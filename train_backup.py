#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:30:15 2021

@author: fanyaoyu
"""

import torch
from dataset import Mydataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train_function(disc_face, disc_comic, gen_face, gen_comic, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler):
    
    face_reals = 0
    comic_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (comic, face) in enumerate(loop):
        comic = comic.to(config.DEVICE)
        face = face.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            # for the face discriminator 
            face_fake = gen_face(comic)
            disc_face_real = disc_face(face)
            disc_face_fake = disc_face(face_fake.detach())
            #H_reals += D_H_real.mean().item()
            #H_fakes += D_H_fake.mean().item()
            disc_face_real_loss = MSE(disc_face_real, torch.ones_like(disc_face_real))
            disc_face_fake_loss = MSE(disc_face_fake, torch.zeros_like(disc_face_fake))
            disc_face_loss = disc_face_real_loss + disc_face_fake_loss
            
            # for the comic discriminator
            comic_fake = gen_comic(face)
            disc_comic_real = disc_comic(comic)
            disc_comic_fake = disc_comic(comic_fake.detach())
            disc_comic_real_loss = MSE(disc_comic_real, torch.ones_like(disc_comic_real))
            disc_comic_fake_loss = MSE(disc_comic_fake, torch.zeros_like(disc_comic_fake))
            disc_comic_loss = disc_comic_real_loss + disc_comic_fake_loss

            # Add losses together
            disc_loss = (disc_face_loss + disc_comic_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators 
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            disc_face_fake = disc_face(face_fake)
            disc_comic_fake = disc_comic(comic_fake)
            gen_face_loss = MSE(disc_face_fake, torch.ones_like(disc_face_fake))
            gen_comic_loss = MSE(disc_comic_fake, torch.ones_like(disc_comic_fake))

            # cycle loss
            cycle_comic = gen_comic(face_fake)
            cycle_face = gen_face(comic_fake)
            cycle_comic_loss = L1(comic, cycle_comic)
            cycle_face_loss = L1(face, cycle_face)


            # add all togethor
            gen_loss = (gen_face_loss + gen_comic_loss
                + cycle_face_loss * config.LAMBDA_CYCLE
                + cycle_comic_loss * config.LAMBDA_CYCLE)

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(face_fake*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(comic_fake*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(face_real=face_reals/(idx+1), face_fake=face_fakes/(idx+1))



def main():
    disc_face = Discriminator(in_channels=3).to(config.DEVICE)
    disc_comic = Discriminator(in_channels=3).to(config.DEVICE)
    gen_comic = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_face = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_face.parameters()) + list(disc_comic.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_comic.parameters()) + list(gen_face.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_face, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_comic, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_face, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_comic, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/faces", root_zebra=config.TRAIN_DIR+"/comics", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_function(disc_face, disc_comic, gen_face, gen_comic, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_face, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_comic, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_face, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_comic, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
