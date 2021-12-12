import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import CycleGAN
import time
from arguments import Arguments
from dataset import FaceDataSet
from PIL import Image
import random

def tensor2img(img):
    if not isinstance(img, np.ndarray):
      if isinstance(img, torch.Tensor):
        img = img.data
      img = img[0].cpu().float().numpy()
      img = (np.transpose(img, (1, 2, 0)) + 1)/2.0 * 255
    return img.astype(np.uint8)

def img_save(img, path, name):
    img = Image.fromarray(img)
    img.save(os.path.join(path, name))

def test(dataset, model, config):
    loader = dataset.train_loader
    n_iter = 0
    for data in loader:
        n_iter += config.batch_size
        if n_iter >= len(loader)-10:
            A = data[0]
            B = data[1]
            model.Optimize(A, B, False)
            real_A, real_B = tensor2img(model.real_A), tensor2img(model.real_B)
            fake_A, fake_B = tensor2img(model.fake_A), tensor2img(model.fake_B)
            rec_A, rec_B = tensor2img(model.rec_A), tensor2img(model.rec_B)
            img_save(real_A, config.print_dir, 'test_'+ str(n_iter) + '_real_A.jpg')
            img_save(real_B, config.print_dir, 'test_'+ str(n_iter) + '_real_B.jpg')
            img_save(fake_A, config.print_dir, 'test_'+ str(n_iter) + '_fake_A.jpg')
            img_save(fake_B, config.print_dir, 'test_'+ str(n_iter) + '_fake_B.jpg')
            img_save(rec_A, config.print_dir, 'test_'+ str(n_iter) + '_rec_A.jpg')
            img_save(rec_B, config.print_dir, 'test_'+ str(n_iter) + '_rec_B.jpg')


def train(dataset, model, config):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    G_A, G_B, D_A, D_B, cycle_A, cycle_B = [],[],[],[],[],[]
    for epoch in range(config.n_epochs):
        n_iter = 0
        # epoch_start = time.time()
        print_iter = random.randint(0,len(train_loader)-11) * config.batch_size
        data_size = len(train_loader)
        
        for batch_data in train_loader:
            # print(len(batch_data))
            A = batch_data[0]
            B = batch_data[1]
            # iter_start = time.time()
            n_iter += config.batch_size
            if n_iter < data_size - 10:
                model.Optimize(A, B, True)
                if n_iter == print_iter:
                    real_A, real_B = tensor2img(model.real_A), tensor2img(model.real_B)
                    fake_A, fake_B = tensor2img(model.fake_A), tensor2img(model.fake_B)
                    rec_A, rec_B = tensor2img(model.rec_A), tensor2img(model.rec_B)
                    img_save(real_A, config.print_dir, str(epoch) + '_real_A.jpg')
                    img_save(real_B, config.print_dir, str(epoch) + '_real_B.jpg')
                    img_save(fake_A, config.print_dir, str(epoch) + '_fake_A.jpg')
                    img_save(fake_B, config.print_dir, str(epoch) + '_fake_B.jpg')
                    img_save(rec_A, config.print_dir, str(epoch) + '_rec_A.jpg')
                    img_save(rec_B, config.print_dir, str(epoch) + '_rec_B.jpg')
            
            if n_iter % config.print_freq == 0:
                print('print loss after %d iter' % (n_iter))
                print("G_A: %f, G_B: %f, D_A: %f, D_B: %f, cycle_A: %f, cycle_B: %f" % (model.genLoss_A,model.genLoss_B,model.disLoss_A,model.disLoss_B,model.cycleLoss_A,model.cycleLoss_B))
            

        model.lr_update()

        G_A.append(model.genLoss_A)
        G_B.append(model.genLoss_A)
        D_A.append(model.disLoss_A)
        D_B.append(model.disLoss_B)
        cycle_A.append(model.cycleLoss_A)
        cycle_B.append(model.cycleLoss_B)

    plt.title('G_A')
    plt.plot(G_A)
    plt.show()
    plt.title('G_B')
    plt.plot(G_B)
    plt.show()
    plt.title('D_A')
    plt.plot(D_A)
    plt.show()
    plt.title('D_B')
    plt.plot(D_B)
    plt.show()
    plt.title('cycle_A')
    plt.plot(cycle_A)
    plt.show()
    plt.title('cycle_B')
    plt.plot(cycle_B)
    plt.show()
	
    return 
            

if __name__ == '__main__':
    
    config = Arguments().parse()
    dataSet = FaceDataSet(config, batch_size=config.batch_size, dataset_path=config.dataset_path)
    trainModel = CycleGAN(config)
    
    if config.is_train:
        train(dataSet, trainModel, config)
    
    test(dataSet, trainModel, config)