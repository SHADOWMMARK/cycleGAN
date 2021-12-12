import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, set):
        comic = []
        face = []
        print(set.class_to_idx)
        self.dataset = []
        for (_, img) in enumerate(set):
            if img[1] == 0:
                comic.append(img[0])
            else:
                face.append(img[0])
        size = len(comic)
        self.dataset = []
        for i in range(size):
            self.dataset.append([face[i],comic[i]])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class FaceDataSet:
    def __init__(self, args, batch_size=5, dataset_path=""):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.resize = args.resize
        self.crop_size = args.crop_size
        self.trans = self.get_transform()
        self.train_loader, self.val_loader = self.get_dataloader()
        
    
    def get_transform(self):
        
        trans_list = [
            transforms.Resize(self.resize, Image.BICUBIC),
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        trans = transforms.Compose(trans_list)
        
        return trans
        
    def get_dataloader(self):
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path,'train'), transform=self.trans)
        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path,'train'), transform=self.trans)
        
        train_dataset= Mydataset(train_set)
        val_dataset = Mydataset(val_set)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)
        return train_loader, val_loader
