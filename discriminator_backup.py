#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:27:16 2021

@author: fanyaoyu
"""
import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels), # normalize pixel-wise
            nn.LeakyReLU(0.2))
    
    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__() 
        # initialize the first layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2))
        
        layers = [] # set up the rest layers
        in_channels = features[0] 
        for feature in features[1:]:
            layers.append(DiscBlock(in_channels, out_channels = feature, stride=1 if feature==features[-1] else 2)) # set the stride for each layer to be 2 exept the last layer to be 1
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers) # the astroid unwraps the layers
            
    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
    
def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)
        
if __name__ == '__main__':
    test()
