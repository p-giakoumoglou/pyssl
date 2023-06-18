# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
from utils import Solarization


__all__ = ['BarlowTwins']


class BarlowTwins(nn.Module):
    """ 
    Barlow Twins
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://arxiv.org/abs/2103.03230
    """
    def __init__(self, backbone, feature_size, projection_dim=8192, hidden_dim=8192, lamda=0.005,
                 image_size=224, mean=(0.5,), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.lamda = lamda
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim, projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                Solarization(p=0.0),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_prime = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                Solarization(p=0.2),
                T.Normalize(mean=mean, std=std),
                ])
        
    def forward(self, x):
        x1, x2 = self.augment(x), self.augment_prime(x)
        z1, z2 = self.encoder(x1), self.encoder(x2)
        bz = z1.shape[0]
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(bz)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss
    
    
class Projector(nn.Module):
    """ Projector for Barlow Twins """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer3 = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim, bias=False),
                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
    
def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = BarlowTwins(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')