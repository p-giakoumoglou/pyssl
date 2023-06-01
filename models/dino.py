"""
 ____ ___ _   _  ___  
|  _ \_ _| \ | |/ _ \ 
| | | | ||  \| | | | |
| |_| | || |\  | |_| |
|____/___|_| \_|\___/ 

DINO: Emerging Properties in Self-Supervised Vision Transformers
Link: https://arxiv.org/abs/2104.14294
"""

import torch
from torch import nn
import torch.nn.functional as F 
import copy


__all__ = ['DINO']


def cross_entropy_loss(z_t, z_s, temp_s, temp_t, C):
    z_t = z_t.detach() # stop gradient
    z_s = F.log_softmax(z_s / temp_s, dim=1)
    z_t = F.log_softmax((z_t - C) / temp_t, dim=1) # center + sharpen
    return - (z_t * torch.log(z_s)).sum(dim=1).mean()


class Projection(nn.Module):
    """ Projection Head for DINO """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=256, bottleneck_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x 


class DINO(nn.Module):
    """ Distillation-based Self-Supervised Learning: BYOL """
    def __init__(self, backbone, feature_size, temp_s, temp_t, m):
        super().__init__()
        
        assert backbone is not None and feature_size>0
        
        self.temp_s = temp_s
        self.temp_t = temp_t
        self.C = 0
        self.m = m
        
        self.backbone = backbone        
        self.projector = Projection(feature_size, hidden_dim=2048, out_dim=256)
        
        self.student = nn.Sequential(self.backbone, self.projector)
        self.teacher = copy.deepcopy(self.student)
        
    def forward(self, x1, x2):
        z1_s, z2_s = self.student(x1), self.student(x2)
        
        with torch.no_grad():
            z1_t, z2_t = self.teacher(x1), self.teacher(x2)
            
        loss = cross_entropy_loss(z1_t, z2_s, self.temp_s, self.temp_t, self.C)/2 + \
               cross_entropy_loss(z2_t, z1_s, self.temp_s, self.temp_t, self.C)/2
        self.C = self.m*self.C + (1-self.m)*torch.cat([z1_t, z2_t]).mean(dim=0)
            
        return loss


if __name__ == '__main__':
    from networks import model_dict
    model_fun, feature_size = model_dict['resnet18']
    backbone = model_fun()
    
    model = DINO(backbone, feature_size, 0.5, 0.5, 0.5)
    
    images1 = images2 = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(images1, images2)
        print(f'loss = {loss}')