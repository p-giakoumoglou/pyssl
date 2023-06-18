# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
from PIL import Image
import copy


__all__ = ['DINO']


class DINO(nn.Module):
    """ 
    DINO: Emerging Properties in Self-Supervised Vision Transformers
    Link: https://arxiv.org/abs/2104.14294
    Implementation: https://github.com/facebookresearch/dino
    """
    def __init__(self, backbone, feature_size, projection_dim=256, hidden_dim=2048, bottleneck_dim=256, temp_s=0.1, temp_t=0.5, m=0.5, lamda=0.996, num_crops=6,
                 image_size=224, mean=(0.5,), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.projection_dim = projection_dim
        self.temp_s = temp_s
        self.temp_t = temp_t
        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.m = m
        self.lamda = lamda # EMA update
        self.backbone = backbone   
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.head_student = Head(feature_size, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, out_dim=projection_dim)
        self.student = self.encoder = nn.Sequential(self.backbone, self.head_student)
        self.head_teacher = Head(feature_size, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, out_dim=projection_dim)
        self.teacher = nn.Sequential(copy.deepcopy(backbone), self.head_teacher)
        self._init_teacher()
        self.num_crops = num_crops
        self.augment_global1 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.04, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_global2 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.04, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(threshold=128, p=0.2),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_local = T.Compose([
                T.RandomResizedCrop(int(image_size*3/7), scale=(0.05, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std),
                ])
        
    def forward(self, x):
        x1, x2 = self.augment_global1(x), self.augment_global2(x)
        
        xc = []
        if self.num_crops > 0:
            for _ in range(self.num_crops):
                xc.append(self.augment_local(x))
          
        z1_s, z2_s = self.student(x1), self.student(x2)
           
        zc_s = []
        for x in xc:
            zc_s.append(self.student(x))
        
        with torch.no_grad():
            self._momentum_update_teacher()
            z1_t, z2_t = self.teacher(x1), self.teacher(x2)
            
        z_s = [z1_s, z2_s] + zc_s
        z_t = [z1_t, z2_t]  
        
        loss, loss_terms = 0, 0
        for iq, q in enumerate(z_t):
            for iv, v in enumerate(z_s):
                if iv==iq:
                    continue
                loss += cross_entropy_loss(q, v, self.temp_s, self.temp_t, self.center)
                loss_terms += 1
        loss /= loss_terms 
        
        self._update_center(z1_t, z2_t)
        return loss
    
    def _init_teacher(self):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient
            
    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data = self.lamda * param_k.data  + (1. - self.lamda) * param_q.data
           
    @torch.no_grad()
    def _update_center(self, z1_t, z2_t):
        self.center = self.m*self.center + (1-self.m)*torch.cat([z1_t, z2_t]).mean(dim=0)
            

def cross_entropy_loss(z_t, z_s, temp_s, temp_t, center):
    z_t = z_t.detach() # stop gradient
    z_s = z_s / temp_s
    z_t = F.softmax((z_t - center) / temp_t, dim=1) # center + sharpen
    return - (z_t * F.log_softmax(z_s, dim=1)).sum(dim=1).mean()
    

class Head(nn.Module):
    """ Projection Head for DINO """
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, out_dim=256, ):
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
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = DINO(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')