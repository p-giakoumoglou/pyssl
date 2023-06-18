# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
import torchvision.transforms as T
import copy
from PIL import Image
from utils import Solarization


__all__ = ['MoCoV3']
    

class MoCoV3(nn.Module):
    """ 
    MoCo v3: Momentum Contrast v3
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/moco-v3
    """
    def __init__(self, backbone, feature_size, projection_dim=256, hidden_dim=2048, temperature=0.5, m=0.999,
                 image_size=224, mean=(0.5,), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.temperature = temperature
        self.m = m
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.encoder_q = nn.Sequential(self.backbone, self.projector)
        self.predictor = Predictor(in_dim=projection_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()
        self.encoder = copy.deepcopy(self.encoder_q)
        self.augment1 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std)
                ])
        self.augment2 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomApply([Solarization(p=1.0)], p=0.2),
                T.Normalize(mean=mean, std=std)
                ])
        
    def forward(self, x):
        x1, x2 = self.augment1(x), self.augment2(x)
        q1 = self.predictor(self.encoder_q(x1))
        q2 = self.predictor(self.encoder_q(x2))
        with torch.no_grad():
            self._update_momentum_encoder()
            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)
        loss = contrastive_loss(q1, k2, self.temperature) + contrastive_loss(q2, k1, self.temperature)
        return loss
        
    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_b, param_m in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
            
    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False 
            
    @torch.no_grad()
    def eval(self):
        super().eval()
        self.encoder = copy.deepcopy(self.encoder_q)
         

def contrastive_loss(q, k, temperature):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,mc->nm', [q, k]) / temperature
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long)).to(q.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * temperature)


class Projector(nn.Module):
    """ Projector for SimCLR v2, used in MoCo v3 too """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=256):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer3 = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.BatchNorm1d(out_dim, eps=1e-5, affine=True),
                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
    
class Predictor(nn.Module):
    """ Projection Head and Prediction Head for BYOL, used in MoCo v3 too """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x       
    
    
if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = MoCoV3(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')