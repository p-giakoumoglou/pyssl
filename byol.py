"""
 ______   _____  _     
| __ ) \ / / _ \| |    
|  _ \\ V / | | | |    
| |_) || || |_| | |___ 
|____/ |_| \___/|_____|

BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
Link: https://arxiv.org/abs/2006.07733
Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol

TODO
    - Cosine schedule for momentum update in EMA
"""

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import copy
from PIL import Image
from utils import Solarization


__all__ = ['BYOL']


def mean_squared_error(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


class MLP(nn.Module):
    """ Projection Head and Prediction Head for BYOL """
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


class BYOL(nn.Module):
    """ Distillation-based Self-Supervised Learning: BYOL """
    def __init__(self, backbone, feature_size, projection_dim=256, tau=0.996,
                 image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()

        # Parameters
        self.projection_dim = projection_dim
        self.tau = tau # EMA update
        self.backbone = backbone
        self.projector = MLP(feature_size, hidden_dim=4096, out_dim=projection_dim)
        
        # Online network
        self.online_encoder = nn.Sequential(self.backbone, self.projector)
        self.online_predictor = MLP(in_dim=projection_dim, hidden_dim=4096, out_dim=projection_dim)
        
        # Target network
        self.target_encoder = copy.deepcopy(self.online_encoder) # target must be a deepcopy of online, since we will use the backbone trained by online
        self._init_target_encoder()
        
        self.encoder = copy.deepcopy(self.online_encoder)
        
        # Augmentation
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
        
        z1_o, z2_o = self.online_encoder(x1), self.online_encoder(x2)
        p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
        
        with torch.no_grad():
            self._momentum_update_target_encoder()
            z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2)
            
        loss =  mean_squared_error(p1_o, z2_t) / 2 + mean_squared_error(p2_o, z1_t) / 2 
            
        return loss
    
    def _init_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
            
    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.tau * param_t.data  + (1. - self.tau) * param_o.data
                  
    @torch.no_grad()
    def eval(self):
        super().eval()
        self.encoder = copy.deepcopy(self.online_encoder)


if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
        
    model = BYOL(backbone, feature_size, tau=0.996)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')