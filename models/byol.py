"""
 ______   _____  _     
| __ ) \ / / _ \| |    
|  _ \\ V / | | | |    
| |_) || || |_| | |___ 
|____/ |_| \___/|_____|

BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
Link: https://arxiv.org/abs/2006.07733

TODO
    - Cosine schedule for momentum update in EMA
"""

import torch
from torch import nn
import torch.nn.functional as F 
import copy
import numpy as np


__all__ = ['BYOL']


def mean_squared_error(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1).mean()


class MLP(nn.Module):
    """ Projection Head for BYOL """
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
    def __init__(self, backbone, feature_size, tau=0.996):
        super().__init__()
        
        assert backbone is not None and feature_size>0, "BYOL `backbone` and `feature_size` error"
        assert tau <=1 and tau >0, "BYOL `tau` must be in [0,1]"
        
        self.tau = tau # EMA update
        self.backbone = backbone
        self.projector = MLP(feature_size, hidden_dim=4096, out_dim=256)
        
        self.online_predictor = MLP(in_dim=256, hidden_dim=4096, out_dim=256)
        self.online_encoder = nn.Sequential(self.backbone, self.projector)
        
        self.target_encoder = copy.deepcopy(self.online_encoder) # target must be a deepcopy of online, since we will use the backbone trained by online !!!
        self._init_target_encoder()

    def forward(self, x1, x2):
        z1_o, z2_o = self.online_encoder(x1), self.online_encoder(x2)
        p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
        
        with torch.no_grad():
            z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2)
            self._update_target_encoder_parameters
            
        loss =  mean_squared_error(p1_o, z2_t) / 2 + mean_squared_error(p2_o, z1_t) / 2 
            
        return loss
    
    def _init_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)  # initialize
            param_t.requires_grad = False     # not update by gradient
            
    @torch.no_grad()
    def _update_target_encoder_parameters(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.tau * param_t.data  + (1. - self.tau) * param_o.data
            
    def adjust_tau(self, k, K): # TODO
        self.tau = 1 - (1 - self.tau) * (np.cos(np.pi * k/K) + 1) / 2
        raise NotImplementedError("BYOL tau not implemented yet")
      
    @torch.no_grad()
    def eval(self): # XXX
        super().eval()
        self.encoder = copy.deepcopy(self.online_encoder)


if __name__ == '__main__':
    from networks import model_dict
    model_fun, feature_size = model_dict['resnet18']
    backbone = model_fun()
    
    model = BYOL(backbone, feature_size)
    
    images1 = images2 = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(images1, images2)
        print(f'loss = {loss}')