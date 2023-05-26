"""
 ______   _____  _     
| __ ) \ / / _ \| |    
|  _ \\ V / | | | |    
| |_) || || |_| | |___ 
|____/ |_| \___/|_____|

BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
Link: https://arxiv.org/abs/2006.07733
"""

import torch
from torch import nn
import torch.nn.functional as F 


__all__ = ['BYOL']


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass