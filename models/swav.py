"""
 ______        _____     __
/ ___\ \      / / \ \   / /
\___ \\ \ /\ / / _ \ \ / / 
 ___) |\ V  V / ___ \ V /  
|____/  \_/\_/_/   \_\_/  

SWAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
Link: https://arxiv.org/abs/2006.09882
"""

import torch
from torch import nn
import torch.nn.functional as F 


__all__ = ['SWAV']


class SWAV(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass