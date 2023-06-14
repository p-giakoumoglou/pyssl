"""
 ____             _                 _____          _           
| __ )  __ _ _ __| | _____      __ |_   _|_      _(_)_ __  ___ 
|  _ \ / _` | '__| |/ _ \ \ /\ / /   | | \ \ /\ / / | '_ \/ __|
| |_) | (_| | |  | | (_) \ V  V /    | |  \ V  V /| | | | \__ \
|____/ \__,_|_|  |_|\___/ \_/\_/     |_|   \_/\_/ |_|_| |_|___/
 
Barlow Twins
Link: https://arxiv.org/abs/2104.02057
Implementation: https://arxiv.org/abs/2103.03230

+ does not require large batch size
+ does not require asymmetry between the network twins such as a predictor network
+ does not require gradient stopping
+ does not require  moving average on the weight updates
- benefits from high-dimensional embeddings (projection_dim)
+ cross-correlation matrix computed from twin embeddings as close to the identity matrix as possible
"""

import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
from utils import Solarization


__all__ = ['BarlowTwins']


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


class BarlowTwins(nn.Module):
    """ Contrastive-based Self-Supervised Learning: Barlow Twins"""
    def __init__(self, backbone, feature_size, projection_dim=8192, hidden_dim=8192, lamda=0.005,
                 image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        
        # Parameters
        self.lamda = lamda
        
        # Encoder
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim, projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)
        
        # Augmentation
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
        
        z1, z2 = self.encoder(x1), self.encoder(x2) # NxD
        
        bz = z1.shape[0] # N
        
        c = self.bn(z1).T @ self.bn(z2) # DxD
        c.div_(bz)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss
    
    
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