"""
 ____  _           ____  _                 
/ ___|(_)_ __ ___ / ___|(_) __ _ _ __ ___  
\___ \| | '_ ` _ \\___ \| |/ _` | '_ ` _ \ 
 ___) | | | | | | |___) | | (_| | | | | | |
|____/|_|_| |_| |_|____/|_|\__,_|_| |_| |_|

SimSiam: Exploring Simple Siamese Representation Learning
Link: https://arxiv.org/abs/2011.10566
Implementation: https://github.com/facebookresearch/simsiam
"""

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T


__all__ = ['SimSiam']


def negative_cosine_similarity(p, z):
    """ Negative Cosine Similarity """
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z).sum(dim=1).mean()
    

class Projector(nn.Module):
    """ Projection Head for SimSiam """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 


class Predictor(nn.Module):
    """ Predictor for SimSiam """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class SimSiam(nn.Module):
    """ Distillation-based Self-Supervised Learning: SimSiam """
    def __init__(self, backbone, feature_size, projection_dim=2048,
                 image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()

        # Parameters
        self.projection_dim = projection_dim
        
        # Dataset
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # Encoder & Predictor
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim=2048, out_dim=projection_dim)
        self.predictor = Predictor(in_dim=projection_dim, hidden_dim=512, out_dim=projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        
        # Augmentation
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=mean, std=std)
                ])
        
    def forward(self, x):
        x1, x2 = self.augment(x), self.augment(x)
        z1, z2 = self.encoder(x1), self.encoder(x2) 
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2
        return loss


if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = SimSiam(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')
