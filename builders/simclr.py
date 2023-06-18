# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T


__all__ = ['SimCLR']


class SimCLR(nn.Module):
    """ 
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Implementation: https://github.com/google-research/simclr
    """
    def __init__(self, backbone, feature_size, projection_dim=128, temperature=0.5, 
                 image_size=224, mean=(0.5,), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim=feature_size, out_dim=projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std),
                ])
        
    def forward(self, x):
        x1, x2 = self.augment(x), self.augment(x)
        z1, z2 = self.encoder(x1), self.encoder(x2)
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss


def nt_xent_loss(z1, z2, temperature=0.5):
    """ NT-Xent loss """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)
    
    
class Projector(nn.Module):
    """ Projector for SimCLR """
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        
        if hidden_dim is None:
            self.layer1 = nn.Linear(in_dim, out_dim)
        else:
            self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
    def forward(self, x):
        x = self.layer1(x)
        return x 
    
    
if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = SimCLR(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')