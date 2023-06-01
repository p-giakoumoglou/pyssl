"""
 ____  _            ____ _     ____  
/ ___|(_)_ __ ___  / ___| |   |  _ \ 
\___ \| | '_ ` _ \| |   | |   | |_) |
 ___) | | | | | | | |___| |___|  _ < 
|____/|_|_| |_| |_|\____|_____|_| \_\

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Link: https://arxiv.org/abs/2002.05709
"""

import torch
from torch import nn
import torch.nn.functional as F 


__all__ = ['SimCLR']


def NT_XentLoss(z1, z2, temperature=0.5):
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


class SimCLR(nn.Module):
    """ Contrastive-based Self-Supervised Learning: SimCLR"""
    def __init__(self, backbone, feature_size):
        super().__init__()
        
        assert backbone is not None and feature_size>0
        
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim=feature_size, out_dim=128)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = NT_XentLoss(z1, z2)
        return loss


if __name__ == '__main__':
    from networks import model_dict
    model_fun, feature_size = model_dict['resnet18']
    backbone = model_fun()
    
    model = SimCLR(backbone, feature_size)
    
    images1 = images2 = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(images1, images2)
        print(f'loss = {loss}')