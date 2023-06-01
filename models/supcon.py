"""
 ____               ____            
/ ___| _   _ _ __  / ___|___  _ __  
\___ \| | | | '_ \| |   / _ \| '_ \ 
 ___) | |_| | |_) | |__| (_) | | | |
|____/ \__,_| .__/ \____\___/|_| |_|

SupCon: Supervised Contrastive Learning
Link: https://arxiv.org/abs/2004.11362
"""

import torch
from torch import nn


__all__ = ['SupCon']


def SupConLoss(features, labels=None, mask=None, temperature=0.07, contrast_mode='all', base_temperature=0.07):
    """ 
    Supervised Contrastive Loss. It also supports the unsupervised contrastive loss in SimCLR 
    If both labels and mask are None, it degenerates to SimCLR unsupervised loss
    """
    device = features.device 
    if len(features.shape) < 3:
        raise ValueError('features needs to be [bsz, n_views, ...], at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)
    if contrast_mode not in ['all', 'one']:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))
    bz = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both labels and mask')
    elif labels is None and mask is None:
        mask = torch.eye(bz, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != bz:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    mask = mask.repeat(anchor_count, contrast_count)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(bz * anchor_count).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, bz).mean()
    return loss


class Projector(nn.Module):
    """ Projector for SupCon """
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


class SupCon(nn.Module):
    """ Contrastive-based Self-Supervised Learning: SupCon"""
    def __init__(self, backbone, feature_size):
        super().__init__()
        
        assert backbone is not None and feature_size>0
        
        self.backbone = backbone
        self.projector = Projector(feature_size, hidden_dim=feature_size, out_dim=128)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        
    def forward(self, x1, x2, labels):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        loss = SupConLoss(z, labels)
        return loss


if __name__ == '__main__':
    from networks import model_dict
    model_fun, feature_size = model_dict['resnet18']
    backbone = model_fun()
    
    model = SupCon(backbone, feature_size)
    
    images1 = images2 = torch.rand(4, 3, 224, 224)
    labels = torch.rand(4)
    with torch.no_grad():
        loss = model.forward(images1, images2, labels)
        print(f'loss = {loss}')