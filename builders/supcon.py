# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torchvision.transforms as T


__all__ = ['SupCon']


class SupCon(nn.Module):
    """
    SupCon: Supervised Contrastive Learning
    Link: https://arxiv.org/abs/2004.11362
    Implementation: https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, backbone, feature_size, projection_dim=128, temperature=0.07,
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
                T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(mean=mean, std=std)
                ])
        
    def forward(self, x, y):
        x1, x2 = self.augment(x), self.augment(x)
        z1, z2 = self.encoder(x1), self.encoder(x2)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        loss = sup_con_loss(z, y, temperature=self.temperature)
        return loss


def sup_con_loss(features, labels=None, mask=None, temperature=0.07, contrast_mode='all', base_temperature=0.07):
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
    loss = loss.view(anchor_count, bz)
    return loss.mean()


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
    
    
if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = SupCon(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    y = torch.rand(4)
    with torch.no_grad():
        loss = model.forward(x, y)
        print(f'loss = {loss}')