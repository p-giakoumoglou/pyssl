"""
 __  __        ____              ____  
|  \/  | ___  / ___|___   __   _|___ \ 
| |\/| |/ _ \| |   / _ \  \ \ / / __) |
| |  | | (_) | |__| (_) |  \ V / / __/ 
|_|  |_|\___/ \____\___/    \_/ |_____|
                                       
MoCo v2: Momentum Contrast v2
Link: https://arxiv.org/abs/2003.04297
Implementation: https://github.com/facebookresearch/moco
"""

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import copy


__all__ = ['MoCoV2']


def loss_fn(q, k, queue, temperature):
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
    loss = F.cross_entropy(logits, labels)
    return loss


class Projector(nn.Module):
    """ Projector for MoCov2: Copy from SimCLR"""
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        if hidden_dim == None: hidden_dim=in_dim
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
    def forward(self, x):
        x = self.layer1(x)
        return x 
    

class MoCoV2(nn.Module):
    """ Contrastive-based Self-Supervised Learning: MoCov2"""
    def __init__(self, backbone, feature_size, projection_dim=128, K=65536, m=0.999, temperature=0.07,
                 image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        
        # Parameters
        self.projection_dim = projection_dim
        self.K = K
        self.m = m
        self.temperature = temperature     
        self.backbone = backbone
        self.projector = Projector(feature_size, feature_size, projection_dim)
        
        # Encoder - queue
        self.encoder_q  = nn.Sequential(self.backbone, self.projector)
        
        # Encoder - key
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()
        
        # Queue        
        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.encoder = copy.deepcopy(self.encoder_q)
        
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
        x_q, x_k = self.augment(x), self.augment(x)        
        
        q = self.encoder_q(x_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_encoder_k()
            x_k, idx_unshuffle = self._batch_shuffle_single_gpu(x_k)
            k = self.encoder_k(x_k) 
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        loss = loss_fn(q, k, self.queue, self.temperature)
        self._dequeue_and_enqueue(k)
        return loss
    
    @torch.no_grad()
    def eval(self):
        super().eval()
        self.encoder = copy.deepcopy(self.encoder_q)
    
    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False 
        
    @torch.no_grad()
    def _momentum_update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % bz == 0
        self.queue[:, ptr:(ptr + bz)] = keys.t() 
        ptr = (ptr + bz) % self.K
        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]
            

if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = MoCoV2(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')