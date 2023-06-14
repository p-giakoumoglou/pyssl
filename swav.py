"""
 ____           ___     __
/ ___|_      __/ \ \   / /
\___ \ \ /\ / / _ \ \ / / 
 ___) \ V  V / ___ \ V /  
|____/ \_/\_/_/   \_\_/  

SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
Link: https://arxiv.org/abs/2006.09882
Implementation: https://github.com/facebookresearch/swav
"""

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T


__all__ = ['SwAV']


def cross_entropy_loss(q, p):
    return torch.mean(torch.sum(q * F.log_softmax(p, dim=1), dim=1))


def swav_loss(c1, c2, c_c, q1, q2, temperature, num_crops):
    loss = 0
    
    p1, p2 = c1/temperature, c2/temperature
    loss += cross_entropy_loss(q1, p2) / (num_crops - 1)
    loss += cross_entropy_loss(q2, p1) / (num_crops - 1)
    
    for c in range(len(c_c)):
        p = c_c[c] / temperature
        loss += cross_entropy_loss(q1, p) / (num_crops - 1)
        loss += cross_entropy_loss(q2, p) / (num_crops - 1)
    
    return loss/2 


class Projector(nn.Module):
    """ Projector for SwAV """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        
        if hidden_dim is None:
            self.layer1 = nn.Linear(in_dim, out_dim)
        else:
            self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
    def forward(self, x):
        x = self.layer1(x)
        return x 


class SwAV(nn.Module):
    """ Clustering-based Self-Supervised Learning: SwAV """
    
    def __init__(self, backbone, feature_size, projection_dim=128, temperature=0.1, epsilon=0.05, 
                 sinkhorn_iterations=3, num_prototypes=3000, queue_length=64, use_the_queue=True, num_crops=6,
                 image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
               
        # Parameters
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.num_prototypes = num_prototypes
        self.queue_length = queue_length
        self.use_the_queue = use_the_queue
        
        # Queue
        self.register_buffer("queue", torch.zeros(2, self.queue_length, self.projection_dim))
        
        # Network
        self.backbone = backbone
        self.projector = Projector(feature_size, 2048, projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.prototypes = nn.Linear(self.projection_dim, self.num_prototypes, bias=False)
                
        self._init_weights()
        
        # Augmentation
        self.num_crops = num_crops
        self.augment_global = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.14, 1.0)),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std),
                ])
        
        self.augment_local = T.Compose([
                T.RandomResizedCrop(int(image_size*3/7), scale=(0.05, 0.14)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.Normalize(mean=mean, std=std),
                ])
        
    def forward(self, x):
        x1, x2 = self.augment_global(x), self.augment_global(x)
        
        if self.num_crops >0:
            xc = []
            for _ in range(self.num_crops):
                xc.append(self.augment_local(x))
        
        bz = x1.shape[0]
        
        with torch.no_grad(): # normalize prototypes
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        
        z1, z2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = nn.functional.normalize(z1, dim=1, p=2), nn.functional.normalize(z2, dim=1, p=2)
        z1, z2 = z1.detach(), z2.detach()
        
        c1, c2 = self.prototypes(z1), self.prototypes(z2)
               
        _c1, _c2 = c1.detach(), c2.detach()
        with torch.no_grad():
            if self.queue is not None:
                if self.use_the_queue:
                    _c1 = torch.cat((torch.mm(self.queue[0], self.prototypes.weight.t()), _c1))
                    _c2 = torch.cat((torch.mm(self.queue[1], self.prototypes.weight.t()), _c2))
                    self.queue[0, bz:] = self.queue[0, :-bz].clone()
                    self.queue[0, :bz] = z1
                    self.queue[1, bz:] = self.queue[1, :-bz].clone()
                    self.queue[1, :bz] = z2
            q1, q2 = self.sinkhorn(_c1)[:bz, :], self.sinkhorn(_c2)[:bz, :]
                
        z_c, c_c = [], []
        for x in xc:
            z = self.encoder(x)
            z = nn.functional.normalize(z, dim=1, p=2)
            z = z.detach()
            z_c.append(z)
            c_c.append(self.prototypes(z))
            
        loss = swav_loss(c1, c2, c_c, q1, q2, self.temperature, 2+len(xc))
        
        return loss 
    
    @torch.no_grad()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
         
    @torch.no_grad()
    def freeze_prototypes(self):
        for name, p in self.prototypes.named_parameters():
            if "prototypes" in name:
                p.grad = None
                    
    @torch.no_grad()
    def sinkhorn(self, Q):
        with torch.no_grad():
            Q = torch.exp(Q / self.epsilon).t()
            B = Q.shape[1]
            K = Q.shape[0]
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            for _ in range(self.sinkhorn_iterations):
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q /= K
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B
            Q *= B
            return Q.t()


if __name__ == '__main__':
    import torchvision
    backbone = torchvision.models.resnet50(pretrained=False)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    
    model = SwAV(backbone, feature_size)
    
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        loss = model.forward(x)
        print(f'loss = {loss}')