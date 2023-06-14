from .barlow_twins import BarlowTwins 
from .byol import BYOL
from .dino import DINO
from .moco import MoCo
from .mocov2 import MoCov2
from .mocov3 import MoCov3
from .simclr import SimCLR
from .simsiam import SimSiam
from .simclrv2 import SimCLRv2
from .supcon import SupCon
from .swav import SwAV


__all__ = [
        'BarlowTwins',
        'BYOL',
        'DINO',
        'MoCo',
        'MoCov2',
        'MoCov3',
        'SimCLR',
        'SimCLRv2',
        'SimSiam',
        'SupCon',
        'SwAV',
        ]


model_dict = {
    'barlowtwins': BarlowTwins,
    'byol': BYOL,
    'dino': DINO,
    'moco': MoCo,
    'mocov2': MoCov2,
    'mocov3': MoCov3,
    'simclr': SimCLR,
    'simclrv2': SimCLRv2,
    'simsiam': SimSiam,
    'supcon': SupCon,
    'swav': SwAV,
}
