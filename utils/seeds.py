import numpy as np
import random
import torch

def set_all_seeds(seed):
    """ Initialize random number generator seeds """
    if seed is not None:
        print((f"\U000026A0 Random number generator initialized with seed={seed}"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = True
    
    
def set_deterministic(seed):
    """ Initialize deterministic """
    if seed is not None:
        print((f"\U000026A0 Deterministic initialized with seed={seed}"))
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False