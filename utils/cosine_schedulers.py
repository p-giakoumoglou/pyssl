import numpy as np
from abc import ABC, abstractmethod


class CosineDecay(ABC):
    """ Cosine decay scheduler with warm-up """
    def __init__(self, optimizer, warmup_value, base_value, final_value, warmup_epochs, num_epochs, iter_per_epoch):
        self.optimizer = optimizer
        self.warmup_value = warmup_value
        self.base_value = base_value
        self.final_value = final_value
        self.current_value = 0
        self.iter = 0

        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * iter_per_epoch
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(warmup_value, base_value, warmup_iters)

        iters = np.arange(num_epochs * iter_per_epoch - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, schedule))
        assert len(self.schedule) == num_epochs * iter_per_epoch, "Scheduler length does not match iterations"

    @abstractmethod
    def step(self):
        pass

        
class CosineDecayLR(CosineDecay):
    """ Cosine decay learning rate scheduler with warm-up """
    def __init__(self, optimizer, warmup_lr, base_lr, final_lr, warmup_epochs_lr, num_epochs, iter_per_epoch):
        super().__init__(optimizer, warmup_lr, base_lr, final_lr, warmup_epochs_lr, num_epochs, iter_per_epoch)

    def step(self):
        value = 0
        for param_group in self.optimizer.param_groups:
            value = param_group["lr"] = self.schedule[self.iter]

        self.iter += 1
        self.current_value = value
        return value

    def get_lr(self):
        return self.current_value


class CosineDecayWD(CosineDecay):
    """ Cosine decay weight decay scheduler with warm-up """
    def __init__(self, optimizer, warmup_wd, base_wd, final_wd, warmup_epochs_lr, num_epochs, iter_per_epoch):
        super().__init__(optimizer, warmup_wd, base_wd, final_wd, warmup_epochs_lr, num_epochs, iter_per_epoch)

    def step(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # only the first group is regularized
                value = param_group["weight_decay"] = self.schedule[self.iter]
            
        self.iter += 1
        self.current_value = value
        return value
        
    def get_wd(self):
        return self.current_value
    
if __name__ == "__main__":
    import torchvision
    import torch
    import matplotlib.pyplot as plt
    
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    
    num_epochs = 100
    iter_per_epoch = 50
    
    lr_scheduler = CosineDecayLR(optimizer, 0, 1.0, 0.5, 10, num_epochs, iter_per_epoch)
    wd_scheduler = CosineDecayWD(optimizer, 0.0, 0.04, 0.4, 0, num_epochs, iter_per_epoch)
    
    plt.plot(lr_scheduler.schedule)
    plt.title('Learning Rate Schedule as initialized')
    plt.show()
    
    plt.plot(wd_scheduler.schedule)
    plt.title('Weight Decau Schedule as initialized')
    plt.show()
    
    lrs, wds = [], []
    for epoch in range(num_epochs):
        for it in range(iter_per_epoch):
            lr = lr_scheduler.step()
            lrs.append(lr)
            wd = wd_scheduler.step()
            wds.append(wd)
            
    plt.plot(lrs)
    plt.title('Learning Rate after running step()')
    plt.show()
    
    plt.plot(wds)
    plt.title('Weight Decay after running step()')
    plt.show()
            
    
    
    
    
    
    