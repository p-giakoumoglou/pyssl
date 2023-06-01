class EarlyStopping:
    """ Early Stopping """
    def __init__(self, patience=30):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness: 
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch 
        self.possible_stop = delta >= (self.patience - 1) 
        stop = delta >= self.patience
        if stop:
            print(f'Stopping training early as no improvement observed in last {self.patience} epochs \n'  
                  f'Best results observed at epoch {self.best_epoch}')
        return stop
