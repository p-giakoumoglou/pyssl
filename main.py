import os
import warnings
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dotmap import DotMap
from sklearn.manifold import TSNE
from tqdm import tqdm
import pyfiglet
warnings.simplefilter("ignore")

import networks
import models
import transformations
from utils import AverageMeter, LearningRateScheduler
from utils import set_deterministic, set_all_seeds


def parse_arguements():
    """ Arguements for pre-training, we use DotMap rather than argparse """
    args = DotMap()
    
    # General
    args.debug = True
    args.data_dir = './data/'
    args.ckpt_dir = './ckpts/'
    args.logs_dir = './logs/'
    args.device = 'cuda'
    args.seed = 123
    
    # SimSiam
    args.model.name = "supcon" # simsiam, simclr, supcon
    args.model.backbone = 'resnet18'
    args.model.save_freq = 10
    
    # Dataset
    args.dataset.name = "cifar10"
    args.dataset.image_size = 32 
    args.dataset.batch_size = 128
    args.dataset.num_workers = 12
    
    # Train - Optimizer
    args.train.optimizer = "sgd"
    args.train.weight_decay = 0.0005
    args.train.momentum = 0.9
    args.train.stop_at_epoch = 800
    
    # Train - Scheduler
    args.train.warmup_epochs = 10
    args.train.warmup_lr = 0
    args.train.base_lr = 0.03
    args.train.final_lr = 0
    args.train.num_epochs = 800
        
    if args.debug:
        args.train.stop_at_epoch = 2

    print(pyfiglet.figlet_format(args.model.name.upper()))
    set_deterministic(args.seed)
    set_all_seeds(args.seed)
    print(f'Using {args.device.upper()}')
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)
    if not os.path.exists(args.logs_dir): os.makedirs(args.logs_dir)
    
    return args


def set_loader(args):
    """ Data loaders for the training and validation on CIFAR 10 """
    transform = transformations.transform_dict[args.model.name.lower()]
    if args.dataset.name.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transformations.TwoCropTransform(*transform))
        valid_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transformations.TRANSFORM_EVAL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.dataset.batch_size, shuffle=True, num_workers=args.dataset.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(args.dataset.batch_size//2), shuffle=False, num_workers=args.dataset.num_workers)
    else:
        raise NotImplementedError(f'\U0000274C Dataset {args.dataset.name} not implemented')
    print(f'Initialized loaders for {args.dataset.name.upper()}')
    return train_loader, valid_loader


def set_model(args):
    """
    Self-Supervised Learning Model: BYOL/SimCLR/SimSiam/SupCon/SWAV
    """
    model_fun, feature_size = networks.model_dict[args.model.backbone]
    backbone = model_fun()
    print(f'\U0001F680 Backbone {args.model.backbone.upper()} initialized')
    
    if args.model.name.lower() == "simsiam":
        model = models.SimSiam(backbone, feature_size) 
    elif args.model.name.lower() == "simclr":
        model = models.SimCLR(backbone, feature_size) 
    elif args.model.name.lower() == "supcon":
        model = models.SupCon(backbone, feature_size) 
        print('\U000026A0 Using SupCon, labels are parsed in loss calculation')
    else:
        NotImplementedError(f'Model {args.model.name.upper()} not implemented')
    model = model.to(args.device)
    print(f'\U0001F680 Model {args.model.name.upper()} initialized')
    
    return model, feature_size


def set_optimizer(optimizer_name, model, lr, momentum, weight_decay):
    """ Optimizer for Self-Supervised Learning """
    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        NotImplementedError(f'Optimizer {optimizer_name.upper()} not implemented')
    print(f'Optimizer {optimizer.__class__.__name__} initialized (lr={lr}, mom={momentum}, wd={weight_decay})')
    return optimizer


def set_lr_scheduler(optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
    """ Learning Rate Scheduler for Self-Supervised Learning """
    scheduler = LearningRateScheduler(optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, True)
    print(f'Learning Rate Scheduler initialized (warmup_epochs={warmup_epochs}, warmup_lr={warmup_lr}, base_lr={final_lr}, base_lr={final_lr})')
    return scheduler


def pretrain(loader, model, optimizer, lr_scheduler, epoch, args):
    """
    One epoch Self-Supervised Learning pre-training
    """
    model.train()
    
    losses = AverageMeter('loss')
    lrs = AverageMeter('lr')
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.train.stop_at_epoch}')
    for idx, ((images1, images2), labels) in enumerate(pbar):
        if args.debug == True and idx > 5: break
        bsz = images1.shape[0]       
        model.zero_grad()
        images1, images2 = images1.to(args.device, non_blocking=True), images2.to(args.device, non_blocking=True)
        if args.model.name.lower() == 'supcon':
            loss = model.forward(images1, images2, labels).mean()
        else:
            loss = model.forward(images1, images2).mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        losses.update(loss.item(), bsz)
        lrs.update(lr_scheduler.get_lr(), bsz)
        
        pbar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_lr())
        
    return losses.avg, lrs.avg


def tsne(loader, model, args):
    """ t-SNE to project encoder (backbone & head) to 2 dimensions """
    latent_vectors = []
    classes = []
    
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc='t-SNE evaluation')
        for idx, (images, labels) in enumerate(pbar):
            if args.debug == True and idx > 10: break
            images = images.to(args.device, non_blocking=True)
            latent_vectors.append(model.encoder(images).view(len(images),-1))
            classes.extend(labels.cpu().detach().numpy())
        latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy() 
    
    tsne = TSNE(2)
    clustered = tsne.fit_transform(latent_vectors)
    fig = plt.figure(figsize=(24,10))
    cmap = plt.get_cmap('Spectral', 10)
    plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
    plt.colorbar(drawedges=True)
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(args.logs_dir + '/tsne_plot.png')
    plt.show()
    plt.close(fig)
    return fig
  

def main(args):
    """ Self-Supervised Learning: Backbone pre-training """
    
    # Get loaders
    train_loader, valid_loader = set_loader(args)
    
    # Get mode: SimCLR, BYOL, SimSiam, SupCon, SWAV, etc.
    model, _ = set_model(args)
    
    ######################
    ####### Part 1 #######
    ######################
    
    # Set optimizer
    optimizer = set_optimizer(args.train.optimizer,
                              model,
                              lr=args.train.base_lr*args.dataset.batch_size/256, 
                              momentum=args.train.momentum,
                              weight_decay=args.train.weight_decay,
                              )
    
    # Set learning rate scheduler
    lr_scheduler = set_lr_scheduler(optimizer,
                                args.train.warmup_epochs,
                                args.train.warmup_lr*args.dataset.batch_size/256,
                                args.train.num_epochs,
                                args.train.base_lr*args.dataset.batch_size/256,
                                args.train.final_lr*args.dataset.batch_size/256, 
                                len(train_loader),
                                True,
                                )

    #########################################
    ####### 1.1 Backbone Pre-training #######
    #########################################
    print('\nStarting backbone pre-training')
    writer = SummaryWriter(log_dir=args.logs_dir)
    for epoch in range(1, args.train.stop_at_epoch+1):
        loss, lr = pretrain(train_loader, model, optimizer, lr_scheduler, epoch, args)
        writer.add_scalar('pretrain/loss', loss, epoch)
        writer.add_scalar('pretrain/lr', lr, epoch)
        if epoch % args.model.save_freq == 0:
            torch.save(model.state_dict(), args.ckpt_dir + args.model.name + '_' + str(epoch) +".pth")
    
    # Save final model
    torch.save(model.state_dict(), args.ckpt_dir + args.model.name + "_final.pth")
    print('Model saved successfully under', args.ckpt_dir + args.model.name + "_final.pth")

    ##########################
    ####### 1.2 t-SNE ########
    ##########################
    model.load_state_dict(torch.load(args.ckpt_dir + args.model.name + "_final.pth"))
    print('\nt-SNE visualization in 2 dimensions')
    fig = tsne(valid_loader, model, args)
    writer.add_figure('t-sne', fig)
    writer.flush()
    writer.close()        

if __name__ == '__main__':
    args = parse_arguements()
    main(args)