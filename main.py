import os
import warnings
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from datetime import datetime
import pyfiglet
warnings.simplefilter("ignore")

from utils import AverageMeter, CosineDecayLR, CosineDecayWD
from utils import set_deterministic, set_all_seeds
from utilities import set_model, set_optimizer, set_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser - Training")
    
    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    
    # Directories/Device/Seed
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory for data')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/', help='Directory for checkpoints')
    parser.add_argument('--logs_dir', type=str, default='./logs/', help='Directory for logs')
    parser.add_argument('--device', type=str, default='cuda', help='Device for computation')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    # Model
    parser.add_argument('--model_name', type=str, default='simsiam', help='Name of the model')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving model checkpoints')
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='Name of the dataset')
    parser.add_argument('--image_size', type=int, default=32, help='Size of input images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Training - Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs (affects scheduler as well)')
    
    # Training - Scheduler Learning Rate
    parser.add_argument('--warmup_epochs_lr', type=int, default=10, help='Number of warm-up epochs')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Learning rate during warm-up')
    parser.add_argument('--base_lr', type=float, default=0.03, help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=0, help='Final learning rate')
    
    # Training - Scheduler Weight Decay
    parser.add_argument('--warmup_epochs_wd', type=int, default=0, help='Number of warm-up epochs')
    parser.add_argument('--warmup_wd', type=float, default=0, help='Learning rate during warm-up')
    parser.add_argument('--base_wd', type=float, default=0.0005, help='Base learning rate')
    parser.add_argument('--final_wd', type=float, default=0.05, help='Final learning rate')
    
    
    args = parser.parse_args()
    
    if args.debug:
        args.num_epochs = 3
        args.warmup_epochs_lr = 2
        args.warmup_epochs_wd=2
                    
    print(pyfiglet.figlet_format(args.model_name.upper()))
    set_deterministic(args.seed)
    set_all_seeds(args.seed)
    print(f'Using {args.device.upper()}')
    args.logs_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args.model_name.lower() +  "/"
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)
    if not os.path.exists(args.logs_dir): os.makedirs(args.logs_dir)
    
    return args
      

def pretrain(loader, model, optimizer, lr_scheduler, wd_scheduler, epoch, args):
    """
    One epoch Self-Supervised Learning pre-training
    """    
    model.train()
    
    losses = AverageMeter('loss')
    lrs = AverageMeter('lr')
    wds = AverageMeter('wd')
    
    pbar = tqdm(loader, ascii=True, desc=f'Epoch {epoch}/{args.num_epochs} (pretraining)', unit='batches')
    for idx, ((images1, images2), labels) in enumerate(pbar):
        if args.debug == True and idx > 10: break    
        bsz = images1.shape[0]       
        model.zero_grad()
        images1, images2 = images1.to(args.device, non_blocking=True), images2.to(args.device, non_blocking=True)
        if args.model_name.lower() == 'supcon':
            loss = model.forward(images1, images2, labels).mean()
        else:
            loss = model.forward(images1, images2).mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        wd_scheduler.step()
        losses.update(loss.item(), bsz)
        lrs.update(lr_scheduler.get_lr(), bsz)
        wds.update(wd_scheduler.get_wd(), bsz)
        
        pbar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_lr(), wd=wd_scheduler.get_wd())
        
    return losses.avg, lrs.avg, wds.avg


def knn(loader, model, epoch, args, k=(5,20)):
    """ k-nearest neighbors (KNN) evaluation """
    latent_vectors = []
    classes = []
    
    model.to(args.device)
    model.eval()
        
    with torch.no_grad():
        pbar = tqdm(loader, ascii=True, desc=f'Epoch {epoch}/{args.num_epochs} (k-NN evaluation)', unit='batches')
        for idx, (images, labels) in enumerate(pbar):
            if args.debug == True and idx > 10: break
            images = images.to(args.device, non_blocking=True)
            latent_vectors.append(model.encoder(images).view(len(images),-1))
            classes.extend(labels.cpu().detach().numpy())
        latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy() 
        classes = np.array(classes)
        
    acc = []
    for kk in k:
        knn = KNeighborsClassifier(kk)
        knn.fit(latent_vectors, classes)
        pred = knn.predict(latent_vectors)
        acc.append(np.mean(pred == classes) * 100)    
    return acc


def tsne(loader, model, args):
    """ t-SNE to project encoder (backbone & head) to 2 dimensions """
    latent_vectors = []
    classes = []
    
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, ascii = True, desc='t-SNE evaluation', unit='batches')
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
    train_loader, valid_loader = set_loader(args, use_two_crop=True)
    
    # Get mode: SimCLR, BYOL, SimSiam, SupCon, SWAV, etc.
    model, _ = set_model(args)
    
    
    # Set optimizer
    optimizer = set_optimizer(args.optimizer,
                              model,
                              lr=args.base_lr*args.batch_size/256, 
                              momentum=args.momentum,
                              weight_decay=args.base_wd,
                              )
    
    # Set schedulers
    lr_scheduler = CosineDecayLR(optimizer,
                                 args.warmup_lr*args.batch_size/256, 
                                 args.base_lr*args.batch_size/256, 
                                 args.final_lr*args.batch_size/256, 
                                 args.warmup_epochs_lr, 
                                 args.num_epochs, 
                                 len(train_loader),
                                 )
    print(f'Cosine decay scheduler initialized (warmup_epochs={args.warmup_epochs_lr}, warmup_lr={args.warmup_lr*args.batch_size/256}, base_lr={args.base_lr*args.batch_size/256}, final_lr={args.final_lr*args.batch_size/256})')
    
    wd_scheduler = CosineDecayWD(optimizer,
                                 args.warmup_wd, 
                                 args.base_wd, 
                                 args.final_wd, 
                                 args.warmup_epochs_wd, 
                                 args.num_epochs, 
                                 len(train_loader),
                                 )
    print(f'Cosine decay scheduler initialized (warmup_epochs={args.warmup_epochs_wd}, warmup_wd={args.warmup_wd}, base_wd={args.base_wd}, final_wd={args.final_wd})')

    # Epoch training
    print('\nStarting backbone pre-training\n')
    writer = SummaryWriter(log_dir=args.logs_dir)
    for epoch in range(1, args.num_epochs+1):
        
        # Backbone pretraining
        loss, lr, wd = pretrain(train_loader, model, optimizer, lr_scheduler, wd_scheduler, epoch, args)
        writer.add_scalar('pretrain/loss', loss, epoch)
        writer.add_scalar('pretrain/lr', lr, epoch)
        writer.add_scalar('pretrain/wd', wd, epoch)
        
        # k-NN evaluation 
        top1_knn_5nn, top1_knn_20nn = knn(valid_loader, model, epoch, args, k=(5,20))
        writer.add_scalar('eval/top1_knn_5nn', top1_knn_5nn, epoch)
        writer.add_scalar('eval/top1_knn_20nn', top1_knn_20nn, epoch)
        
        # Save
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), args.ckpt_dir + args.model_name + '_' + str(epoch) +".pth")
    
    # Save final model
    torch.save(model.state_dict(), args.ckpt_dir + args.model_name + "_final.pth")
    print('Model saved successfully under', args.ckpt_dir + args.model_name + "_final.pth")

    # t-SNE visualization
    model.load_state_dict(torch.load(args.ckpt_dir + args.model_name + "_final.pth"))
    print('\nt-SNE visualization in 2 dimensions\n')
    fig = tsne(valid_loader, model, args)
    writer.add_figure('t-sne', fig)
    writer.flush()
    writer.close()        

if __name__ == '__main__':
    args = parse_args()
    main(args)