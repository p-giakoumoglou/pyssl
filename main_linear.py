import os
import warnings
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from datetime import datetime
import pyfiglet
warnings.simplefilter("ignore")

import networks
import transformations
from main import set_model, set_optimizer, set_lr_scheduler
from utils import calculate_topk_accuracy, AverageMeter
from utils import set_deterministic, set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser - Evaluation")
    
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
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    
    # Evaluation - Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer for training')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    
    # Evaluation - Scheduler
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Learning rate during warm-up')
    parser.add_argument('--base_lr', type=float, default=30, help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=0, help='Final learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    if args.debug:
        args.stop_at_epoch = 2

    print(pyfiglet.figlet_format(args.model_name.upper()))
    set_deterministic(args.seed)
    set_all_seeds(args.seed)
    print(f'Using {args.device.upper()}')
    args.logs_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args.model_name.lower() +  "_linear/"
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)
    if not os.path.exists(args.logs_dir): os.makedirs(args.logs_dir)
    
    return args


def set_loader(args):
    """ Data loaders for the training and validation on CIFAR 10 """
    if args.dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transformations.TRANSFORM_LINEAR)
        valid_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transformations.TRANSFORM_EVAL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise NotImplementedError(f'\U0000274C Dataset {args.dataset_name} not implemented')
    print(f'Initialized loaders for {args.dataset_name.upper()}')
    return train_loader, valid_loader

    
def train(loader, model, classifier, optimizer, lr_scheduler, epoch, args):
    """ 
    One epoch fine-tuning: Pass images through pre-trained frozen backbone and linear classifier
    """
    model.eval()
    classifier.train()
    
    losses = AverageMeter('loss')
    lrs = AverageMeter('lr')
    
    pbar = tqdm(loader, ascii = True, desc=f'Epoch {epoch}/{args.num_epochs}')
    for idx, (images, labels) in enumerate(pbar):
        if args.debug == True and idx > 5: break
        bsz = labels.shape[0]
        classifier.zero_grad()
        images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)
        with torch.no_grad():
            feature = model.backbone(images)
        preds = classifier(feature)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        losses.update(loss.item(), bsz)
        lrs.update(lr_scheduler.get_lr(), bsz)
        
        pbar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_lr())
    
    return losses.avg, lrs.avg


def validate(loader, model, classifier, args):
    """ 
    Validation: : Pass images through pre-trained frozen backbone and frozen linear classifier
    """
    model.eval()
    classifier.eval()
    
    losses = AverageMeter('loss')
    acc1 = AverageMeter('acc@1')
    acc5 = AverageMeter('acc@5')
    
    pbar = tqdm(loader, ascii = True, desc='Evaluating')
    for idx, (images, labels) in enumerate(pbar):
        if args.debug == True and idx > 5: break
        bsz = labels.shape[0]
        images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)
        with torch.no_grad():
            feature = model.backbone(images)
            preds = classifier(feature.detach())
        loss = F.cross_entropy(preds, labels)
        
        top1, top5 = calculate_topk_accuracy(preds, labels, topk=(1,5))
    
        losses.update(loss, bsz)
        acc1.update(top1, bsz)
        acc5.update(top5, bsz)
        
        pbar.set_postfix(loss=loss.item(), acc1=top1, acc5=top5)
    
    return losses.avg, acc1.avg, acc5.avg


def main(args):
    """ Self-Supervised Learning: Classifier training """
    
    # Get loaders
    train_loader, valid_loader = set_loader(args)
    
    # Get mode: SimCLR, BYOL, SimSiam, SupCon, SWAV, etc.
    model, feature_size = set_model(args)
    
    ######################
    ####### Part 2 #######
    ######################
    
    try:
        model.load_state_dict(torch.load(args.ckpt_dir + args.model_name + "_final.pth"))
    except:
        raise ValueError(f'\U0000274C Pre-trained model {args.model_name} not found')
    
    classifier = networks.LinearClassifier(feature_size, args.num_classes).to(args.device)
    
    # Set optimizer
    optimizer = set_optimizer(args.optimizer,
                              classifier,
                              lr=args.base_lr*args.batch_size/256, 
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              )
    
     # Set learning rate scheduler
    lr_scheduler = set_lr_scheduler(optimizer,
                                args.warmup_epochs,
                                args.warmup_lr*args.batch_size/256,
                                args.num_epochs,
                                args.base_lr*args.batch_size/256,
                                args.final_lr*args.batch_size/256, 
                                len(train_loader),
                                True,
                                )
    
    # Fine-tuning classifier
    top1, top5 = 0, 0
    print('\nStarting fine-tuning linear classifier on top of frozen pre-trained backbone')
    writer = SummaryWriter(log_dir=args.logs_dir)
    for epoch in range(1, args.num_epochs+1):
        # Train classifier
        loss, lr = train(train_loader, model, classifier, optimizer, lr_scheduler, epoch, args)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        # Validate classifier
        loss, acc1, acc5 = validate(valid_loader, model, classifier, args)
        writer.add_scalar('valid/loss', loss, epoch)
        writer.add_scalar('valid/acc1', acc1, epoch)
        writer.add_scalar('valid/acc5', acc5, epoch)
        if acc1 > top1:
            top1 = acc1
            top5 = acc5
            print(f'\U00002705 Found higher Acc@1 = {top1:.4f}% where Acc@5 = {top5:.4f}%')
        if epoch % args.save_freq == 0:
            torch.save(classifier.state_dict(), args.ckpt_dir + args.model_name + "_classifier_" + str(epoch) + ".pth")
        
    # Save final classifier
    torch.save(classifier.state_dict(), args.ckpt_dir + args.model_name + "_classifier_final.pth")
    print('Classifier saved successfully under', args.ckpt_dir + args.model_name + "_classifier_final.pth")
    print(f'\n\U00002705 Higher Acc@1 = {top1:.4f}% with Acc@5 = {top5:.4f}%')


if __name__ == '__main__':
    args = parse_args()
    main(args)