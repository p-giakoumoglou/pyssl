import os
import warnings
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dotmap import DotMap
from tqdm import tqdm
import pyfiglet
warnings.simplefilter("ignore")

import networks
import transformations
from main import set_model, set_optimizer, set_lr_scheduler
from utils import calculate_topk_accuracy, AverageMeter
from utils import set_deterministic, set_all_seeds


def parse_arguements():
    """ Arguements for linear evaluation, we use DotMap rather than argparse """
    args = DotMap()
    
    # General
    args.debug = True
    args.data_dir = './data/'
    args.ckpt_dir = './ckpts/'
    args.logs_dir = './logs/'
    args.device = 'cuda'
    args.seed = 123
    
    # SimSiam
    args.model.name = "simsiam" 
    args.model.backbone = 'resnet18'
    args.model.save_freq = 10
    
    # Dataset
    args.dataset.name = "cifar10"
    args.dataset.image_size = 32 
    args.dataset.batch_size = 256
    args.dataset.num_workers = 6
    args.dataset.num_classes = 10
       
    # Evaluation - Optimizer
    args.eval.optimizer = "sgd"
    args.eval.weight_decay = 0
    args.eval.momentum = 0.9
    
    # Evaluation - Scheduler
    args.eval.warmup_epochs = 0
    args.eval.warmup_lr = 0
    args.eval.base_lr = 30
    args.eval.final_lr = 0
    args.eval.num_epochs = 100
    
    if args.debug:
        args.eval.num_epochs = 5
        
    print(pyfiglet.figlet_format(args.model.name.capitalize()))
    set_deterministic(args.seed)
    set_all_seeds(args.seed)
    print(f'Using {args.device.upper()}')
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)
    if not os.path.exists(args.logs_dir): os.makedirs(args.logs_dir)

    return args


def set_loader(args):
    """ Data loaders for the training and validation on CIFAR 10 """
    if args.dataset.name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transformations.TRANSFORM_LINEAR)
        valid_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transformations.TRANSFORM_EVAL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.dataset.batch_size), shuffle=True, num_workers=args.dataset.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(args.dataset.batch_size), shuffle=False, num_workers=args.dataset.num_workers)
    else:
        raise NotImplementedError(f'\U0000274C Dataset {args.dataset.name} not implemented')
    print(f'Initialized loaders for {args.dataset.name.upper()}')
    return train_loader, valid_loader

    
def train(loader, model, classifier, optimizer, lr_scheduler, epoch, args):
    """ 
    One epoch fine-tuning: Pass images through pre-trained frozen backbone and linear classifier
    """
    model.eval()
    classifier.train()
    
    losses = AverageMeter('loss')
    lrs = AverageMeter('lr')
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}')
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
    
    pbar = tqdm(loader, desc='Evaluating')
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
    
    model.load_state_dict(torch.load(args.ckpt_dir + args.model.name + "_final.pth"))
    
    classifier = networks.LinearClassifier(feature_size, args.dataset.num_classes).to(args.device)
    
    # Set optimizer
    optimizer = set_optimizer(args.eval.optimizer,
                              classifier,
                              lr=args.eval.base_lr*args.dataset.batch_size/256, 
                              momentum=args.eval.momentum,
                              weight_decay=args.eval.weight_decay,
                              )
    
     # Set learning rate scheduler
    lr_scheduler = set_lr_scheduler(optimizer,
                                args.eval.warmup_epochs,
                                args.eval.warmup_lr*args.dataset.batch_size/256,
                                args.eval.num_epochs,
                                args.eval.base_lr*args.dataset.batch_size/256,
                                args.eval.final_lr*args.dataset.batch_size/256, 
                                len(train_loader),
                                True,
                                )
    
    # Fine-tuning classifier
    print('\nStarting fine-tuning linear classifier on top of frozen pre-trained backbone')
    writer = SummaryWriter(log_dir=args.logs_dir)
    for epoch in range(1, args.eval.num_epochs+1):
        # Train classifier
        loss, lr = train(train_loader, model, classifier, optimizer, lr_scheduler, epoch, args)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        # Validate classifier
        loss, acc1, acc5 = validate(valid_loader, model, classifier, args)
        writer.add_scalar('valid/loss', loss, epoch)
        writer.add_scalar('valid/acc1', lr, epoch)
        writer.add_scalar('valid/acc5', lr, epoch)
        if epoch % args.model.save_freq == 0:
            torch.save(classifier.state_dict(), args.ckpt_dir + args.model.name + "_classifier_" + str(epoch) + ".pth")
        
    # Save final classifier
    torch.save(classifier.state_dict(), args.ckpt_dir + args.model.name + "_classifier_final.pth")
    print('Classifier saved successfully under', args.ckpt_dir + args.model.name + "_classifier_final.pth")


if __name__ == '__main__':
    args = parse_arguements()
    main(args)