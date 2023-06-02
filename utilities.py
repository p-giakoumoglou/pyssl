import torch
import torchvision
import networks
import models
import transformations
import optimizers


def set_loader(args, use_two_crop=True):
    """ Data loaders for the training and validation on CIFAR 10 and CIFAR 100 """
    if args.dataset_name == "cifar10":
        if use_two_crop:
            transform = transformations.transform_dict[args.model_name.lower()]
            train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transformations.TwoCropTransform(*transform))
        else:
            train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transformations.TRANSFORM_LINEAR)
            assert args.num_classes == 10, "Chose CIFAR-10, but the number of classes is not 10"
        valid_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transformations.TRANSFORM_EVAL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset_name == "cifar100":
        if use_two_crop:
            transform = transformations.transform_dict[args.model_name.lower()]
            train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transformations.TwoCropTransform(*transform))
        else:
            train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transformations.TRANSFORM_LINEAR)
            assert args.num_classes == 100, "Chose CIFAR-100, but the number of classes is not 100"
        valid_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transformations.TRANSFORM_EVAL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
    else:
        raise NotImplementedError(f'\U0000274C Dataset {args.dataset_name} not implemented')
    print(f'Loaders for {args.dataset_name.upper()} initialized')
    return train_loader, valid_loader


def set_model(args):
    """
    Self-Supervised Learning Models
    """
    model_fun, feature_size = networks.model_dict[args.backbone]
    backbone = model_fun()
    print(f'\U0001F680 Backbone {args.backbone.upper()} initialized')
    
    if args.model_name.lower() == "simsiam":
        model = models.SimSiam(backbone, feature_size) 
    elif args.model_name.lower() == "byol":
        model = models.BYOL(backbone, feature_size) 
    elif args.model_name.lower() == "simclr":
        model = models.SimCLR(backbone, feature_size) 
    elif args.model_name.lower() == "supcon":
        model = models.SupCon(backbone, feature_size) 
        print('\U000026A0 Using SupCon, labels are parsed in loss calculation')
    else:
        NotImplementedError(f'Model {args.model_name.upper()} not implemented')
    model = model.to(args.device)
    print(f'\U0001F680 Model {args.model_name.upper()} initialized')
    
    return model, feature_size


def set_optimizer(optimizer_name, model, lr, momentum, weight_decay):
    """ Optimizer for Self-Supervised Learning """
    parameters = model.parameters() 
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        txt = f'(lr={lr}, mom={momentum}, wd={weight_decay})'
    elif optimizer_name.lower() == 'sgd_nesterov':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        txt = f'with Nesterov momentum (lr={lr}, mom={momentum}, wd={weight_decay})'
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        txt = f'(lr={lr}, wd={weight_decay})'
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        txt = f'(lr={lr}, wd={weight_decay})'
    elif optimizer_name.lower() == 'larc':
        optimizer = optimizers.LARC(torch.optim.SGD(parameters, lr=lr,  momentum=momentum, weight_decay=weight_decay), trust_coefficient=0.001, clip=False)
        txt = f'(lr={lr}, mom={momentum}, wd={weight_decay}, trust_coefficient=0.001, clip=False)'
    elif optimizer_name.lower() == 'lars':
        optimizer = optimizers.LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        txt = f'(lr={lr}, wd={weight_decay})'
    elif optimizer_name.lower() == 'lars_simclr':
        optimizer = optimizers.LARS_SimCLR(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        txt = f'(lr={lr}, mom={momentum}, wd={weight_decay})'
    else:
        NotImplementedError(f'Optimizer {optimizer_name.upper()} not implemented')
    print(f'Optimizer {optimizer.__class__.__name__} initialized', txt)
    return optimizer