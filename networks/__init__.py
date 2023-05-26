from .alexnet import alexnet
from .densenet import densenet121, densenet161, densenet169, densenet201
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .classifier import LinearClassifier


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
		   'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'
		   'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
		   'densenet121', 'densenet161', 'densenet121', 'densenet201',
		   'alexnet',
           'LinearClassifier',
		   ]


model_dict = {
    'alexnet': [alexnet, 9216],
    'densenet121': [densenet121, 1024],
    'densenet161': [densenet161, 2208],
    'densenet169': [densenet169, 1664],
    'densenet201': [densenet201, 1920],
    'efficientnet_b0': [efficientnet_b0, 1280],
    'efficientnet_b1': [efficientnet_b1, 1280],
    'efficientnet_b2': [efficientnet_b2, 1280],
    'efficientnet_b3': [efficientnet_b3, 1280],
    'efficientnet_b4': [efficientnet_b4, 999],
    'efficientnet_b5': [efficientnet_b5, 999],
    'efficientnet_b6': [efficientnet_b6, 999],
    'efficientnet_b7': [efficientnet_b7, 999],
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet152': [resnet152, 2048],
    'vgg11': [vgg11, 25088],
    'vgg13': [vgg13, 25088],
    'vgg16': [vgg16, 25088],
    'vgg19': [vgg19, 25088],
    'vgg11_bn': [vgg11_bn, 25088],
    'vgg13_bn': [vgg13_bn, 25088],
    'vgg16_bn': [vgg16_bn, 25088],
    'vgg19_bn': [vgg19_bn, 25088],
}

#model_fun, dim_in = model_dict['resnet18']
#backbone = model_fun()