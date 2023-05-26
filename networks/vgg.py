import torch
import torch.nn as nn


__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


vgg_cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_cfg, batch_norm=False, in_channels=3):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(vgg_cfg, in_channels, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def _make_layers(self, vgg_cfg, in_channels=3, batch_norm=False):
        layers = nn.Sequential()
        for v in vgg_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d((2, 2), (2, 2)))
            else:
                v = int(v)
                if batch_norm:
                    layers.append(nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1)))
                    layers.append(nn.BatchNorm2d(v))
                    layers.append(nn.ReLU(True))
                else:
                    layers.append(nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1)))
                    layers.append(nn.ReLU(True))
                in_channels = v

        return layers

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def vgg11(**kwargs):
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(vgg_cfgs["vgg11"], True, **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(vgg_cfgs["vgg19"], True, **kwargs)
    return model


model_dict = {
    'vgg11': [vgg11, 25088],
    'vgg13': [vgg13, 25088],
    'vgg16': [vgg16, 25088],
    'vgg19': [vgg19, 25088],
    'vgg11_bn': [vgg11_bn, 25088],
    'vgg13_bn': [vgg13_bn, 25088],
    'vgg16_bn': [vgg16_bn, 25088],
    'vgg19_bn': [vgg19_bn, 25088],
}


if __name__ == '__main__':
    for name in model_dict:
        model_fun, dim = model_dict[name]
        model = model_fun()
        x = torch.rand(3, 224, 224) # can pass any size, e.g., 224, 300, 600
        features = model(x.unsqueeze(0))
        print("Passed image of size {} to {}, got features of size {}".format(list(x.unsqueeze(0).shape), name.upper(), list(features.shape)))
        assert dim == features.shape[1]
