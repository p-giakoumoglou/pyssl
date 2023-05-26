import torch
from torch import nn
from collections import OrderedDict


__all__ = ['densenet121', 'densenet161', 'densenet121', 'densenet201']


class DenseLayer(nn.Module):
    def __init__(self, channels, growth_rate, bottle_neck_size, dropout_rate):
        super(DenseLayer, self).__init__()
        growth_channels = int(bottle_neck_size * growth_rate)
        self.dropout_rate = float(dropout_rate)

        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(channels, growth_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.norm2 = nn.BatchNorm2d(growth_channels)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(growth_channels, growth_rate, (3, 3), (1, 1), (1, 1), bias=False)
        self.dropout = nn.Dropout(dropout_rate, True)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        else:
            x = x
        out = torch.cat(x, 1)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        return out


class DenseBlock(nn.ModuleDict):
    def __init__(self, repeat_times, channels, growth_rate, bottle_neck_size, dropout_rate):
        super(DenseBlock, self).__init__()
        for i in range(repeat_times):
            layer = DenseLayer(channels=channels+i*growth_rate, growth_rate=growth_rate, bottle_neck_size=bottle_neck_size, dropout_rate=dropout_rate)
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x):
        out = [x]
        for _, layer in self.items():
            denselayer_out = layer(out)
            out.append(denselayer_out)
        out = torch.cat(out, 1)
        return out


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.pool = nn.AvgPool2d((2, 2), (2, 2))


class DenseNet(nn.Module):
    def __init__(self, block_cfg=(6, 12, 24, 16), channels=64, growth_rate=32, bottle_neck_size=4, dropout_rate=0.0, in_channels=3):
        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(in_channels, channels, (7, 7), (2, 2), (3, 3), bias=False)),
                    ("norm0", nn.BatchNorm2d(channels)),
                    ("relu0", nn.ReLU(True)),
                    ("pool0", nn.MaxPool2d((3, 3), (2, 2), (1, 1))),
                ]
            )
        )

        for i, repeat_times in enumerate(block_cfg):
            block = DenseBlock(
                repeat_times=repeat_times,
                channels=channels,
                growth_rate=growth_rate,
                bottle_neck_size=bottle_neck_size,
                dropout_rate=dropout_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            channels = channels + int(repeat_times * growth_rate)
            if i != len(block_cfg) - 1:
                trans = Transition(channels, channels // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                channels = channels // 2

        self.features.add_module("norm5", nn.BatchNorm2d(channels))
        self.features.add_module("relu5", nn.ReLU(True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)


def densenet121(**kwargs):
    model = DenseNet((6, 12, 24, 16), 64, 32, **kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet((6, 12, 36, 24), 96, 48, **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet((6, 12, 32, 32), 64, 32, **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet((6, 12, 48, 32), 64, 32, **kwargs)
    return model


model_dict = {
    'densenet121': [densenet121, 1024],
    'densenet161': [densenet161, 2208],
    'densenet169': [densenet169, 1664],
    'densenet201': [densenet201, 1920],
}


if __name__ == '__main__':
    for name in model_dict:
        model_fun, dim = model_dict[name]
        model = model_fun()
        x = torch.rand(3, 224, 224) # can pass any size, e.g., 224, 300, 600
        features = model(x.unsqueeze(0))
        print("Passed image of size {} to {}, got features of size {}".format(list(x.unsqueeze(0).shape), name.upper(), list(features.shape)))
        assert dim == features.shape[1]