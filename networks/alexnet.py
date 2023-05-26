import torch
import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model


model_dict = {
    'alexnet': [alexnet, 9216],
}


if __name__ == '__main__':
    for name in model_dict:
        model_fun, dim = model_dict[name]
        model = model_fun()
        x = torch.rand(3, 224, 224) # can pass any size, e.g., 224, 300, 600
        features = model(x.unsqueeze(0))
        print("Passed image of size {} to {}, got features of size {}".format(list(x.unsqueeze(0).shape), name.upper(), list(features.shape)))
        assert dim == features.shape[1]