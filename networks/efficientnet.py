############################
##########        ##########
########## to be  ##########
########## fixed  ##########
##########        ##########
############################


import torch
import torch.nn as nn
import math


__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


def RoundRepeats(r):
    return int(math.ceil(r))


def DropPath(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = torch.autograd.Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y
        return y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_channels, eps=1e-3, momentum=0.01),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size, stride, kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels, eps=1e-3, momentum=0.01),
            Swish()
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + DropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    config = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1],
    ]

    def __init__(self, param, stem_channels=32, feature_size=1280, drop_connect_rate=0.2, in_channels=3):
        super(EfficientNet, self).__init__()
        self.in_channels = in_channels

        # scaling width
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = RoundChannels(stem_channels*width_coefficient)
            for conf in self.config:
                conf[0] = RoundChannels(conf[0]*width_coefficient)
                conf[1] = RoundChannels(conf[1]*width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = RoundRepeats(conf[6]*depth_coefficient)

        # scaling resolution
        input_size = param[2]

        # stem convolution
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels, eps=1e-3, momentum=0.01),
            Swish(),
            )

        # total blocks
        total_blocks = 0
        for conf in self.config:
            total_blocks += conf[6]

        # mobile inverted bottleneck
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in self.config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)

        # last several layers
        self.head_conv = nn.Sequential(
            nn.Conv2d(self.config[-1][1], feature_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size, eps=1e-3, momentum=0.01),
            Swish()
        )
        self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.dropout = nn.Dropout(param[3])

        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    
def efficientnet_b0(**kwargs):
    param = (1.0, 1.0, 224, 0.2)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b1(**kwargs):
    param = (1.0, 1.1, 240, 0.2)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b2(**kwargs):
    param = (1.1, 1.2, 260, 0.3)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b3(**kwargs):
    param = (1.2, 1.4, 300, 0.3)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b4(**kwargs):
    param = (1.4, 1.8, 380, 0.4)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b5(**kwargs):
    param = (1.6, 2.2, 456, 0.4)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b6(**kwargs):
    param = (1.8, 2.6, 528, 0.5)
    model = EfficientNet(param, **kwargs)
    return model


def efficientnet_b7(**kwargs):
    param = (2.0, 3.1, 600, 0.5)
    model = EfficientNet(param, **kwargs)
    return model


model_dict = {
    'efficientnet_b0': [efficientnet_b0, 1280],
    'efficientnet_b1': [efficientnet_b1, 1280],
    'efficientnet_b2': [efficientnet_b2, 1280],
    'efficientnet_b3': [efficientnet_b3, 1280],
    'efficientnet_b4': [efficientnet_b4, 999],
    'efficientnet_b5': [efficientnet_b5, 999],
    'efficientnet_b6': [efficientnet_b6, 999],
    'efficientnet_b7': [efficientnet_b7, 999],
}


if __name__ == '__main__':
    for name in model_dict:
        model_fun, dim = model_dict[name]
        model = model_fun()
        x = torch.rand(3, 224, 224) # can pass any size, e.g., 224, 300, 600
        features = model(x.unsqueeze(0))
        print("Passed image of size {} to {}, got features of size {}".format(list(x.unsqueeze(0).shape), name.upper(), list(features.shape)))
        #assert dim == features.shape[1]