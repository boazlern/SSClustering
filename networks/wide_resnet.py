import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.my_model import MyModel
from collections import OrderedDict


class mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = PSBatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = mish()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = PSBatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = mish()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(MyModel):
    def __init__(self, num_classes, grayscale, sobel, **kwargs):
        self.n = 4
        self.num_blocks = kwargs['depth'] % 8 - 1
        self.channels = [16] + [16 * kwargs['widen_factor'] << i for i in range(self.num_blocks)]
        assert(self.num_blocks * 9 + 1 == kwargs['depth'])
        self.block = BasicBlock
        self.drop_rate = kwargs['dropout']
        self.num_classes = num_classes
        super().__init__(grayscale=grayscale, sobel=sobel)

    def build_features(self):
        modules_dict = OrderedDict({'conv': nn.Conv2d(self.num_input_channels, self.channels[0], kernel_size=3,
                                                      stride=1, padding=1, bias=False)})
        for i in range(1, self.num_blocks + 1):
            if i == 1:
                cur_block = NetworkBlock(self.n, self.channels[i - 1], self.channels[i], self.block, 1, self.drop_rate,
                                         activate_before_residual=True)
            else:
                cur_block = NetworkBlock(self.n, self.channels[i - 1], self.channels[i], self.block, 2, self.drop_rate)

            modules_dict['block{}'.format(i)] = cur_block
        modules_dict.update(OrderedDict({'bn': PSBatchNorm2d(self.channels[-1], momentum=0.001), 'relu': mish(),
                                         'gap': nn.AdaptiveAvgPool2d(1)}))
        return nn.Sequential(modules_dict)

    def get_num_features(self):
        return self.channels[-1]

    def build_classifier(self):
        return nn.Linear(self.channels[-1], self.num_classes)

    def build_rotnet(self):
        return nn.Sequential(OrderedDict({
            'linear1': nn.Linear(self.channels[-1], self.channels[-1]),
            'bn1': nn.BatchNorm1d(self.channels[-1]),
            'relu1': nn.ReLU(inplace=True),
            'linear2': nn.Linear(self.channels[-1], self.channels[-1]),
            'bn2': nn.BatchNorm1d(self.channels[-1]),
            'relu2': nn.ReLU(inplace=True),
            'linear3': nn.Linear(self.channels[-1], 4)
        }))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, PSBatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
