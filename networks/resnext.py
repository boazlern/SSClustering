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


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride,
                 cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super().__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = PSBatchNorm2d(D, momentum=0.001)
        self.conv_conv = nn.Conv2d(D, D,
                                   kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False)
        self.bn = PSBatchNorm2d(D, momentum=0.001)
        self.act = mish()
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = PSBatchNorm2d(out_channels, momentum=0.001)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias=False))
            self.shortcut.add_module(
                'shortcut_bn', PSBatchNorm2d(out_channels, momentum=0.001))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = self.act(self.bn_reduce.forward(bottleneck))
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.act(self.bn.forward(bottleneck))
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return self.act(residual + bottleneck)


class CifarResNeXt(MyModel):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, num_classes, grayscale, sobel, cardinality=4, depth=28,
                 base_width=4, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 *
                       self.widen_factor, 256 * self.widen_factor]
        super().__init__(grayscale=grayscale, sobel=sobel)

    def build_features(self):
        return nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(self.num_input_channels, 64, 3, 1, 1, bias=False),
            'bn': PSBatchNorm2d(64, momentum=0.001),
            'relu': mish(),
            'block1': self.block('stage_1', self.stages[0], self.stages[1], 1),
            'block2': self.block('stage_2', self.stages[1], self.stages[2], 2),
            'block3': self.block('stage_3', self.stages[2], self.stages[3], 2),
            'gap': nn.AdaptiveAvgPool2d(1)
        }))

    def get_num_features(self):
        return self.stages[3]

    def build_classifier(self):
        return nn.Linear(self.stages[3], self.num_classes)

    def build_rotnet(self):
        return nn.Sequential(OrderedDict({
            'linear1': nn.Linear(self.stages[3], self.stages[3]),
            'bn1': nn.BatchNorm1d(self.stages[3]),
            'relu1': nn.ReLU(inplace=True),
            'linear2': nn.Linear(self.stages[3], self.stages[3]),
            'bn2': nn.BatchNorm1d(self.stages[3]),
            'relu2': nn.ReLU(inplace=True),
            'linear3': nn.Linear(self.stages[3], 4)
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

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels,
                                                          out_channels,
                                                          pool_stride,
                                                          self.cardinality,
                                                          self.base_width,
                                                          self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels,
                                                   out_channels,
                                                   1,
                                                   self.cardinality,
                                                   self.base_width,
                                                   self.widen_factor))
        return block
