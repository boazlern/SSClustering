#!/usr/bin/python
import math
import torch.nn as nn
from networks.my_model import MyModel
from collections import OrderedDict

# Necessary import to be be able to instantiate resnet directly by importing this file.
from torchvision.models.resnet import (
    Bottleneck,
    BasicBlock
)


class ResNet(MyModel):
    """
    Implement Resnet model.
    """

    def __init__(self, block, layers, num_classes, sobel=True, grayscale=True,
                 num_features=512, **kwargs):
        """
        Initialize the model

        :param block: the class used to instantiate a resnet block
        :param layers: the number of layers per block
        :param grayscale: whether the input image is grayscale or RGB.
        :param sobel: whether to use a fixed sobel operator at the start of the network instead of the raw pixels.
        """
        self.inplanes = 64
        self.block = block
        self.layers = layers
        self.num_features = num_features
        self.num_classes = num_classes
        super().__init__(grayscale=grayscale, sobel=sobel)

    def build_features(self):
        return nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(self.num_input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            'bn': nn.BatchNorm2d(self.inplanes),
            'relu': nn.ReLU(inplace=True),
            'max_pool': nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            'layer1': self._make_layer(self.block, self.num_features // 8, self.layers[0]),
            'layer2': self._make_layer(self.block, self.num_features // 4, self.layers[1], stride=2),
            'layer3': self._make_layer(self.block, self.num_features // 2, self.layers[2], stride=2),
            'layer4': self._make_layer(self.block, self.num_features, self.layers[3], stride=2),
            'gap': nn.AdaptiveAvgPool2d(1)
        }))

    def get_num_features(self):
        return self.num_features

    def build_classifier(self):
        return nn.Linear(self.num_features, self.num_classes)

    def build_rotnet(self):
        return nn.Sequential(OrderedDict({
            'linear1': nn.Linear(self.num_features, self.num_features),
            'bn1': nn.BatchNorm1d(self.num_features),
            'relu1': nn.ReLU(inplace=True),
            'linear2': nn.Linear(self.num_features, self.num_features),
            'bn2': nn.BatchNorm1d(self.num_features),
            'relu2': nn.ReLU(inplace=True),
            'linear3': nn.Linear(self.num_features, 4)
        }))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict({
                'conv': nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                'bn': nn.BatchNorm2d(planes * block.expansion),
            }))

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
