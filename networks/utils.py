from networks.resnet import ResNet, BasicBlock, Bottleneck
from networks.wide_resnet import WideResNet
from networks.resnext import CifarResNeXt


def get_model(arch, num_classes, grayscale, sobel, **model_specifics):
    if arch == 'resnet' or arch == 'resnet34':
        return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes,
                      grayscale=grayscale, sobel=sobel, **model_specifics)
    elif arch == 'resnet18':
        return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes,
                      grayscale=grayscale, sobel=sobel, **model_specifics)
    elif arch == 'wide_resnet':
        return WideResNet(num_classes=num_classes, grayscale=grayscale, sobel=sobel, **model_specifics)
    elif arch == 'resnext':
        return CifarResNeXt(num_classes=num_classes, grayscale=grayscale, sobel=sobel)
    else:
        raise NotImplementedError
