import torch.nn as nn
import torch
import numpy as np


class MyModel(nn.Module):
    def __init__(self, grayscale, sobel):
        super().__init__()
        self.grayscale = grayscale
        self.sobel = sobel
        self.num_input_channels = self.get_n_input_channels()
        self.features = self.build_features()
        self.num_features = self.get_num_features()
        self.classifier = self.build_classifier()
        self.rot_net = self.build_rotnet()

        sobel_input_channels = 1 if grayscale else 3

        # Hard coded block that computes the image gradients from grayscale.
        # Not learnt.
        conv_gradients = nn.Conv2d(sobel_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)

        if self.sobel:
            self.pre_processing = nn.Sequential(
                conv_gradients,
            )
            # override weights for preprocessing part.
            dx = np.array([[[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]]], dtype=np.float32)  # TODO maybe normalize the filter (divide by 8).
            dy = np.array([[[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]]], dtype=np.float32)
            _conv_grad = torch.from_numpy(
                np.repeat(
                    np.concatenate([dx, dy])[:, np.newaxis, :, :],
                    sobel_input_channels,
                    axis=1
                )
            )
            conv_gradients.weight = nn.Parameter(data=_conv_grad, requires_grad=False)
        self.initialize_weights()

    def build_features(self):
        raise NotImplementedError

    def get_num_features(self):
        raise NotImplementedError

    def build_classifier(self):
        raise NotImplementedError

    def build_rotnet(self):
        raise NotImplementedError

    def initialize_weights(self):
        raise NotImplementedError

    def get_n_input_channels(self):
        if self.sobel:
            return 2
        elif self.grayscale:
            return 1
        return 3

    def forward(self, x, rot_net=False, return_features=False):
        if self.sobel:
            x = self.pre_processing(x)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        if rot_net:
            if return_features:
                return features, self.rot_net(features)
            return self.rot_net(features)
        if return_features:
            return features, self.classifier(features)
        return self.classifier(features)
