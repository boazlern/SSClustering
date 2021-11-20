import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, STL10
from utils.utils import generate_gmm_targets
from PIL import Image


class DynamicDataset(object):
    def __init__(self, with_nat=True, nat_std=0.05, alpha=1, transform=None):
        self.transform = transform
        self.data, self.labels = self.build_data()
        self.num_channels = self.get_num_channels()
        self.num_classes = self.get_num_classes()
        self.transform_mode = None
        self.nat = None
        self.nat_std = nat_std
        self.alpha = alpha
        if with_nat:
            self.initialize_nat()

    def get_num_classes(self):
        raise NotImplementedError

    def get_num_channels(self):
        raise NotImplementedError

    def build_data(self):
        raise NotImplementedError

    def change_transform_mode(self, mode):
        self.transform_mode = mode

    def update_targets(self, indices, new_targets):
        self.nat[indices, :] = new_targets

    def initialize_nat(self, **kwargs):
        self.nat = generate_gmm_targets(n=len(self.data), num_classes=self.num_classes, std=self.nat_std,
                                        alpha=self.alpha, **kwargs)

    def add_images(self, new_images, new_labels):
        self.data = np.vstack([self.data, new_images])
        self.labels = np.hstack([self.labels, new_labels])

    def remove_images(self, indices):
        new_indices = np.ones(len(self.data), dtype=np.bool)
        new_indices[indices] = False
        removed_images = self.data[indices]
        self.data = self.data[new_indices]
        if self.nat is not None:
            self.nat = self.nat[new_indices]
        self.labels = self.labels[new_indices]
        return removed_images

    def get_class_images(self, label):
        class_indices = self.labels == label
        return self.data[class_indices]

    def __getitem__(self, index):

        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, self.transform_mode)

        if self.nat is not None:
            nat = self.nat[index, :]
            return index, img, label, nat

        return img, label


class Cifar10Dataset(CIFAR10, DynamicDataset):

    def __init__(self, root, indices=None, with_nat=True, nat_std=0.05, alpha=1, train=True, transform=None,
                 download=False):
        CIFAR10.__init__(self, root=root, train=train, download=download)
        self.indices = indices
        DynamicDataset.__init__(self, with_nat=with_nat, nat_std=nat_std, alpha=alpha, transform=transform)

    def get_num_classes(self):
        return 10

    def get_num_channels(self):
        return 3

    def build_data(self):
        if self.indices is not None:
            return self.data[self.indices], np.array(self.targets)[self.indices]
        return self.data, np.array(self.targets)

    def __getitem__(self, index):
        return DynamicDataset.__getitem__(self, index)


class Cifar100Dataset(CIFAR100, DynamicDataset):

    def __init__(self, root, indices=None, with_nat=True, nat_std=0.05, alpha=1, train=True, transform=None,
                 download=False):
        CIFAR100.__init__(self, root=root, train=train, download=download)
        self.indices = indices
        DynamicDataset.__init__(self, with_nat=with_nat, nat_std=nat_std, alpha=alpha, transform=transform)

    def get_num_classes(self):
        return 100

    def get_num_channels(self):
        return 3

    def build_data(self):
        if self.indices is not None:
            return self.data[self.indices], np.array(self.targets)[self.indices]
        return self.data, np.array(self.targets)

    def __getitem__(self, index):
        return DynamicDataset.__getitem__(self, index)


class SVHNDataset(SVHN, DynamicDataset):

    def __init__(self, root, indices=None, with_nat=True, nat_std=0.05, alpha=1, split='train', transform=None,
                 download=False):
        SVHN.__init__(self, root=root, split=split, download=download)
        self.indices = indices
        DynamicDataset.__init__(self, with_nat=with_nat, nat_std=nat_std, alpha=alpha, transform=transform)

    def get_num_classes(self):
        return 10

    def get_num_channels(self):
        return 3

    def build_data(self):
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        if self.indices is not None:
            return self.data[self.indices], np.array(self.labels)[self.indices]
        return self.data, np.array(self.labels)

    def __getitem__(self, index):
        return DynamicDataset.__getitem__(self, index)


class STL10Dataset(STL10, DynamicDataset):

    def __init__(self, root, indices=None, with_nat=True, nat_std=0.05, alpha=1, split='train', transform=None,
                 download=False):
        STL10.__init__(self, root=root, split=split, download=download)
        self.indices = indices
        DynamicDataset.__init__(self, with_nat=with_nat, nat_std=nat_std, alpha=alpha, transform=transform)

    def get_num_classes(self):
        return 10

    def get_num_channels(self):
        return 3

    def build_data(self):
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        if self.indices is not None:
            return self.data[self.indices], np.array(self.labels, dtype=np.long)[self.indices]
        return self.data, np.array(self.labels, dtype=np.long)

    def __getitem__(self, index):
        return DynamicDataset.__getitem__(self, index)


class DummyNATDataset(Dataset):
    def __init__(self, dataset, only_images=False):
        self.dataset = dataset
        self.only_images = only_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _, x, y, __ = self.dataset.__getitem__(index)
        if self.only_images:
            return x
        return x, y


class BasicDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform:
            return self.transform(image)
        return image
