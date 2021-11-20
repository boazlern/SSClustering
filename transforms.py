from torchvision import transforms
from strong_augmentations.rand_augment import RandAugment
from strong_augmentations.ct_augment import CTAugment
import numpy as np
from functools import partial


class MODES:
    CLUSTERING_MODE = 0
    EVAL_MODE = 1
    ROTNET_MODE = 2
    SS_MODE = 3
    TRAIN_MODE = 4


class MultiTransform:
    def __init__(self):
        self.mode_to_func = {MODES.TRAIN_MODE: self.train_mode, MODES.EVAL_MODE: self.eval_mode,
                             MODES.ROTNET_MODE: self.rotnet_mode, MODES.SS_MODE: self.ss_mode}

    def train_mode(self, x):
        raise NotImplementedError

    def eval_mode(self, x):
        raise NotImplementedError

    def rotnet_mode(self, x):
        raise NotImplementedError

    def ss_mode(self, x):
        pass

    def __call__(self, x, modes):
        try:
            if len(modes) == 1:
                return self.mode_to_func[modes[0]](x)
            transformed = []
            for mode in modes:
                transformed.append(self.mode_to_func[mode](x))
            return transformed
        except KeyError:
            print('unknown mode')
            raise KeyError


class UnlabeledTransform(MultiTransform):
    def __init__(self, mean=None, std=None, cta=False, **kwargs):
        super().__init__()
        self.train_transform = []
        self.eval_transform = []
        self.rotnet_transform = []
        self.weak_transform = []
        self.repetitions = kwargs['r'] if 'r' in kwargs else 1
        color_jitter = kwargs['color_jitter']
        if color_jitter > 0:
            self.add_color_jitter(color_jitter)
        if kwargs['h_flip']:
            self.add_h_flip()
        crop_size = kwargs['crop_size']
        if crop_size is not None:
            out_size = kwargs['out_size']
            self.add_crop(crop_size, out_size)

        self.strong_augment = CTAugment() if cta else RandAugment()
        self.strong_transform = self.weak_transform + [self.strong_augment]
        self.build_transforms(grayscale=kwargs['grayscale'], mean=mean, std=std)

    def add_color_jitter(self, color_range):
        self.train_transform.append(transforms.ColorJitter(brightness=color_range, contrast=color_range,
                                                           saturation=color_range, hue=0.125))

    def add_h_flip(self):
        self.train_transform.append(transforms.RandomHorizontalFlip())
        self.weak_transform.append(transforms.RandomHorizontalFlip())

    def add_crop(self, crop_size, out_size):
        if crop_size == out_size:
            random_crop = transforms.RandomCrop(size=crop_size, padding=int(crop_size * 0.125),
                                                padding_mode='reflect')
            self.train_transform.append(random_crop)
            self.rotnet_transform.append(random_crop)
            self.weak_transform.append(random_crop)
        else:
            self.train_transform += [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.rotnet_transform += [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.weak_transform += [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.eval_transform += [transforms.CenterCrop(size=crop_size), transforms.Resize(out_size)]

    def build_transforms(self, grayscale, mean, std):
        my_transforms = [self.train_transform, self.eval_transform, self.rotnet_transform,
                         self.weak_transform, self.strong_transform]
        for i, transform in enumerate(my_transforms):
            if grayscale:
                transform.append(transforms.Grayscale())
            transform.append(transforms.ToTensor())
            if mean is None:
                transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            else:
                transform.append(transforms.Normalize(mean=mean, std=std))
        self.assign_transforms()

    def assign_transforms(self):
        self.train_transform = transforms.Compose(self.train_transform)
        self.eval_transform = transforms.Compose(self.eval_transform)
        self.rotnet_transform = transforms.Compose(self.rotnet_transform)
        self.weak_transform = transforms.Compose(self.weak_transform)
        self.strong_transform = transforms.Compose(self.strong_transform)

    def train_mode(self, x):
        transformed = []
        for r in range(self.repetitions):
            transformed.append(self.train_transform(x))
        return self.eval_transform(x), transformed

    def rotnet_mode(self, x):
        transformed = []
        for i in range(4):
            transformed.append(self.rotnet_transform(x.rotate(i * 90)))
        return transformed

    def eval_mode(self, x):
        return self.eval_transform(x)

    def ss_mode(self, x):
        return self.weak_transform(x), self.strong_transform(x)


class LabeledTransform(MultiTransform):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__()
        self.eval_transform = []
        self.train_transform = []
        self.rotnet_transform = []
        color_jitter = kwargs['color_jitter']
        if color_jitter > 0:
            self.add_color_jitter(color_jitter)
        if kwargs['h_flip']:
            self.add_h_flip()
        crop_size = kwargs['crop_size']
        if crop_size is not None:
            out_size = kwargs['out_size']
            self.add_crop(crop_size, out_size)
        self.build_transforms(grayscale=kwargs['grayscale'], mean=mean, std=std)

    def add_color_jitter(self, color_range):
        self.train_transform.append(transforms.ColorJitter(brightness=color_range, contrast=color_range,
                                                           saturation=color_range, hue=0.125))

    def add_h_flip(self):
        self.train_transform.append(transforms.RandomHorizontalFlip())

    def add_crop(self, crop_size, out_size):
        if crop_size == out_size:
            random_crop = transforms.RandomCrop(size=crop_size, padding=int(crop_size * 0.125),
                                                padding_mode='reflect')
            self.train_transform.append(random_crop)
            self.rotnet_transform.append(random_crop)
        else:
            self.train_transform += [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.rotnet_transform += [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.eval_transform += [transforms.CenterCrop(size=crop_size), transforms.Resize(out_size)]

    def build_transforms(self, grayscale, mean, std):
        my_transforms = [self.train_transform, self.rotnet_transform, self.eval_transform]
        for transform in my_transforms:
            if grayscale:
                transform.append(transforms.Grayscale())
            transform.append(transforms.ToTensor())
            if mean is None:
                transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            else:
                transform.append(transforms.Normalize(mean=mean, std=std))
        self.assign_transforms()

    def assign_transforms(self):
        self.train_transform = transforms.Compose(self.train_transform)
        self.rotnet_transform = transforms.Compose(self.rotnet_transform)
        self.eval_transform = transforms.Compose(self.eval_transform)

    def eval_mode(self, x):
        return self.eval_transform(x)

    def train_mode(self, x):
        return self.train_transform(x)

    def rotnet_mode(self, x):
        transformed = []
        for i in range(4):
            transformed.append(self.rotnet_transform(x.rotate(i * 90)))
        return transformed


class WeakFixMatchTransform(UnlabeledTransform):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)
        self.strong_transform = self.weak_transform
        self.weak_transform = self.eval_transform


class StrongClusteringTransform(UnlabeledTransform):
    def train_mode(self, x):
        transformed = []
        for r in range(self.repetitions):
            transformed.append(self.strong_transform(x))
        return self.weak_transform(x), transformed


class ContrastiveTransform(UnlabeledTransform):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)

    def train_mode(self, x):
        return self.train_transform(x), self.train_transform(x)


class StrongLabeledTransform(UnlabeledTransform):
    def train_mode(self, x):
        return self.strong_transform(x)


class LabeledRotateTransform(LabeledTransform):
    def train_mode(self, x):
        i = np.random.choice(4)
        return self.train_transform(x.rotate(i * 90))


class LabeledCTATransform(LabeledTransform):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)
        self.strong_augment = CTAugment()
        self.strong_transform = []
        if kwargs['h_flip']:
            self.strong_transform.append(transforms.RandomHorizontalFlip())
        crop_size = kwargs['crop_size']
        if crop_size is not None:
            out_size = kwargs['out_size']
            if crop_size == out_size:
                crop = [transforms.RandomCrop(size=crop_size, padding=int(crop_size * 0.125),
                                              padding_mode='reflect')]
            else:
                crop = [transforms.RandomCrop(size=crop_size), transforms.Resize(out_size)]
            self.strong_transform += crop

        self.strong_transform.append(partial(self.strong_augment, probe=True))
        self.strong_transform = transforms.Compose(self.strong_transform)
        self.normalize = []
        if kwargs['grayscale']:
            self.normalize.append(transforms.Grayscale())
        self.normalize.append(transforms.ToTensor())
        if mean is None:
            self.normalize.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        else:
            self.normalize.append(transforms.Normalize(mean=mean, std=std))
        self.normalize = transforms.Compose(self.normalize)

    def train_mode(self, x):
        strong_transform, ops_indices, mag_indices = self.strong_transform(x)
        return self.train_transform(x), (self.normalize(strong_transform), ops_indices, mag_indices)

    def update_rates(self, proximity, ops_indices, mag_indices, print_probs=False):
        self.strong_augment.update(proximity, ops_indices, mag_indices, print_probs)

    def restore_rates(self, rates):
        self.strong_augment.rates = rates

    @property
    def cta_rates(self):
        return self.strong_augment.rates


class UnlabeledCTATransform(UnlabeledTransform):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(mean=mean, std=std, cta=True, **kwargs)

    def update_rates(self, rates):
        self.strong_augment.rates = rates


class CTAClusteringTransform(UnlabeledCTATransform):

    def ss_mode(self, x):
        transformed = []
        for r in range(self.repetitions):
            transformed.append(self.strong_transform(x))
        return self.weak_transform(x), transformed


class MixMatchUnlabeledTransform(UnlabeledTransform):
    def ss_mode(self, x):
        return self.weak_transform(x), self.weak_transform(x)


class MixMatchLabeledTransform(LabeledTransform):
    def add_color_jitter(self, color_range):
        pass


def get_transforms(labeled_name, unlabeled_name, mean, std, **kwargs):
    labeled_transform = None
    unlabeled_transform = None
    if labeled_name is None:
        labeled_transform = LabeledTransform(mean=mean, std=std, **kwargs)
    elif labeled_name == 'strong':
        labeled_transform = StrongLabeledTransform(mean=mean, std=std, **kwargs)
    elif labeled_name == 'rotate':
        labeled_transform = LabeledRotateTransform(mean=mean, std=std, **kwargs)
    elif labeled_name == 'cta':
        labeled_transform = LabeledCTATransform(mean=mean, std=std, **kwargs)
    elif labeled_name == 'mixmatch':
        labeled_transform = MixMatchLabeledTransform(mean=mean, std=std, **kwargs)
    if unlabeled_name is None:
        unlabeled_transform = UnlabeledTransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'weak_fixmatch':
        unlabeled_transform = WeakFixMatchTransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'strong_clustering':
        unlabeled_transform = StrongClusteringTransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'cta':
        unlabeled_transform = UnlabeledCTATransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'cta_clustering':
        unlabeled_transform = CTAClusteringTransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'contrastive':
        unlabeled_transform = ContrastiveTransform(mean=mean, std=std, **kwargs)
    elif unlabeled_name == 'mixmatch':
        unlabeled_transform = MixMatchUnlabeledTransform(mean=mean, std=std, **kwargs)
    return labeled_transform, unlabeled_transform
