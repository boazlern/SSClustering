# #!/usr/bin/python
from datasets import *
from transforms import get_transforms, MODES
from options import DATASETS
import os

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = [0.4376, 0.4437, 0.4728]
svhn_std = [0.1980, 0.2010, 0.1970]
stl10_mean = [0.4376, 0.4437, 0.4728]
stl10_std = [0.1980, 0.2010, 0.1970]
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def create_partially_labeled_dataset(name, root, n_labels, alpha=1, transductive=False, nat_std=0.05,
                                     seeds=None, **kwargs):
    assert name in DATASETS
    dataset_func = {'cifar10': load_cifar10, 'cifar100': load_cifar100, 'svhn': load_svhn, 'stl10': load_stl10}
    return dataset_func[name](root=root, n_labels=n_labels, transductive=transductive, alpha=alpha, seeds=seeds,
                              nat_std=nat_std, **kwargs)


def fixmatch_partition(labels, n_labels, n_classes, seed, svhn):
    '''
    This function is taken from FixMatch in order to use the exact same partitions for labeled and unlabeled data.
    '''
    np.random.seed(seed)
    class_indices = {}
    if svhn:  # hack because in fixmatch the labels are shifted: 0 -> 1, 1 -> 2, ..., 9 -> 0
        classes = list(range(1, n_classes)) + [0]
    else:
        classes = list(range(n_classes))
    for i in classes:
        cur_indices = np.where(labels == i)[0]
        np.random.shuffle(cur_indices)
        class_indices[i] = cur_indices
    train_stats = np.array([len(class_indices[i]) for i in range(n_classes)], np.float64)
    train_stats /= train_stats.max()

    npos = np.zeros(n_classes, np.int64)
    labeled_indices = []
    for i in range(n_labels):
        c = np.argmax(train_stats - npos / max(npos.max(), 1))
        labeled_indices.append(class_indices[c][npos[c]])
        npos[c] += 1
    labeled_indices = sorted(labeled_indices)  # for having the same order as fixmatch - easy comparison.
    return labeled_indices, seed


def partition_dataset(labels, n_labels, n_classes, seeds=None, svhn=False):
    labels = np.array(labels)
    if isinstance(seeds, int):  # we want exact FixMatch partition.
        return fixmatch_partition(labels, n_labels, n_classes, seeds, svhn)
    labels_per_class = n_labels // n_classes
    remaining_labels = n_labels % n_classes
    labeled_indices = []
    random_seeds = seeds is None or seeds.size == 0
    new_seeds = np.zeros((n_classes, labels_per_class), dtype=int) if random_seeds else seeds
    for i in range(n_classes):
        class_indices = np.where(labels == i)[0]
        if random_seeds:
            chosen_indices = np.random.choice(len(class_indices), labels_per_class, replace=False)
            class_labeled_indices = class_indices[chosen_indices]
            labeled_indices.extend(class_labeled_indices)
            new_seeds[i] = chosen_indices
        elif len(seeds.shape) == 1:  # support old version of shuffling with random seeds
            np.random.seed(seeds[i])
            np.random.shuffle(class_indices)
            remaining_label = i < remaining_labels
            labeled_indices.extend(class_indices[:labels_per_class + remaining_label])
        else:  # the seeds are indices of the class indices
            labeled_indices.extend(class_indices[seeds[i, :labels_per_class]])
    return labeled_indices, new_seeds


def load_cifar10(root, n_labels, transductive, nat_std, alpha, seeds=None, **kwargs):
    kwargs['out_size'] = 32
    mean = cifar10_mean if kwargs['normalize_data'] else None
    std = cifar10_std if kwargs['normalize_data'] else None
    labeled_transform, unlabeled_transform = get_transforms(labeled_name=kwargs['labeled_transform'],
                                                            unlabeled_name=kwargs['unlabeled_transform'],
                                                            mean=mean, std=std, **kwargs)
    base_dataset = CIFAR10(root, train=True, download=True)
    labeled_indices, seeds = partition_dataset(labels=base_dataset.targets, n_labels=n_labels, n_classes=10,
                                               seeds=seeds)
    labeled_trainset = Cifar10Dataset(root=root, indices=labeled_indices, with_nat=False, train=True,
                                      transform=labeled_transform)
    unlabeled_trainset = Cifar10Dataset(root=root, nat_std=nat_std, train=True, alpha=alpha,
                                        transform=unlabeled_transform)
    validation_set = Cifar10Dataset(root=root, train=False, transform=labeled_transform,
                                    with_nat=False, download=False)
    validation_set.change_transform_mode(mode=[MODES.EVAL_MODE])
    if transductive:
        unlabeled_trainset.add_images(new_images=validation_set.data, new_labels=validation_set.labels)
        unlabeled_trainset.initialize_nat()
    return labeled_trainset, unlabeled_trainset, validation_set, seeds


def load_cifar100(root, n_labels, transductive, nat_std, alpha, seeds=None, **kwargs):
    kwargs['out_size'] = 32
    mean = cifar100_mean if kwargs['normalize_data'] else None
    std = cifar100_std if kwargs['normalize_data'] else None
    labeled_transform, unlabeled_transform = get_transforms(labeled_name=kwargs['labeled_transform'],
                                                            unlabeled_name=kwargs['unlabeled_transform'],
                                                            mean=mean, std=std, **kwargs)
    base_dataset = CIFAR100(root, train=True, download=True)
    labeled_indices, seeds = partition_dataset(labels=base_dataset.targets, n_labels=n_labels, n_classes=100,
                                               seeds=seeds)
    labeled_trainset = Cifar100Dataset(root=root, indices=labeled_indices, with_nat=False, train=True,
                                       transform=labeled_transform)
    unlabeled_trainset = Cifar100Dataset(root=root, nat_std=nat_std, alpha=alpha, train=True,
                                         transform=unlabeled_transform)
    validation_set = Cifar100Dataset(root=root, train=False, transform=labeled_transform,
                                     with_nat=False, download=False)
    validation_set.change_transform_mode(mode=[MODES.EVAL_MODE])
    if transductive:
        unlabeled_trainset.add_images(new_images=validation_set.data, new_labels=validation_set.labels)
        unlabeled_trainset.initialize_nat()
    return labeled_trainset, unlabeled_trainset, validation_set, seeds


def load_svhn(root, n_labels, transductive, nat_std, alpha, seeds=None, **kwargs):
    kwargs['out_size'] = 32
    mean = svhn_mean if kwargs['normalize_data'] else None
    std = svhn_std if kwargs['normalize_data'] else None
    labeled_transform, unlabeled_transform = get_transforms(labeled_name=kwargs['labeled_transform'],
                                                            unlabeled_name=kwargs['unlabeled_transform'],
                                                            mean=mean, std=std, **kwargs)
    base_dataset = SVHN(root, split='train', download=True)
    labeled_indices, seeds = partition_dataset(labels=base_dataset.labels, n_labels=n_labels, n_classes=10,
                                               seeds=seeds, svhn=True)
    labeled_trainset = SVHNDataset(root=root, indices=labeled_indices, with_nat=False, transform=labeled_transform)
    unlabeled_trainset = SVHNDataset(root=root, nat_std=nat_std, alpha=alpha, transform=unlabeled_transform)
    validation_set = SVHNDataset(root=root, split='test', transform=labeled_transform,
                                 with_nat=False, download=True)
    validation_set.change_transform_mode(mode=[MODES.EVAL_MODE])
    if transductive:
        unlabeled_trainset.add_images(new_images=validation_set.data, new_labels=validation_set.labels)
        unlabeled_trainset.initialize_nat()
    return labeled_trainset, unlabeled_trainset, validation_set, seeds


def load_stl10(root, n_labels, transductive, nat_std, alpha, seeds=None, **kwargs):
    kwargs['out_size'] = 96
    mean = stl10_mean if kwargs['normalize_data'] else None
    std = stl10_std if kwargs['normalize_data'] else None
    labeled_transform, unlabeled_transform = get_transforms(labeled_name=kwargs['labeled_transform'],
                                                            unlabeled_name=kwargs['unlabeled_transform'],
                                                            mean=mean, std=std, **kwargs)
    base_dataset = STL10(root, split='train', download=True)
    if isinstance(seeds, int):
        with open(os.path.join(root, 'stl10_fold_indices.txt'), 'r') as fold_indices:
            data = fold_indices.read()
            labeled_indices = sorted(list(map(int, data.split('\n')[seeds].split())))
            if n_labels < 1000:
                labeled_set = STL10Dataset(root=root, indices=labeled_indices, with_nat=False)
                in_labeled_indices, seeds = partition_dataset(labels=labeled_set.labels, n_labels=n_labels,
                                                              n_classes=10, seeds=seeds)
                labeled_indices = np.array(labeled_indices)[in_labeled_indices]
    else:
        labeled_indices, seeds = partition_dataset(labels=base_dataset.labels, n_labels=n_labels,
                                                   n_classes=10, seeds=seeds)
    labeled_trainset = STL10Dataset(root=root, indices=labeled_indices, with_nat=False, transform=labeled_transform)
    # adding the labeled images to the unlabeled set and then initialize targets to all data.
    unlabeled_trainset = STL10Dataset(root=root, split='unlabeled', with_nat=False, nat_std=nat_std, alpha=alpha,
                                      transform=unlabeled_transform)
    all_labeled_images = np.transpose(base_dataset.data, (0, 2, 3, 1))
    unlabeled_trainset.add_images(new_images=all_labeled_images, new_labels=base_dataset.labels)
    unlabeled_trainset.initialize_nat()
    validation_set = STL10Dataset(root=root, split='test', transform=labeled_transform,
                                  with_nat=False, download=True)
    validation_set.change_transform_mode(mode=[MODES.EVAL_MODE])
    if transductive:
        unlabeled_trainset.add_images(new_images=validation_set.data, new_labels=validation_set.labels)
        unlabeled_trainset.initialize_nat()
    return labeled_trainset, unlabeled_trainset, validation_set, seeds


def get_basic_dataset(data, transform=None):
    return BasicDataset(data, transform)
