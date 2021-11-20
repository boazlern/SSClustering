import numpy as np
import random
import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def generate_random_targets(n, z):
    """
    Generate a matrix of random target assignment.
    Each target assignment vector has unit length (hence can be view as random point on hypersphere)
    :param n: the number of samples to generate.
    :param z: the latent space dimensionality
    :return: the sampled representations
    """

    # Generate random targets using gaussian distrib.
    samples = np.random.normal(0, 1, (n, z)).astype(np.float32)
    # rescale such that fit on unit sphere.
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples), axis=1)), 1)
    # return rescaled targets
    return samples/radiuses


def generate_gmm_targets(n, num_classes, predictions=None, std=0.05, normalize=True, alpha=1):
    """
    :param n: number of samples to generate
    :param dim: dimension of Z
    :param num_classes: number of classes\components
    :param std: variance
    :param normalize: boolean flag of whether or not to normalize the samples (to the unit sphere)
    :param alpha: the ratio of real targets out of the n targets. All the rest will be zeros.
    :return: a numpy array (n x dim_z) of the generated samples clustered by the components.
    """
    real_targets = int(n * alpha)
    c_size = real_targets // num_classes
    remainder = real_targets % num_classes

    means = np.eye(num_classes)
    # covs = [np.diag([std] * num_classes) for _ in range(num_classes)]  # uniform diagonal covariance matrix

    samples = np.zeros((n, num_classes), dtype=np.float32)
    for i in range(num_classes):
        if i < remainder:
            start_index = i * c_size + i
            end_index = start_index + c_size + 1
            size = c_size + 1
        else:
            start_index = i * c_size + remainder
            end_index = start_index + c_size
            size = c_size
        samples[start_index: end_index] = np.random.normal(means[i], std, size=(size, num_classes))

    np.random.shuffle(samples)
    if normalize:
        norm = np.linalg.norm(samples, axis=1)
        norm[norm == 0] = 1
        return samples / norm[:, None]

    return samples


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_seed(gpu, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def get_predictions(model, loader, device, return_probs=False):
    with torch.no_grad():
        pred_labels = np.zeros(len(loader.dataset), dtype=np.int)
        probs = np.zeros(len(loader.dataset))
        i = 0
        for x, y in loader:
            x = x.to(device)
            out = torch.softmax(model(x), dim=-1)
            cur_probs, cur_pred = torch.max(out, dim=1)
            pred_labels[i: i + len(x)] = cur_pred.detach().cpu().numpy()
            probs[i: i + len(x)] = cur_probs.detach().cpu().numpy()
            i += len(x)
    if return_probs:
        return pred_labels, probs
    return pred_labels


def find_clusters_mean_features(model, loader, num_classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean_features = np.zeros((num_classes, model.num_features))
    num_samples = np.zeros(num_classes)

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            features, out = model(x, return_features=True)
            predictions = np.argmax(out.detach().cpu().numpy(), axis=1)
            features = features.detach().cpu().numpy()
            for i in range(len(predictions)):
                pred = predictions[i]
                num_samples[pred] += 1
                mean_features[pred] += features[i]
        mean_features /= num_samples[:, None]
    return mean_features
