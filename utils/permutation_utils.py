import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, euclidean_distances
from torch.utils.data import DataLoader
from utils.dataset_utils import get_basic_dataset


def find_best_switch(mean_distances):
    best_switch = None
    switch_options = np.diag(mean_distances)[:, None] - mean_distances
    ideal_indices = np.where(np.logical_and(switch_options > 0, switch_options.T > 0))

    if len(ideal_indices[0]) > 0:
        print('ideal switch was found. The options were: {}'.format(ideal_indices))
        best_index = switch_options[ideal_indices].argmax()
        best_switch = ideal_indices[0][best_index], ideal_indices[1][best_index]
    else:
        switch_options = switch_options + switch_options.T
        switch_indices = np.where(switch_options > 0)
        if len(switch_indices[0]) > 0 and random.random() < 0.5:
            print('non-ideal switch was found. The options were: {}'.format(switch_indices))
            best_index = switch_options[switch_indices].argmax()
            best_switch = switch_indices[0][best_index], switch_indices[1][best_index]
        elif random.random() < 0.1:
            np.fill_diagonal(switch_options, -np.inf)
            best_switch = np.unravel_index(switch_options.argmax(), switch_options.shape)
            print('no switch was found but we randomly switched: {}'.format(best_switch))
        else:
            print('no switch has been made')

    return best_switch


def get_perm_cost_matrix(model, dataset, transform, rotate=True, n_repetitions=10, j=4, means=None):
    mean_distances = np.zeros((dataset.num_classes, dataset.num_classes))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for c in range(dataset.num_classes):
            class_images = dataset.get_class_images(label=c)
            class_dataset = get_basic_dataset(class_images, transform)
            loader = DataLoader(class_dataset,
                                batch_size=len(class_dataset),
                                shuffle=False,
                                num_workers=j,
                                pin_memory=torch.cuda.is_available())
            for i in range(n_repetitions):
                for x in loader:
                    if isinstance(x, list):
                        x = torch.cat(x, dim=0)
                    x = x.to(device)
                    features, out = model(x, return_features=True)
                    if means is None:
                        distances = pairwise_distances(out.detach().cpu().numpy(), np.eye(dataset.num_classes),
                                                       metric='l1')
                    else:
                        distances = pairwise_distances(features.detach().cpu().numpy(), means, metric='l1')
                    if rotate:
                        mean_distances[c] += 0.1 * np.sum(distances[:4], axis=0) + 0.9 * np.sum(distances[4:], axis=0)
                    else:
                        mean_distances[c] += np.sum(distances, axis=0)

        return mean_distances


def compute_best_labels_permutation(pred_labels, true_labels, num_classes):
    num_correct = np.zeros((num_classes, num_classes))
    for c_1 in range(num_classes):
        for c_2 in range(num_classes):
            num_correct[c_1, c_2] = int(((pred_labels == c_1) * (true_labels == c_2)).sum())

    _, assignment = linear_sum_assignment(len(true_labels) - num_correct)

    reordered_pred_labels = np.zeros(len(true_labels))

    for c in range(num_classes):
        reordered_pred_labels[pred_labels == c] = assignment[c]

    return assignment, reordered_pred_labels


def compute_optimal_target_permutation(feats, targets):
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.

    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    cost_matrix = euclidean_distances(feats, targets)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute the targets based on hungarian algorithm optimisation
    new_targets = targets[col_ind]
    n_switches = np.sum(np.argmax(targets, axis=1) != np.argmax(new_targets, axis=1))
    return n_switches, new_targets


def get_ensemble_permutation(perms):
    ensemble_perm = np.full(perms.shape[1], -1)
    # phase 1: assign label to every class which has majority.
    for i, perm in enumerate(perms.T):
        values, min_perm_indices, counts = np.unique(perm, return_counts=True, return_index=True)
        max_count_indices = np.where(counts == counts.max())[0]
        if len(max_count_indices) == 1:
            ensemble_perm[i] = values[max_count_indices[0]]

    # phase 2: handle ties.
    for j in np.where(ensemble_perm == -1)[0]:
        values, min_perm_indices, counts = np.unique(perms[:, j], return_counts=True, return_index=True)
        max_count_indices = np.where(counts == counts.max())[0]
        tied_values = values[max_count_indices]
        print('there was a tie around model class number: {}. The possible '
              'classes were: {}'.format(j, tied_values))
        used_values = np.isin(tied_values, ensemble_perm)
        if not np.all(used_values):
            max_count_indices = max_count_indices[~used_values]

        ensemble_vote = values[max_count_indices[np.argmin(min_perm_indices[max_count_indices])]]
        ensemble_perm[j] = ensemble_vote
    print('the ensemble perm is: {}'.format(ensemble_perm))
    return ensemble_perm


def find_best_nearby_permutations(perms, perm, k, tolerance=4):
    best_perms = [perm]
    indices = []
    i = 1
    while len(best_perms) < k:
        cur_perm = np.array(perms.__next__()[0])
        num_matches = np.sum(cur_perm == perm)
        if num_matches != len(perm) and (tolerance == 0 or len(perm) - num_matches < tolerance):
            best_perms.append(cur_perm)
            indices.append(i)
        i += 1
    print(indices)
    return best_perms
