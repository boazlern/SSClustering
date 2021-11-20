import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options import EvaluationOptions, TRASNFORMS_ARGS
from easydict import EasyDict as e_dict
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import torch
from utils.dataset_utils import create_partially_labeled_dataset
from networks.utils import get_model
from torch.utils.data import DataLoader
from utils.utils import get_predictions
from utils.permutation_utils import get_perm_cost_matrix, compute_best_labels_permutation, find_best_nearby_permutations, \
    get_ensemble_permutation
from datasets import DummyNATDataset
from transforms import MODES
import time
import matplotlib.pyplot as plt
from functools import partial
from utils.murty_algo import murty
from scipy.stats import mode


def main(extra_args=None):
    if extra_args is None:
        args = e_dict(vars(EvaluationOptions().parse()))
    else:
        parser = EvaluationOptions().parser
        args = e_dict({action.dest: action.default for action in parser._actions})
        args.update(extra_args)
    if args.lab_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.lab_gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpts = args.ckpt if isinstance(args.ckpt, list) else [args.ckpt]
    ckpt = torch.load(ckpts[0], map_location='cpu')
    ckpt_args = e_dict(ckpt['args'])
    labels_permutation = ckpt['labels_permutation'] if 'labels_permutation' in ckpt else None
    print(labels_permutation)
    augmentations = {k: v for k, v in ckpt_args.items() if k in TRASNFORMS_ARGS}
    if 'labeled_transform' not in augmentations:
        augmentations['labeled_transform'] = None
    if 'unlabeled_transform' not in augmentations:
        augmentations['unlabeled_transform'] = None
    labeled_trainset, unlabeled_trainset, validation_set, __ = \
        create_partially_labeled_dataset(name=ckpt_args.dataset, root=args.data_dir, seeds=ckpt['data_seeds'],
                                         n_labels=ckpt_args.n_labels, transductive=args.eval_data == 'all',
                                         **augmentations)
    num_classes = unlabeled_trainset.num_classes
    widen_factor = ckpt_args.widen_factor if 'widen_factor' in ckpt_args else 2
    depth = ckpt_args.depth if 'depth' in ckpt_args else 28
    model = get_model(arch=ckpt_args.arch, num_classes=num_classes,
                      grayscale=ckpt_args.grayscale, sobel=ckpt_args.sobel, **{'dropout': 0.0, 'widen_factor':
                                                                               widen_factor, 'depth': depth})
    model.train(False)

    if args.eval_data == 'validation':
        true_labels = validation_set.labels
        dataset = validation_set
    elif args.eval_data == 'train':
        true_labels = labeled_trainset.labels
        labeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        dataset = labeled_trainset
    else:
        true_labels = unlabeled_trainset.labels
        unlabeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        dataset = DummyNATDataset(unlabeled_trainset)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=torch.cuda.is_available())

    predictions = np.zeros((len(ckpts), len(dataset)), dtype=np.int)
    for i in range(len(ckpts)):
        if i != 0:
            ckpt = torch.load(ckpts[i], map_location='cpu')
            labels_permutation = ckpt['labels_permutation'] if 'labels_permutation' in ckpt else None

        if args.ema:
            if args.us:
                # to support old version where all the names were with 'c'.
                state_str = 'c_ema_state_dict' if 'c_ema_state_dict' in ckpt else 'us_ema_state_dict'
                model_state_dict = ckpt[state_str]
            else:
                # to support old version where all the names were with 'ss'.
                state_str = 'ss_ema_state_dict' if 'ss_ema_state_dict' in ckpt else 's_ema_state_dict'
                model_state_dict = ckpt[state_str]
        else:
            model_state_dict = ckpt['model_state_dict']
            if len({k for k in model_state_dict.keys() if 'rot_net' in k}) == 0:
                rotnet_state = {k: v for k, v in model.state_dict().items() if 'rot_net' in k}
                model_state_dict.update(rotnet_state)

        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model = model.to('cpu')
        model.load_state_dict(model_state_dict)
        model = model.to(device)

        start = time.time()
        pred_labels, max_probs = get_predictions(model=model, loader=loader, device=device, return_probs=True)

        if args.runs_ensemble:
            if args.top_k == 0:
                if labels_permutation is not None:
                    pred_labels = labels_permutation[pred_labels]
                predictions[i] = pred_labels
                continue

        if args.clustering_score:
            nmi_score = normalized_mutual_info_score(pred_labels, true_labels)
            assignment, reordered_pred_labels = compute_best_labels_permutation(pred_labels, true_labels, num_classes)
            acc_score = np.mean(reordered_pred_labels == true_labels)
            print("best permutation is: {}".format(assignment))
            print("the nmi score of the model is: {}".format(nmi_score))
            print("the classification score of the model is: {}".format(acc_score))
        else:
            if args.top_k > 0:
                transform = partial(dataset.transform, modes=[MODES.ROTNET_MODE])
                mean_distances = get_perm_cost_matrix(model, labeled_trainset, transform, j=args.workers)
                assignments = murty(mean_distances.T)
                cur_labels_perm = np.arange(num_classes) if labels_permutation is None else labels_permutation
                k_best_perms = find_best_nearby_permutations(perms=assignments, perm=cur_labels_perm, k=args.top_k,
                                                             tolerance=args.tolerance)
                print(k_best_perms)
                if args.perms_ensemble:
                    ensemble_perm = get_ensemble_permutation(np.stack(k_best_perms, axis=0))
                    classification_acc = np.mean(ensemble_perm[pred_labels] == true_labels)
                    predictions[i] = ensemble_perm[pred_labels]
                else:
                    perms_predictions = np.stack([perm[pred_labels] for perm in k_best_perms])
                    classification_acc = np.max(np.mean(perms_predictions == true_labels[None], axis=1))
                    predictions[i] = cur_labels_perm[pred_labels]
            else:
                if labels_permutation is not None:
                    pred_labels = labels_permutation[pred_labels]
                classification_acc = np.mean(pred_labels == true_labels)
            if extra_args is not None and not args.runs_ensemble:
                return classification_acc
            print("the classification accuracy of the model is: {}".format(classification_acc))

            if not extra_args:
                classes_accuracy = []
                classes_predictions = []
                for i in range(num_classes):
                    class_indices = true_labels == i
                    class_num = np.sum(class_indices)
                    class_accuracy = np.sum(pred_labels[class_indices] == i) / class_num
                    classes_accuracy.append(class_accuracy)

                    class_preds = pred_labels == i
                    classes_predictions.append(np.sum(class_preds))
                    class_most_confident = np.flip(max_probs[class_preds].argsort())[:args.K]
                    num_correct = np.sum(true_labels[class_preds][class_most_confident] == i)
                    print("the model was correct on {}/{} among the most confident images of class {}".format(num_correct,
                                                                                                              args.K, i))
                prediction_distribution = np.array(classes_predictions) / np.sum(classes_predictions)
                print('predictions distribution: {}'.format(prediction_distribution))
                print('the distance from uniform distribution is {}'.format(np.linalg.norm((np.ones(num_classes) / num_classes)
                                                                                           - prediction_distribution)))
                plt.bar(range(num_classes), classes_accuracy, width=1)
                plt.show()

            print("predictions time took {} seconds".format(time.time() - start))

    return np.mean(mode(predictions).mode == true_labels)


if __name__ == '__main__':
    main()
