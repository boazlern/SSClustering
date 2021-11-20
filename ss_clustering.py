#!/usr/bin/python
import json
import sys
import os
import string
import torch.nn as nn
from datasets import DummyNATDataset
from utils.dataset_utils import create_partially_labeled_dataset
import sklearn.cluster
from utils.permutation_utils import find_best_switch, get_perm_cost_matrix, compute_best_labels_permutation
from transforms import MODES
from networks.utils import get_model
from utils.ema import ModelEMA
from utils.utils import *
from utils.schedulers import WarmupCosineLrScheduler, LinearScheduler
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from options import TRASNFORMS_ARGS, RESTORE_ARGS, OPTIMIZER_ARGS, MODEL_SPECIFICS_ARGS
from functools import partial
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

NUM_ROTATIONS = 4
EXPERIMENTS_FOLDER = 'experiments'


class SemiSupervisedClustering:
    def __init__(self, args):
        self.args = args
        if 'cta' in self.args.s_algo or self.args.s_algo == 'remixmatch':  # all algorithms which use cta
            if self.args.s_algo == 'cta_clustering':
                self.args.unlabeled_transform = 'cta_clustering'
            else:
                self.args.unlabeled_transform = 'cta'
            self.args.labeled_transform = 'cta'
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.initial_iteration = 0
        self.data_seeds = None
        self.labels_permutation = None  # for going from labels predicted by the model to normal labels.
        self.best_classifying_acc = 0.0
        self.best_clustering_acc = 0.0
        self.us_lowest_loss = np.inf
        self.s_lowest_loss = np.inf
        self.cur_dir = ''
        ckpt = None
        if self.args.resume:
            ckpt = self.load_checkpoint()
            if ckpt:
                self.restore_args(ckpt)
        self.create_dirs()

        self.log_file = open(os.path.join(self.log_dir, 'prints.txt'), "a")
        transform_args = {k: v for k, v in self.args.items() if k in TRASNFORMS_ARGS}
        self.update_data_seeds()
        self.labeled_trainset, self.unlabeled_trainset, self.validation_set, self.data_seeds = \
            create_partially_labeled_dataset(name=self.args.dataset, root=self.args.data_dir,
                                             n_labels=self.args.n_labels, alpha=self.args.alpha,
                                             transductive=self.args.transductive, nat_std=self.args.nat_std,
                                             seeds=self.data_seeds, **transform_args)
        self.update_nat(ckpt)
        if self.args.debug:  # debug mode
            self.validation_set.data = self.validation_set.data[:100]
            self.validation_set.labels = self.validation_set.labels[:100]
            self.unlabeled_trainset.data = self.unlabeled_trainset.data[:1000]
            self.unlabeled_trainset.labels = self.unlabeled_trainset.labels[:1000]
            self.args.workers = 0
        self.val_loader = DataLoader(self.validation_set,
                                     batch_size=512,
                                     shuffle=False,
                                     num_workers=self.args.workers,
                                     pin_memory=False)

        self.num_classes = self.labeled_trainset.num_classes
        self.model, self.us_optim, self.s_optim = self.build_models(ckpt)
        self.s_scheduler, self.us_scheduler = self.get_schedulers()
        self.s_ema, self.us_ema = self.get_ema(ckpt)

        self.us_loss_fn = nn.MSELoss().to(self.device)
        self.s_loss_fn = nn.CrossEntropyLoss().to(self.device)

        if self.args.estimate_perm and self.args.resume:
            # used when we're resuming from clustering, and want to add labels (use semi-supervised algorithms).
            # So we start by estimating the permutation between labels and clusters. If 'estimate_perm' is off, we
            # basically try to fit the clustering model learned to the identity permutation.
            self.initial_perm_estimation()
        # we want to call the method from the teacher and not from the student.  Not really used anymore.
        if (self.args.K > 0 and self.args.teacher_path is None) or self.args.save_pseudo:
            self.pseudo_label()
        self.restore_cta_rates(ckpt)
        if torch.cuda.device_count() > 1:
            print('gpu count = ', torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.save_args()

    def save_args(self):
        with open(self.cur_dir + '/args.txt', 'w') as outfile:
            json.dump(self.args, outfile, indent=4)

    def update_data_seeds(self):
        # data seeds can be one of the following:
        # 1. a model .pkl file path for loading partition we used in a different experiment.
        # 2. 'random'. used when we resume and want to use a different seed than the one used in the resumed model.
        # 3. a model .pkl file path for loading partition we used in a different experiment.
        # 4. .npy numpy file path. If the matrix is 1-dimensional, it is treated as random seeds for each class.
        #    If it's 2-dimensional, it's treated as indices of the class indices.
        # 5. a number. Then, it is used to extract the exact same partition as in FixMatch.
        if self.args.data_seeds is not None:
            if self.args.data_seeds.endswith('pkl'):
                self.data_seeds = torch.load(self.args.data_seeds, map_location='cpu')['data_seeds']
            elif self.args.data_seeds == 'random':
                self.data_seeds = None
            elif self.args.data_seeds.endswith('npy'):
                self.data_seeds = np.load(self.args.data_seeds)  # numpy file
            elif self.args.data_seeds.isdigit():
                self.data_seeds = int(self.args.data_seeds)

    def update_nat(self, ckpt):
        # load learned targets of the resumed model.
        if ckpt is not None and 'nat' in ckpt:
            self.unlabeled_trainset.nat = ckpt['nat']

    def restore_cta_rates(self, ckpt):
        # load cta rates learned by the resumed model.
        if ckpt is not None and 'cta_rates' in ckpt and 'cta' in self.args.s_algo:
            cta_rates = ckpt['cta_rates']
            if cta_rates is not None:
                self.labeled_trainset.transform.restore_rates(cta_rates)

    def restore_args(self, ckpt):
        """
        Restore the argument based on the state of the checkpoint.
        :param ckpt: the checkpoint object.
        :return:
        """
        if ckpt is not None:
            saved_args = ckpt['args']
            for arg in saved_args:
                if arg in RESTORE_ARGS or (arg in OPTIMIZER_ARGS and not self.args.fresh_optim):
                    try:
                        self.args[arg] = saved_args[arg]
                    except:
                        sys.exit("We lack some arguments")
            if 'end_iteration' in ckpt:
                self.initial_iteration = ckpt['end_iteration'] + 1
            if 'cur_dir' in ckpt:
                self.cur_dir = ckpt['cur_dir']
            if 'best_classifying_acc' in ckpt:
                self.best_classifying_acc = ckpt['best_classifying_acc']
            if 'best_clustering_acc' in ckpt:
                self.best_clustering_acc = ckpt['best_clustering_acc']
            if 'data_seeds' in ckpt:
                self.data_seeds = ckpt['data_seeds']

    def load_checkpoint(self):
        """
        Loads the checkpoint from the given directory
        """

        if not os.path.exists(self.args.resume):
            print("=> no checkpoint found at '{}'\n".format(self.args.resume))
            return None
        else:
            print("=> loading checkpoint '{}'\n".format(self.args.resume))

        return torch.load(self.args.resume, map_location='cpu')

    def get_ema(self, ckpt):
        s_ema = None
        us_ema = None
        if self.args.s_ema_eval or self.args.s_ema_teacher:
            s_ema = ModelEMA(model=self.model, device=self.device)
            if ckpt is not None:
                # for backward compatibility.
                s_ema_state_dict = ckpt.get('s_ema_state_dict') or ckpt.get('ss_ema_state_dict')
                if s_ema_state_dict is not None:
                    s_ema.ema.load_state_dict(s_ema_state_dict)

        if self.args.us_ema_eval or self.args.us_ema_teacher:
            us_ema = ModelEMA(model=self.model, device=self.device)
            if ckpt is not None:
                # for backward compatibility.
                us_ema_state_dict = ckpt.get('us_ema_state_dict') or ckpt.get('c_ema_state_dict')
                if us_ema_state_dict is not None:
                    us_ema.ema.load_state_dict(us_ema_state_dict)

        return s_ema, us_ema

    def get_schedulers(self):
        s_steps = max(sum(self.args.s_epochs) + (self.args.iterations - len(self.args.s_epochs)) *
                      self.args.s_epochs[-1], 1)
        us_steps = max(sum(self.args.us_epochs) + (self.args.iterations - len(self.args.us_epochs)) *
                       self.args.us_epochs[-1], 1)
        s_scheduler = None
        us_scheduler = None
        if self.args.s_scheduler == 'linear':
            linear_scheduler = LinearScheduler(min_value=self.args.min_lr, max_value=self.args.s_lr,
                                               num_iters=s_steps)
            s_scheduler = LambdaLR(optimizer=self.s_optim, lr_lambda=linear_scheduler)
        elif self.args.s_scheduler == 'step':
            s_scheduler = MultiStepLR(optimizer=self.s_optim, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.s_scheduler == 'cosine':
            s_scheduler = WarmupCosineLrScheduler(optimizer=self.s_optim, max_iter=s_steps)
        if self.args.us_scheduler == 'step':
            us_scheduler = MultiStepLR(optimizer=self.us_optim, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.us_scheduler == 'cosine':
            us_scheduler = WarmupCosineLrScheduler(optimizer=self.us_optim, max_iter=us_steps)
        elif self.args.us_scheduler == 'linear':
            linear_scheduler = LinearScheduler(min_value=self.args.min_lr, max_value=self.args.us_lr,
                                               num_iters=us_steps)
            s_scheduler = LambdaLR(optimizer=self.us_optim, lr_lambda=linear_scheduler)
        return s_scheduler, us_scheduler

    def separate_parameters(self, model):
        '''
        separate parameters for different weight decay and learning rate values. The learning rate values are separated
        if we freeze/semi-freeze (use the arguments freeze and/or frozen_lr) some layers. The weight decay is applied
        to all layers except for the 1-dimensional ones (linear, batch norm..)
        :param model: our model.
        :return: list of parameters for the optimizer
        '''
        param_list = [{'params': []}, {'params': [], 'weight_decay': 0}, {'params': [], 'lr': self.args.frozen_lr},
                      {'params': [], 'lr': self.args.frozen_lr, 'weight_decay': 0}]
        if self.args.freeze > 0 and self.args.frozen_lr > 0:
            low_lr = model.features[:self.args.freeze].parameters()
            regular_lr = list(model.features[self.args.freeze:].parameters()) + \
                         list(model.classifier.parameters()) + list(model.rot_net.parameters())
        else:
            low_lr = []
            regular_lr = model.parameters()
        for param in low_lr:
            if len(param.size()) == 1:
                param_list[3]['params'].append(param)
            else:
                param_list[2]['params'].append(param)

        for param in regular_lr:
            if len(param.size()) == 1:
                param_list[1]['params'].append(param)
            else:
                param_list[0]['params'].append(param)

        return [group for group in param_list if len(group['params']) > 0]

    def build_models(self, ckpt):
        grayscale = self.args.grayscale or self.labeled_trainset.num_channels == 1
        model_specifics = {k: v for k, v in self.args.items() if k in MODEL_SPECIFICS_ARGS}
        model = get_model(arch=self.args.arch, num_classes=self.num_classes,
                          grayscale=grayscale, sobel=self.args.sobel, **model_specifics)
        if ckpt is not None:
            state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
            if len({k for k in state_dict.keys() if 'rot_net' in k}) == 0:
                rotnet_state = {k: v for k, v in model.state_dict().items() if 'rot_net' in k}
                state_dict.update(rotnet_state)
            model.load_state_dict(state_dict)
        model = model.to(self.device)

        optim_params = self.separate_parameters(model)
        us_optim = torch.optim.SGD(optim_params, lr=self.args.us_lr, weight_decay=self.args.us_wd,
                                   momentum=self.args.momentum)
        optim_params = self.separate_parameters(model)
        s_optim = torch.optim.SGD(optim_params, lr=self.args.s_lr, weight_decay=self.args.s_wd,
                                  momentum=self.args.momentum, nesterov=self.args.nesterov)

        if ckpt is not None and not self.args.fresh_optim:  # if we resume and want to continue with optimizer state.
            if np.any(self.args.us_epochs):
                state_dict = ckpt['c_optim_state_dict'] if 'c_optim_state_dict' in ckpt \
                    else ckpt['us_optim_state_dict']
                us_optim.load_state_dict(state_dict)
            if np.any(self.args.s_epochs):
                state_dict = ckpt['ss_optim_state_dict'] if 'ss_optim_state_dict' in ckpt \
                    else ckpt['s_optim_state_dict']
                s_optim.load_state_dict(state_dict)

        self.log_file.write("=> Successfully restored All model parameters. Restarting from iteration: {}\n".format(
            self.initial_iteration + 1))
        return model, us_optim, s_optim

    def create_dirs(self):
        # if we are resuming and give the same name as the resumed experiment,
        # we will continue everything on the same folder. Otherwise, new folder is created.
        create_new_dir = self.args.resume is None or \
                         os.path.basename(self.cur_dir.rstrip(string.digits + '_')) != self.args.rn
        # search for the first number which doesn't exist yet with our name.
        if create_new_dir:
            cur_dir = os.path.join(EXPERIMENTS_FOLDER, self.args.rn)
            i = 0
            while os.path.isdir(cur_dir):
                cur_dir = cur_dir.rstrip(string.digits + '_') + '_%d' % (i)
                i += 1
            Path(cur_dir).mkdir(parents=True, exist_ok=True)
            self.cur_dir = cur_dir
        self.checkpoint_dir = os.path.join(self.cur_dir, self.args.checkpoint_dir)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = os.path.join(self.cur_dir, self.args.log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def save_state(self, iteration, phase='s'):

        self.log_file.write('saving best model\n')

        us_ema_state_dict = None if self.us_ema is None else self.us_ema.ema.state_dict()
        s_ema_state_dict = None if self.s_ema is None else self.s_ema.ema.state_dict()
        cta_rates = self.labeled_trainset.transform.cta_rates if 'cta' in self.args.s_algo else None

        state = {
            'args': vars(self.args),
            'model_state_dict': self.model.state_dict(),
            'us_optim_state_dict': self.us_optim.state_dict(),
            's_optim_state_dict': self.s_optim.state_dict(),
            'us_ema_state_dict': us_ema_state_dict,
            's_ema_state_dict': s_ema_state_dict,
            'best_classifying_acc': self.best_classifying_acc,
            'best_clustering_acc': self.best_clustering_acc,
            's_lowest_loss': self.s_lowest_loss,
            'us_lowest_loss': self.us_lowest_loss,
            'end_iteration': iteration,
            'cur_dir': self.cur_dir,
            'data_seeds': self.data_seeds,
            'labels_permutation': self.labels_permutation,
            'nat': self.unlabeled_trainset.nat,
            'cta_rates': cta_rates
        }
        try:
            torch.save(state, os.path.join(self.checkpoint_dir, 'ckpt-{}.pkl'.format(phase)))
        except:
            print('problem with saving')

    def estimate_labels_permutation(self):
        '''
        This method is called during training every few iterations if we are trying to learn a good permutation.
        The heuristic used here is to switch the best pair of classes in our current permutation if they both
        are willing to switch. This is determined by the cost matrix which is the distances of the rotated labeled
        images from all one-hot vectors.
        :return:
        '''
        self.model.train(False)
        model = self.s_ema.ema if self.args.s_ema_eval else self.model
        transform = partial(self.labeled_trainset.transform, modes=[MODES.ROTNET_MODE])
        # we get the cost matrix M. M[i, j] is the mean distance of rotated images from class i and the jth unit vector.
        mean_distances = get_perm_cost_matrix(model, self.labeled_trainset, transform, j=self.args.workers)
        labels_perm = np.arange(self.num_classes) if self.labels_permutation is None else self.labels_permutation
        # given our current permutation, we are looking for the best pair of classes/clusters to switch.
        best_switch = find_best_switch(mean_distances[labels_perm, :])

        if best_switch is not None:
            # we found a good switch and make that switch.
            self.labels_permutation = labels_perm
            self.labels_permutation[list(best_switch)] = self.labels_permutation[list(best_switch[::-1])]

        self.model.train(True)
        print('labels_permutation: {}'.format(self.labels_permutation))

    def initial_perm_estimation(self):
        '''
        Unlike the 'estimate_labels_permutation' method, this method is called only in the beginning of training if
        we resume from clustering and are willing to start training with labels as well. Then, the labels permutation
        are computed with the hungarian algorithm, where the cost matrix is the distance of the mean features of each
        cluster from the one-hot vectors.
        :return:
        '''
        self.model.train(False)
        self.unlabeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        loader = DataLoader(DummyNATDataset(self.unlabeled_trainset, only_images=True),
                            batch_size=512,
                            shuffle=False,
                            num_workers=self.args.workers,
                            pin_memory=False)
        self.labeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        model = self.s_ema.ema if self.args.s_ema_eval else self.model
        transform = partial(self.validation_set.transform, modes=[MODES.EVAL_MODE])

        clusters_mean_features = find_clusters_mean_features(self.model, loader, self.num_classes)
        mean_distances = get_perm_cost_matrix(model, self.labeled_trainset, transform, rotate=False,
                                              n_repetitions=1, j=self.args.workers, means=clusters_mean_features)

        _, assignment = linear_sum_assignment(mean_distances.T)

        self.model.train(True)
        self.labels_permutation = assignment
        print('labels_permutation: {}'.format(self.labels_permutation))

    def freeze_layers(self):
        '''
        freeze layers for fine-tunning.
        :return:
        '''
        for param in self.model.features[:self.args.freeze].parameters():
            param.requires_grad = False

    def rotnet_epoch(self, optim):
        '''
        rotnet epoch. Can be optimized with either s_potim or us_optim.
        :param optim: the optimizer. Depending on which phase we are on.
        :return: loss and accuracy.
        '''
        self.unlabeled_trainset.change_transform_mode(mode=[MODES.ROTNET_MODE])
        data_loader = DataLoader(self.unlabeled_trainset,
                                 batch_size=self.args.us_batch_size,
                                 shuffle=True,
                                 num_workers=self.args.workers,
                                 pin_memory=True)  # TODO maybe more generic batch size (as argument)

        epoch_loss = 0.0
        epoch_acc = 0.0
        # Stream Training dataset with NAT
        for idx, images, label, nat in data_loader:
            images = torch.cat(images, dim=0).to(self.device)

            cur_loss, cur_acc = self.rotnet_batch(x=images, optim=optim)
            epoch_loss += cur_loss
            epoch_acc += cur_acc

        epoch_loss /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        epoch_acc /= len(self.unlabeled_trainset) * NUM_ROTATIONS

        return epoch_loss, epoch_acc

    def rotnet_batch(self, x, optim):
        output = self.model(x, rot_net=True)
        original_images_num = len(x) // NUM_ROTATIONS
        labels = torch.tensor(np.repeat(range(NUM_ROTATIONS), original_images_num)).to(self.device)

        loss = self.s_loss_fn(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        accuracy = np.sum(np.argmax(output.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
        return loss.detach().item() * len(x), accuracy

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        pass

    def s_epoch(self, iteration, epoch, rotnet=False, **kwargs):
        pass

    def prepare_s_iteration(self):
        pass

    def prepare_us_iteration(self):
        pass

    def eval_classifier(self, no_ema=False):
        '''
        evaluate the classifier on the validation set.
        :param no_ema: if True, use the raw model. Else, use the model EMA.
        :return: validation accuracy
        '''
        true_labels = self.validation_set.labels
        model = self.s_ema.ema if self.args.s_ema_eval and not no_ema else self.model
        self.model.train(False)
        pred_labels = get_predictions(model=model, loader=self.val_loader, device=self.device)
        self.model.train(True)
        if self.labels_permutation is not None:
            pred_labels = self.labels_permutation[pred_labels]
        validation_accuracy = np.mean(true_labels == pred_labels)
        if validation_accuracy > self.best_classifying_acc:
            self.best_classifying_acc = validation_accuracy
        return validation_accuracy

    def eval_train(self):
        '''
        evaluate the classifier on the labeled trainset.
        :return: accuracy on the labeled trainset
        '''
        true_labels = self.labeled_trainset.labels
        model = self.s_ema.ema if self.args.s_ema_eval else self.model
        self.model.train(False)
        self.labeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        loader = DataLoader(self.labeled_trainset,
                            batch_size=512,
                            shuffle=False,
                            num_workers=self.args.workers,
                            pin_memory=False)
        pred_labels = get_predictions(model=model, loader=loader, device=self.device)
        self.model.train(True)
        if self.labels_permutation is not None:
            pred_labels = self.labels_permutation[pred_labels]
        train_accuracy = np.mean(true_labels == pred_labels)
        return train_accuracy

    def eval_clustering(self):
        '''
        evaluate the classifier with the clustering accuracy score and NMI score.
        :return: NMI score and clustering accuracy score.
        '''
        self.unlabeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        loader = DataLoader(DummyNATDataset(self.unlabeled_trainset),
                            batch_size=512,
                            shuffle=False,
                            num_workers=self.args.workers,
                            pin_memory=False)
        true_labels = self.unlabeled_trainset.labels
        model = self.us_ema.ema if self.args.us_ema_eval else self.model
        self.model.train(False)
        pred_labels = get_predictions(model=model, loader=loader, device=self.device)
        self.model.train(True)
        nmi_score = sklearn.metrics.normalized_mutual_info_score(pred_labels, true_labels)
        # given the predicted labels and the real labels, compute the best matching between them/the best permutation.
        assignment, reordered_pred_labels = compute_best_labels_permutation(pred_labels, true_labels, self.num_classes)
        acc_score = np.mean(reordered_pred_labels == true_labels)

        if acc_score > self.best_clustering_acc:
            self.best_clustering_acc = acc_score

        return nmi_score, acc_score

    def pseudo_label(self):
        '''
        not really used anymore.
        :return:
        '''
        self.unlabeled_trainset.change_transform_mode(mode=[MODES.EVAL_MODE])
        loader = DataLoader(DummyNATDataset(self.unlabeled_trainset),
                            batch_size=512,
                            shuffle=False,
                            num_workers=self.args.workers,
                            pin_memory=False)
        model = self.s_ema.ema if self.args.s_ema_teacher else self.model  # TODO change that
        self.model.train(False)
        pred_labels, max_probs = get_predictions(model=model, loader=loader, device=self.device, return_probs=True)
        self.model.train(True)

        labels = self.labeled_trainset.labels
        images = self.labeled_trainset.data
        indices = self.labeled_trainset.indices

        for i in range(self.num_classes):
            class_preds = pred_labels == i
            class_confident_indices = np.flip(max_probs[class_preds].argsort())[:self.args.K]
            class_confident_images = self.unlabeled_trainset.data[class_preds][class_confident_indices]
            label = i if self.labels_permutation is None else self.labels_permutation[i]
            num_correct = np.sum(self.unlabeled_trainset.labels[class_preds][class_confident_indices] == label)
            print("the model was correct on {}/{} among the most confident images of class {}".format(num_correct,
                                                                                                      self.args.K, i))
            class_labels = np.ones(len(class_confident_images), dtype=int) * label
            self.labeled_trainset.add_images(class_confident_images, class_labels)

            labels = np.hstack([labels, class_labels])
            images = np.vstack([images, class_confident_images])
            indices = np.hstack([indices, np.where(class_preds)[0][class_confident_indices]])

        if self.args.save_pseudo:
            Path(self.args.save_pseudo).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(self.args.save_pseudo, 'labels.npy'), labels)
            np.save(os.path.join(self.args.save_pseudo, 'images.npy'), images)
            with open(os.path.join(self.args.save_pseudo, 'indices.json'), 'w') as outfile:
                indices_dict = {'indexes': indices.tolist(), 'distribution': [1.0] * self.num_classes}
                json.dump(indices_dict, outfile)
