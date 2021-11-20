from ss_clustering import SemiSupervisedClustering, NUM_ROTATIONS
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import torch.nn.functional as F
from transforms import MODES
from utils.contrastive_loss import NTXentLoss
from sklearn.metrics import euclidean_distances
import lap


class DeepClustering(SemiSupervisedClustering):
    """
    Our deep clustering algorithm which is explained thoroughly in the paper. This is the main unsupervised module we
    experimented with, and all the results in the paper are achieved with this module.
    """
    def __init__(self, args):
        super().__init__(args)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.TRAIN_MODE]
        if rotnet:
            mode.append(MODES.ROTNET_MODE)
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        data_loader = DataLoader(self.unlabeled_trainset,
                                 batch_size=self.args.us_batch_size,
                                 shuffle=True,
                                 num_workers=self.args.workers,
                                 pin_memory=True)

        clustering_loss = 0.0
        rotnet_loss = 0.0
        rotnet_accuracy = 0.0
        num_switches = 0
        # Stream Training dataset with NAT
        for idx, images, label, nat in data_loader:
            if rotnet:
                (x, augmented), rotnet_batch = images
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_accuracy += cur_rotnet_acc
            else:
                x, augmented = images

            targets = nat.numpy()
            x = x.to(self.device)
            augmented = torch.cat(augmented, dim=0).to(self.device)
            cur_clustering_loss, batch_switches = self.clustering_batch(x=x, augmented=augmented, targets=targets,
                                                                        indices=idx)
            num_switches += batch_switches
            clustering_loss += cur_clustering_loss

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        clustering_loss /= len(self.unlabeled_trainset) * self.args.r
        rotnet_loss /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_accuracy /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        us_stats = {'clustering_loss': clustering_loss, 'num_switches': num_switches}
        if clustering_loss < self.us_lowest_loss:
            self.us_lowest_loss = clustering_loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        return rotnet_loss, rotnet_accuracy, us_stats

    def clustering_batch(self, x, augmented, targets, indices):
        n_switches = 0
        with torch.no_grad():
            if self.args.us_ema_teacher:
                output = F.normalize(self.us_ema.ema(x), dim=1, p=2)
            else:
                self.model.train(False)
                output = F.normalize(self.model(x), dim=1, p=2)
                self.model.train(True)
        output = output.detach().cpu().numpy()
        new_targets = np.copy(targets)
        real_targets = targets[np.sum(targets, axis=1) != 0]  # the non-zero targets.

        # finding the best assignment to targets according to the l2 distance of the model's output (projected to
        # the unit sphere) and all one-hot vectors.
        cost = euclidean_distances(output, real_targets)
        _, assignments, __ = lap.lapjv(cost, extend_cost=True)

        for i in range(len(new_targets)):
            if assignments[i] == -1:  # means that the image hasn't got a target.
                new_targets[i] = np.zeros(self.num_classes)
            else:
                new_targets[i] = real_targets[assignments[i]]
            no_real_target = np.logical_xor(np.any(new_targets[i]), np.any(targets[i]))  # whether either the old
            # target or new target are non-targets (zeros). Used to calculate the switches.
            target_switch = np.argmax(new_targets[i]) != np.argmax(targets[i])  # whether there was a cluster switch.
            n_switches += int(no_real_target or target_switch)

        self.unlabeled_trainset.update_targets(indices, new_targets)  # update the assignment to targets.
        one_hot_pseudo_labels = np.eye(self.num_classes)[np.argmax(output, axis=1)]
        confidence = np.linalg.norm(output - one_hot_pseudo_labels, axis=1)
        new_augmented, y = [], []
        for i in range(len(new_targets)):
            if np.sum(new_targets[i]) != 0:  # has target
                t = new_targets[i]
            elif confidence[i] < self.args.rho:  # high confidence sample receives a psuedo-target
                t = one_hot_pseudo_labels[i]
            else:  # the sample has no target and is not confident and hence is not processed.
                continue

            for j in range(self.args.r):  # include the r repetitions of the processed images.
                new_augmented.append(augmented[j * len(x) + i])
                y.append(t)

        new_augmented = torch.stack(new_augmented)
        output = F.normalize(self.model(new_augmented), dim=1, p=2)
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        loss = self.us_loss_fn(output, y)
        self.us_optim.zero_grad()
        loss.backward()
        self.us_optim.step()
        if self.us_ema is not None:
            self.us_ema.update_params()
        return loss.detach().item() * len(augmented), n_switches


class USFixMatch(SemiSupervisedClustering):
    """
    As explained in 'ssl_algorithms.py', this is an attempt to separate FixMatch to supervised and unsupervised phase.
    """

    def __init__(self, args):
        super().__init__(args)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.SS_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.SS_MODE]
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        train_loader = DataLoader(self.unlabeled_trainset,
                                  batch_size=self.args.us_batch_size,
                                  drop_last=False,
                                  num_workers=self.args.workers,
                                  pin_memory=False)
        # inverse function for going from normal labels to labels predicted by the model.
        labels_perm_inverse = np.where(self.labels_permutation == np.arange(self.num_classes)[:, None])[1]

        loss = torch.tensor(0.0, device=self.device)
        confident_samples_ratio = torch.tensor(0.0, device=self.device)
        pseudo_acc = torch.tensor(0.0, device=self.device)
        rotnet_loss = 0.0
        rotnet_acc = 0.0

        batch_mean_time = 0.0
        for i, data in enumerate(train_loader):
            if rotnet:
                idx, ((weak_batch, strong_batch), rotnet_batch), label, nat = data
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_acc += cur_rotnet_acc
            else:
                idx, (weak_batch, strong_batch), label, nat = data

            if self.labels_permutation is not None:
                label = torch.from_numpy(labels_perm_inverse[label])
            start = time.time()
            cur_loss, num_confident, correct_pseudo = self.us_batch(weak_batch, strong_batch, label)
            batch_mean_time += (time.time() - start)
            loss += cur_loss
            confident_samples_ratio += num_confident
            pseudo_acc += correct_pseudo

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        print("batch mean time took {} seconds".format(batch_mean_time / len(train_loader)))
        confident_samples_ratio = confident_samples_ratio.item()
        loss = loss.item() / max(confident_samples_ratio, 1)
        rotnet_samples = len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_loss /= rotnet_samples
        rotnet_acc /= rotnet_samples
        pseudo_acc = pseudo_acc.item() / max(confident_samples_ratio, 1)
        confident_samples_ratio /= len(self.unlabeled_trainset)
        us_stats = {'unlabeled_loss': loss, 'confidence_ratio': confident_samples_ratio,
                    'pseudo_acc': pseudo_acc}
        if loss < self.us_lowest_loss:
            self.us_lowest_loss = loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        return rotnet_loss, rotnet_acc, us_stats

    def us_batch(self, weak_batch, strong_batch, label):
        with torch.no_grad():
            weak_batch = weak_batch.to(self.device)
            if self.args.us_ema_teacher:
                logits_weak = self.us_ema.ema(weak_batch)
            else:
                logits_weak = self.model(weak_batch)
            weak_probs = F.softmax(logits_weak.detach_(), dim=-1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=-1)
            idx = max_probs > self.args.confidence_threshold
            pseudo_labels = pseudo_labels[idx].detach()

        loss = torch.tensor(0.0)
        strong_batch = strong_batch[idx]
        if strong_batch.size(0) > 0:
            strong_batch = strong_batch.to(self.device)
            logits = self.model(strong_batch)
            loss = self.s_loss_fn(logits, pseudo_labels)

            self.us_optim.zero_grad()
            loss.backward()
            self.us_optim.step()

            if self.us_ema is not None:
                self.us_ema.update_params()

        confident_pseudo = strong_batch.size(0)
        correct_pseudo = torch.sum(pseudo_labels.detach() == label[idx].to(self.device))
        loss = loss.detach() * confident_pseudo
        return loss, confident_pseudo, correct_pseudo

    def prepare_us_iteration(self):
        for param in self.model.features[self.args.freeze:].parameters():
            param.requires_grad = True


class USContrastive(SemiSupervisedClustering):
    """
    Use the contrastive loss from the paper 'A Simple Framework for Contrastive Learning of Visual Representations'
    on the unlabeled data as the unsupervised phase.
    """

    def __init__(self, args):
        super().__init__(args)
        self.contrastive_loss = NTXentLoss(device=self.device, temperature=0.5, use_cosine_similarity=True)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.TRAIN_MODE]
        if rotnet:
            mode.append(MODES.ROTNET_MODE)
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        data_loader = DataLoader(self.unlabeled_trainset,
                                 batch_size=self.args.us_batch_size,
                                 shuffle=True,
                                 num_workers=self.args.workers,
                                 pin_memory=False)

        contrastive_loss = 0.0
        rotnet_loss = 0.0
        rotnet_accuracy = 0.0

        for idx, images, label, nat in data_loader:
            if rotnet:
                (x1, x2), rotnet_batch = images
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_accuracy += cur_rotnet_acc
            else:
                x1, x2 = images

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            cur_contrastive_loss = self.us_batch(x1, x2)
            contrastive_loss += cur_contrastive_loss

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        contrastive_loss /= (len(self.unlabeled_trainset) * 2)
        rotnet_loss /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_accuracy /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        us_stats = {'contrastive_loss': contrastive_loss}
        if contrastive_loss < self.us_lowest_loss:
            self.us_lowest_loss = contrastive_loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        return rotnet_loss, rotnet_accuracy, us_stats

    def us_batch(self, x1, x2):
        x1_logits = self.model(x1)
        x2_logits = self.model(x2)
        loss = self.contrastive_loss(x1_logits, x2_logits)

        self.us_optim.zero_grad()
        loss.backward()
        self.us_optim.step()

        if self.us_ema is not None:
            self.us_ema.update_params()

        return loss.detach() * x1.size(0) * 2