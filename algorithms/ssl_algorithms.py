from ss_clustering import SemiSupervisedClustering, NUM_ROTATIONS
from torch.utils.data import RandomSampler, DataLoader
import torch
import numpy as np
import time
import torch.nn.functional as F
from transforms import MODES
from utils.contrastive_loss import NTXentLoss
import lap
from scipy.special import logsumexp


class FixMatch(SemiSupervisedClustering):
    """
    The FixMatch algorithm using RandAugment.
    """

    def __init__(self, args):
        self.interleave = args.interleave
        super().__init__(args)

    def get_ss_iterators(self):
        labeled_sampler = RandomSampler(self.labeled_trainset, replacement=True,
                                        num_samples=len(self.unlabeled_trainset) * self.args.s_batch_size)
        labeled_train_loader = DataLoader(self.labeled_trainset,
                                          batch_size=self.args.s_batch_size,
                                          sampler=labeled_sampler,
                                          drop_last=True,
                                          num_workers=self.args.workers,
                                          pin_memory=True)
        unlabeled_train_loader = DataLoader(self.unlabeled_trainset,
                                            batch_size=self.args.s_batch_size * self.args.mu,
                                            shuffle=True,
                                            drop_last=self.interleave,
                                            num_workers=self.args.workers,
                                            pin_memory=True)
        return labeled_train_loader, unlabeled_train_loader

    def s_epoch(self, iteration, epoch, rotnet=False, **kwargs):
        labeled_mode = [MODES.TRAIN_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.TRAIN_MODE]
        unlabeled_mode = [MODES.SS_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.SS_MODE]
        self.labeled_trainset.change_transform_mode(mode=labeled_mode)
        self.unlabeled_trainset.change_transform_mode(mode=unlabeled_mode)
        labeled_train_loader, unlabeled_train_loader = self.get_ss_iterators()
        # inverse function for going from normal labels to labels predicted by the model.
        labels_perm_inverse = np.where(self.labels_permutation == np.arange(self.num_classes)[:, None])[1]

        train_loader = zip(labeled_train_loader, unlabeled_train_loader)

        l_loss = torch.tensor(0.0, device=self.device)
        ul_loss = torch.tensor(0.0, device=self.device)
        confident_samples_ratio = torch.tensor(0.0, device=self.device)
        pseudo_acc = torch.tensor(0.0, device=self.device)
        rotnet_loss = 0.0
        rotnet_acc = 0.0
        batch_mean_time = 0.0

        for i, (labeled_data, unlabeled_data) in enumerate(train_loader):
            if rotnet:
                (x, labeled_rotnet), y = labeled_data
                idx, ((weak_batch, strong_batch), unlabeled_rotnet), label, nat = unlabeled_data
                rotnet_batch = torch.cat([torch.cat([labeled_rotnet[i], unlabeled_rotnet[i]], dim=0)
                                          for i in range(NUM_ROTATIONS)], dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.s_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_acc += cur_rotnet_acc
                del rotnet_batch
            else:
                x, y = labeled_data
                idx, (weak_batch, strong_batch), label, nat = unlabeled_data

            if self.labels_permutation is not None:
                y = torch.from_numpy(labels_perm_inverse[y])
                label = torch.from_numpy(labels_perm_inverse[label])
            start = time.time()
            cur_l_loss, cur_ul_loss, num_confident, correct_pseudo = self.ss_batch(x, y, weak_batch,
                                                                                   strong_batch, label, idx, nat)
            batch_mean_time += (time.time() - start)
            l_loss += cur_l_loss
            ul_loss += cur_ul_loss
            confident_samples_ratio += num_confident
            pseudo_acc += correct_pseudo

        if self.s_scheduler is not None:
            self.s_scheduler.step()
        if self.s_ema is not None:
            self.s_ema.update_buffer()

        print("batch mean time took {} seconds".format(batch_mean_time / len(unlabeled_train_loader)))
        confident_samples_ratio = confident_samples_ratio.item()
        l_loss = l_loss.item() / (len(unlabeled_train_loader) * self.args.s_batch_size)
        ul_loss = ul_loss.item() / max(confident_samples_ratio, 1)
        rotnet_samples = (len(self.unlabeled_trainset) + self.args.s_batch_size * len(unlabeled_train_loader)) * \
                         NUM_ROTATIONS
        rotnet_loss /= rotnet_samples
        rotnet_acc /= rotnet_samples
        pseudo_acc = pseudo_acc.item() / max(confident_samples_ratio, 1)
        confident_samples_ratio /= len(self.unlabeled_trainset)
        s_stats = {'s_labeled_loss': l_loss, 's_unlabeled_loss': ul_loss,
                   'confidence_ratio': confident_samples_ratio, 'pseudo_acc': pseudo_acc}
        epoch_loss = l_loss + self.args.lambda_pseudo * ul_loss
        if epoch_loss < self.s_lowest_loss:
            self.s_lowest_loss = epoch_loss
            self.save_state(iteration=iteration, phase='s')
        self.save_state(iteration=iteration, phase='end_s')
        return rotnet_loss, rotnet_acc, s_stats

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        if self.interleave:
            interleave_batch = self.interleave_batch(torch.cat([x, weak_batch, strong_batch], dim=0))
            logits = self.deinterleave_batch(self.model(interleave_batch))
            logits_x = logits[:x.size(0)]
            logits_weak, logits_strong = logits[x.size(0):].chunk(2)
            del logits
            with torch.no_grad():
                weak_probs = F.softmax(logits_weak.detach(), dim=-1)
                max_probs, pseudo_labels = torch.max(weak_probs, dim=-1)
                idx = max_probs > self.args.confidence_threshold
                pseudo_labels = pseudo_labels[idx].detach()
            logits_strong = logits_strong[idx]
        else:
            with torch.no_grad():
                weak_batch = weak_batch.to(self.device)
                if self.args.s_ema_teacher:
                    logits_weak = self.s_ema.ema(weak_batch)
                else:
                    logits_weak = self.model(weak_batch)
                weak_probs = F.softmax(logits_weak.detach_(), dim=-1)
                max_probs, pseudo_labels = torch.max(weak_probs, dim=-1)
                idx = max_probs > self.args.confidence_threshold
                pseudo_labels = pseudo_labels[idx].detach()

            strong_batch = strong_batch[idx]
            inputs = x.to(self.device)
            if strong_batch.size(0) > 0:
                inputs = torch.cat([x, strong_batch], dim=0).to(self.device)
            logits = self.model(inputs)
            logits_x, logits_strong = logits[:x.size(0)], logits[x.size(0):]

        y = y.to(self.device)
        l_loss = self.s_loss_fn(logits_x, y)
        ul_loss = torch.tensor(0.0)
        if logits_strong.size(0) > 0:
            ul_loss = self.s_loss_fn(logits_strong, pseudo_labels)
        loss = l_loss + self.args.lambda_pseudo * ul_loss

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        confident_pseudo = strong_batch.size(0)
        correct_pseudo = torch.sum(pseudo_labels.detach() == label[idx].to(self.device))
        batch_l_loss = l_loss.detach() * x.size(0)
        batch_ul_loss = ul_loss.detach() * confident_pseudo
        return batch_l_loss, batch_ul_loss, confident_pseudo, correct_pseudo

    def interleave_batch(self, x):
        image_shape = list(x.shape[1:])
        interleave_batch = x.reshape([-1, self.args.mu * 2 + 1] + image_shape)
        return interleave_batch.transpose(1, 0).reshape([-1] + image_shape).detach().to(self.device)

    def deinterleave_batch(self, x):
        image_shape = list(x.shape[1:])
        interleave_batch = x.reshape([self.args.mu * 2 + 1, -1] + image_shape)
        return interleave_batch.transpose(1, 0).reshape([-1] + image_shape).to(self.device)


class CTAFixMatch(FixMatch):
    """
    The FixMatch algorithm with Control-Theory Augmentations. This is the main class that is used as the semi-supervised
    module in the paper.
    """
    def __init__(self, args):
        super().__init__(args)

    def s_epoch(self, iteration, epoch, rotnet=False, **kwargs):
        labeled_mode = [MODES.TRAIN_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.TRAIN_MODE]
        unlabeled_mode = [MODES.SS_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.SS_MODE]
        self.labeled_trainset.change_transform_mode(mode=labeled_mode)
        self.unlabeled_trainset.change_transform_mode(mode=unlabeled_mode)
        labeled_train_loader, unlabeled_train_loader = self.get_ss_iterators()
        # inverse function for going from normal labels to labels predicted by the model.
        labels_perm_inverse = np.where(self.labels_permutation == np.arange(self.num_classes)[:, None])[1]

        train_loader = zip(labeled_train_loader, unlabeled_train_loader)

        l_loss = torch.tensor(0.0, device=self.device)
        ul_loss = torch.tensor(0.0, device=self.device)
        confident_samples_ratio = torch.tensor(0.0, device=self.device)
        pseudo_acc = torch.tensor(0.0, device=self.device)
        rotnet_loss = 0.0
        rotnet_acc = 0.0

        batch_mean_time = 0.0
        for i, (labeled_data, unlabeled_data) in enumerate(train_loader):
            if rotnet:
                ((x, (probe, ops_indices, mag_indices)), labeled_rotnet), y = labeled_data
                idx, ((weak_batch, strong_batch), unlabeled_rotnet), labels, nat = unlabeled_data
                rotnet_batch = torch.cat([torch.cat([labeled_rotnet[i], unlabeled_rotnet[i]], dim=0)
                                          for i in range(NUM_ROTATIONS)], dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.s_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_acc += cur_rotnet_acc
                del rotnet_batch
            else:
                (x, (probe, ops_indices, mag_indices)), y = labeled_data
                idx, (weak_batch, strong_batch), labels, nat = unlabeled_data

            if self.labels_permutation is not None:
                y = torch.from_numpy(labels_perm_inverse[y])
                labels = torch.from_numpy(labels_perm_inverse[labels])
            start = time.time()
            cur_l_loss, cur_ul_loss, num_confident, correct_pseudo = self.ss_batch(x, y, weak_batch,
                                                                                   strong_batch, labels, idx, nat)
            # TODO not robust - when resume from iterations which is divisible by 10.
            print_probs = (epoch % 10 == 0 or (iteration != 0 and iteration % 10 == 0)) and i == 0
            self.update_cta_rates(x=probe, labels=y, ops_indices=ops_indices, mag_indices=mag_indices,
                                  print_probs=print_probs)
            batch_mean_time += (time.time() - start)
            l_loss += cur_l_loss
            ul_loss += cur_ul_loss
            confident_samples_ratio += num_confident
            pseudo_acc += correct_pseudo

        if self.s_scheduler is not None:
            self.s_scheduler.step()
        if self.s_ema is not None:
            self.s_ema.update_buffer()

        print("batch mean time took {} seconds".format(batch_mean_time / len(unlabeled_train_loader)))
        confident_samples_ratio = confident_samples_ratio.item()
        l_loss = l_loss.item() / (len(unlabeled_train_loader) * self.args.s_batch_size)
        ul_loss = ul_loss.item() / max(confident_samples_ratio, 1)
        rotnet_samples = (len(self.unlabeled_trainset) + self.args.s_batch_size * len(unlabeled_train_loader)) * \
                         NUM_ROTATIONS
        rotnet_loss /= rotnet_samples
        rotnet_acc /= rotnet_samples
        pseudo_acc = pseudo_acc.item() / max(confident_samples_ratio, 1)
        confident_samples_ratio /= len(self.unlabeled_trainset)
        s_stats = {'s_labeled_loss': l_loss, 's_unlabeled_loss': ul_loss,
                   'confidence_ratio': confident_samples_ratio, 'pseudo_acc': pseudo_acc}
        epoch_loss = l_loss + self.args.lambda_pseudo * ul_loss
        if epoch_loss < self.s_lowest_loss:
            self.s_lowest_loss = epoch_loss
            self.save_state(iteration=iteration, phase='s')
        self.save_state(iteration=iteration, phase='end_s')
        return rotnet_loss, rotnet_acc, s_stats

    def update_cta_rates(self, x, labels, ops_indices, mag_indices, print_probs=False):
        with torch.no_grad():
            self.model.train(False)
            x = x.to(self.device)
            output = F.softmax(self.model(x), dim=-1).cpu().numpy()
            self.model.train(True)

            output[range(len(x)), labels] -= 1
            proximity = 1 - 0.5 * np.abs(output).sum(axis=1)
            # update the rates in both labeled and unlabeled transforms. The new rates are computed in the labeled
            # transform method 'update_rates' and then just plugged into the unlabeled transform.
            self.labeled_trainset.transform.update_rates(proximity=proximity, ops_indices=ops_indices,
                                                         mag_indices=mag_indices, print_probs=print_probs)
            rates = self.labeled_trainset.transform.cta_rates
            self.unlabeled_trainset.transform.update_rates(rates)


class CTAClustering(CTAFixMatch):
    """
    Our deep clustering algorithm as Semi-Supervised method: Use CTA on unlabeled data and cross-entropy
     for labeled data.
    """

    def __init__(self, args):
        super().__init__(args)

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        nat = nat.numpy()
        weak_batch = weak_batch.to(self.device)
        n_switches = 0
        with torch.no_grad():
            if self.args.us_ema_teacher:
                output = self.s_ema.ema(weak_batch)
            else:
                self.model.train(False)
                output = self.model(weak_batch)
                self.model.train(True)
        output = output.detach().cpu().numpy()
        new_targets = np.copy(nat)
        real_targets = nat[np.sum(nat, axis=1) != 0]

        j, c = output.shape[0], real_targets.shape[0]
        # instead of the euclidean distances as in our regular clustering, here we use the cross entropy. So cost[i, j]
        # is the cross entropy loss between output i and target j.
        cost = -output[np.repeat(np.arange(j), c), np.tile(real_targets.argmax(axis=1), j)].reshape((j, c)) + \
               logsumexp(output, axis=1)[:, None]

        _, assignments, __ = lap.lapjv(cost, extend_cost=True)

        for i in range(len(new_targets)):
            if assignments[i] == -1:  # means that the image hasn't got a target.
                new_targets[i] = np.zeros(self.num_classes)
            else:
                new_targets[i] = real_targets[assignments[i]]
            no_real_target = np.logical_xor(np.any(new_targets[i]), np.any(nat[i]))  # whether either the old
            # target or new target are non-targets (zeros).
            target_switch = np.argmax(new_targets[i]) != np.argmax(nat[i])  # whether there was a cluster switch.
            n_switches += int(no_real_target or target_switch)
        self.unlabeled_trainset.update_targets(indices, new_targets)  # update the assignment to targets.
        pseudo_labels = np.argmax(output, axis=1)
        confidence = np.max(output, axis=1)
        new_strong, z = [], []
        for i in range(len(new_targets)):
            if np.sum(new_targets[i]) != 0:  # has target
                t = new_targets[i].argmax()
            elif confidence[i] > self.args.confidence_threshold:  # high confidence sample receives a psuedo-target
                t = pseudo_labels[i]
            else:  # the sample has no target and is not confident and hence is not processed.
                continue
            for j in range(self.args.r):  # include the r repetitions of the processed images.
                new_strong.append(strong_batch[j][i])
                z.append(t)

        new_strong = torch.stack(new_strong).to(self.device)
        output = self.model(new_strong)
        z = torch.tensor(z, dtype=torch.long, device=self.device)
        ul_loss = self.s_loss_fn(output, z)  # we use cross-entropy loss for the unlabeled data instead of euclidean.

        x = x.to(self.device)
        logits_x = self.model(x)
        y = y.to(self.device)
        l_loss = self.s_loss_fn(logits_x, y)  # regular cross-entropy for labeled data.
        loss = l_loss + self.args.lambda_pseudo * ul_loss

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        confident_pseudo = new_strong.size(0)
        batch_l_loss = l_loss.detach() * x.size(0)
        batch_ul_loss = ul_loss.detach() * confident_pseudo
        return batch_l_loss, batch_ul_loss, confident_pseudo, 0


class ContrastiveFixMatch(FixMatch):
    """
    Combines FixMatch with SimClr's contrastive loss from the paper "A Simple Framework for Contrastive Learning of
    Visual Representations". The labeled data is handled as in FixMatch, while the unlabeled data is augmented like in
    FixMatch but is optimized with the contrastive loss.
    """

    def __init__(self, args):
        super().__init__(args)
        self.contrastive_loss = NTXentLoss(device=self.device, temperature=0.5, use_cosine_similarity=True)

    def s_epoch(self, iteration, epoch, rotnet=False, **kwargs):
        rotnet_loss, rotnet_acc, s_stats = super().s_epoch(iteration, epoch, rotnet, **kwargs)
        s_stats['s_unlabeled_loss'] /= (len(self.unlabeled_trainset) * 2)
        return rotnet_loss, rotnet_acc, s_stats

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        if hasattr(self.unlabeled_trainset.transform, 'choice'):
            self.unlabeled_trainset.transform.choice = np.random.choice(4)
        weak_batch = weak_batch.to(self.device)
        strong_batch = strong_batch.to(self.device)
        x = x.to(self.device)
        weak_logits = self.model(weak_batch)
        strong_logits = self.model(strong_batch)
        x_logits = self.model(x)
        y = y.to(self.device)
        l_loss = self.s_loss_fn(x_logits, y)
        ul_loss = self.contrastive_loss(weak_logits, strong_logits)
        loss = l_loss + self.args.lambda_pseudo * ul_loss

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        batch_l_loss = l_loss.detach() * x.size(0)
        batch_ul_loss = ul_loss.detach() * 2 * weak_batch.size(0)
        return batch_l_loss, batch_ul_loss, 0, 0


class BalancedFixMatch(FixMatch):
    """
    Not really used.
    """

    def __init__(self, args):
        super().__init__(args)

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        with torch.no_grad():
            weak_batch = weak_batch.to(self.device)
            if self.args.s_ema_teacher:
                logits_weak = self.s_ema.ema(weak_batch)
            else:
                logits_weak = self.model(weak_batch)
            weak_probs = F.softmax(logits_weak.detach_(), dim=-1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=-1)
            mask = torch.ones_like(pseudo_labels, dtype=torch.float)
            labels, counts = np.unique(pseudo_labels.cpu(), return_counts=True)
            max_count = max(counts)
            if max_count > 0:
                ratios = [x / max_count for x in counts]
                idx = (mask == 0.0)
                for i in range(len(labels)):
                    temp = (max_probs * (pseudo_labels == labels[i]).float()).ge(self.args.confidence_threshold
                                                                                 - 0.25 * (1 - ratios[i]))
                    idx = idx | temp
                labels, counts = np.unique(pseudo_labels[idx].cpu(), return_counts=True)
                ratio = torch.zeros_like(mask, dtype=torch.float)
                for i in range(len(labels)):
                    ratio += ((1 / counts[i]) * (pseudo_labels == labels[i]).float())  # Magnitude of mask elements
                Z = torch.sum(mask[idx])
                mask = ratio[idx]
                if Z > 0:
                    mask = Z * mask / torch.sum(mask)
            else:
                idx = (max_probs > self.args.confidence_threshold)
                mask[idx] = 1.0

        pseudo_labels = pseudo_labels[idx]

        strong_batch = strong_batch[idx]
        inputs = x.to(self.device)
        if strong_batch.size(0) > 0:
            inputs = torch.cat([x, strong_batch], dim=0).to(self.device)
        logits = self.model(inputs)
        logits_x, logits_strong = logits[:x.size(0)], logits[x.size(0):]
        y = y.to(self.device)
        l_loss = self.s_loss_fn(logits_x, y)
        ul_loss = torch.tensor(0.0)
        if strong_batch.size(0) > 0:
            ul_loss = (F.cross_entropy(logits_strong, pseudo_labels, reduction='none') * mask).mean()
        loss = l_loss + self.args.lambda_pseudo * ul_loss

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        confident_pseudo = strong_batch.size(0)
        correct_pseudo = torch.sum(pseudo_labels.detach() == label[idx].to(self.device))
        batch_l_loss = l_loss.detach() * x.size(0)
        batch_ul_loss = ul_loss.detach() * confident_pseudo
        return batch_l_loss, batch_ul_loss, confident_pseudo, correct_pseudo


class SFixMatch(SemiSupervisedClustering):
    """
    Class for experimentation with separating FixMatch into 2 different modules, this one which handles only labeled
    data and USFixMatch from 'unsupervised_algorithms.py' file that handles only unlabeled data.
    """
    def __init__(self, args):
        super().__init__(args)

    def s_epoch(self, iteration, epoch, rotnet=False, **kwargs):
        mode = [MODES.TRAIN_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.TRAIN_MODE]
        self.labeled_trainset.change_transform_mode(mode=mode)
        sampler = RandomSampler(self.labeled_trainset, replacement=True,
                                num_samples=self.args.n_batches * self.args.s_batch_size)
        train_loader = DataLoader(self.labeled_trainset,
                                  batch_size=self.args.s_batch_size,
                                  sampler=sampler,
                                  drop_last=True,
                                  num_workers=self.args.workers,
                                  pin_memory=False)
        # inverse function for going from normal labels to labels predicted by the model.
        labels_perm_inverse = np.where(self.labels_permutation == np.arange(self.num_classes)[:, None])[1]

        loss = torch.tensor(0.0, device=self.device)
        rotnet_loss = 0.0
        rotnet_acc = 0.0

        batch_mean_time = 0.0
        for i, data in enumerate(train_loader):
            if rotnet:
                (x, rotnet_batch), y = data
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.s_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_acc += cur_rotnet_acc
            else:
                x, y = data

            if self.labels_permutation is not None:
                y = torch.from_numpy(labels_perm_inverse[y])
            start = time.time()
            cur_loss = self.ss_batch(x, y)
            batch_mean_time += (time.time() - start)
            loss += cur_loss

        if self.s_scheduler is not None:
            self.s_scheduler.step()
        if self.s_ema is not None:
            self.s_ema.update_buffer()

        print("batch mean time took {} seconds".format(batch_mean_time / self.args.n_batches))
        loss = loss.item() / self.args.n_batches
        rotnet_samples = self.args.n_batches * self.args.s_batch_size * NUM_ROTATIONS
        rotnet_loss /= rotnet_samples
        rotnet_acc /= rotnet_samples
        s_stats = {'s_labeled_loss': loss}
        if loss < self.s_lowest_loss:
            self.s_lowest_loss = loss
            self.save_state(iteration=iteration, phase='s')
        self.save_state(iteration=iteration, phase='end_s')
        return rotnet_loss, rotnet_acc, s_stats

    def ss_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss = self.s_loss_fn(logits, y)

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        loss = loss.detach() * x.size(0)
        return loss

    def prepare_s_iteration(self):
        for param in self.model.features[self.args.freeze:].parameters():
            param.requires_grad = False
