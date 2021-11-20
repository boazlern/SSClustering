from algorithms.ssl_algorithms import CTAFixMatch
import torch
import numpy as np
import torch.nn.functional as F


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def softmax_cross_entropy_with_logits(logits, targets):
    return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))


class ReMixMatch(CTAFixMatch):
    def __init__(self, args):
        super().__init__(args)
        self.model_buf = torch.ones((128, self.num_classes), device=self.device) / self.num_classes
        self.p_model = torch.ones(self.num_classes, device=self.device) / self.num_classes
        self.p_data = self.compute_labels_prob().to(self.device)
        self.T = 0.5
        self.beta = 0.75
        self.warmup_steps = 2 ** 17
        self.step = 0
        self.w_kl = 0.5
        self.w_rot = 2
        self.w_match = 1.5

    def compute_labels_prob(self):
        return torch.from_numpy(np.bincount(self.labeled_trainset.labels) / self.labeled_trainset.labels.shape[0])

    def update_model_probs(self, cur_probs):
        self.model_buf = torch.cat([self.model_buf[1:], torch.mean(cur_probs, dim=0, keepdim=True)])
        self.p_model = torch.mean(self.model_buf, dim=0)
        self.p_model /= self.p_model.sum()

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.num_classes).scatter_(1, y.view(-1, 1).long(), 1)
        weak_batch = weak_batch.to(self.device)
        strong_batch = strong_batch.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        rot_batch = strong_batch.clone()
        rot_size = batch_size // 4
        rot_batch[rot_size: 2 * rot_size] = rot_batch[rot_size: 2 * rot_size].flip(2).transpose(2, 3)
        rot_batch[2 * rot_size: 3 * rot_size] = rot_batch[2 * rot_size: 3 * rot_size].flip(2).flip(3)
        rot_batch[3 * rot_size:] = rot_batch[3 * rot_size:].transpose(2, 3).flip(2)
        rot_batch.detach_()
        rot_labels = torch.arange(4).repeat_interleave(rot_size).to(self.device)
        rot_output = self.model(x, rot_net=True)
        rot_loss = self.s_loss_fn(rot_output, rot_labels)

        with torch.no_grad():
            weak_logits = self.model(weak_batch)
        strong_logits = self.model(strong_batch)
        unlabeled_probs = F.softmax(weak_logits, dim=-1).repeat(2, 1).detach()
        p_ratio = (1e-6 + self.p_data) / (1e-6 + self.p_model)
        p_weighted = unlabeled_probs * p_ratio[None]
        p_weighted /= torch.sum(p_weighted, dim=1)[:, None]
        p_target = p_weighted ** (1 / self.T)
        p_target /= torch.sum(p_target, dim=1)[:, None]
        p_target.detach_()
        loss_kl = softmax_cross_entropy_with_logits(strong_logits, p_target[:batch_size])

        # mixup
        all_inputs = torch.cat([x, weak_batch, strong_batch], dim=0)
        all_targets = torch.cat([y, *torch.split(p_target, 2)], dim=0)

        l = np.random.beta(self.beta, self.beta)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [self.model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(self.model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        del logits

        Lx = softmax_cross_entropy_with_logits(logits=logits_x, targets=mixed_target[:batch_size])
        Lu = softmax_cross_entropy_with_logits(logits=logits_u, targets=mixed_target[batch_size:])

        w_kl = self.w_kl * np.clip(self.step / self.warmup_steps, 0, 1)  # TODO check the issue of warmup steps
        w_match = self.w_match * np.clip(self.step / self.warmup_steps, 0, 1)

        unlabeled_loss = w_kl * loss_kl + w_match * Lu + self.w_rot * rot_loss
        loss = Lx + unlabeled_loss

        self.update_model_probs(unlabeled_probs)
        self.step += 1

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        return Lx.detach(), unlabeled_loss.detach(), 0, 0
