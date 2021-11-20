from algorithms.ssl_algorithms import FixMatch
import torch
import numpy as np
import torch.nn.functional as F


class SemiLoss(object):
    def __init__(self, final_w, warmup_steps):
        self.step = 0
        self.warmup_steps = warmup_steps
        self.final_w = final_w

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        cur_w = self.final_w * float(np.clip(self.step / self.warmup_steps, 0.0, 1.0))
        return Lx, Lu, cur_w


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


class MixMatch(FixMatch):
    def __init__(self, args):
        super().__init__(args)
        self.T = 0.5
        self.beta = 0.75
        self.loss = SemiLoss(final_w=75, warmup_steps=2 ** 14)

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.num_classes).scatter_(1, y.view(-1, 1).long(), 1)
        weak_batch = weak_batch.to(self.device)
        strong_batch = strong_batch.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            output1 = self.model(weak_batch)
            output2 = self.model(strong_batch)
            p = (torch.softmax(output1, dim=1) + torch.softmax(output2, dim=1)) / 2
            pt = p ** (1 / self.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([x, weak_batch, strong_batch], dim=0)
        all_targets = torch.cat([y, targets_u, targets_u], dim=0)

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

        Lx, Lu, w = self.loss(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:])
        self.loss.step += 1
        loss = Lx + w * Lu

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        return Lx.item(), Lu.item(), 0, 0
