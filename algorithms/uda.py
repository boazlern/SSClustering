from algorithms.ssl_algorithms import FixMatch, CTAFixMatch
import torch
import torch.nn.functional as F


class UDA(FixMatch):
    def __init__(self, args):
        super().__init__(args)
        self.T = 0.4

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        batch_size = x.size(0)
        y = y.to(self.device)
        data_all = torch.cat([x, weak_batch, strong_batch]).to(self.device)
        preds_all = self.model(data_all)
        preds_labeled = preds_all[:batch_size]
        labeled_loss = self.s_loss_fn(preds_labeled, y)  # loss for supervised learning

        preds_unlabeled = preds_all[batch_size:]
        preds1, preds2 = torch.chunk(preds_unlabeled, 2)
        del preds_unlabeled
        preds1_probs = F.softmax(preds1, dim=1).detach()
        max_probs, _ = torch.max(preds1_probs, dim=-1)
        idx = max_probs > self.args.confidence_threshold
        preds2 = F.log_softmax(preds2[idx], dim=1)
        target_probs = F.softmax(preds1[idx] / self.T, dim=1).detach()

        # loss for unsupervised
        loss_kldiv = F.kl_div(preds2, target_probs, reduction='batchmean')
        loss = labeled_loss + self.args.lambda_pseudo * loss_kldiv

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        return labeled_loss.item(), loss_kldiv.item(), 0, 0


class CTAUDA(CTAFixMatch):
    def __init__(self, args):
        super().__init__(args)
        self.T = 0.4

    def ss_batch(self, x, y, weak_batch, strong_batch, label, indices=None, nat=None):
        batch_size = x.size(0)
        y = y.to(self.device)
        data_all = torch.cat([x, weak_batch, strong_batch]).to(self.device)
        preds_all = self.model(data_all)
        preds_labeled = preds_all[:batch_size]
        labeled_loss = self.s_loss_fn(preds_labeled, y)  # loss for supervised learning

        preds_unlabeled = preds_all[batch_size:]
        preds1, preds2 = torch.chunk(preds_unlabeled, 2)
        del preds_unlabeled
        preds1_probs = F.softmax(preds1, dim=1).detach()
        max_probs, _ = torch.max(preds1_probs, dim=-1)
        idx = max_probs > self.args.confidence_threshold
        preds2 = F.log_softmax(preds2[idx], dim=1)
        target_probs = F.softmax(preds1[idx] / self.T, dim=1).detach()

        # loss for unsupervised
        loss_kldiv = F.kl_div(preds2, target_probs, reduction='batchmean')
        loss = labeled_loss + self.args.lambda_pseudo * loss_kldiv

        self.s_optim.zero_grad()
        loss.backward()
        self.s_optim.step()

        if self.s_ema is not None:
            self.s_ema.update_params()

        return labeled_loss.item(), loss_kldiv.item(), 0, 0