"""Losses shared by the knowledge-base stages (docs/knowledge-base-stack.md, 5.2):
KL between readings, the retrieval auxiliary loss, its key-teacher distillation term and
the reward for spread-out use of the KB."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from schnitz.kb.decoder import _kl as kl


def retrieval_loss(scores: torch.Tensor, positive: torch.Tensor,
                   ks: tuple[int, ...] = (1, 5, 20)) -> tuple[torch.Tensor, dict]:
    """Listwise retrieval loss over each query's candidates: -log of the softmax mass
    on its positives. ``scores`` (Q, C); ``positive`` (Q, C) bool, or nonnegative
    weights (e.g. responsibility shares of rewritten items), used as
    log-weights. Queries without positives are skipped. Returns the mean loss and
    recall@k (any positive in the top k) over the counted queries."""
    weights = positive.float()
    has = weights.sum(-1) > 0
    if not bool(has.any()):
        return scores.sum() * 0.0, {}
    s, w = scores[has].float(), weights[has]
    log_p = torch.log_softmax(s, -1)
    loss = -(torch.logsumexp(log_p + torch.log(w.clamp_min(1e-30)), -1)).mean()
    order = s.argsort(-1, descending=True)
    hits = (w.gather(-1, order) > 0)
    stats = {f'recall@{k}': float(hits[:, :k].any(-1).float().mean()) for k in ks
             if k <= s.shape[-1]}
    return loss, stats


def teacher_kl(scores: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """KL(softmax(teacher) || softmax(scores)) over one candidate list (last dimension):
    the key-teacher distillation term. ``teacher`` is the teacher's logits (cosine over
    its temperature) and carries no gradient; exactly 0 when the two are equal."""
    log_t = torch.log_softmax(teacher.detach().float(), -1)
    log_s = torch.log_softmax(scores.float(), -1)
    return (log_t.exp() * (log_t - log_s)).sum(-1).mean()


class UsageEMA:
    """Exponential moving average of each item's share of read mass in one KB space,
    for the spread-out-use reward and its coverage statistics."""

    def __init__(self, items: int, decay: float = 0.999, device=None):
        self.share = torch.full((items,), 1.0 / max(items, 1), device=device)
        self.decay = decay
        self.touched = torch.zeros(items, dtype=torch.bool, device=device)

    def grow(self, items: int) -> None:
        """New items (appended to the KB) start at the mean share."""
        if items > self.share.shape[0]:
            extra = items - self.share.shape[0]
            self.share = torch.cat([self.share, self.share.new_full((extra,), 1.0 / items)])
            self.share /= self.share.sum()
            self.touched = torch.cat([self.touched, self.touched.new_zeros(extra)])

    @torch.no_grad()
    def update(self, item_ids: torch.Tensor, mass: torch.Tensor) -> None:
        batch = torch.zeros_like(self.share).index_add_(0, item_ids.to(self.share.device),
                                                          mass.detach().float().to(self.share.device))
        if float(batch.sum()) > 0:
            self.share.mul_(self.decay).add_(batch / batch.sum(), alpha=1 - self.decay)
        self.touched[item_ids.to(self.share.device)] = True

    def stats(self) -> dict:
        n = self.share.shape[0]
        p = self.share / self.share.sum()
        effective = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
        return {'items': n, 'touched_share': float(self.touched.float().mean()),
                'effective_share': effective / max(n, 1)}


def balance_loss(item_ids: torch.Tensor, gates: torch.Tensor, usage: UsageEMA) -> torch.Tensor:
    """Reward for spread-out use: n * sum_j P_j f_j, with P_j the batch's
    (differentiable) share of gate mass on item j and f_j its moving-average share of
    read mass (``usage``, not differentiated). 1 when mass goes to items in
    proportion to uniform use; larger when it concentrates on already popular items.
    ``item_ids`` (N,) and ``gates`` (N,) list every (read, candidate) pair."""
    total = gates.sum()
    if float(total.detach()) <= 0:
        return gates.sum() * 0.0
    f = usage.share.to(gates.device)[item_ids.to(gates.device)]
    return usage.share.shape[0] * (gates / total * f).sum()


def read_entropy(gates: torch.Tensor) -> torch.Tensor:
    """Effective number of items one read uses (exp of the gate-mass entropy); logged
    to keep reads sparse while use spreads across reads."""
    p = gates.float().clamp_min(0)
    p = p / p.sum().clamp_min(1e-30)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())


def reconstruction_losses(model, stack, examples, outs: list[torch.Tensor],
                          weights: dict) -> tuple[torch.Tensor, dict]:
    """The stack stages' span objective: the frozen decoder reads each produced span
    and reconstructs the example's text (NLL), with a KL to reading the example's
    own span (``ex['span']``), plus a cosine to it measured in the stack's
    standardized space (``Stack.standardize``)."""
    spans = [ex['span'] for ex in examples]
    cos = 1 - F.cosine_similarity(stack.standardize(torch.cat(outs)),
                                  stack.standardize(torch.cat(spans)), dim=-1).mean()
    logits, targets = model.read(examples, outs)
    nll = F.cross_entropy(logits, targets)
    with torch.no_grad():
        t_logits, _ = model.read(examples, spans)
    divergence = kl(logits, t_logits)
    loss = weights['cos'] * cos + weights['nll'] * nll + weights['kl'] * divergence
    return loss, {'cos': cos.item(), 'nll': nll.item(), 'kl': divergence.item()}
