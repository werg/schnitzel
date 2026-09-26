"""Key-space geometry: collapse and hubness metrics, and spreading regularizers.

Utility-driven key learning feeds back on itself: only retrieved records get
gradient, and a collapsed space retrieves the same few records for everyone.
These terms keep each space's geometry spread at small weight. They act within
one space and never reward differences between spaces.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def effective_rank(x: Tensor) -> float:
    """exp(entropy) of the normalized singular values of centered rows."""
    x = x.float()
    values = torch.linalg.svdvals(x - x.mean(0))
    p = values / values.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def mean_direction_cosine(x: Tensor) -> float:
    """Mean cosine of unit rows with their mean direction; 1 means collinear."""
    unit = F.normalize(x.float(), dim=-1)
    return float((unit @ F.normalize(unit.mean(0), dim=0)).mean())


def uniformity(x: Tensor, t: float = 2.0, sample: int = 4096,
               generator: torch.Generator | None = None) -> float:
    """Wang and Isola's log E exp(-t |xi - xj|^2) on unit rows (lower is more uniform)."""
    unit = F.normalize(x.float(), dim=-1)
    if len(unit) > sample:
        index = torch.randperm(len(unit), generator=generator)[:sample].to(unit.device)
        unit = unit[index]
    distances = torch.pdist(unit).pow(2)
    return float(torch.logsumexp(-t * distances, 0) - math.log(len(distances)))


def occurrence_statistics(neighbors: Tensor, records: int) -> dict:
    """Hubness from top-k lists [queries, k] over ``records`` keys."""
    counts = torch.bincount(neighbors.reshape(-1).long().cpu(), minlength=records).float()
    mean, std = counts.mean(), counts.std()
    skew = float(((counts - mean) ** 3).mean() / std.clamp_min(1e-12) ** 3)
    ordered = counts.sort().values
    rank = torch.arange(1, records + 1, dtype=torch.float32)
    gini = float(((2 * rank - records - 1) * ordered).sum()
                 / (records * ordered.sum()).clamp_min(1e-12))
    return {'skewness': skew, 'gini': gini,
            'never_retrieved': float((counts == 0).float().mean()),
            'max_share': float(counts.max() / counts.sum().clamp_min(1e-12))}


def geometry_summary(keys: Tensor, queries: Tensor, k: int = 16) -> dict:
    """Collapse and hubness metrics for one space's keys and query addresses."""
    unit_keys = F.normalize(keys.float(), dim=-1)
    unit_queries = F.normalize(queries.float(), dim=-1)
    neighbors = torch.cat([(chunk @ unit_keys.T).topk(min(k, len(unit_keys)), -1).indices
                           for chunk in unit_queries.split(1024)])
    return {'key_effective_rank': effective_rank(keys),
            'query_effective_rank': effective_rank(queries),
            'key_mean_direction_cosine': mean_direction_cosine(keys),
            'query_mean_direction_cosine': mean_direction_cosine(queries),
            'key_uniformity': uniformity(keys),
            f'hubness_at_{k}': occurrence_statistics(neighbors, len(keys))}


def variance_covariance_loss(x: Tensor, gamma: float = 1.0) -> tuple[Tensor, Tensor]:
    """VICReg-style terms on unit rows: per-dimension spread and decorrelation.

    A uniform distribution on the unit sphere has per-dimension standard
    deviation 1/sqrt(d); the variance hinge keeps every dimension near that.
    """
    unit = F.normalize(x.float(), dim=-1)
    n, d = unit.shape
    centered = unit - unit.mean(0)
    std = (centered.pow(2).mean(0) + 1e-6).sqrt()
    variance = F.relu(gamma / math.sqrt(d) - std).mean() * math.sqrt(d)
    correlation = (centered.T @ centered) / (n - 1) / (std[:, None] * std[None])
    off = correlation - torch.diag(torch.diagonal(correlation))
    covariance = off.pow(2).sum() / (d * (d - 1))
    return variance, covariance


def koleo_loss(x: Tensor, eps: float = 1e-8) -> Tensor:
    """KoLeo: raise each unit row's distance to its nearest neighbour (even local density)."""
    unit = F.normalize(x.float(), dim=-1)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(-2.0)
    nearest = similarity.max(1).values
    distance = (2 - 2 * nearest).clamp_min(0).add(eps).sqrt()
    return -distance.add(eps).log().mean()


class RetrievalLoad:
    """Running retrieval counts per record, a hub load penalty and exploration sampling."""

    def __init__(self, records: int, decay: float = 0.999, threshold: float = 4.0) -> None:
        if not 0 < decay < 1 or threshold <= 1:
            raise ValueError('Invalid retrieval-load settings')
        self.counts = torch.zeros(records, dtype=torch.float64)
        self.decay, self.threshold = decay, threshold
        self.updates = 0

    def update(self, retrieved: Tensor) -> None:
        self.counts.mul_(self.decay)
        self.counts.index_add_(0, retrieved.reshape(-1).long().cpu(),
                               torch.ones(retrieved.numel(), dtype=torch.float64))
        self.updates += 1

    def overload(self, records: Tensor) -> Tensor:
        """log(load / threshold × expected load), floored at zero, for the given records."""
        expected = self.counts.sum() / len(self.counts)
        if expected <= 0:
            return torch.zeros(len(records), dtype=torch.float32)
        ratio = self.counts[records.long().cpu()] / (self.threshold * expected)
        return ratio.clamp_min(1e-12).log().clamp_min(0).float()

    def penalty(self, queries: Tensor, keys: Tensor, records: Tensor) -> Tensor:
        """Push overloaded field keys away from the query mass; zero for balanced keys."""
        weight = self.overload(records).to(keys.device)
        if not bool(weight.any()):
            return keys.sum() * 0
        similarity = F.normalize(queries.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T
        return (similarity.mean(0) * weight).sum() / weight.gt(0).sum()

    def explore(self, candidates: Tensor, count: int,
                generator: torch.Generator | None = None) -> Tensor:
        """Sample ``count`` candidates with probability proportional to 1/(1 + load)."""
        if count <= 0 or not len(candidates):
            return candidates[:0]
        weights = 1.0 / (1.0 + self.counts[candidates.long().cpu()])
        chosen = torch.multinomial(weights.float(), min(count, len(candidates)),
                                   replacement=False, generator=generator)
        return candidates[chosen.to(candidates.device)]

    def state_dict(self) -> dict:
        return {'counts': self.counts.clone(), 'updates': self.updates,
                'decay': self.decay, 'threshold': self.threshold}

    def load_state_dict(self, state: dict) -> None:
        if (len(state['counts']) != len(self.counts) or state['decay'] != self.decay
                or state['threshold'] != self.threshold):
            raise ValueError('Retrieval-load state does not match this bank')
        self.counts = state['counts'].clone()
        self.updates = int(state['updates'])


class BankLoad:
    """Per-space retrieval load over one bank's record identities.

    Exploration and the load penalty read a snapshot frozen at the start of each
    optimizer step, and retrievals recorded during the step are committed after
    it, so pipelined retrieval threads cannot make a step nondeterministic.
    """

    def __init__(self, record_ids, spaces: int, decay: float = 0.999,
                 threshold: float = 4.0) -> None:
        self.ids = [str(record_id) for record_id in record_ids]
        self.position = {record_id: i for i, record_id in enumerate(self.ids)}
        self.loads = [RetrievalLoad(len(self.ids), decay, threshold) for _ in range(spaces)]
        self.snapshots = [load.counts.clone() for load in self.loads]
        self.pending: list[list[Tensor]] = [[] for _ in range(spaces)]
        self.active: Tensor | None = None

    def restrict(self, record_ids) -> None:
        """Limit load expectations, exploration and statistics to an active subset."""
        if record_ids is None:
            self.active = None
            return
        self.active = torch.zeros(len(self.ids), dtype=torch.bool)
        self.active[self.positions(sorted(record_ids))] = True

    def begin_step(self) -> None:
        self.snapshots = [load.counts.clone() for load in self.loads]
        self.pending = [[] for _ in self.loads]

    def positions(self, record_ids) -> Tensor:
        return torch.tensor([self.position[record_id] for record_id in record_ids],
                            dtype=torch.long)

    def record(self, space: int, record_ids) -> None:
        if record_ids:
            self.pending[space].append(self.positions(record_ids))

    def commit(self) -> None:
        for load, pending in zip(self.loads, self.pending, strict=True):
            load.update(torch.cat(pending) if pending else torch.zeros(0, dtype=torch.long))
        self.pending = [[] for _ in self.loads]

    def overload(self, space: int, record_ids) -> Tensor:
        load = self.loads[space]
        counts = self.snapshots[space]
        # Expected load per record among those that can be retrieved at all.
        population = len(counts) if self.active is None else int(self.active.sum())
        expected = counts.sum() / max(population, 1)
        if expected <= 0:
            return torch.zeros(len(record_ids), dtype=torch.float32)
        ratio = counts[self.positions(record_ids)] / (load.threshold * expected)
        return ratio.clamp_min(1e-12).log().clamp_min(0).float()

    def penalty(self, space: int, query: Tensor, keys: Tensor, record_ids) -> Tensor:
        weight = self.overload(space, record_ids).to(keys.device)
        if not bool(weight.any()):
            return keys.sum() * 0
        similarity = (F.normalize(query.float(), dim=-1)
                      @ F.normalize(keys.float(), dim=-1).T).mean(0)
        return (similarity * weight).sum() / weight.gt(0).sum()

    def explore(self, space: int, count: int, seed: int, exclude=(),
                eligible=None) -> list[str]:
        """Draw proposals with probability proportional to 1/(1 + load).

        ``eligible`` (a boolean mask in record order) restricts the draw to records
        the query may read; the caller still filters proposals for causal and
        authorization eligibility. Without it, low-load records that the query
        cannot read would crowd out every proposal.
        """
        if count <= 0:
            return []
        weights = 1.0 / (1.0 + self.snapshots[space].float())
        if self.active is not None:
            weights = weights * self.active
        if eligible is not None:
            weights = weights * torch.as_tensor(eligible, dtype=torch.bool)
        if not bool((weights > 0).any()):
            return []
        for record_id in exclude:
            position = self.position.get(record_id)
            if position is not None:
                weights[position] = 0
        generator = torch.Generator().manual_seed(seed)
        drawn = torch.multinomial(weights, min(count, int((weights > 0).sum())),
                                  replacement=False, generator=generator)
        return [self.ids[i] for i in drawn.tolist()]

    def statistics(self) -> list[dict]:
        rows = []
        for load in self.loads:
            counts = load.counts.float()
            if self.active is not None:
                counts = counts[self.active]
            ordered = counts.sort().values
            n = len(counts)
            rank = torch.arange(1, n + 1, dtype=torch.float32)
            gini = float(((2 * rank - n - 1) * ordered).sum()
                         / (n * ordered.sum()).clamp_min(1e-12))
            rows.append({'gini': gini, 'cold_fraction': float((counts < 1e-3).float().mean()),
                         'max_share': float(counts.max() / counts.sum().clamp_min(1e-12))})
        return rows

    def state_dict(self) -> dict:
        return {'ids_sha256': self._ids_digest(),
                'loads': [load.state_dict() for load in self.loads]}

    def _ids_digest(self) -> str:
        return hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()

    def load_state_dict(self, state: dict) -> None:
        if (len(state['loads']) != len(self.loads)
                or state['ids_sha256'] != self._ids_digest()):
            raise ValueError('Bank load state belongs to another bank or space count')
        for load, saved in zip(self.loads, state['loads'], strict=True):
            load.load_state_dict(saved)
        self.begin_step()
