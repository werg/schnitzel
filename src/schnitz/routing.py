"""Discrete candidate plans and group-aware training within those candidates."""
from __future__ import annotations

import itertools
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cosine_similarities(query: Tensor, keys: Tensor) -> Tensor:
    """Normalized similarities without conflating geometry with loss temperature."""
    return F.normalize(query.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T


def cosine_scores(query: Tensor, keys: Tensor, temperature: float = 0.1) -> Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return cosine_similarities(query, keys) / temperature


class AdaptiveDistanceGate(nn.Module):
    """Density-relative nonnegative record gates with bounded learned bandwidth.

    Candidate discovery remains hard.  The forward reader receives ordinary
    materialized records multiplied by these continuous relevance weights.  The
    local reference similarity is detached so keys cannot improve every gate by
    manipulating the density estimator itself.
    """

    def __init__(self, query_dim: int, *, density_k: int = 8,
                 min_temperature: float = 0.02, max_temperature: float = 0.5,
                 initial_temperature: float = 0.1,
                 max_radius_adjustment: float = 0.25,
                 density_fraction: float = 0.0, floor_mode: str = 'max') -> None:
        super().__init__()
        if (query_dim < 1 or density_k < 1 or not 0 < min_temperature
                < initial_temperature < max_temperature
                or max_radius_adjustment < 0 or not 0 <= density_fraction <= 1
                or floor_mode not in {'max', 'additive'}):
            raise ValueError('Invalid adaptive distance-gate configuration')
        self.density_k = density_k
        # With large reads, a fixed k-th-neighbour radius shuts the gate on almost
        # every record read. A fraction scales the boundary with the read size.
        self.density_fraction = density_fraction
        # 'max' floors supported records but blocks their gate gradient;
        # 'additive' (floor + (1 - floor) * w) keeps it everywhere.
        self.floor_mode = floor_mode
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_radius_adjustment = max_radius_adjustment
        self.adjust = nn.Linear(query_dim, 2)
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)
        fraction = ((initial_temperature - min_temperature)
                    / (max_temperature - min_temperature))
        with torch.no_grad():
            self.adjust.bias[1] = torch.logit(torch.tensor(fraction))

    def forward(self, query: Tensor, selected_scores: Tensor, candidate_scores: Tensor,
                selected_mask: Tensor, candidate_mask: Tensor,
                support_mask: Tensor | None = None,
                support_floor: float = 0.0) -> tuple[Tensor, dict[str, Tensor]]:
        if (query.ndim != 2 or selected_scores.ndim != 2 or candidate_scores.ndim != 2
                or selected_mask.shape != selected_scores.shape
                or candidate_mask.shape != candidate_scores.shape
                or query.shape[0] != selected_scores.shape[0]
                or query.shape[0] != candidate_scores.shape[0]):
            raise ValueError('Adaptive gate tensors have incompatible shapes')
        if not 0 <= support_floor <= 1:
            raise ValueError('Support gate floor must be between zero and one')
        if support_mask is not None and support_mask.shape != selected_scores.shape:
            raise ValueError('Support mask differs from selected records')
        boundaries = []
        for scores, valid, selected in zip(candidate_scores, candidate_mask, selected_mask,
                                           strict=True):
            available = scores[valid]
            if not available.numel():
                raise ValueError('Adaptive gating needs at least one candidate per query')
            k = self.density_k
            if self.density_fraction:
                k = max(k, round(self.density_fraction * int(selected.sum())))
            k = min(k, available.numel())
            boundaries.append(available.topk(k).values[-1])
        boundary = torch.stack(boundaries).detach()
        adjustment_input = F.normalize(query.to(self.adjust.weight.dtype), dim=-1)
        adjustments = self.adjust(adjustment_input).to(selected_scores.dtype)
        radius = boundary + adjustments[:, 0].tanh() * self.max_radius_adjustment
        fraction = adjustments[:, 1].sigmoid()
        temperature = self.min_temperature + fraction * (
            self.max_temperature - self.min_temperature)
        weights = ((selected_scores - radius[:, None]) / temperature[:, None]).sigmoid()
        # Gradient of each weight with respect to its score, before any floor.
        slope = (weights * (1 - weights) / temperature[:, None]).detach()
        if support_mask is not None and support_floor:
            if self.floor_mode == 'additive':
                floored = support_floor + (1 - support_floor) * weights
                slope = torch.where(support_mask, slope * (1 - support_floor), slope)
            else:
                floored = torch.maximum(weights, torch.full_like(weights, support_floor))
                slope = torch.where(support_mask & (weights < support_floor),
                                    torch.zeros_like(slope), slope)
            weights = torch.where(support_mask, floored, weights)
        weights = weights * selected_mask.to(weights.dtype)
        mass = weights.sum(-1)
        effective = mass.square() / weights.square().sum(-1).clamp_min(1e-12)
        return weights, {'radius': radius, 'temperature': temperature,
                         'mass': mass, 'effective_records': effective, 'slope': slope}


def union_support_loss(space_scores: list[Tensor],
                       positive_indices: list[tuple[int, ...]]) -> Tensor:
    """Gently anchor every labelled fact in at least one retrieval space.

    Labels are positive evidence only.  Unlabelled records remain alternatives
    rather than being declared false, and spaces are free to specialize.
    """
    if (not space_scores or len(space_scores) != len(positive_indices)
            or any(scores.ndim != 1 for scores in space_scores)):
        raise ValueError('One score vector and positive index tuple per space required')
    count = len(positive_indices[0])
    if count < 1 or any(len(indices) != count for indices in positive_indices):
        raise ValueError('Support identities must align across spaces')
    losses = []
    for positive in range(count):
        probabilities = []
        for scores, indices in zip(space_scores, positive_indices, strict=True):
            index = indices[positive]
            if not 0 <= index < scores.numel():
                raise ValueError('Support index is outside its candidate field')
            probabilities.append(scores.softmax(-1)[index])
        probability = 1 - torch.stack([1 - value for value in probabilities]).prod()
        losses.append(-probability.clamp_min(1e-12).log())
    return torch.stack(losses).mean()


def group_plan_loss(scores: Tensor, sufficient_groups: list[tuple[int, ...]]) -> Tensor:
    """Negative log probability of a sufficient unordered prefix under PL sampling.

    Groups must have equal cardinality and be distinct. Enumerates <=4-member
    groups. This rewards useful starters even when their isolated utility is zero.
    Hard top-k candidate discovery itself is NOT made differentiable by this loss.
    """
    if scores.ndim != 1 or not sufficient_groups:
        raise ValueError("Expected 1D candidate scores and at least one sufficient group")
    groups = {tuple(sorted(g)) for g in sufficient_groups}
    lengths = {len(g) for g in groups}
    if len(lengths) != 1 or not 1 <= next(iter(lengths)) <= 4:
        raise ValueError("Reference requires equal-size groups with 1..4 members")
    if any(len(g) != len(set(g)) or min(g) < 0 or max(g) >= scores.numel() for g in groups):
        raise ValueError("Invalid support group")
    probabilities = []
    for group in groups:
        for order in itertools.permutations(group):
            available = torch.ones_like(scores, dtype=torch.bool)
            log_probability = scores.sum() * 0
            for index in order:
                log_probability = log_probability + scores[index] - torch.logsumexp(
                    scores.masked_fill(~available, -torch.inf), dim=0)
                available = available.clone()
                available[index] = False
            probabilities.append(log_probability)
    return -torch.logsumexp(torch.stack(probabilities), dim=0)


def utility_ranking_loss(scores: Tensor, utilities: Tensor, tolerance: float = 1e-6) -> Tensor:
    """Do not interpret indistinguishable isolated utilities as anti-group evidence."""
    if scores.shape != utilities.shape:
        raise ValueError("Shape mismatch")
    if (utilities.max() - utilities.min()).detach().item() <= tolerance:
        return scores.sum() * 0
    return -(utilities.detach().softmax(-1) * scores.log_softmax(-1)).sum()


def complete_support_recall(selected: set[str], groups: list[set[str]]) -> float:
    return float(any(group <= selected for group in groups))


def validate_routing_dataset(config, episodes) -> None:
    """Do not turn supplied teacher context into fictitious sufficient-set labels."""
    if config.train.retrieval != 'learned' or config.train.arm not in {'memory', 'direct_latent'}:
        return
    informative = 0
    for episode in episodes:
        if episode.support_annotation != 'verified':
            raise ValueError('Learned group routing requires verified support labels; provided_context '
                             'episodes support oracle training only. Collect utility/sufficiency labels first.')
        if not 1 <= len(episode.required_ids) <= 4:
            raise ValueError('Learned group routing requires 1..4 verified required records')
        informative += len(episode.supports) > len(episode.required_ids)
    if not informative and not config.train.bank_dir:
        raise ValueError('Learned routing has no competing candidates: every candidate is required. '
                         'Provide causally eligible distractors or use oracle retrieval.')
