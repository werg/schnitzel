"""Temporary full-cluster compaction and contribution-preserving training losses.

Persistent arbitrary-child selection is intentionally not approximated by a
full-cluster code. FullClusterCode.require_selection fails closed for subsets.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .readers import SetReader, Statistics, accumulation_dtype


def mean_and_mass(x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    if weights.shape != x.shape[:2] or (weights < 0).any():
        raise ValueError("Expected nonnegative [B,N] weights")
    dtype = accumulation_dtype(x)
    mass = weights.to(dtype).sum(1, keepdim=True)
    mean = (x.to(dtype) * weights.to(dtype)[..., None]).sum(1, keepdim=True)
    mean = mean / mass[..., None].clamp_min(torch.finfo(dtype).tiny)
    return mean.to(x.dtype), mass.to(x.dtype)


class SyntheticCompactor(nn.Module):
    """Amortized, permutation-invariant cluster -> r synthetic ordinary records.

    Effective multiplicity is conserved. The initial output is near a replicated
    mean; shared encoder parameters, not independently optimized per-cluster codes.
    """
    def __init__(self, input_dim: int, width: int = 128, records: int = 2) -> None:
        super().__init__()
        if min(input_dim, width, records) < 1:
            raise ValueError("Dimensions must be positive")
        self.records, self.input_dim = records, input_dim
        self.encode = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.codes = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(),
                                   nn.Linear(width, records * input_dim))
        self.multiplicity = nn.Linear(width + 1, records)
        nn.init.normal_(self.codes[-1].weight, std=1e-3)
        nn.init.zeros_(self.codes[-1].bias)

    def forward(self, x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        mean, mass = mean_and_mass(x, weights)
        feature, _ = mean_and_mass(self.encode(x), weights)
        feature = torch.cat((feature[:, 0], torch.log1p(mass)), -1)
        codes = self.codes(feature).reshape(x.shape[0], self.records, self.input_dim)
        codes = mean + codes
        weights_out = self.multiplicity(feature).softmax(-1) * mass
        return codes, weights_out


def statistics_loss(student: Statistics, teacher: Statistics) -> Tensor:
    """Match numerator AND mass at a common log scale; teacher is detached."""
    scale = torch.maximum(student.log_scale, teacher.log_scale).detach()
    sn = student.numerator * (student.log_scale - scale).exp()
    tn = teacher.numerator.detach() * (teacher.log_scale.detach() - scale).exp()
    sm = student.mass * (student.log_scale - scale).exp()
    tm = teacher.mass.detach() * (teacher.log_scale.detach() - scale).exp()
    # Reference normalization avoids making large clusters dominate the loss.
    denominator = tm.detach().clamp_min(1.0)
    return F.mse_loss(sn / denominator, tn / denominator) + F.mse_loss(sm / denominator, tm / denominator)


def contribution_loss(reader: SetReader, raw: Tensor, raw_weights: Tensor,
                      compact: Tensor, compact_weights: Tensor, query: Tensor,
                      rollout_weight: float = 1.0) -> Tensor:
    """Teacher-state conditional statistics plus free compact rollout matching.

    The uncompressed branch is detached; the compact branch can train the reader,
    writer and/or compactor. Caller decides which parameters are frozen.
    """
    with torch.no_grad():
        teacher = reader(raw, query, raw_weights, diagnostics=True)
    loss = compact.sum() * 0
    for t, (state, target) in enumerate(zip(teacher.states, teacher.statistics, strict=True)):
        current = reader.aggregate(t, compact, query, state.detach(), compact_weights)
        loss = loss + statistics_loss(current, target) / reader.rounds
    if rollout_weight:
        predicted = reader(compact, query, compact_weights).tokens
        loss = loss + rollout_weight * F.mse_loss(predicted, teacher.tokens.detach())
    return loss


def local_merge_loss(reader: SetReader, raw: Tensor, weights: Tensor, query: Tensor) -> Tensor:
    mean, mass = mean_and_mass(raw, weights)
    return contribution_loss(reader, raw, weights, mean, mass, query)


def field_responsibilities(points: Tensor, centers: Tensor, memberships: int = 2,
                           temperature: float = 1.0) -> Tensor:
    """Two/few-nearest-center partition of unity; field shares sum to one per point."""
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("Expected [N,D] points and [C,D] centers")
    if not 1 <= memberships <= centers.shape[0] or temperature <= 0:
        raise ValueError("Invalid membership count or temperature")
    distances = torch.cdist(points.float(), centers.float()).square()
    distance, indices = distances.topk(memberships, largest=False, dim=-1)
    coefficients = (-distance / temperature).softmax(-1)
    return torch.zeros_like(distances).scatter(1, indices, coefficients).to(points.dtype)


def adaptive_field_responsibilities(points: Tensor, centers: Tensor, *, memberships: int = 2,
                                    density_k: int = 8, minimum_scale: float = 1e-3,
                                    log_scale_adjustment: Tensor | None = None) -> Tensor:
    """Overlapping local charts with density-relative bandwidth and conserved mass.

    The k-nearest input radius supplies each center's scale.  A compactor may
    provide a bounded learned log-scale adjustment, but absolute key position is
    not an input.  Responsibilities remain a partition of unity per raw record.
    """
    if (points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]
            or not 1 <= memberships <= centers.shape[0] or density_k < 1
            or minimum_scale <= 0):
        raise ValueError('Invalid adaptive compaction field geometry')
    distances = torch.cdist(points.float(), centers.float()).square()
    k = min(density_k, points.shape[0])
    scale = distances.topk(k, dim=0, largest=False).values[-1].clamp_min(
        minimum_scale ** 2).detach()
    if log_scale_adjustment is not None:
        if log_scale_adjustment.shape != (centers.shape[0],):
            raise ValueError('One scale adjustment is required per compaction center')
        scale = scale * log_scale_adjustment.float().tanh().exp()
    local_distance, indices = distances.topk(memberships, largest=False, dim=-1)
    local_scale = scale[indices]
    coefficients = (-local_distance / local_scale).softmax(-1)
    return torch.zeros_like(distances).scatter(1, indices, coefficients).to(points.dtype)


def random_partition(n: int, group_size: int, generator: torch.Generator) -> list[list[int]]:
    """Alternative training views; a point occurs exactly once in each view."""
    if n < 0 or group_size < 1:
        raise ValueError("Invalid partition size")
    perm = torch.randperm(n, generator=generator).tolist()
    return [perm[i:i + group_size] for i in range(0, n, group_size)]


def storage_noise(x: Tensor, *, noise_std: float = 0.0, quantization_step: float = 0.0) -> Tensor:
    """Training-only perturbation; replay captures torch RNG. No inference regeneration."""
    if noise_std < 0 or quantization_step < 0:
        raise ValueError("Noise levels cannot be negative")
    if noise_std:
        x = x + noise_std * torch.randn_like(x)
    if quantization_step:
        x = x + (torch.rand_like(x) - 0.5) * quantization_step
    return x


@dataclass(frozen=True)
class FullClusterCode:
    children: tuple[str, ...]
    authorization_domain: str
    reader_version: str

    def require_selection(self, selected: tuple[str, ...], domain: str) -> None:
        if domain != self.authorization_domain:
            raise PermissionError("Compacted payload crosses an authorization boundary")
        if set(selected) != set(self.children) or len(selected) != len(self.children):
            raise ValueError("Full-cluster code cannot answer an arbitrary partial selection")


@dataclass
class CompactView:
    values: Tensor
    weights: Tensor
    groups: list[list[int]]
    input_records: int
    output_records: int


def compact_view(x: Tensor, weights: Tensor, *, method: str = "mean",
                 compactor: SyntheticCompactor | None = None, grouping: str = "whole",
                 group_size: int = 4, memberships: int = 2,
                 generator: torch.Generator | None = None,
                 geometry: Tensor | None = None) -> CompactView:
    """Temporary local replacement, including exact partition-of-unity field shares.

    A training read has batch size one; grouping is discrete and detached. Every
    field share participates, so this is not an approximate field retrieval plan.
    A small group stays raw rather than being expanded into more synthetic records.
    Overlap can still increase the total record count: report it, do not hide it.
    """
    if x.ndim != 3 or x.shape[0] != 1 or weights.shape != x.shape[:2]:
        raise ValueError("compact_view expects [1,N,D] values and [1,N] weights")
    if group_size < 1 or memberships < 1 or grouping not in {"whole", "random", "local", "overlap"}:
        raise ValueError("Invalid grouping")
    if method not in {"mean", "synthetic"} or (method == "synthetic" and compactor is None):
        raise ValueError("Invalid compaction method")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonnegative finite weights required")
    if geometry is not None and (geometry.ndim != 2 or geometry.shape[0] != x.shape[1]):
        raise ValueError('Compaction geometry must provide one key per record')
    n = x.shape[1]
    if n == 0:
        return CompactView(x, weights, [], 0, 0)
    # CPU permutations work identically with a CPU generator for CPU/CUDA payloads.
    order = torch.randperm(n, generator=generator).tolist() if grouping != "whole" else list(range(n))
    groups, shares = [], []
    if grouping == "whole":
        groups = [order]
        shares = [weights]
    elif grouping == "random":
        groups = [order[i:i + group_size] for i in range(0, n, group_size)]
        shares = [weights[:, g] for g in groups]
    elif grouping == "local":
        remaining = set(order)
        points = x[0].detach().float()
        for seed in order:
            if seed not in remaining:
                continue
            candidates = sorted(remaining)
            distances = (points[candidates] - points[seed]).square().sum(-1)
            nearest = distances.argsort(stable=True)[:group_size].tolist()
            group = [candidates[i] for i in nearest]
            groups.append(group)
            shares.append(weights[:, group])
            remaining.difference_update(group)
    else:
        count = max(1, (n + group_size - 1) // group_size)
        points = (x[0] if geometry is None else geometry).detach().float()
        coefficients = adaptive_field_responsibilities(
            points, points[order[:count]], memberships=min(memberships, count),
            density_k=min(group_size, n)).detach()
        for field in range(count):
            ids = (coefficients[:, field] > 0).nonzero().flatten().tolist()
            if ids:
                groups.append(ids)
                shares.append(weights[:, ids] * coefficients[ids, field][None].to(weights.dtype))
    values_out, weights_out = [], []
    size = 1 if method == "mean" else compactor.records
    for ids, share in zip(groups, shares, strict=True):
        value = x[:, ids]
        if len(ids) <= size:
            compact, mass = value, share
        elif method == "mean":
            compact, mass = mean_and_mass(value, share)
        else:
            compact, mass = compactor(value, share)
        values_out.append(compact)
        weights_out.append(mass)
    values = torch.cat(values_out, 1)
    masses = torch.cat(weights_out, 1)
    return CompactView(values, masses, groups, n, values.shape[1])
