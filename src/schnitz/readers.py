"""Fixed-slot, permutation-invariant readers with explicit additive merge boundaries.

Record order is exchangeable, output slot identity is not. Complete each aggregate
before updating residual state. Checkpointing replays chunks without truncating
query, gate, state, or parameter gradient paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def accumulation_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


@dataclass
class Statistics:
    """Actual numerator/mass equal stored tensors multiplied by exp(log_scale)."""
    numerator: Tensor  # [batch, slots, width]
    mass: Tensor  # [batch, slots, 1]
    log_scale: Tensor  # [batch, slots, 1]; also stabilizes very small learned MLP gates

    def mean(self) -> Tensor:
        present = self.mass > 0
        denominator = torch.where(present, self.mass, torch.ones_like(self.mass))
        return torch.where(present, self.numerator / denominator, torch.zeros_like(self.numerator))

    def log_mass(self) -> Tensor:
        return torch.where(
            self.mass > 0,
            self.mass.clamp_min(torch.finfo(self.mass.dtype).tiny).log() + self.log_scale,
            torch.full_like(self.mass, -torch.inf),
        )


def merge_statistics(parts: list[Statistics]) -> Statistics:
    """Associative up to rounding; also supports attention's log-scaled statistics."""
    if not parts:
        raise ValueError("At least one statistics object is required")
    scale = torch.stack([p.log_scale for p in parts]).amax(0).detach()
    numerator = sum(p.numerator * (p.log_scale - scale).exp() for p in parts)
    mass = sum(p.mass * (p.log_scale - scale).exp() for p in parts)
    return Statistics(numerator, mass, scale)


@dataclass
class ReaderOutput:
    tokens: Tensor
    states: list[Tensor]  # state BEFORE each round; optional training diagnostics
    statistics: list[Statistics]


class MLPRound(nn.Module):
    def __init__(self, input_dim: int, query_dim: int, width: int, slots: int,
                 gate: str = "learned", activation: str = "silu") -> None:
        super().__init__()
        if gate not in {"unit", "learned"} or activation not in {"silu", "relu"}:
            raise ValueError("Invalid gate or activation")
        self.input = nn.Linear(input_dim, width)
        self.query = nn.Linear(query_dim, width, bias=False)
        self.state = nn.Linear(width, width, bias=False)
        self.slot = nn.Parameter(torch.randn(slots, width) * 0.02)
        # Bias must not be added only once AFTER a weighted sum.
        self.output = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1) if gate == "learned" else None
        self.activation = F.silu if activation == "silu" else F.relu

    def forward(self, x: Tensor, query: Tensor, state: Tensor,
                weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.activation(
            self.input(x)[:, :, None, :]
            + self.state(state)[:, None, :, :]
            + self.query(query)[:, None, None, :]
            + self.slot[None, None, :, :]
        )
        dtype = accumulation_dtype(x)
        if self.gate is not None:
            logits = self.gate(h)
            log_gate = F.logsigmoid(logits.to(dtype))
            largest = log_gate.amax(1).detach()
            # Preserve the ordinary serialized-precision path. When every gate
            # in a slot becomes tiny, factor out a common scale BEFORE pooling
            # and division: the mathematically cancelling 1/mass gradients must
            # not overflow in BF16/FP32. Keep the scale for absolute mass and
            # cross-chunk/compaction merging, just as the attention reader does.
            scale = torch.where(largest < -40, largest, torch.zeros_like(largest))
            g = torch.where((scale < 0)[:, None], (log_gate - scale[:, None]).exp(),
                            logits.sigmoid().to(dtype))
        else:
            g = torch.ones_like(h[..., :1], dtype=dtype)
            scale = torch.zeros_like(g[:, 0])
        g = g * weights[:, :, None, None].to(dtype)
        pooled = (g * h.to(dtype)).sum(1)
        mass = g.sum(1)
        # Project once per output slot. Accumulation remains FP32 in BF16 runs.
        numerator = self.output(pooled.to(self.output.weight.dtype)).to(dtype)
        return numerator, mass, scale


class AttentionRound(nn.Module):
    """Single-head fixed-slot cross-attention, with a stable compaction boundary.

    Multiple fixed slots interact through the same residual updater as the MLP.
    This intentionally simple comparator is not a claim to optimize all attention variants.
    """
    def __init__(self, input_dim: int, query_dim: int, width: int, slots: int) -> None:
        super().__init__()
        self.key = nn.Linear(input_dim, width, bias=False)
        self.value = nn.Linear(input_dim, width, bias=False)
        self.query = nn.Linear(query_dim, width, bias=False)
        self.state = nn.Linear(width, width, bias=False)
        self.slot = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.width = width

    def forward(self, x: Tensor, query: Tensor, state: Tensor,
                weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dtype = accumulation_dtype(x)
        q = self.state(state) + self.query(query)[:, None] + self.slot[None]
        score = torch.einsum("bmd,bnd->bmn", q.to(dtype), self.key(x).to(dtype))
        score = score / math.sqrt(self.width)
        # Use a weighted exponential, not log(weights), to preserve derivatives
        # through nonnegative multiplicities, including zero-weight records.
        # Include zero-weight records in the numerical shift: masking their scores
        # would silently change the derivative w.r.t. a multiplicity at zero.
        # The shift cancels from the normalized output and from merged statistics.
        scale = score.amax(-1, keepdim=True).detach()
        unnormalized = (score - scale).exp() * weights[:, None, :].to(dtype)
        numerator = torch.einsum("bmn,bnd->bmd", unnormalized, self.value(x).to(dtype))
        mass = unnormalized.sum(-1, keepdim=True)
        return numerator, mass, scale


class SetReader(nn.Module):
    def __init__(self, input_dim: int, query_dim: int, output_dim: int, *,
                 width: int = 128, slots: int = 8, rounds: int = 3,
                 kind: str = "mlp", gate: str = "learned", activation: str = "silu",
                 chunk_size: int = 64, checkpoint_chunks: bool = False) -> None:
        super().__init__()
        if min(input_dim, query_dim, output_dim, width, slots, rounds, chunk_size) < 1:
            raise ValueError("All reader dimensions/counts must be positive")
        if kind not in {"mlp", "attention"}:
            raise ValueError(f"Unknown reader: {kind}")
        self.input_dim, self.query_dim = input_dim, query_dim
        self.width, self.slots, self.rounds = width, slots, rounds
        self.kind, self.chunk_size = kind, chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.initial_query = nn.Linear(query_dim, width)
        self.initial_slots = nn.Parameter(torch.randn(slots, width) * 0.02)
        if kind == "mlp":
            self.blocks = nn.ModuleList([
                MLPRound(input_dim, query_dim, width, slots, gate, activation)
                for _ in range(rounds)
            ])
        else:
            self.blocks = nn.ModuleList([
                AttentionRound(input_dim, query_dim, width, slots) for _ in range(rounds)
            ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 1, 2 * width), nn.SiLU(),
                          nn.Linear(2 * width, width)) for _ in range(rounds)
        ])
        self.slot_mix = nn.ModuleList([nn.Linear(slots, slots, bias=False) for _ in range(rounds)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(rounds)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
        self.null_tokens = nn.Parameter(torch.randn(slots, output_dim) * 0.02)

    def initial_state(self, query: Tensor) -> Tensor:
        return self.initial_query(query)[:, None, :] + self.initial_slots[None]

    def aggregate(self, t: int, x: Tensor, query: Tensor, state: Tensor,
                  weights: Tensor) -> Statistics:
        if x.shape[1] == 0:
            dtype = accumulation_dtype(x)
            numerator = torch.zeros_like(state, dtype=dtype)
            mass = torch.zeros((*state.shape[:2], 1), device=state.device, dtype=dtype)
            return Statistics(numerator, mass, torch.zeros_like(mass))
        parts = []
        for start in range(0, x.shape[1], self.chunk_size):
            args = (x[:, start:start + self.chunk_size], query, state,
                    weights[:, start:start + self.chunk_size])
            block = self.blocks[t]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                tensors = checkpoint(block, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                tensors = block(*args)
            parts.append(Statistics(*tensors))
        return merge_statistics(parts)

    def update(self, t: int, state: Tensor, stats: Statistics) -> Tensor:
        mean = stats.mean().to(state.dtype)
        log_count = F.softplus(stats.log_mass()).to(state.dtype)  # log(1 + actual mass)
        features = torch.cat((self.norms[t](state), mean, log_count), -1)
        delta = self.updates[t](features)
        # Fixed-size slot mixing, never neighborhood self-attention.
        delta = delta + self.slot_mix[t](delta.transpose(1, 2)).transpose(1, 2)
        return state + delta / math.sqrt(self.rounds)

    def forward(self, x: Tensor, query: Tensor, weights: Tensor | None = None,
                *, diagnostics: bool = False) -> ReaderOutput:
        if x.ndim != 3 or query.ndim != 2 or x.shape[0] != query.shape[0]:
            raise ValueError("Expected values [B,N,D] and queries [B,Q]")
        if x.shape[-1] != self.input_dim or query.shape[-1] != self.query_dim:
            raise ValueError("Reader dimension mismatch")
        if weights is None:
            weights = x.new_ones(x.shape[:2])
        if weights.shape != x.shape[:2] or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be finite nonnegative [B,N]")
        state = self.initial_state(query)
        states, statistics = [], []
        for t in range(self.rounds):
            stats = self.aggregate(t, x, query, state, weights)
            if diagnostics:
                states.append(state)
                statistics.append(stats)
            state = self.update(t, state, stats)
        tokens = self.output(state)
        tokens = torch.where((weights.sum(1) > 0)[:, None, None], tokens, self.null_tokens[None])
        return ReaderOutput(tokens, states, statistics)


class MultiSpaceReader(nn.Module):
    """Multiple payload widths, one shared residual stream across all rounds.

    Each local branch exposes the same aggregate API for compaction probes.
    Single-space uses SetReader directly; this branch is an opt-in experiment.
    """
    def __init__(self, input_dims: list[int], query_dim: int, output_dim: int, **kwargs) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("At least one space is required")
        self.local = nn.ModuleList([SetReader(d, query_dim, output_dim, **kwargs) for d in input_dims])
        first = self.local[0]
        self.width, self.rounds, self.slots = first.width, first.rounds, first.slots
        self.fusion = nn.ModuleList([
            nn.Linear(len(input_dims) * self.width, self.width, bias=False)
            for _ in range(self.rounds)
        ])

    def forward(self, values: list[Tensor], query: Tensor, weights: list[Tensor]) -> Tensor:
        if len(values) != len(self.local) or len(weights) != len(self.local):
            raise ValueError("Space count mismatch")
        state = self.local[0].initial_state(query)
        for t in range(self.rounds):
            changes = []
            for reader, x, weight in zip(self.local, values, weights, strict=True):
                stats = reader.aggregate(t, x, query, state, weight)
                changes.append(reader.update(t, state, stats) - state)
            state = state + self.fusion[t](torch.cat(changes, -1))
        tokens = self.local[0].output(state)
        present = torch.stack([w.sum(1) > 0 for w in weights]).any(0)
        return torch.where(present[:, None, None], tokens, self.local[0].null_tokens[None])
