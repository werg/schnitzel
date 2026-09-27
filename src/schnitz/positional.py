"""Position-preserving memory codecs and block neural operator readers.

Stored tensors remain flat at the persistence boundary, but their declared
layout is ``[record, source_slot, channels]``.  Operator parameters are shared
over source/target positions; learned position and relative-position features
retain the ordered latent interface without a dense flattened projection.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .readers import ReaderOutput, Statistics, accumulation_dtype, merge_statistics


class PositionalCodec(nn.Module):
    """One shared projection from each canonical writer state to stored channels."""

    def __init__(self, input_dim: int, channels: int) -> None:
        super().__init__()
        if min(input_dim, channels) < 1:
            raise ValueError("Codec dimensions must be positive")
        self.input_dim, self.channels = input_dim, channels
        self.projection = nn.Linear(input_dim, channels)

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim != 3 or states.shape[-1] != self.input_dim:
            raise ValueError("Positional codec expects [batch, slots, width]")
        return self.projection(states).flatten(1)


class JointTokenCodec(nn.Module):
    """Jointly mix every writer slot into a few full-width stored tokens.

    Learned queries cross-attend over all canonical writer states, so each
    stored token can draw on the whole record. Parameters do not depend on the
    writer slot count, which lets slot expansion keep the codec unchanged.
    """

    def __init__(self, width: int, tokens: int, heads: int = 8) -> None:
        super().__init__()
        if min(width, tokens, heads) < 1 or width % heads:
            raise ValueError("Joint codec needs positive tokens and divisible heads")
        self.width, self.tokens = width, tokens
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, width)

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim != 3 or states.shape[-1] != self.width:
            raise ValueError("Joint codec expects [batch, slots, width]")
        memory = self.norm(states)
        queries = self.queries[None].expand(states.shape[0], -1, -1).to(memory.dtype)
        mixed, _ = self.attention(queries, memory, memory, need_weights=False)
        return self.output(mixed).flatten(1)


class PositionOperatorRound(nn.Module):
    """A shared input-position -> target-position residual message operator."""

    def __init__(self, input_dim: int, query_dim: int, width: int,
                 source_slots: int, target_slots: int,
                 target_position_ids: Tensor | None = None) -> None:
        super().__init__()
        if min(input_dim, query_dim, width, source_slots, target_slots) < 1:
            raise ValueError("Operator dimensions must be positive")
        self.source_slots, self.target_slots = source_slots, target_slots
        self.input = nn.Linear(input_dim, width)
        self.query = nn.Linear(query_dim, width, bias=False)
        self.state = nn.Linear(width, width, bias=False)
        self.source_position = nn.Parameter(torch.randn(source_slots, width) * 0.02)
        self.target_position = nn.Parameter(torch.randn(target_slots, width) * 0.02)
        self.relative = nn.Linear(1, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1)
        source = torch.linspace(0, 1, source_slots)
        if target_position_ids is None:
            target = torch.linspace(0, 1, target_slots)
        else:
            if target_position_ids.shape != (target_slots,):
                raise ValueError("Expected one local position for every target slot")
            denominator = max(source_slots - 1, 1)
            target = target_position_ids.float() / denominator
        self.register_buffer("relative_offsets", source[:, None] - target[None, :], persistent=False)

    def forward(self, x: Tensor, query: Tensor, state: Tensor,
                weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4 or x.shape[2] != self.source_slots:
            raise ValueError("Operator expects [batch, records, source_slots, channels]")
        # [B,N,S,T,W].  Record chunks bound this temporary tensor.
        h = self.input(x)[:, :, :, None, :]
        h = h + self.state(state)[:, None, None, :, :]
        h = h + self.query(query)[:, None, None, None, :]
        h = h + self.source_position[None, None, :, None, :]
        h = h + self.target_position[None, None, None, :, :]
        relative = self.relative(self.relative_offsets[..., None].to(x.dtype))
        h = F.silu(h + relative[None, None])
        dtype = accumulation_dtype(x)
        log_gate = F.logsigmoid(self.gate(h).to(dtype))
        largest = log_gate.amax((1, 2)).detach()
        scale = torch.where(largest < -40, largest, torch.zeros_like(largest))
        gates = torch.where((scale < 0)[:, None, None],
                            (log_gate - scale[:, None, None]).exp(),
                            log_gate.exp())
        # A record's shares across its source positions sum to its record weight.
        shares = weights[:, :, None, None, None].to(dtype) / self.source_slots
        gates = gates * shares
        messages = self.message(h).to(dtype)
        numerator = (gates * messages).sum((1, 2))
        mass = gates.sum((1, 2))
        return numerator, mass, scale


class PositionalSetReader(nn.Module):
    """Repeated block neural operator over stored positions and target slots."""

    def __init__(self, input_dim: int, source_slots: int, query_dim: int,
                 output_dim: int, *, width: int = 128, slots: int = 8,
                 rounds: int = 3, kind: str = "mlp", chunk_size: int = 64,
                 checkpoint_chunks: bool = False) -> None:
        super().__init__()
        if input_dim % source_slots:
            raise ValueError("Payload width must divide into source slots")
        if kind != "mlp":
            raise ValueError("The positional interface currently uses the MLP operator")
        self.input_dim, self.source_slots = input_dim, source_slots
        self.channels = input_dim // source_slots
        self.width, self.slots, self.rounds = width, slots, rounds
        self.kind, self.chunk_size = kind, chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.initial_query = nn.Linear(query_dim, width)
        self.initial_slots = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.blocks = nn.ModuleList([
            PositionOperatorRound(self.channels, query_dim, width, source_slots, slots)
            for _ in range(rounds)
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
        return self.initial_query(query)[:, None] + self.initial_slots[None]

    def aggregate(self, t: int, x: Tensor, query: Tensor, state: Tensor,
                  weights: Tensor) -> Statistics:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("Expected flat stored values [batch, records, payload_dim]")
        if x.shape[1] == 0:
            dtype = accumulation_dtype(x)
            numerator = torch.zeros_like(state, dtype=dtype)
            mass = torch.zeros((*state.shape[:2], 1), device=state.device, dtype=dtype)
            return Statistics(numerator, mass, torch.zeros_like(mass))
        structured = x.reshape(*x.shape[:2], self.source_slots, self.channels)
        parts = []
        for start in range(0, x.shape[1], self.chunk_size):
            args = (structured[:, start:start + self.chunk_size], query, state,
                    weights[:, start:start + self.chunk_size])
            block = self.blocks[t]
            tensors = (checkpoint(block, *args, use_reentrant=False, preserve_rng_state=True)
                       if self.checkpoint_chunks and self.training and torch.is_grad_enabled()
                       else block(*args))
            parts.append(Statistics(*tensors))
        return merge_statistics(parts)

    def update(self, t: int, state: Tensor, stats: Statistics) -> Tensor:
        mean = stats.mean().to(state.dtype)
        log_count = F.softplus(stats.log_mass()).to(state.dtype)
        delta = self.updates[t](torch.cat((self.norms[t](state), mean, log_count), -1))
        delta = delta + self.slot_mix[t](delta.transpose(1, 2)).transpose(1, 2)
        return state + delta / math.sqrt(self.rounds)

    def forward(self, x: Tensor, query: Tensor, weights: Tensor | None = None,
                *, diagnostics: bool = False) -> ReaderOutput:
        if x.ndim != 3 or query.ndim != 2 or x.shape[0] != query.shape[0]:
            raise ValueError("Expected values [B,N,D] and queries [B,Q]")
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
        tokens = torch.where((weights.sum(1) > 0)[:, None, None], tokens,
                             self.null_tokens[None])
        return ReaderOutput(tokens, states, statistics)


class MultiSpacePositionalReader(nn.Module):
    """Per-space positional operators coupled through one target residual stream."""

    def __init__(self, input_dims: list[int], source_slots: int | list[int], query_dim: int,
                 output_dim: int, **kwargs) -> None:
        super().__init__()
        counts = ([source_slots] * len(input_dims) if isinstance(source_slots, int)
                  else list(source_slots))
        if len(counts) != len(input_dims):
            raise ValueError("One stored position count is required per space")
        self.local = nn.ModuleList([
            PositionalSetReader(d, count, query_dim, output_dim, **kwargs)
            for d, count in zip(input_dims, counts, strict=True)
        ])
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
        return torch.where(present[:, None, None], tokens,
                           self.local[0].null_tokens[None])


class PositionalCompactor(nn.Module):
    """Operator compactor from positional records to positional synthetic records."""

    def __init__(self, input_dim: int, source_slots: int, width: int = 128,
                 records: int = 2, rounds: int = 3, chunk_size: int = 32) -> None:
        super().__init__()
        if input_dim % source_slots:
            raise ValueError("Compactor payload must divide into source slots")
        self.input_dim, self.source_slots = input_dim, source_slots
        self.channels, self.records = input_dim // source_slots, records
        self.width, self.rounds, self.chunk_size = width, rounds, chunk_size
        targets = records * source_slots
        local_positions = torch.arange(source_slots).repeat(records)
        self.initial_slots = nn.Parameter(torch.randn(targets, width) * 0.02)
        self.blocks = nn.ModuleList([
            PositionOperatorRound(self.channels, 1, width, source_slots, targets,
                                  target_position_ids=local_positions)
            for _ in range(rounds)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(rounds)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 1, 2 * width), nn.SiLU(),
                          nn.Linear(2 * width, width)) for _ in range(rounds)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.channels))
        self.multiplicity = nn.Linear(width + 1, 1)

    def forward(self, x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or weights.shape != x.shape[:2]:
            raise ValueError("Positional compactor expects [B,N,D] and [B,N]")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Compaction weights must be finite and nonnegative")
        batch = x.shape[0]
        state = self.initial_slots[None].expand(batch, -1, -1)
        # Cluster mass is a density/scale feature, not an absolute key position.
        query = torch.log1p(weights.sum(1, keepdim=True)).to(x.dtype)
        structured = x.reshape(batch, x.shape[1], self.source_slots, self.channels)
        last_mass = None
        for t, block in enumerate(self.blocks):
            parts = []
            for start in range(0, x.shape[1], self.chunk_size):
                parts.append(Statistics(*block(
                    structured[:, start:start + self.chunk_size], query, state,
                    weights[:, start:start + self.chunk_size])))
            stats = merge_statistics(parts)
            mean = stats.mean().to(state.dtype)
            log_count = F.softplus(stats.log_mass()).to(state.dtype)
            delta = self.updates[t](torch.cat((self.norms[t](state), mean, log_count), -1))
            state = state + delta / math.sqrt(self.rounds)
            last_mass = stats.log_mass().to(state.dtype)
        values = self.output(state).reshape(batch, self.records, self.input_dim)
        grouped = state.reshape(batch, self.records, self.source_slots, self.width).mean(2)
        log_mass = last_mass.reshape(batch, self.records, self.source_slots, 1).mean(2)
        logits = self.multiplicity(torch.cat((grouped, F.softplus(log_mass)), -1)).squeeze(-1)
        total = weights.sum(1, keepdim=True)
        multiplicity = logits.softmax(-1) * total
        return values, multiplicity
