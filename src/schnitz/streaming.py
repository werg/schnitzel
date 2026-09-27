"""Bounded-payload inference through the additive reader boundary.

A factory reopens the same immutable, visibility-checked read plan each round.
Only one payload chunk plus fixed-size reader statistics is staged on the device.
This is intentionally inference-only; training uses chunk checkpointing/replay.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from .readers import SetReader, ReaderOutput, Statistics, merge_statistics, accumulation_dtype


@torch.no_grad()
def read_stream(reader: SetReader, query: Tensor,
                chunks: Callable[[], Iterable[tuple[Tensor, Tensor]]], *, diagnostics: bool = False) -> ReaderOutput:
    if reader.training or torch.is_grad_enabled():
        raise ValueError('Streamed reader requires eval mode')
    state = reader.initial_state(query)
    states, statistics = [], []
    expected_count = None
    has_values = torch.zeros(query.shape[0], device=query.device, dtype=torch.bool)
    for t in range(reader.rounds):
        stats, count = None, 0
        for values, weights in chunks():
            if values.ndim != 3 or values.shape[0] != query.shape[0] or values.shape[-1] != reader.input_dim:
                raise ValueError('Invalid streamed payload shape')
            if weights.shape != values.shape[:2] or (weights < 0).any() or not torch.isfinite(weights).all():
                raise ValueError('Invalid streamed multiplicities')
            if values.shape[1] == 0:
                continue
            count += values.shape[1]
            if t == 0:
                has_values |= (weights.sum(1) > 0).to(query.device)
            # Transfers are synchronous by design. Pinned/double-buffered overlap is a later adapter.
            values = values.to(device=query.device, dtype=reader.blocks[t].input.weight.dtype
                               if reader.kind == 'mlp' else reader.blocks[t].key.weight.dtype)
            weights = weights.to(device=query.device, dtype=values.dtype)
            part = Statistics(*reader.blocks[t](values, query, state, weights))
            stats = part if stats is None else merge_statistics([stats, part])
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise ValueError('The streamed read plan changed between reader rounds')
        if stats is None:
            dtype = accumulation_dtype(query)
            mass = torch.zeros((*state.shape[:2], 1), device=query.device, dtype=dtype)
            stats = Statistics(torch.zeros_like(state, dtype=dtype), mass, torch.zeros_like(mass))
        if diagnostics:
            states.append(state)
            statistics.append(stats)
        state = reader.update(t, state, stats)
    tokens = reader.output(state)
    tokens = torch.where(has_values[:, None, None], tokens, reader.null_tokens[None].to(tokens.dtype))
    return ReaderOutput(tokens, states, statistics)
