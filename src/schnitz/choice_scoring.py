"""Opt-in candidate batching for a previously captured, fixed memory result.

The serial path remains the numerical reference. Native BF16 batch-shape changes
can alter scores slightly and must be measured before adopting this in a study.
"""
from dataclasses import replace

import torch
from torch import Tensor

from .recurrence import LoopMemory


@torch.no_grad()
def candidate_nll(agent, prompt, memory, targets, *, max_batch=4):
    """Full sequence NLL including EOS; right-padding losses are never counted.

All candidates receive the same prefix and captured memory. No query, retrieval,
writer or compactor runs here. Causal attention/recurrence keeps right-padding
tokens after each candidate's last scored position.
"""
    if agent.training:
        raise ValueError('Candidate batching requires a frozen eval-mode model')
    if max_batch < 1 or prompt.shape[0] != 1:
        raise ValueError('Positive batch limit and exactly one prompt required')
    if any(t.ndim != 2 or t.shape[0] != 1 or t.shape[1] < 1 for t in targets):
        raise ValueError('Each candidate must be a nonempty [1, tokens] tensor')
    if max_batch == 1:
        return [float(agent.conditioned_nll(prompt, target, memory, reduction='sum')) for target in targets]
    result = []
    for start in range(0, len(targets), max_batch):
        group = targets[start:start + max_batch]
        count = len(group)
        lengths = torch.tensor([t.shape[1] for t in group], device=prompt.device)
        padded = group[0].new_zeros(count, max(t.shape[1] for t in group))
        for index, target in enumerate(group):
            padded[index, :target.shape[1]] = target[0]
        def expand(value):
            if value is None:
                return None
            if value.shape[0] != 1:
                raise ValueError('Captured memory must belong to one prompt')
            return value.expand(count, -1, -1)
        if isinstance(memory, LoopMemory):
            captured = replace(memory, events=tuple(expand(event) for event in memory.events))
        elif isinstance(memory, Tensor):
            captured = expand(memory)
        elif memory is None:
            captured = None
        else:
            raise TypeError('Unsupported captured memory type')
        losses = agent.conditioned_nll(prompt.expand(count, -1), padded, captured, reduction='none').reshape(count, -1)
        valid = torch.arange(padded.shape[1], device=prompt.device)[None] < lengths[:, None]
        result.extend(losses.masked_fill(~valid, 0.).sum(1).tolist())
    return result
