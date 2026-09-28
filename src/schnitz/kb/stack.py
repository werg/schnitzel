"""The knowledge-base stack's modules (docs/knowledge-base-stack.md, sections 3-4):
spaces, forward codecs, recombiner, superposition operators and key heads, all
built from the MLP-matrix operator (``schnitz.mlp_matrix``). Every stage uses
these definitions.
"""
from __future__ import annotations

import math

import torch

from schnitz.mlp_matrix import MLPMatrix

# space: (positions per span rep, width); the widths times the ratios sum to 960
SPACES = {'A': (1.0, 384), 'B': (0.5, 512), 'C': (0.25, 768), 'D': (0.125, 1024)}


class Stack(torch.nn.Module):
    def __init__(self, target_norm: float, state: int, hidden: int, layers: int,
                 checkpointing: bool):
        super().__init__()
        common = dict(state=state, hidden=hidden, layers=layers,
                      checkpoint_layers=checkpointing)
        self.codecs = torch.nn.ModuleDict({
            name: MLPMatrix({'span': 1024}, width, out_norm=math.sqrt(width), **common)
            for name, (_, width) in SPACES.items()})
        self.recombiner = MLPMatrix({name: width for name, (_, width) in SPACES.items()}, 1024,
                                    out_norm=target_norm, **common)

    def encode(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        n = span.shape[0]
        return {name: self.codecs[name]([('span', span, 1.0)], max(1, math.ceil(ratio * n)))[0]
                for name, (ratio, _) in SPACES.items()}

    def decode(self, items: dict[str, torch.Tensor], keep: dict[str, float], count: int):
        return self.recombiner([(name, items[name], keep[name]) for name in SPACES], count)[0]
