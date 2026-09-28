"""Span protocol tokens in a running decoder (restart plan 3.2, B4; WP1).

The LFM2 decoder ties its input embeddings and LM head. ``ProtocolTokens``
unties them (an exact copy, so the decoder's function is unchanged) and gives
the protocol tokens of ``schnitz.span_tokens`` their own small input and output
tables, applied by forward hooks on the embedding lookup and on the LM head.
Every path that embeds token ids or computes logits therefore sees them, including
chat-template text that contains the reserved tokens, and the tables can train
at their own learning rate while the rest of the decoder trains slowly.

Inside a span the next-token choice is restricted to ``<|rep|>`` against
``<|/bg|>`` (``span_logits``), with a learned bias pair, replacing B2/B3's two-way
stop head; ``<|bg|>`` is an ordinary output token, so the model can open a span
itself. ``marker(factor)`` is ``<|bg|>``'s input embedding plus the writer's ratio
code. A small ratio head predicts log2 of the compression factor at the position
that emits ``<|bg|>``, for prompts that do not state it.
"""
from __future__ import annotations

import torch
from torch import nn

from schnitz.span_tokens import SPAN_TOKENS

NAMES = tuple(SPAN_TOKENS)


class ProtocolTokens(nn.Module):
    def __init__(self, embed: nn.Embedding, lm_head: nn.Linear, ratio: nn.Module,
                 init: dict[str, torch.Tensor] | None = None):
        super().__init__()
        ids = torch.tensor([SPAN_TOKENS[n][1] for n in NAMES])
        self.register_buffer('ids', ids, persistent=False)
        weight = embed.weight.detach().float()
        width = weight.shape[1]
        self.inputs = nn.Parameter(weight[ids].clone())
        self.outputs = nn.Parameter(lm_head.weight.detach().float()[ids].clone())
        self.span_bias = nn.Parameter(torch.zeros(2))
        self.__dict__['ratio'] = ratio  # the writer's ratio code, owned (and saved) by the writer
        self.ratio_head = nn.Linear(width, 1)
        nn.init.zeros_(self.ratio_head.weight)
        nn.init.zeros_(self.ratio_head.bias)
        for name, vector in (init or {}).items():
            table, key = name.split(':')
            getattr(self, table).data[NAMES.index(key)] = vector.float()
        self._hooks = []

    def index(self, name: str) -> int:
        return NAMES.index(name)

    def token(self, name: str) -> int:
        return SPAN_TOKENS[name][1]

    def embedding(self, name: str) -> torch.Tensor:
        return self.inputs[self.index(name)]

    def marker(self, factor: torch.Tensor) -> torch.Tensor:
        return self.inputs[self.index('bg')] + self.ratio(factor)

    def span_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits over (``<|rep|>``, ``<|/bg|>``) at span positions."""
        rows = self.outputs[[self.index('rep'), self.index('bg_end')]]
        return hidden.float() @ rows.t() + self.span_bias

    def install(self, embed: nn.Embedding, lm_head: nn.Linear) -> None:
        """Hook the tables into ``embed`` and ``lm_head`` (untied beforehand)."""
        def on_embed(module, args, output):
            ids = args[0]
            hit = (ids[..., None] == self.ids).any(-1)
            if not hit.any():
                return output
            which = (ids[hit][:, None] == self.ids).float().argmax(-1)
            output = output.clone()
            output[hit] = self.inputs[which].to(output.dtype)
            return output

        def on_head(module, args, output):
            output = output.clone()
            output[..., self.ids] = (args[0].float() @ self.outputs.t()).to(output.dtype)
            return output

        self._hooks = [embed.register_forward_hook(on_embed),
                       lm_head.register_forward_hook(on_head)]


def untie(lm) -> None:
    """Give a tied causal LM its own LM-head weight (an exact copy)."""
    head, embed = lm.lm_head, lm.get_input_embeddings()
    if head.weight.data_ptr() == embed.weight.data_ptr():
        head.weight = nn.Parameter(embed.weight.detach().clone())
    if hasattr(lm.config, 'tie_word_embeddings'):
        lm.config.tie_word_embeddings = False
    if hasattr(lm.config, 'tie_embedding'):
        lm.config.tie_embedding = False
