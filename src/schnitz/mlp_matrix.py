"""MLP-matrix operator for the knowledge-base stack (docs/knowledge-base-stack.md, section 4).

Dense joint recombination of variable-size inputs into a variable number of
outputs. Every (source position j, target position i) pair gets its own
contribution, a two-layer MLP of the source content, both normalized positions,
their difference, the target's residual state and an optional condition (a
target key). Contributions are combined per target in numerator-and-mass form
with the sources' gates, then a per-target feed-forward updates the target
state; several such layers follow each other.

    z_ij = W_src[kind_j] x_j + P φ(p_j) + T LN(h_i) + Q φ(t_i) + D φ(p_j - t_i) + C c
    a_i  = Σ_j w_ij σ(z_ij) / Σ_j w_ij,      w_ij = w_j · K(p_j − t_i)
    w_j  = gate of j's item / its length
    h_i ← h_i + O a_i ;  h_i ← h_i + FFN(LN(h_i))

K is a locality kernel (with ``relative=True``): exp(−Δ² / 2σ²), σ = β · s, s the
larger of the target spacing 1/m and the source item's spacing 1/n, β learnable
per layer and input kind (softplus, starting at 1). Without it every target
averages all sources equally and a single source's signal is diluted by their
number; with it a layer can be local at first and widen (large β is a flat
average) where dense superposition helps. Operators over unordered
neighbourhoods (``relative=False``) use no kernel.

The output map O is linear, so it is applied after the weighted sum: the cost per
pair is one hidden vector. Gates only modulate mass (they are not features): a
gate of 0 removes an item exactly, scaling all gates together changes nothing
but the returned mass, and each item's total mass is its gate, spread evenly over
its positions, so a long item does not outweigh a short one by length alone.
Items carry no order feature, so the output does not depend on the order of the
input items; positions are within-item.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from schnitz.bgkit_span import fourier, interface_rms


class MatrixLayer(nn.Module):
    def __init__(self, sources: dict[str, int], state: int, hidden: int, feats: int,
                 cond: int, relative: bool):
        super().__init__()
        self.source = nn.ModuleDict({kind: nn.Linear(width, hidden) for kind, width in sources.items()})
        self.source_position = nn.Linear(feats, hidden, bias=False)
        self.target_norm = nn.LayerNorm(state)
        self.target = nn.Linear(state, hidden, bias=False)
        self.target_position = nn.Linear(feats, hidden, bias=False)
        self.relative = nn.Linear(feats, hidden, bias=False) if relative else None
        # locality bandwidth per input kind, softplus(0.5413) = 1.0 spacing
        self.bandwidth = nn.ParameterDict({kind: nn.Parameter(torch.tensor(0.5413))
                                           for kind in sources}) if relative else None
        self.condition = nn.Linear(cond, hidden, bias=False) if cond else None
        self.out = nn.Linear(hidden, state)
        self.ffn = nn.Sequential(nn.LayerNorm(state), nn.Linear(state, 2 * state), nn.SiLU(),
                                 nn.Linear(2 * state, state))

    def forward(self, h, sources, source_pos_feats, relative_feats, weights, target_pos_feats,
                cond, locality=None):
        target = self.target(self.target_norm(h)) + self.target_position(target_pos_feats)
        if self.condition is not None and cond is not None:
            target = target + self.condition(cond)
        z = (sources + self.source_position(source_pos_feats))[None] + target[:, None]
        if self.relative is not None:
            z = z + self.relative(relative_feats)
        if locality is None:
            a = torch.einsum('mnh,n->mh', torch.nn.functional.silu(z), weights) / weights.sum()
        else:
            delta, spacing, kinds = locality                          # (m, n), (m, n), per source
            beta = torch.stack([torch.nn.functional.softplus(self.bandwidth[k]) for k in kinds])
            sigma = beta[None] * spacing
            pair = weights[None] * torch.exp(-0.5 * (delta / sigma) ** 2)   # (m, n)
            a = torch.einsum('mnh,mn->mh', torch.nn.functional.silu(z), pair) \
                / pair.sum(1, keepdim=True).clamp_min(1e-30)
        h = h + self.out(a)
        return h + self.ffn(h)


class MLPMatrix(nn.Module):
    """``sources``: input kinds and their widths; ``width``: output width.

    ``forward(items, count, cond=None)`` takes ``items`` as a list of
    ``(kind, x (n, width_kind), gate)`` and returns ``(count, width)`` outputs and
    the total input mass (sum of gates)."""

    def __init__(self, sources: dict[str, int], width: int, *, state: int = 512,
                 hidden: int = 256, layers: int = 3, frequencies: int = 8, cond: int = 0,
                 relative: bool = True, out_norm: float | None = None,
                 checkpoint_layers: bool = False):
        super().__init__()
        feats = 2 * frequencies + 1
        self.frequencies, self.out_norm, self.cond = frequencies, out_norm, cond
        self.checkpoint_layers = checkpoint_layers
        self.init = nn.Sequential(nn.Linear(2 * feats + cond, state), nn.SiLU(),
                                  nn.Linear(state, state))
        self.layers = nn.ModuleList(MatrixLayer(sources, state, hidden, feats, cond, relative)
                                    for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(state), nn.Linear(state, width))

    def forward(self, items: list[tuple[str, torch.Tensor, torch.Tensor | float]], count: int,
                cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        device = items[0][1].device
        gates = torch.stack([torch.as_tensor(g, dtype=torch.float, device=device).clamp_min(0)
                             for _, _, g in items])
        lengths = torch.tensor([x.shape[0] for _, x, _ in items], device=device, dtype=torch.float)
        weights = torch.cat([(g / n).expand(int(n)) for g, n in zip(gates, lengths)])
        positions = torch.cat([(torch.arange(x.shape[0], device=device) + 0.5) / x.shape[0]
                               for _, x, _ in items])
        targets = (torch.arange(count, device=device) + 0.5) / count
        f = self.frequencies
        source_pos = fourier(positions, f)
        target_pos = fourier(targets, f)
        delta = positions[None] - targets[:, None]                           # (m, n)
        relative = fourier(delta, f)                                          # (m, n, feats)
        locality = None
        if self.layers[0].bandwidth is not None:
            item_spacing = torch.cat([torch.full((x.shape[0],), 1.0 / x.shape[0], device=device)
                                      for _, x, _ in items])
            spacing = torch.clamp(item_spacing[None], min=1.0 / count).expand(count, -1)
            kinds = [kind for kind, x, _ in items for _ in range(x.shape[0])]
            locality = (delta, spacing, kinds)
        # size feature: input length (items with a nonzero gate) per output position
        n_eff = float((lengths * (gates > 0)).sum().clamp_min(1))
        ratio = torch.full((count,), math.log2(n_eff / max(count, 1)) / 8, device=device)
        start = [target_pos, fourier(ratio, f)]
        if self.cond:
            if cond is None:
                raise ValueError('this operator is conditioned; pass cond')
            cond = cond.float().expand(count, self.cond)
            start.append(cond)
        h = self.init(torch.cat(start, dim=-1))
        for layer in self.layers:
            sources = torch.cat([layer.source[kind](x.float()) for kind, x, _ in items])
            args = (h, sources, source_pos, relative, weights, target_pos, cond, locality)
            if self.checkpoint_layers and torch.is_grad_enabled():
                h = checkpoint(layer, *args, use_reentrant=False)
            else:
                h = layer(*args)
        out = self.head(h)
        if self.out_norm is not None:
            out = interface_rms(out, self.out_norm)
        return out, gates.sum()
