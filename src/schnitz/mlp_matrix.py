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

Every input kind passes a LayerNorm before its projection. Inputs arrive at very
different scales (decoder-space spans have a per-dimension RMS near 0.025, the
positional Fourier features near 1); without the normalization the content term
of z_ij starts some 30 times weaker than the position terms and the operator
settles on a position-only code.

Items may carry per-item extra features (``extra`` > 0), e.g. the item's key for a
superposition operator that sees its neighbours' keys (K3b). They enter z_ij
through their own normalization and a projection that starts at zero, so an
operator trained without them (K3a) continues unchanged when they are switched on.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from schnitz.bgkit_span import fourier, interface_rms


def _to(t: torch.Tensor, device) -> torch.Tensor:
    """A host tensor on ``device`` by a pinned non-blocking copy (no wait for the queue)."""
    if torch.device(device).type == 'cuda':
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


class MatrixLayer(nn.Module):
    def __init__(self, sources: dict[str, int], state: int, hidden: int, feats: int,
                 cond: int, relative: bool, extra: int = 0):
        super().__init__()
        self.extra = nn.Linear(extra, hidden, bias=False) if extra else None
        if self.extra is not None:
            nn.init.zeros_(self.extra.weight)
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
                cond, locality=None, extra=None):
        if self.extra is not None and extra is not None:
            sources = sources + self.extra(extra)
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

    def forward_packed(self, h, sources, source_pos_feats, relative_feats, weights,
                       target_pos_feats, cond, pair_t, pair_s, locality=None, extra=None):
        """``forward`` for many independent (targets, sources) groups at once: the pairs
        are listed explicitly (``pair_t``, ``pair_s``: every target with every source of
        its own group) and the weighted means are segment sums over each target's pairs."""
        if self.extra is not None and extra is not None:
            sources = sources + self.extra(extra)
        target = self.target(self.target_norm(h)) + self.target_position(target_pos_feats)
        if self.condition is not None and cond is not None:
            target = target + self.condition(cond)
        src = sources + self.source_position(source_pos_feats)
        z = src[pair_s] + target[pair_t]
        if self.relative is not None:
            z = z + self.relative(relative_feats)
        w = weights[pair_s]
        if locality is not None:
            delta, spacing, kind_index, names = locality              # per pair
            beta = torch.stack([torch.nn.functional.softplus(self.bandwidth[k])
                                for k in names])[kind_index]
            w = w * torch.exp(-0.5 * (delta / (beta * spacing)) ** 2)
        prod = torch.nn.functional.silu(z) * w[:, None].to(z.dtype)
        num = torch.zeros(h.shape[0], prod.shape[1], device=prod.device, dtype=prod.dtype)
        num = num.index_add(0, pair_t, prod)
        den = torch.zeros(h.shape[0], device=w.device, dtype=w.dtype).index_add(0, pair_t, w)
        a = num / den.clamp_min(1e-30)[:, None].to(num.dtype)
        h = h + self.out(a)
        return h + self.ffn(h)


class MLPMatrix(nn.Module):
    """``sources``: input kinds and their widths; ``width``: output width.

    ``forward(items, count, cond=None)`` takes ``items`` as a list of
    ``(kind, x (n, width_kind), gate)`` or ``(kind, x, gate, extra (extra,))`` and returns ``(count, width)`` outputs and
    the total input mass (sum of gates)."""

    def __init__(self, sources: dict[str, int], width: int, *, state: int = 512,
                 hidden: int = 256, layers: int = 3, frequencies: int = 8, cond: int = 0,
                 relative: bool = True, out_norm: float | None = None,
                 checkpoint_layers: bool = False, extra: int = 0):
        super().__init__()
        self.extra_width = extra
        self.extra_norm = nn.LayerNorm(extra) if extra else None
        feats = 2 * frequencies + 1
        self.frequencies, self.out_norm, self.cond = frequencies, out_norm, cond
        self.checkpoint_layers = checkpoint_layers
        self.input_norm = nn.ModuleDict({kind: nn.LayerNorm(width) for kind, width in sources.items()})
        self.init = nn.Sequential(nn.Linear(2 * feats + cond, state), nn.SiLU(),
                                  nn.Linear(state, state))
        self.layers = nn.ModuleList(MatrixLayer(sources, state, hidden, feats, cond, relative,
                                                extra)
                                    for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(state), nn.Linear(state, width))

    def forward(self, items: list[tuple[str, torch.Tensor, torch.Tensor | float]], count: int,
                cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        items = [(it[0], it[1], it[2], it[3] if len(it) > 3 else None) for it in items]
        device = items[0][1].device
        gates = torch.stack([torch.as_tensor(g, dtype=torch.float, device=device).clamp_min(0)
                             for _, _, g, _ in items])
        lengths = torch.tensor([x.shape[0] for _, x, _, _ in items], device=device, dtype=torch.float)
        weights = torch.cat([(g / n).expand(int(n)) for g, n in zip(gates, lengths)])
        positions = torch.cat([(torch.arange(x.shape[0], device=device) + 0.5) / x.shape[0]
                               for _, x, _, _ in items])
        targets = (torch.arange(count, device=device) + 0.5) / count
        f = self.frequencies
        source_pos = fourier(positions, f)
        target_pos = fourier(targets, f)
        delta = positions[None] - targets[:, None]                           # (m, n)
        relative = fourier(delta, f)                                          # (m, n, feats)
        locality = None
        if self.layers[0].bandwidth is not None:
            item_spacing = torch.cat([torch.full((x.shape[0],), 1.0 / x.shape[0], device=device)
                                      for _, x, _, _ in items])
            spacing = torch.clamp(item_spacing[None], min=1.0 / count).expand(count, -1)
            kinds = [kind for kind, x, _, _ in items for _ in range(x.shape[0])]
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
        normed = [(kind, self.input_norm[kind](x.float())) for kind, x, _, _ in items]
        extra = None
        if self.extra_norm is not None:  # per-item features, zero where an item has none
            extra = torch.cat([
                (self.extra_norm(e.float()) if e is not None
                 else torch.zeros(self.extra_width, device=device))[None].expand(x.shape[0], -1)
                for _, x, _, e in items])
        for layer in self.layers:
            sources = torch.cat([layer.source[kind](x) for kind, x in normed])
            args = (h, sources, source_pos, relative, weights, target_pos, cond, locality, extra)
            if self.checkpoint_layers and torch.is_grad_enabled():
                h = checkpoint(layer, *args, use_reentrant=False)
            else:
                h = layer(*args)
        out = self.head(h)
        if self.out_norm is not None:
            out = interface_rms(out, self.out_norm)
        return out, gates.sum()

    def forward_many(self, groups: list[tuple[list, int, torch.Tensor | None]],
                     max_pairs: int | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``forward`` of many independent groups ``(items, count, cond)`` in one pass per
        chunk of groups (at most ``max_pairs`` (target, source) position pairs per chunk).
        Each group's result equals ``forward(items, count, cond)`` up to floating-point
        summation order; returns [(output (count, width), mass)] in group order."""
        out: list = [None] * len(groups)
        chunk: list[int] = []
        pairs = 0
        for n, (items, count, _) in enumerate(groups):
            size = int(count) * sum(int(it[1].shape[0]) for it in items)
            if chunk and max_pairs and pairs + size > max_pairs:
                for m, got in zip(chunk, self._packed([groups[i] for i in chunk])):
                    out[m] = got
                chunk, pairs = [], 0
            chunk.append(n)
            pairs += size
        if chunk:
            for m, got in zip(chunk, self._packed([groups[i] for i in chunk])):
                out[m] = got
        return out

    def _packed(self, groups) -> list[tuple[torch.Tensor, torch.Tensor]]:
        items = [[(it[0], it[1], it[2], it[3] if len(it) > 3 else None) for it in g[0]]
                 for g in groups]
        counts = [int(g[1]) for g in groups]
        flat = [it for group in items for it in group]
        device = flat[0][1].device
        f = self.frequencies
        # index arithmetic on the host, one pinned non-blocking copy each (no device syncs)
        n_items = torch.tensor([len(group) for group in items])
        lengths_h = torch.tensor([int(x.shape[0]) for _, x, _, _ in flat])
        count_h = torch.tensor(counts)
        item_of_h = torch.repeat_interleave(torch.arange(len(flat)), lengths_h)
        starts_h = torch.cumsum(lengths_h, 0) - lengths_h
        within_h = torch.arange(item_of_h.shape[0]) - starts_h[item_of_h]
        n_pos_h = lengths_h[item_of_h].float()
        positions_h = (within_h.float() + 0.5) / n_pos_h
        group_of_item_h = torch.repeat_interleave(torch.arange(len(groups)), n_items)
        n_src_h = torch.zeros(len(groups), dtype=torch.long).index_add(0, group_of_item_h,
                                                                       lengths_h)
        tgt_group_h = torch.repeat_interleave(torch.arange(len(groups)), count_h)
        tgt_start_h = torch.cumsum(count_h, 0) - count_h
        tgt_local_h = torch.arange(tgt_group_h.shape[0]) - tgt_start_h[tgt_group_h]
        targets_h = (tgt_local_h.float() + 0.5) / count_h[tgt_group_h].float()
        per_pair_h = count_h * n_src_h        # every target with every source of its group
        pair_group_h = torch.repeat_interleave(torch.arange(len(groups)), per_pair_h)
        pair_start_h = torch.cumsum(per_pair_h, 0) - per_pair_h
        local_h = torch.arange(pair_group_h.shape[0]) - pair_start_h[pair_group_h]
        src_start_h = torch.cumsum(n_src_h, 0) - n_src_h
        pair_t_h = tgt_start_h[pair_group_h] + torch.div(local_h, n_src_h[pair_group_h],
                                                         rounding_mode='floor')
        pair_s_h = src_start_h[pair_group_h] + local_h % n_src_h[pair_group_h]
        spacing_h = (1.0 / n_pos_h[pair_s_h]).maximum(1.0 / count_h[pair_group_h].float())
        kinds = sorted({kind for kind, _, _, _ in flat})
        kind_item_h = torch.tensor([kinds.index(k) for k, _, _, _ in flat])
        (lengths, count_t, item_of, positions, group_of_item, tgt_group, targets, pair_t,
         pair_s, spacing, kind_pos) = (_to(t, device) for t in (
             lengths_h, count_h, item_of_h, positions_h, group_of_item_h, tgt_group_h,
             targets_h, pair_t_h, pair_s_h, spacing_h, kind_item_h[item_of_h]))
        gates = torch.stack([torch.as_tensor(g, dtype=torch.float, device=device)
                             for _, _, g, _ in flat]).clamp_min(0)
        weights = (gates / lengths.float())[item_of]
        delta = positions[pair_s] - targets[pair_t]
        relative = fourier(delta, f)
        locality = None
        if self.layers[0].bandwidth is not None:
            locality = (delta, spacing, kind_pos[pair_s], kinds)
        # size feature per group: input length of items with a nonzero gate
        live = torch.zeros(len(groups), device=device).index_add(
            0, group_of_item, lengths.float() * (gates > 0).float()).clamp_min(1)
        ratio = (torch.log2(live / count_t.float().clamp_min(1)) / 8)[tgt_group]
        start = [fourier(targets, f), fourier(ratio, f)]
        cond = None
        if self.cond:
            if any(c is None for _, _, c in groups):
                raise ValueError('this operator is conditioned; pass cond')
            cond = torch.stack([c.float().to(device).reshape(self.cond)
                                for _, _, c in groups])[tgt_group]
            start.append(cond)
        h = self.init(torch.cat(start, dim=-1))
        target_pos = start[0]
        source_pos = fourier(positions, f)
        # inputs normalized per kind (row-wise LayerNorms, so one call per kind)
        normed = [(kind, self.input_norm[kind](torch.cat([x.float() for k, x, _, _ in flat
                                                          if k == kind])))
                  for kind in kinds]
        inverse = None
        if len(kinds) > 1:      # sources are computed kind by kind; back to position order
            order = torch.argsort(kind_pos, stable=True)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.shape[0], device=device)
        extra = None
        if self.extra_norm is not None:
            has = [i for i, it in enumerate(flat) if it[3] is not None]
            per_item = torch.zeros(len(flat), self.extra_width, device=device)
            if has:
                per_item = per_item.index_copy(0, _to(torch.tensor(has), device),
                                               self.extra_norm(torch.stack(
                                                   [flat[i][3].float() for i in has])))
            extra = per_item[item_of]
        for layer in self.layers:
            sources = torch.cat([layer.source[kind](x) for kind, x in normed])
            if inverse is not None:
                sources = sources[inverse]
            args = (h, sources, source_pos, relative, weights, target_pos, cond, pair_t, pair_s,
                    locality, extra)
            if self.checkpoint_layers and torch.is_grad_enabled():
                h = checkpoint(layer.forward_packed, *args, use_reentrant=False)
            else:
                h = layer.forward_packed(*args)
        out = self.head(h)
        if self.out_norm is not None:
            out = interface_rms(out, self.out_norm)
        masses = torch.zeros(len(groups), device=device).index_add(0, group_of_item, gates)
        return list(zip(torch.split(out, counts), masses.unbind(0)))
