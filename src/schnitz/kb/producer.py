"""The producer path of KB items and the L2 objective (docs/knowledge-base-stack.md, 5.1
step 8, and 5.2 "Gradients into producers"): source record -> writer span -> forward
codecs -> items per space (-> S_s rewrites for items that came from rewrites).

Used by L2 (producers reproduce the items L1a trained in place: the two-step route) and
by B9 (the items a round writes). Bank creation (``schnitz.kb.bank``, ``train.py l1
build``) runs the same path without gradients.

- ``write_span``: the writer's span of a source under the memory prompt at a ratio level,
  with gradients into the writer's span heads (``SpanWriter.rep`` and the ratio code).
  ``feed='teacher'`` feeds a given span (the bank's cached span of the source) and
  predicts each rep from the ones before it; ``feed='free'`` first free-runs the writer
  without gradients (exactly what bank creation stores), then one gradient pass fed with
  those reps, so each predicted rep equals the free-running one and the gradient reaches
  one step back. The prompt and source are a cached no-gradient prefix: the decoder is
  frozen and nothing trainable acts on it.
- ``produce_items``: the stack's codecs map the span to one item per space.
- ``rewrite_item``: S_s over the produced inputs of a rewrite output, each input at gate
  share x mass (numerator and mass, invariants 5 and 7), conditioned on the output's key.
- ``item_losses`` / ``key_loss`` / ``functional_loss``: the L2 objective. Values per
  space by cosine per position and MSE relative to the target's mean square (each
  space's own scale; the codecs' outputs are rms-normalized per space); keys by cosine
  of the item-key heads' key of the produced values to the target's key; functionally,
  the frozen decoder's reading of R(produced items) against R(target items)
  (KL on the source's reconstruction).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from schnitz.kb.decoder import LEVELS, length_factors
from schnitz.kb.stack import SPACES


def level_index(level: str | int) -> int:
    return LEVELS.index(level) if isinstance(level, str) else int(level)


def span_length(tokens: int, level: str | int) -> tuple[float, int]:
    """(factor, reps) of a write at a ratio level: the B1 length schedule, as
    ``schnitz.kb.bank.write_spans`` uses it."""
    factor = length_factors(int(tokens))[level_index(level)]
    return factor, max(1, math.ceil(int(tokens) / factor))


def _scope(model):
    core = getattr(model, 'core', None)
    return core.autocast() if core is not None else contextlib.nullcontext()


def write_spans(model, examples: Sequence[dict], counts: Sequence[int], feed: str = 'teacher',
                teacher: Sequence[Tensor] | None = None) -> list[Tensor]:
    """Writer spans of ``examples`` (``Model.write`` examples: ``ids`` + ``prompt`` or
    ``inputs``, and ``factor``) with gradients into the writer's span heads.
    ``counts[i]`` reps each. ``feed`` is 'teacher' (``teacher[i]`` fed, (counts[i], width))
    or 'free' (the writer's own free-running reps fed, computed without gradients)."""
    if feed not in ('teacher', 'free'):
        raise ValueError(f'unknown feed {feed!r}')
    with _scope(model):
        if feed == 'free':
            with torch.no_grad():
                own, _ = model.free_run(list(examples), list(counts))
            fed = [o.detach().float() for o in own]
        else:
            if teacher is None:
                raise ValueError('teacher feed needs the spans to feed')
            fed = [t.to(model.device).float() for t in teacher]
            for t, n in zip(fed, counts):
                if t.shape[0] != n:
                    raise ValueError(f'teacher span has {t.shape[0]} reps, expected {n}')
        prefix = model.prefix(list(examples))
        states = model.write(list(examples), fed, prefix)
        return [model.writer.rep(h[:-1]).float() for h in states]


def produce_items(stack, span: Tensor) -> dict[str, Tensor]:
    """The codecs' item of every space for one span (the stack's ``encode``)."""
    return {s: v.float() for s, v in stack.encode(span).items()}


def rewrite_item(operator, inputs: Sequence[tuple[Tensor, float, Tensor | None]],
                 key: Tensor, count: int, neighbour_keys: bool = False) -> Tensor:
    """S_s's output for a rewrite: ``inputs`` are (produced values, share x mass, key).
    Gates only scale mass, so an input with share 0 is exactly absent."""
    out, _ = operator([(v, g, k) for v, g, k in inputs], key.float(), int(count),
                      neighbour_keys=neighbour_keys)
    return out.float()


# -- losses ---------------------------------------------------------------------------
def item_losses(produced: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Per space: ``cos_<s>`` (1 - mean cosine over positions) and ``mse_<s>`` (MSE over the
    target's mean square). Spaces missing on either side are skipped; position counts
    must agree."""
    out = {}
    for s in SPACES:
        if s not in produced or s not in target:
            continue
        p, t = produced[s].float(), target[s].float().to(produced[s].device)
        if p.shape != t.shape:
            raise ValueError(f'space {s}: produced {tuple(p.shape)} vs target {tuple(t.shape)}')
        out[f'cos_{s}'] = 1 - F.cosine_similarity(p, t, dim=-1).mean()
        out[f'mse_{s}'] = (p - t).square().mean() / t.square().mean().clamp_min(1e-12)
    return out


def key_loss(keys, space: str, produced: Tensor, target_key: Tensor) -> Tensor:
    """1 - cosine of the item-key heads' key of the produced values to the target key."""
    k = keys.item_key(space, produced)
    return 1 - F.cosine_similarity(k, target_key.float().to(k.device), dim=-1)


def recombine(stack, items: Mapping[str, Tensor], count: int) -> Tensor:
    """R over the items of every space present (gate 1 each; absent spaces gate 0)."""
    keep = {s: float(s in items) for s in SPACES}
    width = {s: w for s, (_, w) in SPACES.items()}
    full = {s: items[s] if s in items else next(iter(items.values())).new_zeros(1, width[s])
            for s in SPACES}
    return stack.decode(full, keep, int(count))


def functional_loss(model, examples: Sequence[dict], produced: Sequence[Tensor],
                    target: Sequence[Tensor]) -> tuple[Tensor, dict]:
    """KL of the frozen decoder reading ``produced`` spans against reading ``target``
    spans (no gradient) on each example's reconstruction; also both NLLs."""
    from schnitz.kb.losses import kl
    logits, labels = model.read(list(examples), list(produced))
    with torch.no_grad():
        t_logits, _ = model.read(list(examples), [t.detach() for t in target])
    divergence = kl(logits, t_logits)
    return divergence, {'kl': divergence.item(),
                        'nll_produced': F.cross_entropy(logits, labels).item(),
                        'nll_target': F.cross_entropy(t_logits, labels).item()}


@dataclass
class L2Weights:
    cos: float = 1.0
    mse: float = 1.0
    key: float = 1.0
    kl: float = 1.0

    @classmethod
    def parse(cls, text: str) -> L2Weights:
        out = cls()
        for pair in filter(None, text.split(',')):
            name, value = pair.split('=')
            if not hasattr(out, name):
                raise ValueError(f'unknown L2 weight {name!r}')
            setattr(out, name, float(value))
        return out


def l2_loss(parts: Mapping[str, Tensor], weights: L2Weights) -> Tensor:
    """Weighted sum of ``item_losses`` means (cos, mse), the key loss and the KL."""
    total = None

    def add(value, weight):
        nonlocal total
        if weight and value is not None:
            total = weight * value if total is None else total + weight * value
    cos = [v for k, v in parts.items() if k.startswith('cos_')]
    mse = [v for k, v in parts.items() if k.startswith('mse_')]
    add(torch.stack(cos).mean() if cos else None, weights.cos)
    add(torch.stack(mse).mean() if mse else None, weights.mse)
    add(parts.get('key'), weights.key)
    add(parts.get('kl'), weights.kl)
    if total is None:
        raise ValueError('no L2 loss term')
    return total
