"""L1 read path at a ``memory_search()`` call (docs/knowledge-base-stack.md, 3, 5.1 steps 4 and 7).

Built on the shared stack modules (``schnitz.kb.stack``: ``KeyHeads``,
``SuperpositionOperator``, ``Stack``'s recombiner, ``read_count``;
``schnitz.kb.losses``: ``retrieval_loss``).

At a call the frozen decoder's hidden state after ``query_layer`` layers, at the
call's own position, is the query state; the query heads map it to one unit key per
space (queries are vectors, never text). Items get their keys from the item-key heads
applied to their values. Per space:

1. **Retrieval.** Exact top-k (``kb_store.search_kbs``) over the stored (in the L1
   writer: live) keys of the KBs the caller is authorized for (``allowed`` lists
   datasets; learned selection is never authorization, invariant 6), limited to
   items whose time is at or before the query time (invariant 2). The search is
   discrete and runs on detached queries; the stored keys are a cache of the item-key
   heads' output, refreshed by the trainer (``rekey``).
2. **Scores and gates.** Every candidate is scored from its key recomputed live from
   its current values (``KeyHeads.scores``: cosine times a learned scale), so scores
   carry gradients into the query heads, the item-key heads and the item values. The
   gate is ``sigmoid(scale_s (cos - b_s))`` with a learnable offset ``b_s``. Reads are
   sparse (owner, 28 September): only the top ``keep_s`` candidates by score enter
   the read; the others have gate exactly 0, which removes them exactly. Gates only
   scale mass (MLP-matrix gate semantics), so the task loss trains keys and heads.
3. **Superposition operator** S_s (``SuperpositionOperator``, no locality kernel,
   conditioned on the query key) combines the read items of space s into one read
   of ``m_s`` positions (the gate-weighted mean item length) and returns the
   space's total gate mass.

The **recombiner** R (the stack's) takes each space read with its mass as gate
(numerator and mass, invariant 5: space reads are never averaged without their
masses) and produces the span; its count is ``read_count`` (the gate-mass-weighted
mean length of the items read in decoder reps, capped by the per-read budget), and
the span is de-standardized with the stack's statistics (``mean + std * y``).

The **retrieval loss** (``retrieval_loss``) runs per space over the scored
candidates plus any target item the search missed (scored from its values, never
read): -log of the softmax mass on the slot's items.

Gold mode skips retrieval: the items of the slot's target records, gate 1 (the
information-matched control, never a training read). Items come from an
``ItemCache``, one leaf tensor per item for a step, so the gradients of all reads
of a step accumulate before one sparse live update per touched (KB, space).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from schnitz.kb.losses import retrieval_loss
from schnitz.kb.stack import SPACES, KeyHeads, Stack, SuperpositionOperator, read_count
from schnitz.kb_store import KnowledgeBase, search_kbs

# fine spaces retrieve few items, coarse spaces many (docs 3): about 8 writer reps'
# worth of positions per candidate set and space for equal-length items
DEFAULT_CANDIDATES = {'A': 8, 'B': 16, 'C': 32, 'D': 64}
# extreme sparsity (owner, 28 September): a read keeps a handful of items per space
DEFAULT_KEEP = {'A': 2, 'B': 2, 'C': 3, 'D': 4}

Ref = tuple[str, str]    # (dataset, item id) within one space


@dataclass
class ReadConfig:
    candidates: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CANDIDATES))
    keep: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_KEEP))
    hidden: int = 1024            # the decoder's hidden width (query input)
    span_width: int = 1024        # the decoder's input-embedding width (span output)
    target_norm: float = 1.0      # provenance only; R's output is de-standardized
    state: int = 512
    op_hidden: int = 256
    layers: int = 3
    key_hidden: int = 512
    gate_offset: float = 0.5      # initial b_s (cosine units)
    max_reps: int = 16            # per-read span budget
    checkpointing: bool = True


class ItemCache:
    """Item values of one step as leaf tensors (one per item), so gradients from all
    reads of a step accumulate; ``apply`` makes one sparse Adam update per touched
    (KB, space). ``train=False`` reads without gradients and never writes."""

    def __init__(self, device: torch.device | str = 'cpu', train: bool = True):
        self.device, self.train = torch.device(device), train
        self.values: dict[tuple[str, str, str], Tensor] = {}
        self.times: dict[tuple[str, str, str], int] = {}
        self.kbs: dict[str, KnowledgeBase] = {}

    def get(self, kb: KnowledgeBase, space: str, ids: Sequence[str]) -> list[tuple[Tensor, int]]:
        self.kbs[kb.dataset] = kb
        missing = [i for i in dict.fromkeys(ids) if (kb.dataset, space, i) not in self.values]
        if missing:
            live = kb.is_live(space) and kb.writable
            for item in kb.read(space, missing, live=live):
                ref = (kb.dataset, space, item.id)
                values = item.values.float().to(self.device)
                self.values[ref] = values.requires_grad_(self.train)
                self.times[ref] = item.time
        return [(self.values[(kb.dataset, space, i)], self.times[(kb.dataset, space, i)])
                for i in ids]

    def touched(self) -> dict[tuple[str, str], list[str]]:
        out: dict[tuple[str, str], list[str]] = {}
        for (dataset, space, item_id), value in self.values.items():
            if value.grad is not None:
                out.setdefault((dataset, space), []).append(item_id)
        return out

    @torch.no_grad()
    def apply(self, item_lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
              weight_decay: float = 0.0) -> dict[str, int]:
        """One Adam step on every item that received a gradient: one ``live_step`` per
        touched (KB, space)."""
        if not self.train:
            raise ValueError('an evaluation cache does not update items')
        count = 0
        if item_lr > 0:
            for (dataset, space), ids in self.touched().items():
                grads = torch.cat([self.values[(dataset, space, i)].grad for i in ids])
                self.kbs[dataset].live_step(space, ids, grads, lr=item_lr, betas=betas,
                                            eps=eps, weight_decay=weight_decay)
                count += len(ids)
        return {'items': count}


@dataclass
class SpaceRead:
    refs: list[Ref]               # the items read (nonzero gates)
    gates: Tensor                 # detached, per item read
    mass: float
    positions: int
    recall: float | None = None       # share of target items among the scored candidates
    recall_read: float | None = None  # ... among the items read
    scored: list[Ref] = field(default_factory=list)
    scored_gates: Tensor | None = None   # differentiable gates of every scored candidate


@dataclass
class Read:
    span: Tensor                  # (n, span_width); n = 0 for an empty read
    spaces: dict[str, SpaceRead]
    aux: Tensor | None = None     # retrieval loss (targets present, retrieved mode)
    n: int = 0
    recall_at: dict = field(default_factory=dict)


def current_ids(kb: KnowledgeBase, space: str) -> list[str]:
    """Ids of the current items of ``space`` at the KB's cursor, in row order."""
    visible = kb._visible(kb._map(space, 'rows.i64'), kb.cursor)
    return [kb._row_ids[space][r] for r in np.flatnonzero(visible).tolist()]


def source_index(kb: KnowledgeBase, space: str) -> dict[str, list[str]]:
    """Current item ids of ``space`` per source record id (from item provenance)."""
    out: dict[str, list[str]] = {}
    visible = kb._visible(kb._map(space, 'rows.i64'), kb.cursor)
    for row, item_id in enumerate(kb._row_ids[space]):
        if visible[row]:
            for source in kb._meta(space, row)['sources']:
                out.setdefault(source, []).append(item_id)
    return out


def _fetch(cache: ItemCache, by_dataset: Mapping[str, KnowledgeBase], space: str,
           refs: Sequence[Ref]) -> list[tuple[Tensor, int]]:
    """``cache.get`` for refs of several KBs, one store read per KB, in ``refs`` order."""
    out: dict[Ref, tuple[Tensor, int]] = {}
    for dataset in dict.fromkeys(d for d, _ in refs):
        ids = [i for d, i in refs if d == dataset]
        out.update({(dataset, i): got for i, got in
                    zip(ids, cache.get(by_dataset[dataset], space, ids))})
    return [out[r] for r in refs]


class L1Reader(nn.Module):
    """Query and item-key heads, gate offsets, per-space S_s and the stack's R."""

    def __init__(self, config: ReadConfig, stack: Stack | None = None):
        super().__init__()
        self.config = c = config
        self.spaces = list(SPACES)
        self.keys = KeyHeads(c.hidden, c.key_hidden)
        self.gate_offset = nn.ParameterDict({s: nn.Parameter(torch.tensor(c.gate_offset))
                                             for s in self.spaces})
        self.operators = nn.ModuleDict({
            s: SuperpositionOperator(s, c.state, c.op_hidden, c.layers, c.checkpointing)
            for s in self.spaces})
        self.stack = stack if stack is not None else Stack(
            c.target_norm, c.state, c.op_hidden, c.layers, c.checkpointing, width=c.span_width)
        for param in self.stack.codecs.parameters():     # R is used; the codecs are not
            param.requires_grad_(False)

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def gates(self, space: str, scores: Tensor) -> Tensor:
        scale = self.keys.log_scale[space].exp()
        return torch.sigmoid(scores - scale * self.gate_offset[space])

    def read(self, state: Tensor, kbs: Sequence[KnowledgeBase], allowed: Sequence[str],
             query_time: int, cache: ItemCache, *,
             targets: Mapping[str, Sequence[Ref]] | None = None, gold: bool = False) -> Read:
        """One read for the query-layer ``state`` (hidden,) at a call. ``targets``
        names the slot's items per space (retrieval loss and recall; in ``gold`` mode
        they are the read). Every KB must be authorized."""
        c = self.config
        allowed = set(allowed)
        denied = [kb.dataset for kb in kbs if kb.dataset not in allowed]
        if denied:
            raise PermissionError(f'not authorized to read {denied}')
        by_dataset = {kb.dataset: kb for kb in kbs}
        reads, masses, info, aux, recall_at = {}, {}, {}, [], {}
        read_positions, read_spaces, read_gates = [], [], []
        for s in self.spaces:
            q = self.keys.query_key(s, state)
            wanted = list(dict.fromkeys((targets or {}).get(s, ())))
            for dataset, _ in wanted:
                if dataset not in allowed or dataset not in by_dataset:
                    raise PermissionError(f'target item of {dataset!r} is not readable here')
            if gold:
                refs = wanted
            else:
                live = all(kb.writable and kb.is_live(s) for kb in kbs)
                hits = search_kbs(kbs, allowed, s, q.detach()[None], c.candidates[s],
                                  query_time=query_time, live=live)
                refs = list(zip(hits.datasets[0], hits.ids[0]))
            got = [(ref, values) for ref, (values, time)
                   in zip(refs, _fetch(cache, by_dataset, s, refs))
                   if time <= query_time]            # causal even for gold items
            none = None if not wanted else 0.0
            if not got:
                info[s] = SpaceRead([], torch.zeros(0), 0.0, 0, none, none)
                continue
            refs = [r for r, _ in got]
            scored, scored_gates = list(refs), None
            if gold:
                gates = torch.ones(len(got), device=q.device)
            else:
                keys = torch.stack([self.keys.item_key(s, v.to(q.device)) for _, v in got])
                scores = self.keys.scores(s, q[None], keys)[0]
                if wanted:
                    loss, at = self._retrieval_loss(s, q, refs, scores, wanted, by_dataset,
                                                    cache, query_time)
                    if loss is not None:
                        aux.append(loss)
                        recall_at.update({f'{k}_{s}': v for k, v in at.items()})
                scored_gates = self.gates(s, scores)
                keep = min(c.keep.get(s, len(got)), len(got))
                order = torch.topk(scores.detach(), keep).indices.sort().values.tolist()
                got = [got[i] for i in order]
                refs = [refs[i] for i in order]
                gates = scored_gates[order]
            g = gates.detach().float()
            size = torch.tensor([float(v.shape[0]) for _, v in got], device=g.device)
            count = max(1, round(float((g * size).sum() / g.sum().clamp_min(1e-12))))
            out, mass = self.operators[s]([(v.to(q.device), gt, None) for (_, v), gt
                                           in zip(got, gates)], q, count)
            reads[s], masses[s] = out, mass
            read_positions += [int(v.shape[0]) for _, v in got]
            read_spaces += [s] * len(got)
            read_gates.append(g)
            recall = recall_read = None
            if wanted:
                recall = sum(r in set(scored) for r in wanted) / len(wanted)
                recall_read = sum(r in set(refs) for r in wanted) / len(wanted)
            info[s] = SpaceRead(refs, g.cpu(), float(mass.detach()), count, recall, recall_read,
                                scored, scored_gates)
        aux_loss = torch.stack(aux).mean() if aux else None
        live = [s for s in self.spaces if s in reads and float(masses[s].detach()) > 0]
        if not live:
            return Read(torch.zeros(0, c.span_width, device=state.device), info, aux_loss, 0,
                        recall_at)
        n = read_count(read_positions, read_spaces, torch.cat(read_gates), budget=c.max_reps)
        y, _ = self.stack.recombiner([(s, reads[s], masses[s]) for s in live], n)
        span = self.stack.mean + self.stack.std * y
        return Read(span, info, aux_loss, n, recall_at)

    def _retrieval_loss(self, space, q, refs, scores, wanted, by_dataset, cache, query_time):
        extra = [r for r in wanted if r not in set(refs)]
        fetched = _fetch(cache, by_dataset, space, extra) if extra else []
        missed = [self.keys.item_key(space, v.to(q.device)) for v, time in fetched
                  if time <= query_time]
        if missed:
            scores = torch.cat([scores, self.keys.scores(space, q[None], torch.stack(missed))[0]])
        positive = torch.tensor([r in set(wanted) for r in refs] + [True] * len(missed),
                                device=q.device)
        if not positive.any():
            return None, {}
        return retrieval_loss(scores[None], positive[None])

    @torch.no_grad()
    def rekey(self, kb: KnowledgeBase, batch: int = 4096) -> int:
        """Refresh the stored live keys of every current item of ``kb`` from its live
        values with the current item-key heads (one ``set_live_keys`` per space)."""
        device = next(self.parameters()).device
        count = 0
        for s in self.spaces:
            if not (kb.writable and kb.is_live(s)):
                continue
            ids = current_ids(kb, s)
            keys = []
            for start in range(0, len(ids), batch):
                items = kb.read(s, ids[start:start + batch], live=True)
                keys += [self.keys.item_key(s, it.values.float().to(device)).cpu() for it in items]
            if ids:
                kb.set_live_keys(s, ids, torch.stack(keys))
                count += len(ids)
        return count


def splice(embeds: Tensor, mem_positions: Sequence[int], spans: Sequence[Tensor]
           ) -> tuple[Tensor, Tensor]:
    """Insert ``spans[i]`` right after the ``<|mem|>`` token at ``mem_positions[i]``
    (so it sits between ``<|mem|>`` and ``<|/mem|>``). Returns the new sequence and,
    for every original position, its index in the new sequence."""
    if len(mem_positions) != len(spans):
        raise ValueError('one span per memory slot')
    order = sorted(range(len(spans)), key=lambda i: mem_positions[i])
    parts, index, cursor, offset = [], [], 0, 0
    for i in order:
        p = mem_positions[i]
        if not cursor <= p < embeds.shape[0]:
            raise ValueError('memory positions must be distinct and inside the sequence')
        parts.append(embeds[cursor:p + 1])
        index.append(torch.arange(cursor, p + 1) + offset)
        span = spans[i].to(embeds.device, embeds.dtype)
        parts.append(span)
        offset += span.shape[0]
        cursor = p + 1
    parts.append(embeds[cursor:])
    index.append(torch.arange(cursor, embeds.shape[0]) + offset)
    return torch.cat(parts), torch.cat(index)
