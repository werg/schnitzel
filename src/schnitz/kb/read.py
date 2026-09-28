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
candidates plus any target item the search missed and the caller's in-batch
negatives (each scored from its values, never read; negatives must belong to the
read's authorized KBs): -log of the softmax mass on the slot's items. ``exclude``
keeps named items (an episode's own writes) out of a read. In L1b a ``Producer``
supplies the values of the items a read keeps, recomputed from their sources with
gradients into the producers; selection stays on the current values' scores.

Live items of a resident KB are gathered once per (KB, space) and call, onto the
reader's device (``KnowledgeBase.read(..., device=...)``).

Gold mode skips retrieval: the items of the slot's target records, gate 1 (the
information-matched control, never a training read). Items come from an
``ItemCache``, one leaf tensor per item for a step, so the gradients of all reads
of a step accumulate before one sparse live update per touched (KB, space).
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from schnitz.kb.losses import retrieval_loss
from schnitz.kb.stack import KEY_WIDTH, SPACES, KeyHeads, Stack, SuperpositionOperator, read_count
from schnitz.mlp_matrix import MLPMatrix
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
    # 'r': R reads the retrieved items of every space directly, conditioned on the query
    # keys (owner, 28 September; the L1 default); 's_s': per-space S_s, then R (the path of
    # the earlier smokes; the default here so stored reader configs keep their meaning)
    read_combine: str = 's_s'
    # keys as free parameters (owner, 28 September): scores use each item's live key (a
    # leaf moved by the retrieval and gate gradients, ``KeyOptimizer``), not the item-key
    # heads applied to its values; the search index is the live keys themselves
    learned_keys: bool = False


class ItemCache:
    """Item values of one step as leaf tensors (one per item), so gradients from all
    reads of a step accumulate; ``apply`` makes one sparse Adam update per touched
    (KB, space). ``train=False`` reads without gradients and never writes. ``keys``
    gives the items' live keys as leaves too (``learned_keys``), updated by ``apply``
    through a ``KeyOptimizer``."""

    def __init__(self, device: torch.device | str = 'cpu', train: bool = True):
        self.device, self.train = torch.device(device), train
        self.values: dict[tuple[str, str, str], Tensor] = {}
        self.key_leaves: dict[tuple[str, str, str], Tensor] = {}
        self.times: dict[tuple[str, str, str], int] = {}
        self.masses: dict[tuple[str, str, str], float] = {}     # stored item mass
        self.kbs: dict[str, KnowledgeBase] = {}

    def _load(self, kb: KnowledgeBase, space: str, ids: Sequence[str]) -> None:
        self.kbs[kb.dataset] = kb
        missing = [i for i in dict.fromkeys(ids) if (kb.dataset, space, i) not in self.values]
        if missing:
            live = kb.is_live(space) and kb.writable
            # resident live state: one gather straight onto the reader's device
            device = self.device if live and kb.live_device is not None else None
            for item in kb.read(space, missing, live=live, device=device):
                ref = (kb.dataset, space, item.id)
                values = item.values.float().to(self.device)
                key = item.key.float().to(self.device)
                if device is not None:     # a view of the gathered block: own storage
                    values, key = values.clone(), key.clone()
                self.values[ref] = values.requires_grad_(self.train)
                self.key_leaves[ref] = key.requires_grad_(self.train)
                self.times[ref] = item.time
                self.masses[ref] = item.mass

    def get(self, kb: KnowledgeBase, space: str, ids: Sequence[str]) -> list[tuple[Tensor, int]]:
        self._load(kb, space, ids)
        return [(self.values[(kb.dataset, space, i)], self.times[(kb.dataset, space, i)])
                for i in ids]

    def keys(self, kb: KnowledgeBase, space: str, ids: Sequence[str]) -> list[Tensor]:
        """The items' live (or stored) keys as leaves (``learned_keys``)."""
        self._load(kb, space, ids)
        return [self.key_leaves[(kb.dataset, space, i)] for i in ids]

    def mass(self, dataset: str, space: str, item_id: str) -> float:
        """The stored mass of an item fetched this step."""
        return self.masses[(dataset, space, item_id)]

    def touched(self) -> dict[tuple[str, str], list[str]]:
        out: dict[tuple[str, str], list[str]] = {}
        for (dataset, space, item_id), value in self.values.items():
            if value.grad is not None:
                out.setdefault((dataset, space), []).append(item_id)
        return out

    @torch.no_grad()
    def apply(self, item_lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
              weight_decay: float = 0.0, key_optimizer: KeyOptimizer | None = None
              ) -> dict[str, int]:
        """One Adam step on every item that received a gradient: one ``live_step`` per
        touched (KB, space); with ``key_optimizer`` also one step on every key that
        received a gradient (``KeyOptimizer.step``)."""
        if not self.train:
            raise ValueError('an evaluation cache does not update items')
        count, keys = 0, 0
        if item_lr > 0:
            for (dataset, space), ids in self.touched().items():
                grads = torch.cat([self.values[(dataset, space, i)].grad for i in ids])
                self.kbs[dataset].live_step(space, ids, grads, lr=item_lr, betas=betas,
                                            eps=eps, weight_decay=weight_decay)
                count += len(ids)
        if key_optimizer is not None:
            by: dict[tuple[str, str], list[str]] = {}
            for (dataset, space, item_id), key in self.key_leaves.items():
                if key.grad is not None:
                    by.setdefault((dataset, space), []).append(item_id)
            for (dataset, space), ids in by.items():
                refs = [(dataset, space, i) for i in ids]
                keys += key_optimizer.step(self.kbs[dataset], space, ids,
                                           [self.key_leaves[r].detach() for r in refs],
                                           [self.key_leaves[r].grad for r in refs])
        return {'items': count, 'keys': keys} if key_optimizer is not None else {'items': count}


class KeyOptimizer:
    """Per-key Adam for live keys as free parameters (``ReadConfig.learned_keys``): each
    key its own step count and moments; after the step the key is renormalized to unit
    length (scores are cosines) and written with ``KnowledgeBase.set_live_keys``. The
    moments are the trainer's state (``state_dict``), saved with its checkpoint. With
    ``normalize=False`` the live keys hold unconstrained vectors (the leaves' key
    corrections of ``superpose.head_leaf_key``)."""

    def __init__(self, lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
                 normalize: bool = True):
        self.lr, self.betas, self.eps, self.normalize = lr, betas, eps, normalize
        self.state: dict[str, tuple[Tensor, Tensor, int]] = {}

    @torch.no_grad()
    def step(self, kb: KnowledgeBase, space: str, ids: Sequence[str], keys: Sequence[Tensor],
             grads: Sequence[Tensor]) -> int:
        if self.lr <= 0 or not ids:
            return 0
        b1, b2 = self.betas
        # all keys of the call at once (row-wise the same arithmetic as one key at a time)
        key = torch.stack([k for k in keys])
        g = torch.stack([gr for gr in grads]).to(key.device, key.dtype)
        names = [f'{kb.dataset}/{space}/{i}' for i in ids]
        zero = torch.zeros(key.shape[1], dtype=key.dtype)
        old = [self.state.get(n, (zero, zero, 0)) for n in names]
        m = torch.stack([o[0] for o in old]).to(key.device, key.dtype)
        v = torch.stack([o[1] for o in old]).to(key.device, key.dtype)
        t = torch.tensor([o[2] + 1 for o in old], dtype=torch.float64)
        c1 = (1 - b1 ** t).to(key.device, key.dtype)[:, None]
        c2 = (1 - b2 ** t).to(key.device, key.dtype)[:, None]
        m = m * b1 + (1 - b1) * g
        v = v * b2 + (1 - b2) * g * g
        update = self.lr * (m / c1) / ((v / c2).sqrt() + self.eps)
        new = key - update
        if self.normalize:
            new = nn.functional.normalize(new, dim=-1)
        m_cpu, v_cpu = m.cpu().unbind(0), v.cpu().unbind(0)
        for name, mi, vi, ti in zip(names, m_cpu, v_cpu, t.tolist()):
            self.state[name] = (mi.clone(), vi.clone(), int(ti))
        kb.set_live_keys(space, list(ids), new.float().cpu())
        return len(ids)

    def state_dict(self) -> dict:
        return {'lr': self.lr, 'state': {k: (m, v, t) for k, (m, v, t) in self.state.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        self.state = dict(state.get('state', {}))


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
    recomputed: int = 0           # L1b: items read with the producers' recomputed values
    scales: list[float] = field(default_factory=list)   # per item read: stored mass x weight
    values: list[Tensor] = field(default_factory=list)  # the values read (with their graph)


@dataclass
class Read:
    span: Tensor                  # (n, span_width); n = 0 for an empty read
    spaces: dict[str, SpaceRead]
    aux: Tensor | None = None     # retrieval loss (targets present, retrieved mode)
    n: int = 0
    recall_at: dict = field(default_factory=dict)
    state: Tensor | None = None   # the query-layer state the read was made for (detached)


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


def producer_index(kb: KnowledgeBase, space: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """(producer, sources) of every current item of ``space``, by item id, in row order."""
    visible = kb._visible(kb._map(space, 'rows.i64'), kb.cursor)
    out = {}
    for row in np.flatnonzero(visible).tolist():
        meta = kb._meta(space, row)
        out[kb._row_ids[space][row]] = (meta['producer'], tuple(meta['sources']))
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


class Producer(Protocol):
    """L1b: the values of read items recomputed from their stored sources, with
    gradients into the producers (``schnitz.kb.producer.Producers``); None for an
    item without a recomputable source."""

    def values(self, space: str, refs: Sequence[Ref]) -> list[Tensor | None]: ...


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
        if c.read_combine not in ('r', 's_s'):
            raise ValueError(f'read_combine is r or s_s, not {c.read_combine!r}')
        # 'r': a query-conditioned copy of the stack's R (condition columns start at zero,
        # so it begins as the stack's R); the stack's own R stays for decoding (L2, K1)
        self.read_r = conditioned(self.stack.recombiner, sum(KEY_WIDTH[s] for s in self.spaces)) \
            if c.read_combine == 'r' else None

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def recombiner(self) -> nn.Module:
        """The R the reads use (``read_r`` in 'r' mode, the stack's R in 's_s' mode)."""
        return self.read_r if self.read_r is not None else self.stack.recombiner

    def gates(self, space: str, scores: Tensor) -> Tensor:
        scale = self.keys.log_scale[space].exp()
        return torch.sigmoid(scores - scale * self.gate_offset[space])

    def read(self, state: Tensor, kbs: Sequence[KnowledgeBase], allowed: Sequence[str],
             query_time: int, cache: ItemCache, *,
             targets: Mapping[str, Sequence[Ref]] | None = None, gold: bool = False,
             negatives: Mapping[str, Sequence[Ref]] | None = None,
             exclude: Mapping[str, Collection[Ref]] | None = None,
             producer: Producer | None = None,
             weights: Mapping[str, float] | None = None,
             alternatives: Mapping[str, Sequence[Ref]] | None = None,
             neutral: Mapping[str, Sequence[Ref]] | None = None) -> Read:
        """One read for the query-layer ``state`` (hidden,) at a call. ``targets``
        names the slot's items per space (retrieval loss and recall; in ``gold`` mode
        they are the read). ``alternatives`` (per space): other items that alone hold
        the slot's content (redundant copies); they are extra positives of the
        retrieval loss and never its negatives, recall then counts a read as a hit
        when any positive is among the items (``any``), and the gold read stays
        ``targets``. ``neutral`` (per space): items that are neither positives nor
        negatives of the retrieval loss (near-duplicates of the positives that do not
        hold the slot's content, e.g. overlapping windows of the same document): they
        leave the loss's list (scored candidates and ``negatives``) but may still be
        retrieved and read; a positive named neutral stays positive, and recall is
        unchanged. Every KB must be authorized, and so must every target, alternative,
        negative and neutral item.

        ``negatives`` (per space): extra items scored as negatives in the retrieval
        loss (in-batch negatives; the caller passes only items of the read's own KBs).
        ``exclude`` (per space): items this read may not retrieve (an episode's own
        writes). ``producer`` (L1b): the values of the items read (not of the other
        scored candidates) come from ``producer.values`` - the producers' recomputation
        from the items' stored sources - and so do their gates; selection stays on the
        scores of the current (live) values.

        Each read item's gate is multiplied by its stored mass and by ``weights[id]``
        (item id -> multiplier, default 1; weight 0 removes an item exactly).

        A superposed cache (``schnitz.kb.superpose.SuperposedCache``) serves rows computed
        from the KB's items: the search runs over the rows, targets, negatives and
        exclusions name the KB's items and are mapped to the rows covering them (positives
        weighted by their share of the row's mass; a row covering a neutral item is
        neutral unless it covers a positive), and ``producer`` is the cache's."""
        c = self.config
        allowed = set(allowed)
        denied = [kb.dataset for kb in kbs if kb.dataset not in allowed]
        if denied:
            raise PermissionError(f'not authorized to read {denied}')
        by_dataset = {kb.dataset: kb for kb in kbs}
        for what, named in (('target', targets), ('alternative', alternatives),
                            ('negative', negatives), ('neutral', neutral)):
            for dataset, _ in (r for rs in (named or {}).values() for r in rs):
                if dataset not in allowed or dataset not in by_dataset:
                    raise PermissionError(f'{what} item of {dataset!r} is not readable here')
        superposed = getattr(cache, 'superposed', False)
        if superposed:
            producer = None                  # the cache recomputes rows from their leaves
        reads, masses, info, aux, recall_at = {}, {}, {}, [], {}
        read_positions, read_spaces, read_gates, entries, queries = [], [], [], [], {}
        for s in self.spaces:
            q = self.keys.query_key(s, state)
            queries[s] = q
            wanted = list(dict.fromkeys((targets or {}).get(s, ())))
            copies = [r for r in dict.fromkeys((alternatives or {}).get(s, ()))
                      if r not in set(wanted)]
            banned = set((exclude or {}).get(s, ()))
            # positives: the slot's items and their copies (weight 1), or the rows
            # covering them (share)
            positive = cache.positives(s, wanted + copies) if superposed \
                else dict.fromkeys(wanted + copies, 1.0)
            if gold:
                refs = cache.gold(s, [r for r in wanted if r not in banned], banned) \
                    if superposed else [r for r in wanted if r not in banned]
            elif superposed:
                refs = cache.search(kbs, allowed, s, q.detach(), c.candidates[s], query_time,
                                    banned)
            else:
                live = all(kb.writable and kb.is_live(s) for kb in kbs)
                options = {'exclude': {i for _, i in banned}} if banned else {}
                hits = search_kbs(kbs, allowed, s, q.detach()[None], c.candidates[s],
                                  query_time=query_time, live=live, **options)
                refs = [r for r in zip(hits.datasets[0], hits.ids[0]) if r not in banned]
            got = [(ref, values) for ref, (values, time)
                   in zip(refs, _fetch(cache, by_dataset, s, refs))
                   if time <= query_time]            # causal even for gold items
            none = None if not wanted else 0.0
            if not got:
                info[s] = SpaceRead([], torch.zeros(0), 0.0, 0, none, none)
                continue
            refs = [r for r, _ in got]
            scored, scored_gates, recomputed = list(refs), None, 0
            if gold:
                gates = torch.ones(len(got), device=q.device)
            else:
                keys = self._item_keys(s, cache, by_dataset, refs, [v for _, v in got], q.device)
                scores = self.keys.scores(s, q[None], keys)[0]
                if positive:
                    others = [r for r in (negatives or {}).get(s, ()) if r not in banned]
                    if superposed and others:
                        others = cache.gold(s, others, banned)
                    # neutral items (rows covering them) leave the list; positives stay
                    idle = list(dict.fromkeys((neutral or {}).get(s, ())))
                    if superposed and idle:
                        idle = cache.gold(s, idle, banned)
                    idle = {r for r in idle if r not in positive}
                    loss, at = self._retrieval_loss(s, q, refs, scores, positive, by_dataset,
                                                    cache, query_time, others, idle)
                    if loss is not None:
                        aux.append(loss)
                        recall_at.update({f'{k}_{s}': v for k, v in at.items()})
                scored_gates = self.gates(s, scores)
                keep = min(c.keep.get(s, len(got)), len(got))
                order = torch.topk(scores.detach(), keep).indices.sort().values.tolist()
                got = [got[i] for i in order]
                refs = [refs[i] for i in order]
                gates = scored_gates[order]
                if producer is not None:
                    # L1b: the read items' values are the producers' recomputation and
                    # their gates follow the recomputed keys; an item without a
                    # recomputable source keeps its current value
                    fresh = producer.values(s, refs)
                    recomputed = sum(v is not None for v in fresh)
                    if recomputed and c.learned_keys:      # keys are parameters, not values'
                        got = [(ref, old if v is None else v)
                               for (ref, old), v in zip(got, fresh)]
                    elif recomputed:
                        got = [(ref, old if v is None else v)
                               for (ref, old), v in zip(got, fresh)]
                        keys = torch.stack([self.keys.item_key(s, v.to(q.device))
                                            for _, v in got])
                        gates = self.gates(s, self.keys.scores(s, q[None], keys)[0])
            # gate x stored mass (invariant 5: a rewrite output carries its inputs' mass)
            # x the caller's per-item weight (B9's receding gold weight); gates only
            # scale mass, so this is each item's exact share, and read_count sees it
            scale = [cache.mass(d, s, i) * (weights or {}).get(i, 1.0) for d, i in refs]
            values = [v for _, v in got]
            if c.read_combine == 'r':
                g = self._scaled(gates, scale)
                entries += [(s, v.to(q.device), gt) for v, gt in zip(values, g)]
                mass, count, g = g.sum(), sum(int(v.shape[0]) for v in values), g.detach().float()
            else:
                out, mass, count, g = self._combine(s, q, values, gates, scale)
                reads[s] = out
            masses[s] = mass
            read_positions += [int(v.shape[0]) for v in values]
            read_spaces += [s] * len(values)
            read_gates.append(g)
            recall = recall_read = None
            if positive and copies:          # redundant copies: any positive is a hit
                recall = float(any(r in set(scored) for r in positive))
                recall_read = float(any(r in set(refs) for r in positive))
            elif positive:
                total = sum(positive.values())
                recall = sum(w for r, w in positive.items() if r in set(scored)) / total
                recall_read = sum(w for r, w in positive.items() if r in set(refs)) / total
            elif wanted:
                recall = recall_read = 0.0
            info[s] = SpaceRead(refs, g.cpu(), float(mass.detach()), count, recall, recall_read,
                                scored, scored_gates, recomputed, list(scale), list(values))
        aux_loss = torch.stack(aux).mean() if aux else None
        if c.read_combine == 'r':
            span, n = self._span_r(entries, queries, read_positions, read_spaces, read_gates,
                                   state)
        else:
            span, n = self._span(reads, masses, read_positions, read_spaces, read_gates, state)
        return Read(span, info, aux_loss, n, recall_at, state.detach())

    def _item_keys(self, space: str, cache, by_dataset, refs: Sequence[Ref],
                   values: Sequence[Tensor], device) -> Tensor:
        """Unit keys of the items ``refs``: their live keys as leaves (``learned_keys``),
        else the item-key heads applied to their values."""
        if self.config.learned_keys:
            out: dict[Ref, Tensor] = {}
            for dataset in dict.fromkeys(d for d, _ in refs):
                ids = [i for d, i in refs if d == dataset]
                out.update({(dataset, i): k for i, k in
                            zip(ids, cache.keys(by_dataset[dataset], space, ids))})
            return nn.functional.normalize(torch.stack([out[r].to(device) for r in refs]), dim=-1)
        return torch.stack([self.keys.item_key(space, v.to(device)) for v in values])

    @staticmethod
    def _scaled(gates: Tensor, scale: Sequence[float]) -> Tensor:
        if any(x != 1.0 for x in scale):
            return gates * torch.tensor(list(scale), device=gates.device, dtype=gates.dtype)
        return gates

    def _combine(self, space: str, q: Tensor, values: Sequence[Tensor], gates: Tensor,
                 scale: Sequence[float]) -> tuple[Tensor, Tensor, int, Tensor]:
        """S_s over the items read in ``space`` at ``gates`` x ``scale`` (stored mass x
        caller weight): the space read, its mass, its count (the gate-weighted mean item
        length) and the detached gates."""
        gates = self._scaled(gates, scale)
        g = gates.detach().float()
        size = torch.tensor([float(v.shape[0]) for v in values], device=g.device)
        count = max(1, round(float((g * size).sum() / g.sum().clamp_min(1e-12))))
        out, mass = self.operators[space]([(v.to(q.device), gt, None) for v, gt
                                           in zip(values, gates)], q, count)
        return out, mass, count, g

    def _span_r(self, entries, queries: Mapping[str, Tensor], read_positions, read_spaces,
                read_gates, state: Tensor) -> tuple[Tensor, int]:
        """'r' mode: R over every space's read items at their gates (x stored mass x
        weight), conditioned on the query keys of all spaces, de-standardized."""
        c = self.config
        if not entries or float(sum(float(g.detach()) for _, _, g in entries)) <= 0:
            return torch.zeros(0, c.span_width, device=state.device), 0
        n = read_count(read_positions, read_spaces, torch.cat(read_gates), budget=c.max_reps)
        cond = torch.cat([queries[s] for s in self.spaces])[None]
        y, _ = self.read_r(entries, n, cond=cond)
        return self.stack.mean + self.stack.std * y, n

    def _span(self, reads, masses, read_positions, read_spaces, read_gates,
              state: Tensor) -> tuple[Tensor, int]:
        """R over the space reads with mass as gate, de-standardized; (span, count)."""
        c = self.config
        live = [s for s in self.spaces if s in reads and float(masses[s].detach()) > 0]
        if not live:
            return torch.zeros(0, c.span_width, device=state.device), 0
        n = read_count(read_positions, read_spaces, torch.cat(read_gates), budget=c.max_reps)
        y, _ = self.stack.recombiner([(s, reads[s], masses[s]) for s in live], n)
        return self.stack.mean + self.stack.std * y, n

    def reread(self, state: Tensor, values: Mapping[str, Sequence[Tensor]],
               scales: Mapping[str, Sequence[float]],
               keys: Mapping[str, Sequence[Tensor]] | None = None) -> Tensor:
        """The span of a read with its selection fixed: ``values[s]`` are the items read
        in space s (in read order) and ``scales[s]`` their stored mass x caller weight;
        gates from the items' keys against the query heads' keys of ``state`` (as a read
        whose items are recomputed, L1b), then S_s and R. Gradients reach the values
        (B9's chain through an earlier write's reads)."""
        reads, masses, read_positions, read_spaces, read_gates = {}, {}, [], [], []
        entries, queries = [], {}
        for s in self.spaces:
            q = self.keys.query_key(s, state)
            queries[s] = q
            items = list(values.get(s, ()))
            if not items:
                continue
            if keys is not None and s in keys:     # learned keys (``learned_keys``)
                k = nn.functional.normalize(torch.stack([x.to(q.device) for x in keys[s]]),
                                            dim=-1)
            else:
                k = torch.stack([self.keys.item_key(s, v.to(q.device)) for v in items])
            gates = self.gates(s, self.keys.scores(s, q[None], k)[0])
            if self.config.read_combine == 'r':
                g = self._scaled(gates, scales[s])
                entries += [(s, v.to(q.device), gt) for v, gt in zip(items, g)]
                g = g.detach().float()
            else:
                out, mass, _, g = self._combine(s, q, items, gates, scales[s])
                reads[s], masses[s] = out, mass
            read_positions += [int(v.shape[0]) for v in items]
            read_spaces += [s] * len(items)
            read_gates.append(g)
        if self.config.read_combine == 'r':
            return self._span_r(entries, queries, read_positions, read_spaces, read_gates,
                                state)[0]
        return self._span(reads, masses, read_positions, read_spaces, read_gates, state)[0]

    def fixed_read(self, state: Tensor, donor: Read) -> Read:
        """The read of ``donor``'s items at ``donor``'s gates, conditioned on the query of
        ``state``: the information-matched content control of L1's contrast term
        (``--contrast-weight``). Same items, same gates (stored mass x weight included),
        so the same per-space counts and the same span length; the item payloads and the
        gates enter detached, so only R (and S_s), the query heads through R's condition,
        and whatever produced ``state`` receive gradients."""
        reads, masses, read_positions, read_spaces, read_gates = {}, {}, [], [], []
        entries, queries, info = [], {}, {}
        for s in self.spaces:
            q = self.keys.query_key(s, state)
            queries[s] = q
            got = donor.spaces.get(s)
            if got is None or not got.refs:
                continue
            if len(got.values) != len(got.refs):
                raise ValueError('the donor read keeps no item values')
            items = [v.detach().to(q.device) for v in got.values]
            gates = got.gates.detach().to(q.device, torch.float)
            if self.config.read_combine == 'r':
                entries += [(s, v, g) for v, g in zip(items, gates)]
                mass, count = gates.sum(), sum(int(v.shape[0]) for v in items)
            else:
                out, mass, count, _ = self._combine(s, q, items, gates, [1.0] * len(items))
                reads[s] = out
            masses[s] = mass
            read_positions += [int(v.shape[0]) for v in items]
            read_spaces += [s] * len(items)
            read_gates.append(gates)
            info[s] = SpaceRead(list(got.refs), gates.cpu(), float(mass), count,
                                scales=list(got.scales), values=items)
        if self.config.read_combine == 'r':
            span, n = self._span_r(entries, queries, read_positions, read_spaces, read_gates,
                                   state)
        else:
            span, n = self._span(reads, masses, read_positions, read_spaces, read_gates, state)
        return Read(span, info, None, n, {}, state.detach())

    def _retrieval_loss(self, space, q, refs, scores, wanted: Mapping[Ref, float], by_dataset,
                        cache, query_time, negatives: Sequence[Ref] = (),
                        neutral: Collection[Ref] = frozenset()):
        """Over the scored candidates, plus the positives the search missed and
        ``negatives`` not already among them (in-batch negatives), each scored from its
        current values; items later than the query time are left out. ``wanted`` maps each
        positive to its weight (1 for the slot's own items; a covering row's share).
        ``neutral`` items (never positives) are neither: scored candidates among them
        leave the list and negatives among them are not added."""
        if neutral:
            keep = [k for k, r in enumerate(refs) if r not in neutral or r in wanted]
            if len(keep) < len(refs):
                refs = [refs[k] for k in keep]
                scores = scores[torch.tensor(keep, dtype=torch.long, device=scores.device)]
            negatives = [r for r in negatives if r not in neutral]
        seen = set(refs)
        extra = [r for r in wanted if r not in seen]
        extra += [r for r in dict.fromkeys(negatives) if r not in seen and r not in wanted]
        fetched = _fetch(cache, by_dataset, space, extra) if extra else []
        kept = [(r, v) for r, (v, time) in zip(extra, fetched) if time <= query_time]
        pairs = []
        if kept:
            keys = self._item_keys(space, cache, by_dataset, [r for r, _ in kept],
                                   [v for _, v in kept], q.device)
            pairs = [(wanted.get(r, 0.0), k) for (r, _), k in zip(kept, keys)]
        if pairs:
            more = self.keys.scores(space, q[None], torch.stack([k for _, k in pairs]))[0]
            scores = torch.cat([scores, more])
        positive = torch.tensor([wanted.get(r, 0.0) for r in refs] + [p for p, _ in pairs],
                                device=q.device, dtype=torch.float)
        if not bool((positive > 0).any()):
            return None, {}
        return retrieval_loss(scores[None], positive[None])

    @torch.no_grad()
    def rekey(self, kb: KnowledgeBase, batch: int = 4096) -> int:
        """Refresh the stored live keys of every current item of ``kb`` from its live
        values with the current item-key heads (one ``set_live_keys`` per space). With
        ``learned_keys`` the live keys are parameters and stay as they are (returns 0)."""
        if self.config.learned_keys:
            return 0
        device = next(self.parameters()).device
        count = 0
        for s in self.spaces:
            if not (kb.writable and kb.is_live(s)):
                continue
            ids = current_ids(kb, s)
            keys = []
            for start in range(0, len(ids), batch):
                items = kb.read(s, ids[start:start + batch], live=True,
                                device=device if kb.live_device is not None else None)
                keys.append(self.item_keys(s, [it.values for it in items]).cpu())
            if ids:
                kb.set_live_keys(s, ids, torch.cat(keys))
                count += len(ids)
        return count

    def item_keys(self, space: str, values: Sequence[Tensor]) -> Tensor:
        """``KeyHeads.item_key`` of many variable-length items in one pass: the head
        acts per position, so the items are concatenated and mean-pooled per item."""
        device = next(self.parameters()).device
        lengths = torch.tensor([v.shape[0] for v in values], device=device)
        out = self.keys.item[space](torch.cat([v.float().to(device) for v in values]))
        owner = torch.repeat_interleave(torch.arange(len(values), device=device), lengths)
        pooled = torch.zeros(len(values), out.shape[1], device=device, dtype=out.dtype)
        pooled.index_add_(0, owner, out)
        return nn.functional.normalize(pooled / lengths[:, None].to(out.dtype), dim=-1)


def conditioned(recombiner: MLPMatrix, cond: int) -> MLPMatrix:
    """A copy of ``recombiner`` that also takes a condition of width ``cond`` (the query
    keys): every weight is copied and the condition's columns and projections start at
    zero, so the copy computes exactly what the original does until they train."""
    layer = recombiner.layers[0]
    sources = {kind: norm.normalized_shape[0] for kind, norm in recombiner.input_norm.items()}
    out = MLPMatrix(sources, recombiner.head[1].out_features,
                    state=recombiner.init[2].out_features, hidden=layer.out.in_features,
                    layers=len(recombiner.layers), frequencies=recombiner.frequencies, cond=cond,
                    relative=layer.relative is not None, out_norm=recombiner.out_norm,
                    checkpoint_layers=recombiner.checkpoint_layers, extra=recombiner.extra_width)
    state = out.state_dict()
    with torch.no_grad():
        for name, value in recombiner.state_dict().items():
            if state[name].shape == value.shape:
                state[name] = value.clone()
            elif name == 'init.0.weight':
                grown = torch.zeros_like(state[name])
                grown[:, :value.shape[1]] = value
                state[name] = grown
        for name in state:
            if '.condition.' in name:
                state[name] = torch.zeros_like(state[name])
    out.load_state_dict(state)
    return out


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
