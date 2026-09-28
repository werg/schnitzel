"""L1 read path at a ``memory_search()`` call (docs/knowledge-base-stack.md, 3 and 5.1 step 6).

At a call the frozen decoder's hidden state after ``query_layer`` layers, at the
call's own position, is the query: one key head per space projects it to that
space's key width (queries are vectors, never text). Per space:

1. **Retrieval.** Exact top-k (``kb_store.search_kbs``) over the current (live, in the
   writer during L1) keys of the
   KBs the caller is authorized for (``allowed`` lists datasets; learned selection is
   never authorization, invariant 6), limited to items whose time is at or before
   the query time (invariant 2). The search itself is discrete and runs on detached
   queries.
2. **Gates.** Every candidate's gate comes from its query-key cosine,
   ``g = sigmoid(exp(a_s) (cos - b_s))`` (default) or a softmax over the candidates
   with the same logits; ``a_s``, ``b_s`` are learnable per space. Gates are computed
   with gradients from the live keys and the query, and only scale mass (MLP-matrix
   gate semantics), so the task loss trains keys and key heads.
   Reads are sparse (owner, 28 September): of the scored candidates only the top
   ``keep_s`` by gate logit enter the read (the others have gate exactly 0, which
   removes them exactly); all candidates are scored by the retrieval loss.
3. **Superposition operator** ``S_s`` (an ``MLPMatrix`` over the candidates of space s,
   no locality kernel since a neighbourhood is unordered, conditioned on the query)
   produces one per-space read of ``m_s`` positions, ``m_s`` the gate-weighted mean
   candidate length, and returns the space's total gate mass.

The **recombiner** R (an ``MLPMatrix`` over the space reads, the K1 recombiner's form)
takes each space read with its mass as gate (numerator and mass, invariant 5: space
reads are never averaged without their masses) and produces the span of ``n`` reps in
the decoder's input space. Choice of ``n`` (documented in the stack doc, WP5):

    L = sum_s mu_s (mbar_s / r_s) / sum_s mu_s     (the implied length of one record)
    M = mean_s mu_s, clamped to [1, max_items]     (with sigmoid gates, about the number
                                                    of relevant items found per space)
    n = clamp(ceil(L * M), min_reps, max_reps)

with ``mu_s`` a space's total gate mass and ``mbar_s`` its gate-weighted mean item
length (r_s positions per writer rep). A read that finds one record gets about that
record's span length, a read over several records proportionally more, capped. With
softmax gates every space has mass 1 and ``n`` is one record's length. ``n`` is
computed from detached values (a count is not differentiable).

Gold mode skips retrieval: the items of the slot's target records, gate 1 (the
information-matched control, never used in training reads). Items come from an
``ItemCache``, which holds one leaf tensor per item and key for a training step, so
gradients of several reads accumulate before one sparse live update per item
(``KnowledgeBase.live_step`` and ``live_key_step``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, SpaceSpec, search_kbs
from schnitz.mlp_matrix import MLPMatrix

# fine spaces retrieve few items, coarse spaces many (docs 3); about 8 writer reps'
# worth of positions per candidate set and space for equal-length items
DEFAULT_CANDIDATES = {'A': 8, 'B': 16, 'C': 32, 'D': 64}
# extreme sparsity (owner, 28 September): a read keeps a handful of items per space
DEFAULT_KEEP = {'A': 2, 'B': 2, 'C': 3, 'D': 4}

Ref = tuple[str, str]    # (dataset, item id) within one space


@dataclass
class ReadConfig:
    spaces: dict[str, SpaceSpec] = field(default_factory=lambda: dict(DEFAULT_SPACES))
    candidates: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CANDIDATES))
    keep: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_KEEP))
    hidden: int = 1024            # the decoder's hidden width (query input)
    span_width: int = 1024        # the decoder's input-embedding width (span output)
    target_norm: float = 1.0      # interface norm of span reps
    state: int = 512
    op_hidden: int = 256
    layers: int = 3
    gate: str = 'sigmoid'         # or 'softmax'
    max_items: float = 4.0
    min_reps: int = 1
    max_reps: int = 16
    checkpointing: bool = True


class KeyHeads(nn.Module):
    """One linear head per space from the (layer-normalized) query-layer state."""

    def __init__(self, hidden: int, spaces: Mapping[str, SpaceSpec]):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleDict({s: nn.Linear(hidden, spec.key_width)
                                    for s, spec in spaces.items()})

    def forward(self, state: Tensor) -> dict[str, Tensor]:
        x = self.norm(state.float())
        return {s: head(x) for s, head in self.heads.items()}


class Router(nn.Module):
    """Gate logits exp(a_s) (cos(q, k) - b_s) per space."""

    def __init__(self, spaces: Sequence[str], scale: float = 10.0, bias: float = 0.0):
        super().__init__()
        self.log_scale = nn.ParameterDict({s: nn.Parameter(torch.tensor(math.log(scale)))
                                           for s in spaces})
        self.bias = nn.ParameterDict({s: nn.Parameter(torch.tensor(bias)) for s in spaces})

    def logits(self, space: str, query: Tensor, keys: Tensor) -> Tensor:
        cos = F.cosine_similarity(query.float()[None], keys.float(), dim=-1)
        return self.log_scale[space].exp() * (cos - self.bias[space])


class ItemCache:
    """Items and keys of one step as leaf tensors (one per item), so gradients from all
    reads of a step accumulate; ``apply`` makes one sparse Adam update per touched item.
    ``train=False`` reads without gradients and never writes."""

    def __init__(self, device: torch.device | str = 'cpu', train: bool = True):
        self.device, self.train = torch.device(device), train
        self.values: dict[tuple[str, str, str], Tensor] = {}
        self.keys: dict[tuple[str, str, str], Tensor] = {}
        self.times: dict[tuple[str, str, str], int] = {}
        self.kbs: dict[str, KnowledgeBase] = {}

    def get(self, kb: KnowledgeBase, space: str, ids: Sequence[str]
            ) -> list[tuple[Tensor, Tensor, int]]:
        self.kbs[kb.dataset] = kb
        missing = [i for i in dict.fromkeys(ids) if (kb.dataset, space, i) not in self.values]
        if missing:
            live = kb.is_live(space) and kb.writable
            for item in kb.read(space, missing, live=live):
                ref = (kb.dataset, space, item.id)
                values = item.values.float().to(self.device)
                key = item.key.float().to(self.device)
                self.values[ref] = values.requires_grad_(self.train)
                self.keys[ref] = key.requires_grad_(self.train)
                self.times[ref] = item.time
        return [(self.values[(kb.dataset, space, i)], self.keys[(kb.dataset, space, i)],
                 self.times[(kb.dataset, space, i)]) for i in ids]

    def touched(self) -> int:
        return sum(v.grad is not None for v in self.values.values())

    @torch.no_grad()
    def apply(self, item_lr: float, key_lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
              weight_decay: float = 0.0) -> dict[str, int]:
        """One Adam step on every item (values) and key that received a gradient."""
        if not self.train:
            raise ValueError('an evaluation cache does not update items')
        groups: dict[tuple[str, str], list[str]] = {}
        for (dataset, space, item_id), value in self.values.items():
            key = self.keys[(dataset, space, item_id)]
            if value.grad is not None or key.grad is not None:
                groups.setdefault((dataset, space), []).append(item_id)
        counts = {'items': 0, 'keys': 0}
        for (dataset, space), ids in groups.items():
            kb = self.kbs[dataset]
            refs = [(dataset, space, i) for i in ids]
            with_values = [r for r in refs if self.values[r].grad is not None]
            if with_values and item_lr > 0:
                kb.live_step(space, [r[2] for r in with_values],
                             [self.values[r].grad for r in with_values], lr=item_lr,
                             betas=betas, eps=eps, weight_decay=weight_decay)
                counts['items'] += len(with_values)
            with_keys = [r for r in refs if self.keys[r].grad is not None]
            if with_keys and key_lr > 0:
                kb.live_key_step(space, [r[2] for r in with_keys],
                                 torch.stack([self.keys[r].grad for r in with_keys]),
                                 lr=key_lr, betas=betas, eps=eps)
                counts['keys'] += len(with_keys)
        return counts


@dataclass
class SpaceRead:
    refs: list[Ref]
    gates: Tensor                 # detached, per candidate
    mass: float
    positions: int
    recall: float | None = None   # share of target items among the scored candidates
    recall_read: float | None = None  # ... among the items read (nonzero gates)


@dataclass
class Read:
    span: Tensor                  # (n, span_width); n = 0 for an empty read
    spaces: dict[str, SpaceRead]
    aux: Tensor | None = None     # retrieval loss (targets present and retrieved mode)
    n: int = 0


def source_index(kb: KnowledgeBase, space: str) -> dict[str, list[str]]:
    """Current item ids of ``space`` per source record id (from item provenance)."""
    out: dict[str, list[str]] = {}
    rows = kb._map(space, 'rows.i64')
    visible = kb._visible(rows, kb.cursor)
    for row, item_id in enumerate(kb._row_ids[space]):
        if visible[row]:
            for source in kb._meta(space, row)['sources']:
                out.setdefault(source, []).append(item_id)
    return out


class L1Reader(nn.Module):
    """Key heads, router, per-space superposition operators S_s and recombiner R."""

    def __init__(self, config: ReadConfig):
        super().__init__()
        self.config = c = config
        self.spaces = list(c.spaces)
        self.keys = KeyHeads(c.hidden, c.spaces)
        self.router = Router(self.spaces)
        common = dict(state=c.state, hidden=c.op_hidden, layers=c.layers,
                      checkpoint_layers=c.checkpointing)
        self.operators = nn.ModuleDict({
            s: MLPMatrix({s: spec.width}, spec.width, cond=spec.key_width, relative=False,
                         out_norm=math.sqrt(spec.width), **common)
            for s, spec in c.spaces.items()})
        # the K1 recombiner's form, so its trained weights load directly
        self.recombiner = MLPMatrix({s: spec.width for s, spec in c.spaces.items()},
                                    c.span_width, out_norm=c.target_norm, **common)

    def _gates(self, logits: Tensor) -> Tensor:
        if self.config.gate == 'softmax':
            return torch.softmax(logits, 0)
        return torch.sigmoid(logits)

    def read(self, state: Tensor, kbs: Sequence[KnowledgeBase], allowed: Sequence[str],
             query_time: int, cache: ItemCache, *,
             targets: Mapping[str, Sequence[Ref]] | None = None, gold: bool = False) -> Read:
        """One read for the query-layer ``state`` (hidden,) at a call.

        ``targets`` names the slot's target items per space (for the retrieval loss and
        recall; in ``gold`` mode they are the read). Every KB must be authorized."""
        c = self.config
        allowed = set(allowed)
        denied = [kb.dataset for kb in kbs if kb.dataset not in allowed]
        if denied:
            raise PermissionError(f'not authorized to read {denied}')
        by_dataset = {kb.dataset: kb for kb in kbs}
        queries = self.keys(state)
        reads, masses, lengths, aux = {}, {}, {}, []
        info: dict[str, SpaceRead] = {}
        for s in self.spaces:
            q = queries[s]
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
            got = [(ref, values, key) for ref, (values, key, time)
                   in zip(refs, _fetch(cache, by_dataset, s, refs))
                   if time <= query_time]           # causal even for gold items
            if not got:
                none = None if not wanted else 0.0
                info[s] = SpaceRead([], torch.zeros(0), 0.0, 0, none, none)
                continue
            refs = [r for r, _, _ in got]
            scored = set(refs)
            keys = torch.stack([k for _, _, k in got])
            if gold:
                gates = torch.ones(len(got), device=q.device)
            else:
                logits = self.router.logits(s, q, keys.to(q.device))
                if wanted:
                    aux.append(self._retrieval_loss(s, q, refs, logits, wanted, by_dataset,
                                                    cache, query_time))
                # sparse read: only the top ``keep`` candidates by gate logit carry mass;
                # the rest are scored (retrieval loss) but read with gate exactly 0
                keep = min(c.keep.get(s, len(got)), len(got))
                top = torch.topk(logits.detach(), keep).indices
                order = top.sort().values.tolist()
                got = [got[i] for i in order]
                refs = [refs[i] for i in order]
                gates = self._gates(logits)[order] if c.gate == 'sigmoid' \
                    else torch.softmax(logits[order], 0)
            items = [(s, v.to(q.device), g) for (_, v, _), g in zip(got, gates)]
            g = gates.detach().float()
            size = torch.tensor([float(v.shape[0]) for _, v, _ in got], device=g.device)
            mean_len = float((g * size).sum() / g.sum().clamp_min(1e-12))
            count = max(1, round(mean_len))
            out, mass = self.operators[s](items, count, cond=q[None])
            reads[s], masses[s], lengths[s] = out, mass, mean_len
            recall = recall_read = None
            if wanted:
                recall = sum(r in scored for r in wanted) / len(wanted)
                recall_read = sum(r in set(refs) for r in wanted) / len(wanted)
            info[s] = SpaceRead(refs, g.cpu(), float(mass.detach()), count, recall, recall_read)
        aux_loss = torch.stack(aux).mean() if aux else None
        live = [s for s in self.spaces if s in reads and masses[s].detach() > 0]
        if not live:
            width = c.span_width
            return Read(torch.zeros(0, width, device=state.device), info, aux_loss, 0)
        mu = torch.stack([masses[s].detach().float() for s in live])
        implied = torch.tensor([lengths[s] / c.spaces[s].ratio for s in live], device=mu.device)
        record_len = float((mu * implied).sum() / mu.sum())
        found = min(max(float(mu.mean()), 1.0), c.max_items)
        n = int(min(max(math.ceil(record_len * found), c.min_reps), c.max_reps))
        span, _ = self.recombiner([(s, reads[s], masses[s]) for s in live], n)
        return Read(span, info, aux_loss, n)

    def _retrieval_loss(self, space, q, refs, logits, wanted, by_dataset, cache, query_time):
        """Balanced BCE on gate logits: target items (retrieved or not) up, other
        candidates down. Missing targets are scored with their live keys but do not
        enter the read."""
        extra = [r for r in wanted if r not in set(refs)]
        all_logits, labels = [logits], [torch.tensor([float(r in set(wanted)) for r in refs])]
        if extra:
            fetched = _fetch(cache, by_dataset, space, extra)
            keys = [key for _, key, time in fetched if time <= query_time]
            if keys:
                all_logits.append(self.router.logits(space, q, torch.stack(keys).to(q.device)))
                labels.append(torch.ones(len(keys)))
        z = torch.cat(all_logits).float()
        y = torch.cat(labels).to(z.device)
        pos, neg = y > 0, y == 0
        loss = z.new_zeros(())
        if pos.any():
            loss = loss + F.softplus(-z[pos]).mean()
        if neg.any():
            loss = loss + F.softplus(z[neg]).mean()
        return loss


def _fetch(cache: ItemCache, by_dataset: Mapping[str, KnowledgeBase], space: str,
           refs: Sequence[Ref]) -> list[tuple[Tensor, Tensor, int]]:
    """``cache.get`` for refs of several KBs, one store read per KB, in ``refs`` order."""
    out: dict[Ref, tuple[Tensor, Tensor, int]] = {}
    for dataset in dict.fromkeys(d for d, _ in refs):
        ids = [i for d, i in refs if d == dataset]
        out.update({(dataset, i): got for i, got in
                    zip(ids, cache.get(by_dataset[dataset], space, ids))})
    return [out[r] for r in refs]


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
