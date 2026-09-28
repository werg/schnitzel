"""The superposed KB: rows as combiner outputs over overlapping fields (docs/knowledge-base-stack.md,
5.1 steps 7 and 8; owner-confirmed spec and schedule, 28 September).

A space's KB is a set of **rows**, one per stored learnable record, fewer than the
source records (the storage budget). Each row is an anchor in key space (its key). The
write side has a combiner output (S_s, the aggregator) at every row, consuming a
**field** of write inputs: the lower level's items nearest that row. Level 0 are the
leaves (the source-level items: the codecs' items of the source records); level 1 is a
combiner output at every row over a field of leaves; at depth L >= 2 each level again
has an output at every row over a field of the level below's row outputs, and the top
level's outputs stand for the rows. Reads retrieve rows by the query key and the
query-conditioned recombiner R turns them into the span (``schnitz.kb.read``); no
aggregator runs at a query's or a source record's key.

- **Fields.** Each input's candidates are its c nearest rows (``--overlap``; redrawn
  at every rebuild), and its shares over them are ``softmax(tau cos(k_input, k_row))``,
  differentiable in both keys and in tau (learnable per level), recomputed at every
  combine (the stored shares are those of the last rebuild): an input's shares sum to one
  and its mass is split over its rows, never duplicated or dropped (invariant 7); a
  row's mass is the share-weighted sum of its inputs' masses, so each level carries
  the KB's total mass (invariant 5). A row whose field would be empty takes its nearest
  input as an extra member (that input's shares are renormalized over its c + 1 rows).
  At level 1 the mean fill is c N / M; the row count per space follows the field size,
  M = ceil(c / f_s x N) (``--field A=8,B=16,C=16,D=16``: about the read's neighbourhood
  per space), or ``--budget``.
- **Positions.** The aggregator is conditioned on its row's key (``target_key``) and each
  input enters with its key offset from the row (``key_input - key_row`` as the
  operator's per-item extra feature, ``neighbour_keys=True``), so a field is not a bag;
  level-2 inputs are positioned by their rows' keys.
- **Time.** A row's time is the latest time of its inputs (recursively, its leaves); a
  read never sees a row later than its query time (invariant 2): exact and conservative.
- **One KB.** Fields are built per KB, one authorization domain (invariant 6).
- **Writes** become leaves: ``insert`` places a new or rewritten leaf by its key into the
  fields of its c nearest rows and re-runs only the combiners whose fields changed
  (and the rows above them); the rows stay. ``rebuild`` reassigns every field from the
  rows' current keys (every ``--graph-every`` steps and at every checkpoint).

Schedule (owner): ``build_rows`` makes the rows KB (rows placed by farthest-point sampling
over the leaves' keys, initialized as the share-weighted mean of their field so they
have content and keys; lineage and shares recorded by ``kb_store.rewrite``). A long read
phase trains the rows as free learnable parameters by reads alone (L1a on the rows KB:
R, key and query heads, rows; the row drift from initialization is logged per space).
The write fit (L2 ``--producer stack``) then regresses the rows' values and keys from the
leaves through the aggregator levels with the read side cut off (``fit_losses``); the
read side's functional check runs only at its evaluations. Reads that see the stack's
outputs instead of the free rows (``l1 train --rows-from-stack``: this module's
``SuperposedKB`` and ``SuperposedCache``) are an available switch, not the default.

With ``--rows-from-stack`` evaluation is lazy (``SuperposedCache``): only the rows a read
retrieves are computed, from level-(L-1) inputs held in a cache whose entries expire
after ``--cache-every`` steps (valid at least for the step they were computed in; a
rebuild clears it). The task gradient reaches the read's rows as leaf tensors
(gradients of all reads of a step accumulate); ``SuperposedCache.backward`` then
recomputes each row with a graph: its level-(L-1) inputs are the cached values, and a
sampled fraction ``--deep-grad`` p of them is recomputed one level further down
(recursively, to the leaves) and enters as ``cached + (fresh - fresh.detach()) / p``.
The forward stays the cached value bit for bit; the gradient reaches the leaves and the
lower aggregators through the fresh recomputation, scaled to be unbiased.
**Approximation:** the gradient is taken at the fresh values while the forward used
cached values up to ``cache_every`` steps old, and only a sampled part of each field
propagates per step (p = 1 with fresh caches is the exact gradient; tested). At depth 1
the inputs are the leaves themselves (exact). Gradients stop at the leaves.

**Keys are first-class** (owner, 28 September). In the read phase each row's key is a
free parameter next to its value (``ReadConfig.learned_keys``: the live key, moved by the
retrieval and gate gradients through ``read.KeyOptimizer``; the search scans the live
keys). On the write side a row's key follows its field: the inputs' keys weighted by the
aggregator's input weights (shares x mass, normalized) plus a zero-initialized key head
on the row's mean output (``field_key``); rows are re-anchored at their current keys and
the fields reassigned at each rebuild. ``build_rows`` keys new rows by the field mean.

Search runs over the rows' keys (refreshed on the trainer's rekey schedule). A slot's retrieval positives are the rows whose fields
cover its leaves, weighted by the leaves' share of the row's mass (``covering``); the
gold control reads, per leaf, the row holding most of its mass; a read that must not
see some leaves (an episode's own writes) excludes every row holding any of them.

``consolidate`` re-fits the leaves after an aggregator update so the touched rows return
to their previous values (item-preserving; off by default). ``export`` materializes the
rows into a frozen KB (``kb_store``): the leaves, then each level as one ``rewrite`` with
the fields' shares (lineage, responsibility shares and masses down to the sources).

API for other stages (K3, L2): ``SuperposeConfig.row_count`` / ``place_rows`` /
``build_rows`` (rows per space: keys, values, learnable in a KB; ``rows_of`` reads them),
``build_graph`` (field assignment of lower-level items to rows, per level), ``aggregate``
and ``SuperposedKB.grad_value`` (the stack forward with gradients), ``fit_losses`` (the
stack fit to row targets).

Training-only; inference reads stored payloads (invariant 1).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import scipy.sparse as sparse
import torch
from torch import Tensor, nn

from schnitz.kb.read import ItemCache, current_ids
from schnitz.kb.stack import KEY_WIDTH, SPACES
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance

DEFAULT_FIELD = {'A': 8, 'B': 16, 'C': 16, 'D': 16}
Ref = tuple[str, str]


@dataclasses.dataclass
class SuperposeConfig:
    depth: int = 2                  # L; 0: the rows are free parameters (plain rows KB)
    field: dict = dataclasses.field(default_factory=lambda: dict(DEFAULT_FIELD))
    overlap: int = 3                # c: fields (rows) each input joins
    budget: dict | None = None      # space -> rows as a fraction of the leaves
    temperature: float = 0.1        # share softmax over cosine to the input's rows
    deep_grad: float = 0.25         # p: sampled inputs recomputed one level down
    cache_every: int = 10           # level-(L-1) cache entries live this many steps
    graph_every: int = 100          # field reassignment period (and every checkpoint)
    max_positives: int = 8          # retrieval positives kept per read and space
    per_level: bool = True          # an aggregator per level (else one for all levels)
    seed: int = 0
    batched: bool = True            # all rows of a level in one aggregator pass
    max_pairs: int = 1 << 18        # (output, input) position pairs per batched pass

    def field_range(self, space: str) -> tuple[float, float]:
        """The level-1 field size of a space as (low, high): a number or a range ``lo:hi``."""
        f = self.field.get(space, 8)
        lo, hi = (f, f) if isinstance(f, (int, float)) else f
        return float(lo), float(hi)

    def field_size(self, space: str) -> float:
        """The nominal field size: the geometric middle of the range."""
        lo, hi = self.field_range(space)
        return math.sqrt(lo * hi)

    def sample_field(self, space: str, rng: random.Random) -> float:
        """A field size drawn log-uniformly from the range (``--field A=4:16``)."""
        lo, hi = self.field_range(space)
        return math.exp(rng.uniform(math.log(lo), math.log(hi))) if hi > lo else lo

    def level1_overlap(self, field: float, leaves: int, rows: int) -> int:
        """Candidate rows per leaf that give a level-1 field of about ``field`` inputs:
        fill = c N / M."""
        return max(1, min(rows, round(field * rows / max(leaves, 1))))

    def row_count(self, space: str, n: int) -> int:
        """Rows of a space over ``n`` leaves: ceil(c / f_s x n) at the nominal field size,
        or the budget fraction."""
        if n <= 0:
            return 0
        budget = (self.budget or {}).get(space)
        if budget is None:
            budget = min(1.0, self.overlap / max(1.0, self.field_size(space)))
        return max(1, min(n, math.ceil(budget * n)))

    @classmethod
    def from_dict(cls, data: Mapping) -> SuperposeConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def parse_fields(text: str | None) -> dict | None:
    """``A=4:16,B=16`` -> {'A': (4, 16), 'B': 16}; a bare value applies to every space."""
    if text is None or text == '':
        return None

    def one(v: str):
        lo, _, hi = v.partition(':')
        return (int(lo), int(hi)) if hi else int(lo)
    if '=' not in text:
        return {s: one(text) for s in SPACES}
    return {k.strip(): one(v) for k, v in (p.split('=') for p in text.split(',') if p)}


def parse_pairs(text: str | None, cast=float) -> dict | None:
    """``A=8,B=16`` -> {'A': 8, 'B': 16}; a bare number applies to every space."""
    if text is None or text == '':
        return None
    if '=' not in text:
        return {s: cast(text) for s in SPACES}
    return {k.strip(): cast(v) for k, v in (p.split('=') for p in text.split(',') if p)}


# -- rows and fields ------------------------------------------------------------------------
def farthest_points(keys: Tensor, m: int) -> list[int]:
    """``m`` rows of unit ``keys`` by farthest-point sampling (cosine), float64 on the CPU;
    starts at the row farthest from the mean key (ties: the lower row)."""
    keys = keys.detach().double().cpu()
    n = keys.shape[0]
    m = min(m, n)
    if m <= 0:
        return []
    first = int(torch.argmin(keys @ keys.mean(0)))
    chosen = [first]
    near = keys @ keys[first]          # similarity to the nearest chosen row
    near[first] = math.inf
    for _ in range(m - 1):
        nxt = int(torch.argmin(near))
        chosen.append(nxt)
        near = torch.maximum(near, keys @ keys[nxt])
        near[chosen] = math.inf
    return chosen


SCAN_ELEMENTS = 1 << 25    # similarities per block of an exact scan (256 MB in float64)


def exact_topk(queries: Tensor, keys: Tensor, k: int, chunk: int | None = None
               ) -> tuple[Tensor, Tensor]:
    """Exact top-``k`` cosine neighbours among ``keys`` of every query (float64 on the CPU,
    blocks of ``chunk`` queries, by default ``SCAN_ELEMENTS`` similarities per block; no
    approximate index)."""
    q = queries.detach().double().cpu()
    kk = keys.detach().double().cpu()
    k = min(k, kk.shape[0])
    chunk = chunk or max(1, SCAN_ELEMENTS // max(kk.shape[0], 1))
    values, indices = [], []
    for start in range(0, q.shape[0], chunk):
        top = (q[start:start + chunk] @ kk.T).topk(k, dim=1)
        values.append(top.values)
        indices.append(top.indices)
    return torch.cat(values), torch.cat(indices)


def exact_argmax(queries: Tensor, keys: Tensor, chunk: int | None = None) -> Tensor:
    """For every query the index of its most similar key (exact, blocks of ``chunk``
    keys, by default ``SCAN_ELEMENTS`` similarities per block; ties: the lower index)."""
    q = queries.detach().double().cpu()
    kk = keys.detach().double().cpu()
    chunk = chunk or max(1, SCAN_ELEMENTS // max(q.shape[0], 1))
    best = torch.full((q.shape[0],), -math.inf, dtype=torch.float64)
    arg = torch.zeros(q.shape[0], dtype=torch.long)
    for start in range(0, kk.shape[0], chunk):
        sim = q @ kk[start:start + chunk].T
        val, idx = sim.max(dim=1)
        better = val > best
        best = torch.where(better, val, best)
        arg = torch.where(better, idx + start, arg)
    return arg


def kmeanspp(keys: Tensor, k: int, gen: torch.Generator) -> Tensor:
    """k-means++ seeding (cosine distance 1 - cos, D^2 sampling) of ``k`` rows of unit
    ``keys``; returns their indices. O(n k)."""
    keys = keys.detach().double().cpu()
    n = keys.shape[0]
    k = min(k, n)
    first = int(torch.randint(n, (1,), generator=gen))
    chosen = [first]
    dist = (1 - keys @ keys[first]).clamp_min(0)
    for _ in range(k - 1):
        w = dist ** 2
        total = float(w.sum())
        nxt = int(torch.multinomial(w / total, 1, generator=gen)) if total > 0 else \
            int(torch.randint(n, (1,), generator=gen))
        chosen.append(nxt)
        dist = torch.minimum(dist, (1 - keys @ keys[nxt]).clamp_min(0))
    return torch.tensor(chosen)


def place_rows(leaf_keys: Tensor, m: int, bucket: int = 4096, seed: int = 0) -> Tensor:
    """Initial row positions: the keys of ``m`` leaves chosen by farthest-point sampling
    (spread over the occupied key space, reproducible). Up to ``2 bucket`` leaves one
    exact FPS (O(N m)); beyond, FPS within buckets: ``ceil(N / bucket)`` centres by
    k-means++ seeding on a sample, every leaf assigned exactly to its nearest centre,
    and each bucket's share of the ``m`` rows (proportional to its leaves) by FPS inside
    it: O(N N/bucket + bucket m). The rows' own keys follow from their values once
    initialized."""
    keys = nn.functional.normalize(leaf_keys.detach().float().cpu(), dim=-1)
    n = keys.shape[0]
    m = min(m, n)
    if n <= 2 * bucket:
        return keys[sorted(farthest_points(keys, m))]
    gen = torch.Generator().manual_seed(seed)
    centres_n = math.ceil(n / bucket)
    sample = torch.randperm(n, generator=gen)[:min(n, max(20 * centres_n, 20000))]
    centres = keys[sample[kmeanspp(keys[sample], centres_n, gen)]]
    owner = exact_argmax(keys, centres)
    members = [torch.nonzero(owner == c).flatten() for c in range(centres.shape[0])]
    sizes = np.array([len(x) for x in members], np.float64)
    quota = sizes / sizes.sum() * m
    take = np.floor(quota).astype(int)
    for c in np.argsort(-(quota - take))[:m - int(take.sum())]:   # largest remainders
        take[c] += 1
    take = np.minimum(take, sizes.astype(int))
    chosen = []
    for idx, t in zip(members, take.tolist()):
        if t > 0:
            chosen += idx[farthest_points(keys[idx], t)].tolist()
    return keys[sorted(chosen)]


def memberships(keys: Tensor, rows: Tensor, overlap: int,
                tau: float) -> list[list[tuple[int, float]]]:
    """Per input: its ``overlap`` nearest rows (its candidates) and its shares over them,
    ``softmax(tau cos)`` (sum one)."""
    if not len(keys) or not len(rows):
        return [[] for _ in range(len(keys))]
    values, indices = exact_topk(keys, rows, overlap)
    return [_shares(list(zip(idx, val)), tau)
            for idx, val in zip(indices.tolist(), values.tolist())]


def _shares(pairs: Sequence[tuple[int, float]], tau: float) -> list[tuple[int, float]]:
    w = np.exp(tau * (np.array([v for _, v in pairs]) - 1.0))
    return list(zip([o for o, _ in pairs], (w / w.sum()).tolist()))


def fields_of(keys: Tensor, rows: Tensor, overlap: int, tau: float
              ) -> tuple[list[list[tuple[int, float]]], list[list[int]]]:
    """Per row its field ((input, share) pairs) and per input its candidate rows. Every
    input's candidates are its c nearest rows; a row left empty becomes a candidate of
    its nearest input (whose shares are renormalized over its c + 1 rows)."""
    member = memberships(keys, rows, overlap, tau)
    if len(keys) and len(rows):
        filled = {o for pairs in member for o, _ in pairs}
        empty = [o for o in range(rows.shape[0]) if o not in filled]
        if empty:
            k64, r64 = keys.detach().double().cpu(), rows.detach().double().cpu()
            nearest = exact_argmax(r64[empty], k64).tolist()
            for o, i in zip(empty, nearest):
                member[i] = _shares([(r, float(k64[i] @ r64[r])) for r, _ in member[i]]
                                    + [(o, float(k64[i] @ r64[o]))], tau)
    fields: list[list[tuple[int, float]]] = [[] for _ in range(len(rows))]
    for i, pairs in enumerate(member):
        for o, share in pairs:
            fields[o].append((i, share))
    return fields, [[o for o, _ in pairs] for pairs in member]


def dynamic_shares(key: Tensor, anchors: Tensor, candidates: Sequence[int], tau: Tensor) -> Tensor:
    """An input's shares over its candidate rows, ``softmax(tau cos(k_input, k_row))``,
    differentiable in the input key and tau (sum one: invariant 7)."""
    rows = anchors[list(candidates)].to(key.device)
    return torch.softmax(tau * (rows @ nn.functional.normalize(key.float(), dim=-1)), dim=0)


def candidate_shares(keys: Tensor, anchors: Tensor, candidates: Sequence[Sequence[int]],
                     tau: Tensor) -> tuple[Tensor, Tensor]:
    """``dynamic_shares`` of many inputs at once: (P, C) shares over each input's
    candidate rows (padded to the longest list; padding has share 0) and the (P, C)
    candidate row index."""
    width = max(len(c) for c in candidates)
    index = to_device(torch.tensor([list(c) + [0] * (width - len(c)) for c in candidates]),
                      keys.device)
    valid = to_device(torch.tensor([[True] * len(c) + [False] * (width - len(c))
                                    for c in candidates]), keys.device)
    sims = torch.einsum('pcd,pd->pc', anchors.to(keys.device).float()[index],
                        nn.functional.normalize(keys.float(), dim=-1))
    return torch.softmax((tau * sims).masked_fill(~valid, -math.inf), dim=1), index


@dataclasses.dataclass
class Level:
    """Combiner outputs of one level, one per row: field inputs (rows of the level
    below) with their shares as of the last reassignment, mass, time and position count;
    ``candidates[i]`` are input i's candidate rows (the normalization set of its shares)."""
    inputs: list[list[int]]
    shares: list[list[float]]
    mass: np.ndarray
    time: np.ndarray
    count: list[int]
    candidates: list[list[int]] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        return {'inputs': self.inputs, 'shares': self.shares, 'mass': self.mass.tolist(),
                'time': self.time.tolist(), 'count': self.count, 'candidates': self.candidates}

    def matrix(self, n_in: int) -> sparse.csr_matrix:
        rows = [j for j, ins in enumerate(self.inputs) for _ in ins]
        cols = [i for ins in self.inputs for i in ins]
        vals = [s for sh in self.shares for s in sh]
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.inputs), n_in))


def _count(mass, length: Sequence[int], ins: Sequence[int], sh: Sequence[float]) -> int:
    """A combiner emits the gate-weighted mean length of its inputs."""
    g = np.array([s * mass[i] for i, s in zip(ins, sh)])
    n = np.array([length[i] for i in ins], np.float64)
    total = g.sum()
    return max(1, round(float((g * n).sum() / total)) if total > 0 else round(float(n.mean())))


def summarize(fields, mass, time, length, candidates=None) -> Level:
    inputs = [[i for i, _ in sorted(fld)] for fld in fields]
    shares = [[s for _, s in sorted(fld)] for fld in fields]
    masses = np.array([sum(s * mass[i] for i, s in zip(ins, sh))
                       for ins, sh in zip(inputs, shares)], np.float64)
    times = np.array([max(int(time[i]) for i in ins) if ins else 0 for ins in inputs], np.int64)
    counts = [_count(mass, length, ins, sh) if ins else 1 for ins, sh in zip(inputs, shares)]
    if candidates is None:
        candidates = [[] for _ in range(len(mass))]
        for o, ins in enumerate(inputs):
            for i in ins:
                candidates[i].append(o)
    return Level(inputs, shares, masses, times, counts, [list(c) for c in candidates])


def _level_id(dataset: str, space: str, level: int, row: str) -> str:
    return 's' + hashlib.sha1(f'{dataset}#{space}#L{level}#{row}'.encode()).hexdigest()[:31]


def row_id(dataset: str, space: str, n: int) -> str:
    return 'r' + hashlib.sha1(f'{dataset}#{space}#row#{n}'.encode()).hexdigest()[:31]


class SpaceGraph:
    """One space of a superposed KB: the leaves (ids, masses, times, lengths, unit keys),
    the rows (ids, unit keys) and the levels (one combiner output per row and level).
    ``comp`` (rows x leaves) holds each row's source composition (share x mass through
    the levels; rows sum to one); ``holder[i]`` is the row holding most of leaf i's mass."""

    def __init__(self, dataset: str, space: str, ids: Sequence[str], mass, time, length,
                 keys: Tensor, row_ids: Sequence[str], row_keys: Tensor,
                 levels: Sequence[Level], row_lengths: Sequence[int] | None = None):
        self.dataset, self.space = dataset, space
        self.row_lengths = None if row_lengths is None else [int(x) for x in row_lengths]
        self.ids = list(ids)
        self.mass = np.asarray(mass, np.float64)
        self.time = np.asarray(time, np.int64)
        self.length = [int(x) for x in length]
        self.keys = keys
        self.row_ids, self.row_keys = list(row_ids), row_keys
        self.levels = list(levels)
        self.level_ids = [self.ids] + [[_level_id(dataset, space, n, r) for r in self.row_ids]
                                       for n in range(1, len(self.levels))] + [self.row_ids]
        self.refresh()

    def refresh(self) -> None:
        """Indexes and composition after the leaves or fields changed; the top level
        emits the stored rows' own position counts when known."""
        if self.row_lengths is not None and self.levels:
            self.levels[-1].count = list(self.row_lengths)
        self.index = {i: n for n, i in enumerate(self.ids)}
        self.level_ids[0] = self.ids
        self.top_index = {i: n for n, i in enumerate(self.row_ids)}
        self._compose()

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def top_ids(self) -> list[str]:
        return self.row_ids

    def level_mass(self, level: int) -> np.ndarray:
        return self.mass if level == 0 else self.levels[level - 1].mass

    def level_time(self, level: int) -> np.ndarray:
        return self.time if level == 0 else self.levels[level - 1].time

    def level_length(self, level: int) -> list[int]:
        return self.length if level == 0 else self.levels[level - 1].count

    def level_keys(self, level: int) -> Tensor:
        """Unit keys of a level's items: the leaves' own, above them the rows'."""
        return self.keys if level == 0 else self.row_keys

    @property
    def top_mass(self) -> np.ndarray:
        return self.level_mass(self.depth)

    @property
    def top_time(self) -> np.ndarray:
        return self.level_time(self.depth)

    def _compose(self) -> None:
        n = len(self.ids)
        comp = sparse.identity(n, format='csr')
        below = self.mass
        for level in self.levels:
            s = level.matrix(comp.shape[0])
            inv = np.where(level.mass > 0, 1.0 / np.maximum(level.mass, 1e-300), 0.0)
            comp = sparse.diags(inv) @ s @ sparse.diags(below) @ comp
            below = level.mass
        self.comp = comp.tocsr()
        carried = (sparse.diags(self.top_mass) @ self.comp).tocsc()
        holder = np.full(n, -1, np.int64)
        for i in range(n):
            a, b = carried.indptr[i], carried.indptr[i + 1]
            if b > a:
                holder[i] = int(carried.indices[a + int(np.argmax(carried.data[a:b]))])
        self.holder = holder
        self._comp_csc = self.comp.tocsc()

    def tops_holding(self, leaves: Sequence[int]) -> np.ndarray:
        """Rows holding any share of the given leaf rows."""
        if not len(leaves):
            return np.zeros(0, np.int64)
        return np.unique(self._comp_csc[:, list(leaves)].nonzero()[0])

    def covering(self, leaf_ids: Sequence[str]) -> dict[str, float]:
        """The rows whose fields cover the leaves (through the levels), each with the
        leaves' share of its mass, largest first."""
        rows = [self.index[i] for i in leaf_ids if i in self.index]
        if not rows:
            return {}
        w = np.asarray(self.comp[:, rows].sum(axis=1)).ravel()
        order = [j for j in np.argsort(-w, kind='stable') if w[j] > 0]
        return {self.row_ids[j]: float(w[j]) for j in order}

    def composition(self) -> dict[str, dict[str, float]]:
        out = {}
        for j, top in enumerate(self.row_ids):
            a, b = self.comp.indptr[j], self.comp.indptr[j + 1]
            out[top] = {self.ids[i]: float(v) for i, v in
                        zip(self.comp.indices[a:b], self.comp.data[a:b])}
        return out

    def to_json(self) -> dict:
        return {'ids': self.ids, 'rows': self.row_ids, 'mass': self.mass.tolist(),
                'time': self.time.tolist(), 'length': self.length,
                'levels': [lv.to_json() for lv in self.levels], 'level_ids': self.level_ids}

    def stats(self) -> dict:
        """Per level: mean field fill, fields per input, positions; per row the unique
        leaves it depends on."""
        out = {'leaves': len(self.ids), 'rows': len(self.row_ids), 'levels': []}
        n_in = len(self.ids)
        for level in self.levels:
            per_input = np.zeros(n_in)
            for ins in level.inputs:
                per_input[ins] += 1
            out['levels'].append({
                'field_fill': round(float(np.mean([len(x) for x in level.inputs])), 3)
                if level.inputs else 0.0,
                'fields_per_input': round(float(per_input.mean()), 3) if n_in else 0.0,
                'positions': int(sum(level.count))})
            n_in = len(level.inputs)
        nnz = np.diff(self.comp.indptr)
        out['leaves_per_row'] = {'mean': round(float(nnz.mean()), 2) if len(nnz) else 0,
                                 'max': int(nnz.max()) if len(nnz) else 0}
        out['leaf_positions'] = int(sum(self.length))
        out['row_positions'] = int(sum(self.level_length(self.depth)))
        return out


def build_graph(dataset: str, space: str, ids: Sequence[str], keys: Tensor, mass, time,
                length, row_ids: Sequence[str], row_keys: Tensor, config: SuperposeConfig,
                taus: Sequence[float] | None = None, field: float | None = None,
                row_lengths: Sequence[int] | None = None) -> SpaceGraph:
    """Field assignment of every level to the rows: each input's candidates are its c
    nearest rows by key (level 1 over the leaves, each higher level over the level below's
    row outputs, keyed by their rows), with shares ``softmax(tau cos)`` at the given
    per-level ``taus`` (default 1 / temperature). ``field`` sets level 1's field size
    (candidates per leaf by ``level1_overlap``; default the configured overlap c); higher
    levels take c."""
    keys = nn.functional.normalize(keys.detach().float().cpu(), dim=-1)
    row_keys = nn.functional.normalize(row_keys.detach().float().cpu(), dim=-1)
    mass, time = np.asarray(mass, np.float64), np.asarray(time, np.int64)
    levels = []
    k, m_, t_, n_ = keys, mass, time, list(length)
    for n in range(max(1, config.depth)):
        tau = taus[n] if taus is not None else 1.0 / max(config.temperature, 1e-6)
        overlap = config.overlap if n or field is None else \
            config.level1_overlap(field, len(ids), len(row_ids))
        fields, candidates = fields_of(k, row_keys, overlap, tau)
        level = summarize(fields, m_, t_, n_, candidates)
        levels.append(level)
        k, m_, t_, n_ = row_keys, level.mass, level.time, level.count
    return SpaceGraph(dataset, space, ids, mass, time, length, keys, row_ids, row_keys, levels,
                      row_lengths)


def mean_key(keys: Sequence[Tensor], gates: Sequence[float]) -> Tensor:
    """The unit, gate-weighted mean of unit keys (``field_key`` without correction)."""
    w = torch.tensor(list(gates), dtype=torch.float)
    w = w / w.sum().clamp_min(1e-12)
    return nn.functional.normalize((w[:, None] * torch.stack([k.float().cpu() for k in keys]))
                                   .sum(0), dim=-1)


def resample(values: Tensor, count: int) -> Tensor:
    """An item's positions linearly resampled to ``count`` (positions at (i + 0.5) / n)."""
    if values.shape[0] == count:
        return values
    return nn.functional.interpolate(values.T[None].float(), size=count, mode='linear',
                                     align_corners=False)[0].T


def mean_rows(graph: SpaceGraph, leaves: Sequence[Tensor]) -> list[Tensor]:
    """The untrained aggregation of level 1: each row as the gate-weighted mean of its
    field's leaves (resampled to the row's count)."""
    level = graph.levels[0]
    out = []
    for ins, sh, count in zip(level.inputs, level.shares, level.count):
        gates = [s * graph.mass[i] for i, s in zip(ins, sh)]
        total = sum(gates) or 1.0
        out.append(sum(g / total * resample(leaves[i].float(), count)
                       for i, g in zip(ins, gates)))
    return out


@torch.no_grad()
def build_rows(kb: KnowledgeBase, dest: Path, config: SuperposeConfig,
               name: str | None = None) -> tuple[KnowledgeBase, dict]:
    """A rows KB (the storage budget of learnable records) from a KB of leaves: per space
    ``row_count`` rows placed by farthest-point sampling over the leaves' keys, each
    initialized as the share-weighted mean of its field (``mean_rows``, depth-1 fields),
    keyed by the share-weighted mean of its field's keys (the write-side key rule with a
    zero correction, ``field_key``); the leaves are appended and replaced by the rows in one
    ``rewrite`` per space (lineage, shares, masses, times). Current items are the rows."""
    out = KnowledgeBase.create(dest, name=name or f'{kb.name}@rows', dataset=kb.dataset,
                               spaces=kb.spaces, origin={'command': 'superpose.build_rows',
                                                         'kb': str(kb.root),
                                                         'config': dataclasses.asdict(config)})
    stats = {}
    for space in kb.spaces:
        ids = current_ids(kb, space)
        if not ids:
            continue
        items = kb.read(space, ids)
        keys = torch.stack([it.key.float() for it in items])
        anchors = place_rows(keys, config.row_count(space, len(ids)), seed=config.seed)
        rows = [row_id(kb.dataset, space, n) for n in range(anchors.shape[0])]
        one = dataclasses.replace(config, depth=1)
        graph = build_graph(kb.dataset, space, ids, keys, [it.mass for it in items],
                            [it.time for it in items], [it.values.shape[0] for it in items],
                            rows, anchors, one)
        values = mean_rows(graph, [it.values.float() for it in items])
        level = graph.levels[0]
        row_keys = [mean_key([graph.keys[i] for i in ins],
                             [s * graph.mass[i] for i, s in zip(ins, sh)])
                    for ins, sh in zip(level.inputs, level.shares)]
        out.append(space, [NewItem(it.values, it.key, it.provenance, it.mass, it.time, it.id)
                           for it in items])
        out.rewrite(space, ids, [
            NewItem(v.cpu(), k.float().cpu(), Provenance((), 'rewrite'),
                    float(level.mass[j]), int(level.time[j]), rows[j])
            for j, (v, k) in enumerate(zip(values, row_keys))],
            shares=level.matrix(len(ids)).toarray())
        stats[space] = graph.stats()
    return out, stats


# -- aggregators ------------------------------------------------------------------------------
class WriteOps(nn.Module):
    """The aggregators S_s (write side only): per level (``per_level``) or one set for
    all levels, one ``SuperpositionOperator`` per space, a key head per space (the row
    key's correction, zero at the start) and a learnable share sharpness tau per space
    (``log_tau``, starting at 1 / temperature). ``load_init`` copies per-space operator
    weights (e.g. K3's) into every level."""

    def __init__(self, depth: int, dims: Mapping[str, int], per_level: bool = True,
                 checkpoint: bool = False, temperature: float = 0.1):
        super().__init__()
        from schnitz.kb.stack import SuperpositionOperator
        self.per_level = per_level
        names = [str(n) for n in range(1, max(1, depth) + 1)] if per_level else ['all']
        self.levels = nn.ModuleDict({n: nn.ModuleDict({
            s: SuperpositionOperator(s, dims['state'], dims['hidden'], dims['layers'], checkpoint)
            for s in SPACES}) for n in names})
        self.key_heads = nn.ModuleDict({n: nn.ModuleDict({
            s: nn.Linear(SPACES[s][1], KEY_WIDTH[s]) for s in SPACES}) for n in names})
        for head in (h for level in self.key_heads.values() for h in level.values()):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        start = math.log(1.0 / max(temperature, 1e-6))
        self.log_tau = nn.ParameterDict({f'{n}_{s}': nn.Parameter(torch.tensor(start))
                                         for n in names for s in SPACES})

    @torch.no_grad()
    def load_init(self, state: Mapping[str, Tensor]) -> int:
        """Per-space operator weights (keys ``<space>.op...``, optionally prefixed by
        ``ops.`` or ``operators.``) into every level. Returns the tensors loaded."""
        picked = {}
        for prefix in ('ops.', 'operators.', ''):
            picked = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
                      and k[len(prefix):].split('.')[0] in SPACES}
            if picked:
                break
        count = 0
        for module in self.levels.values():
            own = module.state_dict()
            for k, v in picked.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v.clone()
                    count += 1
            module.load_state_dict(own)
        return count

    def _name(self, level: int) -> str:
        return str(level) if self.per_level else 'all'

    def op(self, level: int, space: str) -> nn.Module:
        return self.levels[self._name(level)][space]

    def key_head(self, level: int, space: str) -> nn.Module:
        return self.key_heads[self._name(level)][space]

    def tau(self, level: int, space: str) -> Tensor:
        return self.log_tau[f'{self._name(level)}_{space}'].exp()

    def taus(self, space: str, depth: int) -> list[float]:
        return [float(self.tau(n, space).detach()) for n in range(1, depth + 1)]


def head_input(out: Tensor) -> Tensor:
    """The key head's input: the row output's mean over positions, unit length (the
    outputs' rows have norm sqrt(width), so an unnormalized mean would make every Adam
    step of the zero-initialized head move the key by about lr x width)."""
    return nn.functional.normalize(out.float().mean(0), dim=-1)


def field_key(head: nn.Module, keys: Sequence[Tensor], gates, out: Tensor) -> Tensor:
    """A row's key from its field: the inputs' unit keys weighted by the aggregator's
    input weights (gates share x mass, normalized; differentiable), plus the key head's
    correction of the row's mean output; unit length."""
    w = gates if torch.is_tensor(gates) else torch.tensor(list(gates), dtype=torch.float)
    w = w.to(out.device).float()
    w = w / w.sum().clamp_min(1e-12)
    base = (w[:, None] * torch.stack([k.to(out.device).float() for k in keys])).sum(0)
    return nn.functional.normalize(base + head(head_input(out)), dim=-1)


def apply_field(op: nn.Module, values: Sequence[Tensor], gates, offsets: Sequence[Tensor],
                anchor: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """One combiner output: S_s over a field (values at gates share x mass - tensors,
    differentiable - positioned by their key offsets from the row), conditioned on the
    row's key. Returns the output and its mass (the sum of the gates)."""
    out, mass = op([(v, g, o) for v, g, o in zip(values, gates, offsets)], anchor, int(count),
                   neighbour_keys=True)
    return out.float(), mass


def combine(ops: WriteOps, graph: SpaceGraph, level: int, j: int,
            inputs: Sequence[tuple[Tensor, Tensor, Tensor]], device, autocast
            ) -> tuple[Tensor, Tensor, Tensor]:
    """Row ``j`` of ``level`` from its field's inputs ((value, unit key, mass) each):
    each input's gate is its mass times its dynamic share to this row (``dynamic_shares``
    over its candidate rows, learnable tau), so the fit and task losses move input keys
    toward the rows they help. Returns (value, key, mass)."""
    lv = graph.levels[level - 1]
    anchors = graph.row_keys.to(device)
    anchor = anchors[j]
    tau = ops.tau(level, graph.space)
    gates, offsets = [], []
    for k, (_, key, mass) in zip(lv.inputs[j], inputs):
        cands = lv.candidates[k]
        share = dynamic_shares(key.to(device), anchors, cands, tau)[cands.index(j)]
        gates.append(mass.to(device).float() * share)
        offsets.append(key.to(device).float() - anchor)
    gates = torch.stack(gates)
    with autocast():
        out, mass = apply_field(ops.op(level, graph.space), [v.to(device) for v, _, _ in inputs],
                                gates, offsets, anchor, lv.count[j])
    key = field_key(ops.key_head(level, graph.space), [k for _, k, _ in inputs], gates, out)
    return out, key, mass.float()


def combine_many(ops: WriteOps, graph: SpaceGraph, level: int, rows: Sequence[int],
                 inputs: Sequence[Sequence[tuple[Tensor, Tensor, Tensor]]], device, autocast,
                 max_pairs: int | None = None) -> list[tuple[Tensor, Tensor, Tensor]]:
    """``combine`` for many rows of one level in one aggregator pass (packed pairs,
    ``MLPMatrix.forward_many``): the dynamic shares of every (row, input) pair in one
    masked softmax over the inputs' candidate rows, the field keys in one head call.
    Equal to ``[combine(..., j, ...) for j in rows]`` up to summation order."""
    rows = list(rows)
    if not rows:
        return []
    lv = graph.levels[level - 1]
    anchors = graph.row_keys.to(device).float()
    tau = ops.tau(level, graph.space)
    owner, cand_rows, position, keys, masses, values = [], [], [], [], [], []
    for g, (j, ins) in enumerate(zip(rows, inputs)):
        for k, (v, key, mass) in zip(lv.inputs[j], ins):
            cands = lv.candidates[k]
            owner.append(g)
            cand_rows.append(cands)
            position.append(cands.index(j))
            keys.append(key)
            masses.append(mass)
            values.append(v)
    values = gather_to(values, device)
    key_in = torch.stack(gather_to(keys, device)).float()
    shares, _ = candidate_shares(key_in, anchors, cand_rows, tau)
    share = shares.gather(1, to_device(torch.tensor(position), device)[:, None])[:, 0]
    gates = torch.stack(gather_to(masses, device)).float() * share
    owner_t = to_device(torch.tensor(owner), device)
    offsets = key_in - anchors[to_device(torch.tensor(rows), device)][owner_t]
    groups, start = [], 0
    counts = [len(lv.inputs[j]) for j in rows]
    for g, (j, n) in enumerate(zip(rows, counts)):
        groups.append(([(values[i], gates[i], offsets[i])
                        for i in range(start, start + n)], anchors[j], int(lv.count[j])))
        start += n
    with autocast():
        got = ops.op(level, graph.space).forward_many(groups, neighbour_keys=True,
                                                      max_pairs=max_pairs)
    outs = [o.float() for o, _ in got]
    # field keys: gate-normalized mean of the inputs' keys + the head on the mean output
    total = torch.zeros(len(rows), device=device).index_add(0, owner_t, gates)
    w = gates / total[owner_t].clamp_min(1e-12)
    base = torch.zeros(len(rows), key_in.shape[1], device=device).index_add(
        0, owner_t, w[:, None] * key_in)
    head = ops.key_head(level, graph.space)(torch.stack([head_input(o) for o in outs]))
    out_keys = nn.functional.normalize(base + head, dim=-1)
    return [(o, k, m.float()) for o, k, (_, m) in zip(outs, out_keys, got)]


def aggregate(graph: SpaceGraph, ops: WriteOps, leaves: Sequence[tuple[Tensor, Tensor, Tensor]],
              *, device='cpu', autocast=None) -> list[list[tuple[Tensor, Tensor, Tensor]]]:
    """Every level of ``graph`` from the leaves' (value, unit key, mass), with gradients
    when enabled: per level the (value, key, mass) of each row (level 0 the leaves)."""
    autocast = autocast or (lambda: torch.autocast('cpu', enabled=False))
    device = torch.device(device)
    out = [list(leaves)]
    for n, level in enumerate(graph.levels, 1):
        out.append([combine(ops, graph, n, j, [out[-1][k] for k in ins], device, autocast)
                    for j, ins in enumerate(level.inputs)])
    return out


def to_device(t: Tensor, device) -> Tensor:
    """A host tensor on ``device`` without waiting for the device's queue (pinned,
    non-blocking copy): index tensors built on the host must not synchronize."""
    device = torch.device(device)
    if device.type == 'cuda' and t.device.type == 'cpu':
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def gather_to(tensors: Sequence[Tensor], device) -> list[Tensor]:
    """``[t.to(device) for t in tensors]`` with one copy for all tensors not yet there
    (concatenated along the first dimension, or stacked when 0-d); tensors listed twice are
    copied once. Differentiable like ``.to``."""
    device = torch.device(device)
    out = list(tensors)
    away: dict[int, int] = {}
    for n, t in enumerate(tensors):
        if t.device != device and id(t) not in away:
            away[id(t)] = n
    if not away:
        return out
    firsts = list(away.values())
    moving = [tensors[n] for n in firsts]
    if all(t.dim() == 0 for t in moving):
        moved = list(torch.stack(moving).to(device).unbind(0))
    else:
        flat = [t.reshape(-1, *t.shape[1:]) if t.dim() else t.reshape(1) for t in moving]
        same = len({f.shape[1:] for f in flat}) == 1 and len({f.dtype for f in flat}) == 1
        if not same:
            moved = [t.to(device) for t in moving]
        else:
            joined = torch.cat(flat).to(device)
            moved = [piece.reshape(t.shape) for piece, t in
                     zip(torch.split(joined, [f.shape[0] for f in flat]), moving)]
    by_id = {id(t): m for t, m in zip(moving, moved)}
    return [by_id.get(id(t), t) if t.device != device else t for t in out]


def segment_index(lengths: Sequence[int], device) -> Tensor:
    """The segment number of every row of ``lengths``-sized consecutive segments."""
    return to_device(torch.repeat_interleave(torch.arange(len(lengths)),
                                             torch.tensor(list(lengths))), device)


def default_leaf_key(space: str, item_id: str, value: Tensor, raw: Tensor) -> Tensor:
    """A leaf's unit key: its (live) key."""
    return nn.functional.normalize(raw.float(), dim=-1)


def _default_many(space, ids, values, raws):
    return nn.functional.normalize(torch.stack([r.float() for r in raws]), dim=-1)


default_leaf_key.many = _default_many


def head_leaf_key(heads, corrections: Callable | None = None,
                  corrections_many: Callable | None = None) -> Callable:
    """Leaf keys from content: the item-key head of the value (unnormalized mean) plus a
    free per-item correction (``corrections(space, item_id, raw)``; default the item's
    live key, which then holds the correction), normalized. The heads learn to place a
    corpus; the correction holds item-specific placement. ``key.many(space, ids, values,
    raws)`` computes many leaves' keys in one head call (``corrections_many(space, ids,
    raws)`` gives their corrections stacked)."""
    def key(space: str, item_id: str, value: Tensor, raw: Tensor) -> Tensor:
        head = heads.item[space](value.float()).mean(-2)
        corr = raw if corrections is None else corrections(space, item_id, raw)
        return nn.functional.normalize(head + corr.to(head.device).float(), dim=-1)

    def many(space: str, ids: Sequence[str], values: Sequence[Tensor], raws) -> Tensor:
        lengths = [int(v.shape[0]) for v in values]
        out = heads.item[space](torch.cat([v.float() for v in values]))
        seg = segment_index(lengths, out.device)
        heads_mean = torch.zeros(len(values), out.shape[1], device=out.device,
                                 dtype=out.dtype).index_add(0, seg, out)
        heads_mean = heads_mean / to_device(torch.tensor(lengths, dtype=out.dtype),
                                            out.device)[:, None]
        if corrections_many is not None:
            corr = corrections_many(space, ids, raws)
        elif corrections is not None:
            corr = torch.stack([corrections(space, i, r) for i, r in zip(ids, raws)])
        else:
            corr = torch.stack([r.float() for r in raws])
        return nn.functional.normalize(heads_mean + corr.to(out.device).float(), dim=-1)
    key.many = many
    return key


def leaf_keys(fn: Callable, space: str, ids: Sequence[str], values: Sequence[Tensor],
              raws: Sequence[Tensor]) -> list[Tensor]:
    """Unit keys of many leaves: ``fn.many`` in one call when it has one, else per leaf."""
    if not len(ids):
        return []
    if hasattr(fn, 'many'):
        return list(fn.many(space, ids, values, raws).float().unbind(0))
    return [fn(space, i, v, r).float() for i, v, r in zip(ids, values, raws)]


# -- the stack view -----------------------------------------------------------------------
class SuperposedKB:
    """The rows of one KB (all its spaces) as combiner outputs over its leaves.

    ``kb`` holds the leaves; ``rows[space]`` = (row ids, row keys) are the anchors (the
    stored rows of a rows KB); ``rebuild`` redraws every input's candidate rows (and,
    with ``reanchor``, moves the anchors to the rows' current keys). A row's value, key
    and mass come from ``combine`` with dynamic shares. ``values_fn(space, ids)`` gives
    leaf (value, raw key) pairs without gradient (default: the KB's live, or stored,
    values and keys); ``leaf_key(space, id, value, raw)`` makes a leaf's unit key
    (default: the raw key; ``head_leaf_key``: content head plus correction); ``autocast``
    wraps every aggregator call."""

    def __init__(self, kb: KnowledgeBase, ops: WriteOps, config: SuperposeConfig,
                 rows: Mapping[str, tuple[Sequence[str], Tensor]], *,
                 values_fn: Callable | None = None, leaf_key: Callable | None = None,
                 device='cpu', autocast=None, live: bool | None = None):
        if config.depth < 1:
            raise ValueError('a superposed view needs depth >= 1 (depth 0 is the rows KB)')
        self.kb, self.ops, self.config = kb, ops, config
        self.rows = {s: (list(r[0]), nn.functional.normalize(r[1].detach().float().cpu(), dim=-1))
                     for s, r in rows.items()}
        self.row_lengths = {s: list(r[2]) for s, r in rows.items() if len(r) > 2}
        self.values_fn = values_fn
        self.leaf_key = leaf_key or default_leaf_key
        self.device = torch.device(device)
        self.autocast = autocast or (lambda: torch.autocast('cpu', enabled=False))
        self.live = live
        self.graphs: dict[str, SpaceGraph] = {}
        self.top_keys: dict[str, Tensor] = {}
        self._cache: dict[tuple[str, int, int], tuple[tuple[Tensor, Tensor, Tensor], int]] = {}
        self._src: dict[tuple[str, int], tuple[Tensor, Tensor, Tensor]] = {}
        self._top: dict[tuple[str, int], tuple[Tensor, Tensor, Tensor]] = {}
        self.slots: dict[str, dict[str, int]] = {}
        self.step, self.generation = 0, 0
        self.counters: dict[str, int] = {}
        self.churn: dict[str, float] = {}

    @property
    def dataset(self) -> str:
        return self.kb.dataset

    @property
    def depth(self) -> int:
        return self.config.depth

    # -- construction ----------------------------------------------------------------------
    def _is_live(self, space: str) -> bool:
        if self.live is not None:
            return self.live
        return self.kb.writable and self.kb.is_live(space)

    @torch.no_grad()
    def leaf_info(self, space: str, ids: Sequence[str] | None = None, batch: int = 2048):
        """(ids, unit keys, masses, times, lengths) of the current (or given) leaves; the
        keys by ``leaf_key``."""
        ids = current_ids(self.kb, space) if ids is None else list(ids)
        keys, mass, time, length = [], [], [], []
        live = self._is_live(space)
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            items = self.kb.read(space, chunk, live=live)
            pairs = self.values_fn(space, chunk) if self.values_fn is not None else \
                [(it.values, it.key) for it in items]
            got = leaf_keys(self.leaf_key, space, chunk,
                            [v.float() for v in gather_to([v for v, _ in pairs], self.device)],
                            [r.float() for r in gather_to([r for _, r in pairs], self.device)])
            if got:
                keys += list(torch.stack(got).cpu().unbind(0))
            for item in items:
                mass.append(item.mass)
                time.append(item.time)
                length.append(int(item.values.shape[0]))
        width = self.kb.spaces[space].key_width
        return ids, torch.stack(keys) if keys else torch.zeros(0, width), mass, time, length

    def rebuild(self, step: int = 0, reanchor: bool = True,
                fields: Mapping[str, float] | None = None, rekey: bool = True) -> dict:
        """Redraw every input's candidate rows from the leaves' current keys and the rows'
        keys (with ``reanchor`` the rows' current keys once known, else the given anchors);
        the static shares use the current tau. Clears the caches, refreshes the rows' search
        keys and logs the churn (share of leaves whose candidate rows changed)."""
        old = {s: {g.ids[i]: tuple(g.row_ids[o] for o in cands)
                   for i, cands in enumerate(g.levels[0].candidates)}
               for s, g in self.graphs.items()}
        self.graphs = {}
        for space in self.kb.spaces:
            if space not in self.rows:
                continue
            ids, keys, mass, time, length = self.leaf_info(space)
            row_ids, row_keys = self.rows[space]
            if reanchor and space in self.top_keys and len(self.top_keys[space]) == len(row_ids):
                row_keys = self.top_keys[space]
                self.rows[space] = (row_ids, row_keys)
            g = build_graph(self.dataset, space, ids, keys, mass, time, length, row_ids,
                            row_keys, self.config, self.ops.taus(space, self.config.depth),
                            (fields or {}).get(space), self.row_lengths.get(space))
            self.graphs[space] = g
            if space in old:
                before = old[space]
                changed = [set(before[i]) != {g.row_ids[o] for o in cands}
                           for i, cands in zip(g.ids, g.levels[0].candidates) if i in before]
                self.churn[space] = round(sum(changed) / max(len(changed), 1), 4)
        self.generation += 1
        self.clear()
        self.begin(step)
        if rekey:
            self.rekey()
        return {s: g.stats() for s, g in self.graphs.items()}

    def insert(self, space: str, ids: Sequence[str]) -> dict:
        """Place new or rewritten leaves by their keys: their candidates are their c
        nearest rows; the rows stay, and only the combiners whose fields changed (and the
        rows above them) are recomputed."""
        g = self.graphs[space]
        ids = list(dict.fromkeys(ids))
        _, keys, mass, time, length = self.leaf_info(space, ids)
        lv = g.levels[0]
        fields = [list(zip(ins, sh)) for ins, sh in zip(lv.inputs, lv.shares)]
        candidates = [list(c) for c in lv.candidates]
        rows, changed = [], set()
        for n, item_id in enumerate(ids):
            if item_id in g.index:
                r = g.index[item_id]
                for o in candidates[r]:
                    fields[o] = [(i, s) for i, s in fields[o] if i != r]
                    changed.add(o)
                g.keys[r], g.mass[r], g.time[r], g.length[r] = keys[n], mass[n], time[n], length[n]
            else:
                r = len(g.ids)
                g.ids.append(item_id)
                g.keys = torch.cat([g.keys, keys[n:n + 1]])
                g.mass = np.append(g.mass, mass[n])
                g.time = np.append(g.time, time[n])
                g.length.append(length[n])
                candidates.append([])
            rows.append(r)
        tau = float(self.ops.tau(1, space).detach())
        for r, pairs in zip(rows, memberships(g.keys[rows], g.row_keys, self.config.overlap,
                                              tau)):
            candidates[r] = [o for o, _ in pairs]
            for o, share in pairs:
                fields[o].append((r, share))
                changed.add(o)
        if any(not fld for fld in fields):
            raise ValueError('a row lost its whole field; rebuild instead')
        g.levels[0] = summarize(fields, g.mass, g.time, g.length, candidates)
        touched = {1: changed}
        for n in range(2, g.depth + 1):
            up, below = g.levels[n - 1], g.levels[n - 2]
            hit = {j for j, ins in enumerate(up.inputs) if touched[n - 1] & set(ins)}
            for j in hit:
                ins, sh = up.inputs[j], up.shares[j]
                up.mass[j] = sum(s * below.mass[i] for i, s in zip(ins, sh))
                up.time[j] = max(int(below.time[i]) for i in ins)
                up.count[j] = _count(below.mass, below.count, ins, sh)
            touched[n] = hit
        g.refresh()
        for key in [k for k in self._cache if k[0] == space and k[2] in touched.get(k[1], ())]:
            del self._cache[key]
        for key in [k for k in self._top if k[0] == space and k[1] in touched[g.depth]]:
            del self._top[key]
        for r in rows:
            self._src.pop((space, r), None)
        if touched[g.depth] and space in self.top_keys:
            tops = sorted(touched[g.depth])
            for j, (_, key, _) in zip(tops, self.items(space, g.depth, tops)):
                self.top_keys[space][j] = key.cpu()
        return {'leaves': len(rows), 'rows': {n: sorted(v) for n, v in touched.items()}}

    def refield(self, fields: Mapping[str, float], step: int) -> None:
        """Re-draw level 1's candidates at the given field size per space (keys from the
        current ``leaf_key``, anchors unchanged, no rekey): the write fit's per-step field
        size and the evaluation's sweep."""
        self.clear()
        self.begin(step)
        for space, field in fields.items():
            g = self.graphs[space]
            keys = torch.stack([k for _, k, _ in self._sources(space, range(len(g.ids)))]).cpu()
            self.graphs[space] = build_graph(self.dataset, space, g.ids, keys, g.mass, g.time,
                                             g.length, g.row_ids, g.row_keys, self.config,
                                             self.ops.taus(space, self.config.depth), field,
                                             g.row_lengths)
        self.clear()

    def clear(self) -> None:
        self._cache.clear()
        self._src.clear()
        self._top.clear()

    def begin(self, step: int) -> None:
        """A new training step: leaf values and keys may have changed since the last one."""
        if step != self.step:
            self._src.clear()
            self._top.clear()
            self.step = step

    # -- values and keys -------------------------------------------------------------------------
    @torch.no_grad()
    def _sources(self, space: str, rows: Sequence[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        """(value, unit key, mass) of leaves, without gradient, cached for the step."""
        g = self.graphs[space]
        rows = list(rows)
        todo = [r for r in dict.fromkeys(rows) if (space, r) not in self._src]
        if todo:
            ids = [g.ids[r] for r in todo]
            if self.values_fn is not None:
                pairs = self.values_fn(space, ids)
            else:
                live = self._is_live(space)
                device = self.device if live and self.kb.live_device is not None else None
                pairs = [(it.values, it.key) for it in self.kb.read(space, ids, live=live,
                                                                    device=device)]
            values = [v.float() for v in gather_to([v.detach() for v, _ in pairs], self.device)]
            keys = leaf_keys(self.leaf_key, space, ids, values,
                             [r.float() for r in gather_to([r.detach() for _, r in pairs],
                                                           self.device)])
            masses = to_device(torch.tensor(g.mass[todo], dtype=torch.float),
                               self.device).unbind(0)
            for r, v, key, m in zip(todo, values, keys, masses):
                self._src[(space, r)] = (v, key, m)
        return [self._src[(space, r)] for r in rows]

    def _apply(self, space: str, level: int, rows: Sequence[int], inputs
               ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """The rows of one level from their fields' inputs: one batched aggregator pass
        (``combine_many``), or row by row (``config.batched`` False, the reference)."""
        g = self.graphs[space]
        if self.config.batched:
            return combine_many(self.ops, g, level, rows, inputs, self.device, self.autocast,
                                self.config.max_pairs)
        return [combine(self.ops, g, level, j, ins, self.device, self.autocast)
                for j, ins in zip(rows, inputs)]

    def _fresh(self, key: tuple[str, int, int]) -> bool:
        got = self._cache.get(key)
        return got is not None and self.step - got[1] < max(1, self.config.cache_every)

    @torch.no_grad()
    def items(self, space: str, level: int, rows: Sequence[int]
              ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Level items' (value, key, mass) without gradient; lower levels from the cache.
        The missing rows of a level are computed together, their inputs first."""
        rows = list(rows)
        if level == 0:
            return self._sources(space, rows)
        g = self.graphs[space]
        top = level == g.depth
        todo = [j for j in dict.fromkeys(rows)
                if not ((space, j) in self._top if top else self._fresh((space, level, j)))]
        if todo:
            ins = g.levels[level - 1].inputs
            need = sorted({k for j in todo for k in ins[j]})
            below = dict(zip(need, self._sources(space, need) if level == 1 else
                             self.items(space, level - 1, need)))
            outs = self._apply(space, level, todo, [[below[k] for k in ins[j]] for j in todo])
            self.counters['computed'] = self.counters.get('computed', 0) + len(todo)
            for j, out in zip(todo, outs):
                if top:
                    self._top[(space, j)] = out
                else:
                    self._cache[(space, level, j)] = (out, self.step)
        return [self._top[(space, j)] if top else self._cache[(space, level, j)][0]
                for j in rows]

    def item(self, space: str, level: int, j: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.items(space, level, [j])[0]

    def value(self, space: str, level: int, j: int) -> Tensor:
        return self.item(space, level, j)[0]

    def grad_values(self, space: str, level: int, rows: Sequence[int], leaf: Callable, *,
                    deep: float | None = None, seed: str = ''
                    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Rows' (value, key, mass) recomputed with a graph, all rows of the level in one
        pass. Level-1 inputs are ``leaf(space, ids)`` ((value, key, mass) each, with
        gradients where wanted; one call for the union of the fields); higher inputs are
        the cached items, and a sampled fraction ``deep`` (default ``deep_grad``, drawn
        per (row, input)) is recomputed one level down (once per input, shared by the
        rows that sampled it) and enters as ``cached + (fresh - fresh.detach()) / p``: the
        forward is the cached item exactly, the gradient that of the fresh recomputation."""
        rows = list(rows)
        if not rows:
            return []
        g = self.graphs[space]
        ins = g.levels[level - 1].inputs
        need = sorted({k for j in rows for k in ins[j]})
        if level == 1:
            got = dict(zip(need, leaf(space, [g.ids[k] for k in need])))
            inputs = [[got[k] for k in ins[j]] for j in rows]
        else:
            p = self.config.deep_grad if deep is None else deep
            cached = dict(zip(need, self.items(space, level - 1, need)))
            picks = {}
            for j in dict.fromkeys(rows):
                rnd = random.Random(f'{self.config.seed}:{self.step}:{self.dataset}:{space}:'
                                    f'{level}:{j}:{seed}')
                picks[j] = {k for k in ins[j] if p > 0 and rnd.random() < p}
            deep_rows = sorted(set().union(*picks.values()))
            fresh = dict(zip(deep_rows, self.grad_values(space, level - 1, deep_rows, leaf,
                                                         deep=deep, seed=seed)))
            self.counters['deep'] = self.counters.get('deep', 0) + \
                sum(len(picks[j]) for j in rows)
            inputs = [[tuple(c + (f - f.detach()) / min(p, 1.0)
                             for c, f in zip(cached[k], fresh[k])) if k in picks[j]
                       else cached[k] for k in ins[j]] for j in rows]
        return self._apply(space, level, rows, inputs)

    def grad_value(self, space: str, level: int, j: int, leaf: Callable, *,
                   deep: float | None = None, seed: str = '') -> tuple[Tensor, Tensor, Tensor]:
        return self.grad_values(space, level, [j], leaf, deep=deep, seed=seed)[0]

    # -- rows ------------------------------------------------------------------------------------
    def top_item(self, space: str, item_id: str) -> tuple[Tensor, Tensor, Tensor]:
        g = self.graphs[space]
        return self.item(space, g.depth, g.top_index[item_id])

    def top_value(self, space: str, item_id: str) -> Tensor:
        return self.top_item(space, item_id)[0]

    def top_values(self, space: str) -> list[Tensor]:
        g = self.graphs[space]
        return [v for v, _, _ in self.items(space, g.depth, range(len(g.row_ids)))]

    def top_time(self, space: str, item_id: str) -> int:
        g = self.graphs[space]
        return int(g.top_time[g.top_index[item_id]])

    def top_mass(self, space: str, item_id: str) -> float:
        """A row's mass under the current shares (the stored masses are the last
        reassignment's)."""
        return float(self.top_item(space, item_id)[2])

    @torch.no_grad()
    def rekey(self) -> None:
        """The rows' search keys: their current field keys."""
        for space, g in self.graphs.items():
            width = self.kb.spaces[space].key_width
            keys = [k for _, k, _ in self.items(space, g.depth, range(len(g.row_ids)))]
            self.top_keys[space] = torch.stack(keys).cpu() if keys else torch.zeros(0, width)

    def search(self, space: str, query: Tensor, k: int, query_time: int | None,
               exclude: Sequence[str] = ()) -> list[tuple[float, str]]:
        """Exact top-k over the rows' keys; rows later than ``query_time`` and rows
        holding any excluded leaf are never returned."""
        g = self.graphs.get(space)
        keys = self.top_keys.get(space)
        if g is None or keys is None or not len(g.row_ids):
            return []
        q = nn.functional.normalize(query.detach().float().cpu().reshape(1, -1), dim=-1)
        scores = (q @ nn.functional.normalize(keys, dim=-1).T)[0].double()
        ok = torch.ones(len(g.row_ids), dtype=torch.bool)
        if query_time is not None:
            ok &= torch.from_numpy(g.top_time <= query_time)
        rows = [g.index[i] for i in exclude if i in g.index]
        if rows:
            ok[torch.from_numpy(g.tops_holding(rows))] = False
        if not bool(ok.any()):
            return []
        scores = scores.masked_fill(~ok, -math.inf)
        top = scores.topk(min(k, int(ok.sum())))
        return [(float(s), g.row_ids[j]) for s, j in zip(top.values.tolist(), top.indices.tolist())]

    def positives(self, space: str, leaf_ids: Sequence[str]) -> dict[str, float]:
        """The covering rows (``SpaceGraph.covering``), the ``max_positives`` largest."""
        got = self.graphs[space].covering(leaf_ids)
        return dict(list(got.items())[:self.config.max_positives])

    def gold(self, space: str, leaf_ids: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
        """Per leaf, the row holding most of its mass (in leaf order, unique)."""
        g = self.graphs[space]
        rows = [g.index[i] for i in exclude if i in g.index]
        banned = set(g.tops_holding(rows).tolist()) if rows else set()
        out = []
        for i in leaf_ids:
            j = int(g.holder[g.index[i]]) if i in g.index else -1
            if j >= 0 and j not in banned and g.row_ids[j] not in out:
                out.append(g.row_ids[j])
        return out

    def share_of(self, space: str, item_id: str, leaf_ids: set[str]) -> float:
        """The share of a row's mass coming from the given leaves."""
        g = self.graphs[space]
        j = g.top_index[item_id]
        a, b = g.comp.indptr[j], g.comp.indptr[j + 1]
        return float(sum(v for i, v in zip(g.comp.indices[a:b], g.comp.data[a:b])
                         if g.ids[i] in leaf_ids))

    def slot(self, space: str) -> dict[str, int]:
        """A growing index of row ids (usage statistics)."""
        slots = self.slots.setdefault(space, {})
        for i in self.graphs[space].row_ids:
            slots.setdefault(i, len(slots))
        return slots

    # -- metrics and export ----------------------------------------------------------------------
    @torch.no_grad()
    def write_stats(self, space: str) -> dict:
        """The write side now: mean entropy of the leaves' dynamic shares over their
        candidate rows, the rows' load (mass) distribution, the last reassignment's churn,
        and tau per level."""
        from schnitz.kb_eval import distribution
        g = self.graphs[space]
        lv = g.levels[0]
        tau = self.ops.tau(1, space)
        entropy = 0.0
        if lv.candidates:
            keys = torch.stack([k for _, k, _ in self._sources(space, range(len(g.ids)))])
            p, _ = candidate_shares(keys, g.row_keys, lv.candidates, tau)
            entropy = float((-(p * p.clamp_min(1e-30).log()).sum(1)).mean())
        loads = torch.stack([m for _, _, m in self.items(space, g.depth,
                                                         range(len(g.row_ids)))]).tolist()
        return {'share_entropy': round(entropy, 4),
                'row_load': distribution(loads), 'churn': self.churn.get(space),
                'tau': [round(t, 3) for t in self.ops.taus(space, g.depth)]}

    def metrics(self, reads: Mapping[str, list] | None = None) -> dict:
        """Superposition metrics (``kb_eval.superposition_report``), per-level stats and
        the write side's shares, loads and churn."""
        from schnitz.kb_eval import superposition_report
        out = {}
        for space, g in self.graphs.items():
            rep = superposition_report(g.composition(), (reads or {}).get(space, ()))
            out[space] = {**g.stats(), **self.write_stats(space),
                          'sources_per_item': rep['sources_per_item'],
                          'effective_sources_per_item': rep['effective_sources_per_item'],
                          'items_per_source': rep['items_per_source'],
                          'effective_items_per_read': rep['effective_items_per_read']['entropy']}
        return out

    @torch.no_grad()
    def refresh_shares(self, space: str) -> None:
        """Set every level's stored shares, masses and counts to the current dynamic ones
        (per input normalized over its candidates: invariant 7), so the lineage, masses
        and composition describe the rows as they are computed now."""
        g = self.graphs[space]
        for n, lv in enumerate(g.levels, 1):
            below = self._sources(space, range(len(g.ids))) if n == 1 else \
                self.items(space, n - 1, range(len(g.row_ids)))
            tau = self.ops.tau(n, space)
            anchors = g.row_keys.to(self.device)
            fields: list[list[tuple[int, float]]] = [[] for _ in lv.inputs]
            shares, _ = candidate_shares(torch.stack([x[1] for x in below]), anchors,
                                         lv.candidates, tau)
            for i, (cands, p) in enumerate(zip(lv.candidates, shares.tolist())):
                for o, share in zip(cands, p):
                    fields[o].append((i, share))
            mass = torch.stack([x[2] for x in below]).double().cpu().numpy()
            time = g.level_time(n - 1)
            length = g.level_length(n - 1)
            g.levels[n - 1] = summarize(fields, mass, time, length, lv.candidates)
        g.refresh()

    @torch.no_grad()
    def export(self, dest: str | Path, *, step: int = 0, name: str | None = None,
               extra: Mapping | None = None) -> KnowledgeBase:
        """The rows as a frozen KB at ``dest``: the leaves (current values, keys and
        provenance), then each level as one ``rewrite`` with the current dynamic shares
        (``refresh_shares``: lineage, masses, times) and the field keys, exported
        (``export_live``) with ``superpose.json``. Every level is computed fresh. The
        current items are the rows (their ids)."""
        dest = Path(dest)
        build = dest.with_name(dest.name + '.build')
        shutil.rmtree(build, ignore_errors=True)
        self.clear()
        for space in self.graphs:
            self.refresh_shares(space)
        self.clear()
        out = KnowledgeBase.create(build, name=name or f'{self.kb.name}@superposed',
                                   dataset=self.dataset, spaces=self.kb.spaces,
                                   origin={'command': 'superpose.export', 'kb': self.kb.name,
                                           'step': step, 'depth': self.config.depth})
        try:
            for space, g in self.graphs.items():
                if not g.ids:
                    continue
                live = self._is_live(space)
                leaves = self._sources(space, range(len(g.ids)))
                stored = self.kb.read(space, g.ids, live=live)
                out.append(space, [NewItem(v.cpu(), k.float().cpu(), Provenance(
                    it.provenance.sources, 'live-update' if live else it.provenance.producer,
                    step), float(g.mass[n]), int(g.time[n]), g.ids[n])
                    for n, ((v, k, _), it) in enumerate(zip(leaves, stored))])
                below = g.ids
                for n, level in enumerate(g.levels, 1):
                    got = self.items(space, n, range(len(level.inputs)))
                    items = [NewItem(v.cpu(), k.float().cpu(), Provenance((), 'rewrite', step),
                                     float(level.mass[j]), int(level.time[j]),
                                     g.level_ids[n][j])
                             for j, (v, k, _) in enumerate(got)]
                    # sparse: each input has only its c candidate rows
                    out.rewrite(space, below, items, shares=level.matrix(len(below)))
                    below = g.level_ids[n]
            out.close()
            scratch = KnowledgeBase(build, writable=True)
            try:
                frozen = scratch.export_live(dest, name=name or f'{self.kb.name}@superposed')
            finally:
                scratch.close()
        finally:
            shutil.rmtree(build, ignore_errors=True)
        meta = {'depth': self.config.depth, 'config': dataclasses.asdict(self.config),
                'dataset': self.dataset, 'step': step, **(extra or {}),
                'spaces': {s: g.to_json() for s, g in self.graphs.items()}}
        (dest / 'superpose.json').write_text(json.dumps(meta) + '\n')
        return frozen


def rows_of(kb: KnowledgeBase, live: bool = False
            ) -> dict[str, tuple[list[str], Tensor, list[int]]]:
    """(ids, keys, position counts) of the current items of every space of a rows KB (the
    anchors; the stack's rows keep the stored rows' lengths)."""
    out = {}
    for space in kb.spaces:
        ids = current_ids(kb, space)
        if ids:
            items = kb.read(space, ids, live=live)
            out[space] = (ids, torch.stack([it.key.float() for it in items]),
                          [int(it.values.shape[0]) for it in items])
    return out


# -- the step cache ------------------------------------------------------------------------
class SuperposedCache(ItemCache):
    """The read path's item cache over superposed views (one per dataset). Rows' values and
    keys are leaf tensors for a step (their gradients from all reads accumulate);
    ``backward`` recomputes every row that received a gradient with its graph
    (``SuperposedKB.grad_value``) and backpropagates into the leaves (this cache's live
    leaf values and raw keys, or ``producer.values`` in L1b), the aggregators, their key
    heads and tau. ``apply`` then makes the sparse live update of the touched leaves."""

    superposed = True

    def __init__(self, views: Mapping[str, SuperposedKB], device='cpu', train: bool = True,
                 step: int = 0, producer=None):
        super().__init__(device, train)
        self.views, self.step, self.producer = dict(views), step, producer
        self.tops: dict[tuple[str, str, str], Tensor] = {}
        self.top_keys: dict[tuple[str, str, str], Tensor] = {}
        self.top_masses: dict[tuple[str, str, str], float] = {}
        for view in self.views.values():
            view.begin(step)

    def _view(self, dataset: str) -> SuperposedKB:
        if dataset not in self.views:
            raise KeyError(f'no superposed view of {dataset!r}')
        return self.views[dataset]

    def sources(self, kb: KnowledgeBase, space: str, ids: Sequence[str]):
        return ItemCache.get(self, kb, space, ids)

    def _rows(self, kb: KnowledgeBase, space: str, ids: Sequence[str]
              ) -> list[tuple[str, str, str]]:
        """The rows as this step's leaves; the missing ones computed together."""
        missing = [i for i in dict.fromkeys(ids) if (kb.dataset, space, i) not in self.tops]
        if missing:
            view = self._view(kb.dataset)
            g = view.graphs[space]
            got = view.items(space, g.depth, [g.top_index[i] for i in missing])
            masses = torch.stack([m for _, _, m in got]).tolist()
            for item_id, (value, key, _), m in zip(missing, got, masses):
                ref = (kb.dataset, space, item_id)
                self.tops[ref] = value.to(self.device).detach().clone() \
                    .requires_grad_(self.train)
                self.top_keys[ref] = key.to(self.device).detach().clone() \
                    .requires_grad_(self.train)
                self.top_masses[ref] = m
        return [(kb.dataset, space, i) for i in ids]

    def get(self, kb: KnowledgeBase, space: str, ids: Sequence[str]):
        view = self._view(kb.dataset)
        return [(self.tops[ref], view.top_time(space, ref[2]))
                for ref in self._rows(kb, space, ids)]

    def keys(self, kb: KnowledgeBase, space: str, ids: Sequence[str]) -> list[Tensor]:
        return [self.top_keys[ref] for ref in self._rows(kb, space, ids)]

    def mass(self, dataset: str, space: str, item_id: str) -> float:
        ref = (dataset, space, item_id)
        if ref in self.top_masses:
            return self.top_masses[ref]
        return self._view(dataset).top_mass(space, item_id)

    def search(self, kbs: Sequence[KnowledgeBase], allowed, space: str, query: Tensor, k: int,
               query_time: int, banned) -> list[Ref]:
        allowed = set(allowed)
        pool = []
        for kb in kbs:
            if kb.dataset not in allowed:
                raise PermissionError(f'not authorized to read {kb.dataset!r}')
            exclude = [i for d, i in banned if d == kb.dataset]
            pool += [(s, kb.dataset, i) for s, i in
                     self._view(kb.dataset).search(space, query, k, query_time, exclude)]
        pool.sort(key=lambda x: -x[0])
        return [(d, i) for _, d, i in pool[:k]]

    def positives(self, space: str, wanted: Sequence[Ref]) -> dict[Ref, float]:
        out: dict[Ref, float] = {}
        for dataset in dict.fromkeys(d for d, _ in wanted):
            got = self._view(dataset).positives(space, [i for d, i in wanted if d == dataset])
            out.update({(dataset, i): w for i, w in got.items()})
        return out

    def gold(self, space: str, refs: Sequence[Ref], banned=frozenset()) -> list[Ref]:
        out = []
        for dataset in dict.fromkeys(d for d, _ in refs):
            exclude = [i for d, i in banned if d == dataset]
            out += [(dataset, i) for i in self._view(dataset).gold(
                space, [i for d, i in refs if d == dataset], exclude)]
        return out

    def rows(self, dataset: str, space: str) -> dict[str, int]:
        return self._view(dataset).slot(space)

    def share_of(self, dataset: str, space: str, item_id: str, leaves: set[str]) -> float:
        return self._view(dataset).share_of(space, item_id, leaves)

    def leaf_fn(self, dataset: str, kb: KnowledgeBase):
        """The leaves' (value, key, mass) with gradients: this cache's live leaves (values,
        and raw keys as leaves through the view's ``leaf_key``), or in L1b the producers'
        recomputed values."""
        view = self._view(dataset)

        def leaf(space: str, ids: Sequence[str]) -> list[tuple[Tensor, Tensor, Tensor]]:
            fresh = [None] * len(ids)
            if self.producer is not None:
                fresh = self.producer.values(space, [(dataset, i) for i in ids])
            values = [v if f is None else f
                      for (v, _), f in zip(self.sources(kb, space, ids), fresh)]
            raws = ItemCache.keys(self, kb, space, ids)
            g = view.graphs[space]
            keys = leaf_keys(view.leaf_key, space, ids, values, raws)
            masses = to_device(torch.tensor([float(g.mass[g.index[i]]) for i in ids]),
                               values[0].device).unbind(0) if ids else []
            return list(zip(values, keys, masses))
        return leaf

    def backward(self, balance: float = 0.0, usage: dict | None = None) -> dict:
        """Recompute the rows that received gradients and backpropagate. With
        ``balance`` > 0 also the write side's balance loss on those rows' loads (their
        masses under the dynamic shares, differentiable in the leaves' keys and tau)
        against a moving average of each row's load (``usage['write/<kb>/<space>']``,
        ``losses.balance_loss``): mass moving onto already loaded rows costs."""
        from schnitz.kb.losses import UsageEMA, balance_loss
        if not self.train:
            raise ValueError('an evaluation cache does not backpropagate')
        done, drift = 0, 0.0
        balances = []
        for view in self.views.values():
            view.counters = {}
        groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for ref, value in self.tops.items():
            if value.grad is not None or self.top_keys[ref].grad is not None:
                groups.setdefault(ref[:2], []).append(ref)
        tensors, grads = [], []
        for (dataset, space), refs in groups.items():   # all touched rows of a space at once
            view = self.views[dataset]
            g = view.graphs[space]
            got = view.grad_values(space, g.depth, [g.top_index[r[2]] for r in refs],
                                   self.leaf_fn(dataset, view.kb))
            if balance > 0 and usage is not None:
                name = f'write/{dataset}/{space}'
                ema = usage.setdefault(name, UsageEMA(len(g.row_ids)))
                ema.grow(len(g.row_ids))
                rows = torch.tensor([g.top_index[r[2]] for r in refs])
                loads = torch.stack([m for _, _, m in got])
                b = balance_loss(rows, loads, ema)
                ema.update(rows, loads.detach())
                tensors.append(balance * b)
                grads.append(torch.ones_like(b))
                balances.append(float(b.detach()))
            for ref, (out, out_key, _) in zip(refs, got):
                value, key = self.tops[ref], self.top_keys[ref]
                drift = max(drift, float((out.detach().to(value.device)
                                          - value.detach()).abs().max()))
                for fresh, leaf in ((out, value), (out_key, key)):
                    if leaf.grad is not None:
                        tensors.append(fresh)
                        grads.append(leaf.grad.to(fresh.device))
                done += 1
        if tensors:
            torch.autograd.backward(tensors, grads)
        deep = sum(v.counters.get('deep', 0) for v in self.views.values())
        out = {'rows': done, 'deep': deep, 'drift': drift}
        if balances:
            out['write_balance'] = sum(balances) / len(balances)
        return out

    def touched_tops(self) -> list[tuple[str, str, str]]:
        return [ref for ref, leaf in self.tops.items() if leaf.grad is not None]


# -- the stack fit and consolidation ----------------------------------------------------------
def fit_losses(view: SuperposedKB, space: str, row_ids: Sequence[str],
               targets: Sequence[Tensor], target_keys: Sequence[Tensor] | None,
               leaf: Callable, weights=(1.0, 1.0, 1.0)) -> tuple[Tensor, dict, Tensor]:
    """The write stack fitted to row targets (L2's stack objective): each row's combiner
    output and field key, recomputed through every level from the leaves (``leaf``; full
    gradient, dynamic shares), against the row's learned value and key: 1 - cosine and
    relative MSE of the values, 1 - cosine of the keys. The read side is cut off at the
    rows' key/value pairs (no decoder). Returns the mean loss, its parts and the rows'
    masses (loads, differentiable: the balance loss's input)."""
    from schnitz.kb.producer import item_losses
    g = view.graphs[space]
    parts: dict[str, list[Tensor]] = {'cos': [], 'mse': [], 'key': []}
    loads = []
    got_rows = view.grad_values(space, g.depth, [g.top_index[r] for r in row_ids], leaf,
                                deep=1.0, seed='fit')
    for n, ((out, key, mass), target) in enumerate(zip(got_rows, targets)):
        got = item_losses({space: out}, {space: target.to(out.device)})
        parts['cos'].append(got[f'cos_{space}'])
        parts['mse'].append(got[f'mse_{space}'])
        if target_keys is not None:
            parts['key'].append(1 - nn.functional.cosine_similarity(
                key, target_keys[n].to(key.device).float(), dim=-1))
        loads.append(mass)
    means = {k: torch.stack(v).mean() for k, v in parts.items() if v}
    loss = weights[0] * means['cos'] + weights[1] * means['mse']
    if 'key' in means:
        loss = loss + weights[2] * means['key']
    return loss, {k: round(v.item(), 5) for k, v in means.items()}, torch.stack(loads)


def consolidate(views: Mapping[str, SuperposedKB], refs: Sequence[tuple[str, str, str]],
                before: Mapping[tuple[str, str, str], Tensor], *, steps: int, item_lr: float,
                device='cpu') -> dict:
    """After an aggregator update: re-fit the leaves under the touched rows so the rows
    return to ``before`` (1 - cosine plus relative MSE, L2's reproduction loss), every
    level recomputed fresh with the full gradient (p = 1). Returns the mean relative
    distance of the rows to ``before`` after the update and after re-fitting."""
    from schnitz.kb.producer import item_losses

    def distance() -> float:
        rel = []
        for dataset, space, item_id in refs:
            view = views[dataset]
            view.clear()
            now = view.top_value(space, item_id)
            old = before[(dataset, space, item_id)].to(now.device)
            rel.append(float((now - old).norm() / old.norm().clamp_min(1e-12)))
        return sum(rel) / max(len(rel), 1)

    if not refs:
        return {'items': 0}
    moved = distance()
    for _ in range(steps):
        cache = SuperposedCache(views, device, train=True, step=-1)
        total = None
        groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for ref in refs:
            groups.setdefault(ref[:2], []).append(ref)
        for view in views.values():
            view.clear()
        for (dataset, space), members in groups.items():
            view = views[dataset]
            g = view.graphs[space]
            got = view.grad_values(space, g.depth, [g.top_index[r[2]] for r in members],
                                   cache.leaf_fn(dataset, view.kb), deep=1.0, seed='refit')
            for ref, (out, _, _) in zip(members, got):
                parts = item_losses({space: out}, {space: before[ref]})
                loss = parts[f'cos_{space}'] + parts[f'mse_{space}']
                total = loss if total is None else total + loss
        (total / len(refs)).backward()
        cache.apply(item_lr)
    refit = distance() if steps else moved
    for view in views.values():
        view.clear()
    return {'items': len(refs), 'moved_rel': round(moved, 6), 'refit_rel': round(refit, 6)}
