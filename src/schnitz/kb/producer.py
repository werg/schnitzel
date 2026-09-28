"""The producer path of KB items, shared by every stage (docs/knowledge-base-stack.md,
5.1 steps 6-9, and 5.2 "Gradients into producers"): source -> writer span -> forward
codecs -> items per space (-> S_s rewrites for items that came from rewrites).

One implementation for bank creation's codec step (``l1 build``), in-context writes
(L1 ``--writes``, B9 rounds), L1b's and B9's selective producer replay and L2's
reproduction objective:

- ``Writer``: the frozen writer (``schnitz.kb.decoder.Model``: prefix, write, free
  run, span heads) with the stack's codecs and the decoder's input embeddings. Writes
  in place (``generate``: free runs from a causal prefix, batched) and the codecs'
  items of a span (``encode``).
- ``write_spans``: the writer's spans with gradients into its span heads, in one of
  three feeds. ``teacher``: a given span (the bank's cached span) is fed and each rep
  predicted from the ones before it (one pass). ``self``: the writer first free-runs
  without gradients, then one gradient pass fed with those reps (each predicted rep
  equals the free-running one; the gradient reaches one step back). ``free``: the free
  run itself with gradients (``free_run_grad``: every rep fed back, the actual
  producer forward; per-step checkpointing).
- ``ste_round``: a value at its serialized precision in the forward (bf16: the span
  cache, the store), the identity in the backward.
- ``WriteLog``: the sources of in-context writes (prefix token ids, ``<|mem|>``
  positions, the read spans spliced there, ratio, the generated span, the writer
  batch it was generated in; in memory also the written items and, for B9, how each
  read can be recomputed), so a written item can be replayed.
- ``Producers``: the selective producer replay (L1b, B9 ``--backprop-rounds``, L2).
  The items a read retrieves are recomputed from their stored sources, at the stored
  forward's serialized precision and batch composition, so at initialization the
  recomputation equals the stored payload bit for bit (``match_exact``); the
  gradients of all reads accumulate on the recomputed items, then each replay unit
  is recomputed with gradients and backpropagated into the writer's span heads and
  the codecs before the optimizer step (invariant 3). Written items whose write read
  earlier written items chain further back (``depth``, truncated).
- ``rewrite_item``: S_s over the produced inputs of a rewrite output, each input at
  gate share x mass (numerator and mass, invariants 5 and 7), conditioned on the
  output's key.
- ``item_losses`` / ``key_loss`` / ``functional_loss`` / ``l2_loss``: the L2
  objective. Values per space by cosine per position and MSE relative to the target's
  mean square; keys by cosine of the item-key heads' key of the produced values to
  the target's key; functionally, the frozen decoder's reading of R(produced items)
  against R(target items) (KL on the source's reconstruction).

Training-only: inference reads stored payloads and never re-encodes sources
(invariant 1).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from schnitz.kb.decoder import LEVELS, length_factors
from schnitz.kb.stack import SPACES

FEEDS = ('teacher', 'self', 'free')


# -- lengths and precision ----------------------------------------------------------------
def level_index(level: str | int) -> int:
    return LEVELS.index(level) if isinstance(level, str) else int(level)


def span_length(tokens: int, level: str | int) -> tuple[float, int]:
    """(factor, reps) of a write at a ratio level: the B1 length schedule, as
    ``schnitz.kb.bank.write_spans`` uses it."""
    factor = length_factors(int(tokens))[level_index(level)]
    return factor, max(1, math.ceil(int(tokens) / factor))


def ste_round(x: Tensor, dtype=torch.bfloat16) -> Tensor:
    """``x`` at the serialized precision in the forward (what the store or span cache
    holds), the identity in the backward."""
    # rounded + (x - x): exactly the rounded value (x + (rounded - x) is not, in float)
    return x.detach().to(dtype).to(x.dtype) + (x - x.detach())


def _scope(model):
    core = getattr(model, 'core', None)
    return core.autocast() if core is not None else contextlib.nullcontext()


# -- the writer's forward -------------------------------------------------------------------
def prefix_for(model, examples: Sequence[dict]):
    """``Model.prefix`` (the write site's cached no-gradient prefix), or, when gradients
    are enabled and an example's ``inputs`` carry them (B9's chain through an earlier
    write's reads), the same computation with gradients (``Model.prefix`` is a
    ``torch.no_grad`` method; its undecorated function computes the same forward)."""
    wants = torch.is_grad_enabled() and any(
        torch.is_tensor(ex.get('inputs')) and ex['inputs'].requires_grad for ex in examples)
    plain = getattr(model.prefix, '__wrapped__', None)   # a bound method forwards it
    if wants and plain is not None:
        return plain(model, list(examples))
    return model.prefix(list(examples))


def free_run_grad(model, examples, lengths: list[int]) -> list[Tensor]:
    """``Model.free_run`` with gradients (the same operations, no stop decisions): each
    rep is the writer's rep head on the last state of the span so far, fed back."""
    heads = model.writer
    width = model.decoder.embed_tokens.weight.shape[1]
    reps = [torch.zeros(0, width, device=model.device) for _ in examples]
    prefix = prefix_for(model, examples)

    def last(*fed):
        return tuple(h[-1:] for h in model.write(examples, list(fed), prefix))
    for step in range(max(lengths)):
        if torch.is_grad_enabled():
            # each step recomputed in backward: activations O(span) instead of O(span^2)
            hidden = checkpoint(last, *reps, use_reentrant=False)
        else:
            hidden = last(*reps)
        for i, h in enumerate(hidden):
            if step < lengths[i]:
                reps[i] = torch.cat([reps[i], heads.rep(h)])
    return reps


def write_spans(model, examples: Sequence[dict], counts: Sequence[int], feed: str = 'teacher',
                teacher: Sequence[Tensor] | None = None) -> list[Tensor]:
    """Writer spans of ``examples`` (``Model.write`` examples: ``ids`` + ``prompt`` or
    ``inputs``, and ``factor``) with gradients into the writer's span heads when
    enabled; ``counts[i]`` reps each. ``feed`` (``FEEDS``): 'teacher' (``teacher[i]``
    fed, (counts[i], width)), 'self' (the writer's own free-running reps, computed
    without gradients, fed in one gradient pass) or 'free' (the free run replayed with
    gradients through every step)."""
    if feed not in FEEDS:
        raise ValueError(f'unknown feed {feed!r} (one of {FEEDS})')
    examples = list(examples)
    with _scope(model):
        if feed == 'free':
            return [r.float() for r in free_run_grad(model, examples, list(counts))]
        if feed == 'self':
            with torch.no_grad():
                own, _ = model.free_run(examples, list(counts))
            fed = [o.detach().float() for o in own]
        else:
            if teacher is None:
                raise ValueError('teacher feed needs the spans to feed')
            fed = [t.to(model.device).float() for t in teacher]
            for t, n in zip(fed, counts):
                if t.shape[0] != n:
                    raise ValueError(f'teacher span has {t.shape[0]} reps, expected {n}')
        states = model.write(examples, fed, prefix_for(model, examples))
        return [model.writer.rep(h[:-1]).float() for h in states]


def produce_items(stack, span: Tensor) -> dict[str, Tensor]:
    """The codecs' item of every space for one span (the stack's ``encode``)."""
    return {s: v.float() for s, v in stack.encode(span).items()}


class Writer:
    """The writer (``Model``: prefix, write, free run, span heads) with the stack's
    codecs and the decoder's input embeddings (``embed``: token ids -> float rows; only
    in-context writes need it): writes in place and the producers' forward."""

    def __init__(self, model, stack, embed: Callable[[Tensor], Tensor] | None = None,
                 batch: int = 8):
        self.model, self.stack, self.embed, self.batch = model, stack, embed, batch

    def autocast(self):
        return _scope(self.model)

    def inputs(self, prefix_ids: Tensor, mems: Sequence[int], reads: Sequence[Tensor]) -> Tensor:
        """A write's input embeddings: the prefix with its reads spliced in."""
        from schnitz.kb.read import splice
        if self.embed is None:
            raise ValueError('in-context writes need the decoder\'s input embeddings')
        x, _ = splice(self.embed(prefix_ids), list(mems), list(reads))
        return x

    @torch.no_grad()
    def generate(self, examples: Sequence[dict], counts: Sequence[int]) -> list[Tensor]:
        """Each example's span (``inputs`` + ``factor``, or ``ids`` + ``prompt``),
        free-running for ``counts[i]`` reps, ``self.batch`` at a time: exactly what the
        writer stores (the batch composition is part of the forward on the GPU)."""
        out = []
        for start in range(0, len(examples), self.batch):
            chunk = list(examples[start:start + self.batch])
            with self.autocast():
                reps, _ = self.model.free_run(chunk, list(counts[start:start + self.batch]))
            out += [r.float() for r in reps]
        return out

    def encode(self, span: Tensor) -> dict[str, Tensor]:
        """The codecs' items of a span (as the bank build encodes a record's span)."""
        with self.autocast():
            return produce_items(self.stack, span)


def producer_params(model, stack) -> dict[str, list]:
    """The producers' parameter sets: the codecs and the writer's span heads (marker,
    ratio code, rep head; the stop head gets no gradient from a replay)."""
    return {'codecs': list(stack.codecs.parameters()),
            'writer': [] if model is None else list(model.writer.parameters())}


# -- the sources of in-context writes --------------------------------------------------------
class WriteLog:
    """The sources of in-context writes, for producer replay: per write its prefix
    token ids, ``<|mem|>`` positions, the read spans spliced there, the ratio, the
    generated span (float32, the codecs' exact input) and the sources generated in the
    same writer batch (``group``: the replay unit). ``items`` (the written values per
    space) are kept for verification; ``rereads`` (B9: per read, the query state and per
    space the items read, their sources and gate scales) only in memory. One
    safetensors file plus a JSON index per training step under ``root`` (``None``: in
    memory only); the newest entry of a source wins. ``truncate(step)`` drops steps
    after a checkpoint."""

    def __init__(self, root: Path | None):
        self.root, self.index, self.memory, self.pending = root, {}, {}, {}
        self._handles: dict[str, object] = {}
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            for meta in sorted(root.glob('step-*.json')):
                if meta.with_suffix('.safetensors').exists():
                    for source, entry in json.loads(meta.read_text()).items():
                        self.index[source] = {**entry, 'file': meta.stem}

    def __contains__(self, source: str) -> bool:
        return source in self.index or source in self.pending

    def add(self, source: str, *, kb: str, prefix_ids: Tensor, mems: Sequence[int],
            reads: Sequence[Tensor], factor: float, span: Tensor, step: int,
            group: Sequence[str] | None = None, items: Mapping[str, Tensor] | None = None,
            rereads: list | None = None) -> None:
        """``group``: the sources generated in the same writer batch (replay unit)."""
        if rereads is not None and self.root is not None:
            raise ValueError('read recomputation is kept only in an in-memory write log')
        tensors = {'prefix_ids': prefix_ids.to(torch.int32).cpu(),
                   'span': span.detach().float().cpu(),
                   **{f'read{j}': r.detach().float().cpu() for j, r in enumerate(reads)},
                   **{f'item_{s}': v.detach().float().cpu() for s, v in (items or {}).items()}}
        self.pending[source] = {'kb': kb, 'mems': list(mems), 'factor': factor, 'step': step,
                                'group': list(group or [source]), 'tensors': tensors,
                                'rereads': rereads}

    def flush(self, step: int) -> None:
        """Publish the pending writes as step ``step`` (the count of completed steps)."""
        if not self.pending:
            return
        if self.root is None:
            for source, entry in self.pending.items():
                self.memory[source] = entry
                self.index[source] = {k: v for k, v in entry.items()
                                      if k not in ('tensors', 'rereads')}
            self.pending = {}
            return
        from safetensors.torch import save_file
        name = f'step-{step:08d}'
        tensors, meta = {}, {}
        for n, (source, entry) in enumerate(self.pending.items()):
            for key, value in entry['tensors'].items():
                tensors[f'{n}.{key}'] = value.contiguous()
            meta[source] = {'n': n, 'reads': sum(k.startswith('read') for k in entry['tensors']),
                            'items': sorted(k[5:] for k in entry['tensors'] if k.startswith('item_')),
                            **{k: v for k, v in entry.items() if k not in ('tensors', 'rereads')}}
        save_file(tensors, str(self.root / f'{name}.safetensors'))
        (self.root / f'{name}.json').write_text(json.dumps(meta) + '\n')
        for source, entry in meta.items():
            self.index[source] = {**entry, 'file': name}
        self.pending = {}

    def get(self, source: str) -> dict:
        entry = self.index[source]
        rereads = None
        if self.root is None:
            tensors = self.memory[source]['tensors']
            rereads = self.memory[source].get('rereads')
        else:
            from safetensors import safe_open
            handle = self._handles.get(entry['file'])
            if handle is None:
                handle = safe_open(str(self.root / f'{entry["file"]}.safetensors'),
                                   framework='pt')
                self._handles[entry['file']] = handle
            names = ['prefix_ids', 'span'] + [f'read{j}' for j in range(entry['reads'])] + \
                [f'item_{s}' for s in entry.get('items', ())]
            tensors = {key: handle.get_tensor(f'{entry["n"]}.{key}') for key in names}
        return {'prefix_ids': tensors['prefix_ids'].long(), 'span': tensors['span'],
                'mems': entry['mems'], 'factor': entry['factor'],
                'reads': [tensors[f'read{j}'] for j in range(len(entry['mems']))],
                'items': {k[5:]: v for k, v in tensors.items() if k.startswith('item_')},
                'rereads': rereads}

    def truncate(self, step: int) -> None:
        """Forget the writes of steps after ``step`` (their KB commits were discarded)."""
        self.pending = {}
        if self.root is None:
            return
        self._handles = {}
        for meta in sorted(self.root.glob('step-*.json')):
            if int(meta.stem.split('-')[1]) > step:
                meta.with_suffix('.safetensors').unlink(missing_ok=True)
                meta.unlink()
        self.index = {}
        for meta in sorted(self.root.glob('step-*.json')):
            for source, entry in json.loads(meta.read_text()).items():
                self.index[source] = {**entry, 'file': meta.stem}


# -- selective producer replay ------------------------------------------------------------
Key = tuple[str, str, str]      # (producer kind 'codec' | 'write', dataset, source id)


class Producers:
    """The items of stored sources recomputed by their producers (L1b, B9
    ``--backprop-rounds``, L2).

    Sources (``Key``):

    - ``('codec', dataset, record)``: a bank item. The writer's span of the record
      under the memory prompt at the bank's ratio ``level`` (count: the span cache's),
      rounded to the span cache's bf16, then the codecs.
    - ``('write', dataset, source)``: an in-context write. The writer's span from the
      logged prefix and read spans (``WriteLog``), float32, then the codecs.

    ``mode`` is the feed (``write_spans``): 'free' replays the free run itself (the
    producer's actual forward); 'teacher' feeds the stored span and takes the rep
    head's predictions (one pass, cheaper, conditioned on the stored span's rounded
    reps); 'self' feeds the writer's own free run in one gradient pass. Items are
    rounded to the store's bf16 by ``ste_round``. On the GPU the free run's numerics
    depend on its batch composition (padding), so bank items are replayed ``batch`` at
    a time (exact at 1 when the bank's spans were written one at a time, ``l1 build
    --span-batch-size 1``) and writes together with the sources of their writer batch
    (the logged ``group``).

    ``resolve(dataset, space, item_id)`` names an item's source (``None``: no
    recomputable source; the item keeps its stored value). By default it follows
    ``origin`` (dataset -> space -> item id -> (producer, sources); L1's
    ``Context.origin``), accepting a codec item whose record has text and a cached span
    and a write in the log. ``stored(dataset, space, item_id)`` returns the stored
    payload for the exactness check (a write's logged items take precedence).

    Per step: ``begin``; reads call ``values`` (forward without gradient, once per
    source and depth, item values as leaves so the gradients of all reads accumulate);
    ``backward`` recomputes the replay units whose items received gradients, with
    gradients and in the composition of their forward (``drift``: the largest
    difference to it), and backpropagates the accumulated item gradients into the
    writer's span heads and the codecs; ``change`` measures, after the optimizer step,
    how far the step moved the recomputed items.

    Chains (``depth`` > 1, ``reread``): a write whose log carries ``rereads`` read
    items that are themselves recomputable (B9: the task's own record of the previous
    round). In a gradient replay at depth d < ``depth`` each such read is recomputed by
    ``reread(state, values, scales)`` from the depth-(d+1) recomputations of those
    items, and enters the write's prefix as ``logged + (reread - reread.detach())``: the
    forward stays the logged span exactly, the gradient reaches the earlier write.
    Depth-(d+1) leaves are separate from depth-d leaves, so each path's gradient is cut
    after ``depth`` writes."""

    def __init__(self, writer: Writer, *, origin: Mapping | None = None,
                 resolve: Callable[[str, str, str], Key | None] | None = None,
                 stored: Callable[[str, str, str], Tensor] | None = None,
                 texts: Mapping[str, str] | None = None, caches: Mapping | None = None,
                 level: str | int = 0, log: WriteLog | None = None, mode: str = 'free',
                 batch: int = 1, depth: int = 1, reread: Callable | None = None):
        if mode not in FEEDS:
            raise ValueError(f'unknown replay {mode!r} (one of {FEEDS})')
        if depth < 1:
            raise ValueError('depth counts the writes a gradient reaches (at least 1)')
        self.writer, self.origin, self.stored = writer, origin, stored
        self._resolve = resolve
        self.texts, self.caches = dict(texts or {}), dict(caches or {})
        self.level, self.log, self.mode, self.batch = level_index(level), log, mode, batch
        self.depth, self.reread = depth, reread
        self._ids: dict[str, Tensor] = {}
        self.begin()

    def begin(self) -> None:
        self.leaves: dict[tuple[Key, int], dict[str, Tensor]] = {}
        self.unit_of: dict[tuple[Key, int], tuple[Key, ...]] = {}
        self.items: dict[tuple[Key, int], dict[str, str]] = {}
        self.stats: dict[str, list] = {}
        self._trained: list[tuple[tuple[Key, ...], int]] = []

    # -- sources ---------------------------------------------------------------------------
    def ids(self, record: str) -> Tensor:
        if record not in self._ids:
            self._ids[record] = self.writer.model.text_ids(self.texts[record])
        return self._ids[record]

    def source(self, dataset: str, space: str, item_id: str) -> Key | None:
        if self._resolve is not None:
            return self._resolve(dataset, space, item_id)
        if self.origin is None:
            return None
        producer, sources = self.origin[dataset][space].get(item_id, (None, ()))
        if len(sources) != 1:
            return None
        if producer == 'codec' and sources[0] in self.texts and dataset in self.caches \
                and sources[0] in self.caches[dataset]:
            return ('codec', dataset, sources[0])
        if producer == 'write' and self.log is not None and sources[0] in self.log.index:
            return ('write', dataset, sources[0])
        return None

    def _unit(self, key: Key) -> tuple[Key, ...]:
        """The replay unit of a written item: the sources generated together with it
        (the writer's batch at the site, as logged), so the replay has the generation's
        batch composition; the item alone when that is unknown."""
        if key[0] != 'write':
            return (key,)
        group = self.log.index[key[2]].get('group') or [key[2]]
        unit = tuple(('write', self.log.index[g]['kb'], g) for g in group if g in self.log.index)
        return unit if key in unit else (key,)

    # -- forward ---------------------------------------------------------------------------
    def _codec(self, key: Key) -> tuple[dict, int, Tensor | None]:
        _, dataset, record = key
        ids = self.ids(record)
        factor, count = span_length(ids.shape[0], self.level)
        cache = self.caches.get(dataset)
        stored = None
        if cache is not None and record in cache:
            count = int(cache.index[record][2])
            stored = cache.get(record)
        elif self.mode == 'teacher':
            raise ValueError(f'record {record} has no cached span to feed')
        return {'ids': ids, 'prompt': 'memory', 'factor': factor}, count, stored

    def _write(self, key: Key, depth: int) -> tuple[dict, int, Tensor]:
        entry = self.log.get(key[2])
        reads = list(entry['reads'])
        if torch.is_grad_enabled() and depth < self.depth and self.reread is not None:
            for j, rr in enumerate(entry.get('rereads') or []):
                if rr is not None:
                    reads[j] = self._chain(reads[j], rr, depth)
        x = self.writer.inputs(entry['prefix_ids'], entry['mems'], reads)
        span = entry['span']
        return {'inputs': x, 'factor': entry['factor']}, int(span.shape[0]), span

    def _chain(self, logged: Tensor, rr: dict, depth: int) -> Tensor:
        """A logged read span with a gradient path into the depth-(d+1) recomputation of
        the recomputable items it read (forward: the logged span exactly)."""
        spaces = rr['spaces']
        if not any(k is not None for sp in spaces.values() for k in sp['keys']):
            return logged
        values = {s: [v if k is None else self._leaf(tuple(k), depth + 1)[s]
                      for k, v in zip(sp['keys'], sp['values'])] for s, sp in spaces.items()}
        with self.writer.autocast():
            span = self.reread(rr['state'], values, {s: sp['scales'] for s, sp in spaces.items()})
        span = span.float()
        logged = logged.to(span.device).float()
        if span.shape != logged.shape:
            self.stats.setdefault('reread_mismatch', []).append(1.0)
            return logged
        self.stats.setdefault('reread_rel', []).append(
            float((span.detach() - logged).norm() / logged.norm().clamp_min(1e-12)))
        return logged + (span - span.detach())

    def spans(self, keys: Sequence[Key], depth: int = 1) -> list[Tensor]:
        """The producers' spans of each source, as the codecs take them (a bank span at
        the span cache's bf16), with gradients when enabled."""
        parts = [self._codec(k) if k[0] == 'codec' else self._write(k, depth) for k in keys]
        examples = [e for e, _, _ in parts]
        counts = [n for _, n, _ in parts]
        teacher = None
        if self.mode == 'teacher':
            teacher = [f.float() for _, _, f in parts]
        reps = write_spans(self.writer.model, examples, counts, self.mode, teacher)
        return [ste_round(r.float()) if k[0] == 'codec' else r.float()
                for r, k in zip(reps, keys)]

    def encode(self, span: Tensor) -> dict[str, Tensor]:
        """The codecs' items of a producer span, at the store's bf16 (straight-through)."""
        return {s: ste_round(v) for s, v in self.writer.encode(span).items()}

    def forward(self, keys: Sequence[Key], depth: int = 1) -> list[dict[str, Tensor]]:
        """The producers' items of each source (gradients when enabled)."""
        return [self.encode(span) for span in self.spans(keys, depth)]

    # -- reads -----------------------------------------------------------------------------
    def _leaf(self, key: Key, depth: int) -> dict[str, Tensor]:
        """The depth-``depth`` recomputation of a source as leaf tensors (its replay unit
        computed once, without gradient)."""
        if (key, depth) not in self.leaves:
            unit = self._unit(key)
            with torch.no_grad():
                outs = self.forward(list(unit), depth)
            for k, out in zip(unit, outs):
                if (k, depth) not in self.leaves:
                    self.leaves[(k, depth)] = {s: v.detach().requires_grad_()
                                               for s, v in out.items()}
                    self.unit_of[(k, depth)] = unit
                    if k[0] == 'write':
                        self._verify_logged(k, depth)
        return self.leaves[(key, depth)]

    def values(self, space: str, refs: Sequence[tuple[str, str]],
               depth: int = 1) -> list[Tensor | None]:
        """Recomputed values of ``refs`` (dataset, item id) in ``space``; None for an
        item without a recomputable source."""
        keys = [self.source(d, space, i) for d, i in refs]
        todo = [k for k in dict.fromkeys(keys) if k is not None and (k, depth) not in self.leaves]
        codec = [k for k in todo if k[0] == 'codec']
        units = [tuple(codec[i:i + self.batch]) for i in range(0, len(codec), self.batch)]
        units += list(dict.fromkeys(self._unit(k) for k in todo if k[0] == 'write'))
        for unit in units:
            if all((k, depth) in self.leaves for k in unit):
                continue
            with torch.no_grad():
                outs = self.forward(list(unit), depth)
            for key, out in zip(unit, outs):
                if (key, depth) not in self.leaves:
                    self.leaves[(key, depth)] = {s: v.detach().requires_grad_()
                                                 for s, v in out.items()}
                    self.unit_of[(key, depth)] = unit
        for key, (dataset, item_id) in zip(keys, refs):
            if key is not None and space not in self.items.setdefault((key, depth), {}):
                self.items[(key, depth)][space] = item_id
                self._verify(key, depth, space, self._stored(key, dataset, space, item_id))
        return [None if k is None else self.leaves[(k, depth)][space] for k in keys]

    def _stored(self, key: Key, dataset: str, space: str, item_id: str) -> Tensor | None:
        if key[0] == 'write':
            logged = self.log.get(key[2])['items']
            if space in logged:
                return logged[space]
        return None if self.stored is None else self.stored(dataset, space, item_id)

    def _verify_logged(self, key: Key, depth: int) -> None:
        for space, value in self.log.get(key[2])['items'].items():
            self._verify(key, depth, space, value)

    @torch.no_grad()
    def _verify(self, key: Key, depth: int, space: str, stored: Tensor | None) -> None:
        """A recomputed item against its stored payload (bf16)."""
        if stored is None:
            return
        got = self.leaves[(key, depth)][space]
        stored = stored.float().to(got.device)
        if got.shape != stored.shape:
            self.stats.setdefault('match_rel', []).append(float('inf'))
            return
        self.stats.setdefault('match_exact', []).append(float((got == stored).float().mean()))
        self.stats.setdefault('match_rel', []).append(
            float((got - stored).norm() / stored.norm().clamp_min(1e-12)))

    # -- backward --------------------------------------------------------------------------
    def backward(self) -> dict:
        """Backpropagate the accumulated item gradients through the producers, depth by
        depth (a depth-d replay passes gradients to depth-(d+1) leaves)."""
        drift, done, sources = 0.0, 0, 0
        for depth in range(1, self.depth + 1):
            todo = [k for (k, d), leaves in self.leaves.items() if d == depth
                    and any(v.grad is not None for v in leaves.values())]
            sources += len(todo)
            for unit in dict.fromkeys(self.unit_of[(k, depth)] for k in todo):
                outs = self.forward(list(unit), depth)
                tensors, grads = [], []
                for key, out in zip(unit, outs):
                    if self.unit_of.get((key, depth)) != unit:
                        continue
                    for s, value in out.items():
                        leaf = self.leaves[(key, depth)][s]
                        drift = max(drift, float((value.detach() - leaf).abs().max()))
                        if leaf.grad is not None:
                            tensors.append(value)
                            grads.append(leaf.grad)
                if tensors:
                    torch.autograd.backward(tensors, grads)
                    self._trained.append((unit, depth))
                    done += 1
        out = {'sources': len({k for k, _ in self.leaves}),
               'backward_sources': sources,
               'backward_units': done, 'drift': drift,
               'depth_max': max((d for _, d in self._trained), default=0)}
        out.update(self._summary())
        return out

    def _summary(self) -> dict:
        out = {}
        for name, values in self.stats.items():
            out[name] = round(sum(values) / len(values), 6) if values else None
            if name in ('match_rel', 'reread_rel', 'change_rel') and values:
                out[f'{name}_max'] = max(values)
        return out

    @torch.no_grad()
    def change(self, limit: int = 0) -> dict:
        """After the optimizer step: the relative change of the recomputed items of up
        to ``limit`` trained replay units (0: all), ||new - old|| / ||old|| per item."""
        units = self._trained if limit <= 0 else self._trained[:limit]
        rel = []
        for unit, depth in units:
            for key, out in zip(unit, self.forward(list(unit), depth)):
                old = self.leaves.get((key, depth))
                if old is None or self.unit_of.get((key, depth)) != unit:
                    continue
                for s, value in out.items():
                    if value.shape == old[s].shape:
                        rel.append(float((value - old[s]).norm()
                                         / old[s].detach().norm().clamp_min(1e-12)))
        if not rel:
            return {}
        return {'change_rel': round(sum(rel) / len(rel), 6), 'change_rel_max': max(rel),
                'change_units': len(units)}


# -- rewrites --------------------------------------------------------------------------------
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
