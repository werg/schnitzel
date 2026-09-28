"""Knowledge-base store (docs/knowledge-base-stack.md, sections 3 and 5; WP2).

One KB per dataset; a KB is also an authorization domain. It holds spaces (by default
A-D, 384/512/768/1024 wide). An item lives in one space: a variable-length sequence of
vectors of the space's width (bf16 on disk), one key (float32), a mass, provenance
(source record ids, dataset, producer, step), an availability time, a version and
rewrite lineage. Items are identified by opaque ids; a supersede keeps the id and
bumps the version, a rewrite replaces a set of items by new items whose lineage names
the exact (id, version) inputs (compaction is a rewrite with fewer outputs).

Layout, one directory per KB::

    manifest.json               schema, name, dataset, cursor, space specs and counts
    <space>/keys.f32            (items, key_width)  one contiguous matrix for exact scans
    <space>/rows.i64            (items, 8)          offset, length, born, dead, time, ...
    <space>/mass.f32 payload.bf16 ids.txt meta.jsonl
    <space>/live_*.f32|i64      live mode: fp32 values, Adam m and v, per-item step

Commits. Every mutation appends rows past the committed counts, marks superseded rows
``dead = cursor + 1`` in place and then atomically replaces the manifest (``.pending``
then rename), which advances the cursor and the counts. A row is visible at cursor c
when ``born <= c`` and it is not dead at c; readers pin the cursor of the manifest they
loaded, so other processes read a consistent snapshot while one writer appends. On
writer open, bytes past the committed counts, ``dead`` marks past the cursor and
pending files are discarded: an interrupted commit leaves no trace.

Search is an exact chunked scan over the memory-mapped key matrix of one space (not
ANN, invariant 8), limited to one KB; searching several KBs requires listing the
datasets the caller may read (learned selection is never authorization, invariant 6).
Queries see only items whose time is at or before their query time (invariant 2).

Live mode (stage L1) keeps fp32 master values with per-item Adam state beside the
stored rows and updates only the items named in a step. It is mutable training state:
each update is written as a redo journal first, so it is applied completely or not at
all, and ``live_updates`` counts applied steps. Live values and live keys are not
versioned by the cursor (a pinned older cursor sees the current live keys). One process
writes; in-process readers are serialized with a lock. Stored bf16 payloads are never changed by live updates;
``export_live`` writes the current live values as a new frozen KB for later stages.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

SCHEMA = 'schnitz.kb/1'
PRODUCERS = ('codec', 'rewrite', 'live-update')
# rows.i64 columns
OFFSET, LENGTH, BORN, DEAD, TIME, VERSION, META_OFF, META_LEN = range(8)
LIVE_FILES = ('live_values.f32', 'live_m.f32', 'live_v.f32', 'live_step.i64')


@dataclass(frozen=True)
class SpaceSpec:
    width: int
    key_width: int = 256
    ratio: float = 1.0      # positions per writer rep, m_s = ceil(ratio * n)


DEFAULT_SPACES = {'A': SpaceSpec(384, 256, 1.0), 'B': SpaceSpec(512, 256, 0.5),
                  'C': SpaceSpec(768, 256, 0.25), 'D': SpaceSpec(1024, 256, 0.125)}


@dataclass(frozen=True)
class Provenance:
    sources: tuple[str, ...]    # opaque source record ids
    producer: str               # one of PRODUCERS
    step: int = 0               # producer checkpoint or live-update step
    dataset: str = ''           # '' means the KB's dataset; any other dataset is refused


@dataclass
class NewItem:
    values: Tensor              # (positions, width)
    key: Tensor                 # (key_width,)
    provenance: Provenance
    mass: float = 1.0
    time: int = 0               # availability time; queries before it cannot see the item
    id: str | None = None       # None: a fresh opaque id


@dataclass
class Item:
    id: str
    version: int
    space: str
    values: Tensor
    key: Tensor
    mass: float
    time: int
    provenance: Provenance
    lineage: tuple[tuple[str, int], ...]
    current: bool


@dataclass
class SearchHits:
    """Per query: hits in descending score order (fewer than k if fewer are visible)."""
    ids: list[list[str]]
    versions: list[list[int]]
    scores: list[Tensor]
    datasets: list[list[str]]
    cursor: dict[str, int]      # dataset -> cursor the search read
    items: list[list[Item]] | None = field(default=None)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value) -> None:
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as handle:
        json.dump(value, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    _fsync_dir(path.parent)


def _write_at(path: Path, offset: int, data: bytes) -> None:
    with path.open('r+b') as handle:
        handle.seek(offset)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class KnowledgeBase:
    """One KB directory. ``writable=True`` takes the exclusive writer lock and recovers."""

    def __init__(self, root: str | Path, *, writable: bool = False):
        self.root = Path(root)
        self.writable = writable
        self._lock = threading.RLock()
        self._lockfile = None
        self._maps: dict[tuple[str, str], np.ndarray] = {}
        if writable:
            self._lockfile = (self.root / 'writer.lock').open('a+')
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lockfile.close()
                raise RuntimeError(f'{self.root} already has a writer') from None
        self._manifest = json.loads((self.root / 'manifest.json').read_text())
        if self._manifest.get('schema') != SCHEMA:
            raise ValueError(f'unsupported KB schema {self._manifest.get("schema")!r}')
        if writable and self._manifest['frozen']:
            self.close()
            raise PermissionError('frozen KB (an export) cannot be written')
        self.spaces = {name: SpaceSpec(s['width'], s['key_width'], s['ratio'])
                       for name, s in self._manifest['spaces'].items()}
        if writable:
            self._recover()
        self._build_index()

    # -- creation, lifecycle -------------------------------------------------------------

    @classmethod
    def create(cls, root: str | Path, *, name: str, dataset: str,
               spaces: dict[str, SpaceSpec] | None = None,
               origin: dict | None = None) -> KnowledgeBase:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(root)
        if not name or not dataset:
            raise ValueError('a KB needs a name and a dataset')
        spaces = DEFAULT_SPACES if spaces is None else spaces
        root.mkdir(parents=True, exist_ok=True)
        manifest = {'schema': SCHEMA, 'name': name, 'dataset': dataset, 'frozen': False,
                    'cursor': 0, 'live_updates': 0, 'live_spaces': [], 'origin': origin,
                    'spaces': {}}
        for space, spec in spaces.items():
            if not space.isidentifier() or spec.width < 1 or spec.key_width < 1:
                raise ValueError(f'invalid space {space!r}')
            (root / space).mkdir()
            for file in ('keys.f32', 'rows.i64', 'mass.f32', 'payload.bf16', 'ids.txt',
                         'meta.jsonl'):
                (root / space / file).touch()
            manifest['spaces'][space] = {'width': spec.width, 'key_width': spec.key_width,
                                         'ratio': spec.ratio, 'items': 0, 'positions': 0,
                                         'ids_bytes': 0, 'meta_bytes': 0}
        _atomic_json(root / 'manifest.json', manifest)
        return cls(root, writable=True)

    def close(self) -> None:
        self._maps.clear()
        if self._lockfile is not None:
            fcntl.flock(self._lockfile, fcntl.LOCK_UN)
            self._lockfile.close()
            self._lockfile = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def name(self) -> str:
        return self._manifest['name']

    @property
    def dataset(self) -> str:
        return self._manifest['dataset']

    @property
    def cursor(self) -> int:
        return self._manifest['cursor']

    @property
    def live_updates(self) -> int:
        return self._manifest['live_updates']

    def is_live(self, space: str) -> bool:
        return space in self._manifest['live_spaces']

    def refresh(self) -> None:
        """Reader: move to the writer's latest committed cursor."""
        with self._lock:
            self._manifest = json.loads((self.root / 'manifest.json').read_text())
            self._maps.clear()
            self._build_index()

    # -- files ---------------------------------------------------------------------------

    def _sizes(self, space: str) -> dict[str, int]:
        """Committed byte size of every file of a space."""
        s = self._manifest['spaces'][space]
        n, p, w = s['items'], s['positions'], s['width']
        sizes = {'keys.f32': n * s['key_width'] * 4, 'rows.i64': n * 64, 'mass.f32': n * 4,
                 'payload.bf16': p * w * 2, 'ids.txt': s['ids_bytes'],
                 'meta.jsonl': s['meta_bytes']}
        if self.is_live(space):
            sizes.update({'live_values.f32': p * w * 4, 'live_m.f32': p * w * 4,
                          'live_v.f32': p * w * 4, 'live_step.i64': n * 8})
        return sizes

    def _map(self, space: str, file: str) -> np.ndarray:
        """Committed prefix of a file as a (possibly writable) memory map."""
        cached = self._maps.get((space, file))
        if cached is not None:
            return cached
        s = self._manifest['spaces'][space]
        n, p, w = s['items'], s['positions'], s['width']
        dtype, shape = {'keys.f32': (np.float32, (n, s['key_width'])),
                        'rows.i64': (np.int64, (n, 8)), 'mass.f32': (np.float32, (n,)),
                        'payload.bf16': (np.int16, (p, w)),
                        'meta.jsonl': (np.uint8, (s['meta_bytes'],)),
                        'live_values.f32': (np.float32, (p, w)),
                        'live_m.f32': (np.float32, (p, w)), 'live_v.f32': (np.float32, (p, w)),
                        'live_step.i64': (np.int64, (n,))}[file]
        if shape[0] == 0:
            array = np.zeros(shape, dtype)
        else:
            array = np.memmap(self.root / space / file, dtype=dtype,
                              mode='r+' if self.writable else 'r', shape=shape)
        self._maps[(space, file)] = array
        return array

    def _recover(self) -> None:
        """Discard everything an interrupted commit or live update left behind."""
        for pending in self.root.glob('**/*.pending'):
            pending.unlink()
        for space in self.spaces:
            if not self.is_live(space):
                for file in LIVE_FILES:
                    (self.root / space / file).unlink(missing_ok=True)
            for file, size in self._sizes(space).items():
                path = self.root / space / file
                if path.stat().st_size < size:
                    raise ValueError(f'{path} is shorter than its committed size')
                os.truncate(path, size)
            rows = self._map(space, 'rows.i64')
            late = rows[:, DEAD] > self.cursor
            if late.any():
                rows[late, DEAD] = 0
                rows.flush()
        journal = self.root / 'live.journal'
        if journal.exists():
            self._apply_journal(journal)

    def _build_index(self) -> None:
        self._row_ids: dict[str, list[str]] = {}
        self._rows: dict[str, dict[str, list[int]]] = {}
        self._space_of: dict[str, str] = {}
        for space in self.spaces:
            size = self._manifest['spaces'][space]['ids_bytes']
            with (self.root / space / 'ids.txt').open('rb') as handle:
                ids = handle.read(size).decode().split('\n')[:-1]
            by_id: dict[str, list[int]] = {}
            for row, item_id in enumerate(ids):
                by_id.setdefault(item_id, []).append(row)
                self._space_of[item_id] = space
            self._row_ids[space], self._rows[space] = ids, by_id

    # -- visibility, lookup --------------------------------------------------------------

    def _view(self, cursor: int | None) -> int:
        cursor = self.cursor if cursor is None else cursor
        if not 0 <= cursor <= self.cursor:
            raise ValueError(f'cursor {cursor} is not committed (at {self.cursor})')
        return cursor

    @staticmethod
    def _visible(rows: np.ndarray, cursor: int) -> np.ndarray:
        dead = rows[:, DEAD]
        return (rows[:, BORN] <= cursor) & ((dead == 0) | (dead > cursor))

    def _check_space(self, space: str) -> SpaceSpec:
        if space not in self.spaces:
            raise KeyError(f'no space {space!r} in KB {self.name}')
        return self.spaces[space]

    def has(self, item_id: str) -> bool:
        return item_id in self._space_of

    def _row(self, space: str, item_id: str, cursor: int, version: int | None = None) -> int:
        rows = self._map(space, 'rows.i64')
        for row in reversed(self._rows[space].get(item_id, ())):
            if rows[row, BORN] > cursor:
                continue
            if version is None and self._visible(rows[row:row + 1], cursor)[0]:
                return row
            if version is not None and rows[row, VERSION] == version:
                return row
        what = f'{item_id}@{version}' if version is not None else item_id
        raise KeyError(f'no {"" if version is not None else "current "}item {what} '
                       f'in space {space} of KB {self.name} at cursor {cursor}')

    def _meta(self, space: str, row: int) -> dict:
        r = self._map(space, 'rows.i64')[row]
        return json.loads(bytes(self._map(space, 'meta.jsonl')[r[META_OFF]:r[META_OFF] + r[META_LEN]]))

    def _item(self, space: str, row: int, cursor: int, live: bool) -> Item:
        r = self._map(space, 'rows.i64')[row]
        start, length = int(r[OFFSET]), int(r[LENGTH])
        if live:
            values = torch.from_numpy(np.array(self._map(space, 'live_values.f32')[start:start + length]))
        else:
            values = torch.from_numpy(np.array(self._map(space, 'payload.bf16')[start:start + length]))
            values = values.view(torch.bfloat16)
        meta = self._meta(space, row)
        return Item(meta['id'], int(r[VERSION]), space, values,
                    torch.from_numpy(np.array(self._map(space, 'keys.f32')[row])),
                    float(self._map(space, 'mass.f32')[row]), int(r[TIME]),
                    Provenance(tuple(meta['sources']), meta['producer'], meta['step'],
                               meta['dataset']),
                    tuple((i, v) for i, v in meta['lineage']),
                    bool(self._visible(r[None], cursor)[0]))

    def read(self, space: str, ids: Sequence[str], *, versions: Sequence[int | None] | None = None,
             live: bool = False, cursor: int | None = None) -> list[Item]:
        """Items by id: the current version at ``cursor``, or an explicit (possibly
        superseded) version for measurement. ``live`` returns fp32 live values."""
        with self._lock:
            self._check_space(space)
            cursor = self._view(cursor)
            if live and (not self.is_live(space) or cursor != self.cursor):
                raise ValueError('live values exist only for a live space at the current cursor')
            versions = [None] * len(ids) if versions is None else list(versions)
            return [self._item(space, self._row(space, i, cursor, v), cursor, live)
                    for i, v in zip(ids, versions, strict=True)]

    # -- mutations -----------------------------------------------------------------------

    def _validate(self, space: str, item: NewItem) -> None:
        spec = self.spaces[space]
        values, key = item.values, item.key
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != spec.width:
            raise ValueError(f'space {space} items are (positions >= 1, {spec.width})')
        if key.shape != (spec.key_width,):
            raise ValueError(f'space {space} keys are ({spec.key_width},)')
        if not (torch.isfinite(values.float()).all() and torch.isfinite(key.float()).all()):
            raise ValueError('item values and key must be finite')
        if not (np.isfinite(item.mass) and item.mass >= 0 and item.time >= 0):
            raise ValueError('mass must be finite and nonnegative, time nonnegative')
        p = item.provenance
        if p.producer not in PRODUCERS:
            raise ValueError(f'producer must be one of {PRODUCERS}')
        if p.dataset not in ('', self.dataset):
            raise PermissionError(f'item from dataset {p.dataset!r} cannot enter KB of '
                                  f'{self.dataset!r}')
        if not all(isinstance(s, str) and s for s in p.sources):
            raise ValueError('source ids are nonempty strings')

    def _commit(self, space: str, items: Sequence[NewItem], ids: Sequence[str],
                versions: Sequence[int], lineages: Sequence[Sequence[tuple[str, int]]],
                kill: Sequence[int] = ()) -> None:
        """Append rows, mark ``kill`` rows superseded, publish one new cursor."""
        if not self.writable:
            raise PermissionError('KB opened read-only')
        spec, s = self.spaces[space], dict(self._manifest['spaces'][space])
        cursor = self.cursor + 1
        n, p = s['items'], s['positions']
        keys = np.stack([i.key.detach().float().cpu().numpy() for i in items]).astype('<f4')
        values = torch.cat([i.values.detach().cpu().to(torch.bfloat16) for i in items])
        rows = np.zeros((len(items), 8), np.int64)
        meta, id_bytes, offset, meta_offset = [], [], p, s['meta_bytes']
        for j, (item, item_id, version, lineage) in enumerate(zip(items, ids, versions, lineages,
                                                                 strict=True)):
            if not item_id or '\n' in item_id:
                raise ValueError('item ids are nonempty single-line strings')
            line = json.dumps({'id': item_id, 'sources': list(item.provenance.sources),
                               'producer': item.provenance.producer,
                               'step': int(item.provenance.step), 'dataset': self.dataset,
                               'lineage': [list(x) for x in lineage]}).encode() + b'\n'
            rows[j] = (offset, item.values.shape[0], cursor, 0, item.time, version,
                       meta_offset, len(line) - 1)
            offset += item.values.shape[0]
            meta_offset += len(line)
            meta.append(line)
            id_bytes.append(item_id.encode() + b'\n')
        root = self.root / space
        payload = values.contiguous().view(torch.int16).numpy()
        _write_at(root / 'keys.f32', n * spec.key_width * 4, keys.tobytes())
        _write_at(root / 'rows.i64', n * 64, rows.astype('<i8').tobytes())
        _write_at(root / 'mass.f32', n * 4,
                  np.asarray([i.mass for i in items], '<f4').tobytes())
        _write_at(root / 'payload.bf16', p * spec.width * 2, payload.tobytes())
        _write_at(root / 'ids.txt', s['ids_bytes'], b''.join(id_bytes))
        _write_at(root / 'meta.jsonl', s['meta_bytes'], b''.join(meta))
        if self.is_live(space):
            # live values start from the stored (bf16-rounded) payload, moments at zero
            start = values.float().numpy().astype('<f4').tobytes()
            zeros = bytes(len(start))
            for file, data in (('live_values.f32', start), ('live_m.f32', zeros),
                               ('live_v.f32', zeros)):
                _write_at(root / file, p * spec.width * 4, data)
            _write_at(root / 'live_step.i64', n * 8, bytes(8 * len(items)))
        if kill:
            dead = self._map(space, 'rows.i64')
            dead[list(kill), DEAD] = cursor
            dead.flush()
        s.update(items=n + len(items), positions=offset, meta_bytes=meta_offset,
                 ids_bytes=s['ids_bytes'] + sum(map(len, id_bytes)))
        manifest = dict(self._manifest, cursor=cursor,
                        spaces=dict(self._manifest['spaces'], **{space: s}))
        _atomic_json(self.root / 'manifest.json', manifest)
        self._manifest = manifest
        self._maps = {k: v for k, v in self._maps.items() if k[0] != space}
        for j, item_id in enumerate(ids):
            self._row_ids[space].append(item_id)
            self._rows[space].setdefault(item_id, []).append(n + j)
            self._space_of[item_id] = space

    def append(self, space: str, items: Sequence[NewItem]) -> list[str]:
        """Add new items (version 1, no lineage); returns their ids. Items given an id
        that already exists are refused, so a resumed producer can skip with ``has``."""
        with self._lock:
            self._check_space(space)
            if not items:
                return []
            for item in items:
                self._validate(space, item)
            ids = [item.id or uuid.uuid4().hex for item in items]
            if len(set(ids)) != len(ids) or any(self.has(i) for i in ids):
                raise ValueError('appended ids must be new and distinct')
            self._commit(space, items, ids, [1] * len(items), [()] * len(items))
            return ids

    def supersede(self, space: str, items: Sequence[NewItem]) -> list[tuple[str, int]]:
        """Replace current items by new versions of the same ids (``item.id`` required).
        Old versions stay readable by explicit version but are no longer searched."""
        with self._lock:
            self._check_space(space)
            ids = [item.id for item in items]
            if not items or None in ids or len(set(ids)) != len(ids):
                raise ValueError('supersede needs distinct existing ids')
            rows = [self._row(space, i, self.cursor) for i in ids]
            table = self._map(space, 'rows.i64')
            old = [int(table[r, VERSION]) for r in rows]
            fixed = []
            for item, row in zip(items, rows):
                self._validate(space, item)
                # a new version is never available before the version it replaces
                fixed.append(NewItem(item.values, item.key, item.provenance, item.mass,
                                     max(item.time, int(table[row, TIME])), item.id))
            lineage = [((i, v),) for i, v in zip(ids, old)]
            self._commit(space, fixed, ids, [v + 1 for v in old], lineage, rows)
            return [(i, v + 1) for i, v in zip(ids, old)]

    def rewrite(self, space: str, inputs: Sequence[str], outputs: Sequence[NewItem], *,
                mass_tolerance: float = 1e-4) -> list[str]:
        """Replace the current ``inputs`` by ``outputs`` (new ids) in one commit.

        Every output's lineage is the exact (id, version) of all inputs; outputs with no
        sources inherit the union of the inputs' sources; an output's time is at least
        the latest input time. The total mass is conserved (invariants 5 and 7), so a
        compaction (fewer outputs) weighs as much as the items it replaced."""
        with self._lock:
            self._check_space(space)
            if not inputs or not outputs or len(set(inputs)) != len(inputs):
                raise ValueError('rewrite needs distinct inputs and at least one output')
            rows = [self._row(space, i, self.cursor) for i in inputs]
            table, mass = self._map(space, 'rows.i64'), self._map(space, 'mass.f32')
            total_in = float(sum(mass[r] for r in rows))
            total_out = float(sum(item.mass for item in outputs))
            if abs(total_out - total_in) > mass_tolerance * max(1.0, abs(total_in)):
                raise ValueError(f'rewrite changes the total mass ({total_in} -> {total_out})')
            lineage = tuple((i, int(table[r, VERSION])) for i, r in zip(inputs, rows))
            latest = max(int(table[r, TIME]) for r in rows)
            sources = sorted({s for r in rows for s in self._meta(space, r)['sources']})
            fixed = []
            for item in outputs:
                self._validate(space, item)
                provenance = item.provenance
                if not provenance.sources:
                    provenance = Provenance(tuple(sources), provenance.producer, provenance.step,
                                            provenance.dataset)
                fixed.append(NewItem(item.values, item.key, provenance, item.mass,
                                     max(item.time, latest), item.id))
            ids = [item.id or uuid.uuid4().hex for item in fixed]
            if len(set(ids)) != len(ids) or any(self.has(i) for i in ids):
                raise ValueError('rewrite outputs need new, distinct ids')
            self._commit(space, fixed, ids, [1] * len(ids), [lineage] * len(ids), rows)
            return ids

    # -- search --------------------------------------------------------------------------

    def search(self, space: str, queries: Tensor, k: int, *, metric: str = 'cosine',
               query_time: int | Sequence[int] | Tensor | None = None,
               cursor: int | None = None, chunk_rows: int = 32768,
               return_items: bool = False) -> SearchHits:
        """Exact top-k over the current keys of one space of this KB.

        ``queries`` is (batch, key_width); ``metric`` is 'cosine' or 'dot'. Keys are
        streamed from the memory-mapped matrix ``chunk_rows`` at a time, so memory is
        bounded by the chunk, not the KB. Superseded items and items whose time is after
        the query's time are never returned."""
        with self._lock:
            spec = self._check_space(space)
            cursor = self._view(cursor)
            if metric not in ('cosine', 'dot'):
                raise ValueError('metric is cosine or dot')
            if queries.ndim != 2 or queries.shape[1] != spec.key_width or k < 1:
                raise ValueError(f'queries are (batch, {spec.key_width}), k >= 1')
            q = queries.detach().float().cpu()
            if metric == 'cosine':
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            times = torch.full((len(q),), 2**62, dtype=torch.long) if query_time is None \
                else torch.as_tensor(query_time, dtype=torch.long).expand(len(q))
            best = torch.full((len(q), 0), float('-inf'))
            best_rows = torch.zeros((len(q), 0), dtype=torch.long)
            keys, rows = self._map(space, 'keys.f32'), self._map(space, 'rows.i64')
            for start in range(0, len(keys), chunk_rows):
                block = np.array(rows[start:start + chunk_rows])
                ok = torch.from_numpy(self._visible(block, cursor))
                ok = ok[None] & (torch.from_numpy(block[:, TIME])[None] <= times[:, None])
                if not ok.any():
                    continue
                kk = torch.from_numpy(np.array(keys[start:start + chunk_rows]))
                if metric == 'cosine':
                    kk = kk / kk.norm(dim=1, keepdim=True).clamp_min(1e-12)
                scores = (q @ kk.T).masked_fill(~ok, float('-inf'))
                index = torch.arange(start, start + len(block)).expand(len(q), -1)
                best, pick = torch.cat((best, scores), 1).topk(min(k, best.shape[1] + len(block)), 1)
                best_rows = torch.cat((best_rows, index), 1).gather(1, pick)
            hits = SearchHits([], [], [], [], {self.dataset: cursor})
            for b in range(len(q)):
                keep = torch.isfinite(best[b])
                found = best_rows[b][keep].tolist()
                hits.ids.append([self._row_ids[space][r] for r in found])
                hits.versions.append([int(rows[r, VERSION]) for r in found])
                hits.scores.append(best[b][keep])
                hits.datasets.append([self.dataset] * len(found))
            if return_items:
                hits.items = [[self._item(space, r, cursor, False) for r in
                               best_rows[b][torch.isfinite(best[b])].tolist()]
                              for b in range(len(q))]
            return hits

    # -- live mode (L1) ------------------------------------------------------------------

    def enable_live(self, space: str) -> None:
        """Start live mode: fp32 values from the stored payloads, Adam state at zero."""
        with self._lock:
            self._check_space(space)
            if not self.writable or self.is_live(space):
                raise ValueError('live mode needs a writer and a space that is not live yet')
            root, payload = self.root / space, self._map(space, 'payload.bf16')
            for file in LIVE_FILES:
                (root / file).write_bytes(b'')
            with (root / 'live_values.f32').open('ab') as out:
                for start in range(0, len(payload), 65536):
                    chunk = torch.from_numpy(np.array(payload[start:start + 65536]))
                    out.write(chunk.view(torch.bfloat16).float().numpy().astype('<f4').tobytes())
                out.flush()
                os.fsync(out.fileno())
            s = self._manifest['spaces'][space]
            for file, size in (('live_m.f32', s['positions'] * s['width'] * 4),
                               ('live_v.f32', s['positions'] * s['width'] * 4),
                               ('live_step.i64', s['items'] * 8)):
                os.truncate(root / file, size)   # sparse zeros
            manifest = dict(self._manifest, live_spaces=self._manifest['live_spaces'] + [space])
            _atomic_json(self.root / 'manifest.json', manifest)
            self._manifest = manifest
            self._maps = {k: v for k, v in self._maps.items() if k[0] != space}

    def _live_rows(self, space: str, ids: Sequence[str]) -> tuple[list[int], np.ndarray]:
        if not self.writable or not self.is_live(space):
            raise ValueError(f'space {space} is not live in a writer')
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate ids would apply an update twice')
        rows = [self._row(space, i, self.cursor) for i in ids]
        table = self._map(space, 'rows.i64')
        positions = np.concatenate([np.arange(table[r, OFFSET], table[r, OFFSET] + table[r, LENGTH])
                                    for r in rows]) if rows else np.zeros(0, np.int64)
        return rows, positions

    def live_step(self, space: str, ids: Sequence[str], grads: Sequence[Tensor], *, lr: float,
                  betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                  weight_decay: float = 0.0) -> None:
        """One Adam(W) step on the named items only, each with its own step count.

        Matches ``torch.optim.Adam`` (``weight_decay=0``) or ``AdamW`` (decoupled decay)
        applied to each item as its own parameter; items not named are untouched."""
        with self._lock:
            self._check_space(space)
            rows, positions = self._live_rows(space, ids)
            table = self._map(space, 'rows.i64')
            for r, g in zip(rows, grads, strict=True):
                if g.shape != (table[r, LENGTH], self.spaces[space].width):
                    raise ValueError('each gradient has its item\'s shape')
            if not rows:
                return
            p = torch.from_numpy(np.array(self._map(space, 'live_values.f32')[positions]))
            m = torch.from_numpy(np.array(self._map(space, 'live_m.f32')[positions]))
            v = torch.from_numpy(np.array(self._map(space, 'live_v.f32')[positions]))
            g = torch.cat([x.detach().float().cpu() for x in grads])
            steps = torch.from_numpy(np.array(self._map(space, 'live_step.i64')[rows])) + 1
            per_position = torch.repeat_interleave(steps, torch.as_tensor(table[rows, LENGTH]))
            b1, b2 = betas
            # grouped by step count so each group uses torch's exact scalar arithmetic
            for step in steps.unique().tolist():
                sel = per_position == step
                pp, mm, vv, gg = p[sel], m[sel], v[sel], g[sel]
                if weight_decay:
                    pp.mul_(1 - lr * weight_decay)
                mm.lerp_(gg, 1 - b1)
                vv.mul_(b2).addcmul_(gg, gg, value=1 - b2)
                bias1, bias2 = 1 - b1 ** step, 1 - b2 ** step
                denom = (vv.sqrt() / (bias2 ** 0.5)).add_(eps)
                pp.addcdiv_(mm, denom, value=-lr / bias1)
                p[sel], m[sel], v[sel] = pp, mm, vv
            self._journal(space, {'values': p, 'm': m, 'v': v}, positions,
                          {'step': steps}, np.asarray(rows))

    def set_live_keys(self, space: str, ids: Sequence[str], keys: Tensor) -> None:
        """Replace the keys of live items in place (e.g. from the trained key head)."""
        with self._lock:
            rows, _ = self._live_rows(space, ids)
            if keys.shape != (len(rows), self.spaces[space].key_width) \
                    or not torch.isfinite(keys).all():
                raise ValueError('keys are finite (items, key_width)')
            self._journal(space, {}, np.zeros(0, np.int64),
                          {'keys': keys.detach().float().cpu()}, np.asarray(rows))

    def _journal(self, space: str, position_data: dict[str, Tensor], positions: np.ndarray,
                 row_data: dict[str, Tensor], rows: np.ndarray) -> None:
        """Write a redo journal, apply it, count the update, drop the journal."""
        tensors = {'positions': torch.from_numpy(positions.astype(np.int64)),
                   'rows': torch.from_numpy(rows.astype(np.int64)),
                   **{'p.' + k: t.contiguous() for k, t in position_data.items()},
                   **{'r.' + k: t.contiguous() for k, t in row_data.items()}}
        journal = self.root / 'live.journal'
        pending = journal.with_name(journal.name + '.pending')
        save_file(tensors, pending, metadata={'space': space,
                                              'update': str(self.live_updates + 1)})
        with pending.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(pending, journal)
        _fsync_dir(self.root)
        self._apply_journal(journal)

    def _apply_journal(self, journal: Path) -> None:
        from safetensors import safe_open
        with safe_open(journal, 'pt') as handle:
            meta = handle.metadata()
        tensors = load_file(journal)
        space, positions, rows = meta['space'], tensors['positions'].numpy(), tensors['rows'].numpy()
        targets = {'p.values': 'live_values.f32', 'p.m': 'live_m.f32', 'p.v': 'live_v.f32',
                   'r.step': 'live_step.i64', 'r.keys': 'keys.f32'}
        for name, tensor in tensors.items():
            if name in targets:
                array = self._map(space, targets[name])
                array[positions if name.startswith('p.') else rows] = tensor.numpy()
                array.flush()
        manifest = dict(self._manifest, live_updates=int(meta['update']))
        _atomic_json(self.root / 'manifest.json', manifest)
        self._manifest = manifest
        journal.unlink()
        _fsync_dir(self.root)

    def export_live(self, dest: str | Path, *, name: str | None = None,
                    batch: int = 1024) -> KnowledgeBase:
        """Write every current item as a frozen KB at ``dest``: live spaces contribute their
        live values (bf16, producer 'live-update', step = the item's Adam steps), other
        spaces their stored payloads. Ids and versions are kept; the manifest's origin
        names this KB, its cursor and ``live_updates``. Built under ``dest.pending``."""
        with self._lock:
            dest = Path(dest)
            if dest.exists():
                raise FileExistsError(dest)
            pending = dest.with_name(dest.name + '.pending')
            shutil.rmtree(pending, ignore_errors=True)
            origin = {'name': self.name, 'cursor': self.cursor, 'live_updates': self.live_updates}
            out = KnowledgeBase.create(pending, name=name or f'{self.name}@live{self.live_updates}',
                                       dataset=self.dataset, spaces=self.spaces, origin=origin)
            for space in self.spaces:
                live = self.is_live(space)
                current = np.flatnonzero(self._visible(self._map(space, 'rows.i64'), self.cursor))
                for start in range(0, len(current), batch):
                    rows = current[start:start + batch].tolist()
                    items = [self._item(space, r, self.cursor, live) for r in rows]
                    steps = self._map(space, 'live_step.i64')[rows] if live else [0] * len(rows)
                    fresh = [NewItem(i.values, i.key,
                                     Provenance(i.provenance.sources, 'live-update', int(s))
                                     if live else i.provenance, i.mass, i.time, i.id)
                             for i, s in zip(items, steps)]
                    out._commit(space, fresh, [i.id for i in items], [i.version for i in items],
                                [i.lineage for i in items])
            _atomic_json(pending / 'manifest.json', dict(out._manifest, frozen=True))
            out.close()
            (pending / 'writer.lock').unlink()
            os.replace(pending, dest)
            _fsync_dir(dest.parent)
            return KnowledgeBase(dest)

    def stats(self) -> dict:
        with self._lock:
            out = {}
            for space in self.spaces:
                s = self._manifest['spaces'][space]
                out[space] = {'items': s['items'], 'positions': s['positions'],
                              'current': int(self._visible(self._map(space, 'rows.i64'),
                                                           self.cursor).sum()),
                              'live': self.is_live(space)}
            return out


def search_kbs(kbs: Sequence[KnowledgeBase], allowed: Iterable[str], space: str,
               queries: Tensor, k: int, **options) -> SearchHits:
    """Exact top-k across KBs the caller is authorized for.

    ``allowed`` lists the datasets (authorization domains) the caller may read; every
    KB must be one of them, and each dataset may appear once. Results are merged by
    score; ``datasets`` says which KB each hit came from."""
    allowed = set(allowed)
    names = [kb.dataset for kb in kbs]
    if not kbs or len(set(names)) != len(names):
        raise ValueError('list each KB (dataset) once')
    denied = [n for n in names if n not in allowed]
    if denied:
        raise PermissionError(f'not authorized to read {denied}')
    parts = [kb.search(space, queries, k, **options) for kb in kbs]
    merged = SearchHits([], [], [], [], {kb.dataset: part.cursor[kb.dataset]
                                        for kb, part in zip(kbs, parts)})
    merged.items = [] if options.get('return_items') else None
    for b in range(len(queries)):
        pool = [(float(s), j, n) for j, part in enumerate(parts)
                for n, s in enumerate(part.scores[b].tolist())]
        pool.sort(key=lambda x: -x[0])
        pool = pool[:k]
        merged.ids.append([parts[j].ids[b][n] for _, j, n in pool])
        merged.versions.append([parts[j].versions[b][n] for _, j, n in pool])
        merged.datasets.append([parts[j].datasets[b][n] for _, j, n in pool])
        merged.scores.append(torch.tensor([s for s, _, _ in pool]))
        if merged.items is not None:
            merged.items.append([parts[j].items[b][n] for _, j, n in pool])
    return merged
