"""Knowledge-base store (docs/knowledge-base-stack.md, sections 3 and 5; WP2).

One KB per dataset; a KB is also an authorization domain. It holds spaces (by default
A-D, 384/512/768/1024 wide). An item lives in one space: a variable-length sequence of
vectors of the space's width (bf16 on disk), one key (float32), a mass, provenance
(source record ids, dataset, producer, step), an availability time, a version and
lineage. Items are identified by opaque ids; a supersede keeps the id and bumps the
version, a rewrite replaces a set of items by new items (compaction is a rewrite with
fewer outputs).

Lineage and responsibility shares (invariants 5 and 7). Every lineage entry is an exact
(id, version, share). In a rewrite each output declares shares over its inputs; each
input's shares over all outputs sum to one (checked, then normalized exactly) and each
output's mass is the share-weighted sum of its inputs' masses, so mass is conserved by
construction and no evidence is duplicated or dropped. An output lists only inputs with
a positive share and inherits only their sources. A supersede has one entry with share
1. Items whose composition flows through their lineage are ``derived`` (rewrite outputs,
supersedes that name no sources of their own); ``lineage()`` returns the weighted graph
(share times input mass) that ``schnitz.kb_eval.source_composition`` resolves, and
``source_composition()`` applies it to the current items.

Layout, one directory per KB::

    manifest.json               schema, name, dataset, cursor, space specs and counts
    <space>/keys.f32            (items, key_width)  one contiguous matrix for exact scans
    <space>/rows.i64            (items, 8)          offset, length, born, dead, time, ...
    <space>/mass.f32 payload.bf16 ids.txt meta.jsonl
    <space>/segments.jsonl      one line per commit: file ranges, killed rows, checksums
    <space>/live_*              live mode: fp32 values, Adam m and v, per-item step, keys
    history.jsonl               after export or compaction: metadata of dropped rows
    live_checkpoints/<tag>/     live-state checkpoints (checkpoint_live)

Commits. Every mutation appends rows past the committed counts, marks superseded rows
``dead = cursor + 1`` in place and then atomically replaces the manifest (``.pending``
then rename), which advances the cursor and the counts. A row is visible at cursor c
when ``born <= c`` and it is not dead at c; readers pin the cursor of the manifest they
loaded, so other processes read a consistent snapshot while one writer appends. On
writer open, bytes past the committed counts, ``dead`` marks past the cursor and
pending files are discarded: an interrupted commit leaves no trace.

Checksums. Each commit appends one line to the space's ``segments.jsonl`` with the
byte ranges it added to every stored file, the rows it killed and a checksum of each
range (xxh3-128 if ``xxhash`` is installed, else blake2b-128; the manifest names the
algorithm). ``rows.i64`` is hashed with the dead column zeroed; the dead column is
checked against the killed rows instead. Lines are hash-chained and the manifest holds
the chain head, so the manifest authenticates the whole log without growing per commit.
``verify()`` checks everything; ``KnowledgeBase(..., verify=True)`` verifies on open.
Stored files are append-only apart from dead marks; live files are not checksummed
(they are mutable training state; checkpoints carry their own checksums).

Search is an exact chunked scan over the memory-mapped key matrix of one space (not
ANN, invariant 8), limited to one KB; searching several KBs requires listing the
datasets the caller may read (learned selection is never authorization, invariant 6).
Queries see only items whose time is at or before their query time (invariant 2).

Live mode (stage L1) keeps fp32 master values, live keys and per-item Adam state beside
the stored rows and updates only the items named in a step. It is mutable training
state, visible only in the writer process: each update is written as a redo journal
first, so it is applied completely or not at all, and ``live_updates`` counts applied
steps (the live generation). Stored rows, keys and payloads are never changed by live
updates, so cursor-pinned reads (and other processes) see the stored state only; live
reads and searches (``live=True``) see the current live state. ``pin_live()`` returns a
``LiveSnapshot`` of the current cursor and generation: while any snapshot is pinned, an
update first saves the pre-image of each item it changes (copy on write, in memory,
only for items changed since the newest pin), so the snapshot keeps reading the values
and keys of its generation. ``checkpoint_live``/``restore_live`` save and restore the
exact live state (values, keys, Adam moments, step counts, ``live_updates``) for resume
with a model checkpoint. ``export_live`` writes the current live values as a new frozen
KB; ``compact`` writes a new KB without superseded rows.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

SCHEMA = 'schnitz.kb/2'
PRODUCERS = ('codec', 'rewrite', 'live-update')
# rows.i64 columns
OFFSET, LENGTH, BORN, DEAD, TIME, VERSION, META_OFF, META_LEN = range(8)
STORED_FILES = ('keys.f32', 'rows.i64', 'mass.f32', 'payload.bf16', 'ids.txt', 'meta.jsonl')
LIVE_FILES = ('live_values.f32', 'live_m.f32', 'live_v.f32', 'live_step.i64', 'live_keys.f32')
CHECKPOINTS = 'live_checkpoints'
_TAG = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
_CHUNK = 1 << 24


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
    shares: tuple[float, ...] = ()      # responsibility share of each lineage entry
    derived: bool = False               # composition flows through lineage


@dataclass
class SearchHits:
    """Per query: hits in descending score order (fewer than k if fewer are visible)."""
    ids: list[list[str]]
    versions: list[list[int]]
    scores: list[Tensor]
    datasets: list[list[str]]
    cursor: dict[str, int]      # dataset -> cursor the search read
    items: list[list[Item]] | None = field(default=None)


class IntegrityError(ValueError):
    """A stored file does not match its recorded checksum or segment log."""


def _hash_algorithm() -> str:
    try:
        import xxhash  # noqa: F401
        return 'xxh3_128'
    except ImportError:
        return 'blake2b-128'


def _hasher(algorithm: str):
    if algorithm == 'xxh3_128':
        import xxhash
        return xxhash.xxh3_128()
    if algorithm == 'blake2b-128':
        return hashlib.blake2b(digest_size=16)
    raise ValueError(f'unknown checksum algorithm {algorithm!r}')


def _digest(algorithm: str, data: bytes) -> str:
    h = _hasher(algorithm)
    h.update(data)
    return h.hexdigest()


def _file_digest(algorithm: str, path: Path, start: int, end: int, *,
                 row_bytes: int = 0) -> str:
    """Checksum of bytes [start, end) of a file; ``row_bytes`` (rows.i64) zeroes the
    dead column of every row, which is the only part of a stored file changed in place."""
    h = _hasher(algorithm)
    with path.open('rb') as handle:
        handle.seek(start)
        remaining = end - start
        step = _CHUNK - _CHUNK % row_bytes if row_bytes else _CHUNK
        while remaining > 0:
            data = handle.read(min(step, remaining))
            if not data:
                raise IntegrityError(f'{path} is shorter than its committed size')
            remaining -= len(data)
            if row_bytes:
                rows = np.frombuffer(data, '<i8').reshape(-1, 8).copy()
                rows[:, DEAD] = 0
                data = rows.tobytes()
            h.update(data)
    return h.hexdigest()


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


def _copy_hashed(algorithm: str, src: Path, dst: Path, size: int) -> str:
    """Copy the first ``size`` bytes of ``src`` to ``dst`` (fsynced); returns their hash."""
    h = _hasher(algorithm)
    with src.open('rb') as fin, dst.open('wb') as fout:
        remaining = size
        while remaining > 0:
            data = fin.read(min(_CHUNK, remaining))
            if not data:
                raise ValueError(f'{src} is shorter than {size} bytes')
            h.update(data)
            fout.write(data)
            remaining -= len(data)
        fout.flush()
        os.fsync(fout.fileno())
    return h.hexdigest()


def _ref(item_id: str, version: int) -> str:
    return f'{item_id}@{version}'


class LiveSnapshot:
    """A pinned (cursor, live generation) view of a live KB in the writer process.

    Reads and searches see the stored rows of ``cursor`` with the live values and keys
    of ``generation``, whatever updates follow. Release it (or use ``with``) so the
    writer can drop the saved pre-images."""

    def __init__(self, kb: KnowledgeBase, token: int, cursor: int, generation: int):
        self.kb, self._token, self.cursor, self.generation = kb, token, cursor, generation

    def _check(self) -> None:
        if self._token not in self.kb._pins:
            raise RuntimeError('live snapshot was released')

    def read(self, space: str, ids: Sequence[str]) -> list[Item]:
        with self.kb._lock:
            self._check()
            return self.kb._read(space, ids, None, self.kb.is_live(space), self.cursor,
                                 self.generation)

    def search(self, space: str, queries: Tensor, k: int, **options) -> SearchHits:
        with self.kb._lock:
            self._check()
            return self.kb._search(space, queries, k, live=self.kb.is_live(space),
                                   cursor=self.cursor,
                                   generation=self.generation, **options)

    def release(self) -> None:
        self.kb._release(self._token)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class KnowledgeBase:
    """One KB directory. ``writable=True`` takes the exclusive writer lock and recovers;
    ``verify=True`` checks every committed segment's checksum after opening."""

    def __init__(self, root: str | Path, *, writable: bool = False, verify: bool = False):
        self.root = Path(root)
        self.writable = writable
        self._lock = threading.RLock()
        self._lockfile = None
        self._maps: dict[tuple[str, str], np.ndarray] = {}
        self._pins: dict[int, int] = {}                     # token -> generation
        self._pre: dict[str, dict[int, list]] = {}          # space -> row -> pre-images
        self._next_pin = 0
        if writable:
            self._lockfile = (self.root / 'writer.lock').open('a+')
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lockfile.close()
                raise RuntimeError(f'{self.root} already has a writer') from None
        self._manifest = json.loads((self.root / 'manifest.json').read_text())
        if self._manifest.get('schema') != SCHEMA:
            self.close()
            raise ValueError(f'unsupported KB schema {self._manifest.get("schema")!r}')
        if writable and self._manifest['frozen']:
            self.close()
            raise PermissionError('frozen KB (an export) cannot be written')
        self.spaces = {name: SpaceSpec(s['width'], s['key_width'], s['ratio'])
                       for name, s in self._manifest['spaces'].items()}
        if writable:
            self._recover()
        self._build_index()
        if verify:
            self.verify()

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
                    'checksum': _hash_algorithm(), 'history': None, 'spaces': {}}
        for space, spec in spaces.items():
            if not space.isidentifier() or spec.width < 1 or spec.key_width < 1:
                raise ValueError(f'invalid space {space!r}')
            (root / space).mkdir()
            for file in STORED_FILES + ('segments.jsonl',):
                (root / space / file).touch()
            manifest['spaces'][space] = {'width': spec.width, 'key_width': spec.key_width,
                                         'ratio': spec.ratio, 'items': 0, 'positions': 0,
                                         'ids_bytes': 0, 'meta_bytes': 0,
                                         'segments_bytes': 0, 'chain': ''}
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

    @property
    def frozen(self) -> bool:
        return self._manifest['frozen']

    def is_live(self, space: str) -> bool:
        return space in self._manifest['live_spaces']

    def refresh(self) -> None:
        """Reader: move to the writer's latest committed cursor."""
        with self._lock:
            self._manifest = json.loads((self.root / 'manifest.json').read_text())
            self._maps.clear()
            self._build_index()

    # -- files ---------------------------------------------------------------------------

    def _sizes(self, space: str, manifest: dict | None = None) -> dict[str, int]:
        """Committed byte size of every file of a space."""
        manifest = self._manifest if manifest is None else manifest
        s = manifest['spaces'][space]
        n, p, w = s['items'], s['positions'], s['width']
        sizes = {'keys.f32': n * s['key_width'] * 4, 'rows.i64': n * 64, 'mass.f32': n * 4,
                 'payload.bf16': p * w * 2, 'ids.txt': s['ids_bytes'],
                 'meta.jsonl': s['meta_bytes'], 'segments.jsonl': s['segments_bytes']}
        if space in manifest['live_spaces']:
            sizes.update({'live_values.f32': p * w * 4, 'live_m.f32': p * w * 4,
                          'live_v.f32': p * w * 4, 'live_step.i64': n * 8,
                          'live_keys.f32': n * s['key_width'] * 4})
        return sizes

    def _map(self, space: str, file: str) -> np.ndarray:
        """Committed prefix of a file as a (possibly writable) memory map."""
        cached = self._maps.get((space, file))
        if cached is not None:
            return cached
        s = self._manifest['spaces'][space]
        n, p, w, kw = s['items'], s['positions'], s['width'], s['key_width']
        dtype, shape = {'keys.f32': (np.float32, (n, kw)),
                        'rows.i64': (np.int64, (n, 8)), 'mass.f32': (np.float32, (n,)),
                        'payload.bf16': (np.int16, (p, w)),
                        'meta.jsonl': (np.uint8, (s['meta_bytes'],)),
                        'live_values.f32': (np.float32, (p, w)),
                        'live_m.f32': (np.float32, (p, w)), 'live_v.f32': (np.float32, (p, w)),
                        'live_step.i64': (np.int64, (n,)),
                        'live_keys.f32': (np.float32, (n, kw))}[file]
        if shape[0] == 0:
            array = np.zeros(shape, dtype)
        else:
            array = np.memmap(self.root / space / file, dtype=dtype,
                              mode='r+' if self.writable else 'r', shape=shape)
        self._maps[(space, file)] = array
        return array

    def _recover(self) -> None:
        """Discard everything an interrupted commit, live update or checkpoint left
        behind; finish an interrupted restore."""
        for pending in sorted(self.root.glob('**/*.pending'), reverse=True):
            if pending.is_dir():
                shutil.rmtree(pending)
            elif pending.exists():
                pending.unlink()
        marker = self.root / 'live.restore'
        if marker.exists():
            self._finish_restore(json.loads(marker.read_text())['tag'])
            return
        self._truncate()
        journal = self.root / 'live.journal'
        if journal.exists():
            self._apply_journal(journal)

    def _truncate(self) -> None:
        """Cut every file to its committed size and clear dead marks past the cursor."""
        self._maps.clear()
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

    def _live_state(self, space: str, row: int, generation: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Live (values, key) of a row, as of ``generation`` (None: now)."""
        if generation is not None:
            for replaced_at, values, key in self._pre.get(space, {}).get(row, ()):
                if replaced_at > generation:
                    return values, key
        r = self._map(space, 'rows.i64')[row]
        start, length = int(r[OFFSET]), int(r[LENGTH])
        return (np.array(self._map(space, 'live_values.f32')[start:start + length]),
                np.array(self._map(space, 'live_keys.f32')[row]))

    def _item(self, space: str, row: int, cursor: int, live: bool,
              generation: int | None = None) -> Item:
        r = self._map(space, 'rows.i64')[row]
        start, length = int(r[OFFSET]), int(r[LENGTH])
        if live:
            values, key = self._live_state(space, row, generation)
            values, key = torch.from_numpy(values), torch.from_numpy(key)
        else:
            values = torch.from_numpy(np.array(self._map(space, 'payload.bf16')[start:start + length]))
            values = values.view(torch.bfloat16)
            key = torch.from_numpy(np.array(self._map(space, 'keys.f32')[row]))
        meta = self._meta(space, row)
        return Item(meta['id'], int(r[VERSION]), space, values, key,
                    float(self._map(space, 'mass.f32')[row]), int(r[TIME]),
                    Provenance(tuple(meta['sources']), meta['producer'], meta['step'],
                               meta['dataset']),
                    tuple((e[0], e[1]) for e in meta['lineage']),
                    bool(self._visible(r[None], cursor)[0]),
                    tuple(float(e[2]) for e in meta['lineage']), bool(meta['derived']))

    def _check_live(self, space: str, cursor: int | None) -> None:
        if not (self.writable and self.is_live(space)) or cursor not in (None, self.cursor):
            raise ValueError('live state exists only for a live space in the writer, at the '
                             'current cursor (use pin_live for a stable view)')

    def read(self, space: str, ids: Sequence[str], *, versions: Sequence[int | None] | None = None,
             live: bool = False, cursor: int | None = None) -> list[Item]:
        """Items by id: the current version at ``cursor``, or an explicit (possibly
        superseded) version for measurement. ``live`` returns fp32 live values and live
        keys (writer only, current cursor)."""
        with self._lock:
            self._check_space(space)
            if live:
                self._check_live(space, cursor)
            return self._read(space, ids, versions, live, self._view(cursor), None)

    def _read(self, space, ids, versions, live, cursor, generation) -> list[Item]:
        self._check_space(space)
        versions = [None] * len(ids) if versions is None else list(versions)
        return [self._item(space, self._row(space, i, cursor, v), cursor, live, generation)
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
                versions: Sequence[int], lineages: Sequence[Sequence[tuple[str, int, float]]],
                kill: Sequence[int] = (), derived: Sequence[bool] | None = None) -> None:
        """Append rows, mark ``kill`` rows superseded, log the segment, publish one new
        cursor."""
        if not self.writable:
            raise PermissionError('KB opened read-only')
        derived = [False] * len(items) if derived is None else derived
        spec, s = self.spaces[space], dict(self._manifest['spaces'][space])
        algorithm = self._manifest['checksum']
        cursor = self.cursor + 1
        n, p = s['items'], s['positions']
        keys = np.stack([i.key.detach().float().cpu().numpy() for i in items]).astype('<f4')
        values = torch.cat([i.values.detach().cpu().to(torch.bfloat16) for i in items])
        rows = np.zeros((len(items), 8), np.int64)
        meta, id_bytes, offset, meta_offset = [], [], p, s['meta_bytes']
        for j, (item, item_id, version, lineage, der) in enumerate(
                zip(items, ids, versions, lineages, derived, strict=True)):
            if not item_id or '\n' in item_id:
                raise ValueError('item ids are nonempty single-line strings')
            line = json.dumps({'id': item_id, 'sources': list(item.provenance.sources),
                               'producer': item.provenance.producer,
                               'step': int(item.provenance.step), 'dataset': self.dataset,
                               'lineage': [[i, int(v), float(sh)] for i, v, sh in lineage],
                               'derived': bool(der)}).encode() + b'\n'
            rows[j] = (offset, item.values.shape[0], cursor, 0, item.time, version,
                       meta_offset, len(line) - 1)
            offset += item.values.shape[0]
            meta_offset += len(line)
            meta.append(line)
            id_bytes.append(item_id.encode() + b'\n')
        root = self.root / space
        data = {'keys.f32': keys.tobytes(), 'rows.i64': rows.astype('<i8').tobytes(),
                'mass.f32': np.asarray([i.mass for i in items], '<f4').tobytes(),
                'payload.bf16': values.contiguous().view(torch.int16).numpy().tobytes(),
                'ids.txt': b''.join(id_bytes), 'meta.jsonl': b''.join(meta)}
        at = {'keys.f32': n * spec.key_width * 4, 'rows.i64': n * 64, 'mass.f32': n * 4,
              'payload.bf16': p * spec.width * 2, 'ids.txt': s['ids_bytes'],
              'meta.jsonl': s['meta_bytes']}
        for file in STORED_FILES:
            _write_at(root / file, at[file], data[file])
        segment = {'cursor': cursor, 'items': [n, n + len(items)], 'positions': [p, offset],
                   'ids_bytes': [s['ids_bytes'], s['ids_bytes'] + len(data['ids.txt'])],
                   'meta_bytes': [s['meta_bytes'], meta_offset],
                   'kill': sorted(int(k) for k in kill), 'prev': s['chain'],
                   'hash': {file: _digest(algorithm, data[file]) for file in STORED_FILES}}
        seg_line = json.dumps(segment, sort_keys=True).encode() + b'\n'
        _write_at(root / 'segments.jsonl', s['segments_bytes'], seg_line)
        if self.is_live(space):
            # live values start from the stored (bf16-rounded) payload, moments at zero
            start = values.float().numpy().astype('<f4').tobytes()
            zeros = bytes(len(start))
            for file, blob in (('live_values.f32', start), ('live_m.f32', zeros),
                               ('live_v.f32', zeros)):
                _write_at(root / file, p * spec.width * 4, blob)
            _write_at(root / 'live_step.i64', n * 8, bytes(8 * len(items)))
            _write_at(root / 'live_keys.f32', n * spec.key_width * 4, data['keys.f32'])
        if kill:
            dead = self._map(space, 'rows.i64')
            dead[list(kill), DEAD] = cursor
            dead.flush()
        s.update(items=n + len(items), positions=offset, meta_bytes=meta_offset,
                 ids_bytes=s['ids_bytes'] + len(data['ids.txt']),
                 segments_bytes=s['segments_bytes'] + len(seg_line),
                 chain=_digest(algorithm, seg_line))
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
        Old versions stay readable by explicit version but are no longer searched. A new
        version that names no sources is derived from the old one (share 1) and inherits
        its sources; one that names sources is a fresh encoding of those sources."""
        with self._lock:
            self._check_space(space)
            ids = [item.id for item in items]
            if not items or None in ids or len(set(ids)) != len(ids):
                raise ValueError('supersede needs distinct existing ids')
            rows = [self._row(space, i, self.cursor) for i in ids]
            table = self._map(space, 'rows.i64')
            old = [int(table[r, VERSION]) for r in rows]
            fixed, derived = [], []
            for item, row in zip(items, rows):
                self._validate(space, item)
                provenance = item.provenance
                derived.append(not provenance.sources)
                if not provenance.sources:
                    provenance = Provenance(tuple(self._meta(space, row)['sources']),
                                            provenance.producer, provenance.step,
                                            provenance.dataset)
                # a new version is never available before the version it replaces
                fixed.append(NewItem(item.values, item.key, provenance, item.mass,
                                     max(item.time, int(table[row, TIME])), item.id))
            lineage = [((i, v, 1.0),) for i, v in zip(ids, old)]
            self._commit(space, fixed, ids, [v + 1 for v in old], lineage, rows, derived)
            return [(i, v + 1) for i, v in zip(ids, old)]

    @staticmethod
    def _share_matrix(inputs: Sequence[str], n_out: int, shares) -> np.ndarray:
        if shares is None:
            if n_out != 1:
                raise ValueError('a rewrite with several outputs needs explicit shares '
                                 '(invariant 7)')
            return np.ones((1, len(inputs)))
        if isinstance(shares, Sequence) and shares and isinstance(shares[0], Mapping):
            if len(shares) != n_out:
                raise ValueError('one share mapping per output')
            index = {i: j for j, i in enumerate(inputs)}
            matrix = np.zeros((n_out, len(inputs)))
            for o, mapping in enumerate(shares):
                for input_id, share in mapping.items():
                    if input_id not in index:
                        raise ValueError(f'share for {input_id!r}, which is not an input')
                    matrix[o, index[input_id]] = float(share)
            return matrix
        if isinstance(shares, Tensor):
            shares = shares.detach().cpu().double().numpy()
        matrix = np.array(shares, dtype=np.float64)
        if matrix.shape != (n_out, len(inputs)):
            raise ValueError(f'shares are (outputs, inputs) = ({n_out}, {len(inputs)})')
        return matrix

    def rewrite(self, space: str, inputs: Sequence[str], outputs: Sequence[NewItem], *,
                shares: Sequence[Mapping[str, float]] | np.ndarray | Tensor | None = None,
                tolerance: float = 1e-4) -> list[str]:
        """Replace the current ``inputs`` by ``outputs`` (new ids) in one commit.

        ``shares[o]`` gives output o's responsibility share of each input (a mapping
        input id -> share, or an (outputs, inputs) matrix); optional only for a single
        output (share 1 of every input). Shares are nonnegative, each input's shares
        sum to one within ``tolerance`` (then normalized exactly) and every output has
        a positive share of some input. Each output's mass must equal the share-weighted
        sum of its inputs' masses within ``tolerance`` (relative) and is stored as that
        sum, so mass is conserved exactly (invariants 5 and 7). An output's lineage names
        the exact (id, version, share) of the inputs it has a positive share of, its
        sources are theirs (an output naming other sources is refused) and its time is at
        least their latest time. Compaction is the case with fewer outputs."""
        with self._lock:
            self._check_space(space)
            if not inputs or not outputs or len(set(inputs)) != len(inputs):
                raise ValueError('rewrite needs distinct inputs and at least one output')
            rows = [self._row(space, i, self.cursor) for i in inputs]
            table, mass = self._map(space, 'rows.i64'), self._map(space, 'mass.f32')
            masses = np.array([float(mass[r]) for r in rows])
            matrix = self._share_matrix(inputs, len(outputs), shares)
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError('shares are finite and nonnegative')
            totals = matrix.sum(0)
            for input_id, total in zip(inputs, totals):
                if abs(total - 1.0) > tolerance:
                    raise ValueError(f'shares of input {input_id!r} sum to {total}, not 1 '
                                     '(invariant 7)')
            matrix = matrix / totals
            if (matrix.sum(1) <= 0).any():
                raise ValueError('every output needs a positive share of some input')
            expected = matrix @ masses
            versions = [int(table[r, VERSION]) for r in rows]
            times = [int(table[r, TIME]) for r in rows]
            source_lists = [self._meta(space, r)['sources'] for r in rows]
            fixed, lineages = [], []
            for o, item in enumerate(outputs):
                self._validate(space, item)
                if abs(item.mass - expected[o]) > tolerance * max(1.0, abs(expected[o])):
                    raise ValueError(f'output {o} has mass {item.mass}, but its shares of the '
                                     f'inputs carry {expected[o]}')
                used = np.flatnonzero(matrix[o] > 0).tolist()
                sources = tuple(sorted({s for i in used for s in source_lists[i]}))
                provenance = item.provenance
                if provenance.sources and set(provenance.sources) != set(sources):
                    raise ValueError('a rewrite output inherits the sources of its inputs')
                provenance = Provenance(sources, provenance.producer, provenance.step,
                                        provenance.dataset)
                fixed.append(NewItem(item.values, item.key, provenance, float(expected[o]),
                                     max([item.time] + [times[i] for i in used]), item.id))
                lineages.append(tuple((inputs[i], versions[i], float(matrix[o, i]))
                                      for i in used))
            ids = [item.id or uuid.uuid4().hex for item in fixed]
            if len(set(ids)) != len(ids) or any(self.has(i) for i in ids):
                raise ValueError('rewrite outputs need new, distinct ids')
            self._commit(space, fixed, ids, [1] * len(ids), lineages, rows,
                         [True] * len(ids))
            return ids

    # -- lineage, composition ------------------------------------------------------------

    def _history(self) -> list[dict]:
        history = self._manifest.get('history')
        if not history:
            return []
        with (self.root / 'history.jsonl').open('rb') as handle:
            data = handle.read(history['bytes'])
        return [json.loads(line) for line in data.splitlines()]

    def lineage(self, *, cursor: int | None = None) -> dict[str, dict[str, float]]:
        """Weighted lineage graph of every item version born by ``cursor`` (and of the
        history of an exported or compacted KB), keyed ``'<id>@<version>'``, in the form
        ``schnitz.kb_eval.source_composition`` resolves: a derived item weighs each
        lineage input by share times the input's mass (by the shares alone if all those
        masses are zero); any other item weighs its own sources equally."""
        with self._lock:
            cursor = self._view(cursor)
            nodes = []   # (ref, mass, sources, lineage, derived)
            for entry in self._history():
                nodes.append((_ref(entry['id'], entry['version']), entry['mass'],
                              entry['sources'], entry['lineage'], entry['derived']))
            for space in self.spaces:
                table, mass = self._map(space, 'rows.i64'), self._map(space, 'mass.f32')
                for row in np.flatnonzero(table[:, BORN] <= cursor).tolist():
                    meta = self._meta(space, row)
                    nodes.append((_ref(meta['id'], int(table[row, VERSION])),
                                  float(mass[row]), meta['sources'], meta['lineage'],
                                  meta['derived']))
            masses = {ref: m for ref, m, *_ in nodes}
            out: dict[str, dict[str, float]] = {}
            for ref, _, sources, lineage, derived in nodes:
                if derived and lineage:
                    weights = {}
                    for input_id, version, share in lineage:
                        key = _ref(input_id, version)
                        if key not in masses:
                            raise ValueError(f'lineage input {key} of {ref} is missing')
                        weights[key] = share * masses[key]
                    if sum(weights.values()) <= 0:
                        weights = {_ref(i, v): share for i, v, share in lineage}
                else:
                    weights = dict.fromkeys(sources, 1.0)
                out[ref] = weights
            clash = {s for ref, _, sources, lineage, derived in nodes
                     if not (derived and lineage) for s in sources if s in out}
            if clash:
                raise ValueError(f'source ids collide with item refs: {sorted(clash)[:5]}')
            return out

    def source_composition(self, *, cursor: int | None = None) -> dict[str, dict[str, float]]:
        """Source shares of every item current at ``cursor``, resolved through the
        share-weighted lineage (``schnitz.kb_eval.source_composition``)."""
        from schnitz.kb_eval import source_composition
        with self._lock:
            cursor = self._view(cursor)
            composition = source_composition(self.lineage(cursor=cursor))
            out = {}
            for space in self.spaces:
                table = self._map(space, 'rows.i64')
                for row in np.flatnonzero(self._visible(table, cursor)).tolist():
                    item_id = self._row_ids[space][row]
                    out[item_id] = composition[_ref(item_id, int(table[row, VERSION]))]
            return out

    # -- search --------------------------------------------------------------------------

    def search(self, space: str, queries: Tensor, k: int, *, metric: str = 'cosine',
               query_time: int | Sequence[int] | Tensor | None = None,
               cursor: int | None = None, chunk_rows: int = 32768,
               return_items: bool = False, live: bool = False) -> SearchHits:
        """Exact top-k over the current keys of one space of this KB.

        ``queries`` is (batch, key_width); ``metric`` is 'cosine' or 'dot'. Keys are
        streamed from the memory-mapped matrix ``chunk_rows`` at a time, so memory is
        bounded by the chunk, not the KB. Superseded items and items whose time is after
        the query's time are never returned. ``live`` scans the live keys (writer only,
        current cursor; ``pin_live`` for a stable view) instead of the stored keys."""
        with self._lock:
            if live:
                self._check_live(space, cursor)
            return self._search(space, queries, k, metric=metric, query_time=query_time,
                                cursor=cursor, chunk_rows=chunk_rows,
                                return_items=return_items, live=live)

    def _search(self, space, queries, k, *, metric='cosine', query_time=None, cursor=None,
                chunk_rows=32768, return_items=False, live=False,
                generation=None) -> SearchHits:
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
        keys = self._map(space, 'live_keys.f32' if live else 'keys.f32')
        rows = self._map(space, 'rows.i64')
        override = {}
        if live and generation is not None:
            override = {row: self._live_state(space, row, generation)[1]
                        for row in self._pre.get(space, {})}
        for start in range(0, len(keys), chunk_rows):
            block = np.array(rows[start:start + chunk_rows])
            ok = torch.from_numpy(self._visible(block, cursor))
            ok = ok[None] & (torch.from_numpy(block[:, TIME])[None] <= times[:, None])
            if not ok.any():
                continue
            chunk = np.array(keys[start:start + chunk_rows])
            for row, key in override.items():
                if start <= row < start + len(chunk):
                    chunk[row - start] = key
            kk = torch.from_numpy(chunk)
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
            hits.items = [[self._item(space, r, cursor, live, generation) for r in
                           best_rows[b][torch.isfinite(best[b])].tolist()]
                          for b in range(len(q))]
        return hits

    # -- live mode (L1) ------------------------------------------------------------------

    def enable_live(self, space: str) -> None:
        """Start live mode: fp32 values from the stored payloads, live keys from the stored
        keys, Adam state at zero."""
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
            _copy_hashed(self._manifest['checksum'], root / 'keys.f32', root / 'live_keys.f32',
                         s['items'] * s['key_width'] * 4)
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
        """Replace the live keys of items (e.g. from the trained key head); stored keys
        are unchanged. Live searches (``live=True``) scan the live keys."""
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
        update = int(meta['update'])
        if self._pins:   # copy on write for pinned snapshots
            newest = max(self._pins.values())
            saved = self._pre.setdefault(space, {})
            for row in rows.tolist():
                entries = saved.get(row)
                if entries and newest < entries[-1][0]:
                    continue    # no pin has seen the current state of this row
                values, key = self._live_state(space, row, None)
                saved.setdefault(row, []).append((update, values, key))
        targets = {'p.values': 'live_values.f32', 'p.m': 'live_m.f32', 'p.v': 'live_v.f32',
                   'r.step': 'live_step.i64', 'r.keys': 'live_keys.f32'}
        for name, tensor in tensors.items():
            if name in targets:
                array = self._map(space, targets[name])
                array[positions if name.startswith('p.') else rows] = tensor.numpy()
                array.flush()
        manifest = dict(self._manifest, live_updates=update)
        _atomic_json(self.root / 'manifest.json', manifest)
        self._manifest = manifest
        journal.unlink()
        _fsync_dir(self.root)

    def pin_live(self) -> LiveSnapshot:
        """Pin the current cursor and live generation (writer process only)."""
        with self._lock:
            if not self.writable:
                raise ValueError('live state exists only in the writer')
            token = self._next_pin
            self._next_pin += 1
            self._pins[token] = self.live_updates
            return LiveSnapshot(self, token, self.cursor, self.live_updates)

    def _release(self, token: int) -> None:
        with self._lock:
            if self._pins.pop(token, None) is None:
                return
            pins = list(self._pins.values())
            for space, saved in self._pre.items():
                for row in list(saved):
                    kept, previous = [], -1
                    for entry in saved[row]:
                        if any(previous <= g < entry[0] for g in pins):
                            kept.append(entry)
                        previous = entry[0]
                    if kept:
                        saved[row] = kept
                    else:
                        del saved[row]

    # -- live checkpoints ----------------------------------------------------------------

    def checkpoint_live(self, tag: str) -> dict:
        """Save the exact live state (values, keys, Adam moments, step counts,
        ``live_updates``) and the manifest under ``live_checkpoints/<tag>``, built under
        ``<tag>.pending`` and renamed, with a checksum per file. Costs a full copy of the
        live files (three fp32 copies of every payload)."""
        with self._lock:
            if not self.writable:
                raise PermissionError('KB opened read-only')
            if not _TAG.fullmatch(tag) or tag.endswith('.pending'):
                raise ValueError(f'invalid checkpoint tag {tag!r}')
            base = self.root / CHECKPOINTS
            final, pending = base / tag, base / (tag + '.pending')
            if final.exists():
                raise FileExistsError(final)
            shutil.rmtree(pending, ignore_errors=True)
            pending.mkdir(parents=True)
            algorithm = self._manifest['checksum']
            files = {}
            for space in self._manifest['live_spaces']:
                (pending / space).mkdir()
                for file in LIVE_FILES:
                    size = self._sizes(space)[file]
                    files[f'{space}/{file}'] = {
                        'bytes': size, 'hash': _copy_hashed(algorithm, self.root / space / file,
                                                            pending / space / file, size)}
                _fsync_dir(pending / space)
            info = {'tag': tag, 'cursor': self.cursor, 'live_updates': self.live_updates,
                    'checksum': algorithm, 'files': files, 'manifest': self._manifest}
            _atomic_json(pending / 'checkpoint.json', info)
            os.replace(pending, final)
            _fsync_dir(base)
            return {k: info[k] for k in ('tag', 'cursor', 'live_updates')}

    def live_checkpoints(self) -> list[str]:
        base = self.root / CHECKPOINTS
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir()
                      if (p / 'checkpoint.json').exists() and not p.name.endswith('.pending'))

    def drop_live_checkpoint(self, tag: str) -> None:
        with self._lock:
            if not self.writable:
                raise PermissionError('KB opened read-only')
            if tag not in self.live_checkpoints():
                raise KeyError(tag)
            path = self.root / CHECKPOINTS / tag
            os.replace(path, path.with_name(tag + '.pending'))   # atomic removal
            shutil.rmtree(path.with_name(tag + '.pending'))

    def _segment_head(self, space: str, size: int) -> str:
        """Chain hash of the segment log truncated to ``size`` bytes ('' when empty)."""
        if size == 0:
            return ''
        with (self.root / space / 'segments.jsonl').open('rb') as handle:
            data = handle.read(size)
        if len(data) != size or not data.endswith(b'\n'):
            return 'mismatch'
        last = data[:-1].rsplit(b'\n', 1)[-1] + b'\n'
        return _digest(self._manifest['checksum'], last)

    def restore_live(self, tag: str, *, discard_commits: bool = False,
                     verify: bool = True) -> None:
        """Restore the live state saved by ``checkpoint_live(tag)`` exactly.

        The checkpoint's stored state must be a prefix of the current one (same segment
        chain up to its cursor). If commits followed the checkpoint, they are refused
        unless ``discard_commits``, which rolls the stored rows back to the checkpoint's
        cursor too; later cursors are then reused, so no other process may have the KB
        open. Interrupted restores are finished at the next writer open."""
        with self._lock:
            if not self.writable:
                raise PermissionError('KB opened read-only')
            if self._pins:
                raise RuntimeError('release live snapshots before restoring')
            final = self.root / CHECKPOINTS / tag
            info = json.loads((final / 'checkpoint.json').read_text())
            target = info['manifest']
            for space, s in target['spaces'].items():
                current = self._manifest['spaces'].get(space)
                if current is None or current['segments_bytes'] < s['segments_bytes'] \
                        or self._segment_head(space, s['segments_bytes']) != s['chain']:
                    raise ValueError(f'checkpoint {tag} is not an ancestor of the stored state')
            if target['cursor'] != self.cursor and not discard_commits:
                raise ValueError(f'commits followed checkpoint {tag} (cursor '
                                 f'{target["cursor"]} -> {self.cursor}); pass discard_commits')
            if verify:
                for rel, f in info['files'].items():
                    path = final / rel
                    if path.stat().st_size != f['bytes'] or _file_digest(
                            info['checksum'], path, 0, f['bytes']) != f['hash']:
                        raise IntegrityError(f'checkpoint file {path} is corrupt')
            _atomic_json(self.root / 'live.restore', {'tag': tag})
            self._finish_restore(tag)

    def _finish_restore(self, tag: str) -> None:
        final = self.root / CHECKPOINTS / tag
        info = json.loads((final / 'checkpoint.json').read_text())
        target = info['manifest']
        self._maps.clear()
        self._pre.clear()
        for space in target['live_spaces']:
            for file in LIVE_FILES:
                dst = self.root / space / file
                pending = dst.with_name(file + '.pending')
                shutil.copyfile(final / space / file, pending)
                with pending.open('rb') as handle:
                    os.fsync(handle.fileno())
                os.replace(pending, dst)
            _fsync_dir(self.root / space)
        journal = self.root / 'live.journal'
        journal.unlink(missing_ok=True)     # never present at restore; defensive
        _atomic_json(self.root / 'manifest.json', target)
        self._manifest = target
        self._truncate()
        (self.root / 'live.restore').unlink()
        _fsync_dir(self.root)
        self._build_index()

    # -- export, compaction, verification ------------------------------------------------

    def _write_history(self, pending: Path, out: KnowledgeBase) -> None:
        """Metadata of every row not current at the cursor (plus this KB's own history)
        as ``history.jsonl``, so lineage keeps resolving without the dropped payloads."""
        lines = [json.dumps(entry).encode() + b'\n' for entry in self._history()]
        where = f'{self.name}@{self.cursor}'
        for space in self.spaces:
            table, mass = self._map(space, 'rows.i64'), self._map(space, 'mass.f32')
            for row in np.flatnonzero(~self._visible(table, self.cursor)).tolist():
                meta = self._meta(space, row)
                lines.append(json.dumps({
                    'space': space, 'id': meta['id'], 'version': int(table[row, VERSION]),
                    'mass': float(mass[row]), 'time': int(table[row, TIME]),
                    'sources': meta['sources'], 'producer': meta['producer'],
                    'step': meta['step'], 'lineage': meta['lineage'],
                    'derived': meta['derived'], 'born': int(table[row, BORN]),
                    'dead': int(table[row, DEAD]), 'kb': where}).encode() + b'\n')
        data = b''.join(lines)
        (pending / 'history.jsonl').write_bytes(data)
        with (pending / 'history.jsonl').open('rb') as handle:
            os.fsync(handle.fileno())
        out._manifest = dict(out._manifest, history={
            'bytes': len(data), 'hash': _digest(out._manifest['checksum'], data)})

    def _copy_current(self, dest: Path, name: str, origin: dict, *, export: bool,
                      batch: int) -> KnowledgeBase:
        dest = Path(dest)
        if dest.exists():
            raise FileExistsError(dest)
        pending = dest.with_name(dest.name + '.pending')
        shutil.rmtree(pending, ignore_errors=True)
        out = KnowledgeBase.create(pending, name=name, dataset=self.dataset, spaces=self.spaces,
                                   origin=origin)
        carried = []
        for space in self.spaces:
            live = self.is_live(space)
            carry = live and not export
            table = self._map(space, 'rows.i64')
            current = np.flatnonzero(self._visible(table, self.cursor))
            if carry:
                carried.append(space)
                for file in LIVE_FILES:
                    (pending / space / file).write_bytes(b'')
            for start in range(0, len(current), batch):
                rows = current[start:start + batch].tolist()
                items = [self._item(space, r, self.cursor, live and export) for r in rows]
                steps = self._map(space, 'live_step.i64')[rows] if live else [0] * len(rows)
                fresh = [NewItem(i.values, i.key,
                                 Provenance(i.provenance.sources, 'live-update', int(s))
                                 if live and export else i.provenance, i.mass, i.time, i.id)
                         for i, s in zip(items, steps)]
                out._commit(space, fresh, [i.id for i in items], [i.version for i in items],
                            [tuple((a, v, sh) for (a, v), sh in zip(i.lineage, i.shares))
                             for i in items], derived=[i.derived for i in items])
                if carry:   # the live state of the kept rows, in the new row order
                    positions = np.concatenate([np.arange(table[r, OFFSET],
                                                          table[r, OFFSET] + table[r, LENGTH])
                                                for r in rows])
                    for file, index in (('live_values.f32', positions), ('live_m.f32', positions),
                                        ('live_v.f32', positions), ('live_step.i64', rows),
                                        ('live_keys.f32', rows)):
                        with (pending / space / file).open('ab') as handle:
                            handle.write(np.ascontiguousarray(self._map(space, file)[index]).tobytes())
                            handle.flush()
                            os.fsync(handle.fileno())
        self._write_history(pending, out)
        manifest = dict(out._manifest, frozen=export or self.frozen, live_spaces=carried,
                        live_updates=0 if export else self.live_updates)
        _atomic_json(pending / 'manifest.json', manifest)
        out.close()
        (pending / 'writer.lock').unlink()
        os.replace(pending, dest)
        _fsync_dir(dest.parent)
        return KnowledgeBase(dest)

    def export_live(self, dest: str | Path, *, name: str | None = None,
                    batch: int = 1024) -> KnowledgeBase:
        """Write every current item as a frozen KB at ``dest``: live spaces contribute their
        live values and keys (bf16, producer 'live-update', step = the item's Adam steps),
        other spaces their stored payloads. Ids, versions and lineage are kept, the rows
        left behind are summarized in ``history.jsonl``; the manifest's origin names this
        KB, its cursor and ``live_updates``. Built under ``dest.pending``."""
        with self._lock:
            origin = {'name': self.name, 'cursor': self.cursor, 'live_updates': self.live_updates}
            return self._copy_current(dest, name or f'{self.name}@live{self.live_updates}',
                                      origin, export=True, batch=batch)

    def compact(self, dest: str | Path, *, name: str | None = None,
                batch: int = 1024) -> KnowledgeBase:
        """Garbage collection: write the current items (stored payloads and keys, ids,
        versions, lineage, shares, provenance) as a new KB at ``dest``, without the
        superseded rows, whose metadata goes to ``history.jsonl`` so lineage and source
        composition still resolve. Live spaces keep their live state and
        ``live_updates``; a frozen KB's compaction is frozen. Live checkpoints are not
        carried (their row layout is the old one). Returns the new KB read-only."""
        with self._lock:
            origin = {'name': self.name, 'cursor': self.cursor, 'live_updates': self.live_updates,
                      'compacted': True}
            return self._copy_current(dest, name or self.name, origin, export=False, batch=batch)

    def verify(self) -> dict:
        """Check every committed segment against its checksums, the segment chain against
        the manifest, the counts, the dead marks and the history. Raises
        ``IntegrityError``; returns the number of segments per space."""
        with self._lock:
            algorithm = self._manifest['checksum']
            report = {}
            for space in self.spaces:
                s, root = self._manifest['spaces'][space], self.root / space
                unit = {'keys.f32': s['key_width'] * 4, 'rows.i64': 64, 'mass.f32': 4,
                        'payload.bf16': s['width'] * 2, 'ids.txt': 1, 'meta.jsonl': 1}
                span = {'keys.f32': 'items', 'rows.i64': 'items', 'mass.f32': 'items',
                        'payload.bf16': 'positions', 'ids.txt': 'ids_bytes',
                        'meta.jsonl': 'meta_bytes'}
                with (root / 'segments.jsonl').open('rb') as handle:
                    log = handle.read(s['segments_bytes'])
                if len(log) != s['segments_bytes']:
                    raise IntegrityError(f'{root / "segments.jsonl"} is truncated')
                chain, last_cursor = '', 0
                ends = {'items': 0, 'positions': 0, 'ids_bytes': 0, 'meta_bytes': 0}
                dead = np.zeros(s['items'], np.int64)
                lines = log.splitlines(keepends=True)
                for line in lines:
                    seg = json.loads(line)
                    if seg['prev'] != chain or seg['cursor'] <= last_cursor \
                            or seg['cursor'] > self.cursor:
                        raise IntegrityError(f'segment log of space {space} is broken')
                    chain, last_cursor = _digest(algorithm, line), seg['cursor']
                    for what in ends:
                        if seg[what][0] != ends[what]:
                            raise IntegrityError(f'segments of space {space} are not contiguous')
                        ends[what] = seg[what][1]
                    for file in STORED_FILES:
                        a, b = seg[span[file]]
                        got = _file_digest(algorithm, root / file, a * unit[file],
                                           b * unit[file],
                                           row_bytes=64 if file == 'rows.i64' else 0)
                        if got != seg['hash'][file]:
                            raise IntegrityError(f'{root / file} segment at cursor '
                                                 f'{seg["cursor"]} fails its checksum')
                    dead[seg['kill']] = seg['cursor']
                if chain != s['chain'] or any(ends[w] != s[w] for w in ends):
                    raise IntegrityError(f'manifest of space {space} does not match its log')
                marks = np.array(self._map(space, 'rows.i64')[:, DEAD])
                marks[marks > self.cursor] = 0     # a newer writer commit than this view
                if not np.array_equal(marks, dead):
                    raise IntegrityError(f'dead marks of space {space} do not match the log')
                report[space] = len(lines)
            history = self._manifest.get('history')
            if history:
                with (self.root / 'history.jsonl').open('rb') as handle:
                    data = handle.read(history['bytes'])
                if len(data) != history['bytes'] or _digest(algorithm, data) != history['hash']:
                    raise IntegrityError('history.jsonl fails its checksum')
            return report

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
