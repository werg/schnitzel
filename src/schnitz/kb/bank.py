"""Bank creation's record-to-span half (docs/knowledge-base-stack.md, 5.1 step 6), shared
by the L1 KB build (``schnitz.kb.stages.l1 build``) and the writer stage (B4c slot
filling): one implementation of

- ``record_sources``: the records the memory transcripts' slots name, per dataset KB
  (plus optional distractor records of the same KB), with text and ``created_at``;
- ``write_spans``: the writer's single-pass span of each record (the record in
  context under the memory prompt, free-running at the B1 length schedule of a ratio
  level: factor ``length_factors(tokens)[level]``, ``ceil(tokens / factor)`` reps);
- ``SpanCache``: those spans on disk (sharded safetensors plus an index), resumable,
  recording the writer state they came from.

This is an offline bank-creation step by a frozen writer (invariant 1): inference
reads stored payloads and never re-encodes sources.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import contextlib
import json
import math
import os
from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from schnitz.kb.decoder import LEVELS, length_factors


# -- transcripts and their records --------------------------------------------------
class Transcripts:
    """The first ``limit`` transcripts of ``split`` per directory, in file order, read
    lazily by byte offset (the full corpora do not fit in memory as parsed rows)."""

    def __init__(self, dirs: Sequence[Path], split: str, limit: int | None):
        self.entries: list[tuple[Path, int, str]] = []
        for d in dirs:
            path = Path(d) / f'transcripts-{split}.jsonl'
            if not path.exists():
                continue
            with path.open('rb') as handle:
                n = 0
                while limit is None or n < limit:
                    offset = handle.tell()
                    if not handle.readline():
                        break
                    self.entries.append((path, offset, str(d)))
                    n += 1

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        path, offset, d = self.entries[i]
        with path.open('rb') as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        row['_dir'] = d
        return row

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def slots_of(row: dict) -> list[dict]:
    return [m['content']['slot'] for m in row['messages']
            if isinstance(m.get('content'), dict) and 'slot' in m['content']]


def needed_records(rows: Iterable[dict]) -> dict[str, dict[str, str]]:
    """kb -> record id -> transcript dir (whose manifest names the corpus): the records
    a slot reads, its ``alternatives`` (redundant copies of the same fact, all banked
    so a read can find any of them) and its ``neutral`` records (neither positives nor
    negatives of its retrieval loss, but part of the KB and readable)."""
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for slot in slots_of(row):
            if slot['kb'] != row['kb']:
                raise PermissionError(f'{row["episode_id"]}: slot of another KB')
            for r in [*slot['record_ids'], *(slot.get('alternatives') or []),
                      *(slot.get('neutral') or [])]:
                out.setdefault(row['kb'], {})[r] = row['_dir']
    return out


def read_sources(dirs: Iterable[str], wanted: set[str],
                 extra: Mapping[str, int] | None = None) -> dict[str, dict]:
    """Text, time and KB of the ``wanted`` records from each transcript directory's
    corpus (``manifest.json`` ``input``/``sources.jsonl``). ``extra`` asks for up to
    that many more records of a KB (in corpus order: distractors)."""
    records, extra, added = {}, dict(extra or {}), {}
    for d in sorted(set(dirs)):
        manifest = json.loads((Path(d) / 'manifest.json').read_text())
        corpus = Path(manifest['input'])
        with (corpus / 'sources.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                rec = json.loads(line)
                kb = (f'{manifest["kb"]}:{rec.get("domain", "")}'
                      if manifest.get('kb_per_domain') else manifest['kb'])
                take = rec['record_id'] in wanted
                if not take and added.get(kb, 0) < extra.get(kb, 0):
                    take = True
                    added[kb] = added.get(kb, 0) + 1
                if take:
                    records[rec['record_id']] = {'text': rec['text'], 'kb': kb,
                                                 'created_at': int(rec['created_at'])}
    return records


def record_sources(transcript_dirs: Sequence[Path], splits: Mapping[str, int | None],
                   distractors: int = 0, with_writes: bool = False) -> dict[str, dict]:
    """Every record the slots of the first ``splits[split]`` transcripts per directory
    name (``None``: all; ``with_writes``: only transcripts with write sites), plus up to ``distractors`` other records per KB: record id ->
    {'text', 'kb', 'created_at'}. Raises if a slot record is not in its corpus or a
    slot names another KB than its episode."""
    rows = [row for split, limit in splits.items()
            for row in Transcripts(transcript_dirs, split, limit)
            if not with_writes or row.get('write_sites')]
    needed = needed_records(rows)
    wanted = {r for recs in needed.values() for r in recs}
    records = read_sources({d for recs in needed.values() for d in recs.values()}, wanted,
                           {kb: distractors for kb in needed})
    missing = wanted - set(records)
    if missing:
        raise ValueError(f'{len(missing)} slot records are not in the corpus sources')
    return {r: rec for r, rec in records.items() if r in wanted or rec['kb'] in needed}


# -- the write step ---------------------------------------------------------------
def _level(level: str | int) -> int:
    return LEVELS.index(level) if isinstance(level, str) else int(level)


@torch.no_grad()
def write_spans(model, texts: Sequence[str], level: str | int = 's0',
                batch_size: int = 32) -> list[torch.Tensor]:
    """The frozen writer's span of each text: the text in context under the memory
    prompt, free-running for ``ceil(tokens / length_factors(tokens)[level])`` reps.
    Batched by length; returned in input order (float32, on the model's device)."""
    index = _level(level)
    ids = [model.text_ids(t) for t in texts]
    order = sorted(range(len(ids)), key=lambda i: int(ids[i].shape[0]))
    out: list[torch.Tensor | None] = [None] * len(ids)
    scope = model.core.autocast if getattr(model, 'core', None) is not None \
        else contextlib.nullcontext
    for start in range(0, len(order), batch_size):
        picked = order[start:start + batch_size]
        factors = [length_factors(int(ids[i].shape[0]))[index] for i in picked]
        examples = [{'ids': ids[i], 'prompt': 'memory', 'factor': f}
                    for i, f in zip(picked, factors)]
        lengths = [max(1, math.ceil(ids[i].shape[0] / f)) for i, f in zip(picked, factors)]
        with scope():
            spans, _ = model.free_run(examples, lengths)
        for i, span in zip(picked, spans):
            out[i] = span.float()
    return out


def _atomic_json(path: Path, value) -> None:
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=1) + '\n')
    with pending.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(pending, path)


class SpanCache:
    """Writer spans of records on disk: ``shard-NNNNN.safetensors`` (bf16 reps of many
    records concatenated) and ``index.json`` (record id -> shard, offset, count) plus
    ``manifest.json`` (level, writer state, metadata). Built resumably by ``build``;
    ``get`` returns one record's span (bf16)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.index: dict[str, list[int]] = json.loads((self.root / 'index.json').read_text())
        self._handles: dict[int, object] = {}

    def __contains__(self, record_id: str) -> bool:
        return record_id in self.index

    def __len__(self) -> int:
        return len(self.index)

    def get(self, record_id: str) -> torch.Tensor:
        shard, offset, count = self.index[record_id]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.root / f'shard-{shard:05d}.safetensors'), framework='pt')
            self._handles[shard] = handle
        return handle.get_slice('reps')[offset:offset + count]

    @classmethod
    def build(cls, root: str | Path, model, records: Mapping[str, str],
              level: str | int = 's0', *, batch_size: int = 32, shard_records: int = 2048,
              meta: dict | None = None) -> SpanCache:
        """Write the spans of ``records`` (record id -> text) not yet cached. Each shard is
        written then published in the index atomically, so an interrupted build resumes.
        A cache is for one level and one writer: ``meta`` (e.g. the writer state path
        and step) must match an existing cache's."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {'level': LEVELS[_level(level)], 'meta': meta or {}}
        if (root / 'manifest.json').exists():
            old = json.loads((root / 'manifest.json').read_text())
            if old != manifest:
                raise ValueError(f'{root} holds spans of {old}, not {manifest}')
        else:
            _atomic_json(root / 'manifest.json', manifest)
            _atomic_json(root / 'index.json', {})
        index = json.loads((root / 'index.json').read_text())
        shard = 1 + max((s for s, _, _ in index.values()), default=-1)
        todo = [r for r in records if r not in index]
        for start in range(0, len(todo), shard_records):
            chunk = todo[start:start + shard_records]
            spans = write_spans(model, [records[r] for r in chunk], level, batch_size)
            path = root / f'shard-{shard:05d}.safetensors'
            pending = path.with_name(path.name + '.pending')
            save_file({'reps': torch.cat([s.to(torch.bfloat16).cpu() for s in spans])
                       .contiguous()}, str(pending))
            with pending.open('rb') as handle:
                os.fsync(handle.fileno())
            os.replace(pending, path)
            offset = 0
            for r, span in zip(chunk, spans):
                index[r] = [shard, offset, int(span.shape[0])]
                offset += int(span.shape[0])
            _atomic_json(root / 'index.json', index)
            shard += 1
        return cls(root)


def kb_dir(kb: str) -> str:
    """Directory name of a dataset KB (``r6-mixed:musique`` -> ``r6-mixed__musique``):
    the same for its span cache and its store."""
    return re.sub(r'[^A-Za-z0-9_.-]', '__', kb)


def build_caches(root: Path, model, records: Mapping[str, dict], level: str | int = 's1',
                 batch_size: int = 32, meta: dict | None = None) -> dict[str, int]:
    """One ``SpanCache`` per dataset KB under ``root`` for ``records`` (id -> {text,
    kb, ...}, as ``record_sources`` returns). Returns the record count per KB."""
    by_kb: dict[str, dict[str, str]] = {}
    for record_id, rec in records.items():
        by_kb.setdefault(rec['kb'], {})[record_id] = rec['text']
    for kb, texts in sorted(by_kb.items()):
        SpanCache.build(Path(root) / kb_dir(kb), model, texts, level, batch_size=batch_size,
                        meta=meta)
    return {kb: len(texts) for kb, texts in by_kb.items()}
