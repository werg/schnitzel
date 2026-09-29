"""KB-size curriculum: small sub-KBs cut from an existing L1 bank (docs/knowledge-base-stack.md
5.1 step 4; restart plan, 29 September: the retrieval options queue, item 2).

``l1 subkb`` groups the transcripts of a bank (``--group-size`` episodes per group: 1 =
per-episode KBs, the batch size = per-batch KBs), and gives each group its own small KB:
every record its slots read or name as ``alternatives`` (``--with-neutral``: also their
``neutral`` records), plus distractors up to ``--size`` records in all:

- ``random``: records of the same parent KB, uniformly;
- ``teacher``: the teacher's near misses (``schnitz.kb.teacher_keys`` cache): per search
  site of the group its top-ranked records of the parent KB that are neither positives
  nor neutral for *any* slot of the group, round robin over the sites, topped up with
  random ones when the cache runs short;
- ``mixed``: half teacher, half random.

Distractors are never a slot's positive or neutral record, and are visible at the
group's query times. Nothing is re-encoded (invariant 1): the sub-KB's items are the
parent bank's stored items, copied with their ids, payloads (bf16, bit for bit), keys,
masses, times and provenance (sources, producer, step); only leaf items (one bank record
each, no lineage) are copied. A sub-KB is its own authorization domain, cut from exactly
one parent KB (``<parent>#<split><index>``, ``parent_kb``); its transcripts name it as
their KB, so reads, in-batch negatives and hard negatives stay inside it. The teacher
cache and hard negatives of the parent's sites apply to its sub-KBs (their records are
kept only where the sub-KB holds them).

Output: a banks directory like ``l1 build``'s (``banks.json``, ``stack.pt``,
``key_heads_init.pt``, one KB per group) plus ``transcripts/<name>/`` with the rewritten
transcripts (only ``kb`` and the slots' ``kb`` change; ids, records, alternatives,
neutral and provenance are kept, with ``provenance.parent_kb`` added). Train with
``l1 train --banks OUT --transcripts OUT/transcripts/<name>``; a curriculum is a series
of these (small -> the full bank) with ``--init-reader`` from the previous stage.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import shutil

SEP = '#'
MODES = ('random', 'teacher', 'mixed')
TEACHER_TOP = 256


def parent_kb(name: str) -> str:
    """The parent KB of a sub-KB name (a plain KB is its own parent)."""
    return name.split(SEP, 1)[0]


def sub_name(parent: str, split: str, index: int) -> str:
    if SEP in parent:
        raise ValueError(f'{parent!r} is already a sub-KB')
    return f'{parent}{SEP}{split[:1]}{index:05d}'


@dataclass
class ParentIndex:
    """A parent KB's bank records: per record its leaf items per space and its time."""
    items: dict[str, dict[str, list[str]]] = field(default_factory=dict)   # record -> space -> ids
    time: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, kb) -> ParentIndex:
        """From the current items of every space (a record's time is the latest of its
        items' times, so a distractor is visible wherever all its items are)."""
        import numpy as np
        from schnitz.kb_store import TIME
        out = cls()
        for s in kb.spaces:
            table = kb._map(s, 'rows.i64')
            for row in np.flatnonzero(kb._visible(table, kb.cursor)).tolist():
                meta = kb._meta(s, row)
                if meta['producer'] == 'codec' and len(meta['sources']) == 1:
                    r = meta['sources'][0]
                    out.items.setdefault(r, {}).setdefault(s, []).append(kb._row_ids[s][row])
                    out.time[r] = max(out.time.get(r, 0), int(table[row, TIME]))
        return out


@dataclass
class Group:
    name: str
    parent: str
    split: str
    rows: list[dict]
    core: list[str]                 # records the slots read or name (in the parent bank)
    distractors: list[str]
    sources: dict = field(default_factory=dict)   # distractor counts by source

    @property
    def records(self) -> list[str]:
        return [*self.core, *self.distractors]


def _slots(row: Mapping) -> list[dict]:
    from schnitz.kb.bank import slots_of
    return slots_of(row)


def _query_time(row: Mapping) -> int:
    prov = row.get('provenance') or {}
    return int(prov.get('source_query_time', prov.get('query_time', 2)))


def pick_distractors(rows: Sequence[dict], pool: Sequence[str], n: int, mode: str,
                     rng: random.Random, teacher=None, parent: str | None = None
                     ) -> tuple[list[str], dict]:
    """``n`` distractors for a group from ``pool`` (sorted candidate records: the parent
    KB's, visible at the group's query times, no slot's positive or neutral record)."""
    if mode not in MODES:
        raise ValueError(f'distractors must be one of {MODES}')
    allowed = set(pool)
    picked: list[str] = []
    sources: Counter = Counter()
    want_teacher = n if mode == 'teacher' else n // 2 if mode == 'mixed' else 0
    if want_teacher and teacher is not None:
        lists = []
        for row in rows:
            for j in range(len(_slots(row))):
                try:
                    got = teacher.mined(row['episode_id'], j, TEACHER_TOP,
                                        kb=parent or parent_kb(row['kb']))
                except KeyError:
                    sources['teacher_site_missing'] += 1
                    continue
                lists.append([r for r in got if r in allowed])
        position = 0
        chosen = set()
        while len(picked) < want_teacher and any(position < len(v) for v in lists):
            for values in lists:
                if position < len(values) and values[position] not in chosen \
                        and len(picked) < want_teacher:
                    picked.append(values[position])
                    chosen.add(values[position])
                    sources['teacher'] += 1
            position += 1
    elif want_teacher:
        sources['teacher_unavailable'] += 1
    rest = [r for r in pool if r not in set(picked)]
    extra = rng.sample(rest, min(n - len(picked), len(rest)))
    sources['random'] += len(extra)
    return picked + extra, dict(sources)


def plan(rows_by_split: Mapping[str, Sequence[dict]], parents: Mapping[str, ParentIndex], *,
         size: int, mode: str = 'random', group_size: int = 1, seed: int = 0,
         teacher=None, with_neutral: bool = False, counts: Counter | None = None
         ) -> list[Group]:
    """The groups and their records. Episodes are grouped per split and parent KB (in a
    seeded order); an episode whose slot records are not all in the parent bank is
    dropped (``records_missing``)."""
    counts = Counter() if counts is None else counts
    groups: list[Group] = []
    for split, rows in rows_by_split.items():
        by_kb: dict[str, list[dict]] = {}
        for row in rows:
            kb = row['kb']
            if SEP in kb:
                raise ValueError(f'{row["episode_id"]}: already on a sub-KB ({kb})')
            parent = parents.get(kb)
            slots = _slots(row)
            if parent is None or not slots or not all(
                    r in parent.items for slot in slots for r in slot['record_ids']):
                counts['records_missing' if parent is not None else 'kb_not_in_bank'] += 1
                continue
            if any(slot.get('kb', kb) != kb for slot in slots):
                raise PermissionError(f'{row["episode_id"]}: slot of another KB')
            by_kb.setdefault(kb, []).append(row)
        for kb in sorted(by_kb):
            rng = random.Random(f'{seed}:{split}:{kb}')
            rows_kb = sorted(by_kb[kb], key=lambda r: r['episode_id'])
            rng.shuffle(rows_kb)
            parent = parents[kb]
            for start in range(0, len(rows_kb), group_size):
                chunk = rows_kb[start:start + group_size]
                named, core = set(), []
                for row in chunk:
                    for slot in _slots(row):
                        positives = [*slot['record_ids'], *(slot.get('alternatives') or ())]
                        neutral = list(slot.get('neutral') or ())
                        named.update(positives, neutral)
                        core += positives + (neutral if with_neutral else [])
                core = [r for r in dict.fromkeys(core) if r in parent.items]
                qt = min(_query_time(row) for row in chunk)
                pool = sorted(r for r, t in parent.time.items() if t <= qt and r not in named)
                n = max(0, size - len(core))
                if len(core) > size:
                    counts['groups_over_size'] += 1
                picked, sources = pick_distractors(chunk, pool, n, mode, rng, teacher, kb)
                name = sub_name(kb, split, start // group_size)
                groups.append(Group(name, kb, split, chunk, core, picked, sources))
    return groups


def copy_items(parent, child, records: Sequence[str], index: ParentIndex,
               batch: int = 256) -> int:
    """Append the parent's leaf items of ``records`` to ``child``, with their ids,
    payloads, keys, masses, times, sources, producer and step (never re-encoded)."""
    from schnitz.kb_store import NewItem, Provenance
    copied = 0
    for s in parent.spaces:
        ids = [i for r in records for i in index.items[r].get(s, ())]
        for start in range(0, len(ids), batch):
            items = parent.read(s, ids[start:start + batch])
            fresh = []
            for item in items:
                if item.version != 1 or item.lineage or item.derived:
                    raise ValueError(f'{item.id}: sub-KBs copy leaf items only (version 1, '
                                     'no lineage)')
                p = item.provenance
                fresh.append(NewItem(item.values, item.key,
                                     Provenance(p.sources, p.producer, p.step), item.mass,
                                     item.time, item.id))
            child.append(s, fresh)
            copied += len(fresh)
    return copied


def rewrite_row(row: dict, name: str) -> dict:
    """The transcript on its sub-KB: only the episode's and the slots' ``kb`` change."""
    out = json.loads(json.dumps(row))
    parent = out['kb']
    out['kb'] = name
    for m in out['messages']:
        content = m.get('content')
        if isinstance(content, dict) and 'slot' in content:
            content['slot']['kb'] = name
    for site in out.get('write_sites') or ():
        m = out['messages'][site['message']]
        if 'write_span' in m:
            m['write_span']['kb'] = name
    out.setdefault('provenance', {})['parent_kb'] = parent
    out.pop('_dir', None)
    return out


def _bytes_per_record(root: Path, records: int) -> float:
    total = sum(f.stat().st_size for s in ('A', 'B', 'C', 'D') if (root / s).exists()
                for f in (root / s).iterdir() if not f.name.startswith('live'))
    return total / max(records, 1)


def build(banks: Path, output: Path, *, transcripts: Sequence[Path] | None = None,
          limits: Mapping[str, int | None] | None = None, size: int, mode: str = 'random',
          group_size: int = 1, seed: int = 0, teacher_dir: Path | None = None,
          with_neutral: bool = False, reserve_gb: float = 30.0, log=print) -> dict:
    """Write the sub-KB banks and transcripts under ``output`` (see the module doc)."""
    from schnitz.kb.bank import Transcripts, kb_dir
    from schnitz.kb_store import KnowledgeBase
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest = json.loads((banks / 'banks.json').read_text())
    if 'rows' in manifest:
        raise ValueError('sub-KBs are cut from leaf banks (l1 build), not rows banks')
    transcripts = [Path(d) for d in (transcripts or manifest['transcripts'])]
    limits = dict(limits or {'train': manifest.get('limit'),
                             'validation': manifest.get('eval_limit')})
    teacher = None
    if mode != 'random':
        if teacher_dir is None:
            raise ValueError(f'--distractors {mode} needs a teacher-keys cache')
        from schnitz.kb.teacher_keys import TeacherKeys
        teacher = TeacherKeys(teacher_dir)
    parents, handles, per_record = {}, {}, {}
    for name, info in manifest['kbs'].items():
        handles[name] = KnowledgeBase(banks / info['dir'])
        parents[name] = ParentIndex.of(handles[name])
        per_record[name] = _bytes_per_record(banks / info['dir'], len(parents[name].items))
    rows_by_split = {}
    for split, limit in limits.items():
        rows_by_split[split] = [dict(row) for row in Transcripts(transcripts, split, limit)]
    counts: Counter = Counter()
    groups = plan(rows_by_split, parents, size=size, mode=mode, group_size=group_size,
                  seed=seed, teacher=teacher, with_neutral=with_neutral, counts=counts)
    estimate = sum(len(g.records) * per_record[g.parent] for g in groups)
    free = shutil.disk_usage(output.parent if output.parent.exists() else Path('/')).free
    if estimate > free - reserve_gb * 1e9:
        raise OSError(f'sub-KBs need about {estimate / 1e9:.1f} GB; {free / 1e9:.1f} GB free '
                      f'(keeping {reserve_gb} GB)')
    log(json.dumps({'groups': len(groups), 'records': sum(len(g.records) for g in groups),
                    'estimate_gb': round(estimate / 1e9, 2)}))
    pending = output.with_name(output.name + '.pending')
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True)
    kbs = {}
    for g in groups:
        root = pending / kb_dir(g.name)
        child = KnowledgeBase.create(
            root, name=kb_dir(g.name), dataset=g.name,
            origin={'command': 'train.py l1 subkb', 'parent': g.parent,
                    'parent_dir': str(banks / manifest['kbs'][g.parent]['dir']),
                    'parent_cursor': handles[g.parent].cursor, 'size': size,
                    'distractors': mode, 'seed': seed})
        copy_items(handles[g.parent], child, g.records, parents[g.parent])
        kbs[g.name] = {'dir': root.name, 'records': len(g.records), 'parent': g.parent,
                       'split': g.split, 'core': len(g.core),
                       'distractors': len(g.distractors), 'distractor_sources': g.sources,
                       'episodes': [row['episode_id'] for row in g.rows],
                       'stats': child.stats()}
        child.close()
        (root / 'writer.lock').unlink(missing_ok=True)
    out_dirs = []
    for d in transcripts:
        rows = [(g, row) for g in groups for row in g.rows if row.get('_dir') == str(d)]
        if not rows:
            continue
        target = pending / 'transcripts' / Path(d).name
        target.mkdir(parents=True, exist_ok=True)
        for split in limits:
            with (target / f'transcripts-{split}.jsonl').open('w', encoding='utf-8') as out:
                for g, row in rows:
                    if g.split == split:
                        out.write(json.dumps(rewrite_row(row, g.name), ensure_ascii=False) + '\n')
        source = json.loads((Path(d) / 'manifest.json').read_text())
        source['subkb'] = {'banks': str(output), 'parent_banks': str(banks),
                           'source_transcripts': str(d), 'size': size, 'distractors': mode,
                           'group_size': group_size}
        (target / 'manifest.json').write_text(json.dumps(source, indent=2) + '\n')
        out_dirs.append(str(output / 'transcripts' / Path(d).name))
    for file in ('stack.pt', 'key_heads_init.pt'):
        if (banks / file).exists():
            shutil.copy2(banks / file, pending / file)
    for handle in handles.values():
        handle.close()
    keep = {k: manifest[k] for k in ('span_source', 'level', 'span_cache', 'codecs',
                                     'span_batch_size', 'codec_step', 'reader_state')
            if k in manifest}
    out = {'command': 'subkb', 'parent_banks': str(banks), 'transcripts': out_dirs,
           'source_transcripts': [str(d) for d in transcripts], 'limit': None,
           'eval_limit': None, 'source_limits': limits, 'size': size, 'distractors': mode,
           'group_size': group_size, 'with_neutral': with_neutral, 'seed': seed,
           'teacher': str(teacher_dir) if teacher_dir else None, 'counts': dict(counts),
           **keep, 'kbs': kbs}
    (pending / 'banks.json').write_text(json.dumps(out, indent=2) + '\n')
    if output.exists():
        output.rmdir()
    os.replace(pending, output)
    return out
