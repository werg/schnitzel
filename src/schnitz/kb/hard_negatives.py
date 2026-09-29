"""Mined hard negatives for the retrieval loss (docs/knowledge-base-stack.md 5.1 step 4 and
5.2; restart plan, 29 September: the retrieval options queue, item 2).

Data side of ``l1 train --hard-negatives FILE``: per search site (episode id, call index)
a ranked list of *record ids* that the retrieval loss of that site should score as extra
negatives. Two sources, mixed by ``mine``:

- a text-embedding teacher cache (``schnitz.kb.teacher_keys``): the site's teacher
  top-ranked records of its own KB (near misses the teacher itself confuses);
- the reader's own top wrong hits (``collect_hits``/``dump_hits``, written by
  ``l1 train --dump-hits FILE``): per site the scored candidates of its last training
  read, ranked by the read's gates, mapped to their bank records.

Positives never become negatives: a site's slot records, its ``alternatives`` and its
``neutral`` records are excluded when mining and again when the trainer maps records to
items (``HardNegatives.items``). Records map to items of the episode's own KB only (a KB
is an authorization domain); a sub-KB (``schnitz.kb.subkb``) accepts the negatives of
its parent's sites and keeps only the records it holds. The reader's retrieval loss adds
these items to the site's in-batch negatives (``L1Reader.read(negatives=)``, which also
drops items later than the query time). The file is training supervision only; nothing
here is used at inference (invariant 1).

File format (JSON)::

    {"format": "schnitz.kb.hard-negatives/1", "meta": {...},
     "sites": [{"episode_id": ..., "call": 0, "kb": ..., "records": [...],
                "sources": {"teacher": n, "hits": m}}, ...]}

The reader's hits dump (``HITS_FORMAT``) has the same ``sites`` rows, each with the
training step of its read (``step``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path

FORMAT = 'schnitz.kb.hard-negatives/1'
HITS_FORMAT = 'schnitz.kb.reader-hits/1'


def parent_kb(name: str) -> str:
    from schnitz.kb.subkb import parent_kb as parent
    return parent(name)


def excluded(slot: Mapping) -> set[str]:
    """Records a site's hard negatives may never contain: its positives (the slot's
    records and alternatives) and its neutral records."""
    return {*slot.get('record_ids', ()), *(slot.get('alternatives') or ()),
            *(slot.get('neutral') or ())}


def _slots(row: Mapping) -> list[dict]:
    from schnitz.kb.bank import slots_of
    return slots_of(row)


# -- sources ------------------------------------------------------------------------------
def from_teacher(cache, rows: Iterable[Mapping], k: int = 16) -> dict[tuple[str, int], list[str]]:
    """Per search site the teacher's top ``k`` records of the site's KB, the site's
    positives and neutral records left out (``TeacherKeys.mined``). Sites missing from
    the cache are skipped."""
    out = {}
    for row in rows:
        for j, slot in enumerate(_slots(row)):
            try:
                got = cache.mined(row['episode_id'], j, k, kb=parent_kb(row['kb']),
                                  skip=excluded(slot))
            except KeyError:
                continue
            out[row['episode_id'], j] = got
    return out


def load_hits(path: str | Path) -> dict[tuple[str, int], list[str]]:
    """A reader hits dump (``dump_hits``): per site its ranked wrong records."""
    data = json.loads(Path(path).read_text())
    if data.get('format') != HITS_FORMAT:
        raise ValueError(f'{path} is not a reader hits dump ({HITS_FORMAT})')
    return {(s['episode_id'], int(s['call'])): list(s['records']) for s in data['sites']}


def mine(rows: Iterable[Mapping], *, teacher=None, hits: Mapping | None = None, k: int = 16,
         teacher_k: int | None = None) -> list[dict]:
    """Per site up to ``k`` hard negatives, interleaving the reader's own wrong hits and
    the teacher's near misses (hits first), with every site's positives and neutral
    records excluded. ``rows``: the training transcripts (their slots define what is
    excluded; a site with no source gets no entry)."""
    rows = list(rows)
    mined = from_teacher(teacher, rows, teacher_k or k) if teacher is not None else {}
    hits = hits or {}
    out = []
    for row in rows:
        for j, slot in enumerate(_slots(row)):
            key = (row['episode_id'], j)
            skip = excluded(slot)
            lists = [('hits', [r for r in hits.get(key, ()) if r not in skip]),
                     ('teacher', [r for r in mined.get(key, ()) if r not in skip])]
            picked, sources = [], {}
            position = 0
            while len(picked) < k and any(position < len(v) for _, v in lists):
                for name, values in lists:
                    if position < len(values) and values[position] not in picked \
                            and len(picked) < k:
                        picked.append(values[position])
                        sources[name] = sources.get(name, 0) + 1
                position += 1
            if picked:
                out.append({'episode_id': row['episode_id'], 'call': j, 'kb': row['kb'],
                            'records': picked, 'sources': sources})
    return out


def save(path: str | Path, sites: Sequence[dict], meta: dict | None = None,
         fmt: str = FORMAT) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps({'format': fmt, 'meta': meta or {}, 'sites': list(sites)})
                       + '\n')
    os.replace(pending, path)


# -- the trainer side -----------------------------------------------------------------------
class HardNegatives:
    """Loaded hard negatives: per (episode id, call) the KB and the ranked records."""

    def __init__(self, sites: Iterable[dict]):
        self.sites: dict[tuple[str, int], tuple[str, list[str]]] = {}
        for s in sites:
            self.sites[s['episode_id'], int(s['call'])] = (s['kb'], list(s['records']))

    @classmethod
    def load(cls, path: str | Path) -> HardNegatives:
        data = json.loads(Path(path).read_text())
        if data.get('format') != FORMAT:
            raise ValueError(f'{path} is not a hard-negatives file ({FORMAT})')
        return cls(data['sites'])

    def __len__(self) -> int:
        return len(self.sites)

    def records(self, episode_id: str, call: int, kb: str) -> list[str]:
        """The site's records, if they belong to ``kb`` (or ``kb`` is a sub-KB of theirs);
        another KB's list is refused (authorization)."""
        got = self.sites.get((episode_id, int(call)))
        if got is None:
            return []
        owner, records = got
        if owner != kb and owner != parent_kb(kb):
            raise PermissionError(f'hard negatives of {owner!r} offered to a site of {kb!r}')
        return records

    def items(self, ctx, ep, j: int) -> dict[str, list[tuple[str, str]]]:
        """Per space the items of site ``j``'s hard-negative records in the episode's own
        KB (``ctx.index``), the slot's positives and neutral records excluded."""
        skip = excluded(ep.slots[j])
        names = [r for r in self.records(ep.episode_id, j, ep.kb) if r not in skip]
        index = ctx.index[ep.kb]
        return {s: [(ep.kb, i) for r in names for i in index[s].get(r, ())] for s in index}

    def extend(self, ctx, ep, negatives: Mapping[str, Sequence] | None
               ) -> Mapping[str, Sequence] | None:
        """The episode's in-batch ``negatives`` (per space) plus the hard negatives of
        every site of the episode (the reader drops each site's own positives and neutral
        items from the list, so a site's negatives never include another site's
        positives it shares); ``negatives`` itself when there are none."""
        extra: dict[str, list] = {}
        for j in range(len(ep.slots)):
            for s, refs in self.items(ctx, ep, j).items():
                extra.setdefault(s, []).extend(refs)
        if not any(extra.values()):
            return negatives
        spaces = set(extra) | set(negatives or {})
        return {s: list(dict.fromkeys([*(negatives or {}).get(s, ()), *extra.get(s, ())]))
                for s in spaces}


def collect_hits(sink: dict, ctx, ep, reads, top: int = 16, step: int = 0) -> None:
    """Record, per search site of ``ep``, the reader's top wrong hits of this read: the
    scored candidates of every space ranked by their gates (interleaved over spaces),
    mapped to their bank records (items of one source record), without the site's
    positives and neutral records. ``reads``: one read per call, in call order
    (``run_episode`` in retrieve mode)."""
    if len(reads) != len(ep.slots):
        return
    for j, read in enumerate(reads):
        skip = excluded(ep.slots[j])
        per_space = []
        for s, info in read.spaces.items():
            if info.scored_gates is None or not info.scored:
                continue
            gates = info.scored_gates.detach().float().cpu().tolist()
            ranked = [ref for _, ref in sorted(zip(gates, info.scored),
                                               key=lambda x: -x[0])]
            recs = []
            for dataset, item_id in ranked:
                producer, sources = ctx.origin[dataset][s].get(item_id, (None, ()))
                if producer == 'codec' and len(sources) == 1 and sources[0] not in skip:
                    recs.append(sources[0])
            per_space.append(recs)
        merged: list[str] = []
        position = 0
        while len(merged) < top and any(position < len(v) for v in per_space):
            for values in per_space:
                if position < len(values) and values[position] not in merged \
                        and len(merged) < top:
                    merged.append(values[position])
            position += 1
        sink[ep.episode_id, j] = {'episode_id': ep.episode_id, 'call': j, 'kb': ep.kb,
                                  'records': merged, 'step': step}


def dump_hits(path: str | Path, sink: Mapping, step: int) -> None:
    save(path, sorted(sink.values(), key=lambda s: (s['episode_id'], s['call'])),
         {'step': step}, HITS_FORMAT)
