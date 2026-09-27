"""Add background knowledge to a task corpus (plan B9: reference information in the KB).

Reads a finished task corpus (``prepare_task_corpora.py``) and background passage
corpora (``passages.jsonl`` with ``id``, ``title``, ``text``, ``source``,
``topic_key``) and writes a new corpus in which every background passage is a
knowledge-base record (``kind: background``, created before every query) and
each episode lists its ``--top`` best background passages as supports (not as
required evidence). Candidates are first narrowed to passages collected for the
episode's topic (``--topic-field`` of its provenance, e.g. ``db_id``), then ranked
by BM25 against the episode's question; too few candidates fall back to BM25 over
the whole background corpus. Exact scoring, no approximate index.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

WORD = re.compile(r'[a-z0-9]+')
STOP = set('the a an of and or to in on for is are was were be by with as at from that this '
           'which what who whom whose how many much does do did use stored notes question '
           'answer only return write one sqlite query database value values give'.split())


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.length = [sum(t.values()) for t in self.tf]
        self.mean = sum(self.length) / max(len(docs), 1)
        df = Counter(w for t in self.tf for w in t)
        self.idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.postings[w].append(i)

    def top(self, query: str, k: int, allowed: set[int] | None = None) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for w in set(tokens(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                if allowed is not None and i not in allowed:
                    continue
                f = self.tf[i][w]
                norm = f + self.k1 * (1 - self.b + self.b * self.length[i] / self.mean)
                scores[i] += idf * f * (self.k1 + 1) / norm
        return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _keys(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).split('::')[0] for v in value]
    return [str(value).split('::')[0]] if value is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--background', type=Path, action='append', required=True,
                        help='directory with passages.jsonl (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--topic-field', help='provenance field naming the topic, e.g. db_id')
    parser.add_argument('--top', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads((args.corpus / 'manifest.json').read_text())
    domain = manifest['domain']
    passages = []
    for directory in args.background:
        with (directory / 'passages.jsonl').open(encoding='utf-8') as handle:
            passages += [dict(json.loads(line), corpus=directory.name) for line in handle]
    records = []
    for p in passages:
        text = f'Background: {p["title"]}\n{p["text"]}'.strip()
        records.append({'record_id': hashlib.sha256(f'{domain}\0{text}'.encode()).hexdigest()[:32],
                        'text': text, 'domain': domain, 'created_at': 1, 'kind': 'background',
                        'provenance': {'dataset': domain, 'background_corpus': p['corpus'],
                                       'source': p.get('source'), 'passage_id': p['id']}})
    index = BM25([f'{p["title"]} {p["text"]}' for p in passages])
    by_topic: dict[str, set[int]] = defaultdict(set)
    for i, p in enumerate(passages):
        for key in _keys(p.get('topic_key')):
            by_topic[key.lower()].add(i)

    tmp = args.output.with_name(args.output.name + '.pending')
    tmp.mkdir(parents=True)
    used: Counter = Counter()
    for split_file in sorted(args.corpus.glob('episodes-*.jsonl')):
        with split_file.open(encoding='utf-8') as source, \
                (tmp / split_file.name).open('w', encoding='utf-8') as out:
            for line in source:
                row = json.loads(line)
                topic = str(row['provenance'].get(args.topic_field, '')).lower() if args.topic_field else ''
                allowed = by_topic.get(topic) if topic else None
                picked = index.top(row['query'], args.top, allowed) if allowed else []
                if len(picked) < args.top:
                    picked += [i for i in index.top(row['query'], args.top * 2)
                               if i not in picked][:args.top - len(picked)]
                have = {s['record_id'] for s in row['supports']}
                for i in picked:
                    rec = records[i]
                    used[i] += 1
                    if rec['record_id'] not in have:
                        row['supports'].append({'record_id': rec['record_id'], 'text': rec['text'],
                                                'created_at': 1, 'kind': 'background'})
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    seen = set()
    with (args.corpus / 'sources.jsonl').open(encoding='utf-8') as source, \
            (tmp / 'sources.jsonl').open('w', encoding='utf-8') as out:
        for line in source:
            seen.add(json.loads(line)['record_id'])
            out.write(line)
        for rec in records:
            if rec['record_id'] not in seen:
                seen.add(rec['record_id'])
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp.glob('*.jsonl'))}
    manifest.update({'background': {'corpora': [str(d) for d in args.background],
                                    'passages': len(passages), 'top': args.top,
                                    'topic_field': args.topic_field,
                                    'passages_used_as_supports': len(used)},
                     'sources': len(seen), 'sha256': digests, 'parent': str(args.corpus)})
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(tmp, args.output)
    print(json.dumps(manifest['background']), manifest['sources'])


if __name__ == '__main__':
    main()
