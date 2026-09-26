"""Mix prepared episode corpora and write the combined bank source manifest.

Inputs are the existing mixed episodes and source manifest (kept whole) plus
dataset directories from ``public_corpus_common.Writer``. Each added dataset
contributes at most ``--train-cap`` / ``--validation-cap`` episodes, sampled
with a fixed seed; only sources referenced by kept episodes enter the bank
(plus a dataset's unreferenced sources with ``--keep-unreferenced``, used for
full evaluation haystacks). Base source rows are copied byte-for-byte first so
existing record identities, order and writer groups are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random


def _lines(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield line if line.endswith('\n') else line + '\n'


def _sample(path: Path, cap: int, seed: str) -> list[str]:
    if not path.is_file() or cap <= 0:
        return []
    rows = list(_lines(path))
    random.Random(seed).shuffle(rows)
    return rows[:cap]


def _write(path: Path, rows) -> str:
    digest = hashlib.sha256()
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(row)
            digest.update(row.encode())
    os.replace(pending, path)
    return digest.hexdigest()


def mix(*, base_train: Path, base_validation: Path, base_sources: Path,
        datasets: list[Path], output: Path, train_cap: int, validation_cap: int,
        caps: dict[str, int], seed: int, keep_unreferenced: set[str]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    base_ids, base_domains = set(), Counter()
    for line in _lines(base_sources):
        row = json.loads(line)
        base_ids.add(row['record_id'])
        base_domains[row.get('domain', 'research')] += 1
    train, validation = list(_lines(base_train)), list(_lines(base_validation))
    counts = {'base': {'train': len(train), 'validation': len(validation),
                       'sources': len(base_ids)}}
    added_sources: list[str] = []
    for directory in datasets:
        manifest = json.loads((directory / 'manifest.json').read_text())
        domain = manifest['domain']
        if domain in base_domains:
            raise ValueError(f'{domain} already has records in the base bank')
        if manifest.get('role') == 'heldout_evaluation_only':
            raise ValueError(f'{domain} is evaluation-only')
        kept_train = _sample(directory / 'episodes-train.jsonl', caps.get(domain, train_cap),
                             f'{seed}:{domain}:train')
        kept_validation = _sample(directory / 'episodes-validation.jsonl', validation_cap,
                                  f'{seed}:{domain}:validation')
        needed = set()
        for line in (*kept_train, *kept_validation):
            episode = json.loads(line)
            if episode['provenance']['domain'] != domain:
                raise ValueError(f'{directory}: episode outside its domain')
            needed.update(support['record_id'] for support in episode['supports'])
        written = 0
        for line in _lines(directory / 'sources.jsonl'):
            row = json.loads(line)
            if row['domain'] != domain or row['record_id'] in base_ids:
                raise ValueError(f'{directory}: source outside its domain or colliding')
            if row['record_id'] in needed or domain in keep_unreferenced:
                added_sources.append(line)
                needed.discard(row['record_id'])
                written += 1
        if needed:
            raise ValueError(f'{directory}: episodes reference missing sources')
        train.extend(kept_train)
        validation.extend(kept_validation)
        counts[domain] = {'train': len(kept_train), 'validation': len(kept_validation),
                          'sources': written, 'dataset_manifest_sha256': hashlib.sha256(
                              (directory / 'manifest.json').read_bytes()).hexdigest()}
    random.Random(f'{seed}:train').shuffle(train)
    random.Random(f'{seed}:validation').shuffle(validation)
    digests = {
        'train-episodes.jsonl': _write(output / 'train-episodes.jsonl', train),
        'validation-episodes.jsonl': _write(output / 'validation-episodes.jsonl', validation),
        'sources.jsonl': _write(output / 'sources.jsonl',
                                (*_lines(base_sources), *sorted(added_sources))),
    }
    result = {'format': 1, 'base_train': str(base_train), 'base_validation': str(base_validation),
              'base_sources': str(base_sources), 'datasets': [str(d) for d in datasets],
              'train_cap': train_cap, 'validation_cap': validation_cap, 'caps': caps,
              'seed': seed, 'keep_unreferenced': sorted(keep_unreferenced), 'counts': counts,
              'totals': {'train': len(train), 'validation': len(validation),
                         'sources': len(base_ids) + len(added_sources)},
              'sha256': digests}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-train', type=Path, required=True)
    parser.add_argument('--base-validation', type=Path, required=True)
    parser.add_argument('--base-sources', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-cap', type=int, default=20000)
    parser.add_argument('--validation-cap', type=int, default=200)
    parser.add_argument('--cap', action='append', default=[], metavar='DOMAIN=N')
    parser.add_argument('--keep-unreferenced', action='append', default=[])
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    caps = {name: int(value) for name, value in (item.split('=', 1) for item in args.cap)}
    print(json.dumps(mix(
        base_train=args.base_train, base_validation=args.base_validation,
        base_sources=args.base_sources, datasets=args.dataset, output=args.output,
        train_cap=args.train_cap, validation_cap=args.validation_cap, caps=caps,
        seed=args.seed, keep_unreferenced=set(args.keep_unreferenced)), indent=2))
