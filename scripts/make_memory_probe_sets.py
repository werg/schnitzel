"""Evaluation episode sets that test whether stored values deliver their information.

``text-control``
    Copies episodes with the text of their gold passages written into the query
    ("Reference passages: ..."). Evaluated with memory reads disabled, this is
    the information-matched text arm for the same answers; compare it with the
    original episodes under ``oracle_gold`` (gold delivered as stored values)
    and ``no_read`` (prior only). Episode order and IDs are kept, so packing
    produces the same trajectories and answer tokens.
``reconstruction``
    Exact-span questions about bank passages that were never training targets:
    "return words i..j of the stored passage that begins ...". The answer is
    not recoverable from the query or from general knowledge, so answer NLL
    measures what the stored value carries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def text_control(episodes: Path, sources: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in episodes.open(encoding='utf-8') if line.strip()]
    needed = {record_id for row in rows for record_id in row['required_ids']}
    texts = {}
    with sources.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['record_id'] in needed:
                texts[row['record_id']] = row['text']
    if needed - texts.keys():
        raise ValueError('Gold passages are missing from the source manifest')
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            block = '\n'.join(f'[{i + 1}] {texts[r]}' for i, r in enumerate(row['required_ids']))
            copy = dict(row, query=f'Reference passages:\n{block}\n\n{row["query"]}')
            copy['provenance'] = dict(row.get('provenance', {}), text_control=True)
            handle.write(json.dumps(copy) + '\n')
    return {'episodes': len(rows), 'gold_passages': len(needed)}


def reconstruction(sources: Path, train_episodes: Path, output: Path, *, count: int,
                   domain: str = 'research') -> dict:
    excluded = set()
    with train_episodes.open(encoding='utf-8') as handle:
        for line in handle:
            excluded.update(json.loads(line)['required_ids'])
    candidates = []
    with sources.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            title = row.get('provenance', {}).get('article_title')
            if (row.get('domain', 'research') != domain or not title
                    or row['record_id'] in excluded or '\nPassage: ' not in row['text']):
                continue
            passage = row['text'].split('\nPassage: ', 1)[1]
            words = list(re.finditer(r'\S+', passage))
            if len(words) < 14:
                continue
            start = 6 + int(_sha('probe:' + row['record_id'])[:16], 16) % (len(words) - 13)
            answer = passage[words[start].start():words[start + 7].end()]
            locator = passage[words[0].start():words[5].end()]
            query = (f'Article: {title}\nStored passage begins: {locator}\n'
                     f'Return exactly words {start + 1} through {start + 8} of that stored '
                     'passage. Give only those words.')
            if answer in query:
                continue
            candidates.append({
                'episode_id': _sha('probe-reconstruction:' + row['record_id'])[:32],
                'environment': 'probe-reconstruction', 'query': query, 'answer': answer,
                'query_time': 2, 'required_ids': [row['record_id']],
                'sufficient_groups': [[row['record_id']]], 'support_annotation': 'verified',
                'task_family': 'passage_span', 'restore': False, 'allowed_capability': 0,
                'capability': 0, 'choices': [],
                'supports': [{'record_id': row['record_id'], 'text': row['text'],
                              'created_at': row.get('created_at', 1), 'kind': 'passage'}],
                'provenance': {'domain': domain, 'article_title': title,
                               'probe': 'never-a-training-target'}})
    candidates.sort(key=lambda row: _sha('order:' + row['episode_id']))
    with output.open('w', encoding='utf-8') as handle:
        for row in candidates[:count]:
            handle.write(json.dumps(row) + '\n')
    return {'episodes': min(count, len(candidates)), 'eligible': len(candidates),
            'excluded_training_golds': len(excluded)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    text = commands.add_parser('text-control')
    text.add_argument('--episodes', type=Path, required=True)
    text.add_argument('--sources', type=Path, required=True)
    text.add_argument('--output', type=Path, required=True)
    recon = commands.add_parser('reconstruction')
    recon.add_argument('--sources', type=Path, required=True)
    recon.add_argument('--train-episodes', type=Path, required=True)
    recon.add_argument('--output', type=Path, required=True)
    recon.add_argument('--count', type=int, default=256)
    args = parser.parse_args()
    if args.command == 'text-control':
        result = text_control(args.episodes, args.sources, args.output)
    else:
        result = reconstruction(args.sources, args.train_episodes, args.output, count=args.count)
    print(json.dumps(result, indent=2))
