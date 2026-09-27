"""Interleave exact full-passage and indexed-span tasks over the same short sources."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from schnitz.data import episode_from_dict
from schnitz.trajectories import file_sha256


def indexed_span(row: dict, *, span_words: int = 8):
    episode = episode_from_dict(row)
    if (episode.task_family != 'passage_reconstruction' or
            len(episode.supports) != 1 or span_words < 1):
        raise ValueError('An exact one-source reconstruction task is required')
    passage = episode.supports[0].text.split('\nPassage: ', 1)[1]
    spans = list(re.finditer(r'\S+', passage))
    if len(spans) < span_words:
        raise ValueError('Span exceeds stored passage')
    digest = hashlib.sha256(('indexed-span-v1:' + episode.required_ids[0]).encode()).digest()
    start = int.from_bytes(digest[:8], 'big') % (len(spans) - span_words + 1)
    end = start + span_words
    answer = passage[spans[start].start():spans[end - 1].end()]
    query = (f'Return exactly words {start + 1} through {end} of the stored passage, '
             'preserving their spelling and punctuation. Give only those words.')
    if answer in query:
        raise ValueError('Indexed query reveals its answer')
    return replace(episode,
        episode_id=hashlib.sha256(('indexed-span-v1:' + episode.episode_id +
                                   f':{start}:{end}').encode()).hexdigest()[:32],
        query=query, answer=answer, task_family='passage_span',
        provenance=episode.provenance | {'span_words': span_words,
            'span_start_word': start + 1, 'span_end_word': end,
            'task_version': 'indexed-span-v1'})


def prepare(short_data: Path, output: Path, *, span_words: int = 8) -> dict:
    if span_words < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive span length and a fresh output parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Training data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Preparation requires a 10 GiB external disk reserve')
    parent_manifest = json.loads((short_data / 'manifest.json').read_text())
    for split in ('train', 'validation'):
        name = f'{split}.jsonl'
        if file_sha256(short_data / name) != parent_manifest['splits'][split]['sha256'][name]:
            raise ValueError('Short reconstruction input differs from manifest')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        summaries = {}
        for split in ('train', 'validation'):
            rows = [json.loads(line) for line in (short_data / f'{split}.jsonl').open()]
            full, spans = rows, [asdict(indexed_span(row, span_words=span_words))
                                  for row in rows]
            files = {f'{split}.jsonl': [item for pair in zip(full, spans, strict=True)
                                       for item in pair],
                     f'{split}-full.jsonl': full,
                     f'{split}-span.jsonl': spans}
            for name, items in files.items():
                with (pending / name).open('w', encoding='utf-8') as handle:
                    for item in items:
                        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            summaries[split] = {'sources': len(rows), 'episodes': 2 * len(rows),
                'sha256': {name: file_sha256(pending / name) for name in files}}
        result = {'format': 1, 'task': 'full passage plus indexed eight-word span',
                  'parent_manifest_sha256': file_sha256(short_data / 'manifest.json'),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'span_words': span_words, 'splits': summaries,
                  'notice': 'One source version serves both tasks. Indexed positions are '
                            'causal instructions; passage content remains stored only.'}
        (pending / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--span-words', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.short_data, args.output,
                             span_words=args.span_words), indent=2))
