"""Select source-held-out SQuAD questions for an already published train bank."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from schnitz.corpus_data import title_located_question
from schnitz.data import episode_from_dict
from schnitz.operations import atomic_json
from schnitz.text import render_prompt
from schnitz.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(data: Path, output: Path, tokenizer, *, source_offset: int = 1024,
            source_count: int = 64, max_target_tokens: int = 128) -> dict:
    if (source_offset < 0 or min(source_count, max_target_tokens) < 1 or output.exists()
            or not output.parent.is_dir()):
        raise ValueError('Positive budgets and fresh output under an existing parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Evaluation data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Evaluation preparation requires a 10 GiB disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('sources-train.jsonl', 'train.jsonl'):
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError(f'Prepared source changed: {name}')
    source_rows = [json.loads(line) for line in (data / 'sources-train.jsonl').open()]
    if source_offset + source_count > len(source_rows):
        raise ValueError('Held-out source range exceeds published source manifest')
    ids = [row['record_id'] for row in source_rows[source_offset:]]
    eligible_ids = set(ids)
    chosen = {}
    for line in (data / 'train.jsonl').open():
        row = json.loads(line)
        source_id = row['required_ids'][0]
        if source_id in eligible_ids and source_id not in chosen:
            episode = episode_from_dict(row)
            try:
                located = title_located_question(episode)
            except ValueError:
                continue
            if (len(tokenizer.encode(located.answer, add_special_tokens=False))
                    + int(tokenizer.eos_token_id is not None) <= max_target_tokens
                    and len(tokenizer.encode(render_prompt(tokenizer, located.query, ''),
                                             add_special_tokens=False)) <= 1024):
                chosen[source_id] = asdict(located)
    selected_ids = [source_id for source_id in ids if source_id in chosen][:source_count]
    if len(selected_ids) != source_count:
        raise ValueError('Insufficient held-out sources have bounded, title-located questions')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        path = pending / 'episodes.jsonl'
        with path.open('w') as handle:
            for source_id in selected_ids:
                handle.write(json.dumps(chosen[source_id], sort_keys=True, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = {'format': 1, 'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'source_manifest_sha256': file_sha256(data / 'sources-train.jsonl'),
                  'source_offset': source_offset, 'source_count': source_count,
                  'source_ids_sha256': hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest(),
                  'episodes_sha256': file_sha256(path), 'max_target_tokens': max_target_tokens,
                  'tokenizer_revision': REVISION,
                  'query_locator': 'source article title; target answer excluded',
                  'notice': 'Questions and sources are beyond the declared source offset; '
                            'all selected source IDs belong to the published 10k bank.'}
        atomic_json(pending / 'manifest.json', result)
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-offset', type=int, default=1024)
    parser.add_argument('--source-count', type=int, default=64)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             source_offset=args.source_offset, source_count=args.source_count), indent=2))
