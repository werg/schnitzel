"""Prepare content-addressable SQuAD QA episodes for a published source bank."""
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
from schnitz.text import render_prompt
from schnitz.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(data: Path, output: Path, tokenizer, *, source_count: int,
            questions_per_source: int = 1, max_target_tokens: int = 128,
            max_prompt_tokens: int = 1024, source_offset: int = 0) -> dict:
    if (source_offset < 0 or min(source_count, questions_per_source,
                                 max_target_tokens, max_prompt_tokens) < 1
            or output.exists() or not output.parent.is_dir()):
        raise ValueError('Positive budgets and a fresh output parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Bank training data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Bank training preparation requires a 10 GiB disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('sources-train.jsonl', 'train.jsonl'):
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError(f'Prepared source changed: {name}')
    source_ids = [json.loads(line)['record_id'] for line in
                  (data / 'sources-train.jsonl').open()]
    if len(source_ids) < source_offset + source_count or len(set(source_ids)) != len(source_ids):
        raise ValueError('Source budget exceeds the unique prepared corpus')
    eligible_source_ids = source_ids[source_offset:]
    counts = {source_id: 0 for source_id in eligible_source_ids}
    episodes = []
    filtered = 0
    for line in (data / 'train.jsonl').open():
        row = json.loads(line)
        source_id = row['required_ids'][0]
        if source_id not in counts or counts[source_id] >= questions_per_source:
            continue
        episode = episode_from_dict(row)
        try:
            located = title_located_question(episode)
        except ValueError:
            filtered += 1
            continue
        target_tokens = len(tokenizer.encode(located.answer, add_special_tokens=False))
        prompt_tokens = len(tokenizer.encode(render_prompt(tokenizer, located.query, ''),
                                             add_special_tokens=False))
        if (target_tokens + int(tokenizer.eos_token_id is not None) > max_target_tokens
                or prompt_tokens > max_prompt_tokens):
            filtered += 1
            continue
        episodes.append(asdict(located))
        counts[source_id] += 1
    eligible_ids = [source_id for source_id in eligible_source_ids
                    if counts[source_id] == questions_per_source]
    if len(eligible_ids) < source_count:
        raise ValueError('Not enough sources have safe, bounded title-located questions')
    selected_ids = set(eligible_ids[:source_count])
    episodes = [episode for episode in episodes
                if episode['required_ids'][0] in selected_ids]
    episodes.sort(key=lambda row: hashlib.sha256(row['episode_id'].encode()).digest())
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        path = pending / 'train.jsonl'
        with path.open('w') as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, sort_keys=True, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = {'format': 1, 'source_manifest_sha256': file_sha256(data / 'sources-train.jsonl'),
                  'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'source_offset': source_offset, 'source_count': source_count,
                  'questions_per_source': questions_per_source,
                  'selected_source_ids_sha256': hashlib.sha256(json.dumps(
                      eligible_ids[:source_count]).encode()).hexdigest(),
                  'skipped_sources_before_budget': eligible_source_ids.index(
                      eligible_ids[source_count - 1]) + 1 - source_count,
                  'episodes': len(episodes), 'filtered_candidates': filtered,
                  'max_target_tokens': max_target_tokens, 'max_prompt_tokens': max_prompt_tokens,
                  'tokenizer_revision': REVISION, 'train_sha256': file_sha256(path),
                  'query_locator': 'source article title; target answer excluded',
                  'notice': 'All sources precede their questions; supports remain labels only on stored-bank training reads.'}
        (pending / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
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
    parser.add_argument('--source-count', type=int, required=True)
    parser.add_argument('--source-offset', type=int, default=0)
    parser.add_argument('--questions-per-source', type=int, default=1)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             source_count=args.source_count,
                             questions_per_source=args.questions_per_source,
                             source_offset=args.source_offset), indent=2))
