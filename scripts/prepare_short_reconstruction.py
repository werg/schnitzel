"""Build source-disjoint 32–64-token exact reconstruction episodes."""
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

from schnitz.corpus_data import short_reconstruction
from schnitz.data import Source
from schnitz.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def shorten_source(row: dict, tokenizer, *, words: int = 28,
                   min_source_tokens: int = 32, max_source_tokens: int = 64):
    """Return an exact full-passage task and a versioned source, or skip it."""
    original = row['text']
    if '\nPassage: ' not in original:
        return None
    title, passage = original.split('\nPassage: ', 1)
    spans = list(re.finditer(r'\S+', passage))
    if len(spans) < words:
        return None
    segment = passage[spans[0].start():spans[words - 1].end()]
    text = title + '\nPassage: ' + segment
    source_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if not min_source_tokens <= source_tokens <= max_source_tokens:
        return None
    target_tokens = len(tokenizer.encode(segment, add_special_tokens=False))
    if target_tokens > max_source_tokens:
        return None
    source_id = hashlib.sha256(('short-reconstruction-v1:' + row['record_id'] +
                                ':' + text).encode()).hexdigest()[:32]
    source = Source(source_id, text, row['created_at'], row['kind'])
    episode = short_reconstruction(source, max_words=words, tokenizer=tokenizer,
                                   max_target_tokens=max_source_tokens + 1)
    if episode.answer != segment or segment in episode.query:
        raise AssertionError('The task must reconstruct the entire hidden passage')
    episode = replace(episode, provenance=episode.provenance | {
        'parent_source_id': row['record_id'], 'parent_context_sha256':
        row['provenance']['context_sha256'], 'article_title':
        row['provenance']['article_title'], 'source_tokens': source_tokens,
        'target_tokens': target_tokens, 'truncation': 'first-whole-words-v1'})
    return source, episode


def prepare(data: Path, output: Path, tokenizer, *, train_sources: int = 6000,
            validation_sources: int = 512, words: int = 28) -> dict:
    if min(train_sources, validation_sources, words) < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive budgets and fresh output parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Training data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Preparation requires a 10 GiB external disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for split in ('train', 'validation'):
        name = f'sources-{split}.jsonl'
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError('Prepared source bytes differ from manifest')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    seen_text = set()
    try:
        summaries = {}
        for split, count in (('train', train_sources), ('validation', validation_sources)):
            pairs = []
            with (data / f'sources-{split}.jsonl').open() as handle:
                for line in handle:
                    item = shorten_source(json.loads(line), tokenizer, words=words)
                    if item is None:
                        continue
                    source, _ = item
                    digest = hashlib.sha256(source.text.encode()).hexdigest()
                    if digest in seen_text:
                        continue
                    seen_text.add(digest)
                    pairs.append(item)
                    if len(pairs) == count:
                        break
            if len(pairs) != count:
                raise ValueError(f'Only {len(pairs)} eligible {split} sources for {count} requested')
            files = {f'{split}.jsonl': [asdict(e) for _, e in pairs],
                     f'sources-{split}.jsonl': [asdict(s) for s, _ in pairs]}
            for name, rows in files.items():
                with (pending / name).open('w', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            summaries[split] = {'sources': len(pairs), 'episodes': len(pairs),
                'sha256': {name: file_sha256(pending / name) for name in files}}
        result = {'format': 1, 'task': 'exact full-passage reconstruction',
                  'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'tokenizer_revision': REVISION, 'passage_words': words,
                  'source_token_range': [32, 64], 'splits': summaries,
                  'notice': 'Article-disjoint validation; source created before generic query. '
                            'No target text appears in the query.'}
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
    parser.add_argument('--train-sources', type=int, default=6000)
    parser.add_argument('--validation-sources', type=int, default=512)
    parser.add_argument('--words', type=int, default=28)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             train_sources=args.train_sources,
                             validation_sources=args.validation_sources,
                             words=args.words), indent=2))
