"""Build a bounded reconstruction/QA pilot from the external SQuAD source manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import schnitz.corpus_data as corpus_data
from schnitz.corpus_data import short_reconstruction, with_distractor
from schnitz.data import Source, episode_from_dict
from schnitz.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(data: Path, output: Path, tokenizer, *, source_count: int = 1024,
            validation_count: int = 64, max_words: int = 40,
            max_target_tokens: int = 128, add_distractor: bool = False) -> dict:
    if min(source_count, validation_count, max_words, max_target_tokens) < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive budgets and a fresh output under an existing parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Pilot data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Pilot preparation requires a 10 GiB disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('sources-train.jsonl', 'train.jsonl',
                 'sources-validation.jsonl', 'validation.jsonl'):
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError(f'Prepared source changed: {name}')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        summaries = {}
        for split, count in (('train', source_count), ('validation', validation_count)):
            source_rows = [json.loads(line) for line in
                           (data / f'sources-{split}.jsonl').read_text().splitlines()]
            if len(source_rows) < count:
                raise ValueError('Source budget exceeds prepared corpus')
            selected_rows = source_rows[:count]
            selected = {row['record_id']: Source(**{key: row[key] for key in
                                                   ('record_id', 'text', 'created_at', 'kind')})
                        for row in selected_rows}
            qa = {}
            for line in (data / f'{split}.jsonl').open():
                row = json.loads(line)
                source_id = row['required_ids'][0]
                if source_id in selected and source_id not in qa:
                    qa[source_id] = row
            if len(qa) != len(selected):
                raise ValueError('Every selected source needs a verified QA')
            episodes = []
            for source_index, (source_id, source) in enumerate(selected.items()):
                pair = [short_reconstruction(source, max_words=max_words, tokenizer=tokenizer,
                                             max_target_tokens=max_target_tokens),
                        episode_from_dict(qa[source_id])]
                if add_distractor:
                    title = selected_rows[source_index]['provenance']['article_title']
                    start = int(hashlib.sha256(f'distractor:{source_id}'.encode()).hexdigest()[:8], 16) % count
                    alternate = next((selected_rows[(start + j) % count] for j in range(count)
                                      if selected_rows[(start + j) % count]['record_id'] != source_id
                                      and selected_rows[(start + j) % count]['provenance']['article_title'] != title),
                                     None)
                    if alternate is None:
                        raise ValueError('No distinct-article distractor in the selected split')
                    decoy = Source(**{key: alternate[key] for key in
                                      ('record_id', 'text', 'created_at', 'kind')})
                    pair = [with_distractor(episode, decoy) for episode in pair]
                episodes.extend(asdict(episode) for episode in pair)
            for episode in episodes:
                tokens = len(tokenizer.encode(episode['answer'], add_special_tokens=False))
                if tokens + int(tokenizer.eos_token_id is not None) > max_target_tokens:
                    raise ValueError('Complete target exceeds the declared token budget')
            episodes.sort(key=lambda row: hashlib.sha256(row['episode_id'].encode()).digest())
            path = pending / f'{split}.jsonl'
            with path.open('w') as handle:
                for episode in episodes:
                    handle.write(json.dumps(episode, sort_keys=True, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            summaries[split] = {'sources': count, 'episodes': len(episodes),
                                'sha256': file_sha256(path)}
        result = {'format': 2, 'source_manifest_sha256': file_sha256(data / 'sources-train.jsonl'),
                  'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'corpus_module_sha256': file_sha256(Path(corpus_data.__file__)),
                  'source_count': source_count, 'max_words': max_words,
                  'max_target_tokens': max_target_tokens,
                  'add_distractor': add_distractor,
                  'tokenizer_revision': REVISION,
                  'splits': summaries,
                  'notice': ('Verified one-source task labels with an earlier independent distractor.' if add_distractor
                             else 'One source record per task, supplied oracle selection; this stage does not train global retrieval.')}
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
    parser.add_argument('--source-count', type=int, default=1024)
    parser.add_argument('--validation-count', type=int, default=64)
    parser.add_argument('--max-words', type=int, default=40)
    parser.add_argument('--max-target-tokens', type=int, default=128)
    parser.add_argument('--with-distractor', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             source_count=args.source_count,
                             validation_count=args.validation_count,
                             max_words=args.max_words,
                             max_target_tokens=args.max_target_tokens,
                             add_distractor=args.with_distractor), indent=2))
