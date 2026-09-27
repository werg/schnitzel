"""Prepare content-located span queries and a standalone source manifest."""
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


def located_span(row: dict, *, locator_words: int = 6, span_words: int = 8):
    episode = episode_from_dict(row)
    if (episode.task_family != 'passage_reconstruction' or len(episode.supports) != 1
            or locator_words < 1 or span_words < 1):
        raise ValueError('Located span needs one verified short-passage source')
    source = episode.supports[0]
    passage = source.text.split('\nPassage: ', 1)[1]
    words = list(re.finditer(r'\S+', passage))
    if len(words) < locator_words + span_words:
        raise ValueError('Source is too short for a disjoint locator and target')
    locator = passage[words[0].start():words[locator_words - 1].end()]
    choices = len(words) - locator_words - span_words + 1
    digest = hashlib.sha256(('bank-located-span-v1:' + source.record_id).encode()).digest()
    start = locator_words + int.from_bytes(digest[:8], 'big') % choices
    end = start + span_words
    answer = passage[words[start].start():words[end - 1].end()]
    title = episode.provenance['article_title']
    query = (f'Article: {title}\nStored passage begins: {locator}\n'
             f'Return exactly words {start + 1} through {end} of that passage. '
             'Give only those words.')
    if answer in query:
        raise ValueError('Located query reveals its answer')
    return replace(episode,
        episode_id=hashlib.sha256(('bank-located-span-v1:' + episode.episode_id +
                                   f':{start}:{end}').encode()).hexdigest()[:32],
        query=query, answer=answer, task_family='located_passage_span',
        provenance=episode.provenance | {'locator_words': locator_words,
            'span_words': span_words, 'span_start_word': start + 1,
            'span_end_word': end, 'query_locator': 'article-and-passage-prefix-v1'})


def prepare(short_data: Path, output: Path, *, locator_words: int = 6,
            span_words: int = 8) -> dict:
    if min(locator_words, span_words) < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive budgets and a fresh output parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Bank data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Preparation requires a 10 GiB external disk reserve')
    manifest = json.loads((short_data / 'manifest.json').read_text())
    names = [f'{split}.jsonl' for split in ('train', 'validation')] + [
        f'sources-{split}.jsonl' for split in ('train', 'validation')]
    for name in names:
        split = 'validation' if 'validation' in name else 'train'
        if file_sha256(short_data / name) != manifest['splits'][split]['sha256'][name]:
            raise ValueError(f'Short source input differs from manifest: {name}')
    parent_manifest_sha = file_sha256(short_data / 'manifest.json')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        summaries, all_sources, training_ids = {}, [], set()
        for split in ('train', 'validation'):
            rows = [json.loads(line) for line in (short_data / f'{split}.jsonl').open()]
            sources = [json.loads(line) for line in
                       (short_data / f'sources-{split}.jsonl').open()]
            source_ids = {item['record_id'] for item in sources}
            if len(source_ids) != len(rows) or any(row['required_ids'][0] not in source_ids
                                                   for row in rows):
                raise ValueError('One unique source per passage query required')
            if split == 'train':
                training_ids = source_ids
            elif source_ids & training_ids:
                raise ValueError('Training and validation source IDs overlap')
            episode_by_source = {row['required_ids'][0]: row for row in rows}
            for source in sources:
                parent = episode_by_source[source['record_id']]['provenance']
                all_sources.append(source | {'domain': 'research', 'provenance': {
                    'short_source_id': source['record_id'],
                    'parent_source_id': parent['parent_source_id'],
                    'parent_context_sha256': parent['parent_context_sha256'],
                    'article_title': parent['article_title'],
                    'short_corpus_manifest_sha256': parent_manifest_sha,
                    'source_text_sha256': hashlib.sha256(source['text'].encode()).hexdigest()}})
            episodes = [asdict(located_span(row, locator_words=locator_words,
                                           span_words=span_words)) for row in rows]
            name = f'{split}.jsonl'
            with (pending / name).open('w', encoding='utf-8') as handle:
                for episode in episodes:
                    handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            summaries[split] = {'sources': len(sources), 'episodes': len(episodes),
                                'sha256': file_sha256(pending / name)}
        with (pending / 'sources-all.jsonl').open('w', encoding='utf-8') as handle:
            for source in all_sources:
                handle.write(json.dumps(source, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = {'format': 1, 'task': 'content-located indexed span',
                  'parent_manifest_sha256': parent_manifest_sha,
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'locator_words': locator_words, 'span_words': span_words,
                  'splits': summaries, 'bank_sources': len(all_sources),
                  'bank_sources_sha256': file_sha256(pending / 'sources-all.jsonl'),
                  'notice': 'The locator is a passage prefix before the target span. '
                            'All bank source text is prior to its query; validation sources '
                            'are distinct and included only as unlabeled bank records.'}
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
    parser.add_argument('--locator-words', type=int, default=6)
    parser.add_argument('--span-words', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.short_data, args.output,
                             locator_words=args.locator_words,
                             span_words=args.span_words), indent=2))
