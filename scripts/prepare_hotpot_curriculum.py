"""Prepare a 100k-source reconstruction and multi-hop transfer curriculum."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import pyarrow.parquet as pq

from schnitz.data import Episode, Source
from schnitz.operations import atomic_json
from schnitz.text import render_prompt
from schnitz.trajectories import file_sha256


DATASET_REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
MODEL_REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rows(paths):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def paragraph_chunks(title: str, sentences: list[str], tokenizer,
                     max_source_tokens: int) -> tuple[list[Source], dict[int, list[str]]]:
    """Pack complete sentences; split exceptional long sentences at word boundaries."""
    source_title = title.strip()
    title_budget = min(20, max_source_tokens // 3)
    while source_title and len(tokenizer.encode(
            f'Title: {source_title}\nPassage: x', add_special_tokens=True)) > title_budget:
        source_title = source_title[:-1].rstrip()
    prefix = f'Title: {source_title}\nPassage: '
    pieces = []
    for sentence_index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        words = list(re.finditer(r'\S+', sentence))
        start = 0
        while start < len(words):
            lo, hi, best = start + 1, len(words), None
            while lo <= hi:
                mid = (lo + hi) // 2
                text = sentence[words[start].start():words[mid - 1].end()]
                if len(tokenizer.encode(prefix + text, add_special_tokens=True)) <= max_source_tokens:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best is None:
                word = sentence[words[start].start():words[start].end()]
                offset = 0
                while offset < len(word):
                    lo, hi, char_best = offset + 1, len(word), None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if len(tokenizer.encode(prefix + word[offset:mid],
                                                add_special_tokens=True)) <= max_source_tokens:
                            char_best, lo = mid, mid + 1
                        else:
                            hi = mid - 1
                    if char_best is None:
                        raise ValueError('Source prefix leaves no room for content')
                    pieces.append((sentence_index, word[offset:char_best]))
                    offset = char_best
                start += 1
                continue
            text = sentence[words[start].start():words[best - 1].end()]
            pieces.append((sentence_index, text))
            start = best
    packed = []
    for sentence_index, text in pieces:
        if packed:
            trial = packed[-1][1] + ' ' + text
            if len(tokenizer.encode(prefix + trial, add_special_tokens=True)) <= max_source_tokens:
                packed[-1][1] = trial
                packed[-1][2].add(sentence_index)
                continue
        packed.append([sentence_index, text, {sentence_index}])
    sources, by_sentence = [], {}
    for _, passage, sentence_ids in packed:
        text = prefix + passage
        record_id = _sha(f'hotpot:{DATASET_REVISION}:{text}')[:32]
        source = Source(record_id, text, 1, 'passage')
        sources.append(source)
        for sentence_index in sentence_ids:
            by_sentence.setdefault(sentence_index, []).append(record_id)
    return sources, by_sentence


def _contexts(row, tokenizer, max_source_tokens, cache):
    result = {}
    for title, sentences in zip(row['context']['title'], row['context']['sentences'], strict=True):
        key = (title, tuple(sentences))
        if key not in cache:
            cache[key] = paragraph_chunks(title, sentences, tokenizer, max_source_tokens)
        result[title] = cache[key]
    return result


def _episode(row, contexts, *, split: str, max_supports: int,
             max_prompt_tokens: int, max_target_tokens: int, tokenizer):
    required = []
    for title, sentence_index in zip(row['supporting_facts']['title'],
                                     row['supporting_facts']['sent_id'], strict=True):
        if title not in contexts or sentence_index not in contexts[title][1]:
            return None
        required.extend(contexts[title][1][sentence_index])
    required = list(dict.fromkeys(required))
    if not 1 <= len(required) <= max_supports:
        return None
    by_id = {source.record_id: source for sources, _ in contexts.values() for source in sources}
    distractors = [source for title, (sources, _) in contexts.items()
                   if title not in set(row['supporting_facts']['title']) for source in sources]
    offset = int(_sha('distractors:' + row['id'])[:16], 16)
    distractors = (distractors[offset % len(distractors):] +
                   distractors[:offset % len(distractors)]) if distractors else []
    supports = [by_id[record_id] for record_id in required]
    supports.extend(source for source in distractors
                    if source.record_id not in required) 
    supports = supports[:max_supports + 2]
    query = ('Use the previously stored passages. Give only the short response.\n'
             'Question: ' + row['question'].strip())
    answer = row['answer'].strip()
    if (not answer or answer in query
            or len(tokenizer.encode(render_prompt(tokenizer, query, ''),
                                    add_special_tokens=False)) > max_prompt_tokens
            or len(tokenizer.encode(answer, add_special_tokens=False)) +
                    int(tokenizer.eos_token_id is not None) > max_target_tokens):
        return None
    return Episode(_sha(f'hotpot:{split}:{row["id"]}')[:32], f'hotpot-{split}',
        tuple(supports), query, answer, tuple(required), False, 0, 0, 2,
        'hotpot_multihop', (), (tuple(required),), 'verified',
        {'dataset': 'hotpot_qa', 'dataset_revision': DATASET_REVISION,
         'original_id': row['id'], 'original_split': split,
         'question_type': row['type'], 'level': row['level'],
         'license': 'cc-by-sa-4.0', 'distractor_policy': 'same-row-nonsupport-v1'})


def _reconstruction(source: Source, title: str) -> Episode | None:
    passage = source.text.split('\nPassage: ', 1)[1]
    words = list(re.finditer(r'\S+', passage))
    if len(words) < 14:
        return None
    start = 6 + int(_sha('reconstruct:' + source.record_id)[:16], 16) % (len(words) - 13)
    answer = passage[words[start].start():words[start + 7].end()]
    locator = passage[words[0].start():words[5].end()]
    query = (f'Article: {title}\nStored passage begins: {locator}\n'
             f'Return exactly words {start + 1} through {start + 8} of that stored passage. '
             'Give only those words.')
    if answer in query:
        return None
    return Episode(_sha('reconstruct:' + source.record_id)[:32], 'hotpot-reconstruction',
        (source,), query, answer, (source.record_id,), False, 0, 0, 2,
        'passage_span', (), ((source.record_id,),), 'verified',
        {'dataset': 'hotpot_qa', 'dataset_revision': DATASET_REVISION,
         'article_title': title, 'span_start_word': start + 1,
         'span_end_word': start + 8, 'query_locator': 'article-and-passage-prefix-v1'})


def _reconstruction_mix(sources, titles, excluded_ids: set[str], count: int):
    episodes = [episode for rid, source in sources.items()
                if rid not in excluded_ids
                and (episode := _reconstruction(source, titles[rid])) is not None]
    episodes.sort(key=lambda episode: _sha('mix:' + episode.episode_id))
    return episodes[:count]


def prepare(raw: Path, output: Path, tokenizer, *, bank_sources: int = 100_000,
            max_source_tokens: int = 65, max_prompt_tokens: int = 128,
            max_target_tokens: int = 65, max_supports: int = 4,
            validation_episodes: int = 1024) -> dict:
    train_paths = sorted(raw.glob('train-*.parquet'))
    validation_paths = sorted(raw.glob('validation-*.parquet'))
    readme = raw / 'README.md'
    if (not train_paths or not validation_paths or not readme.is_file() or output.exists()
            or not output.parent.is_dir() or bank_sources < 1):
        raise ValueError('Pinned raw splits, fresh output and positive source budget required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Target curriculum must live on the external disk')
    if shutil.disk_usage(output.parent).free < 20 * 1024**3:
        raise OSError('Target curriculum preparation requires a 20 GiB reserve')

    cache, sources, titles, candidates = {}, {}, {}, {'train': [], 'validation': []}
    # Required evidence is admitted before unrelated bank material.
    for split, paths in [('train', train_paths), ('validation', validation_paths)]:
        for row in _rows(paths):
            contexts = _contexts(row, tokenizer, max_source_tokens, cache)
            context_titles = {source.record_id: title for title, (chunks, _) in contexts.items()
                              for source in chunks}
            episode = _episode(row, contexts, split=split, max_supports=max_supports,
                               max_prompt_tokens=max_prompt_tokens,
                               max_target_tokens=max_target_tokens, tokenizer=tokenizer)
            if episode is None:
                continue
            for source in episode.supports[:len(episode.required_ids)]:
                sources[source.record_id] = source
                titles[source.record_id] = context_titles[source.record_id]
            candidates[split].append(episode)
    if len(sources) > bank_sources:
        keep = sorted(sources, key=lambda rid: _sha('required-order:' + rid))[:bank_sources]
        sources = {rid: sources[rid] for rid in keep}
        titles = {rid: titles[rid] for rid in keep}

    # Fill to exactly 100k with independently reusable context chunks.
    if len(sources) < bank_sources:
        cache.clear()
        for paths in (train_paths, validation_paths):
            for row in _rows(paths):
                for title, (chunks, _) in _contexts(row, tokenizer, max_source_tokens, cache).items():
                    for source in chunks:
                        if source.record_id not in sources:
                            sources[source.record_id] = source
                            titles[source.record_id] = title
                            if len(sources) == bank_sources:
                                break
                    if len(sources) == bank_sources:
                        break
                if len(sources) == bank_sources:
                    break
            if len(sources) == bank_sources:
                break
    if len(sources) != bank_sources:
        raise ValueError('Raw corpus did not yield the requested unique source count')

    train_required = {rid for episode in candidates['train'] for rid in episode.required_ids
                      if set(episode.required_ids) <= sources.keys()}
    transfer_train = [episode for episode in candidates['train']
                      if set(episode.required_ids) <= sources.keys()]
    transfer_validation = [episode for episode in candidates['validation']
        if set(episode.required_ids) <= sources.keys()
        and not set(episode.required_ids) & train_required][:validation_episodes]
    validation_required = {rid for episode in transfer_validation
                           for rid in episode.required_ids}
    # Twenty-five percent reconstruction by episode count; deterministic ordering.
    reconstruction = _reconstruction_mix(sources, titles, validation_required,
                                         len(transfer_train) // 3)
    train = transfer_train + reconstruction
    train.sort(key=lambda episode: _sha('train-order:' + episode.episode_id))

    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        paths = {}
        for name, rows in [('sources-all.jsonl', [asdict(sources[rid]) | {
                'domain': 'research', 'provenance': {'dataset': 'hotpot_qa',
                'dataset_revision': DATASET_REVISION, 'article_title': titles[rid],
                'source_text_sha256': _sha(sources[rid].text), 'license': 'cc-by-sa-4.0'}}
                for rid in sorted(sources)]),
                ('train.jsonl', [asdict(episode) for episode in train]),
                ('validation.jsonl', [asdict(episode) for episode in transfer_validation])]:
            path = pending / name
            with path.open('w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            paths[name] = {'rows': len(rows), 'sha256': file_sha256(path)}
        result = {'format': 1, 'curriculum': 'four-space reconstruction and multi-hop transfer',
                  'dataset': 'hotpotqa/hotpot_qa', 'configuration': 'distractor',
                  'dataset_revision': DATASET_REVISION, 'license': 'cc-by-sa-4.0',
                  'model_revision': MODEL_REVISION, 'raw': {path.name: file_sha256(path)
                      for path in [readme, *train_paths, *validation_paths]},
                  'limits': {'max_source_tokens': max_source_tokens,
                      'max_prompt_tokens': max_prompt_tokens,
                      'max_target_tokens': max_target_tokens,
                      'max_supports': max_supports}, 'files': paths,
                  'train_transfer_episodes': len(transfer_train),
                  'train_reconstruction_episodes': len(reconstruction),
                  'validation_required_source_disjoint': True,
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'notice': 'Source text precedes every query. Validation labels are withheld from training; '
                            'all stored validation sources remain unlabeled bank records.'}
        atomic_json(pending / 'manifest.json', result)
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/cache'))
    parser.add_argument('--bank-sources', type=int, default=100_000)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                               trust_remote_code=False)
    print(json.dumps(prepare(args.raw, args.output, tokenizer,
                             bank_sources=args.bank_sources), indent=2))
