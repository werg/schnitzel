"""Standalone passage sources and causally separated questions for bank training."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .data import Episode, Source
from .trajectories import file_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _article_split(title: str) -> str:
    return 'validation' if int(_digest(title)[:8], 16) % 10 == 0 else 'train'


def _write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def short_reconstruction(source: Source, *, max_words: int = 40,
                         tokenizer=None, max_target_tokens: int | None = None) -> Episode:
    """Ask for a bounded exact passage prefix through a single prior record."""
    if max_words < 1 or '\nPassage: ' not in source.text:
        raise ValueError('Expected a passage source and a positive word budget')
    passage = source.text.split('\nPassage: ', 1)[1]
    spans = list(re.finditer(r'\S+', passage))[:max_words]
    def beginning() -> str:
        return passage[spans[0].start():spans[-1].end()] if spans else ''
    if tokenizer is not None:
        if max_target_tokens is None or max_target_tokens < 1:
            raise ValueError('A positive target token budget is required with a tokenizer')
        while spans and (len(tokenizer.encode(beginning(), add_special_tokens=False))
                         + int(tokenizer.eos_token_id is not None) > max_target_tokens):
            spans.pop()
    answer = beginning()
    if not answer:
        raise ValueError('Passage has no words')
    word_count = len(spans)
    return Episode(_digest(f'reconstruct:{source.record_id}:{max_words}:{word_count}')[:32],
                   'squad-reconstruction', (source,),
                   'Reproduce the beginning of the stored passage, preserving its wording. '
                   f'Give only the first {word_count} word{"s" if word_count != 1 else ""}.',
                   answer, (source.record_id,), False, 0, 0, source.created_at + 1,
                   'passage_reconstruction', (), ((source.record_id,),), 'verified',
                   {'source_id': source.record_id, 'max_words': max_words, 'actual_words': word_count,
                    'ordering': 'source-before-query'})


def with_distractor(episode: Episode, distractor: Source) -> Episode:
    """Add an earlier, independently identified source without changing the label."""
    if (distractor.created_at >= episode.query_time or
            distractor.record_id in {source.record_id for source in episode.supports}):
        raise ValueError('Distractor must be distinct and causally eligible')
    return replace(episode, supports=episode.supports + (distractor,),
                   provenance=episode.provenance | {'distractor_ids':
                                                      [distractor.record_id]})


def title_located_question(episode: Episode) -> Episode:
    """Place a source-internal article locator in the causal question prefix."""
    title = episode.provenance.get('article_title')
    if episode.task_family != 'passage_qa' or not isinstance(title, str) or not title:
        raise ValueError('A passage QA with article provenance is required')
    query = f'Article: {title}\nQuestion: {episode.query}'
    answer_words = re.sub(r'[\W_]+', ' ', episode.answer, flags=re.UNICODE).strip().casefold()
    query_words = re.sub(r'[\W_]+', ' ', query, flags=re.UNICODE).strip().casefold()
    if f' {answer_words} ' in f' {query_words} ':
        raise ValueError('Article locator would disclose the target answer')
    return replace(episode,
                   episode_id=_digest(f'title-located:{episode.episode_id}')[:32],
                   query=query,
                   provenance=episode.provenance | {'query_locator': 'source_article_title'})


def prepare_squad(raw: str | Path, output: str | Path, tokenizer, *,
                  max_train_sources: int, max_validation_sources: int,
                  max_source_tokens: int = 512, max_target_tokens: int = 128) -> dict:
    """Freeze complete answerable passages before questions; never insert targets into queries.

    Every source precedes all its questions in this declared corpus-ingest order.
    Article groups and exact context bytes are disjoint across the two splits.
    The source manifest has no question or answer fields.
    """
    if min(max_train_sources, max_validation_sources, max_source_tokens, max_target_tokens) < 1:
        raise ValueError('Positive source and token budgets required')
    raw, output = Path(raw), Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('A fresh output path with an existing parent is required')
    document = json.loads(raw.read_text(encoding='utf-8'))
    if document.get('version') != 'v2.0' or not isinstance(document.get('data'), list):
        raise ValueError('Expected the official SQuAD v2.0 train JSON shape')
    raw_digest = file_sha256(raw)
    candidates = {'train': [], 'validation': []}
    counts = Counter()
    seen_contexts = set()
    for article in document['data']:
        title = article['title']
        split = _article_split(title)
        for paragraph_index, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            context_hash = _digest(context)
            if context_hash in seen_contexts:
                counts['duplicate_context'] += 1
                continue
            seen_contexts.add(context_hash)
            source_text = f'Title: {title}\nPassage: {context}'
            source_tokens = len(tokenizer.encode(source_text, add_special_tokens=False))
            if not 32 <= source_tokens <= max_source_tokens:
                counts['source_length_filtered'] += 1
                continue
            valid_qas = []
            for qa in paragraph['qas']:
                if qa['is_impossible']:
                    counts['unanswerable_filtered'] += 1
                    continue
                question = qa['question'].strip()
                for answer in qa['answers']:
                    text, start = answer['text'], answer['answer_start']
                    if (not isinstance(start, int) or start < 0
                            or context[start:start + len(text)] != text):
                        counts['invalid_answer_span'] += 1
                        continue
                    if (not question or not text or text.casefold() in question.casefold()
                            or len(tokenizer.encode(text, add_special_tokens=False)) > max_target_tokens):
                        counts['answer_filtered'] += 1
                        continue
                    valid_qas.append((qa['id'], question, text, start))
                    break
            if not valid_qas:
                counts['no_eligible_questions'] += 1
                continue
            record_id = _digest(f'SQuAD-v2.0:{title}:{paragraph_index}:{context_hash}')[:32]
            source = Source(record_id, source_text, 1, 'passage')
            order = _digest(f'bank-order:{title}:{paragraph_index}:{context_hash}')
            candidates[split].append((order, title, paragraph_index, context_hash,
                                      source_tokens, source, valid_qas))
    budgets = {'train': max_train_sources, 'validation': max_validation_sources}
    selected = {split: sorted(rows)[:budgets[split]] for split, rows in candidates.items()}
    if any(len(selected[split]) != budgets[split] for split in budgets):
        raise ValueError('Insufficient eligible passages for the declared source budget')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        episode_counts = {}
        for split, rows in selected.items():
            sources, episodes = [], []
            for _, title, paragraph_index, context_hash, source_tokens, source, qas in rows:
                provenance = {'dataset': 'SQuAD-v2.0', 'raw_sha256': raw_digest,
                              'article_title': title, 'paragraph_index': paragraph_index,
                              'context_sha256': context_hash, 'license': 'CC-BY-SA-4.0',
                              'source_tokens': source_tokens, 'ordering': 'corpus-ingest-before-question'}
                sources.append(asdict(source) | {'domain': 'research', 'provenance': provenance})
                for qa_id, question, answer, start in qas:
                    episodes.append(Episode(
                        _digest(f'{source.record_id}:{qa_id}')[:32], f'squad-{split}', (source,),
                        question, answer, (source.record_id,), False, 0, 0, 2,
                        'passage_qa', (), ((source.record_id,),), 'verified',
                        provenance | {'qa_id': qa_id, 'answer_start': start}))
            _write_jsonl(pending / f'sources-{split}.jsonl', sources)
            _write_jsonl(pending / f'{split}.jsonl', (asdict(episode) for episode in episodes))
            episode_counts[split] = len(episodes)
        if not all(episode_counts.values()):
            raise ValueError('Each split needs answerable questions')
        files = ('sources-train.jsonl', 'sources-validation.jsonl', 'train.jsonl', 'validation.jsonl')
        manifest = {'format': 1, 'dataset': 'SQuAD-v2.0', 'source_url':
                    'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
                    'source_sha256': raw_digest, 'license': 'CC-BY-SA-4.0',
                    'split_policy': 'SHA256 article split (10% validation), exact context deduplication',
                    'ordering': 'Source passage ingested at time 1; question at time 2',
                    'sources': {split: len(selected[split]) for split in budgets},
                    'episodes': episode_counts, 'filters': dict(counts),
                    'max_source_tokens': max_source_tokens, 'max_target_tokens': max_target_tokens,
                    'sha256': {name: file_sha256(pending / name) for name in files}}
        (pending / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(pending, output)
    except BaseException:
        import shutil
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return manifest
