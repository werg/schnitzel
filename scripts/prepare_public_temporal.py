"""Causally dated public QA domains: StreamingQA (DeepMind) and TimeQA.

StreamingQA questions are dated, and their evidence is a dated WMT news-crawl
article referenced by a sorting-key ID (publication date and SHA-256 of the
base64 unsplit article line). Only the referenced articles and a bounded pool of
same-/nearby-date distractors are reconstructed, by streaming the document-split
WMT archives once. Every passage's ``created_at`` is its publication date and
the episode ``query_time`` is the question date, both as whole days since
2000-01-01. A question whose evidence is not strictly older than the question is
dropped and counted; dates are never shifted. Distractors are published on the
evidence day or up to ``NEARBY_DAYS`` earlier, so they are causally prior too.
Upstream train questions before the cutoff become training episodes; upstream
valid questions from the cutoff to the end date become validation, so every
validation question is dated after every training question. Upstream eval
(2020) is not used.

TimeQA asks about facts that change over time; the time is in the question, and
the Wikipedia page is the context. There is no principled document date, so
passages have ``created_at=1`` and queries ``query_time=2``. Gold is the chunk
holding the annotated answer span (train) or every chunk containing the answer
string (dev has no spans); distractors are other chunks of the same page, i.e.
the same entity at other time periods. Easy and hard modes are both used.

Answers are supervised targets only, never query text. Source commands/text are
inert data.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import datetime as dt
import gzip
import hashlib
import heapq
import json
from multiprocessing import Pool
from pathlib import Path
import random
import re
from urllib.parse import unquote

from public_corpus_common import (MAX_CHARS, PROMPT, Writer, chunk, clean,
                                  episode, load_tokenizer, source)

EPOCH = dt.date(2000, 1, 1)
NEARBY_DAYS = 3
POOL_CANDIDATES = 12
POOL_PER_DAY = 8
DISTRACTORS = 2
MAX_GOLD = 4
MIN_QUESTION_WORDS = 4
TOXICITY_LIMIT = 0.5
WMT_KEY_SEPARATOR = '\x00\x01'
STOPWORDS = frozenset(
    'a an the of in on at to for from by with and or is was were be been are did does do '
    'what which who whom whose when where why how that this these those it its as his her '
    'their he she they has have had not than then into about after before during'.split())


def day_number(date: dt.date) -> int:
    return (date - EPOCH).days


def day_of_timestamp(ts: int) -> int:
    return day_number(dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date())


def date_of_day(day: int) -> dt.date:
    return EPOCH + dt.timedelta(days=day)


def long_date(day: int) -> str:
    date = date_of_day(day)
    return f'{date:%A}, {date:%B} {date.day}, {date.year}'


def words(text: str) -> set[str]:
    return {w for w in re.findall(r'\w+', text.lower()) if w not in STOPWORDS}


def priority(seed: int, key: str) -> int:
    return int(hashlib.sha256(f'{seed}\0{key}'.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------- StreamingQA

def wmt_key(date_field: str, unsplit: bytes) -> str:
    """StreamingQA sorting key of one WMT doc line (see upstream ``extraction.py``)."""
    stamp = dt.datetime.strptime(date_field, '%Y%m%d').strftime('%Y%m%d%H%M%S%f')
    return WMT_KEY_SEPARATOR.join([stamp, hashlib.sha256(unsplit).hexdigest(), ''])


def key_day(key: str) -> int:
    return day_number(dt.datetime.strptime(key[:8], '%Y%m%d').date())


def article_sentences(split_b64: bytes) -> list[str]:
    text = base64.b64decode(split_b64).decode('utf-8', errors='replace')
    return [clean(line) for line in text.split('\n') if clean(line)]


def dated_chunks(sentences: list[str], day: int) -> list[str]:
    prefix = long_date(day) + '. '
    return [prefix + text for text, _, _ in chunk(sentences, limit=MAX_CHARS - len(prefix))]


def scan_wmt(job: tuple) -> tuple[dict, dict, Counter]:
    """Stream one archive: keep needed articles and bottom-k distractor candidates."""
    path, needed, pool_days, seed = job
    articles, pools, counts = {}, {}, Counter()
    with gzip.open(path) as handle:
        for line in handle:
            counts['lines'] += 1
            try:
                date_field, split_b64, unsplit = line.strip().split(b'\t')
            except ValueError:
                counts['malformed'] += 1
                continue
            day = day_number(dt.datetime.strptime(date_field.decode(), '%Y%m%d').date())
            wanted = day in pool_days
            if not wanted and day not in needed['days']:
                continue
            key = wmt_key(date_field.decode(), unsplit)
            if key in needed['keys']:
                articles[key] = (day, article_sentences(split_b64))
                counts['evidence_found'] += 1
            if wanted:
                # Bottom-k sample by a seeded key hash: deterministic and mergeable.
                heap = pools.setdefault(day, [])
                entry = (-priority(seed, key), key, split_b64)
                if len(heap) < POOL_CANDIDATES:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
    result = {}
    for day, heap in pools.items():
        rows = []
        for negated, key, split_b64 in heap:
            sentences = article_sentences(split_b64)
            texts = dated_chunks(sentences, day) if sentences else []
            if texts:
                rows.append((-negated, key, texts[0]))
        result[day] = rows
    return articles, result, counts


def load_streamingqa(path: Path, split: str, start: int | None, end: int,
                     filters) -> list[dict]:
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            query_day = day_of_timestamp(row['question_ts'])
            if query_day >= end or (start is not None and query_day < start):
                continue
            if max(value for name, value in row.items() if name.startswith('toxicity_')) >= TOXICITY_LIMIT:
                filters.reject('toxicity')
                continue
            if len(row['answers']) != 1:
                filters.reject('answer_count')
                continue
            rows.append({'qa_id': row['qa_id'], 'question': row['question'],
                         'answer': row['answers'][0], 'query_day': query_day,
                         'evidence_day_ts': day_of_timestamp(row['evidence_ts']),
                         'evidence_id': row['evidence_id'], 'recent_or_past': row['recent_or_past'],
                         'written_or_generated': row['written_or_generated'], 'upstream_split': split})
    rows.sort(key=lambda row: row['qa_id'])
    random.Random(f'1701:{split}').shuffle(rows)
    return rows


def dedup_filter(path: Path | None, keys: set[str]) -> set[str]:
    """Subset of ``keys`` present in the upstream deduplicated WMT key list (streamed)."""
    if path is None:
        return set(keys)
    kept = set()
    with gzip.open(path, 'rt', encoding='utf-8', newline='\n') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if line in keys:
                kept.add(line)
    return kept


def streamingqa_episode(row: dict, split: str, article: tuple[int, list[str]] | None,
                        pool: list[tuple[int, str, str]], writer: Writer, tokenizer,
                        max_sources: int) -> bool:
    filters = writer.filters_for(split)
    if article is None:
        filters.reject('evidence_missing')
        return False
    day, sentences = article
    if day != row['evidence_day_ts']:
        filters.reject('evidence_date_mismatch')
        return False
    if day >= row['query_day']:
        filters.reject('evidence_not_before_query')
        return False
    if len(row['question'].split()) < MIN_QUESTION_WORDS:
        filters.reject('question_too_short')
        return False
    answer = clean(row['answer'])
    date_line = f"Date: {long_date(row['query_day'])}\n"
    if answer.lower() in date_line.lower():
        filters.reject('answer_in_query_date')
        return False
    lowered = answer.lower()
    texts = dated_chunks(sentences, day)
    prefix = len(long_date(day)) + 2
    article_id = row['evidence_id'].split(WMT_KEY_SEPARATOR)[1]
    meta = {'article_id': article_id, 'publication_date': date_of_day(day).isoformat()}
    # Gold: the article body states the answer (not merely the date prefix).
    gold = [source('streamingqa', text, created_at=day,
                   provenance={**meta, 'chunk_index': index, 'role_hint': 'evidence_article'})
            for index, text in enumerate(texts) if lowered in text[prefix:].lower()]
    if not gold:
        filters.reject('answer_not_in_article_chunk')
        return False
    if len(gold) > MAX_GOLD:
        filters.reject('answer_in_too_many_chunks')
        return False
    question_words = words(row['question'])
    gold_texts = {item['text'] for item in gold}
    # Distractors must not contain the answer anywhere, date prefix included.
    ranked = sorted(((-len(question_words & words(text)), rank, key, text)
                     for rank, key, text in pool
                     if key != row['evidence_id'] and lowered not in text.lower()
                     and text not in gold_texts), key=lambda item: item[:2])
    distractors = []
    for _, _, key, text in ranked[:DISTRACTORS]:
        other_day = key_day(key)
        distractors.append(source(
            'streamingqa', text, created_at=other_day,
            provenance={'article_id': key.split(WMT_KEY_SEPARATOR)[1],
                        'publication_date': date_of_day(other_day).isoformat(),
                        'chunk_index': 0, 'role_hint': 'nearby_date_lede'}))
    supports = gold + distractors
    new = {item['record_id'] for item in supports} - writer.sources.keys()
    if len(writer.sources) + len(new) > max_sources:
        filters.reject('source_budget')
        return False
    item = episode(
        domain='streamingqa', split=split, identifier=row['qa_id'], question=row['question'],
        answer=answer, gold=gold, supports=supports, filters=filters, tokenizer=tokenizer,
        query_time=row['query_day'], all_required=False, annotation='answer_match',
        task_family='public_temporal_qa', prompt=PROMPT.replace('Question: ', date_line + 'Question: '),
        provenance={'qa_id': row['qa_id'], 'upstream_split': row['upstream_split'],
                    'question_date': date_of_day(row['query_day']).isoformat(),
                    'evidence_date': date_of_day(day).isoformat(), 'evidence_article_id': article_id,
                    'recent_or_past': row['recent_or_past'],
                    'written_or_generated': row['written_or_generated'],
                    'distractors': len(distractors)})
    return writer.add(split, item, supports)


def build_streamingqa(raw: Path, output: Path, *, wmt_files: list[Path], dedup_keys: Path | None,
                      train: int, validation: int, max_sources: int, cutoff: dt.date,
                      end: dt.date, workers: int = 2, oversample: float = 1.6, seed: int = 1701,
                      tokenizer=None) -> dict:
    writer = Writer(output, 'streamingqa')
    start, stop = day_number(cutoff), day_number(end)
    candidates = {
        'train': load_streamingqa(raw / 'streaminqa_train.jsonl.gz', 'train', None, start,
                                  writer.filters_for('train')),
        'validation': load_streamingqa(raw / 'streaminqa_valid.jsonl.gz', 'valid', start, stop,
                                       writer.filters_for('validation')),
    }
    available = {split: len(rows) for split, rows in candidates.items()}
    candidates['train'] = candidates['train'][:int(train * oversample)]
    candidates['validation'] = candidates['validation'][:int(validation * oversample)]
    keys = {row['evidence_id'] for rows in candidates.values() for row in rows}
    gold_days = {key_day(key) for key in keys}
    pool_days = {day - offset for day in gold_days for offset in range(NEARBY_DAYS + 1)}
    needed = {'keys': keys, 'days': gold_days}
    articles, pools, scan = {}, {}, Counter()
    jobs = [(path, needed, pool_days, seed) for path in wmt_files]

    def merge(results) -> None:
        for found, candidates_by_day, counts in results:
            articles.update(found)
            scan.update(counts)
            for day, rows in candidates_by_day.items():
                pools[day] = sorted(pools.get(day, []) + rows)[:POOL_CANDIDATES]

    if workers <= 1:
        merge(map(scan_wmt, jobs))
    else:
        with Pool(min(workers, len(jobs))) as pool:
            merge(pool.imap_unordered(scan_wmt, jobs))
    unique = dedup_filter(dedup_keys, {key for rows in pools.values() for _, key, _ in rows}
                          | set(articles))
    scan['evidence_not_in_dedup_list'] = len(set(articles) - unique)
    for day in pools:
        pools[day] = [row for row in pools[day] if row[1] in unique][:POOL_PER_DAY]
    for split, wanted in (('train', train), ('validation', validation)):
        kept = 0
        for row in candidates[split]:
            if kept == wanted:
                break
            gold_day = key_day(row['evidence_id'])
            pool = [item for offset in range(NEARBY_DAYS + 1)
                    for item in pools.get(gold_day - offset, [])]
            kept += streamingqa_episode(row, split, articles.get(row['evidence_id']), pool,
                                        writer, tokenizer, max_sources)
    times = {split: sorted({row['query_time'] for row in rows})
             for split, rows in writer.episodes.items()}
    overlap = ({row['provenance']['evidence_article_id'] for row in writer.episodes.get('train', [])}
               & {row['provenance']['evidence_article_id']
                  for row in writer.episodes.get('validation', [])})
    manifest = {
        'upstream': 'https://github.com/deepmind/streamingqa (questions CC-BY-4.0, code Apache-2.0)',
        'wmt': 'https://data.statmt.org/news-crawl/doc/en/ document-split English news crawl',
        'wmt_files': [path.name for path in wmt_files],
        'time_unit': 'days since 2000-01-01 (UTC date); created_at=publication date, '
                     'query_time=question date; created_at < query_time strictly',
        'split': {'train': f'upstream train questions dated before {cutoff}',
                  'validation': f'upstream valid questions dated {cutoff} to before {end}',
                  'unused': 'upstream eval (2020 questions); upstream train/valid on or after cutoff',
                  'note': 'upstream train/valid are a random split over 2007-2019; the cutoff makes '
                          'every validation question later than every training question'},
        'available_candidates': available, 'scanned_candidates': {
            split: len(rows) for split, rows in candidates.items()},
        'distinct_query_times': {split: len(values) for split, values in times.items()},
        'distinct_query_times_total': len(set().union(*times.values())) if times else 0,
        'query_time_range': {split: [values[0], values[-1]] for split, values in times.items() if values},
        'evidence_articles_shared_train_validation': len(overlap),
        'scan': dict(scan), 'pool_days': len(pools),
        'distractors': (f'up to {DISTRACTORS} lede chunks of deduplicated articles published on the '
                        f'evidence day or up to {NEARBY_DAYS} days earlier, ranked by question word '
                        'overlap, excluding the evidence article and any chunk containing the answer'),
        'toxicity_limit': TOXICITY_LIMIT, 'seed': seed,
        'annotation': ('answer_match: evidence article from upstream; gold chunks are its chunks '
                       f'whose body contains the answer (at most {MAX_GOLD}, any one suffices)'),
        'min_question_words': MIN_QUESTION_WORDS,
    }
    return writer.close(manifest)


# -------------------------------------------------------------------- TimeQA

def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def sentence_spans(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    spans, begin = [], 0
    for match in re.finditer(r'(?<=[.!?])\s+', text):
        spans.append((text[begin:match.start()], offset + begin, offset + match.start()))
        begin = match.end()
    spans.append((text[begin:], offset + begin, offset + len(text)))
    return [(clean(part), a, b) for part, a, b in spans if clean(part)]


def load_timeqa(path: Path, mode: str, filters) -> tuple[list[dict], dict[str, tuple]]:
    opener = gzip.open if path.suffix in {'.gz', '.gzip'} else open
    rows, pages = [], {}
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            targets = [clean(item) for item in _literal(row['targets']) if clean(item)]
            if not targets:
                filters.reject('unanswerable')
                continue
            if len(targets) > 1:
                filters.reject('multiple_targets')
                continue
            page = row['idx'].split('#')[0]
            context = row['context']
            known = pages.setdefault(page, (page_title(page), context,
                                            body_start(context, _literal(row['paragraphs']))))
            if known[1] != context:
                raise ValueError(f'{page}: two different contexts')
            spans = list(zip(row.get('from') or [], row.get('end') or []))
            rows.append({'idx': row['idx'], 'page': page, 'question': row['question'],
                         'answer': targets[0], 'span': spans[0] if spans else None, 'mode': mode})
    return rows, pages


def page_title(page: str) -> str:
    """Readable page title from the upstream ``/wiki/...`` identifier."""
    return clean(unquote(page.rsplit('/wiki/', 1)[-1]).replace('_', ' '))


def body_start(context: str, paragraphs: list[dict]) -> int:
    """Offset of the first paragraph in the context (after its tokenized page-title prefix).

    The first paragraph's ``title`` is not always the page title (sometimes a section),
    so the prefix is located by the first paragraph text instead.
    """
    first = clean(paragraphs[0]['text'])[:60] if paragraphs else ''
    position = context.find(first) if first else -1
    return max(position, 0)


def page_chunks(context: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Chunks of a page context body as ``(text, start_char, end_char)`` in the raw context."""
    spans = sentence_spans(context[offset:], offset)
    result = []
    for text, _, indices in chunk([part for part, _, _ in spans]):
        result.append((text, spans[indices[0]][1], spans[indices[-1]][2]))
    return result


def timeqa_episode(row: dict, split: str, title: str, chunks: list[tuple[str, int, int]],
                   writer: Writer, tokenizer, max_sources: int) -> bool:
    filters = writer.filters_for(split)
    answer = row['answer']
    lowered = answer.lower()
    meta = {'page': row['page']}
    if row['span'] is not None:
        start, end = row['span']
        hits = [i for i, (text, a, b) in enumerate(chunks)
                if a <= start and end <= b and lowered in text.lower()][:1]
        if not hits:
            filters.reject('answer_span_not_in_one_chunk')
            return False
        annotation = 'verified_span'
    else:
        hits = [i for i, (text, _, _) in enumerate(chunks) if lowered in text.lower()]
        if not hits:
            filters.reject('answer_not_in_context')
            return False
        if len(hits) > MAX_GOLD:
            filters.reject('answer_in_too_many_chunks')
            return False
        annotation = 'answer_match'
    rows = [source('timeqa', text, title=title, provenance={**meta, 'chunk_index': index})
            for index, (text, _, _) in enumerate(chunks)]
    gold = [rows[i] for i in hits]
    question_words = words(row['question'])
    ranked = sorted((-len(question_words & words(text)), index)
                    for index, (text, _, _) in enumerate(chunks)
                    if index not in hits and lowered not in text.lower())
    distractors = [rows[index] for _, index in ranked[:DISTRACTORS]]
    supports = gold + distractors
    new = {item['record_id'] for item in supports} - writer.sources.keys()
    if len(writer.sources) + len(new) > max_sources:
        filters.reject('source_budget')
        return False
    item = episode(
        domain='timeqa', split=split, identifier=f"{row['mode']}-{row['idx']}",
        question=row['question'], answer=answer, gold=gold, supports=supports, filters=filters,
        tokenizer=tokenizer, all_required=False, annotation=annotation,
        task_family='public_temporal_qa',
        provenance={'idx': row['idx'], 'mode': row['mode'], 'page': row['page'],
                    'distractors': len(distractors)})
    return writer.add(split, item, supports)


def build_timeqa(raw: Path, output: Path, *, train: int, validation: int, max_sources: int,
                 seed: int = 1701, tokenizer=None) -> dict:
    writer = Writer(output, 'timeqa')
    files = {'train': [('easy', raw / 'train.easy.json.gzip'), ('hard', raw / 'train.hard.json.gzip')],
             'validation': [('easy', raw / 'dev.easy.json'), ('hard', raw / 'dev.hard.json')]}
    train_pages: set[str] = set()
    modes = {}
    for split, wanted in (('train', train), ('validation', validation)):
        filters = writer.filters_for(split)
        by_mode, pages = {}, {}
        for mode, path in files[split]:
            rows, found = load_timeqa(path, mode, filters)
            for page, value in found.items():
                if pages.setdefault(page, value) != value:
                    raise ValueError(f'{page}: easy/hard contexts differ')
            if split == 'validation':
                held = [row for row in rows if row['page'] not in train_pages]
                for _ in range(len(rows) - len(held)):
                    filters.reject('page_in_train')
                rows = held
            rows.sort(key=lambda row: row['idx'])
            random.Random(f'{seed}:{split}:{mode}').shuffle(rows)
            by_mode[mode] = rows
        if split == 'train':
            train_pages = set(pages)
        cache: dict[str, list] = {}
        quotas = {'easy': wanted - wanted // 2, 'hard': wanted // 2}
        for mode, rows in by_mode.items():
            kept = 0
            for row in rows:
                if kept == quotas[mode]:
                    break
                title, context, offset = pages[row['page']]
                if row['page'] not in cache:
                    cache[row['page']] = page_chunks(context, offset)
                kept += timeqa_episode(row, split, title, cache[row['page']], writer,
                                       tokenizer, max_sources)
            modes[f'{split}/{mode}'] = kept
        del cache, pages, by_mode
    manifest = {
        'upstream': 'https://github.com/wenhuchen/Time-Sensitive-QA (BSD-3-Clause)',
        'split': {'train': 'upstream train.easy + train.hard',
                  'validation': 'upstream dev.easy + dev.hard, pages also in train removed',
                  'unused': 'upstream test and human-annotated files'},
        'episodes_by_mode': modes,
        'time': 'created_at=1, query_time=2 (the asked time is in the question text; no '
                'principled document date)',
        'gold': 'train: chunk containing the annotated answer span (verified_span); validation: '
                f'every chunk containing the answer string (answer_match, at most {MAX_GOLD}, '
                'any one suffices)',
        'distractors': (f'up to {DISTRACTORS} other chunks of the same page not containing the answer, '
                        'ranked by question word overlap (same entity, other time periods)'),
        'excluded': 'unanswerable and multi-target questions', 'seed': seed,
    }
    return writer.close(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=('streamingqa', 'timeqa', 'all'), default='all')
    parser.add_argument('--raw', type=Path, required=True, help='public2-raw-20260926 directory')
    parser.add_argument('--output', type=Path, required=True, help='public2-20260926 directory')
    parser.add_argument('--streamingqa-train', type=int, default=15000)
    parser.add_argument('--streamingqa-validation', type=int, default=500)
    parser.add_argument('--streamingqa-sources', type=int, default=40000)
    parser.add_argument('--cutoff', type=dt.date.fromisoformat, default=dt.date(2016, 4, 1))
    parser.add_argument('--end', type=dt.date.fromisoformat, default=dt.date(2017, 1, 1))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeqa-train', type=int, default=5000)
    parser.add_argument('--timeqa-validation', type=int, default=500)
    parser.add_argument('--timeqa-sources', type=int, default=15000)
    parser.add_argument('--no-tokenizer', action='store_true')
    args = parser.parse_args()
    tokenizer = None if args.no_tokenizer else load_tokenizer()
    summaries = {}
    if args.dataset in {'streamingqa', 'all'}:
        raw = args.raw / 'streamingqa'
        years = range(2007, args.end.year + (args.end.timetuple().tm_yday > 1))
        wmt = [raw / 'wmt' / f'news-docs.{year}.en.filtered.gz' for year in years]
        missing = [path for path in wmt if not path.exists()]
        if missing:
            raise FileNotFoundError(missing)
        summaries['streamingqa'] = build_streamingqa(
            raw, args.output / 'streamingqa', wmt_files=wmt,
            dedup_keys=raw / 'wmt' / 'wmt_sorting_key_ids.txt.gz',
            train=args.streamingqa_train, validation=args.streamingqa_validation,
            max_sources=args.streamingqa_sources, cutoff=args.cutoff, end=args.end,
            workers=args.workers, tokenizer=tokenizer)
    if args.dataset in {'timeqa', 'all'}:
        summaries['timeqa'] = build_timeqa(
            args.raw / 'timeqa', args.output / 'timeqa', train=args.timeqa_train,
            validation=args.timeqa_validation, max_sources=args.timeqa_sources,
            tokenizer=tokenizer)
    print(json.dumps({name: {key: value[key] for key in ('episodes', 'sources', 'filtered')}
                      for name, value in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
