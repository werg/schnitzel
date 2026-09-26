"""Held-out evaluation domains: LongMemEval S/M, LoCoMo, MultiHop-RAG and FRAMES.

These are **evaluation-only** sets (manifest ``role: heldout_evaluation_only``);
they are never training data. Every dataset becomes one domain with a single
``test`` split (plus ``test-abstention``/``test-adversarial``/``test-null``
side splits where the benchmark has unanswerable questions).

Time: a bank query reads a record only if ``created_at < query_time``. Both are
whole days since 2000-01-01 wherever the benchmark supplies dates, except
LongMemEval, whose sessions and questions carry minute timestamps and often fall
on the question's own day: there both are whole minutes since 2000-01-01.

* LongMemEval (cleaned release): every chat turn, or a small run of consecutive
  turns packed to ``MAX_CHARS`` with ``chunk``, is a passage dated by its
  session; the passage text starts with the session timestamp, and the query
  carries the question's current date. Gold is the upstream ``has_answer`` turn
  (narrowed to the best answer-overlapping piece when a long turn spans several
  passages). For knowledge-update questions the latest answer session is gold
  and earlier answer-session turns are hard distractors. Evidence that is not
  strictly older (in minutes) than the question is dropped and counted, never
  shifted. Each question has its own haystack, but all sampled haystacks share
  one domain namespace; the full haystack of every sampled question is in
  ``sources.jsonl`` and questions are sampled (stratified by type) until the
  source cap. Questions whose episode is filtered do not consume the cap.
* LoCoMo: same dialogue passages; the query time is one day after the
  conversation's last session. Gold passages hold the annotated dialogue IDs.
  Adversarial questions (category 5) go to ``test-adversarial`` with the
  benchmark's refusal target.
* MultiHop-RAG: the 609 news articles are chunked and dated by publication.
  Queries carry no date; they were posed against the finished corpus, so every
  query time is one day after the newest article (rather than the newest
  evidence, which would leak the evidence date through the time filter). Gold
  passages hold each evidence ``fact``. Null queries go to ``test-null`` with no
  sufficient group.
* FRAMES: the linked Wikipedia articles are fetched at the last revision before
  ``FRAMES_REVISION_DATE`` (``fetch-frames``: revision ID from the action API,
  that revision's HTML from the REST API; the first records came from
  ``action=parse``, which rate-limited us; revision IDs recorded). Articles
  carry no usable date, so ``created_at=1`` and ``query_time=2``. Gold is
  heuristic (``answer_match``): per needed article, the passage with the
  highest lexical overlap with the question and answer. Chunks per article are
  capped to the source budget; gold and high-overlap passages are kept first.

Distractors in episode supports (at most ``MAX_DISTRACTORS``) are lexical hard
negatives that are causally prior. Answers are supervised targets only, never
query text. Source text is inert data.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from public_corpus_common import (MAX_CHARS, PROMPT, Writer, answer_ok, chunk, clean,
                                  episode, load_tokenizer, source, split_sentences)

EPOCH = dt.date(2000, 1, 1)
ROLE = 'heldout_evaluation_only'
MAX_DISTRACTORS = 8
SEED = 1701
LME_SOURCE_CAP = 60_000
FRAMES_SOURCE_CAP = 40_000
FRAMES_REVISION_DATE = '2024-09-20T00:00:00Z'
USER_AGENT = ('SDKB-heldout-eval-prep/0.4 (offline research evaluation corpus; '
              'at most 3 concurrent requests, <=2 requests/s, honours Retry-After)')
TOKEN = re.compile(r'[a-z0-9]+')
STOPWORDS = frozenset(
    'a an the of in on at to for from by with and or is was were be been are did does do '
    'what which who whom whose when where why how that this these those it its as his her '
    'their he she they has have had not than then into about after before during i my me '
    'you your we our us can could would should will so if but there here all any some '
    'just also very really'.split())
LOCOMO_CATEGORIES = {1: 'multi_hop', 2: 'temporal', 3: 'open_domain', 4: 'single_hop',
                     5: 'adversarial'}
LOCOMO_REFUSAL = 'Not mentioned in the conversation'
SKIP_SECTIONS = frozenset({'references', 'external links', 'see also', 'notes', 'further reading',
                           'bibliography', 'sources', 'citations', 'footnotes',
                           'notes and references', 'works cited'})
SKIP_CLASSES = ('navbox', 'reflist', 'refbegin', 'mw-references-wrap', 'references',
                'mw-editsection', 'metadata', 'ambox', 'hatnote', 'shortdescription',
                'sidebar', 'toc', 'noprint', 'mw-empty-elt', 'reference', 'thumb',
                'gallery', 'mw-kartographer')


def day_number(date: dt.date) -> int:
    return (date - EPOCH).days


def minute_number(when: dt.datetime) -> int:
    return int((when - dt.datetime(2000, 1, 1)).total_seconds() // 60)


def terms(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]


def opaque(*parts: str) -> str:
    return hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:16]


def yes_no(answer: str) -> bool:
    return clean(answer).lower().rstrip('.') in {'yes', 'no'}


def add_sources(writer: Writer, rows) -> None:
    """Store bank passages that no episode needs to reference."""
    for row in rows:
        existing = writer.sources.setdefault(row['record_id'], row)
        if existing['created_at'] != row['created_at']:
            raise ValueError('One passage identity with two creation times')


class Lexical:
    """Small BM25 index over passages, used only to pick hard distractors."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, row in enumerate(rows):
            counts = Counter(terms(row['text']))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((index, count))
        self.average = sum(self.lengths) / max(len(self.lengths), 1)

    def scores(self, query: str) -> dict[int, float]:
        result: dict[int, float] = defaultdict(float)
        total = len(self.rows)
        for term in set(terms(query)):
            posting = self.postings.get(term, ())
            if not posting or len(posting) > total // 2 + 1:
                continue
            idf = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, count in posting:
                norm = count + 1.2 * (0.25 + 0.75 * self.lengths[index] / self.average)
                result[index] += idf * count * 2.2 / norm
        return result

    def top(self, query: str, count: int, *, exclude: set[str], before: int) -> list[dict]:
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        chosen = []
        for index, _ in ranked:
            row = self.rows[index]
            if row['record_id'] in exclude or row['created_at'] >= before:
                continue
            chosen.append(row)
            exclude = exclude | {row['record_id']}
            if len(chosen) == count:
                break
        return chosen


def overlap(text: str, wanted: Counter) -> float:
    have = set(terms(text))
    return sum(weight for term, weight in wanted.items() if term in have)


def best_rows(rows: list[dict], question: str, answer: str) -> list[dict]:
    """Rows with the maximal answer-weighted term overlap (all rows if none overlaps)."""
    wanted = Counter(terms(question))
    for term in terms(answer):
        wanted[term] += 3
    scored = [(overlap(row['text'], wanted), row) for row in rows]
    top = max((score for score, _ in scored), default=0)
    return [row for score, row in scored if score == top] if top > 0 else list(rows)


def dialogue_passages(turns: list[tuple[str, str]], prefix: str) -> list[tuple[str, list[int]]]:
    """Pack speaker-labelled turns into passages of ``prefix`` plus at most the limit.

    A turn is split at sentence boundaries only when it alone exceeds the limit;
    every piece keeps the speaker label. Returns ``(text, turn_indices)``.
    """
    limit = MAX_CHARS - len(prefix)
    pieces, owners = [], []
    for index, (label, body) in enumerate(turns):
        sentences = split_sentences(body)
        if not sentences:
            continue
        for piece, _, _ in chunk(sentences, limit=limit - len(label)):
            pieces.append(label + piece)
            owners.append(index)
    return [(prefix + text, sorted({owners[i] for i in indices}))
            for text, _, indices in chunk(pieces, limit=limit)]


def unanswerable_episode(*, domain: str, split: str, identifier: str, question: str,
                         answer: str, supports: list[dict], query_time: int, annotation: str,
                         provenance: dict, tokenizer=None, filters=None) -> dict | None:
    """Episode whose correct response is a refusal: no sufficient group exists.

    Mirrors ``episode``'s schema; ``required_ids`` and ``sufficient_groups`` are empty.
    """
    query = PROMPT + clean(question)
    if tokenizer is not None and len(tokenizer.encode(query, add_special_tokens=False)) > 120:
        filters.reject('query_tokens')
        return None
    if any(row['created_at'] >= query_time or row['domain'] != domain for row in supports):
        raise ValueError(f'{identifier}: supports must be causally prior and in-domain')
    return {
        'episode_id': f'{domain}-{identifier}', 'environment': f'{domain}-{split}',
        'query': query, 'answer': clean(answer), 'query_time': int(query_time),
        'required_ids': [], 'sufficient_groups': [], 'support_annotation': annotation,
        'task_family': 'public_qa',
        'supports': [{'record_id': row['record_id'], 'text': row['text'],
                      'created_at': row['created_at'], 'kind': 'passage'} for row in supports],
        'provenance': {'dataset': domain, 'domain': domain, 'split': split, **provenance},
    }


# ---------------------------------------------------------------- LongMemEval

def iter_json_array(path: Path, block: int = 1 << 22):
    """Yield the elements of a top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        buffer = handle.read(block).lstrip()
        if not buffer.startswith('['):
            raise ValueError(f'{path} is not a JSON array')
        buffer, done = buffer[1:], False
        while True:
            buffer = buffer.lstrip().lstrip(',').lstrip()
            if buffer.startswith(']'):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if done:
                    raise
                more = handle.read(block)
                done = not more
                buffer += more
                continue
            yield item
            buffer = buffer[end:]


def lme_time(text: str) -> dt.datetime:
    return dt.datetime.strptime(text.strip(), '%Y/%m/%d (%a) %H:%M')


def lme_passages(domain: str, question: dict) -> list[tuple[dict, int, list[int]]]:
    """All haystack passages of one question as ``(row, session_index, turn_indices)``."""
    result = []
    for index, (session_id, date, turns) in enumerate(zip(
            question['haystack_session_ids'], question['haystack_dates'],
            question['haystack_sessions'])):
        minute = minute_number(lme_time(date))
        key = opaque(question['question_id'], session_id, date)
        labelled = [(f"{turn['role']}: ", turn['content']) for turn in turns]
        for text, owners in dialogue_passages(labelled, f'Chat session on {date}. '):
            row = source(domain, text, created_at=minute, provenance={
                'article_title': f'{domain}/session-{key}', 'session_id': session_id,
                'session_date': date})
            result.append((row, index, owners))
    return result


def lme_episode(domain: str, question: dict, passages, writer: Writer, tokenizer, stats: Counter):
    qid, kind = question['question_id'], question['question_type']
    split = 'test-abstention' if qid.endswith('_abs') else 'test'
    filters = writer.filters_for(split)
    query_time = minute_number(lme_time(question['question_date']))
    answer = str(question['answer'])
    answer_sessions = set(question['answer_session_ids'])
    by_session: dict[int, list] = defaultdict(list)
    for row, session, owners in passages:
        by_session[session].append((row, owners))
    evidence = []  # (session time, session index, [rows]) per has_answer session
    heuristic = False
    for index, (session_id, date, turns) in enumerate(zip(
            question['haystack_session_ids'], question['haystack_dates'],
            question['haystack_sessions'])):
        if session_id not in answer_sessions:
            continue
        marked = [turn_index for turn_index, turn in enumerate(turns) if turn.get('has_answer')]
        rows = []
        if marked:
            for turn_index in marked:
                holding = [row for row, owners in by_session[index] if turn_index in owners]
                if len(holding) > 1:
                    heuristic = True
                    holding = best_rows(holding, question['question'], answer)
                rows.extend(holding)
        else:
            heuristic = True
            stats['answer_session_without_has_answer'] += 1
            rows = best_rows([row for row, _ in by_session[index]], question['question'], answer)[:2]
        if rows:
            evidence.append((lme_time(date), index, rows))
    if not evidence:
        filters.reject('no_gold')
        return None
    superseded = []
    if kind == 'knowledge-update' and len(evidence) > 1:
        evidence.sort(key=lambda item: item[0])
        superseded = [row for _, _, rows in evidence[:-1] for row in rows]
        evidence = evidence[-1:]
    gold = list({row['record_id']: row for _, _, rows in evidence for row in rows}.values())
    if any(row['created_at'] >= query_time for row in gold):
        filters.reject('evidence_not_before_query')
        return None
    gold_ids = {row['record_id'] for row in gold}
    hard = [row for row in {r['record_id']: r for r in superseded}.values()
            if row['record_id'] not in gold_ids and row['created_at'] < query_time]
    hard = hard[:MAX_DISTRACTORS]
    index = Lexical([row for row, _, _ in passages])
    hard += index.top(question['question'], MAX_DISTRACTORS - len(hard),
                      exclude=gold_ids | {row['record_id'] for row in hard}, before=query_time)
    # The answer check uses the upstream question: the added date must not reject
    # numeric answers that happen to occur in the timestamp.
    if not answer_ok(answer, question['question'], filters, tokenizer,
                     allow_in_query=yes_no(answer)):
        return split, None
    item = episode(
        domain=domain, split=split, identifier=qid,
        question=f"Current date: {question['question_date']}. {question['question']}",
        answer=answer, gold=gold, supports=gold + hard, filters=filters, tokenizer=tokenizer,
        query_time=query_time, all_required=True,
        annotation='answer_match' if heuristic else 'verified',
        task_family='long_term_chat_memory', allow_answer_in_query=True,
        provenance={'question_id': qid, 'question_type': kind,
                    'abstention': qid.endswith('_abs'),
                    'answer_session_ids': sorted(answer_sessions),
                    'gold_heuristic': heuristic,
                    'superseded_ids': [row['record_id'] for row in superseded],
                    'question_date': question['question_date'],
                    'haystack_sessions': len(question['haystack_session_ids'])})
    return split, item


def lme_order(path: Path, seed: int) -> list[tuple[str, int]]:
    """Question IDs in a stratified round-robin order over types, with text sizes."""
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for question in iter_json_array(path):
        size = sum(len(turn['content']) for session in question['haystack_sessions']
                   for turn in session)
        groups[question['question_type']].append((question['question_id'], size))
    rng = random.Random(f'{seed}:longmemeval')
    for kind in sorted(groups):
        groups[kind].sort()
        rng.shuffle(groups[kind])
    order, depth = [], 0
    while any(depth < len(rows) for rows in groups.values()):
        order.extend(groups[kind][depth] for kind in sorted(groups) if depth < len(groups[kind]))
        depth += 1
    return order


def build_longmemeval(path: Path, output: Path, domain: str, *, tokenizer, max_sources: int,
                      max_questions: int, revision: str, seed: int = SEED) -> dict:
    order = lme_order(path, seed)
    # Passages hold at most MAX_CHARS; oversample candidates by that estimate so
    # questions whose episode is filtered do not consume the source budget.
    estimate, candidates = 0, set()
    for qid, size in order:
        if len(candidates) >= 3 * max_questions or estimate > 3 * max_sources:
            break
        candidates.add(qid)
        estimate += size / MAX_CHARS
    built = {}
    for question in iter_json_array(path):
        if question['question_id'] in candidates:
            built[question['question_id']] = (question, lme_passages(domain, question))
    writer, stats = Writer(output, domain), Counter()
    selected, total = [], set()
    for qid, _ in order:
        if qid not in built or len(selected) >= max_questions:
            continue
        question, passages = built[qid]
        stats['examined_questions'] += 1
        result = lme_episode(domain, question, passages, writer, tokenizer, stats)
        if result is None or result[1] is None:
            continue
        ids = {row['record_id'] for row, _, _ in passages}
        if len(total | ids) > max_sources:
            stats['stopped_at_source_cap'] = 1
            break
        selected.append(qid)
        total |= ids
        add_sources(writer, (row for row, _, _ in passages))
        stats['haystack_passage_slots'] += len(passages)
        query_time = minute_number(lme_time(question['question_date']))
        stats['haystack_passages_not_before_query'] += sum(
            row['created_at'] >= query_time for row, _, _ in passages)
        split, item = result
        writer.add(split, item, [])
    stats['selected_questions'] = len(selected)
    owners = Counter()
    for qid in selected:
        for record in {row['record_id'] for row, _, _ in built[qid][1]}:
            owners[record] += 1
    stats['passages_shared_across_questions'] = sum(count > 1 for count in owners.values())
    stats['passages_unique_to_one_question'] = sum(count == 1 for count in owners.values())
    stats['upstream_questions'] = len(order)
    return writer.close({
        'role': ROLE, 'source_dataset': 'xiaowu0162/longmemeval-cleaned',
        'source_file': path.name, 'source_revision': revision, 'license': 'MIT',
        'time_unit': 'minutes since 2000-01-01T00:00 (session / question timestamp)',
        'selection': {'order': 'stratified round-robin over question_type',
                      'seed': seed, 'max_sources': max_sources,
                      'max_questions': max_questions, 'selected_question_ids': selected},
        'stats': dict(stats),
        'notice': ('Evaluation only. Each upstream question has its own haystack; all sampled '
                   'haystacks share this one domain namespace, so a query can also read '
                   'causally prior passages of other questions\' haystacks. Passage text '
                   'starts with the session timestamp, which differs per question, so reused '
                   'upstream sessions become distinct passages. Gold is the upstream '
                   'has_answer turn; gold_heuristic marks narrowing of long turns or '
                   'session-level fallback. Knowledge-update gold is the latest answer '
                   'session; superseded turns are hard distractors. Long answers (>24 words, '
                   'e.g. preference rubrics or abstention explanations) are filtered by the '
                   'shared answer_ok and counted.'),
    })


# ---------------------------------------------------------------- LoCoMo

def locomo_time(text: str) -> dt.datetime:
    return dt.datetime.strptime(clean(text), '%I:%M %p on %d %B, %Y')


def locomo_turn_text(turn: dict) -> str:
    text = turn.get('text') or ''
    caption = clean(turn.get('blip_caption') or '')
    return f'{text} [shares a photo: {caption}]' if caption else text


def evidence_ids(values) -> tuple[list[str], int]:
    found, bad = [], 0
    for value in values:
        matches = re.findall(r'D:?(\d+):(\d+)', value)
        if not matches:
            bad += 1
        found.extend(f'D{session}:{turn}' for session, turn in matches)
    return list(dict.fromkeys(found)), bad


def build_locomo(path: Path, output: Path, *, tokenizer, revision: str,
                 domain: str = 'locomo') -> dict:
    samples = json.loads(path.read_text(encoding='utf-8'))
    writer, stats = Writer(output, domain), Counter()
    conversations = []
    for sample in samples:
        conversation = sample['conversation']
        speakers = f"{conversation['speaker_a']} and {conversation['speaker_b']}"
        by_dialogue, rows, last = {}, [], None
        number = 1
        while f'session_{number}' in conversation:
            turns = conversation[f'session_{number}']
            stamp = conversation[f'session_{number}_date_time']
            when = locomo_time(stamp)
            last = max(last, when) if last else when
            prefix = f'Conversation of {speakers}, {clean(stamp)}. '
            labelled = [(f"{turn['speaker']}: ", locomo_turn_text(turn)) for turn in turns]
            key = opaque(sample['sample_id'], str(number))
            for text, owners in dialogue_passages(labelled, prefix):
                row = source(domain, text, created_at=day_number(when.date()), provenance={
                    'article_title': f'{domain}/{sample["sample_id"]}-session-{key}',
                    'sample_id': sample['sample_id'], 'session': number,
                    'dialogue_ids': [turns[i]['dia_id'] for i in owners]})
                rows.append(row)
                for owner in owners:
                    by_dialogue.setdefault(turns[owner]['dia_id'], []).append(row)
            number += 1
        stats['dated_sessions_without_turns'] += sum(
            1 for key in conversation if key.endswith('_date_time')
            and key[:-len('_date_time')] not in conversation)
        add_sources(writer, rows)
        conversations.append((sample, rows, by_dialogue, day_number(last.date()) + 1))
    index = Lexical(list(writer.sources.values()))
    for sample, rows, by_dialogue, query_time in conversations:
        for number, qa in enumerate(sample['qa']):
            category = qa['category']
            adversarial = category == 5
            split = 'test-adversarial' if adversarial else 'test'
            filters = writer.filters_for(split)
            wanted, bad = evidence_ids(qa.get('evidence', []))
            stats['malformed_evidence_ids'] += bad
            missing = [key for key in wanted if key not in by_dialogue]
            if missing:
                stats['evidence_id_not_found'] += len(missing)
                wanted = [key for key in wanted if key in by_dialogue]
            if not wanted:
                filters.reject('no_gold')
                continue
            if adversarial:
                answer = str(qa.get('answer') or LOCOMO_REFUSAL)
            else:
                answer = str(qa['answer'])
            gold = []
            for key in wanted:
                holding = by_dialogue[key]
                if len(holding) > 1:
                    holding = best_rows(holding, qa['question'], answer)
                gold.extend(holding)
            gold = list({row['record_id']: row for row in gold}.values())
            gold_ids = {row['record_id'] for row in gold}
            distractors = index.top(qa['question'], MAX_DISTRACTORS, exclude=gold_ids,
                                    before=query_time)
            identifier = f"{sample['sample_id']}-q{number}"
            provenance = {'sample_id': sample['sample_id'], 'qa_index': number,
                          'category': category, 'category_name': LOCOMO_CATEGORIES.get(category),
                          'evidence': wanted}
            if adversarial:
                provenance['adversarial_answer'] = str(qa.get('adversarial_answer'))
                provenance['evidence_role'] = 'related turn the question distorts'
            item = episode(
                domain=domain, split=split, identifier=identifier, question=qa['question'],
                answer=answer, gold=gold, supports=gold + distractors, filters=filters,
                tokenizer=tokenizer, query_time=query_time, all_required=True,
                annotation='adversarial_related' if adversarial else 'verified',
                task_family='long_term_chat_memory', allow_answer_in_query=yes_no(answer),
                provenance=provenance)
            writer.add(split, item, gold + distractors)
    stats['conversations'] = len(samples)
    return writer.close({
        'role': ROLE, 'source_dataset': 'snap-research/locomo data/locomo10.json',
        'source_revision': revision, 'license': 'CC-BY-NC-4.0',
        'time_unit': 'days since 2000-01-01 (session date); query = last session + 1 day',
        'stats': dict(stats),
        'notice': ('Evaluation only (non-commercial license). The ten conversations share one '
                   'domain namespace; a query can also read other conversations\' passages. '
                   'Image turns are represented by their BLIP caption. Adversarial '
                   '(category 5) questions are in test-adversarial: the target is the '
                   'benchmark refusal (or its given answer) and their evidence is the related '
                   'turn the question distorts, not a sufficient support.'),
    })


# ---------------------------------------------------------------- MultiHop-RAG

def published_day(text: str) -> int:
    return day_number(dt.datetime.fromisoformat(text.replace('Z', '+00:00')).date())


def fact_sentences(sentences: list[str], fact: str) -> set[int]:
    """Sentence indices covering ``fact`` in the cleaned body; best overlap otherwise."""
    body = ' '.join(sentences)
    fact = clean(fact)
    start = body.find(fact) if fact else -1
    if start < 0 and fact:
        start = body.lower().find(fact.lower())
    if start >= 0:
        end, offset, found = start + len(fact), 0, set()
        for index, sentence in enumerate(sentences):
            if offset < end and offset + len(sentence) > start:
                found.add(index)
            offset += len(sentence) + 1
        return found
    wanted = set(terms(fact))
    if not wanted:
        return set()
    scored = [(len(wanted & set(terms(sentence))) / len(wanted), index)
              for index, sentence in enumerate(sentences)]
    score, index = max(scored, default=(0, -1))
    return {index} if score >= 0.6 else set()


def build_multihop_rag(raw: Path, output: Path, *, tokenizer, revision: str,
                       domain: str = 'multihop_rag') -> dict:
    corpus = json.loads((raw / 'corpus.json').read_text(encoding='utf-8'))
    queries = json.loads((raw / 'MultiHopRAG.json').read_text(encoding='utf-8'))
    writer, stats = Writer(output, domain), Counter()
    articles = {}
    for article in corpus:
        sentences = split_sentences(article['body'])
        day = published_day(article['published_at'])
        rows = [source(domain, text, title=article['title'], created_at=day, provenance={
            'url': article['url'], 'publisher': article['source'],
            'category': article['category'], 'published_at': article['published_at'],
            'chunk_index': number, 'sentence_indices': indices})
            for number, (text, _, indices) in enumerate(chunk(sentences))]
        articles[article['url']] = (sentences, rows)
        add_sources(writer, rows)
    query_time = max(row['created_at'] for row in writer.sources.values()) + 1
    index = Lexical(list(writer.sources.values()))
    for number, query in enumerate(queries):
        kind = query['question_type']
        identifier = f'q{number:04d}'
        provenance = {'upstream_index': number, 'question_type': kind}
        if kind == 'null_query' or not query['evidence_list']:
            split = 'test-null'
            item = unanswerable_episode(
                domain=domain, split=split, identifier=identifier, question=query['query'],
                answer=query['answer'], supports=index.top(query['query'], MAX_DISTRACTORS,
                                                           exclude=set(), before=query_time),
                query_time=query_time, annotation='no_sufficient_support',
                provenance=provenance, tokenizer=tokenizer, filters=writer.filters_for(split))
            writer.add(split, item, [])
            continue
        split, gold, lost = 'test', [], 0
        for evidence in query['evidence_list']:
            if evidence['url'] not in articles:
                lost += 1
                continue
            sentences, rows = articles[evidence['url']]
            wanted = fact_sentences(sentences, evidence['fact'])
            hits = [row for row in rows if wanted & set(row['provenance']['sentence_indices'])]
            if not hits:
                lost += 1
                continue
            gold.extend(hits)
        if lost:
            stats['evidence_fact_not_located'] += lost
            writer.filters_for(split).reject('evidence_fact_not_located')
            continue
        gold = list({row['record_id']: row for row in gold}.values())
        distractors = index.top(query['query'], MAX_DISTRACTORS,
                                exclude={row['record_id'] for row in gold}, before=query_time)
        provenance['evidence_urls'] = list(dict.fromkeys(e['url'] for e in query['evidence_list']))
        item = episode(domain=domain, split=split, identifier=identifier,
                       question=query['query'], answer=str(query['answer']), gold=gold,
                       supports=gold + distractors, filters=writer.filters_for(split),
                       tokenizer=tokenizer, query_time=query_time, all_required=True,
                       annotation='verified', task_family='multi_document_qa',
                       allow_answer_in_query=yes_no(str(query['answer'])), provenance=provenance)
        writer.add(split, item, gold + distractors)
    stats['articles'] = len(corpus)
    return writer.close({
        'role': ROLE, 'source_dataset': 'yixuantt/MultiHopRAG', 'source_revision': revision,
        'license': 'ODC-BY', 'query_time': query_time,
        'time_unit': 'days since 2000-01-01 (article publication date)',
        'stats': dict(stats),
        'notice': ('Evaluation only. Queries have no date upstream; they were posed against the '
                   'complete corpus, so every query_time is one day after the newest article. '
                   'Gold passages contain the annotated evidence facts (exact match, else a '
                   'sentence with >=60% fact-term overlap). Null queries are in test-null with '
                   'no sufficient group; their supports are lexical distractors.'),
    })


# ---------------------------------------------------------------- FRAMES

class WikiText(HTMLParser):
    """Visible article text as lines; tables become ``a | b | c`` rows."""

    VOID = frozenset({'br', 'img', 'hr', 'meta', 'link', 'input', 'wbr', 'col', 'source',
                      'area', 'base', 'embed', 'param', 'track'})
    BLOCK = frozenset({'p', 'li', 'tr', 'dd', 'dt', 'caption', 'blockquote', 'h1', 'h2', 'h3',
                       'h4', 'h5', 'h6', 'div', 'table', 'ul', 'ol', 'dl', 'figcaption'})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.lines: list[tuple[str, str]] = []
        self.current: list[str] = []
        self.skip_section = False
        self.heading: list[str] | None = None

    def skipping(self) -> bool:
        return self.skip_section or any(flag for _, flag in self.stack)

    def flush(self, kind: str = 'text') -> None:
        text = clean(''.join(self.current))
        self.current = []
        if text and text != '|':
            self.lines.append((kind, text.strip(' |')))

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            if tag == 'br':
                self.current.append(' ')
            return
        attrs = dict(attrs)
        classes = attrs.get('class') or ''
        flag = (tag in {'style', 'script', 'sup'}
                or any(name in classes.split() for name in SKIP_CLASSES)
                or 'display:none' in (attrs.get('style') or '').replace(' ', ''))
        if tag in self.BLOCK and not self.skipping():
            self.flush('row' if tag == 'tr' else 'text')
        if tag == 'h2':
            self.heading = []
        if tag in {'td', 'th'}:
            self.current.append(' | ')
        self.stack.append((tag, flag))

    def handle_endtag(self, tag):
        if tag in self.VOID or not any(name == tag for name, _ in self.stack):
            return
        while self.stack:
            name, _ = self.stack.pop()
            if name == tag:
                break
        if tag == 'h2' and self.heading is not None:
            title = clean(''.join(self.heading)).lower()
            self.heading = None
            self.skip_section = title in SKIP_SECTIONS
            self.current = []
            if not self.skip_section:
                self.lines.append(('heading', title))
            return
        if tag in self.BLOCK and not self.skipping():
            self.flush('row' if tag == 'tr' else 'text')

    def handle_data(self, data):
        if self.heading is not None and not any(flag for _, flag in self.stack):
            self.heading.append(data)
            return
        if not self.skipping():
            self.current.append(data)


def wiki_units(html: str) -> list[str]:
    parser = WikiText()
    parser.feed(html)
    parser.close()
    parser.flush()
    units = []
    for kind, text in parser.lines:
        if kind == 'heading':
            continue
        units.extend([text] if kind == 'row' else split_sentences(text))
    return [unit for unit in units if len(unit) > 1]


def frames_link(url: str) -> tuple[str, str, str] | None:
    """(api host, title, fragment) for a Wikipedia article link."""
    if '://' not in url:
        url = 'https://' + url.lstrip('/')
    parts = urllib.parse.urlparse(url)
    host = parts.netloc.replace('.m.wikipedia.org', '.wikipedia.org')
    if host == 'w.wiki' or not host.endswith('wikipedia.org') or not parts.path.startswith('/wiki/'):
        return None
    title = urllib.parse.unquote(parts.path[len('/wiki/'):]).replace('_', ' ')
    return host, title, urllib.parse.unquote(parts.fragment)


def resolve_short(url: str) -> str:
    request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.geturl()


class RateLimit:
    """Start at most ``rate`` requests per second across threads."""

    def __init__(self, rate: float):
        self.interval, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(max(start - now, 0.0))

    def pause(self, seconds: float) -> None:
        """Hold every thread back after the server asks us to slow down."""
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)


def api_get(host: str, params: dict, limit: RateLimit) -> dict:
    query = urllib.parse.urlencode({**params, 'format': 'json', 'formatversion': 2})
    return json.loads(http_get(f'https://{host}/w/api.php?{query}', limit).decode('utf-8'))


def http_get(url: str, limit: RateLimit) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(8):
        limit.wait()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retried then re-raised
            if attempt == 7:
                raise
            delay = 5 * (attempt + 1)
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                retry = error.headers.get('Retry-After') or ''
                delay = int(retry) if retry.isdigit() else 30 * (attempt + 1)
                limit.pause(delay)
            time.sleep(delay)
    raise AssertionError('unreachable')


def fetch_article(url: str, limit: RateLimit) -> dict:
    record = {'url': url, 'revision_date': FRAMES_REVISION_DATE}
    try:
        if urllib.parse.urlparse(url).netloc == 'w.wiki':
            limit.wait()
            target = resolve_short(url)
        else:
            target = url
        parsed = frames_link(target)
        if parsed is None:
            raise ValueError(f'not an article link: {target}')
        host, title, fragment = parsed
        info = api_get(host, {'action': 'query', 'titles': title, 'redirects': 1,
                              'prop': 'revisions', 'rvprop': 'ids|timestamp', 'rvlimit': 1,
                              'rvstart': FRAMES_REVISION_DATE, 'rvdir': 'older'}, limit)
        page = info['query']['pages'][0]
        revision = page['revisions'][0]
        path = urllib.parse.quote(page['title'].replace(' ', '_'), safe='')
        html = http_get(f"https://{host}/api/rest_v1/page/html/{path}/{revision['revid']}",
                        limit).decode('utf-8')
        record.update(host=host, requested_title=title, fragment=fragment,
                      title=page['title'], pageid=page['pageid'], revid=revision['revid'],
                      timestamp=revision['timestamp'], renderer='rest_v1/page/html',
                      html=html)
    except Exception as error:  # noqa: BLE001 - recorded per article and counted
        record['error'] = f'{type(error).__name__}: {error}'
    return record


def fetch_frames(raw: Path, *, rate: float = 2.0, workers: int = 3,
                 limit: int | None = None) -> dict:
    """Fetch every linked article at its last revision before FRAMES_REVISION_DATE.

    Old-revision renders are slow server-side, so three requests may overlap, but
    no more than ``rate`` requests start per second, and HTTP 429 pauses all
    workers (Retry-After honoured). Appends gzip members to
    ``wikipedia/articles.jsonl.gz`` and resumes by URL (failed URLs are retried).
    """
    rows = list(csv.DictReader((raw / 'test.tsv').open(encoding='utf-8'), delimiter='\t'))
    urls = list(dict.fromkeys(url for row in rows for url in ast.literal_eval(row['wiki_links'])))
    folder = raw / 'wikipedia'
    folder.mkdir(exist_ok=True)
    path = folder / 'articles.jsonl.gz'
    done = set()
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if 'error' not in record:
                    done.add(record['url'])
    counts = Counter(already=len(done))
    pending = [url for url in urls[:limit] if url not in done]
    throttle = RateLimit(rate)
    with ThreadPoolExecutor(workers) as pool:
        for record in pool.map(lambda url: fetch_article(url, throttle), pending):
            counts['failed' if 'error' in record else 'fetched'] += 1
            with gzip.open(path, 'at', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return dict(counts)


def frames_units(raw: Path) -> dict[str, dict]:
    """URL -> {'title', 'revid', 'units', ...}; last record per URL wins."""
    articles = {}
    with gzip.open(raw / 'wikipedia' / 'articles.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if 'error' in record:
                articles.pop(record['url'], None)
                continue
            html = record.pop('html')
            record['units'] = wiki_units(html)
            articles[record['url']] = record
    return articles


def build_frames(raw: Path, output: Path, *, tokenizer, revision: str,
                 max_sources: int = FRAMES_SOURCE_CAP, domain: str = 'frames',
                 articles: dict[str, dict] | None = None) -> dict:
    rows = list(csv.DictReader((raw / 'test.tsv').open(encoding='utf-8'), delimiter='\t'))
    articles = frames_units(raw) if articles is None else articles
    writer, stats = Writer(output, domain), Counter()
    # One passage set per article revision (several URLs may resolve to one page).
    pages: dict[tuple, list[dict]] = {}
    page_of = {}
    for url, record in articles.items():
        key = (record.get('host', ''), record['pageid'], record['revid'])
        page_of[url] = key
        if key not in pages:
            pages[key] = [source(domain, text, title=record['title'], created_at=1, provenance={
                'wikipedia_title': record['title'], 'pageid': record['pageid'],
                'revid': record['revid'], 'revision_timestamp': record['timestamp'],
                'chunk_index': number})
                for number, (text, _, _) in enumerate(chunk(record['units']))]
    stats['articles'] = len(pages)
    stats['article_chunks_before_cap'] = sum(len(rows_) for rows_ in pages.values())
    # Heuristic gold: per question and needed article, the best-overlap passage.
    plans = []
    keep: dict[tuple, set[int]] = defaultdict(set)
    for number, row in enumerate(rows):
        links = ast.literal_eval(row['wiki_links'])
        needed = list(dict.fromkeys(page_of[url] for url in links if url in page_of))
        missing = [url for url in links if url not in page_of]
        gold = []
        for key in needed:
            best = best_rows(pages[key], row['Prompt'], row['Answer'])[0]
            keep[key].add(best['provenance']['chunk_index'])
            gold.append(best)
        plans.append((number, row, needed, missing, gold))
    # Budget: every gold passage, then per-article lead/high-overlap passages.
    budget = max_sources - sum(len(indices) for indices in keep.values())
    per_article = max(budget // max(len(pages), 1), 0)
    for key, passages in pages.items():
        extra = [row['provenance']['chunk_index'] for row in passages
                 if row['provenance']['chunk_index'] not in keep[key]][:per_article]
        keep[key].update(extra)
    remaining = max_sources - sum(len(indices) for indices in keep.values())
    for key, passages in sorted(pages.items(), key=lambda item: str(item[0])):
        if remaining <= 0:
            break
        spare = [row['provenance']['chunk_index'] for row in passages
                 if row['provenance']['chunk_index'] not in keep[key]][:remaining]
        keep[key].update(spare)
        remaining -= len(spare)
    for key, passages in pages.items():
        chosen = [row for row in passages if row['provenance']['chunk_index'] in keep[key]]
        stats['article_chunks_dropped_by_cap'] += len(passages) - len(chosen)
        stats['articles_truncated'] += len(chosen) < len(passages)
        add_sources(writer, chosen)
    index = Lexical(list(writer.sources.values()))
    for number, row, needed, missing, gold in plans:
        split, filters = 'test', writer.filters_for('test')
        if missing:
            stats['questions_with_unfetched_links'] += 1
            filters.reject('unfetched_article')
            continue
        gold = list({item['record_id']: item for item in gold}.values())
        distractors = index.top(row['Prompt'], MAX_DISTRACTORS,
                                exclude={item['record_id'] for item in gold}, before=2)
        item = episode(domain=domain, split=split, identifier=f'q{number:03d}',
                       question=row['Prompt'], answer=row['Answer'], gold=gold,
                       supports=gold + distractors, filters=filters, tokenizer=tokenizer,
                       query_time=2, all_required=True, annotation='answer_match',
                       task_family='multi_document_qa',
                       allow_answer_in_query=yes_no(row['Answer']),
                       provenance={'upstream_index': number,
                                   'reasoning_types': row['reasoning_types'],
                                   'wiki_links': ast.literal_eval(row['wiki_links']),
                                   'gold_heuristic': 'max question+answer term overlap per article'})
        writer.add(split, item, gold + distractors)
    return writer.close({
        'role': ROLE, 'source_dataset': 'google/frames-benchmark', 'source_revision': revision,
        'license': 'Apache-2.0 (questions); Wikipedia text CC BY-SA 4.0',
        'wikipedia_revision_rule': f'last revision at or before {FRAMES_REVISION_DATE}',
        'wikipedia_revisions': {record['title']: record['revid'] for record in articles.values()},
        'max_sources': max_sources, 'chunks_per_article_floor': per_article,
        'time_unit': 'none (created_at=1, query_time=2)', 'stats': dict(stats),
        'notice': ('Evaluation only. Gold is heuristic (answer_match): for each linked article the '
                   'single passage with the highest question+answer term overlap; the true '
                   'supporting fact may be elsewhere or in a table. Articles were chunked from '
                   'parsed HTML (tables as "a | b" rows; references/navigation removed) and '
                   'capped to the source budget, keeping gold passages first then the article '
                   'lead.'),
    })


# ---------------------------------------------------------------- MemoryAgentBench

def fact_list(context: str) -> list[tuple[int, str]]:
    return [(int(number), clean(text))
            for number, text in re.findall(r'^(\d+)\. (.+)$', context, re.M)]


def build_memoryagentbench_cr(path: Path, output: Path, *, tokenizer, revision: str,
                              context: str = 'factconsolidation_sh_32k',
                              domain: str = 'memoryagentbench_cr') -> dict:
    """FactConsolidation single-hop: numbered facts where later facts override earlier.

    Every fact is its own passage (text as upstream, with its number) created at
    its position; queries come after the last fact. Gold is heuristic: the
    latest fact containing the answer among the facts with the highest overlap
    with the question's subject terms; earlier conflicting facts about the same
    subject are hard distractors.
    """
    import pyarrow.parquet as parquet
    table = parquet.read_table(path).to_pylist()
    (row,) = [item for item in table if item['metadata']['source'] == context]
    facts = fact_list(row['context'])
    if [number for number, _ in facts] != list(range(len(facts))):
        raise ValueError('Facts are not numbered consecutively from zero')
    writer, stats = Writer(output, domain), Counter()
    rows = [source(domain, f'{number}. {text}', created_at=number + 1,
                   provenance={'context': context, 'fact_index': number})
            for number, text in facts]
    add_sources(writer, rows)
    query_time = len(rows) + 1
    fact_terms = [set(terms(text)) for _, text in facts]
    lexical = Lexical(rows)
    split, filters = 'test', writer.filters_for('test')
    for number, (question, answers) in enumerate(zip(row['questions'], row['answers'])):
        answer = str(answers[0])
        pattern = re.compile(r'(?<!\w)' + re.escape(answer.lower()) + r'(?!\w)')
        subject = set(terms(question)) - set(terms(answer))
        scores = [len(subject & have) for have in fact_terms]
        top = max(scores)
        tied = [index for index, score in enumerate(scores) if score == top and top > 0]
        holding = [index for index in tied if pattern.search(facts[index][1].lower())]
        if not holding:
            stats['answer_fact_not_found'] += 1
            filters.reject('no_gold')
            continue
        gold_index = holding[-1]
        later = [index for index in tied if index > gold_index]
        if later:
            stats['later_conflicting_fact'] += 1
            filters.reject('gold_not_latest')
            continue
        superseded = [rows[index] for index in tied if index < gold_index][-MAX_DISTRACTORS:]
        extra = lexical.top(question, MAX_DISTRACTORS - len(superseded),
                          exclude={rows[i]['record_id'] for i in [gold_index, *tied]},
                          before=query_time)
        item = episode(domain=domain, split=split, identifier=f'{context}-q{number:03d}',
                       question=question, answer=answer, gold=[rows[gold_index]],
                       supports=[rows[gold_index], *superseded, *extra], filters=filters,
                       tokenizer=tokenizer, query_time=query_time, annotation='answer_match',
                       task_family='conflict_resolution',
                       provenance={'context': context, 'qa_pair_id':
                                   row['metadata']['qa_pair_ids'][number],
                                   'answers': [str(value) for value in answers],
                                   'superseded_ids': [item['record_id'] for item in superseded]})
        writer.add(split, item, [])
    stats['facts'] = len(facts)
    return writer.close({
        'role': ROLE, 'source_dataset': 'ai-hyz/MemoryAgentBench Conflict_Resolution',
        'context': context, 'source_revision': revision, 'license': 'MIT (dataset card)',
        'time_unit': 'fact position + 1 (later facts override earlier ones)',
        'stats': dict(stats),
        'notice': ('Evaluation only. Single-hop FactConsolidation only: multi-hop questions use '
                   'the same context but their fact chains are not annotated, so no gold can be '
                   'assigned without guessing. Gold is heuristic (answer_match): the latest '
                   'fact that contains the answer among the facts that best match the '
                   'question subject; questions whose best-matching fact is later than the '
                   'answer-bearing one are dropped.'),
    })


# ---------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['fetch-frames', 'longmemeval_s', 'longmemeval_m',
                                            'locomo', 'multihop_rag', 'frames',
                                            'memoryagentbench_cr'])
    parser.add_argument('--raw', type=Path, required=True, help='public2 raw root')
    parser.add_argument('--output', type=Path, help='public2 output root')
    parser.add_argument('--max-sources', type=int)
    parser.add_argument('--max-questions', type=int, default=500)
    parser.add_argument('--revision', default='')
    parser.add_argument('--rate', type=float, default=2.0, help='Wikipedia requests/s')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'fetch-frames':
        print(json.dumps(fetch_frames(args.raw / 'frames', rate=args.rate,
                                      workers=args.workers)))
        return
    tokenizer = load_tokenizer()
    target = args.output / args.command
    if args.command.startswith('longmemeval'):
        size = args.command.rsplit('_', 1)[1]
        summary = build_longmemeval(
            args.raw / 'longmemeval' / f'longmemeval_{size}_cleaned.json', target, args.command,
            tokenizer=tokenizer, max_sources=args.max_sources or LME_SOURCE_CAP,
            max_questions=args.max_questions, revision=args.revision)
    elif args.command == 'locomo':
        summary = build_locomo(args.raw / 'locomo' / 'locomo10.json', target,
                               tokenizer=tokenizer, revision=args.revision)
    elif args.command == 'multihop_rag':
        summary = build_multihop_rag(args.raw / 'multihop_rag', target, tokenizer=tokenizer,
                                     revision=args.revision)
    elif args.command == 'memoryagentbench_cr':
        summary = build_memoryagentbench_cr(
            args.raw / 'memoryagentbench' / 'data' / 'Conflict_Resolution-00000-of-00001.parquet',
            target, tokenizer=tokenizer, revision=args.revision)
    else:
        summary = build_frames(args.raw / 'frames', target, tokenizer=tokenizer,
                               revision=args.revision,
                               max_sources=args.max_sources or FRAMES_SOURCE_CAP)
    summary.pop('wikipedia_revisions', None)
    summary.get('selection', {}).pop('selected_question_ids', None)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
