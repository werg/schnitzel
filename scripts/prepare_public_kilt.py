"""KILT (Natural Questions, FEVER, zsRE) and TopiOCQA as answer-bearing episodes.

Each dataset becomes its own domain (``kilt_nq``, ``kilt_fever``, ``kilt_zsre``,
``topiocqa``) through ``public_corpus_common``. Upstream train becomes ``train``
and upstream dev becomes ``validation``.

KILT provenance points into the KILT Wikipedia knowledge source
(``kilt_knowledgesource.json``, one page per line, ~37 GB). The knowledge source is
never loaded: the sampled task rows name the paragraphs they need, and one
streaming pass keeps only those paragraphs plus a few other paragraphs of the same
pages as distractor candidates. A gold source is the <=700-character chunk of a
provenance paragraph that contains the provenance span (and, for answer tasks, the
answer string). Several alternative provenance entries mean any one suffices.

TopiOCQA rows carry their gold passage from the TopiOCQA Wikipedia corpus inline
(``Gold_passage``), so its corpus is not needed. The query holds the causal prefix
of the conversation: earlier questions and answers, oldest turns dropped first
until the query fits. Distractors are other gold passages of the same article.

Answers are supervised targets only, never query text. Everything is static:
sources are created at 1 and queries asked at 2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_corpus_common import (  # noqa: E402
    MAX_QUERY_TOKENS, PROMPT, Writer, chunk, clean, episode, load_tokenizer, source,
)

RAW = Path('/archive/corpora/public2-raw-20260926')
OUTPUT = Path('/archive/corpora/public2-20260926')
KILT_REPO = 'facebook/kilt_tasks'
KILT_REVISION = '0f7881c6bf693742af91a31b4d8f827db2f41233'
TOPIOCQA_REPO = 'McGill-NLP/TopiOCQA'
TOPIOCQA_REVISION = '66cd1dbf5577c653ecb99b385200f08e15e12f30'
KNOWLEDGE_SOURCE = 'kilt_wikipedia/kilt_knowledgesource.json'
KILT_TASKS = {'kilt_nq': 'nq', 'kilt_fever': 'fever', 'kilt_zsre': 'structured_zeroshot'}
LICENSES = {
    'kilt_nq': 'KILT task data MIT; Natural Questions CC BY-SA 3.0; Wikipedia CC BY-SA 3.0',
    'kilt_fever': 'KILT task data MIT; FEVER CC BY-SA 3.0; Wikipedia CC BY-SA 3.0',
    'kilt_zsre': ('KILT task data MIT; zsRE (Levy et al. 2017) publicly released without an '
                  'explicit license; Wikipedia CC BY-SA 3.0'),
    'topiocqa': 'CC BY-NC-SA 4.0 (non-commercial; owner-approved 26 September 2026)',
}
FEVER_QUESTION = 'Is this claim supported or refuted by the stored passages? Claim: '
FEVER_LABELS = {'SUPPORTS': 'supported', 'REFUTES': 'refuted'}
MAX_ALTERNATIVES = 3
DISTRACTOR_CANDIDATES = 6
_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
_MARKUP = re.compile(r'^\s*BULLET::::-?\s*')


# ---------------------------------------------------------------- passages

def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences ``split_sentences`` would produce."""
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def paragraph_chunks(paragraph: str, spans: list[tuple[int, int]] = ()) -> list[tuple[str, bool]]:
    """Chunk one KILT paragraph; a chunk is gold when it overlaps a character span.

    Spans are offsets into the raw paragraph. A leading list marker is removed
    and the spans shifted with it. Chunk texts do not depend on ``spans``.
    """
    marker = _MARKUP.match(paragraph)
    shift = marker.end() if marker else 0
    text = paragraph[shift:]
    bounds = sentence_spans(text)
    gold = set()
    for start, end in spans:
        start, end = start - shift, end - shift
        if start < 0 or end <= start:
            continue
        gold.update(index for index, (a, b) in enumerate(bounds) if a < end and b > start)
    return [(piece, is_gold) for piece, is_gold, _ in
            chunk([text[a:b] for a, b in bounds], gold)]


def eligible_distractor(paragraph: str) -> bool:
    text = clean(_MARKUP.sub('', paragraph))
    return len(text) >= 150 and not text.startswith('Section::::')


def contains(text: str, answer: str) -> bool:
    return clean(answer).lower() in clean(text).lower()


# ---------------------------------------------------------------- KILT rows

def read_kilt(raw: Path, task: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    path = raw / 'kilt_tasks' / task / f'{split}-00000-of-00001.parquet'
    return pq.read_table(path, columns=['id', 'input', 'output']).to_pylist()


def zsre_parts(text: str) -> tuple[str, str]:
    subject, _, relation = text.partition(' [SEP] ')
    return clean(subject), clean(relation)


def plan_nq(row: dict) -> dict | None:
    """First answer with span provenance; its alternatives are outputs with that answer."""
    for output in row['output']:
        if clean(output['answer']) and output['provenance']:
            answer = clean(output['answer'])
            groups = [[prov] for other in row['output']
                      if clean(other['answer']).lower() == answer.lower()
                      for prov in other['provenance'] or []]
            return {'question': clean(row['input']), 'answer': answer, 'groups': groups,
                    'answer_in_gold': True, 'meta': {}}
    return None


def plan_fever(row: dict) -> dict | None:
    labels = {output['answer'] for output in row['output'] if output['answer']}
    groups = [list(output['provenance']) for output in row['output'] if output['provenance']]
    if len(labels) != 1 or not groups or next(iter(labels)) not in FEVER_LABELS:
        return None
    label = next(iter(labels))
    return {'question': FEVER_QUESTION + clean(row['input']), 'answer': FEVER_LABELS[label],
            'groups': groups, 'answer_in_gold': False, 'meta': {'label': label}}


def plan_zsre(row: dict) -> dict | None:
    subject, relation = zsre_parts(row['input'])
    for output in row['output']:
        if clean(output['answer']) and output['provenance']:
            return {'question': f'What is the {relation} of {subject}?',
                    'answer': clean(output['answer']),
                    'groups': [[prov] for prov in output['provenance']],
                    'answer_in_gold': True, 'meta': {'relation': relation}}
    return None


PLANS = {'kilt_nq': plan_nq, 'kilt_fever': plan_fever, 'kilt_zsre': plan_zsre}


def select_rows(domain: str, rows: list[dict], split: str, target: int, seed: int,
                overshoot: float = 1.6) -> list[tuple[dict, dict]]:
    """Shuffle, plan and keep a surplus of candidates (filters run later).

    FEVER keeps equal label counts. zsRE caps every relation: at 3% of the target
    for train, and at an equal share of the target for validation, whose
    relations are disjoint from train (zero-shot) and too few for a 3% cap.
    """
    rows = list(rows)
    random.Random(f'{seed}:{domain}:{split}').shuffle(rows)
    planned = [(row, plan) for row in rows if (plan := PLANS[domain](row)) is not None]
    wanted = math.ceil(target * overshoot) + 50
    if domain == 'kilt_fever':
        per_label: dict[str, list] = defaultdict(list)
        for row, plan in planned:
            per_label[plan['answer']].append((row, plan))
        share = math.ceil(wanted / 2)
        chosen = [item for items in per_label.values() for item in items[:share]]
        random.Random(f'{seed}:{domain}:{split}:mix').shuffle(chosen)
        return chosen
    if domain == 'kilt_zsre':
        relations = Counter(plan['meta']['relation'] for _, plan in planned)
        cap = relation_cap(split, target, len(relations))
        seen: Counter = Counter()
        chosen = []
        for row, plan in planned:
            relation = plan['meta']['relation']
            if seen[relation] < math.ceil(cap * overshoot) + 5:
                seen[relation] += 1
                chosen.append((row, plan))
        return chosen
    return planned[:wanted]


def relation_cap(split: str, target: int, relations: int) -> int:
    return max(1, int(0.03 * target)) if split == 'train' else math.ceil(target / max(relations, 1))


# ---------------------------------------------------------------- knowledge source

_WID = re.compile(r'"wikipedia_id":\s*"([^"]+)"')


def scan_knowledge_source(path: Path, needs: dict[str, set[int]], *, seed: int = 1701,
                          candidates: int = DISTRACTOR_CANDIDATES, log_every: int = 500_000) -> dict:
    """One streaming pass: needed paragraphs plus distractor candidates per page."""
    pages: dict[str, dict] = {}
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if log_every and number % log_every == 0:
                print(f'knowledge source: {number} pages, {len(pages)} kept', flush=True)
            match = _WID.search(line, 0, 400)
            if not match or match.group(1) not in needs:
                continue
            page = json.loads(line)
            wikipedia_id = str(page['wikipedia_id'])
            paragraphs = page['text']
            wanted = {index for index in needs[wikipedia_id] if 0 <= index < len(paragraphs)}
            pool = [index for index in range(1, len(paragraphs))
                    if index not in needs[wikipedia_id] and eligible_distractor(paragraphs[index])]
            extra = random.Random(f'{seed}:{wikipedia_id}').sample(pool, min(candidates, len(pool)))
            pages[wikipedia_id] = {
                'title': clean(page['wikipedia_title']),
                'paragraphs': {index: paragraphs[index] for index in sorted(wanted | set(extra))},
                'distractors': extra, 'count': len(paragraphs)}
    return pages


def kilt_needs(plans: list[tuple[dict, dict]]) -> dict[str, set[int]]:
    needs: dict[str, set[int]] = defaultdict(set)
    for _, plan in plans:
        for group in plan['groups']:
            for prov in group:
                needs[str(prov['wikipedia_id'])].add(int(prov['start_paragraph_id']))
    return needs


# ---------------------------------------------------------------- building

class Budget:
    """Pace new distractor sources so the domain stays under ``max_sources``."""

    def __init__(self, writer: Writer, max_sources: int, episodes: int):
        self.writer, self.max_sources, self.episodes = writer, max_sources, max(episodes, 1)

    def allows(self, extra: int, done: int) -> bool:
        # A small slack keeps the first (validation) episodes from being starved, and
        # one source per remaining episode stays reserved for gold, so the budget
        # never selects episodes by whether their gold passage already exists.
        slack = max(3, int(0.02 * self.max_sources))
        reserve = max(self.episodes - done - 1, 0)
        allowance = min(self.max_sources - reserve,
                        self.max_sources * (done + 1) / self.episodes + slack)
        return len(self.writer.sources) + extra <= allowance

    def full(self, extra: int) -> bool:
        return len(self.writer.sources) + extra > self.max_sources


def pick_distractors(options: list[dict], existing: dict, count: int, budget: Budget,
                     pending: int, done: int) -> list[dict]:
    """Prefer passages already in the domain; add new ones while the budget allows."""
    reuse = [item for item in options if item['record_id'] in existing][:count]
    fresh = [item for item in options if item['record_id'] not in existing]
    chosen = list(reuse)
    for item in fresh:
        if len(chosen) >= count or not budget.allows(pending + len(chosen) - len(reuse) + 1, done):
            break
        chosen.append(item)
    return chosen


def kilt_source(domain: str, page: dict, wikipedia_id: str, paragraph_id: int, text: str) -> dict:
    return source(domain, text, title=page['title'], provenance={
        'wikipedia_id': wikipedia_id, 'paragraph_id': paragraph_id,
        'knowledge_source': 'kilt_knowledgesource.json'})


def resolve_kilt(domain: str, plan: dict, pages: dict, filters) -> tuple[list[list[dict]], set]:
    """Map provenance groups to gold chunk sources; drop unusable groups."""
    resolved, used = [], set()
    for group in plan['groups']:
        chunks: dict[str, dict] = {}
        ok = True
        for prov in group:
            wikipedia_id = str(prov['wikipedia_id'])
            start, end = int(prov['start_paragraph_id']), int(prov['end_paragraph_id'])
            page = pages.get(wikipedia_id)
            used.add((wikipedia_id, start))
            if start != end:
                filters.reject('provenance_multi_paragraph')
                ok = False
                break
            if page is None or start not in page['paragraphs']:
                filters.reject('provenance_missing_in_knowledge_source')
                ok = False
                break
            span = (int(prov['start_character']), int(prov['end_character']))
            pieces = paragraph_chunks(page['paragraphs'][start], [span])
            gold = [text for text, is_gold in pieces if is_gold]
            if plan['answer_in_gold']:
                if not gold:
                    gold = [text for text, _ in pieces]
                gold = [text for text in gold if contains(text, plan['answer'])][:1]
                if not gold:
                    filters.reject('provenance_without_answer')
                    ok = False
                    break
            elif not gold:
                filters.reject('provenance_span_invalid')
                ok = False
                break
            for text in gold:
                item = kilt_source(domain, page, wikipedia_id, start, text)
                chunks[item['record_id']] = item
        if ok and chunks:
            resolved.append(list(chunks.values()))
    return resolved, used


def build_kilt(domain: str, writer: Writer, split: str, plans: list[tuple[dict, dict]],
               pages: dict, target: int, budget: Budget, *, tokenizer=None,
               distractors: int = 2, done_before: int = 0) -> int:
    filters = writer.filters_for(split)
    by_page: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in writer.sources.values():
        by_page[item['provenance']['wikipedia_id']][item['record_id']] = item
    relations: Counter = Counter()
    labels: Counter = Counter()
    cap = None
    if domain == 'kilt_zsre':
        cap = relation_cap(split, target, len({plan['meta']['relation'] for _, plan in plans}))
    kept = 0
    for row, plan in plans:
        if kept >= target:
            break
        if cap is not None and relations[plan['meta']['relation']] >= cap:
            filters.reject('relation_cap')
            continue
        if domain == 'kilt_fever' and labels[plan['answer']] >= math.ceil(target / 2):
            filters.reject('label_balance')
            continue
        groups, used = resolve_kilt(domain, plan, pages, filters)
        singles = [group[0] for group in groups if len(group) == 1]
        if singles:
            gold, all_required = list({g['record_id']: g for g in singles}.values())[:MAX_ALTERNATIVES], False
        elif small := [group for group in groups if len(group) <= MAX_ALTERNATIVES]:
            gold, all_required = small[0], True
        elif groups:
            filters.reject('evidence_too_many_chunks')
            continue
        else:
            filters.reject('no_usable_provenance')
            continue
        gold_ids = {item['record_id'] for item in gold}
        options = []
        for wikipedia_id in dict.fromkeys(item['provenance']['wikipedia_id'] for item in gold):
            page = pages[wikipedia_id]
            existing = [item for key, item in by_page[wikipedia_id].items()
                        if key not in gold_ids and (wikipedia_id, item['provenance']['paragraph_id'])
                        not in used]
            options.extend(existing)
            for index in page['distractors']:
                if (wikipedia_id, index) in used:
                    continue
                text = paragraph_chunks(page['paragraphs'][index])[0][0]
                options.append(kilt_source(domain, page, wikipedia_id, index, text))
        if domain != 'kilt_fever':
            options = [item for item in options if not contains(item['text'], plan['answer'])]
        options = list({item['record_id']: item for item in options
                        if item['record_id'] not in gold_ids}.values())
        pending = sum(item['record_id'] not in writer.sources for item in gold)
        if budget.full(pending):
            filters.reject('source_budget')
            continue
        chosen = pick_distractors(options, writer.sources, distractors, budget, pending,
                                  done_before + kept)
        wikipedia_ids = sorted({item['provenance']['wikipedia_id'] for item in gold})
        item = episode(domain=domain, split=split, identifier=str(row['id']),
                       question=plan['question'], answer=plan['answer'], gold=gold,
                       supports=[*gold, *chosen], filters=filters, tokenizer=tokenizer,
                       all_required=all_required,
                       # Both labels are named by the FEVER question itself.
                       allow_answer_in_query=domain == 'kilt_fever',
                       annotation='answer_match' if domain == 'kilt_zsre' else 'verified',
                       provenance={'upstream_id': str(row['id']), 'upstream_split': (
                           'train' if split == 'train' else 'validation'),
                           'upstream': f'{KILT_REPO}@{KILT_REVISION}',
                           'wikipedia_ids': wikipedia_ids, **plan['meta']})
        if writer.add(split, item, [*gold, *chosen]):
            kept += 1
            relations[plan['meta'].get('relation')] += 1
            labels[plan['answer']] += 1
            for row_source in (*gold, *chosen):
                by_page[row_source['provenance']['wikipedia_id']][row_source['record_id']] = row_source
    return kept


# ---------------------------------------------------------------- TopiOCQA

def read_topiocqa(raw: Path, split: str) -> list[dict]:
    name = 'topiocqa_train.jsonl' if split == 'train' else 'topiocqa_valid.jsonl'
    with (raw / 'topiocqa' / 'data' / name).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def topiocqa_title(title: str) -> str:
    topic, _, section = title.partition(' [SEP] ')
    return f'{clean(topic)}: {clean(section)}' if clean(section) else clean(topic)


def topiocqa_topic(passage: dict) -> str:
    return clean(passage['title'].partition(' [SEP] ')[0])


def topiocqa_chunks(passage: dict, rationale: str = '', answer: str = '') -> list[tuple[str, bool]]:
    """Chunk a TopiOCQA passage; gold chunks overlap the rationale (else the answer)."""
    text = clean(passage['text'])
    bounds = sentence_spans(text)
    gold: set[int] = set()
    for needle in (clean(rationale), clean(answer)):
        if not needle:
            continue
        start = text.lower().find(needle.lower())
        if start >= 0:
            end = start + len(needle)
            gold = {index for index, (a, b) in enumerate(bounds) if a < end and b > start}
            break
    return [(piece, is_gold) for piece, is_gold, _ in chunk([text[a:b] for a, b in bounds], gold)]


def topiocqa_source(passage: dict, text: str) -> dict:
    return source('topiocqa', text, title=topiocqa_title(passage['title']),
                  provenance={'passage_id': passage['id'],
                              'topic': topiocqa_topic(passage)})


def conversational_question(row: dict, tokenizer=None) -> str | None:
    """Current question with as many most-recent earlier turns as fit the query."""
    context = [clean(text) for text in row['Context']]
    turns = [(context[i], context[i + 1]) for i in range(0, len(context) - 1, 2)]
    question = clean(row['Question'])
    for keep in range(len(turns), -1, -1):
        if keep == 0 and turns:
            return None
        history = ' | '.join(f'Q: {q} A: {a}' for q, a in turns[len(turns) - keep:])
        text = (f'Earlier in this conversation: {history} Current question: {question}'
                if history else question)
        if tokenizer is None or len(tokenizer.encode(PROMPT + text, add_special_tokens=False)
                                    ) <= MAX_QUERY_TOKENS:
            return text
    return None


def build_topiocqa(writer: Writer, split: str, rows: list[dict], target: int, budget: Budget, *,
                   tokenizer=None, distractors: int = 2, seed: int = 1701,
                   done_before: int = 0) -> int:
    filters = writer.filters_for(split)
    by_topic: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        passage = row['Gold_passage']
        if passage and passage.get('id') and clean(passage.get('text', '')):
            by_topic[topiocqa_topic(passage)][passage['id']] = passage
    order = list(rows)
    random.Random(f'{seed}:topiocqa:{split}').shuffle(order)
    kept = 0
    for row in order:
        if kept >= target:
            break
        answer = clean(row['Answer'])
        passage = row['Gold_passage']
        if not answer or answer.upper() == 'UNANSWERABLE':
            filters.reject('unanswerable')
            continue
        if not passage or not clean(passage.get('text', '')):
            filters.reject('no_gold_passage')
            continue
        pieces = topiocqa_chunks(passage, row['Rationale'], answer)
        gold_texts = [text for text, is_gold in pieces if is_gold]
        if not gold_texts and len(pieces) == 1:
            gold_texts = [pieces[0][0]]
        if not gold_texts:
            filters.reject('rationale_not_located')
            continue
        question = conversational_question(row, tokenizer)
        if question is None:
            filters.reject('history_does_not_fit')
            continue
        gold = [topiocqa_source(passage, text) for text in gold_texts]
        gold_ids = {item['record_id'] for item in gold}
        options = []
        others = [other for key, other in sorted(by_topic[topiocqa_topic(passage)].items())
                  if key != passage['id']]
        random.Random(f'{seed}:{row["Conversation_no"]}:{row["Turn_no"]}').shuffle(others)
        for other in others:
            if len(options) >= 4 * distractors:
                break
            for text, _ in topiocqa_chunks(other):
                if not contains(text, answer):
                    options.append(topiocqa_source(other, text))
                    break
        options = [item for item in options if item['record_id'] not in gold_ids]
        pending = sum(item['record_id'] not in writer.sources for item in gold)
        if budget.full(pending):
            filters.reject('source_budget')
            continue
        chosen = pick_distractors(options, writer.sources, distractors, budget, pending,
                                  done_before + kept)
        item = episode(domain='topiocqa', split=split,
                       identifier=f"{split}-{row['Conversation_no']}-{row['Turn_no']}",
                       question=question, answer=answer, gold=gold, supports=[*gold, *chosen],
                       filters=filters, tokenizer=tokenizer, all_required=True,
                       task_family='public_conversational_qa',
                       provenance={'conversation': int(row['Conversation_no']),
                                   'turn': int(row['Turn_no']), 'passage_id': passage['id'],
                                   'upstream': f'{TOPIOCQA_REPO}@{TOPIOCQA_REVISION}',
                                   'upstream_split': 'train' if split == 'train' else 'dev'})
        if writer.add(split, item, [*gold, *chosen]):
            kept += 1
    return kept


# ---------------------------------------------------------------- driver

DEFAULTS = {  # train episodes, validation episodes, maximum sources
    'kilt_nq': (15000, 500, 26500),
    'kilt_fever': (10000, 500, 15000),
    'kilt_zsre': (10000, 500, 18500),
    'topiocqa': (8000, 500, 14000),
}


def manifest_for(domain: str, *, seed: int, targets: tuple, distractors: int,
                 knowledge_source: str | None) -> dict:
    upstream = (f'{TOPIOCQA_REPO}@{TOPIOCQA_REVISION}' if domain == 'topiocqa'
                else f'{KILT_REPO}@{KILT_REVISION} ({KILT_TASKS[domain]})')
    notes = {
        'kilt_nq': 'Gold is the chunk of a KILT provenance paragraph holding the annotated '
                   'short-answer span; alternative provenances with the same answer are '
                   'separate sufficient groups (any one suffices).',
        'kilt_fever': 'NOT ENOUGH INFO has no provenance and is absent from KILT. Labels are '
                      'balanced by sampling. Each single-chunk evidence set is an '
                      'alternative; a claim with only multi-chunk evidence requires all '
                      'chunks of its first set of at most 3 chunks. Distractors are other '
                      'paragraphs of the evidence pages and are not verified to be neutral.',
        'kilt_zsre': 'Question rendered as "What is the <relation> of <subject>?". KILT dev '
                     'relations are disjoint from train relations (zero-shot); train caps each '
                     'relation at 3% of the target, validation (12 relations) at an equal '
                     'share. Gold chunks must contain the answer string.',
        'topiocqa': 'Gold passage is taken inline from the dataset (TopiOCQA Wikipedia '
                    'corpus). Queries carry the most recent earlier turns (questions and '
                    'answers) that fit MAX_QUERY_TOKENS. Distractors are other gold passages '
                    'of the same article in the same split.',
    }
    return {'upstream': upstream, 'license': LICENSES[domain], 'seed': seed,
            'targets': {'train': targets[0], 'validation': targets[1], 'max_sources': targets[2]},
            'distractors_per_episode_max': distractors,
            'knowledge_source': knowledge_source, 'created_at': 1, 'query_time': 2,
            'notice': notes[domain] + ' Answers are supervised targets only, never query text.'}


def prepare(raw: Path, output: Path, domains: list[str], *, tokenizer=None, seed: int = 1701,
            targets: dict | None = None, distractors: int = 2,
            knowledge_source: Path | None = None) -> dict:
    targets = {**DEFAULTS, **(targets or {})}
    for domain in domains:
        if (output / domain).exists():
            raise FileExistsError(output / domain)
    summaries = {}
    kilt = [domain for domain in domains if domain in KILT_TASKS]
    if kilt:
        plans = {}
        for domain in kilt:
            for split, upstream, index in (('train', 'train', 0), ('validation', 'validation', 1)):
                rows = read_kilt(raw, KILT_TASKS[domain], upstream)
                plans[domain, split] = select_rows(domain, rows, split, targets[domain][index], seed)
                del rows
        needs: dict[str, set[int]] = defaultdict(set)
        for planned in plans.values():
            for key, value in kilt_needs(planned).items():
                needs[key] |= value
        print(f'knowledge source: {len(needs)} pages needed', flush=True)
        path = knowledge_source or raw / KNOWLEDGE_SOURCE
        pages = scan_knowledge_source(path, needs, seed=seed)
        for domain in kilt:
            writer = Writer(output / domain, domain)
            train, validation, max_sources = targets[domain]
            budget = Budget(writer, max_sources, train + validation)
            done = build_kilt(domain, writer, 'validation', plans[domain, 'validation'], pages,
                              validation, budget, tokenizer=tokenizer, distractors=distractors)
            build_kilt(domain, writer, 'train', plans[domain, 'train'], pages, train, budget,
                       tokenizer=tokenizer, distractors=distractors, done_before=done)
            summaries[domain] = writer.close(manifest_for(
                domain, seed=seed, targets=targets[domain], distractors=distractors,
                knowledge_source='http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json'))
    if 'topiocqa' in domains:
        writer = Writer(output / 'topiocqa', 'topiocqa')
        train, validation, max_sources = targets['topiocqa']
        budget = Budget(writer, max_sources, train + validation)
        done = build_topiocqa(writer, 'validation', read_topiocqa(raw, 'validation'), validation,
                              budget, tokenizer=tokenizer, distractors=distractors, seed=seed)
        build_topiocqa(writer, 'train', read_topiocqa(raw, 'train'), train, budget,
                       tokenizer=tokenizer, distractors=distractors, seed=seed, done_before=done)
        summaries['topiocqa'] = writer.close(manifest_for(
            'topiocqa', seed=seed, targets=targets['topiocqa'], distractors=distractors,
            knowledge_source=None))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--datasets', default=','.join(DEFAULTS))
    parser.add_argument('--knowledge-source', type=Path)
    parser.add_argument('--distractors', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--no-tokenizer', action='store_true',
                        help='skip token-length checks (tests only)')
    for domain, (train, validation, max_sources) in DEFAULTS.items():
        flag = domain.replace('_', '-')
        parser.add_argument(f'--{flag}-train', type=int, default=train)
        parser.add_argument(f'--{flag}-validation', type=int, default=validation)
        parser.add_argument(f'--{flag}-max-sources', type=int, default=max_sources)
    args = parser.parse_args()
    domains = [name for name in args.datasets.split(',') if name]
    unknown = set(domains) - DEFAULTS.keys()
    if unknown:
        parser.error(f'unknown datasets: {sorted(unknown)}')
    targets = {domain: tuple(getattr(args, f'{domain}_{part}') for part in
                             ('train', 'validation', 'max_sources')) for domain in DEFAULTS}
    summaries = prepare(args.raw, args.output, domains,
                        tokenizer=None if args.no_tokenizer else load_tokenizer(),
                        seed=args.seed, targets=targets, distractors=args.distractors,
                        knowledge_source=args.knowledge_source)
    print(json.dumps({domain: {key: summary[key] for key in (
        'episodes', 'sources', 'unreferenced_sources', 'gold_per_episode', 'filtered')}
        for domain, summary in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
