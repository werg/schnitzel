"""Verbatim recall of real text from a highly redundant KB (owner, 28 September).

A task whose target has high entropy for the model on its own (content, not format) but
is trivially solvable from the KB: every document is cut into overlapping token windows
(``window`` tokens every ``stride`` tokens, so each token is in about ``window / stride``
records), plus one title record (title and lead paragraph). Episodes ask for plain prose
verbatim:

- ``continuation``: one or two sentences of the document are given; the target is the
  next 64-256 tokens;
- ``title``: "recite the passage about <title>"; the target is the first 128 tokens;
- ``middle``: a sentence from the middle is given; the target is the sentences that
  follow it (64-192 tokens).

Targets end at a sentence end where one leaves at least 64 tokens, else at a word end.
Every record overlapping the target is a support. Sufficient groups are the covers of
the target by one residue class of windows (windows ``window / stride`` starts apart
tile the text, so there are about ``window / stride`` of them). ``alternatives``
partitions the target at the window grid: per segment, every record that contains it
whole, so a transcript slot names every copy of its part of the target and the L1 bank
build stores all of them. ``neutral`` lists every other record of the episode's
document (windows overlapping only the cue or next to the target, the title record):
they share most of their text with the positives, so the L1 retrieval loss scores
them neither as positives nor as negatives. Train and validation are split by
document; validation documents' windows are in the KB.

Tokens come from a pluggable ``offsets(text) -> [(start, end), ...]``: the LFM2.5
tokenizer's offset mapping for corpora, a regex word/punctuation tokenizer for tests.
Windows are widened to whole words.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Callable

DOMAIN = 'recall_text'
RECORD_CHARS = 1500
PROMPTS = {
    'continuation': ('Continue this passage word for word, exactly as it is stored:\n{cue}',
                     'The stored text contains this passage. Recite what comes right after it, '
                     'verbatim:\n{cue}'),
    'title': ('Recite the beginning of the stored passage about {cue}, word for word.',
              'From the stored text, reproduce the opening of the article "{cue}" exactly.'),
    'middle': ('The stored text contains this sentence:\n{cue}\nRecite the sentences that '
               'follow it, verbatim.',
               'Find this sentence in the stored text and continue it word for word:\n{cue}'),
}
PREFIX = 'Use the stored passages. '
Offsets = Callable[[str], list[tuple[int, int]]]


def regex_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]


def hf_offsets(tokenizer) -> Offsets:
    def offsets(text: str) -> list[tuple[int, int]]:
        spans = tokenizer(text, add_special_tokens=False,
                          return_offsets_mapping=True)['offset_mapping']
        return [(a, b) for a, b in spans if b > a]
    return offsets


def record_id(body: str) -> str:
    return hashlib.sha256(f'{DOMAIN}\0{body}'.encode()).hexdigest()[:32]


def _rng(seed: int, *parts) -> random.Random:
    key = '\0'.join(str(p) for p in (seed, *parts))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


@dataclass
class Document:
    key: str
    title: str
    text: str
    source: str


def load_wikipedia(paths: list[Path]) -> list[Document]:
    """Articles of background ``passages.jsonl`` files (ids ``enwiki:<page>:<chunk>``),
    chunks joined in order; one document per page id."""
    chunks: dict[str, list] = {}
    titles, sources = {}, {}
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                page, _, index = row['id'].rpartition(':')
                if page in sources and sources[page] != str(path):
                    continue                  # the same page in another corpus
                sources[page] = str(path)
                titles[page] = row['title']
                chunks.setdefault(page, []).append((int(index), row['text'].strip()))
    return [Document(page, titles[page], '\n'.join(t for _, t in sorted(chunks[page])),
                     sources[page]) for page in sorted(chunks)]


def _word_bounds(text: str, a: int, b: int) -> tuple[int, int]:
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    return a, b


def _record(text: str, record_type: str, doc: Document, **provenance) -> dict:
    return {'record_id': record_id(text), 'text': text, 'domain': DOMAIN, 'created_at': 1,
            'kind': 'passage', 'provenance': {'dataset': DOMAIN, 'record_type': record_type,
                                              'article_title': f'{DOMAIN}/{doc.title}',
                                              'document': doc.key, **provenance}}


@dataclass
class Cut:
    """A document's tokens, windows (token ranges) and records."""
    doc: Document
    tokens: list[tuple[int, int]]
    windows: list[tuple[int, int]]           # token [start, end) per window record
    records: list[dict]                      # window records, then the title record
    lead_end: int                            # the title record holds tokens [0, lead_end)


def window_spans(n: int, window: int, stride: int) -> list[tuple[int, int, str]]:
    """Token spans of the window records of an ``n``-token document: full windows every
    ``stride`` tokens (the last one ending at the document end), plus shorter head
    windows ``[0, window - j * stride)`` and tail windows ``[n - window + j * stride, n)``,
    so tokens near either end are in as many records as those in the middle."""
    if n <= window:
        return [(0, n, 'full')]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    spans = [(s, s + window, 'full') for s in starts]
    for j in range(1, window // stride):
        spans.append((0, window - j * stride, 'head'))
        spans.append((n - window + j * stride, n, 'tail'))
    return spans


def cut(doc: Document, offsets: Offsets, window: int, stride: int) -> Cut | None:
    tokens = offsets(doc.text)
    header = f'Title: {doc.title}\nPassage: '
    windows, recs = [], []
    for k, (s, e, shape) in enumerate(window_spans(len(tokens), window, stride)):
        a, b = _word_bounds(doc.text, tokens[s][0], tokens[e - 1][1])
        text = header + doc.text[a:b]
        if len(text) > RECORD_CHARS:
            return None
        windows.append((s, e))
        recs.append(_record(text, 'window', doc, window=k, shape=shape, tokens=[s, e]))
    lead = doc.text.split('\n', 1)[0]
    lead_end = sum(1 for t in tokens if t[1] <= len(lead))
    lead_text = f'Title: {doc.title}\nLead: {lead}'
    if len(lead_text) <= RECORD_CHARS and lead_end > 0:
        recs.append(_record(lead_text, 'title', doc, tokens=[0, lead_end]))
    else:
        lead_end = 0
    return Cut(doc, tokens, windows, recs, lead_end)


def _sentence_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])|\n+', text)]


def _token_at(tokens: list[tuple[int, int]], char: int) -> int:
    """The first token ending after ``char`` (a token may carry the space before it)."""
    return next((i for i, (_, b) in enumerate(tokens) if b > char), len(tokens))


def _target_end(c: Cut, start: int, length: int, ends: set[int], minimum: int = 64) -> int:
    """Token end of a target from ``start``: the last sentence end in [start + minimum,
    start + length], else ``start + length`` (a word end)."""
    stop = min(start + length, len(c.tokens))
    for e in range(stop, start + minimum - 1, -1):
        if e in ends:
            return e
    return stop


def target_support(c: Cut, start: int, end: int, stride: int, redundancy: int):
    """Supports, sufficient groups and per-segment alternatives of tokens [start, end)."""
    spans = [*c.windows] + ([(0, c.lead_end)] if c.lead_end else [])
    ids = [r['record_id'] for r in c.records]
    overlapping = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    groups: list[list[str]] = []
    full = [i for i in overlapping if i < len(c.windows)
            and c.records[i]['provenance']['shape'] == 'full']
    for residue in range(redundancy):
        members = [i for i in full if (c.windows[i][0] // stride) % redundancy == residue]
        # gaps (the document's ends, an off-grid last window) filled greedily by the
        # other overlapping records: the one reaching furthest past the first gap
        while True:
            covered = set()
            for i in members:
                covered.update(range(*spans[i]))
            gap = next((t for t in range(start, end) if t not in covered), None)
            if gap is None:
                break
            fill = max((i for i in overlapping if i not in members
                        and spans[i][0] <= gap < spans[i][1]),
                       key=lambda i: spans[i][1], default=None)
            if fill is None:
                members = []
                break
            members.append(fill)
        group = [ids[i] for i in sorted(set(members), key=lambda i: spans[i])]
        if group and group not in groups:
            groups.append(group)
    grid = sorted({start, end, *(p for s, e in spans for p in (s, e) if start < p < end)})
    segments = []
    for a, b in zip(grid, grid[1:]):
        holders = [ids[i] for i in overlapping if spans[i][0] <= a and spans[i][1] >= b]
        if holders and (not segments or holders != segments[-1]):
            segments.append(holders)
    return [ids[i] for i in overlapping], groups, segments


def episodes_for(c: Cut, split: str, seed: int, stride: int, redundancy: int, *,
                 continuations: int = 2, middles: int = 1, title: bool = True) -> list[dict]:
    doc, tokens = c.doc, c.tokens
    rng = _rng(seed, 'episodes', doc.key)
    starts = _sentence_starts(doc.text)
    sent_tok = sorted({_token_at(tokens, s) for s in starts})
    ends = set(sent_tok[1:]) | {len(tokens)}
    by_id = {r['record_id']: r for r in c.records}
    out = []

    def add(kind: str, cue: str, start: int, length: int, index: int):
        end = _target_end(c, start, length, ends)
        if end - start < 64 or not cue.strip():
            return
        a, b = tokens[start][0], tokens[end - 1][1]
        answer = doc.text[a:b].strip()
        supports, groups, segments = target_support(c, start, end, stride, redundancy)
        if not groups:
            return
        rng.shuffle(groups)
        named = set(supports).union(*map(set, segments))
        neutral = list(dict.fromkeys(r['record_id'] for r in c.records
                                     if r['record_id'] not in named))
        query = PREFIX + rng.choice(PROMPTS[kind]).format(cue=cue.strip())
        out.append({
            'episode_id': f'{DOMAIN}-{doc.key}-{kind}-{index}',
            'environment': f'{DOMAIN}-{split}', 'query': query, 'answer': answer,
            'query_time': 2, 'required_ids': supports, 'sufficient_groups': groups,
            'alternatives': segments, 'neutral': neutral, 'support_annotation': 'verified',
            'task_family': 'synthetic_recall',
            'supports': [{'record_id': r, 'text': by_id[r]['text'], 'created_at': 1,
                          'kind': 'passage'} for r in supports],
            'verify': {'type': 'exact', 'answer': answer},
            'provenance': {'dataset': DOMAIN, 'domain': DOMAIN, 'split': split,
                           'document': doc.key, 'title': doc.title, 'recall': kind,
                           'target_tokens': end - start, 'target_span': [start, end],
                           'source': doc.source}})

    if title:
        add('title', doc.title, 0, 128, 0)
    usable = [k for k in range(1, len(sent_tok)) if len(tokens) - sent_tok[k] >= 64]
    for j, k in enumerate(rng.sample(usable, min(continuations, len(usable)))):
        first = max(0, k - rng.choice((1, 2)))
        cue = doc.text[tokens[sent_tok[first]][0]:tokens[sent_tok[k] - 1][1]]
        add('continuation', cue, sent_tok[k], rng.randint(64, 256), j)
    third = len(sent_tok) // 3
    middle = [k for k in usable if third <= k - 1 < 2 * third + 1]
    for j, k in enumerate(rng.sample(middle, min(middles, len(middle)))):
        cue = doc.text[tokens[sent_tok[k - 1]][0]:tokens[sent_tok[k] - 1][1]]
        add('middle', cue, sent_tok[k], rng.randint(64, 192), j)
    return out


def build(documents: list[Document], offsets: Offsets, *, window: int = 96, stride: int = 12,
          seed: int = 0, validation: float = 0.1, min_tokens: int = 200, max_tokens: int = 1500,
          continuations: int = 2, middles: int = 1):
    """Records, episodes per split and a summary."""
    if window % stride:
        raise ValueError('window must be a multiple of stride')
    redundancy = window // stride
    records: dict[str, dict] = {}
    episodes: dict[str, list[dict]] = {'train': [], 'validation': []}
    skipped: Counter = Counter()
    titles: set[str] = set()
    for doc in documents:
        if doc.title in titles:
            skipped['duplicate_title'] += 1
            continue
        n = len(offsets(doc.text))
        if not min_tokens <= n <= max_tokens:
            skipped['length'] += 1
            continue
        c = cut(doc, offsets, window, stride)
        if c is None:
            skipped['record_too_long'] += 1
            continue
        titles.add(doc.title)
        split = 'validation' if _rng(seed, 'split', doc.key).random() < validation else 'train'
        for r in c.records:
            records.setdefault(r['record_id'], r)
        episodes[split] += episodes_for(c, split, seed, stride, redundancy,
                                        continuations=continuations, middles=middles)
    rows = [e for es in episodes.values() for e in es]
    lengths = sorted(e['provenance']['target_tokens'] for e in rows)
    q = (lambda f: lengths[int(f * (len(lengths) - 1))]) if lengths else (lambda f: None)
    summary = {
        'window': window, 'stride': stride, 'redundancy': redundancy, 'seed': seed,
        'documents': len(titles), 'skipped_documents': dict(skipped),
        'records_by_type': dict(Counter(r['provenance']['record_type'] for r in records.values())),
        'record_chars_mean': round(sum(len(r['text']) for r in records.values())
                                   / max(len(records), 1)),
        'episodes_by_recall': dict(Counter(e['provenance']['recall'] for e in rows)),
        'target_tokens': {'min': q(0), 'p10': q(0.1), 'median': q(0.5), 'p90': q(0.9),
                          'max': q(1)},
        'supports_per_episode_median': sorted(len(e['supports']) for e in rows)[len(rows) // 2]
        if rows else None,
        'groups_per_episode': dict(Counter(len(e['sufficient_groups']) for e in rows)),
        'neutral_per_episode_median': sorted(len(e['neutral']) for e in rows)[len(rows) // 2]
        if rows else None,
        'records_per_group_median': sorted(len(e['sufficient_groups'][0]) for e in rows)[
            len(rows) // 2] if rows else None}
    return list(records.values()), episodes, summary
