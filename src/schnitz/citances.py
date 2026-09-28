"""Citation contexts (citances) grouped by cited paper: many independent descriptions
of the same content as a knowledge task.

Source: the unarXive 2022 citation-recommendation release (``saier/unarXive_citrec``,
CC BY-SA 4.0; paragraphs of arXiv papers with one annotated citation marker and the
cited work's OpenAlex id). ``scripts/fetch_citance_metadata.py`` resolves cited works to
arXiv and fetches titles and abstracts (arXiv metadata, CC0) of cited and citing papers.

A KB record is one citing paper's citation context for one cited paper: the citing
sentence with one sentence either side, headed by the citing paper's title. Citation
markers are removed and the cited paper's title is never added. Records of a cited paper
are its *copies*: each alone is a sufficient group, all of them are listed as
``alternatives`` (one hop), and ``redundancy`` counts them.

Episodes (split by cited paper; validation cited papers' citances are in the KB too):

- ``public_citance_abstract``: summarize the cited paper as its abstract states it.
  The target is the abstract's first 128-256 tokens (whole sentences where possible).
  No abstract is ever stored, and a citance sharing a ``NGRAM``-word sequence with any
  target abstract is dropped.
- ``public_citance_description``: the citing sentence of a *held-out citing paper*.
  Per-episode removal of one record is not supported by the KB machinery (one KB per
  corpus), so whole citing papers (``heldout_rate`` of them by hash) are held out: none
  of their citances is stored, and one of them serves as a target given the paper's
  title, the cited paper's title and the neighbouring sentences.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Callable, Iterable

DOMAIN = 'citance_recall'
RECORD_CHARS = 1500
NGRAM = 12
MIN_WORDS = 8           # citing sentence, after marker removal
MAX_MARKERS = 3         # citation markers in the citing sentence
PROMPT = 'Use the stored citation contexts.\n'

_MARKER = re.compile(r'\[\d+\]\}')
_MARKER_RUN = re.compile(r'\s*\[\d+\]\}(?:\s*(?:[,;\u2013-]|and)\s*\[\d+\]\})*')
_MATH = re.compile(r'\\\((?:.|\n)*?\\\)')
_ABBREV = re.compile(r'(?:\b(?:al|e\.g|i\.e|cf|[Ff]igs?|[Ee]qs?|[Rr]efs?|[Ss]ecs?|[Tt]ab|vs|'
                     r'resp|approx|[Nn]o|[Cc]h|[Tt]hm|[Pp]rop|[Dd]ef|[Ll]em|[Cc]or|[Aa]pp|'
                     r'pp?|[Vv]ol|[Ee]d|[Ee]ds|[Cc]hap|[Ss]ect)|\b[A-Z])\.$')


def _hash(*parts) -> int:
    return int(hashlib.sha256('\0'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def record_id(domain: str, body: str) -> str:
    return hashlib.sha256(f'{domain}\0{body}'.encode()).hexdigest()[:32]


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def strip_markers(text: str) -> str:
    """Remove unarXive citation markers (``[3]}``) and the punctuation debris they leave
    (empty brackets, doubled commas, space before punctuation)."""
    text = _MARKER_RUN.sub('', text)
    for _ in range(3):
        text = re.sub(r'[\(\[]\s*(?:[,;]\s*|and\s+)*[\)\]]', '', text)
        text = re.sub(r'(?:\s*,\s*){2,}', ', ', text)
        text = re.sub(r'\s+([,.;:)\]])', r'\1', text)
        text = re.sub(r',\s*([.;:])', r'\1', text)
        text = re.sub(r'\b(?:and|or)\s*([.;:)])', r'\1', text)
    return clean(text)


def split_spans(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end) of ``text``: a split after ``.!?`` followed by
    whitespace and a capital, bracket, backslash or quote, unless the word before the stop is a common abbreviation or initial.
    Periods inside inline math do not split (math is followed by no space inside)."""
    spans, start = [], 0
    for m in re.finditer(r'[.!?](?=\s+[A-Z\[\\(\'"\u201c])', text):
        before = text[start:m.end()]
        if _ABBREV.search(before):
            continue
        if text.count('\\(', start, m.end()) > text.count('\\)', start, m.end()):
            continue
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def math_share(text: str) -> float:
    return sum(len(m) for m in _MATH.findall(text)) / max(len(text), 1)


def context(text: str, offset: int, *, limit: int = RECORD_CHARS - 200) -> dict | None:
    """The citing sentence around character ``offset`` plus one sentence either side,
    markers stripped. None when the citing sentence is too short, math-heavy or cites
    too many works to describe the cited one."""
    spans = split_spans(text)
    at = next((i for i, (s, e) in enumerate(spans) if s <= offset < e), None)
    if at is None:
        return None
    raw = text[spans[at][0]:spans[at][1]]
    if len(_MARKER.findall(raw)) > MAX_MARKERS or math_share(raw) > 0.2:
        return None
    sentence = strip_markers(raw)
    if len(sentence.split()) < MIN_WORDS or len(sentence) > limit:
        return None
    before = strip_markers(text[spans[at - 1][0]:spans[at - 1][1]]) if at > 0 else ''
    after = strip_markers(text[spans[at + 1][0]:spans[at + 1][1]]) if at + 1 < len(spans) else ''
    if math_share(before) > 0.3:
        before = ''
    if math_share(after) > 0.3:
        after = ''
    if len(before) + len(sentence) + len(after) + 2 > limit:
        before = '' if len(before) > len(after) else before
        after = '' if len(before) + len(sentence) + len(after) + 2 > limit else after
    return {'before': before, 'sentence': sentence, 'after': after,
            'markers': len(_MARKER.findall(raw))}


def context_text(ctx: dict) -> str:
    return ' '.join(p for p in (ctx['before'], ctx['sentence'], ctx['after']) if p)


def words(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def ngrams(text: str, n: int = NGRAM) -> set[int]:
    toks = words(text)
    return {hash(tuple(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def titles_agree(a: str | None, b: str | None, threshold: float = 0.6) -> bool:
    """Whether two titles name the same work (the arXiv title against the OpenAlex
    record of the cited id, which OpenAlex sometimes re-points after merges)."""
    import difflib
    a, b = ' '.join(words(a or '')), ' '.join(words(b or ''))
    if not a or not b:
        return False
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def heldout_citer(citing: str, rate: float, seed: int = 0) -> bool:
    """Whole citing papers held out of the KB (their citances become targets)."""
    return _hash(seed, 'heldout', citing) / 2 ** 64 < rate


def citer_order(cited: str, citers: Iterable[str], seed: int = 0) -> list[str]:
    """A deterministic per-cited-paper order of its citing papers (records are taken in it)."""
    return sorted(set(citers), key=lambda c: _hash(seed, 'citer', cited, c))


def paper_order(cited: Iterable[str], seed: int = 0) -> list[str]:
    return sorted(set(cited), key=lambda c: _hash(seed, 'paper', c))


def abstract_target(abstract: str, count: Callable[[str], int], *, low: int = 128,
                    high: int = 256) -> str | None:
    """The abstract's first ``low``-``high`` tokens: whole sentences while they fit
    ``high``; if that gives fewer than ``low`` tokens, words up to ``high``. None when the
    whole abstract is shorter than ``low`` tokens."""
    abstract = clean(abstract)
    if count(abstract) < low:
        return None
    if count(abstract) <= high:
        return abstract
    out = ''
    for s, e in split_spans(abstract):
        trial = (out + ' ' + abstract[s:e].strip()).strip()
        if count(trial) > high:
            break
        out = trial
    if count(out) >= low:
        return out
    cut = abstract.split()
    lo, hi = 0, len(cut)
    while lo < hi:            # longest word prefix within ``high`` tokens
        mid = (lo + hi + 1) // 2
        if count(' '.join(cut[:mid])) <= high:
            lo = mid
        else:
            hi = mid - 1
    return ' '.join(cut[:lo])


def approx_tokens(text: str) -> int:
    """Token estimate without a tokenizer (about 1.35 subword tokens per word)."""
    return round(len(text.split()) * 1.35)


def _source(text: str, citing: str, cited: str, license_: str | None) -> dict:
    return {'record_id': record_id(DOMAIN, text), 'text': text, 'domain': DOMAIN,
            'created_at': 1, 'kind': 'passage',
            'provenance': {'dataset': DOMAIN, 'record_type': 'citance', 'citing_arxiv': citing,
                           'cited': [cited], 'license': license_,
                           'article_title': f'{DOMAIN}/{citing}'}}


def _episode(identifier: str, split: str, query: str, answer: str, records: list[dict],
             family: str, **provenance) -> dict:
    ids = [r['record_id'] for r in records]
    return {
        'episode_id': f'{DOMAIN}-{identifier}', 'environment': f'{DOMAIN}-{split}',
        'query': PROMPT + query, 'answer': answer, 'query_time': 2,
        'required_ids': ids, 'sufficient_groups': [[r] for r in ids], 'alternatives': [ids],
        'redundancy': len(ids), 'support_annotation': 'verified', 'task_family': family,
        'supports': [{'record_id': r['record_id'], 'text': r['text'], 'created_at': 1,
                      'kind': 'passage'} for r in records],
        'verify': {'type': 'reference', 'answer': answer},
        'provenance': {'dataset': DOMAIN, 'domain': DOMAIN, 'split': split, **provenance},
    }


def build(contexts: dict[str, dict[str, dict]], cited: dict[str, dict],
          citing: dict[str, dict], *, min_citances: int = 8, max_citances: int = 32,
          validation: int = 300, train: int | None = None, heldout_rate: float = 0.1,
          description_episodes: bool = True, count: Callable[[str], int] = approx_tokens,
          seed: int = 0) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Records, episodes per split and a summary.

    ``contexts[cited_id][citing_id]`` is the chosen ``context()`` of that citing paper
    for that cited work; ``cited[cited_id]`` has ``title`` and ``abstract``;
    ``citing[arxiv_id]`` has ``title`` (and ``license``). Cited papers are taken in
    ``paper_order``; the first ``validation`` kept ones form the validation split, the
    next ``train`` (all when None) the training split."""
    filters: Counter = Counter()
    targets: dict[str, str] = {}
    for cid in paper_order(contexts, seed):
        meta = cited.get(cid)
        if not meta or not meta.get('abstract') or not meta.get('title'):
            filters['no_abstract'] += 1
            continue
        target = abstract_target(meta['abstract'], count)
        if target is None:
            filters['abstract_short'] += 1
            continue
        targets[cid] = target
    banned: set[int] = set()
    for target in targets.values():
        banned |= ngrams(target)

    records: dict[str, dict] = {}
    papers = []                         # (cited id, stored records, held-out candidates)
    for cid in targets:
        stored, held = [], []
        for citer in citer_order(cid, contexts[cid], seed):
            ctx = contexts[cid][citer]
            paper = citing.get(citer)
            if not paper or not paper.get('title'):
                filters['citing_title_missing'] += 1
                continue
            if heldout_citer(citer, heldout_rate, seed):
                held.append((citer, ctx))
                continue
            if len(stored) >= max_citances:
                continue
            text = f'Citing paper: {clean(paper["title"])}\n{context_text(ctx)}'
            if len(text) > RECORD_CHARS:
                filters['citance_long'] += 1
                continue
            if ngrams(text) & banned:
                filters['citance_copies_abstract'] += 1
                continue
            rec = _source(text, citer, cid, paper.get('license'))
            rec = records.setdefault(rec['record_id'], rec)
            if cid not in rec['provenance']['cited']:
                rec['provenance']['cited'].append(cid)
            stored.append(rec)
        stored = list({r['record_id']: r for r in stored}.values())
        if len(stored) < min_citances:
            filters['too_few_citances'] += 1
            continue
        papers.append((cid, stored, held))
    # a description target must not be stored anywhere (self-reuse across papers): no
    # shared NGRAM-word sequence with, and not a substring of, any built record
    kb_joined = '\n'.join(' '.join(words(r['text'])) for r in records.values())
    kb_grams: set[int] = set()
    for r in records.values():
        kb_grams |= ngrams(r['text'])

    episodes: dict[str, list[dict]] = {'train': [], 'validation': []}
    kept = papers[:validation + (train if train is not None else len(papers))]
    for i, (cid, stored, held) in enumerate(kept):
        split = 'validation' if i < validation else 'train'
        title = clean(cited[cid]['title'])
        episodes[split].append(_episode(
            f'{cid.rsplit("/", 1)[-1]}-abstract', split,
            f'Summarize the paper "{title}" as its abstract states it.', targets[cid], stored,
            'public_citance_abstract', cited=cid, cited_arxiv=cited[cid].get('arxiv_id'),
            cited_title=title))
        if not description_episodes:
            continue
        for citer, ctx in held:
            if not (ctx['before'] or ctx['after']):
                filters['description_no_neighbours'] += 1
                continue
            grams = ngrams(ctx['sentence'])
            if grams & banned or grams & kb_grams or ' '.join(words(ctx['sentence'])) in kb_joined:
                filters['description_target_in_kb'] += 1
                continue
            paper = citing[citer]
            passage = ' '.join(p for p in (ctx['before'], '[...]', ctx['after']) if p)
            query = (f'The paper "{clean(paper["title"])}" cites "{title}" in a sentence '
                     f'missing from this passage:\n{passage}\n'
                     'Write the missing sentence, describing the cited paper.')
            episodes[split].append(_episode(
                f'{cid.rsplit("/", 1)[-1]}-{citer}-description', split, query,
                ctx['sentence'], stored, 'public_citance_description', cited=cid,
                cited_title=title, citing_arxiv=citer, citing_license=paper.get('license')))
            break
    used = {r for rows in episodes.values() for e in rows for r in e['required_ids']}
    kept_ids = {cid for cid, _, _ in kept}
    kb = [r for k, r in sorted(records.items()) if k in used]
    for r in kb:
        r['provenance']['cited'] = [c for c in r['provenance']['cited'] if c in kept_ids]
    red = Counter(e['redundancy'] for rows in episodes.values() for e in rows
                  if e['task_family'] == 'public_citance_abstract')
    summary = {
        'min_citances': min_citances, 'max_citances': max_citances,
        'heldout_rate': heldout_rate, 'seed': seed, 'cited_papers': len(kept),
        'records': len(kb), 'record_chars': {
            'mean': round(sum(len(r['text']) for r in kb) / max(len(kb), 1)),
            'max': max((len(r['text']) for r in kb), default=0)},
        'records_serving_several_cited': sum(len(r['provenance']['cited']) > 1 for r in kb),
        'episodes_by_family': dict(Counter(f'{s}:{e["task_family"]}' for s, rows in
                                           episodes.items() for e in rows)),
        'redundancy': dict(sorted(red.items())),
        'abstract_target_tokens': _quantiles([count(e['answer']) for rows in episodes.values()
                                              for e in rows if e['task_family'] ==
                                              'public_citance_abstract']),
        'build_filters': dict(filters)}
    return kb, episodes, summary


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    return {q: values[min(len(values) - 1, int(p * len(values)))]
            for q, p in (('min', 0), ('p10', .1), ('p50', .5), ('p90', .9), ('max', 1))}


def best_context(current: dict | None, candidate: dict) -> bool:
    """Whether ``candidate`` replaces ``current``: fewer citation markers in the citing
    sentence first, then the hash order (``_key``)."""
    if current is None:
        return True
    return (candidate['markers'], candidate['_key']) < (current['markers'], current['_key'])


def collect(rows: Iterable[dict], sample_citer: dict[str, str],
            wanted: dict[str, set[str]], seed: int = 0) -> tuple[dict, Counter]:
    """One pass over citrec rows: per wanted (cited, citing) pair the best context.
    ``sample_citer`` maps sample id to citing arXiv id; ``wanted[cited]`` are the citing
    papers whose contexts are needed."""
    out: dict[str, dict[str, dict]] = {}
    counts: Counter = Counter()
    for row in rows:
        cid = row['label']
        need = wanted.get(cid)
        if not need:
            continue
        citer = sample_citer.get(row['_id'])
        if citer not in need:
            continue
        counts['rows'] += 1
        ctx = context(row['text'], row['marker_offsets'][0][0])
        if ctx is None:
            counts['context_rejected'] += 1
            continue
        ctx['_key'] = _hash(seed, 'context', row['_id'])
        slot = out.setdefault(cid, {})
        if best_context(slot.get(citer), ctx):
            slot[citer] = ctx
    return out, counts
