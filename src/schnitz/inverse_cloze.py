"""Inverse-cloze retrieval episodes for K2 (docs/knowledge-base-stack.md 5.1 step 4;
restart plan, 29 September: the retrieval options queue, item 2).

Retrieval-only transcripts built from the records of an existing task corpus
(``tasks-*``: ``sources.jsonl`` and ``episodes-{split}.jsonl``). Each episode is one
search: the user turn holds a *cue* drawn from the KB's content, the assistant calls
``memory_search()``, and the slot names the records that hold the cue's content. There
is no answer turn: the only assistant tokens are the call (K2 trains the retrieval loss
only, ``l1 train --retrieval-only``). ``task_family`` is ``inverse_cloze``.

The transcripts name the *source corpus's KB* (``kb`` = its corpus name, per domain
for multi-domain corpora), so the L1 bank build stores the same records in the same KB
and span cache as the corpus's own transcripts, and K2 can mix both over one bank. The
records stay whole in the KB: the cue is a query, not a leak (invariant 1 is about
inference reading stored payloads; nothing is removed or re-encoded).

Cue sources, per corpus type (detected from the records):

- ``verbatim`` (overlapping windows of recall-text, r6 passages, synthetic-people bios):
  a whole sentence of the target record (8-40 words; a sentence cut by the window edge
  never), else a 12-24 word span of it. Positives: the target and its redundant
  copies, i.e. every record of the same group (document, article, person) that
  contains the cue verbatim (``alternatives``). ``neutral``: the group's other records
  (windows overlapping only part of the cue or next to it, the title record, the
  person's other records), neither positives nor negatives of the retrieval loss.
- ``citance`` (citance recall): a *different description* of the cited paper, so the
  cue is not a copy of any positive: a sentence or two of its abstract (never stored),
  a held-out citing paper's citance (never stored), or another stored citance of the
  same paper (``sibling``; that record becomes ``neutral``, since matching it is
  lexical). Positives: every stored citance of the paper (one of them named as the
  slot's record, the rest ``alternatives``).
- ``parallel`` (parallel-version recall): one or two verses of a target translation
  that is never stored (the corpus's WEB/BBE episode targets). Positives: every stored
  translation's record covering those verses; ``neutral``: the chapter's other records.

A cue is *ambiguous* when a record outside its group (in the same KB) contains at least
half of its word 8-grams (templated sentences, boilerplate, repeated verses); ambiguous
cues are skipped, so a cue points at its group only. Episodes are split by group, the
source corpus's own split: validation cues come from groups (documents, papers,
chapters, people) that a source validation episode uses and no source training episode
does, so no inverse-cloze training episode draws a cue from them.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import random
import re

FAMILY = 'inverse_cloze'
FORMAT = 3
NGRAM = 8
AMBIGUOUS = 0.5          # share of a cue's 8-grams in an unrelated record = ambiguous
SENTENCE_WORDS = (8, 40)
SPAN_WORDS = (12, 24)
PARAPHRASE_WORDS = 70
CANDIDATES = 3           # cue candidates per planned episode (the first unambiguous wins)
CITANCE_KINDS = {'abstract': 0.4, 'heldout': 0.3, 'sibling': 0.3}

SYSTEM = (
    'You can consult a knowledge base. Call memory_search() whenever you need knowledge; '
    'each result is a memory span you read directly.',
    'A knowledge base holds reference material. Call memory_search() to find the records '
    'a request is about.',
    'When you need knowledge, call memory_search(); results arrive as memory spans.',
)
ROLE = 'Find the stored records the request refers to.'
PROMPTS = {
    'verbatim': ('Find the stored passage that contains this text:\n{cue}',
                 'Look up the record in the knowledge base that says:\n{cue}',
                 'Which stored passage is this from?\n{cue}'),
    'citance': ('Find what other papers say about the work described here:\n{cue}',
                'Look up stored citation contexts of the paper this describes:\n{cue}',
                'Which stored records discuss this work?\n{cue}'),
    'parallel': ('Find stored translations of this passage:\n{cue}',
                 'Look up other versions of these verses in the knowledge base:\n{cue}',
                 'Which stored passages render this text?\n{cue}'),
}
LOSS_POLICY = {'loss_on': 'assistant', 'assistant_parts': ['content', 'tool_calls'],
               'no_loss': ['system', 'user', 'tool'], 'template_generation_tags': True}
HEADER = re.compile(r'^(?:Title|Citing paper): [^\n]*\n(?:(?:Passage|Lead): )?')
PARALLEL_HEADER = re.compile(r'^[^\n]+ \[[^\]\n]+\], [^\n]+\n')
SENTENCE = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])')
VERSE = re.compile(r'^(\d+) (.+)$')


# -- text ---------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', (text or '').lower())


def shingles(tokens: Sequence[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def normalize(text: str) -> str:
    return ' '.join((text or '').split())


def body(record: dict) -> str:
    """A record's content without its header line (title, citing paper, version and
    reference): cues come from content only, never from a header that names the
    group."""
    text = record['text']
    if record.get('kind') == 'parallel_passage':
        return PARALLEL_HEADER.sub('', text, count=1).strip()
    return HEADER.sub('', text, count=1).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(normalize(text)) if s.strip()]


def verses_text(text: str) -> str:
    """Numbered verse lines as running text, without the verse numbers."""
    out = []
    for line in text.split('\n'):
        found = VERSE.match(line.strip())
        out.append(found.group(2) if found else line.strip())
    return normalize(' '.join(out))


def cap_words(text: str, limit: int = PARAPHRASE_WORDS) -> str:
    parts = normalize(text).split(' ')
    return ' '.join(parts[:limit])


# -- groups ---------------------------------------------------------------------------
def corpus_type(records: Iterable[dict]) -> str:
    """``citance``, ``parallel`` or ``verbatim`` from the records' kinds and types."""
    for rec in records:
        prov = rec.get('provenance') or {}
        if rec.get('kind') == 'parallel_passage':
            return 'parallel'
        if prov.get('record_type') == 'citance':
            return 'citance'
    return 'verbatim'


def group_keys(record: dict) -> tuple[str | None, list[str]]:
    """(primary group, every group) of a record. The primary group is where a cue drawn
    from the record belongs (and its split); the others only make the record neutral or
    related there (a roster that lists several people, a citance of several papers).
    ``None``: the record is never a cue source (rosters, registers)."""
    prov = record.get('provenance') or {}
    if record.get('kind') == 'parallel_passage':
        key = f'chapter:{prov.get("book")}:{prov.get("chapter")}'
        return key, [key]
    if prov.get('record_type') == 'citance':
        keys = [f'cited:{c}' for c in prov.get('cited') or ()]
        return (keys[0] if keys else None), keys
    if prov.get('document') is not None:
        key = f'doc:{prov["document"]}'
        return key, [key]
    if 'person' in prov or 'facts' in prov:
        people = sorted({str(f).split(':')[0] for f in prov.get('facts') or ()})
        keys = [f'person:{p}' for p in people]
        if prov.get('person') is not None:
            own = f'person:{prov["person"]}'
            keys = [own] + [k for k in keys if k != own]
            return (own if prov.get('record_type') == 'bio' else None), keys
        return None, keys
    if prov.get('article_title'):
        key = f'article:{record.get("domain", "")}:{prov["article_title"]}'
        return key, [key]
    key = f'record:{record["record_id"]}'
    return key, [key]


def episode_groups(episode: dict, records: Mapping[str, dict]) -> set[str]:
    """The groups a source episode is about (for the split): its document, person,
    cited paper or chapter, else the primary groups of its required records."""
    prov = episode.get('provenance') or {}
    if prov.get('document') is not None:
        return {f'doc:{prov["document"]}'}
    if prov.get('person') is not None:
        return {f'person:{prov["person"]}'}
    if prov.get('cited'):
        cited = prov['cited']
        return {f'cited:{c}' for c in (cited if isinstance(cited, list) else [cited])}
    if prov.get('book') is not None and prov.get('chapter') is not None:
        return {f'chapter:{prov["book"]}:{prov["chapter"]}'}
    out = set()
    for r in episode.get('required_ids') or ():
        if r in records:
            primary, _ = group_keys(records[r])
            if primary:
                out.add(primary)
    return out


def split_groups(episodes: Mapping[str, Iterable[dict]], records: Mapping[str, dict]
                 ) -> tuple[set[str], set[str]]:
    """(train groups, validation groups): validation groups are those a source
    validation episode uses and no source training episode does; every other group
    (including groups no episode uses) is a training group."""
    seen: dict[str, set[str]] = {}
    for split, rows in episodes.items():
        for ep in rows:
            seen.setdefault(split, set()).update(episode_groups(ep, records))
    held = seen.get('validation', set()) - seen.get('train', set())
    every = {group_keys(r)[0] for r in records.values()} - {None}
    return every - held, held


# -- cues -----------------------------------------------------------------------------
@dataclass
class Cue:
    """One planned episode: the cue text and the slot it asks for."""
    text: str
    kind: str                      # sentence, span, abstract, heldout, sibling, version
    group: str
    kb: str
    target: str                    # the slot's record
    alternatives: list[str]        # every positive (the target included)
    neutral: list[str] = field(default_factory=list)
    source: str | None = None      # the stored record the cue text comes from, if any
    meta: dict = field(default_factory=dict)


class Groups:
    """Records by group and KB (a KB is the authorization domain: groups, positives,
    neutral records and the ambiguity check never cross it)."""

    def __init__(self, records: Mapping[str, dict], kb_of: Callable[[str], str]):
        self.records = records
        self.kb_of = kb_of
        self.primary: dict[str, str | None] = {}
        self.keys: dict[str, list[str]] = {}
        self.members: dict[tuple[str, str], list[str]] = {}
        for rid in sorted(records):
            primary, keys = group_keys(records[rid])
            self.primary[rid], self.keys[rid] = primary, keys
            for k in keys:
                self.members.setdefault((kb_of(rid), k), []).append(rid)
        self._bodies: dict[str, str] = {}

    def body(self, rid: str) -> str:
        got = self._bodies.get(rid)
        if got is None:
            got = self._bodies[rid] = normalize(body(self.records[rid]))
        return got

    def of(self, kb: str, group: str) -> list[str]:
        return self.members.get((kb, group), [])

    def related(self, rid: str, kb: str, group: str) -> bool:
        return self.kb_of(rid) == kb and group in self.keys[rid]


def specificity(text: str) -> int:
    """How specific a sentence is, coarsely: names and numbers after its first word
    (a generic sentence such as "They jumped at the chance." points at no record)."""
    parts = normalize(text).split(' ')[1:]
    return min(sum(bool(re.match(r'["\'(]?[A-Z0-9]', p)) for p in parts), 4)


def clean_cue(text: str) -> bool:
    """A paraphrase cue is usable: at least 8 words and no inline LaTeX."""
    return len(text.split()) >= SENTENCE_WORDS[0] and not re.search(r'\\[(\[]|\$', text)


def verbatim_candidates(record: dict, rng: random.Random, count: int = CANDIDATES) -> list[tuple[str, str]]:
    """Up to ``count`` (cue, kind) candidates from one record: whole sentences of
    ``SENTENCE_WORDS`` words (a sentence cut by a window edge is never one), shuffled,
    else one word span of ``SPAN_WORDS`` words from the record's interior."""
    prov = record.get('provenance') or {}
    parts = sentences(body(record))
    if not parts:
        return []
    starts_whole = prov.get('record_type') == 'title' or prov.get('shape') in (None, 'head')
    ends_whole = bool(re.search(r'[.!?]["\')\]]*$', parts[-1]))
    keep = parts[(0 if starts_whole else 1):(len(parts) if ends_whole else len(parts) - 1)]
    lo, hi = SENTENCE_WORDS
    good = [s for s in keep if lo <= len(s.split()) <= hi]
    rng.shuffle(good)
    good.sort(key=specificity, reverse=True)      # stable: ties keep the shuffled order
    out = [(s, 'sentence') for s in good[:count]]
    if out:
        return out
    tokens = normalize(body(record)).split(' ')
    lo, hi = SPAN_WORDS
    if len(tokens) < lo + 4:
        return []
    size = rng.randint(lo, min(hi, len(tokens) - 4))
    start = rng.randint(2, len(tokens) - size - 2) if len(tokens) - size - 2 >= 2 else 0
    return [(' '.join(tokens[start:start + size]), 'span')]


class Ambiguity:
    """Which cues occur (at least ``AMBIGUOUS`` of their word 8-grams) in a record
    outside their group: one pass over the KB's records, indexing only the 8-grams of
    the candidate cues."""

    def __init__(self, groups: Groups, cues: Iterable[Cue], share: float = AMBIGUOUS):
        self.groups, self.share = groups, share
        wanted: set[tuple[str, ...]] = set()
        for cue in cues:
            wanted |= shingles(words(cue.text))
        self.postings: dict[tuple[str, ...], list[str]] = {}
        for rid in groups.records:
            for sh in shingles(words(groups.body(rid))) & wanted:
                self.postings.setdefault(sh, []).append(rid)

    def unrelated_hits(self, cue: Cue) -> list[str]:
        own = shingles(words(cue.text))
        if not own:
            return []
        counts = Counter(r for sh in own for r in self.postings.get(sh, ()))
        need = self.share * len(own)
        return sorted(r for r, n in counts.items() if n >= need
                      and self.groups.kb_of(r) == cue.kb
                      and not self.groups.related(r, cue.kb, cue.group))

    def ambiguous(self, cue: Cue) -> bool:
        return bool(self.unrelated_hits(cue))


def _containing(groups: Groups, kb: str, group: str, text: str) -> list[str]:
    needle = normalize(text)
    return [r for r in groups.of(kb, group) if needle in groups.body(r)]


def plan_verbatim(groups: Groups, chosen: set[str], n: int, rng: random.Random,
                  counts: Counter) -> list[list[Cue]]:
    """Per planned episode its candidate cues: round robin over the chosen groups (a
    random unused record each), so episodes spread over documents."""
    by_group: dict[tuple[str, str], list[str]] = {}
    for rid, primary in groups.primary.items():
        if primary in chosen:
            by_group.setdefault((groups.kb_of(rid), primary), []).append(rid)
    order = sorted(by_group)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(by_group[key])
    plans: list[list[Cue]] = []
    while len(plans) < n and any(by_group[k] for k in order):
        for kb, group in order:
            if len(plans) >= n:
                break
            if not by_group[(kb, group)]:
                continue
            rid = by_group[(kb, group)].pop()
            options = []
            for text, kind in verbatim_candidates(groups.records[rid], rng):
                positives = _containing(groups, kb, group, text)
                if rid not in positives:          # whitespace or header oddities
                    counts['cue_not_in_source'] += 1
                    continue
                neutral = [r for r in groups.of(kb, group) if r not in set(positives)]
                options.append(Cue(text, kind, group, kb, rid, positives, neutral, rid))
            if options:
                plans.append(options)
            else:
                counts['record_without_cue'] += 1
    return plans


def plan_citance(groups: Groups, chosen: set[str], n: int, rng: random.Random,
                 abstracts: Mapping[str, str], heldout: Mapping[str, list[str]],
                 counts: Counter, kinds: Mapping[str, float] = CITANCE_KINDS
                 ) -> list[list[Cue]]:
    """Paraphrase cues for citance groups (cited papers): abstract sentences, a held-out
    citance or a sibling citance (made neutral); every stored citance of the paper is a
    positive."""
    keys = sorted((kb, g) for (kb, g) in groups.members if g in chosen
                  and len(groups.members[(kb, g)]) >= 2)
    rng.shuffle(keys)
    plans: list[list[Cue]] = []
    rounds = 0
    while len(plans) < n and keys and rounds < 64:
        rounds += 1
        for kb, group in keys:
            if len(plans) >= n:
                break
            members = groups.of(kb, group)
            cited = group.split(':', 1)[1]
            options = []
            for _ in range(CANDIDATES):
                avail = {k: w for k, w in kinds.items() if w > 0 and (
                    (k == 'abstract' and cited in abstracts)
                    or (k == 'heldout' and heldout.get(cited))
                    or (k == 'sibling' and len(members) >= 3))}
                if not avail:
                    break
                kind = rng.choices(sorted(avail), [avail[k] for k in sorted(avail)])[0]
                source, neutral = None, []
                if kind == 'abstract':
                    parts = sentences(abstracts[cited])
                    start = rng.randrange(len(parts))
                    text = parts[start]
                    if len(text.split()) < 12 and start + 1 < len(parts):
                        text += ' ' + parts[start + 1]
                elif kind == 'heldout':
                    text = rng.choice(heldout[cited])
                else:
                    source = rng.choice(members)
                    parts = sentences(body(groups.records[source]))
                    text = parts[len(parts) // 2] if parts else body(groups.records[source])
                    if len(text.split()) < SENTENCE_WORDS[0]:
                        text = body(groups.records[source])
                    neutral = [source]
                text = cap_words(text)
                if not clean_cue(text):
                    counts[f'cue_unclean_{kind}'] += 1
                    continue
                positives = [r for r in members if r not in set(neutral)]
                target = rng.choice(positives)
                options.append(Cue(text, kind, group, kb, target, positives, neutral, source,
                                   {'cited': cited}))
            if options:
                plans.append(options)
    return plans


def plan_parallel(groups: Groups, episodes: Sequence[dict], n: int, rng: random.Random,
                  kb: str, counts: Counter) -> list[list[Cue]]:
    """Cues from the never-stored target translation of the source episodes (their
    answers, one numbered verse per line): one or two verses; positives are every
    stored record covering them, ``neutral`` the chapter's other records."""
    rows = list(episodes)
    rng.shuffle(rows)
    plans: list[list[Cue]] = []
    for ep in rows:
        if len(plans) >= n:
            break
        prov = ep.get('provenance') or {}
        group = f'chapter:{prov.get("book")}:{prov.get("chapter")}'
        verses = [(int(m.group(1)), m.group(2)) for m in
                  (VERSE.match(line.strip()) for line in (ep.get('answer') or '').split('\n'))
                  if m]
        members = groups.of(kb, group)
        options = []
        for _ in range(CANDIDATES):
            if not verses:
                break
            k = rng.randrange(len(verses))
            chosen = [verses[k]]
            if len(chosen[0][1].split()) < 10 and k + 1 < len(verses):
                chosen.append(verses[k + 1])
            a, b = chosen[0][0], chosen[-1][0]
            positives = []
            for r in members:
                p = groups.records[r].get('provenance') or {}
                if int(p.get('verse_start', 0)) <= a and b <= int(p.get('verse_end', -1)):
                    positives.append(r)
            if not positives:
                counts['verses_not_covered'] += 1
                continue
            neutral = [r for r in members if r not in set(positives)]
            options.append(Cue(cap_words(' '.join(v for _, v in chosen)), 'version', group,
                               kb, rng.choice(positives), positives, neutral, None,
                               {'target_version': prov.get('target_version'),
                                'verses': [a, b], 'source_episode': ep['episode_id']}))
        if options:
            plans.append(options)
    return plans


def choose(plans: Sequence[Sequence[Cue]], ambiguity: Ambiguity | None, n: int,
           counts: Counter) -> list[Cue]:
    """The first unambiguous candidate of each plan, up to ``n`` episodes."""
    out = []
    for options in plans:
        if len(out) >= n:
            break
        pick = None
        for cue in options:
            if ambiguity is not None and ambiguity.ambiguous(cue):
                counts[f'ambiguous_{cue.kind}'] += 1
                continue
            pick = cue
            break
        if pick is None:
            counts['plan_without_unambiguous_cue'] += 1
            continue
        out.append(pick)
    return out


# -- transcripts ----------------------------------------------------------------------
def episode_id(corpus: str, split: str, n: int, cue: Cue) -> str:
    digest = hashlib.sha256(f'{cue.group}\0{cue.target}\0{cue.text}'.encode()).hexdigest()[:10]
    return f'{FAMILY}-{corpus}-{split}-{n:05d}-{digest}'


def transcript(cue: Cue, *, corpus: str, split: str, n: int, mode: str, rng: random.Random,
               memory_tools: list[dict], kind: str, query_time: int = 2,
               source_corpus: str = '') -> dict:
    """A format-3 memory transcript with one search: system, the cue in the user turn,
    ``memory_search()`` without arguments, and the slot (no answer turn)."""
    system = f'{SYSTEM[rng.randrange(len(SYSTEM))]} {ROLE}'
    user = PROMPTS[mode][rng.randrange(len(PROMPTS[mode]))].format(cue=cue.text)
    alternatives = list(dict.fromkeys([cue.target, *cue.alternatives]))
    flags = {'space_hint': 'fine', 'alternatives': alternatives}
    neutral = [r for r in dict.fromkeys(cue.neutral) if r not in set(alternatives)]
    if neutral:
        flags['neutral'] = neutral
    slot = {'kb': cue.kb, 'record_ids': [cue.target], **flags}
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': '',
                 'tool_calls': [{'type': 'function',
                                 'function': {'name': 'memory_search', 'arguments': {}}}]},
                {'role': 'tool', 'name': 'memory_search', 'content': {'slot': slot}}]
    site = {'message': 2, 'call': 0, 'result': 3, 'kind': kind, 'records': 1,
            'record_ids': [cue.target], 'step': 0, 'trigger': FAMILY, **flags}
    return {'episode_id': episode_id(corpus, split, n, cue), 'kb': cue.kb, 'split': split,
            'format': FORMAT, 'task_family': FAMILY, 'messages': messages,
            'tools': list(memory_tools), 'answer': '', 'verify': None,
            'provenance': {'dataset': FAMILY, 'source_corpus': source_corpus,
                           'corpus_type': mode, 'cue_kind': cue.kind, 'group': cue.group,
                           'cue_record': cue.source, 'split': split,
                           'source_query_time': query_time, **cue.meta},
            'search_sites': [site], 'write_sites': [], 'loss_mask': LOSS_POLICY}


def build(records: Mapping[str, dict], episodes: Mapping[str, Sequence[dict]],
          kb_of: Callable[[str], str], sizes: Mapping[str, int], *, seed: int = 0,
          corpus: str = 'corpus', memory_tools: list[dict] | None = None,
          query_time: int = 2, source_corpus: str = '',
          citance_kinds: Mapping[str, float] = CITANCE_KINDS
          ) -> tuple[dict[str, list[dict]], dict]:
    """Inverse-cloze transcripts per split (``sizes``: split -> episode count) from a
    corpus's records (record id -> record, only records created before ``query_time``
    are used) and its source episodes (split -> rows). Returns (rows per split,
    summary)."""
    from schnitz.span_tokens import MEMORY_TOOLS
    tools = memory_tools if memory_tools is not None else MEMORY_TOOLS
    usable = {r: rec for r, rec in records.items() if int(rec.get('created_at', 0)) < query_time}
    mode = corpus_type(usable.values())
    groups = Groups(usable, kb_of)
    train_groups, held = split_groups(episodes, usable)
    chosen = {'train': train_groups, 'validation': held}
    counts: Counter = Counter()
    abstracts: dict[str, str] = {}
    heldout: dict[str, list[str]] = {}
    if mode == 'citance':
        for rows in episodes.values():
            for ep in rows:
                prov = ep.get('provenance') or {}
                cited = prov.get('cited')
                if not cited or not ep.get('answer'):
                    continue
                if ep.get('task_family') == 'public_citance_abstract':
                    abstracts[cited] = ep['answer']
                elif ep.get('task_family') == 'public_citance_description':
                    heldout.setdefault(cited, []).append(ep['answer'])
    out: dict[str, list[dict]] = {}
    summary: dict = {'corpus_type': mode, 'groups': {'train': len(train_groups),
                                                      'validation': len(held)}}
    for split, n in sizes.items():
        rng = random.Random(f'{seed}:{corpus}:{split}')
        if mode == 'verbatim':
            plans = plan_verbatim(groups, chosen[split], int(n * 1.5) + 8, rng, counts)
        elif mode == 'citance':
            plans = plan_citance(groups, chosen[split], int(n * 1.5) + 8, rng, abstracts,
                                 heldout, counts, citance_kinds)
        else:
            src = [ep for ep in episodes.get(split, ())
                   if episode_groups(ep, usable) <= chosen[split]]
            kbs = sorted({kb_of(r) for r in usable})
            plans = plan_parallel(groups, src, int(n * 1.5) + 8, rng, kbs[0], counts)
        ambiguity = Ambiguity(groups, (c for options in plans for c in options))
        cues = choose(plans, ambiguity, n, counts)
        rows = []
        for i, cue in enumerate(cues):
            kind = usable[cue.target].get('kind', 'passage')
            rows.append(transcript(cue, corpus=corpus, split=split, n=i, mode=mode,
                                   rng=random.Random(f'{seed}:{corpus}:{split}:{i}'),
                                   memory_tools=tools, kind=kind, query_time=query_time,
                                   source_corpus=source_corpus))
        out[split] = rows
        summary[split] = {
            'episodes': len(rows), 'requested': n,
            'cue_kinds': dict(Counter(r['provenance']['cue_kind'] for r in rows)),
            'groups_used': len({r['provenance']['group'] for r in rows}),
            'alternatives_median': _median([len(r['search_sites'][0]['alternatives'])
                                            for r in rows]),
            'neutral_median': _median([len(r['search_sites'][0].get('neutral') or ())
                                       for r in rows]),
            'cue_words_median': _median([len(r['messages'][1]['content'].split('\n', 1)[-1]
                                             .split()) for r in rows])}
    overlap = {r['provenance']['group'] for r in out.get('train', ())} & \
        {r['provenance']['group'] for r in out.get('validation', ())}
    if overlap:
        raise AssertionError(f'{len(overlap)} groups in both splits')
    summary['filters'] = dict(sorted(counts.items()))
    return out, summary


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
