"""Parallel-version recall: a knowledge task with a highly redundant KB.

The knowledge base holds passages of many public-domain Bible translations, all
verse-aligned; an episode asks for a passage in a *target* translation that is never
stored. The target text has high entropy for a model on its own (the exact wording of a
modern, little-memorized translation) but is nearly determined by the stored versions
of the same verses. ``redundancy`` (stored versions covering the target) is recorded per
episode, as is how close the nearest stored version comes to the target.

Data: the CSV exports of scrollmapper/bible_databases (``Book,Chapter,Verse,Text``;
the legacy 2024 branch ``id,b,c,v,t`` for the World English Bible). This module holds
the pure functions (cleaning, alignment, chunking, episodes); the subcommand
``parallel-recall`` of ``scripts/prepare_task_corpora.py`` reads the files and writes the
corpus.

Guarantees (tested in ``tests/test_parallel_recall.py``):

- target versions are never stored, so no record contains a target text;
- every episode has at least ``min_versions`` covering versions; each version's
  covering records form one sufficient group, ``required_ids`` is their union;
- the train/validation split is by chapter: no chapter has episodes in both splits.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import difflib
import hashlib
from pathlib import Path
import random
import re
from typing import Callable

RECORD_CHARS = 1500          # as prepare_task_corpora.RECORD_CHARS
DOMAIN = 'parallel-recall'
FAMILY = 'parallel_recall'
KIND = 'parallel_passage'


@dataclass(frozen=True)
class Version:
    code: str
    title: str
    language: str
    licence: str        # as declared in scrollmapper's ``translations`` table
    file: str           # relative to the raw root
    legacy: bool = False
    note: str = ''


M = 'master/formats/csv/'
PD = 'Public Domain'
# Public-domain (or CC0) versions obtained; licence strings are scrollmapper's
# ``translations.license`` rows (legacy branch: ``bible_version_key.csv``).
VERSIONS: dict[str, Version] = {v.code: v for v in (
    Version('WEB', 'World English Bible', 'en', PD, 'branch-2024/csv/t_web.csv', legacy=True,
            note='legacy 2024 branch; footnotes in braces removed'),
    Version('BBE', 'Bible in Basic English', 'en', PD, M + 'BBE.csv'),
    Version('KJV', 'King James Version (Pure Cambridge Edition)', 'en', PD, M + 'KJVPCE.csv',
            note='KJVPCE; the Strong\'s-tagged KJV module is GPL and not used'),
    Version('ASV', 'American Standard Version', 'en', PD, M + 'ASV.csv'),
    Version('YLT', "Young's Literal Translation", 'en', PD, M + 'YLT.csv'),
    Version('Darby', 'Darby Bible', 'en', PD, M + 'Darby.csv'),
    Version('Webster', 'Webster Bible', 'en', PD, M + 'Webster.csv'),
    Version('RWebster', 'Revised Webster Version', 'en', PD, M + 'RWebster.csv'),
    Version('DRC', 'Douay-Rheims Bible, Challoner Revision', 'en', PD, M + 'DRC.csv'),
    Version('CPDV', 'Catholic Public Domain Version', 'en', PD, M + 'CPDV.csv'),
    Version('ACV', 'A Conservative Version', 'en', PD, M + 'ACV.csv'),
    Version('BSB', 'Berean Standard Bible', 'en', 'Creative Commons CC0', M + 'BSB.csv'),
    Version('Geneva', 'Geneva Bible (1599)', 'en', PD, M + 'Geneva1599.csv'),
    Version('JPS', 'Jewish Publication Society Old Testament (1917)', 'en', PD, M + 'JPS.csv'),
    Version('NHEB-JE', 'New Heart English Bible, Jehovah Edition', 'en', PD, M + 'NHEBJE.csv'),
    Version('NHEB-ME', 'New Heart English Bible, Messianic Edition', 'en', PD, M + 'NHEBME.csv'),
    Version('Noyes', 'Noyes Translation (1869)', 'en', PD, M + 'Noyes.csv'),
    Version('OEB', 'Open English Bible', 'en', 'Creative Commons: CC0', M + 'OEB.csv'),
    Version('RNKJV', 'Restored Name King James Version', 'en', PD, M + 'RNKJV.csv'),
    Version('Rotherham', 'Rotherham Emphasised Bible', 'en', PD, M + 'Rotherham.csv'),
    Version('Twenty', 'Twentieth Century New Testament', 'en', PD, M + 'Twenty.csv'),
    Version('Tyndale', 'Tyndale Bible (1525/1530)', 'en', PD, M + 'Tyndale.csv'),
    Version('UKJV', 'Updated King James Version', 'en', PD, M + 'UKJV.csv'),
    Version('Anderson', "Anderson's New Testament (1864)", 'en', PD, M + 'Anderson.csv'),
    Version('Haweis', 'Haweis New Testament (1795)', 'en', PD, M + 'Haweis.csv'),
    Version('FreCrampon', 'Bible Crampon (1923)', 'fr', PD, M + 'FreCrampon.csv'),
    Version('FreJND', 'Bible J.N. Darby (French)', 'fr', PD, M + 'FreJND.csv'),
    Version('FreBDM1744', 'Bible David Martin (1744)', 'fr', PD, M + 'FreBDM1744.csv'),
    Version('FrePGR', 'Bible Perret-Gentil et Rilliet', 'fr', PD, M + 'FrePGR.csv'),
    Version('FreSynodale1921', 'Version Synodale (1921), NT and Psalms', 'fr', PD,
            M + 'FreSynodale1921.csv'),
    Version('FreOltramare1874', 'Nouveau Testament Oltramare (1874)', 'fr', PD,
            M + 'FreOltramare1874.csv'),
    Version('FreStapfer1889', 'Nouveau Testament Stapfer (1889)', 'fr', PD,
            M + 'FreStapfer1889.csv'),
    Version('FreGeneve1669', 'Nouveau Testament de Geneve (1669)', 'fr', PD,
            M + 'FreGeneve1669.csv'),
    Version('GerElb1871', 'Elberfelder (1871)', 'de', PD, M + 'GerElb1871.csv'),
    Version('GerElb1905', 'Unrevidierte Elberfelder (1905)', 'de', PD, M + 'GerElb1905.csv'),
    Version('GerTextbibel', 'Textbibel (1906)', 'de', PD, M + 'GerTextbibel.csv'),
    Version('GerMenge', 'Menge-Bibel (1939)', 'de', PD, M + 'GerMenge.csv'),
    Version('GerTafel', 'Tafelbibel (1911)', 'de', PD, M + 'GerTafel.csv'),
    Version('GerAlbrecht', 'Albrecht NT und Psalmen', 'de', PD, M + 'GerAlbrecht.csv'),
    Version('GerBoLut', 'Luther 1545 (moderne Rechtschreibung)', 'de', PD, M + 'GerBoLut.csv'),
    Version('SpaRV', 'Reina-Valera (1909)', 'es', PD, M + 'SpaRV.csv'),
    Version('SpaRV1865', 'Reina-Valera (1865)', 'es', PD, M + 'SpaRV1865.csv'),
    Version('ChiUn', 'Chinese Union Version (traditional)', 'zh', PD, M + 'ChiUn.csv'),
    Version('ChiUnL', 'Chinese Union Version (Wenli)', 'zh', PD, M + 'ChiUnL.csv'),
    Version('JapBungo', 'Japanese Meiji/Taisho Bungo-yaku', 'ja', PD, M + 'JapBungo.csv'),
    Version('JapKougo', 'Japanese Kougo-yaku (1954/1955)', 'ja', PD, M + 'JapKougo.csv'),
    Version('JapDenmo', 'Japanese Denmo', 'ja', PD, M + 'JapDenmo.csv'),
)}
REFERENCE = 'KJV'     # versification reference for the alignment checks
# Book names of the scrollmapper files -> display names
DISPLAY = {'I ': '1 ', 'II ': '2 ', 'III ': '3 '}
DISPLAY_FULL = {'Revelation of John': 'Revelation', 'Song of Solomon': 'Song of Solomon'}

_CJK = r'\u3000-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef'


def display_book(book: str) -> str:
    if book in DISPLAY_FULL:
        return DISPLAY_FULL[book]
    for roman, arabic in DISPLAY.items():
        if book.startswith(roman):
            return arabic + book[len(roman):]
    return book


def clean_verse(text: str, language: str = 'en') -> str:
    """Verse text without footnotes (``{...}``), tags (``<...>``), italics brackets and
    extra whitespace; CJK text loses the word-segmentation spaces of some modules."""
    text = re.sub(r'\{[^{}]*\}', '', text or '')
    text = re.sub(r'<[^<>]*>', '', text)
    text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    text = re.sub(r'\s+', ' ', text).strip()
    if language in ('zh', 'ja'):
        text = re.sub(rf'(?<=[{_CJK}]) (?=[{_CJK}])', '', text)
        text = re.sub(rf' (?=[{_CJK}，。、；：！？「」])|(?<=[，。、；：！？「」]) ', '', text)
    return text


Verses = dict[tuple[str, int, int], str]       # (book, chapter, verse) -> cleaned text


def load_version(root: Path, version: Version, book_order: list[str] | None = None
                 ) -> tuple[Verses, Counter]:
    """Cleaned non-empty verses and the raw verse count per (book, chapter)."""
    verses: Verses = {}
    counts: Counter = Counter()
    with (root / version.file).open(encoding='utf-8', newline='') as handle:
        rows = csv.reader(handle)
        next(rows)
        for row in rows:
            if version.legacy:
                if book_order is None:
                    raise ValueError('A legacy file needs the book order')
                book, chapter, verse, text = book_order[int(row[1]) - 1], row[2], row[3], row[4]
            else:
                book, chapter, verse, text = row[0], row[1], row[2], row[3]
            key = (book, int(chapter), int(verse))
            counts[key[:2]] += 1
            text = clean_verse(text, version.language)
            if text:
                verses[key] = text
    return verses, counts


def _words(text: str) -> list[str]:
    return re.findall(r'[a-z]+', text.lower())


def lexically_aligned(verses: Verses, reference: Verses, book: str, chapter: int,
                      n: int) -> bool:
    """English only: in most verses the reference verse with the same number is the
    closest of its neighbours (catches shifted numbering with equal verse counts)."""
    hits = total = 0
    for v in range(1, n + 1):
        own = set(_words(verses.get((book, chapter, v), '')))
        if not own:
            continue
        scores = {}
        for u in (v - 1, v, v + 1):
            ref = set(_words(reference.get((book, chapter, u), '')))
            if ref:
                scores[u] = len(own & ref) / len(own | ref)
        if scores:
            total += 1
            hits += max(scores, key=scores.get) == v
    return total == 0 or hits >= 0.6 * total


def aligned_chapters(version: Version, verses: Verses, counts: Counter, ref_counts: Counter,
                     ref_verses: Verses, book_share: float = 0.9) -> set[tuple[str, int]]:
    """Chapters whose versification matches the reference: the book's chapters agree
    on their verse counts for at least ``book_share`` (else the whole book is dropped, e.g.
    Vulgate-numbered Psalms), the chapter's count agrees, and for English the text
    lines up verse by verse."""
    by_book: dict[str, list[tuple[str, int]]] = {}
    for key in counts:
        by_book.setdefault(key[0], []).append(key)
    keep = set()
    for book, chapters in by_book.items():
        ref_chapters = [k for k in ref_counts if k[0] == book]
        if not ref_chapters:
            continue
        same = [k for k in chapters if counts[k] == ref_counts.get(k)]
        if len(same) < book_share * len(ref_chapters):
            continue
        for key in same:
            if version.language == 'en' and version.code != REFERENCE and \
                    not lexically_aligned(verses, ref_verses, *key, ref_counts[key]):
                continue
            keep.add(key)
    return keep


def record_id(domain: str, body: str) -> str:
    """As ``public_corpus_common.record_id``."""
    return hashlib.sha256(f'{domain}\0{body}'.encode()).hexdigest()[:32]


def header(version: Version, book: str, chapter: int, first: int, last: int) -> str:
    ref = f'{display_book(book)} {chapter}:{first}' + (f'-{last}' if last != first else '')
    return f'{version.title} [{version.code}], {ref}'


def verse_lines(verses: Verses, book: str, chapter: int, first: int, last: int) -> str:
    return '\n'.join(f'{v} {verses[(book, chapter, v)]}' for v in range(first, last + 1))


def chunk_version(version: Version, verses: Verses, chapters: set[tuple[str, int]],
                  seed: int, *, min_verses: int = 4, max_verses: int = 10,
                  limit: int = RECORD_CHARS, domain: str = DOMAIN) -> list[dict]:
    """KB records: runs of ``min_verses``-``max_verses`` consecutive non-empty verses of one
    chapter, at most ``limit`` characters (fewer verses when long; a single longer verse
    is cut). Boundaries are drawn per version, so versions do not share them."""
    rng = random.Random(f'{seed}\0chunks\0{version.code}')
    by_chapter: dict[tuple[str, int], list[int]] = {}
    for (book, chapter, verse) in verses:
        if (book, chapter) in chapters:
            by_chapter.setdefault((book, chapter), []).append(verse)
    records = []
    for (book, chapter) in sorted(by_chapter):
        numbers = sorted(by_chapter[(book, chapter)])
        i = 0
        while i < len(numbers):
            want = rng.randint(min_verses, max_verses)
            j = i + 1
            while j < len(numbers) and j - i < want and numbers[j] == numbers[j - 1] + 1:
                trial = verse_lines(verses, book, chapter, numbers[i], numbers[j])
                if len(header(version, book, chapter, numbers[i], numbers[j])) + 1 + \
                        len(trial) > limit:
                    break
                j += 1
            first, last = numbers[i], numbers[j - 1]
            text = header(version, book, chapter, first, last) + '\n' + \
                verse_lines(verses, book, chapter, first, last)
            text = text[:limit]
            records.append({
                'record_id': record_id(domain, text), 'text': text, 'domain': domain,
                'created_at': 1, 'kind': KIND,
                'provenance': {'dataset': domain, 'version': version.code,
                               'language': version.language, 'licence': version.licence,
                               'book': book, 'chapter': chapter, 'verse_start': first,
                               'verse_end': last}})
            i = j
    return records


def approx_tokens(text: str) -> int:
    """Token estimate without a tokenizer (about 4 characters per token in English)."""
    return max(1, round(len(text) / 4))


class Index:
    """Records by (version, book, chapter), for covering lookups."""

    def __init__(self, records: list[dict]):
        self.by_chapter: dict[tuple[str, str, int], list[dict]] = {}
        for rec in records:
            p = rec['provenance']
            self.by_chapter.setdefault((p['version'], p['book'], p['chapter']), []).append(rec)

    def covering(self, version: str, book: str, chapter: int, first: int, last: int
                 ) -> list[dict] | None:
        """The version's records overlapping verses ``first``-``last``, or None when some
        of those verses are not stored."""
        recs = [r for r in self.by_chapter.get((version, book, chapter), [])
                if r['provenance']['verse_start'] <= last and r['provenance']['verse_end'] >= first]
        have = {v for r in recs for v in range(r['provenance']['verse_start'],
                                               r['provenance']['verse_end'] + 1)}
        if not set(range(first, last + 1)) <= have:
            return None
        return sorted(recs, key=lambda r: r['provenance']['verse_start'])


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _words(a), _words(b), autojunk=False).ratio()


def split_chapters(chapters: list[tuple[str, int]], validation_share: float, seed: int
                   ) -> dict[tuple[str, int], str]:
    order = sorted(chapters, key=lambda k: hashlib.sha256(
        f'{seed}\0split\0{k[0]}\0{k[1]}'.encode()).hexdigest())
    cut = round(len(order) * validation_share)
    return {k: ('validation' if i < cut else 'train') for i, k in enumerate(order)}


def make_episode(target: Version, target_verses: Verses, stored: dict[str, Version],
                 stored_verses: dict[str, Verses], index: Index, book: str, chapter: int,
                 first: int, last: int, split: str, mode: str, rng: random.Random, *,
                 min_versions: int, domain: str = DOMAIN) -> tuple[dict | None, str | None]:
    answer = verse_lines(target_verses, book, chapter, first, last)
    groups, texts = [], {}
    for code in stored:
        recs = index.covering(code, book, chapter, first, last)
        if recs:
            groups.append((code, recs))
            texts[code] = verse_lines(stored_verses[code], book, chapter, first, last)
    if len(groups) < min_versions:
        return None, 'too_few_versions'
    for code, recs in groups:       # the exact target, in one record or across several
        if texts[code] == answer or any(answer in r['text'] for r in recs):
            return None, 'target_in_kb'
    rng.shuffle(groups)
    ref = f'{display_book(book)} {chapter}:{first}' + (f'-{last}' if last > first else '')
    n = last - first + 1
    if mode == 'continuation':
        prev = target_verses[(book, chapter, first - 1)]
        query = (f'In the {target.title} ({target.code}) one verse reads: "{prev}"\n'
                 f'Give the next {n} verse{"s" if n > 1 else ""} in the {target.code} wording, '
                 'one per line with verse numbers.')
    else:
        query = (f'Give {ref} in the {target.title} ({target.code}) wording, one verse per '
                 'line with verse numbers.')
    required = [r for _, recs in groups for r in recs]
    required_ids = list(dict.fromkeys(r['record_id'] for r in required))
    verbatim = sum(any(stored_verses[c].get((book, chapter, v)) == target_verses[(book, chapter, v)]
                       for c, _ in groups) for v in range(first, last + 1))
    sims = {c: similarity(answer, t) for c, t in texts.items()}
    nearest = max(sims, key=sims.get)
    languages = Counter(stored[c].language for c, _ in groups)
    ident = f'{split}-{target.code}-{book.replace(" ", "_")}-{chapter}-{first}-{last}-{mode[0]}'
    return {
        'episode_id': f'{domain}-{ident}', 'environment': f'{domain}-{split}',
        'query': query, 'answer': answer, 'query_time': 2,
        'required_ids': required_ids,
        'sufficient_groups': [[r['record_id'] for r in recs] for _, recs in groups],
        'support_annotation': 'verified', 'task_family': FAMILY,
        'supports': [{'record_id': r['record_id'], 'text': r['text'], 'created_at': r['created_at'],
                      'kind': r['kind']} for r in {r['record_id']: r for r in required}.values()],
        'verify': {'type': 'exact', 'answer': answer},
        'redundancy': len(groups),
        'provenance': {'dataset': domain, 'domain': domain, 'split': split,
                       'target_version': target.code, 'target_licence': target.licence,
                       'book': book, 'chapter': chapter, 'verse_start': first, 'verse_end': last,
                       'query_mode': mode, 'covering_versions': [c for c, _ in groups],
                       'covering_languages': dict(languages),
                       'verbatim_verses': verbatim, 'verses': n,
                       'nearest_version': nearest, 'nearest_similarity': round(sims[nearest], 4),
                       'mean_similarity': round(sum(sims.values()) / len(sims), 4)},
    }, None


def build(root: Path, *, stored_codes: list[str], target_codes: list[str], seed: int = 0,
          train_episodes: int = 4000, validation_episodes: int = 300,
          validation_share: float = 0.1, min_versions: int = 3, min_tokens: int = 64,
          max_tokens: int = 200, max_verses: int = 12, per_chapter: int = 3,
          continuation_rate: float = 0.5, count_tokens: Callable[[str], int] = approx_tokens,
          domain: str = DOMAIN) -> dict:
    """Records, episodes per split and summary statistics. Target versions are removed
    from ``stored_codes`` so the KB never holds a target text."""
    unknown = [c for c in [*stored_codes, *target_codes] if c not in VERSIONS]
    if unknown:
        raise ValueError(f'Unknown versions: {unknown}')
    stored_codes = [c for c in dict.fromkeys(stored_codes) if c not in target_codes]
    ref_version = VERSIONS[REFERENCE]
    ref_verses, ref_counts = load_version(root, ref_version)
    book_order = list(dict.fromkeys(k[0] for k in ref_counts))   # file order
    loaded: dict[str, tuple[Verses, Counter]] = {}
    for code in dict.fromkeys([*stored_codes, *target_codes]):
        loaded[code] = (ref_verses, ref_counts) if code == REFERENCE else \
            load_version(root, VERSIONS[code], book_order)
    aligned = {code: aligned_chapters(VERSIONS[code], *loaded[code], ref_counts, ref_verses)
               for code in loaded}
    records, stats = [], {}
    for code in stored_codes:
        recs = chunk_version(VERSIONS[code], loaded[code][0], aligned[code], seed, domain=domain)
        records += recs
        stats[code] = {'records': len(recs), 'aligned_chapters': len(aligned[code]),
                       'verses': len(loaded[code][0])}
    index = Index(records)
    stored = {c: VERSIONS[c] for c in stored_codes}
    stored_verses = {c: loaded[c][0] for c in stored_codes}
    chapters = sorted({k for c in target_codes for k in aligned[c]})
    split_of = split_chapters(chapters, validation_share, seed)
    rng = random.Random(f'{seed}\0episodes')
    wanted = {'train': train_episodes, 'validation': validation_episodes}
    episodes: dict[str, list[dict]] = {'train': [], 'validation': []}
    rejects: dict[str, Counter] = {'train': Counter(), 'validation': Counter()}
    numbers_of: dict[str, dict[tuple[str, int], list[int]]] = {}
    for code in target_codes:
        for (b, c, v) in sorted(loaded[code][0]):
            numbers_of.setdefault(code, {}).setdefault((b, c), []).append(v)
    order = list(chapters)
    rng.shuffle(order)
    for book, chapter in order:
        split = split_of[(book, chapter)]
        if len(episodes[split]) >= wanted[split]:
            continue
        for code in target_codes:
            if (book, chapter) not in aligned[code]:
                continue
            tv = loaded[code][0]
            numbers = numbers_of[code].get((book, chapter), [])
            taken: set[int] = set()
            made = 0
            for start in rng.sample(numbers, len(numbers)):
                if made >= per_chapter or len(episodes[split]) >= wanted[split]:
                    break
                if start in taken:
                    continue
                goal = rng.randint(min_tokens, (min_tokens + max_tokens) // 2)
                last = start
                text = verse_lines(tv, book, chapter, start, last)
                while count_tokens(text) < goal and last - start + 1 < max_verses and \
                        (book, chapter, last + 1) in tv and last + 1 not in taken:
                    last += 1
                    text = verse_lines(tv, book, chapter, start, last)
                tokens = count_tokens(text)
                if not min_tokens <= tokens <= max_tokens:
                    rejects[split]['target_length'] += 1
                    continue
                mode = 'continuation' if start > 1 and (book, chapter, start - 1) in tv and \
                    rng.random() < continuation_rate else 'reference'
                ep, why = make_episode(VERSIONS[code], tv, stored, stored_verses, index, book,
                                       chapter, start, last, split, mode, rng,
                                       min_versions=min_versions, domain=domain)
                if ep is None:
                    rejects[split][why] += 1
                    continue
                ep['provenance']['target_tokens'] = tokens
                episodes[split].append(ep)
                taken |= set(range(start - 1, last + 2))
                made += 1
    return {'records': records, 'episodes': episodes, 'split_of': split_of,
            'stored': stats, 'rejected': {k: dict(v) for k, v in rejects.items()},
            'stored_codes': stored_codes, 'target_codes': target_codes}


def summary(result: dict) -> dict:
    out = {}
    for split, rows in result['episodes'].items():
        red = Counter(r['redundancy'] for r in rows)
        sims = sorted(r['provenance']['nearest_similarity'] for r in rows)
        tokens = sorted(r['provenance']['target_tokens'] for r in rows)
        out[split] = {
            'episodes': len(rows),
            'by_target': dict(Counter(r['provenance']['target_version'] for r in rows)),
            'by_mode': dict(Counter(r['provenance']['query_mode'] for r in rows)),
            'redundancy': dict(sorted(red.items())),
            'chapters': len({(r['provenance']['book'], r['provenance']['chapter']) for r in rows}),
            'target_tokens_quartiles': [tokens[len(tokens) * q // 4] for q in (0, 1, 2, 3)]
            + [tokens[-1]] if tokens else [],
            'nearest_similarity_quartiles': [sims[len(sims) * q // 4] for q in (0, 1, 2, 3)]
            + [sims[-1]] if sims else [],
            'nearest_version': dict(Counter(r['provenance']['nearest_version']
                                            for r in rows).most_common(8)),
            'nearest_similarity_median_by_target': {
                t: sorted(r['provenance']['nearest_similarity'] for r in rows
                          if r['provenance']['target_version'] == t)[
                    sum(r['provenance']['target_version'] == t for r in rows) // 2]
                for t in sorted({r['provenance']['target_version'] for r in rows})},
            'verbatim_verse_share': round(sum(r['provenance']['verbatim_verses'] for r in rows)
                                          / max(1, sum(r['provenance']['verses'] for r in rows)), 4),
        }
    return out
