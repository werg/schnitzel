"""MuSiQue, 2WikiMultihopQA and HoVer as multi-hop answer-bearing episodes.

Each dataset becomes its own domain (``musique``, ``2wikimultihopqa``, ``hover``)
through ``public_corpus_common``. Upstream train becomes ``train`` and upstream
dev becomes ``validation`` (upstream test labels are hidden). Every source is a
<=700-character sentence-packed chunk of one provided paragraph; an episode stores
all of its gold chunks plus at most ``--distractors`` distractor chunks from the
same example's provided context. All gold chunks are required together
(``all_required``). Everything is static: sources are created at 1 and queries
asked at 2. Answers are supervised targets only, never query text.

MuSiQue (full v1.0). Support is paragraph-level. A supporting paragraph that fits
one chunk is gold; otherwise the gold chunks are those containing that hop's
answer (the decomposition step whose ``paragraph_support_idx`` names it, or the
final answer and aliases), falling back to every chunk of the paragraph. Following
the official MuSiQue guidance, episodes are dropped when a single-hop component
matches (normalised text, with ``#k`` placeholders filled by earlier hop answers)
a question listed in ``dev_test_singlehop_questions_v1.0.json`` for a seed dataset
our banks train on (``--musique-leak-seeds``, default ``squad2``). The unanswerable
halves of MuSiQue-Full pairs are written separately as ``train-unanswerable`` and
``validation-unanswerable`` with the answer ``unanswerable``; they are built first.
Gold is the supporting evidence still present. No chunk of a paragraph removed
from the answerable twin may be stored anywhere in the domain (causal eligibility
is the whole domain): only pairs whose removed paragraphs support no other
answerable question qualify, later episodes never draw such a chunk as a
distractor, and the answerable twin is skipped.

2WikiMultihopQA (data_ids_april7). Support is sentence-level; a chunk is gold if
it holds a supporting sentence. Yes/no answers are kept up to
``--2wiki-max-yes-no-fraction`` (default 25%) of each split and counted.

HoVer (v1.1). Claims become ``supported`` / ``not supported`` episodes; the
instruction lives in the prompt and the claim is the question, so the label is
never checked against the instruction text. Supporting sentences index the
abstracts of the HotpotQA 2017 Wikipedia dump, which is streamed once and only
the needed articles are kept. HoVer provides no distractor context; distractors
are non-gold chunks of the supporting articles when an abstract spans several.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import tarfile
import unicodedata
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_corpus_common import (  # noqa: E402
    PROMPT, Writer, chunk, clean, episode, load_tokenizer, source, split_sentences,
)

RAW = Path('/archive/corpora/public2-raw-20260926')
OUTPUT = Path('/archive/corpora/public2-20260926')
MUSIQUE_ZIP_SHA256 = '98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd'
HOVER_COMMIT = '39b84697f196308f398a251a7aea9b82ae0f0562'
HOVER_WIKI = 'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'
SOURCES = {
    'musique': ('https://github.com/StonyBrookNLP/musique (Google Drive '
                '1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h, musique_v1.0.zip)'),
    '2wikimultihopqa': ('https://github.com/Alab-NII/2wikimultihop '
                        '(Dropbox ms2m13252h6xubs/data_ids_april7.zip)'),
    'hover': (f'https://github.com/hover-nlp/hover@{HOVER_COMMIT} data/hover; wiki: '
              f'https://nlp.stanford.edu/projects/hotpotqa/{HOVER_WIKI}'),
}
LICENSES = {
    'musique': 'CC BY 4.0',
    '2wikimultihopqa': 'Apache-2.0',
    'hover': 'CC BY-SA 4.0 (HoVer data); Wikipedia abstracts (HotpotQA dump) CC BY-SA 4.0',
}
HOVER_PROMPT = ('Use the previously stored passages. Give only the short response.\n'
                'Is this claim supported by the stored passages? '
                'Answer supported or not supported.\nClaim: ')
HOVER_LABELS = {'SUPPORTED': 'supported', 'NOT_SUPPORTED': 'not supported'}
UNANSWERABLE = 'unanswerable'
YES_NO = {'yes', 'no'}
# domain: (train episodes, validation episodes, max sources)
DEFAULTS = {'musique': (20000, 500, 50000), '2wikimultihopqa': (20000, 500, 50000),
            'hover': (10000, 500, 25000)}


def _key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f'{seed}\0{identifier}'.encode()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r'[^a-z0-9#]+', ' ', unicodedata.normalize('NFKC', text).lower()).strip()


def _title(text: str) -> str:
    return unicodedata.normalize('NFC', clean(text))


def iter_json_array(handle, block: int = 1 << 20):
    """Yield the elements of one top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    buffer, position, started = '', 0, False
    while True:
        chunk_text = handle.read(block)
        buffer = buffer[position:] + (chunk_text or '')
        position = 0
        while True:
            while position < len(buffer) and buffer[position] in ' \t\r\n,':
                position += 1
            if not started and position < len(buffer):
                if buffer[position] != '[':
                    raise ValueError('Expected a JSON array')
                started, position = True, position + 1
                continue
            if position < len(buffer) and buffer[position] == ']':
                return
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if not chunk_text:
                    raise
                break
            yield item
            position = end
        if not chunk_text:
            raise ValueError('Unterminated JSON array')


def _ordered(rows, seed: int, identifier) -> list:
    """Deterministic seeded order that does not depend on file order."""
    return sorted(rows, key=lambda row: _key(seed, identifier(row)))


def paragraph_chunks(domain: str, title: str, sentences: list[str], gold: set[int], *,
                     provenance: dict) -> list[tuple[dict, bool, str]]:
    """``(source, holds_gold_sentence, chunk_text)`` for one provided paragraph."""
    return [(source(domain, text, title=title, provenance=provenance), is_gold, text)
            for text, is_gold, _ in chunk(sentences, gold)]


def _fits(writer: Writer, rows: list[dict], max_sources: int) -> bool:
    new = {row['record_id'] for row in rows} - writer.sources.keys()
    return len(writer.sources) + len(new) <= max_sources


def _distractors(options: list[dict], gold: list[dict], count: int, rng: random.Random):
    gold_ids = {row['record_id'] for row in gold}
    options = list({row['record_id']: row for row in options
                    if row['record_id'] not in gold_ids}.values())
    rng.shuffle(options)
    return options[:count]


def _add(writer: Writer, split: str, item: dict | None, gold: list[dict],
         distractors: list[dict], max_sources: int) -> bool:
    if item is None:
        return False
    rows = [*gold, *distractors]
    if not _fits(writer, rows, max_sources):
        writer.filters_for(split).reject('source_budget')
        return False
    return writer.add(split, item, rows)


# --------------------------------------------------------------------------- MuSiQue


def musique_rows(raw: Path, split: str):
    name = {'train': 'train', 'validation': 'dev'}[split]
    path = raw / 'musique' / 'data' / f'musique_full_v1.0_{name}.jsonl'
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def musique_pairs(raw: Path, split: str) -> dict[str, dict[bool, dict]]:
    pairs: dict[str, dict[bool, dict]] = {}
    for row in musique_rows(raw, split):
        pairs.setdefault(row['id'], {})[bool(row['answerable'])] = row
    return pairs


def musique_leak_questions(raw: Path, seeds: set[str]) -> dict[str, str]:
    path = raw / 'musique' / 'data' / 'dev_test_singlehop_questions_v1.0.json'
    if not seeds or not path.exists():
        return {}
    listed = json.loads(path.read_text(encoding='utf-8'))
    unknown = seeds - listed.keys()
    if unknown:
        raise ValueError(f'Unknown MuSiQue seed datasets: {sorted(unknown)}')
    return {_norm(item['question']): name for name in sorted(seeds) for item in listed[name]}


def musique_hop_questions(row: dict) -> list[str]:
    answers = [step['answer'] for step in row['question_decomposition']]

    def fill(match):
        index = int(match.group(1)) - 1
        return answers[index] if 0 <= index < len(answers) else match.group(0)

    result = []
    for step in row['question_decomposition']:
        result.extend([step['question'], re.sub(r'#(\d+)', fill, step['question'])])
    return result


def musique_leak(row: dict, leaks: dict[str, str]) -> str | None:
    for question in musique_hop_questions(row):
        if _norm(question) in leaks:
            return leaks[_norm(question)]
    return None


def musique_paragraph_answers(row: dict, answerable: bool) -> dict[int, list[str]]:
    """Answer strings that locate the evidence in each supporting paragraph."""
    result: dict[int, list[str]] = {}
    steps = row['question_decomposition']
    for position, step in enumerate(steps):
        index = step.get('paragraph_support_idx')
        if index is None:
            continue
        answers = [step['answer']]
        if position == len(steps) - 1 and answerable:
            answers += [row['answer'], *row.get('answer_aliases', [])]
        result.setdefault(index, []).extend(answer for answer in answers if clean(answer))
    return result


def musique_paragraph(domain: str, row: dict, paragraph: dict, answers: list[str]):
    """Gold and non-gold chunk sources of one MuSiQue paragraph."""
    sentences = split_sentences(paragraph['paragraph_text'])
    provenance = {'musique_id': row['id'], 'paragraph_idx': paragraph['idx']}
    pieces = paragraph_chunks(domain, paragraph['title'], sentences, set(), provenance=provenance)
    if not pieces:
        return [], []
    if len(pieces) == 1 or not paragraph['is_supporting']:
        flags = [True] * len(pieces)
    else:
        lowered = [answer.lower() for answer in answers]
        flags = [any(answer in text.lower() for answer in lowered) for _, _, text in pieces]
        flags = flags if any(flags) else [True] * len(pieces)
    gold = [item for item, flag in zip((p[0] for p in pieces), flags, strict=True) if flag]
    other = [item for item, flag in zip((p[0] for p in pieces), flags, strict=True) if not flag]
    return gold, other


def musique_episode(row: dict, split: str, writer: Writer, *, tokenizer, distractors: int,
                    seed: int, answerable: bool, exclude: set[str] = frozenset()):
    domain = writer.domain
    answers = musique_paragraph_answers(row, answerable)
    rng = random.Random(f'{seed}:{row["id"]}:{answerable}')
    gold, options = [], []
    for paragraph in row['paragraphs']:
        if paragraph['is_supporting']:
            part, _ = musique_paragraph(domain, row, paragraph, answers.get(paragraph['idx'], []))
            gold.extend(part)
        else:
            part, _ = musique_paragraph(domain, row, paragraph, [])
            if part and part[0]['record_id'] not in exclude:
                options.append(part[0])
    chosen = _distractors(options, gold, distractors, rng)
    hops = row['id'].split('__')[0]
    item = episode(
        domain=domain, split=split, identifier=f'{row["id"]}-{"ans" if answerable else "unans"}',
        question=row['question'], answer=row['answer'] if answerable else UNANSWERABLE,
        gold=gold, supports=chosen, filters=writer.filters_for(split), tokenizer=tokenizer,
        annotation='verified' if answerable else 'unanswerable_contrast',
        task_family='public_multihop_qa' if answerable else 'public_multihop_unanswerable',
        provenance={'musique_id': row['id'], 'hops': hops, 'answerable': answerable,
                    'answer_aliases': row.get('answer_aliases', []) if answerable else [],
                    **({} if answerable else {'answerable_twin_answer': row['answer']})})
    return item, gold, chosen


def _paragraph_key(paragraph: dict) -> tuple[str, str]:
    return clean(paragraph['title']), clean(paragraph['paragraph_text'])


def musique_removed_paragraphs(pair: dict[bool, dict]) -> list[tuple[str, str]]:
    present = {_paragraph_key(p) for p in pair[False]['paragraphs']}
    return [_paragraph_key(p) for p in pair[True]['paragraphs']
            if p['is_supporting'] and _paragraph_key(p) not in present]


def musique_removed_evidence(domain: str, pair: dict[bool, dict]) -> set[str]:
    """Every chunk of each answerable-twin supporting paragraph absent from the
    unanswerable version; none of them may be stored in the domain."""
    removed, keys = set(), set(musique_removed_paragraphs(pair))
    for paragraph in pair[True]['paragraphs']:
        if paragraph['is_supporting'] and _paragraph_key(paragraph) in keys:
            gold, other = musique_paragraph(domain, pair[True], paragraph, [])
            removed.update(item['record_id'] for item in (*gold, *other))
    return removed


def build_musique(raw: Path, output: Path, *, train: int, validation: int, max_sources: int,
                  unanswerable_train: int, unanswerable_validation: int,
                  leak_seeds: set[str], tokenizer, distractors: int, seed: int) -> dict:
    writer = Writer(output / 'musique', 'musique')
    leaks = musique_leak_questions(raw, leak_seeds)
    pairs = {split: musique_pairs(raw, split) for split in ('validation', 'train')}
    order = {split: _ordered(list(rows), seed, lambda key: key) for split, rows in pairs.items()}
    leaked = {split: {key for key in order[split] if True in pairs[split][key]
                      and musique_leak(pairs[split][key][True], leaks)} for split in pairs}
    # The contrast set is built first. Each kept unanswerable episode forbids every
    # chunk of the paragraphs removed from its answerable twin; later episodes never
    # pick a forbidden distractor and are skipped when their gold needs one (the twin
    # always does), because causal eligibility is the whole domain. Only pairs whose
    # removed paragraphs support no other answerable question qualify, so the
    # contrast set costs little more than the twins.
    usage = Counter(_paragraph_key(paragraph) for split in pairs
                    for key, pair in pairs[split].items()
                    if True in pair and key not in leaked[split]
                    for paragraph in pair[True]['paragraphs'] if paragraph['is_supporting'])
    forbidden: set[str] = set()
    contrast_ids = {split: set() for split in pairs}
    for split, wanted in (('validation', unanswerable_validation), ('train', unanswerable_train)):
        name = f'{split}-unanswerable'
        filters, kept = writer.filters_for(name), 0
        for identifier in order[split]:
            if kept >= wanted:
                break
            pair = pairs[split][identifier]
            if False not in pair or True not in pair:
                filters.reject('no_contrast_pair')
                continue
            if identifier in leaked[split]:
                filters.reject(f'seed_overlap_{musique_leak(pair[True], leaks)}')
                continue
            removed = musique_removed_evidence(writer.domain, pair)
            if not removed:
                filters.reject('no_removed_evidence')
                continue
            if any(usage[key] > 1 for key in musique_removed_paragraphs(pair)):
                filters.reject('removed_evidence_shared')
                continue
            if removed & writer.sources.keys():
                filters.reject('removed_evidence_in_domain')
                continue
            item, gold, chosen = musique_episode(pair[False], name, writer, tokenizer=tokenizer,
                                                 distractors=distractors, seed=seed,
                                                 answerable=False, exclude=forbidden | removed)
            if item is not None and {row['record_id'] for row in (*gold, *chosen)} & (
                    forbidden | removed):
                filters.reject('would_store_removed_evidence')
                continue
            if _add(writer, name, item, gold, chosen, max_sources):
                forbidden |= removed
                contrast_ids[split].add(identifier)
                kept += 1
    counts = Counter()
    for split, wanted in (('validation', validation), ('train', train)):
        filters, kept = writer.filters_for(split), 0
        for identifier in order[split]:
            if kept >= wanted:
                break
            row = pairs[split][identifier].get(True)
            if row is None:
                filters.reject('no_answerable_version')
                continue
            if identifier in leaked[split]:
                filters.reject(f'seed_overlap_{musique_leak(row, leaks)}')
                continue
            if identifier in contrast_ids[split]:
                filters.reject('contrast_twin')
                continue
            item, gold, chosen = musique_episode(row, split, writer, tokenizer=tokenizer,
                                                 distractors=distractors, seed=seed,
                                                 answerable=True, exclude=forbidden)
            if item is not None and {row['record_id'] for row in (*gold, *chosen)} & forbidden:
                filters.reject('would_store_removed_evidence')
                continue
            if _add(writer, split, item, gold, chosen, max_sources):
                kept += 1
                counts[f'{split}/{row["id"].split("__")[0]}'] += 1
    if forbidden & writer.sources.keys():
        raise AssertionError('Removed MuSiQue evidence ended up in the domain')
    manifest = _manifest('musique', seed=seed, targets=(train, validation, max_sources),
                         distractors=distractors)
    manifest.update({
        'unanswerable_targets': {'train': unanswerable_train,
                                 'validation': unanswerable_validation},
        'forbidden_removed_chunks': len(forbidden),
        'leak_seeds': sorted(leak_seeds),
        'leak_file_used': bool(leaks),
        'seed_overlap_ids': {split: len(rows) for split, rows in leaked.items()},
        'kept_by_hops': dict(sorted(counts.items())),
        'raw_sha256': {'musique_v1.0.zip': MUSIQUE_ZIP_SHA256},
        'gold_rule': ('paragraph-level support; a multi-chunk supporting paragraph keeps the '
                      'chunks that contain its hop answer, else all chunks'),
        'unanswerable_rule': ('MuSiQue-Full unanswerable halves with answer "unanswerable", built '
                              'first; gold is the remaining supporting evidence; no chunk of a '
                              'paragraph removed from the answerable twin is stored in the '
                              'domain, so answerable episodes needing one are skipped'),
    })
    return writer.close(manifest)


# ------------------------------------------------------------------ 2WikiMultihopQA


def twowiki_rows(raw: Path, split: str):
    name = {'train': 'train.json', 'validation': 'dev.json'}[split]
    folder = raw / '2wikimultihopqa'
    archive = folder / 'data_ids_april7.zip'
    if (folder / name).exists():
        with (folder / name).open(encoding='utf-8') as handle:
            yield from iter_json_array(handle)
        return
    with zipfile.ZipFile(archive) as bundle:
        member = next(item for item in bundle.namelist()
                      if item.rsplit('/', 1)[-1] == name and '__MACOSX' not in item)
        with bundle.open(member) as handle:
            yield from iter_json_array(io.TextIOWrapper(handle, encoding='utf-8'))


def _sample_ids(rows, seed: int, limit: int, identifier) -> set[str]:
    keys = sorted((_key(seed, identifier(row)), identifier(row)) for row in rows)
    return {value for _, value in keys[:limit]}


def twowiki_episode(row: dict, split: str, writer: Writer, *, tokenizer, distractors: int,
                    seed: int):
    domain, filters = writer.domain, writer.filters_for(split)
    context: dict[str, list[str]] = {}
    for title, sentences in row['context']:
        context.setdefault(title, sentences)
    support: dict[str, set[int]] = {}
    for title, index in row['supporting_facts']:
        support.setdefault(title, set()).add(int(index))
    gold, options = [], []
    for title, indices in support.items():
        sentences = context.get(title)
        if sentences is None or max(indices) >= len(sentences):
            filters.reject('missing_support')
            return None, [], []
    for title, sentences in context.items():
        provenance = {'twowiki_id': row['_id']}
        for item, is_gold, _ in paragraph_chunks(domain, title, sentences,
                                                 support.get(title, set()),
                                                 provenance=provenance):
            (gold if is_gold else options).append(item)
    rng = random.Random(f'{seed}:{row["_id"]}')
    chosen = _distractors(options, gold, distractors, rng)
    item = episode(
        domain=domain, split=split, identifier=row['_id'], question=row['question'],
        answer=row['answer'], gold=gold, supports=chosen, filters=filters, tokenizer=tokenizer,
        task_family='public_multihop_qa',
        provenance={'twowiki_id': row['_id'], 'type': row['type'],
                    'evidences': row.get('evidences', []),
                    'yes_no': clean(row['answer']).lower() in YES_NO})
    return item, gold, chosen


def build_twowiki(raw: Path, output: Path, *, train: int, validation: int, max_sources: int,
                  tokenizer, distractors: int, seed: int, candidates: int,
                  yes_no_fraction: float = 0.25) -> dict:
    writer = Writer(output / '2wikimultihopqa', '2wikimultihopqa')
    kept_types = Counter()
    for split, wanted in (('validation', validation), ('train', train)):
        chosen_ids = _sample_ids(twowiki_rows(raw, split), seed,
                                 min(max(4 * wanted, 2000), candidates), lambda row: row['_id'])
        rows = [row for row in twowiki_rows(raw, split) if row['_id'] in chosen_ids]
        kept = yes_no = 0
        for row in _ordered(rows, seed, lambda row: row['_id']):
            if kept >= wanted:
                break
            if (clean(row['answer']).lower() in YES_NO
                    and yes_no + 1 > yes_no_fraction * (kept + 1)):
                writer.filters_for(split).reject('yes_no_cap')
                continue
            item, gold, chosen = twowiki_episode(row, split, writer, tokenizer=tokenizer,
                                                 distractors=distractors, seed=seed)
            if _add(writer, split, item, gold, chosen, max_sources):
                kept += 1
                kept_types[f'{split}/{row["type"]}'] += 1
                if item['provenance']['yes_no']:
                    yes_no += 1
                    kept_types[f'{split}/yes_no_answer'] += 1
        del rows
    manifest = _manifest('2wikimultihopqa', seed=seed, targets=(train, validation, max_sources),
                         distractors=distractors)
    manifest.update({'kept_by_type': dict(sorted(kept_types.items())),
                     'gold_rule': 'sentence-level supporting_facts',
                     'yes_no_policy': (f'kept up to {yes_no_fraction:.0%} of each split '
                                       '(upstream ~40% of usable train rows); counted in '
                                       'kept_by_type */yes_no_answer')})
    return writer.close(manifest)


# -------------------------------------------------------------------------- HoVer


def hover_rows(raw: Path, split: str) -> list[dict]:
    name = {'train': 'train', 'validation': 'dev'}[split]
    return json.loads((raw / 'hover' / f'hover_{name}_release_v1.1.json').read_text('utf-8'))


def scan_hotpot_abstracts(path: Path, titles: set[str]) -> dict[str, list[str]]:
    """Stream the HotpotQA abstracts dump and keep only the named articles."""
    found: dict[str, list[str]] = {}
    with tarfile.open(path, mode='r|bz2') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.bz2'):
                continue
            data = bz2.decompress(archive.extractfile(member).read())
            for line in data.decode('utf-8').splitlines():
                if not line.strip():
                    continue
                article = json.loads(line)
                title = _title(article['title'])
                if title in titles and title not in found:
                    found[title] = [sentence for sentence in article['text']]
    return found


def hover_episode(row: dict, split: str, writer: Writer, wiki: dict[str, list[str]], *,
                  tokenizer, distractors: int, seed: int):
    domain, filters = writer.domain, writer.filters_for(split)
    support: dict[str, set[int]] = {}
    for title, index in row['supporting_facts']:
        support.setdefault(_title(title), set()).add(int(index))
    gold, options = [], []
    for title, indices in support.items():
        sentences = wiki.get(title)
        if sentences is None:
            filters.reject('missing_wiki_article')
            return None, [], []
        if max(indices) >= len(sentences):
            filters.reject('missing_support_sentence')
            return None, [], []
        for item, is_gold, _ in paragraph_chunks(domain, title, sentences, indices,
                                                 provenance={'hover_uid': row['uid']}):
            (gold if is_gold else options).append(item)
    chosen = _distractors(options, gold, distractors, random.Random(f'{seed}:{row["uid"]}'))
    item = episode(
        domain=domain, split=split, identifier=row['uid'], question=row['claim'],
        answer=HOVER_LABELS[row['label']], gold=gold, supports=chosen, filters=filters,
        tokenizer=tokenizer, prompt=HOVER_PROMPT, task_family='public_claim_verification',
        allow_answer_in_query=False,
        provenance={'hover_uid': row['uid'], 'num_hops': row['num_hops'],
                    'label': row['label'], 'hpqa_id': row.get('hpqa_id')})
    return item, gold, chosen


def build_hover(raw: Path, output: Path, *, train: int, validation: int, max_sources: int,
                tokenizer, distractors: int, seed: int, wiki_path: Path | None) -> dict:
    writer = Writer(output / 'hover', 'hover')
    rows = {split: _ordered(hover_rows(raw, split), seed, lambda row: row['uid'])
            for split in ('validation', 'train')}
    titles = {_title(title) for split_rows in rows.values() for row in split_rows
              for title, _ in row['supporting_facts']}
    wiki = scan_hotpot_abstracts(wiki_path or raw / 'hover' / HOVER_WIKI, titles)
    print(f'hover: {len(wiki)} of {len(titles)} supporting articles found', flush=True)
    kept_labels = Counter()
    for split, wanted in (('validation', validation), ('train', train)):
        kept = 0
        for row in rows[split]:
            if kept >= wanted:
                break
            item, gold, chosen = hover_episode(row, split, writer, wiki, tokenizer=tokenizer,
                                               distractors=distractors, seed=seed)
            if _add(writer, split, item, gold, chosen, max_sources):
                kept += 1
                kept_labels[f'{split}/{row["label"]}'] += 1
                kept_labels[f'{split}/hops{row["num_hops"]}'] += 1
    manifest = _manifest('hover', seed=seed, targets=(train, validation, max_sources),
                         distractors=distractors)
    manifest.update({'kept_by_label': dict(sorted(kept_labels.items())),
                     'wiki_articles_found': len(wiki), 'wiki_articles_needed': len(titles),
                     'prompt': HOVER_PROMPT,
                     'gold_rule': ('sentence-level supporting_facts over HotpotQA 2017 '
                                   'abstract sentences'),
                     'distractor_rule': 'non-gold chunks of the supporting articles only'})
    return writer.close(manifest)


# ---------------------------------------------------------------------------- CLI


def _manifest(domain: str, *, seed: int, targets: tuple[int, int, int], distractors: int) -> dict:
    train, validation, max_sources = targets
    return {'source': SOURCES[domain], 'license': LICENSES[domain], 'seed': seed,
            'targets': {'train': train, 'validation': validation, 'max_sources': max_sources},
            'distractors_per_episode': distractors, 'created_at': 1, 'query_time': 2,
            'prompt': PROMPT, 'all_required': True}


def prepare(raw: Path, output: Path, datasets: list[str], *, targets: dict, tokenizer,
            distractors: int = 2, seed: int = 1701, musique_unanswerable: tuple[int, int] = (
                2000, 250), musique_leak_seeds: set[str] = frozenset(
                {'squad2'}), twowiki_candidates: int = 40000, twowiki_yes_no_fraction: float = 0.25,
            hover_wiki: Path | None = None) -> dict:
    summaries = {}
    if 'musique' in datasets:
        train, validation, max_sources = targets['musique']
        summaries['musique'] = build_musique(
            raw, output, train=train, validation=validation, max_sources=max_sources,
            unanswerable_train=musique_unanswerable[0],
            unanswerable_validation=musique_unanswerable[1],
            leak_seeds=set(musique_leak_seeds), tokenizer=tokenizer, distractors=distractors,
            seed=seed)
    if '2wikimultihopqa' in datasets:
        train, validation, max_sources = targets['2wikimultihopqa']
        summaries['2wikimultihopqa'] = build_twowiki(
            raw, output, train=train, validation=validation, max_sources=max_sources,
            tokenizer=tokenizer, distractors=distractors, seed=seed,
            candidates=twowiki_candidates, yes_no_fraction=twowiki_yes_no_fraction)
    if 'hover' in datasets:
        train, validation, max_sources = targets['hover']
        summaries['hover'] = build_hover(
            raw, output, train=train, validation=validation, max_sources=max_sources,
            tokenizer=tokenizer, distractors=distractors, seed=seed, wiki_path=hover_wiki)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--datasets', default=','.join(DEFAULTS))
    parser.add_argument('--distractors', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--no-tokenizer', action='store_true',
                        help='skip token-length checks (tests only)')
    for domain, (train, validation, max_sources) in DEFAULTS.items():
        flag = domain.replace('2wikimultihopqa', '2wiki')
        parser.add_argument(f'--{flag}-train', type=int, default=train)
        parser.add_argument(f'--{flag}-validation', type=int, default=validation)
        parser.add_argument(f'--{flag}-max-sources', type=int, default=max_sources)
    parser.add_argument('--musique-unanswerable-train', type=int, default=2000)
    parser.add_argument('--musique-unanswerable-validation', type=int, default=250)
    parser.add_argument('--musique-leak-seeds', default='squad2',
                        help='seed datasets whose MuSiQue dev/test single hops are excluded')
    parser.add_argument('--2wiki-candidates', dest='twowiki_candidates', type=int,
                        default=40000, help='sampled train rows held in memory')
    parser.add_argument('--2wiki-max-yes-no-fraction', dest='twowiki_yes_no_fraction',
                        type=float, default=0.25)
    parser.add_argument('--hover-wiki', type=Path)
    args = parser.parse_args()
    datasets = [name for name in args.datasets.split(',') if name]
    unknown = set(datasets) - DEFAULTS.keys()
    if unknown:
        parser.error(f'unknown datasets: {sorted(unknown)}')
    targets = {domain: tuple(getattr(args, f'{domain.replace("2wikimultihopqa", "2wiki")}_{part}')
                             for part in ('train', 'validation', 'max_sources'))
               for domain in DEFAULTS}
    summaries = prepare(
        args.raw, args.output, datasets, targets=targets,
        tokenizer=None if args.no_tokenizer else load_tokenizer(),
        distractors=args.distractors, seed=args.seed,
        musique_unanswerable=(args.musique_unanswerable_train,
                              args.musique_unanswerable_validation),
        musique_leak_seeds={name for name in args.musique_leak_seeds.split(',') if name},
        twowiki_candidates=args.twowiki_candidates,
        twowiki_yes_no_fraction=args.twowiki_yes_no_fraction, hover_wiki=args.hover_wiki)
    print(json.dumps({domain: {key: summary[key] for key in (
        'episodes', 'sources', 'unreferenced_sources', 'gold_per_episode', 'filtered')}
        for domain, summary in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
