"""Answer-bearing episodes from long non-Wikipedia documents: QASPER and ConditionalQA.

QASPER (allenai, CC BY 4.0) supplies NLP papers with information-seeking questions,
annotated evidence paragraphs and extractive, free-form or yes/no answers.
ConditionalQA (haitian-sun/ConditionalQA, BSD-2-Clause repository license; the
documents are UK government guidance) supplies a user scenario, a question,
answers with conditions and evidence elements of one gov.uk document.

Each document section is packed into passages of at most ``MAX_CHARS`` characters
with ``chunk``. Gold passages are the chunks containing the annotated evidence; one
group containing all of them is required. Up to two other chunks of the same
document are causally available distractors. All passages are created at time 1
and queried at time 2. Only passages referenced by selected episodes are written;
the document remainder is not stored.

QASPER questions ("what baselines do they use?") only make sense for one paper,
so the query names the paper title. ConditionalQA queries are scenario plus
question; queries above ``MAX_QUERY_TOKENS`` are dropped and counted, never
truncated. Unanswerable questions have no gold evidence and are dropped and counted
(``unanswerable``). Answers are supervised targets only, never query text.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import re
import tarfile

from public_corpus_common import (
    MAX_CHARS, Filters, Writer, chunk, clean, episode, load_tokenizer, source,
    split_sentences)

QASPER = 'qasper'
CONDITIONALQA = 'conditionalqa'
QASPER_FILES = {'train': 'qasper-train-v0.3.json', 'validation': 'qasper-dev-v0.3.json'}
CONDITIONALQA_FILES = {'train': 'train.json', 'validation': 'dev.json'}
YES_NO = {'yes', 'no'}


def _document_chunks(domain: str, sections: list[tuple[str, list[str]]], *,
                     document_title: str, provenance: dict) -> tuple[list[dict], dict]:
    """Chunk ``(section_name, units)`` pairs; return sources and unit -> source indices.

    A unit is one paragraph (QASPER) or one HTML element (ConditionalQA). Units are
    split into sentences so chunk boundaries never fall inside a sentence.
    """
    rows, owner = [], {}
    for section_index, (section_name, units) in enumerate(sections):
        sentences, unit_of = [], []
        for unit_index, unit in enumerate(units):
            for sentence in split_sentences(unit):
                sentences.append(sentence)
                unit_of.append(unit_index)
        title = f'{document_title} — {section_name}' if section_name else document_title
        for chunk_index, (text, _, indices) in enumerate(chunk(sentences, limit=MAX_CHARS)):
            units_here = sorted({unit_of[i] for i in indices})
            row = source(domain, text, title=title, created_at=1, provenance={
                **provenance, 'section': clean(section_name), 'section_index': section_index,
                'chunk_index': chunk_index, 'units': units_here})
            for unit_index in units_here:
                owner.setdefault((section_index, unit_index), []).append(len(rows))
            rows.append(row)
    return rows, owner


def _distractors(rows: list[dict], gold: list[dict], rng: random.Random,
                 count: int) -> list[dict]:
    gold_ids = {row['record_id'] for row in gold}
    pool = list({row['record_id']: row for row in rows
                 if row['record_id'] not in gold_ids}.values())
    return rng.sample(pool, min(count, len(pool)))


def _fits_budget(writer: Writer, rows: list[dict], budget: int) -> bool:
    new = {row['record_id'] for row in rows} - writer.sources.keys()
    return len(writer.sources) + len(new) <= budget


# QASPER --------------------------------------------------------------------------

def load_qasper(raw: Path, split: str) -> dict:
    archive = raw / 'qasper-train-dev-v0.3.tgz'
    with tarfile.open(archive) as handle:
        member = handle.extractfile(QASPER_FILES[split])
        return json.loads(member.read())


def qasper_answer(annotation: dict) -> tuple[str, str]:
    answer = annotation['answer']
    if answer.get('yes_no') is not None:
        return ('yes' if answer['yes_no'] else 'no'), 'yes_no'
    spans = [clean(span) for span in answer.get('extractive_spans') or [] if clean(span)]
    if spans:
        return ', '.join(spans), 'extractive'
    return clean(answer.get('free_form_answer') or ''), 'free_form'


def _qasper_sections(paper: dict) -> list[tuple[str, list[str]]]:
    sections = [('Abstract', [paper.get('abstract') or ''])]
    for section in paper.get('full_text') or []:
        sections.append((clean(section.get('section_name') or ''),
                         [paragraph or '' for paragraph in section.get('paragraphs') or []]))
    return sections


def _narrow(rows: list[dict], candidates: list[int], highlights: list[str]) -> list[int]:
    """Of a paragraph split over several chunks, keep those holding a highlight."""
    if len(candidates) < 2:
        return candidates
    probes = []
    for text in highlights:
        text = clean(text)
        if len(text) >= 20:
            probes.extend({text[:48], text[-48:]})
    kept = [index for index in candidates
            if any(probe in rows[index]['text'] for probe in probes)]
    return kept or candidates


def qasper_episodes(papers: dict, split: str, writer: Writer, *, limit: int,
                    source_budget: int, distractors: int, max_gold: int, seed: int,
                    tokenizer=None) -> None:
    filters = writer.filters_for(split)
    order = sorted(papers)
    random.Random(f'{seed}:{QASPER}:{split}').shuffle(order)
    kept = 0
    for paper_id in order:
        paper = papers[paper_id]
        title = clean(paper.get('title') or paper_id)
        sections = _qasper_sections(paper)
        rows, owner = _document_chunks(QASPER, sections, document_title=title,
                                       provenance={'paper_id': paper_id})
        location = {}
        for section_index, (_, paragraphs) in enumerate(sections):
            for paragraph_index, paragraph in enumerate(paragraphs):
                location.setdefault(clean(paragraph), (section_index, paragraph_index))
        for qa in paper.get('qas') or []:
            if kept >= limit:
                return
            annotations = [item for item in qa.get('answers') or []
                           if not item['answer'].get('unanswerable')
                           and item['answer'].get('evidence')]
            if not annotations:
                unanswerable = any(item['answer'].get('unanswerable')
                                   for item in qa.get('answers') or [])
                filters.reject('unanswerable' if unanswerable else 'no_evidence')
                continue
            first_reasons = None
            for rank, annotation in enumerate(annotations):
                attempt = Filters()
                built = _qasper_attempt(
                    paper_id, title, qa, annotation, rank, rows, owner, location, split,
                    attempt, distractors=distractors, max_gold=max_gold, seed=seed,
                    tokenizer=tokenizer)
                if built is not None:
                    break
                first_reasons = first_reasons or attempt.counts
            if built is None:
                filters.counts.update(first_reasons)
                continue
            item, supports = built
            if not _fits_budget(writer, supports, source_budget):
                filters.reject('source_budget')
                return
            kept += writer.add(split, item, supports)


def _qasper_attempt(paper_id, title, qa, annotation, rank, rows, owner, location, split,
                    filters, *, distractors, max_gold, seed, tokenizer):
    """One annotator's episode, or None with the reason counted in ``filters``."""
    evidence = [clean(text) for text in annotation['answer']['evidence'] if clean(text)]
    if any(text.startswith('FLOAT SELECTED') for text in evidence):
        filters.reject('evidence_table_or_figure')
        return None
    if any(text not in location for text in evidence):
        filters.reject('evidence_not_in_text')
        return None
    gold_indices = []
    for text in evidence:
        candidates = owner.get(location[text], [])
        gold_indices.extend(_narrow(rows, candidates,
                                    annotation['answer'].get('highlighted_evidence') or []))
    gold = list({rows[i]['record_id']: rows[i] for i in gold_indices}.values())
    if len(gold) > max_gold:
        filters.reject('too_many_gold')
        return None
    answer, kind = qasper_answer(annotation)
    extra = _distractors(rows, gold, random.Random(f'{seed}:{qa["question_id"]}'), distractors)
    question = f'In the paper "{title}": {clean(qa["question"])}'
    item = episode(
        domain=QASPER, split=split, identifier=qa['question_id'], question=question,
        answer=answer, gold=gold, supports=gold + extra, filters=filters,
        tokenizer=tokenizer, all_required=True, annotation='verified',
        task_family='public_document_qa', allow_answer_in_query=kind == 'yes_no',
        provenance={'paper_id': paper_id, 'question_id': qa['question_id'],
                    'answer_type': kind, 'annotation_id': annotation.get('annotation_id'),
                    'annotator_rank': rank, 'evidence_paragraphs': len(evidence),
                    'gold_chunks': len(gold), 'distractors': len(extra),
                    'upstream_split': split})
    return None if item is None else (item, gold + extra)


# ConditionalQA ---------------------------------------------------------------------

def strip_html(fragment: str) -> str:
    return clean(html.unescape(re.sub(r'<[^>]+>', ' ', fragment or '')))


def _conditionalqa_sections(document: dict) -> tuple[list[tuple[str, list[str]]], dict]:
    """Split a document at ``<h1>``; return sections and element -> (section, unit)."""
    sections: list[tuple[str, list[str]]] = []
    location = {}
    for element in document['contents']:
        if element.startswith('<h1'):
            sections.append((strip_html(element), []))
            continue
        if not sections:
            sections.append(('', []))
        text = strip_html(element)
        if not text:
            continue
        location.setdefault(element, (len(sections) - 1, len(sections[-1][1])))
        sections[-1][1].append(text)
    return sections, location


def load_conditionalqa(raw: Path, split: str) -> tuple[list[dict], dict]:
    questions = json.loads((raw / CONDITIONALQA_FILES[split]).read_text(encoding='utf-8'))
    documents = json.loads((raw / 'documents.json').read_text(encoding='utf-8'))
    return questions, {document['url']: document for document in documents}


def conditionalqa_answer(row: dict) -> str:
    texts = list(dict.fromkeys(clean(text) for text, _ in row.get('answers') or [] if clean(text)))
    return '; '.join(texts)


def conditionalqa_episodes(questions: list[dict], documents: dict, split: str,
                           writer: Writer, *, limit: int, source_budget: int,
                           distractors: int, max_gold: int, seed: int,
                           tokenizer=None) -> None:
    filters = writer.filters_for(split)
    order = list(questions)
    random.Random(f'{seed}:{CONDITIONALQA}:{split}').shuffle(order)
    cache: dict[str, tuple] = {}
    kept = 0
    for row in order:
        if kept >= limit:
            return
        if row.get('not_answerable') or not row.get('answers'):
            filters.reject('unanswerable')
            continue
        document = documents.get(row['url'])
        if document is None:
            filters.reject('document_missing')
            continue
        if row['url'] not in cache:
            sections, location = _conditionalqa_sections(document)
            rows, owner = _document_chunks(
                CONDITIONALQA, sections, document_title=clean(document['title']),
                provenance={'url': row['url']})
            cache[row['url']] = (rows, owner, location)
        rows, owner, location = cache[row['url']]
        evidence = row.get('evidences') or []
        if not evidence:
            filters.reject('no_evidence')
            continue
        if any(element not in location for element in evidence):
            filters.reject('evidence_not_in_document')
            continue
        gold_indices = [index for element in evidence for index in owner[location[element]]]
        gold = list({rows[i]['record_id']: rows[i] for i in gold_indices}.values())
        if len(gold) > max_gold:
            filters.reject('too_many_gold')
            continue
        answer = conditionalqa_answer(row)
        conditions = [strip_html(text) for _, items in row['answers'] for text in items]
        extra = _distractors(rows, gold, random.Random(f'{seed}:{row["id"]}'), distractors)
        question = f'{clean(row["scenario"])} {clean(row["question"])}'
        item = episode(
            domain=CONDITIONALQA, split=split, identifier=row['id'], question=question,
            answer=answer, gold=gold, supports=gold + extra, filters=filters,
            tokenizer=tokenizer, all_required=True, annotation='verified',
            task_family='public_document_qa',
            allow_answer_in_query=answer.lower() in YES_NO,
            provenance={'question_id': row['id'], 'url': row['url'],
                        'document_title': clean(document['title']),
                        'answers': len(row['answers']), 'conditional': bool(conditions),
                        'conditions': conditions, 'evidence_elements': len(evidence),
                        'gold_chunks': len(gold), 'distractors': len(extra),
                        'upstream_split': split})
        if item is None:
            continue
        if not _fits_budget(writer, gold + extra, source_budget):
            filters.reject('source_budget')
            return
        kept += writer.add(split, item, gold + extra)


# Entry points ---------------------------------------------------------------------

def build_qasper(raw: Path, output: Path, *, train: int = 4000, validation: int = 500,
                 source_budget: int = 20000, distractors: int = 2, max_gold: int = 4,
                 seed: int = 2609, tokenizer=None) -> dict:
    writer = Writer(output, QASPER)
    for split, limit in (('train', train), ('validation', validation)):
        qasper_episodes(load_qasper(raw, split), split, writer, limit=limit,
                        source_budget=source_budget, distractors=distractors,
                        max_gold=max_gold, seed=seed, tokenizer=tokenizer)
    return writer.close({
        'source': 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz',
        'license': 'CC-BY-4.0', 'upstream_splits': {'train': 'train', 'validation': 'dev'},
        'limits': {'train': train, 'validation': validation, 'sources': source_budget},
        'distractors': distractors, 'max_gold': max_gold, 'seed': seed,
        'max_chars': MAX_CHARS, 'tokenizer': tokenizer is not None,
        'notice': ('Query names the paper title. First answerable annotator with textual evidence '
                   'that yields a valid episode (annotator_rank in provenance). '
                   'Unanswerable and table/figure-evidence questions are dropped and counted. '
                   'Gold = chunks holding the evidence paragraphs (narrowed by highlighted '
                   'evidence when a paragraph spans chunks); all gold required.')})


def build_conditionalqa(raw: Path, output: Path, *, train: int = 2000, validation: int = 300,
                        source_budget: int = 10000, distractors: int = 2, max_gold: int = 6,
                        seed: int = 2609, tokenizer=None) -> dict:
    writer = Writer(output, CONDITIONALQA)
    for split, limit in (('train', train), ('validation', validation)):
        questions, documents = load_conditionalqa(raw, split)
        conditionalqa_episodes(questions, documents, split, writer, limit=limit,
                               source_budget=source_budget, distractors=distractors,
                               max_gold=max_gold, seed=seed, tokenizer=tokenizer)
    return writer.close({
        'source': 'https://github.com/haitian-sun/ConditionalQA/tree/'
                  '77bd295952daf415548b3244db10880d3d55cfe0/v1_0',
        'license': 'BSD-2-Clause (repository); documents are gov.uk guidance (OGL v3.0)',
        'upstream_splits': {'train': 'train', 'validation': 'dev'},
        'limits': {'train': train, 'validation': validation, 'sources': source_budget},
        'distractors': distractors, 'max_gold': max_gold, 'seed': seed,
        'max_chars': MAX_CHARS, 'tokenizer': tokenizer is not None,
        'notice': ('Query = scenario + question; over-long queries are dropped. Answer = '
                   'distinct answer texts joined by "; ", conditions kept only in provenance. '
                   'Research use only: answers are not legally verified.')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True,
                        help='Directory holding qasper/ and conditionalqa/ downloads')
    parser.add_argument('--output', type=Path, required=True,
                        help='Parent directory; writes <output>/qasper and <output>/conditionalqa')
    parser.add_argument('--datasets', nargs='+', default=[QASPER, CONDITIONALQA],
                        choices=[QASPER, CONDITIONALQA])
    parser.add_argument('--seed', type=int, default=2609)
    parser.add_argument('--no-tokenizer', action='store_true')
    args = parser.parse_args()
    tokenizer = None if args.no_tokenizer else load_tokenizer()
    for name in args.datasets:
        build = build_qasper if name == QASPER else build_conditionalqa
        summary = build(args.raw / name, args.output / name, seed=args.seed, tokenizer=tokenizer)
        print(json.dumps({key: summary[key] for key in (
            'domain', 'episodes', 'sources', 'gold_per_episode', 'filtered')}, indent=2))
