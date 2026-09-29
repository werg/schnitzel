"""Inverse-cloze retrieval transcripts (``schnitz.inverse_cloze``,
``scripts/prepare_inverse_cloze.py``); CPU, project-authored fixtures, no downloads."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
import random

from schnitz import inverse_cloze as ic

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_inverse_cloze.py'
spec = importlib.util.spec_from_file_location('prepare_inverse_cloze', SCRIPT)
pic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pic)

CITIES = ['Zurich', 'Lisbon', 'Quito', 'Oslo', 'Hanoi', 'Lagos', 'Perth', 'Cusco']


def doc_sentences(doc: str, n: int = 8) -> list[str]:
    year = 1900 + sum(map(ord, doc)) % 97
    return [f'{doc} fact {k} names {CITIES[k % len(CITIES)]} in {year + 7 * k} '
            f'with further detail number {k} of the {doc} article.' for k in range(n)]


BOILER = 'This boilerplate sentence appears word for word in two unrelated stored articles.'


def window_records() -> dict[str, dict]:
    """Three documents cut into windows of three sentences every sentence (redundancy
    3); documents Alpha and Gamma share one boilerplate sentence."""
    records = {}
    docs = {'Alpha': doc_sentences('Alpha') + [BOILER], 'Beta': doc_sentences('Beta'),
            'Gamma': [BOILER] + doc_sentences('Gamma')}
    for doc, sents in docs.items():
        for w in range(len(sents) - 2):
            rid = f'{doc[0].lower()}{w:02d}'
            records[rid] = {'record_id': rid, 'kind': 'passage', 'created_at': 1,
                            'domain': 'recall_text',
                            'text': f'Title: {doc}\nPassage: ' + ' '.join(sents[w:w + 3]),
                            'provenance': {'record_type': 'window', 'document': doc,
                                           'window': w, 'shape': 'head'}}
    return records


def window_episodes() -> dict[str, list[dict]]:
    return {'train': [{'episode_id': 't1', 'provenance': {'document': 'Alpha'}},
                      {'episode_id': 't2', 'provenance': {'document': 'Gamma'}}],
            'validation': [{'episode_id': 'v1', 'provenance': {'document': 'Beta'}}]}


def test_verbatim_cues_name_the_record_its_copies_and_its_documents_neutral():
    records = window_records()
    rows, summary = ic.build(records, window_episodes(), lambda r: 'kb',
                             {'train': 12, 'validation': 4}, seed=3, corpus='rt')
    assert summary['corpus_type'] == 'verbatim'
    assert summary['groups'] == {'train': 2, 'validation': 1}
    assert len(rows['train']) == 12 and len(rows['validation']) == 4
    for split, want in (('train', {'doc:Alpha', 'doc:Gamma'}), ('validation', {'doc:Beta'})):
        for row in rows[split]:
            assert row['task_family'] == ic.FAMILY and row['kb'] == 'kb'
            assert row['provenance']['group'] in want        # split by document
            cue = row['messages'][1]['content'].split('\n', 1)[1]
            assert BOILER not in cue                        # ambiguous: skipped
            slot = row['messages'][3]['content']['slot']
            site = row['search_sites'][0]
            doc = row['provenance']['group'].split(':')[1]
            same = [r for r, rec in records.items() if rec['provenance']['document'] == doc]
            holding = [r for r in same if cue in ic.normalize(ic.body(records[r]))]
            assert slot['record_ids'][0] in holding and slot['record_ids'] == site['record_ids']
            assert sorted(slot['alternatives']) == sorted(holding)
            assert sorted(slot['neutral']) == sorted(set(same) - set(holding))
            # the cue comes from the content, never from the title header
            assert not cue.startswith('Title')
            # no answer turn: the call is the only assistant message
            assert [m['role'] for m in row['messages']] == ['system', 'user', 'assistant', 'tool']
            assert row['messages'][2]['tool_calls'][0]['function'] == {
                'name': 'memory_search', 'arguments': {}}
    again, _ = ic.build(records, window_episodes(), lambda r: 'kb',
                        {'train': 12, 'validation': 4}, seed=3, corpus='rt')
    assert again == rows                                    # deterministic


def test_ambiguity_counts_only_unrelated_records_of_the_same_kb():
    records = window_records()
    groups = ic.Groups(records, lambda r: 'kb')
    shared = ic.Cue(BOILER, 'sentence', 'doc:Alpha', 'kb', 'a06', ['a06'])
    unique = ic.Cue(doc_sentences('Alpha')[3], 'sentence', 'doc:Alpha', 'kb', 'a01', ['a01'])
    amb = ic.Ambiguity(groups, [shared, unique])
    assert amb.ambiguous(shared) and amb.unrelated_hits(shared)[0].startswith('g')
    assert not amb.ambiguous(unique)
    # another KB's copy is not in this KB's search, so it does not make a cue ambiguous
    split = ic.Groups(records, lambda r: 'other' if r.startswith('g') else 'kb')
    assert not ic.Ambiguity(split, [shared]).ambiguous(shared)


def test_window_edges_never_give_a_cut_sentence():
    rec = {'record_id': 'x', 'kind': 'passage',
           'text': 'Title: T\nPassage: of a sentence cut at the start by the window edge here. '
                   'A whole sentence in the middle of this window names Quito in 1911 today. '
                   'And a final one that the window edge cuts before its',
           'provenance': {'record_type': 'window', 'shape': 'full'}}
    got = ic.verbatim_candidates(rec, random.Random(0))
    assert got == [('A whole sentence in the middle of this window names Quito in 1911 '
                    'today.', 'sentence')]


def citance_records() -> tuple[dict, dict]:
    records = {}
    works = {'W1': 4, 'W2': 3, 'W3': 3}
    for work, n in works.items():
        for k in range(n):
            rid = f'{work.lower()}c{k}'
            records[rid] = {
                'record_id': rid, 'kind': 'passage', 'created_at': 1,
                'text': f'Citing paper: Paper {rid}\nEarlier work set the scene for {rid}. '
                        f'The method of {work} solves task {k} with a learned index for {rid}. '
                        f'We extend it in section {k} of this {rid} study.',
                'provenance': {'record_type': 'citance', 'cited': [f'https://x/{work}']}}
    abstract = {w: f'We introduce method {w} for retrieval over large stores. It learns keys '
                   f'from data and beats baselines by a wide margin on {w} benchmarks.'
                for w in works}
    episodes = {'train': [], 'validation': []}
    for work in works:
        split = 'validation' if work == 'W2' else 'train'
        episodes[split].append({'episode_id': f'{work}-abs', 'answer': abstract[work],
                                'task_family': 'public_citance_abstract',
                                'provenance': {'cited': f'https://x/{work}'}})
        episodes[split].append({'episode_id': f'{work}-desc', 'task_family':
                                'public_citance_description',
                                'answer': f'A held out paper describes {work} as a fast '
                                          'learned index for dense retrieval tasks.',
                                'provenance': {'cited': f'https://x/{work}'}})
    return records, episodes


def test_citance_cues_are_other_descriptions_and_sibling_sources_are_neutral():
    records, episodes = citance_records()
    rows, summary = ic.build(records, episodes, lambda r: 'kb', {'train': 30, 'validation': 6},
                             seed=1, corpus='cit')
    assert summary['corpus_type'] == 'citance'
    kinds = Counter(r['provenance']['cue_kind'] for r in rows['train'])
    assert set(kinds) == {'abstract', 'heldout', 'sibling'}
    for split in ('train', 'validation'):
        for row in rows[split]:
            work = row['provenance']['cited'].rsplit('/', 1)[1]
            assert (work == 'W2') == (split == 'validation')
            slot = row['messages'][3]['content']['slot']
            members = sorted(r for r in records if r.startswith(work.lower()))
            cue = row['messages'][1]['content'].split('\n', 1)[1]
            if row['provenance']['cue_kind'] == 'sibling':
                source = row['provenance']['cue_record']
                assert slot['neutral'] == [source] and source not in slot['alternatives']
                assert sorted(slot['alternatives'] + [source]) == members
            else:
                assert 'neutral' not in slot and sorted(slot['alternatives']) == members
                assert not any(cue in records[r]['text'] for r in members)   # a paraphrase
            assert slot['record_ids'][0] in slot['alternatives']


def test_parallel_cues_come_from_the_unstored_version_and_name_covering_records():
    records = {}
    verses = {1: 'In the start God made the heavens', 2: 'The earth was empty and dark then',
              3: 'God said let there be light now', 4: 'God saw the light and it was good'}
    for version, chunks in (('KJV', [(1, 2), (3, 4)]), ('ASV', [(1, 3), (4, 4)]),
                            ('DBY', [(1, 4)])):
        for a, b in chunks:
            rid = f'{version}{a}{b}'.lower()
            lines = '\n'.join(f'{v} {verses[v]} ({version} wording)' for v in range(a, b + 1))
            records[rid] = {'record_id': rid, 'kind': 'parallel_passage', 'created_at': 1,
                            'text': f'{version} Bible [{version}], Genesis 1:{a}-{b}\n{lines}',
                            'provenance': {'version': version, 'book': 'Genesis', 'chapter': 1,
                                           'verse_start': a, 'verse_end': b}}
    answer = '\n'.join(f'{v} {w}, as the target puts it plainly' for v, w in verses.items())
    episodes = {'train': [{'episode_id': 'e1', 'answer': answer,
                           'provenance': {'book': 'Genesis', 'chapter': 1,
                                          'target_version': 'WEB'}}]}
    rows, summary = ic.build(records, episodes, lambda r: 'kb', {'train': 1}, seed=0,
                             corpus='par')
    assert summary['corpus_type'] == 'parallel'
    row = rows['train'][0]
    a, b = row['provenance']['verses']
    slot = row['messages'][3]['content']['slot']
    covering = sorted(r for r, rec in records.items()
                      if rec['provenance']['verse_start'] <= a and b <= rec['provenance']['verse_end'])
    assert sorted(slot['alternatives']) == covering
    assert sorted(slot.get('neutral', [])) == sorted(set(records) - set(covering))
    cue = row['messages'][1]['content'].split('\n', 1)[1]
    assert 'target puts it' in cue and not any(cue in rec['text'] for rec in records.values())
    assert 'Genesis' not in cue                               # no reference in the cue


def test_script_writes_audited_transcripts_on_the_source_kb(tmp_path):
    corpus = tmp_path / 'tasks-recall-text-r3-20260929'
    corpus.mkdir()
    records = window_records()
    (corpus / 'sources.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records.values()))
    for split, rows in window_episodes().items():
        (corpus / f'episodes-{split}.jsonl').write_text(''.join(json.dumps(r) + '\n'
                                                                for r in rows))
    (corpus / 'manifest.json').write_text('{}')
    out = tmp_path / 'memory-inverse-cloze-recall-text-r3-t'
    manifest = pic.run_corpus(corpus, out, {'train': 10, 'validation': 3}, seed=0)
    assert manifest['kb'] == 'recall-text-r3' and manifest['input'] == str(corpus)
    for split, n in (('train', 10), ('validation', 3)):
        assert manifest['splits'][split]['written'] == n
        assert manifest['splits'][split]['rejected'] == {}
        rows = [json.loads(line) for line in (out / f'transcripts-{split}.jsonl').open()]
        assert all(r['kb'] == 'recall-text-r3' and r['split'] == split for r in rows)
    # the L1 bank build banks every named record from the source corpus's KB
    from schnitz.kb.bank import record_sources
    got = record_sources([out], {'train': None, 'validation': None})
    assert set(got) <= set(records) and {v['kb'] for v in got.values()} == {'recall-text-r3'}
    again = pic.run_corpus(corpus, tmp_path / 'again', {'train': 10, 'validation': 3}, seed=0)
    assert again['sha256'] == manifest['sha256']
