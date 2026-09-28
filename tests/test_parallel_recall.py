"""Parallel-version recall corpus (``schnitz.parallel_recall``) on synthetic fixture
translations; no downloads."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from schnitz import parallel_recall as pr

BOOKS = ['Genesis', 'Exodus', 'Matthew']
CHAPTERS, VERSES = 6, 14
STORED = ['KJV', 'ASV', 'YLT', 'Darby', 'Webster']
TARGETS = ['WEB', 'BBE']


def tag(book: int, chapter: int, verse: int) -> str:
    """Letters-only verse marker (the alignment check reads [a-z] words)."""
    return ''.join(chr(97 + int(d)) for d in f'{book}{chapter:02d}{verse:02d}')


def verse_text(code: str, b: int, c: int, v: int) -> str:
    t = tag(b, c, v)
    style = code.lower().replace('-', '')
    return (f'{style} renders {t} as {t}alpha and {t}beta, then {style}ish words about '
            f'{t}gamma follow in this verse of the passage')


def write_fixture(root: Path, *, shifted: str | None = None, copy_of_web: str | None = None):
    for code in [*STORED, *TARGETS]:
        v = pr.VERSIONS[code]
        path = root / v.file
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as handle:
            out = csv.writer(handle)
            out.writerow(['id', 'b', 'c', 'v', 't'] if v.legacy else
                         ['Book', 'Chapter', 'Verse', 'Text'])
            for b, book in enumerate(BOOKS):
                for c in range(1, CHAPTERS + 1):
                    n = VERSES + (1 if code == shifted and book == 'Exodus' else 0)
                    for verse in range(1, n + 1):
                        src = 'WEB' if code == copy_of_web and book == 'Genesis' and c == 1 \
                            else code
                        text = verse_text(src, b, c, verse)
                        if code == 'WEB':
                            text = text.replace(' as ', ' as{a footnote} ', 1)
                        if v.legacy:
                            out.writerow([f'{b + 1}{c:03d}{verse:03d}', b + 1, c, verse, text])
                        else:
                            out.writerow([book, c, verse, text])


def build(root: Path, **kw):
    options = dict(stored_codes=STORED + TARGETS, target_codes=TARGETS, seed=0,
                   train_episodes=40, validation_episodes=8, validation_share=0.2,
                   min_versions=3, min_tokens=40, max_tokens=120, per_chapter=2)
    options.update(kw)
    return pr.build(root, **options)


def all_episodes(result):
    return [e for rows in result['episodes'].values() for e in rows]


@pytest.fixture
def fixture(tmp_path):
    write_fixture(tmp_path)
    return tmp_path


def test_clean_verse():
    assert pr.clean_verse('In the beginning God{After "God," a note.} created [the] heavens.') \
        == 'In the beginning God created the heavens.'
    assert pr.clean_verse('現在 你們要敬畏 耶和華，誠心<WAHb>實意地', 'zh') == '現在你們要敬畏耶和華，誠心實意地'
    assert pr.display_book('I Samuel') == '1 Samuel'
    assert pr.display_book('Revelation of John') == 'Revelation'


def test_target_text_never_in_kb(fixture):
    result = build(fixture)
    assert all_episodes(result)
    stored = {r['provenance']['version'] for r in result['records']}
    assert stored == set(STORED)            # target versions are never stored
    kb = [r['text'] for r in result['records']]
    for ep in all_episodes(result):
        assert ep['answer'] not in '\n'.join(kb)
        for line in ep['answer'].split('\n'):
            assert not any(line.split(' ', 1)[1] in text for text in kb)
        assert 'footnote' not in ep['answer'] and '{' not in ep['answer']


def test_every_episode_has_k_covering_versions(fixture):
    result = build(fixture, min_versions=4)
    by_id = {r['record_id']: r for r in result['records']}
    for ep in all_episodes(result):
        p = ep['provenance']
        assert ep['redundancy'] == len(ep['sufficient_groups']) == len(p['covering_versions']) >= 4
        assert p['target_version'] not in p['covering_versions']
        union = [r for g in ep['sufficient_groups'] for r in g]
        assert set(ep['required_ids']) == set(union) == {s['record_id'] for s in ep['supports']}
        for group in ep['sufficient_groups']:
            versions = {by_id[r]['provenance']['version'] for r in group}
            assert len(versions) == 1
            covered = {v for r in group for v in range(by_id[r]['provenance']['verse_start'],
                                                       by_id[r]['provenance']['verse_end'] + 1)}
            assert set(range(p['verse_start'], p['verse_end'] + 1)) <= covered
            assert all(by_id[r]['provenance']['chapter'] == p['chapter'] for r in group)
        assert 40 <= p['target_tokens'] <= 120
    with pytest.raises(ValueError):
        build(fixture, stored_codes=['NoSuchVersion'])


def test_split_disjoint_by_chapter(fixture):
    result = build(fixture)
    chapters = {split: {(e['provenance']['book'], e['provenance']['chapter']) for e in rows}
                for split, rows in result['episodes'].items()}
    assert chapters['train'] and chapters['validation']
    assert not chapters['train'] & chapters['validation']
    for split, rows in result['episodes'].items():
        assert all(e['provenance']['split'] == split for e in rows)
        assert all(result['split_of'][(e['provenance']['book'], e['provenance']['chapter'])]
                   == split for e in rows)
    assert len({e['episode_id'] for e in all_episodes(result)}) == len(all_episodes(result))


def test_records_are_record_sized_and_chunked_per_version(fixture):
    result = build(fixture)
    bounds = {}
    for rec in result['records']:
        p = rec['provenance']
        assert len(rec['text']) <= pr.RECORD_CHARS
        assert rec['kind'] == pr.KIND and rec['created_at'] == 1
        assert rec['text'].startswith(f'{pr.VERSIONS[p["version"]].title} [{p["version"]}], ')
        assert 1 <= p['verse_end'] - p['verse_start'] + 1 <= 10
        bounds.setdefault(p['version'], set()).add((p['book'], p['chapter'], p['verse_start']))
    assert len({frozenset(b) for b in bounds.values()}) > 1   # boundaries differ by version


def test_shifted_versification_drops_the_book(tmp_path):
    write_fixture(tmp_path, shifted='Darby')
    result = build(tmp_path)
    darby = {r['provenance']['book'] for r in result['records']
             if r['provenance']['version'] == 'Darby'}
    assert darby == {'Genesis', 'Matthew'}
    assert all('Darby' not in e['provenance']['covering_versions'] for e in all_episodes(result)
               if e['provenance']['book'] == 'Exodus')


def test_verbatim_copy_of_target_is_rejected(tmp_path):
    write_fixture(tmp_path, copy_of_web='ASV')
    result = build(tmp_path, train_episodes=400, validation_episodes=100)
    for ep in all_episodes(result):
        p = ep['provenance']
        if (p['book'], p['chapter'], p['target_version']) == ('Genesis', 1, 'WEB'):
            raise AssertionError('an episode whose target a stored record contains verbatim')
    assert sum(r.get('target_in_kb', 0) for r in result['rejected'].values()) > 0


def test_memory_transcript_one_search_per_version(fixture):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_memory_transcripts.py'
    spec = importlib.util.spec_from_file_location('prepare_memory_transcripts_pr', script)
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    result = build(fixture)
    index = {r['record_id']: (r['created_at'], r['kind'], pr.DOMAIN) for r in result['records']}
    builder = mt.Builder('parallel-recall', index, mt.Options(sequential_rate=0.0),
                         per_domain=False)
    ep = result['episodes']['train'][0]
    row, reason, _ = builder.build(ep, 'train')
    assert reason is None
    sites = row['search_sites']
    assert len(sites) == ep['redundancy']
    assert {tuple(s['record_ids']) for s in sites} == {tuple(g) for g in ep['sufficient_groups']}
    assert row['messages'][-1] == {'role': 'assistant', 'content': ep['answer']}
    assert not row['write_sites']
    assert 'translations' in row['messages'][0]['content']
