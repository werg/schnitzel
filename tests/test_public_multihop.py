"""MuSiQue / 2WikiMultihopQA / HoVer preparation on tiny project-authored fixtures."""
from __future__ import annotations

import bz2
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import prepare_public_multihop as multihop  # noqa: E402
from public_corpus_common import PROMPT  # noqa: E402

FILLER = 'This filler sentence describes the local weather and nothing else of interest.'


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _paragraph(idx, title, text, supporting=False):
    return {'idx': idx, 'title': title, 'paragraph_text': text, 'is_supporting': supporting}


def _musique_pair(tag: str, *, town_tag: str | None = None,
                  leak_question: str | None = None) -> list[dict]:
    """A 2-hop pair: who founded the town where X was born?"""
    town_tag = town_tag or tag
    person, town, founder = f'Ada {tag}', f'Brookvale {town_tag}', f'Corin {town_tag}'
    long_town = ' '.join([FILLER] * 8 + [f'{town} was founded by {founder} in 1650.'])
    distractors = [_paragraph(i, f'Other {tag} {i}', f'Other {tag} {i} is a quiet lake. {FILLER}')
                   for i in range(2, 6)]
    hop1 = leak_question or f'Where was {person} born?'
    decomposition = [
        {'id': 1, 'question': hop1, 'answer': town, 'paragraph_support_idx': 0},
        {'id': 2, 'question': 'Who founded #1?', 'answer': founder, 'paragraph_support_idx': 1},
    ]
    question = f'Who founded the town where {person} was born?'
    answerable = {
        'id': f'2hop__{tag}', 'question': question, 'answer': founder,
        'answer_aliases': [], 'answerable': True, 'question_decomposition': decomposition,
        'paragraphs': [_paragraph(0, person, f'{person} is a poet. She was born in {town}.', True),
                       _paragraph(1, town, long_town, True), *distractors]}
    replaced = _paragraph(1, f'Elsewhere {tag}', f'Elsewhere {tag} is a village. {FILLER}')
    unanswerable = {
        **answerable, 'answerable': False,
        'question_decomposition': [dict(decomposition[0]),
                                   {**decomposition[1], 'paragraph_support_idx': None}],
        'paragraphs': [answerable['paragraphs'][0], replaced, *distractors]}
    return [answerable, unanswerable]


def _musique_raw(raw: Path) -> None:
    data = raw / 'musique' / 'data'
    _jsonl(data / 'musique_full_v1.0_train.jsonl', _musique_pair('t1') + _musique_pair('t2'))
    _jsonl(data / 'musique_full_v1.0_dev.jsonl',
           _musique_pair('d1') + _musique_pair('d2', leak_question='Which harbor city is old?')
           + _musique_pair('d3', town_tag='d1'))
    listed = {'squad2': [{'id': 'x', 'question': 'which harbor city is old'}],
              'natural_questions': [], 'zerore': [], 'mlqaen': [], 'trex': []}
    (data / 'dev_test_singlehop_questions_v1.0.json').write_text(json.dumps(listed))


def _twowiki_raw(raw: Path) -> None:
    folder = raw / '2wikimultihopqa'
    folder.mkdir(parents=True)

    def row(identifier, question, answer, kind, support, extra=()):
        context = [['Film Alpha', ['Film Alpha is a 1990 drama.', 'It was directed by Ben Ray.']],
                   ['Ben Ray', ['Ben Ray is a director.', 'He was born in Oslo.']],
                   ['Film Beta', ['Film Beta is a 1985 comedy.', 'It was directed by Cy Lo.']],
                   *extra]
        return {'_id': identifier, 'question': question, 'answer': answer, 'type': kind,
                'supporting_facts': support, 'context': context, 'evidences': [],
                'entity_ids': ''}

    train = [
        row('w1', 'Where was the director of Film Alpha born?', 'Oslo', 'compositional',
            [['Film Alpha', 1], ['Ben Ray', 1]],
            extra=[['Noise', ['Noise is filler.']], ['More Noise', ['Still filler.']]]),
        row('w2', 'Which film came out first, Film Alpha or Film Beta?', 'Film Beta',
            'comparison', [['Film Alpha', 0], ['Film Beta', 0]]),
        row('w3', 'Are Film Alpha and Film Beta both dramas?', 'no', 'comparison',
            [['Film Alpha', 0], ['Film Beta', 0]]),
        row('w4', 'Who directed Film Gamma?', 'Dee', 'compositional', [['Film Gamma', 0]]),
    ]
    dev = [row('v1', 'Who directed the 1990 drama Film Alpha?', 'Ben Ray', 'inference',
               [['Film Alpha', 1]])]
    (folder / 'train.json').write_text(json.dumps(train), encoding='utf-8')
    (folder / 'dev.json').write_text(json.dumps(dev), encoding='utf-8')


def _hover_raw(raw: Path) -> None:
    folder = raw / 'hover'
    folder.mkdir(parents=True)
    articles = [
        {'id': '1', 'title': 'Mara Vell', 'text': ['Mara Vell is a sculptor.',
                                                   ' She studied under Tor Brin in 1901.']},
        {'id': '2', 'title': 'Tor Brin', 'text': ['Tor Brin was a painter from Skagen.',
                                                  ' He favoured naturalism.']},
        {'id': '3', 'title': 'Unused Page', 'text': ['Nothing here matters.']},
    ]
    inner = bz2.compress(''.join(json.dumps(a) + '\n' for a in articles).encode())
    with tarfile.open(folder / multihop.HOVER_WIKI, 'w:bz2') as archive:
        info = tarfile.TarInfo('dump/AA/wiki_00.bz2')
        info.size = len(inner)
        archive.addfile(info, io.BytesIO(inner))
    support = [['Mara Vell', 1], ['Tor Brin', 1]]
    train = [
        {'uid': 'h1', 'claim': 'The sculptor Mara Vell studied under a naturalist painter.',
         'supporting_facts': support, 'label': 'SUPPORTED', 'num_hops': 2, 'hpqa_id': 'a'},
        {'uid': 'h2', 'claim': 'Mara Vell studied under a painter who rejected naturalism.',
         'supporting_facts': support, 'label': 'NOT_SUPPORTED', 'num_hops': 2, 'hpqa_id': 'a'},
        {'uid': 'h3', 'claim': 'Missing Person painted landscapes.',
         'supporting_facts': [['Missing Person', 0]], 'label': 'SUPPORTED', 'num_hops': 2},
    ]
    dev = [{'uid': 'h4', 'claim': 'Tor Brin was a painter from Skagen.',
            'supporting_facts': [['Tor Brin', 0]], 'label': 'SUPPORTED', 'num_hops': 2}]
    (folder / 'hover_train_release_v1.1.json').write_text(json.dumps(train))
    (folder / 'hover_dev_release_v1.1.json').write_text(json.dumps(dev))


def _check_common(folder: Path) -> dict[str, dict]:
    sources = {row['record_id']: row for row in _read(folder / 'sources.jsonl')}
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['sources'] == len(sources)
    for path in folder.glob('episodes-*.jsonl'):
        for row in _read(path):
            assert row['sufficient_groups'] == [row['required_ids']]
            assert row['answer'].lower() not in row['query'][len(PROMPT):].lower() or \
                row['answer'] in {'supported', 'not supported'}
            assert set(row['required_ids']) <= {s['record_id'] for s in row['supports']}
            assert len(row['supports']) - len(row['required_ids']) <= 2
            assert all(s['created_at'] < row['query_time'] for s in row['supports'])
            assert all(s['record_id'] in sources for s in row['supports'])
    return sources


def test_json_array_streaming_with_tiny_blocks():
    rows = [{'a': i, 'text': 'x, ] [ y' * i} for i in range(5)]
    handle = io.StringIO(' \n' + json.dumps(rows, indent=1))
    assert list(multihop.iter_json_array(handle, block=7)) == rows
    assert list(multihop.iter_json_array(io.StringIO('[]'))) == []


def test_musique_answerable_leak_and_unanswerable_contrast(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _musique_raw(raw)
    summary = multihop.build_musique(
        raw, out, train=5, validation=5, max_sources=100, unanswerable_train=1,
        unanswerable_validation=1, leak_seeds={'squad2'}, tokenizer=None,
        distractors=2, seed=3)
    folder = out / 'musique'
    sources = _check_common(folder)
    train = _read(folder / 'episodes-train.jsonl')
    unanswerable = _read(folder / 'episodes-train-unanswerable.jsonl')
    # One train pair becomes the contrast episode; its answerable twin is never stored.
    assert len(train) == 1 and len(unanswerable) == 1
    item = train[0]
    tag = item['provenance']['musique_id'].split('__')[1]
    other = unanswerable[0]['provenance']['musique_id'].split('__')[1]
    assert {tag, other} == {'t1', 't2'}
    gold = [sources[key]['text'] for key in item['required_ids']]
    # Hop 1 paragraph fits one chunk; the long town paragraph keeps only its answer chunk.
    assert len(gold) == 2
    assert any(f'born in Brookvale {tag}' in text for text in gold)
    assert any(f'founded by Corin {tag}' in text for text in gold)
    assert all(len(text.split('Passage: ', 1)[1]) <= 700 for text in gold)
    assert item['answer'] == f'Corin {tag}'
    contrast = unanswerable[0]
    assert contrast['answer'] == 'unanswerable'
    assert contrast['support_annotation'] == 'unanswerable_contrast'
    assert not any(f'Brookvale {other}' in row['text'] and 'founded' in row['text']
                   for row in sources.values())
    assert [sources[key]['text'] for key in contrast['required_ids']] == [
        f'Title: Ada {other}\nPassage: Ada {other} is a poet. She was born in Brookvale {other}.']
    assert summary['filtered']['train'] == {'contrast_twin': 1}
    # Validation: d2 overlaps a SQuAD dev/test single hop. d1 and d3 share the town
    # paragraph, so neither can be a contrast episode (its removed evidence supports
    # another question); both answerable halves are kept.
    assert not (folder / 'episodes-validation-unanswerable.jsonl').exists()
    assert summary['filtered']['validation-unanswerable'] == {
        'seed_overlap_squad2': 1, 'removed_evidence_shared': 2}
    validation = _read(folder / 'episodes-validation.jsonl')
    assert sorted(row['provenance']['musique_id'] for row in validation) == [
        '2hop__d1', '2hop__d3']
    assert summary['filtered']['validation'] == {'seed_overlap_squad2': 1}
    assert summary['seed_overlap_ids'] == {'validation': 1, 'train': 0}


def test_twowiki_sentence_support_and_filters(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _twowiki_raw(raw)
    summary = multihop.build_twowiki(raw, out, train=10, validation=10, max_sources=100,
                                     tokenizer=None, distractors=2, seed=1, candidates=100,
                                     yes_no_fraction=1.0)
    folder = out / '2wikimultihopqa'
    sources = _check_common(folder)
    train = {row['provenance']['twowiki_id']: row
             for row in _read(folder / 'episodes-train.jsonl')}
    assert set(train) == {'w1', 'w3'}
    assert summary['filtered']['train'] == {'answer_in_query': 1, 'missing_support': 1}
    gold = [sources[key]['text'] for key in train['w1']['required_ids']]
    assert sorted(gold) == ['Title: Ben Ray\nPassage: Ben Ray is a director. He was born in Oslo.',
                            'Title: Film Alpha\nPassage: Film Alpha is a 1990 drama. '
                            'It was directed by Ben Ray.']
    assert len(train['w1']['supports']) == 4
    assert train['w3']['provenance']['yes_no'] is True
    assert summary['kept_by_type']['train/yes_no_answer'] == 1
    assert len(_read(folder / 'episodes-validation.jsonl')) == 1
    capped = multihop.build_twowiki(raw, tmp_path / 'capped', train=10, validation=10,
                                    max_sources=100, tokenizer=None, distractors=2, seed=1,
                                    candidates=100, yes_no_fraction=0.25)
    assert capped['filtered']['train']['yes_no_cap'] == 1
    assert 'train/yes_no_answer' not in capped['kept_by_type']


def test_hover_claims_use_wiki_sentences(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _hover_raw(raw)
    summary = multihop.build_hover(raw, out, train=10, validation=10, max_sources=100,
                                   tokenizer=None, distractors=2, seed=1, wiki_path=None)
    folder = out / 'hover'
    sources = _check_common(folder)
    train = {row['provenance']['hover_uid']: row for row in _read(folder / 'episodes-train.jsonl')}
    assert set(train) == {'h1', 'h2'}
    assert summary['filtered']['train'] == {'missing_wiki_article': 1}
    assert train['h1']['answer'] == 'supported' and train['h2']['answer'] == 'not supported'
    assert train['h1']['query'] == (multihop.HOVER_PROMPT + 'The sculptor Mara Vell studied '
                                    'under a naturalist painter.')
    texts = sorted(sources[key]['text'] for key in train['h1']['required_ids'])
    assert texts == ['Title: Mara Vell\nPassage: Mara Vell is a sculptor. She studied under '
                     'Tor Brin in 1901.',
                     'Title: Tor Brin\nPassage: Tor Brin was a painter from Skagen. '
                     'He favoured naturalism.']
    assert summary['wiki_articles_found'] == 2 and summary['wiki_articles_needed'] == 3
    assert not any('Unused Page' in row['text'] for row in sources.values())


def test_cli_runs_all_three(tmp_path, monkeypatch):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    _musique_raw(raw)
    _twowiki_raw(raw)
    _hover_raw(raw)
    monkeypatch.setattr(sys, 'argv', [
        'prepare_public_multihop.py', '--raw', str(raw), '--output', str(out), '--no-tokenizer',
        '--2wiki-candidates', '10'])
    multihop.main()
    assert {path.name for path in out.iterdir()} == {'musique', '2wikimultihopqa', 'hover'}
    for folder in out.iterdir():
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['domain'] == folder.name
