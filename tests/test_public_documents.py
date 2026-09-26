"""QASPER / ConditionalQA episode construction on tiny project-authored fixtures."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import prepare_public_documents as documents
    from public_corpus_common import PROMPT
finally:
    sys.path.pop(0)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _answer(evidence, *, spans=(), free='', yes_no=None, unanswerable=False, highlights=()):
    return {'answer': {'unanswerable': unanswerable, 'extractive_spans': list(spans),
                       'yes_no': yes_no, 'free_form_answer': free, 'evidence': list(evidence),
                       'highlighted_evidence': list(highlights)},
            'annotation_id': 'a' + str(len(evidence))}


def _paper(prefix: str) -> dict:
    method = f'{prefix} The zebra encoder uses sparse gates. It was tuned on held-out data.'
    data = f'{prefix} We train on the Quokka corpus of forum posts. It has ten splits.'
    result = f'{prefix} Accuracy improved by four points. Gains were largest for rare words.'
    return {
        'title': f'{prefix} Sparse Gates',
        'abstract': f'{prefix} We study sparse gates for encoders.',
        'full_text': [{'section_name': 'Method', 'paragraphs': [method, '']},
                      {'section_name': 'Data', 'paragraphs': [data]},
                      {'section_name': 'Results', 'paragraphs': [result]}],
        'qas': [
            {'question': 'Which corpus is used?', 'question_id': f'{prefix}-q1',
             'answers': [_answer([data], spans=['Quokka corpus'])]},
            {'question': 'Does the method help rare words and what encoder is used?',
             'question_id': f'{prefix}-q2',
             'answers': [_answer([method, result], yes_no=True)]},
            {'question': 'What is the license?', 'question_id': f'{prefix}-q3',
             'answers': [_answer([], unanswerable=True)]},
            {'question': 'What is in table 2?', 'question_id': f'{prefix}-q4',
             'answers': [_answer(['FLOAT SELECTED: Table 2: scores'], free='scores'),
                         _answer([result], free='four points')]},
            {'question': 'Describe everything.', 'question_id': f'{prefix}-q5',
             'answers': [_answer([result], free=' '.join(['word'] * 30))]},
        ],
        'figures_and_tables': []}


def _qasper_archive(root: Path) -> Path:
    root.mkdir(parents=True)
    path = root / 'qasper-train-dev-v0.3.tgz'
    with tarfile.open(path, 'w:gz') as handle:
        for name, prefix in (('qasper-train-v0.3.json', 'Alpha'), ('qasper-dev-v0.3.json', 'Beta')):
            data = json.dumps({f'{prefix}-paper': _paper(prefix)}).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    return root


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_qasper_gold_query_and_filters(tmp_path):
    raw = _qasper_archive(tmp_path / 'raw')
    summary = documents.build_qasper(raw, tmp_path / 'out', tokenizer=WordTokenizer())
    assert summary['episodes'] == {'train': 3, 'validation': 3}
    filtered = summary['filtered']['train']
    assert filtered['unanswerable'] == 1 and filtered['answer_too_long'] == 1
    rows = {row['episode_id']: row for row in _read(tmp_path / 'out/episodes-train.jsonl')}
    sources = {row['record_id']: row for row in _read(tmp_path / 'out/sources.jsonl')}

    single = rows['qasper-Alpha-q1']
    assert single['query'] == PROMPT + 'In the paper "Alpha Sparse Gates": Which corpus is used?'
    assert single['answer'] == 'Quokka corpus'
    assert len(single['required_ids']) == 1
    gold = sources[single['required_ids'][0]]
    assert 'Quokka corpus' in gold['text'] and gold['provenance']['section'] == 'Data'
    assert gold['text'].startswith('Title: Alpha Sparse Gates — Data\nPassage: ')
    assert gold['provenance']['article_title'] == 'qasper/Alpha Sparse Gates — Data'
    assert single['query_time'] == 2 and all(s['created_at'] == 1 for s in single['supports'])
    distractors = [s for s in single['supports'] if s['record_id'] not in single['required_ids']]
    assert len(distractors) == 2
    assert all(sources[s['record_id']]['provenance']['paper_id'] == 'Alpha-paper'
               for s in distractors)

    multi = rows['qasper-Alpha-q2']
    assert multi['answer'] == 'yes' and len(multi['required_ids']) == 2
    assert multi['sufficient_groups'] == [multi['required_ids']]

    # A table-only first annotator falls back to the next annotator with textual evidence.
    fallback = rows['qasper-Alpha-q4']
    assert fallback['answer'] == 'four points'
    assert fallback['provenance']['annotator_rank'] == 1
    validation = _read(tmp_path / 'out/episodes-validation.jsonl')
    assert {row['provenance']['paper_id'] for row in validation} == {'Beta-paper'}


def test_qasper_paragraph_spanning_chunks_narrowed_by_highlight():
    long = ' '.join(f'Sentence {i} is filler text about encoders and gates.' for i in range(30))
    paragraph = long + ' The key number is 4242 for the zebra model.'
    rows, owner = documents._document_chunks(
        'qasper', [('Method', [paragraph])], document_title='T', provenance={})
    candidates = owner[(0, 0)]
    assert len(candidates) > 1
    kept = documents._narrow(rows, candidates, ['The key number is 4242 for the zebra model.'])
    assert len(kept) == 1 and '4242' in rows[kept[0]]['text']
    assert documents._narrow(rows, candidates, []) == candidates


def _conditionalqa(root: Path) -> Path:
    root.mkdir(parents=True)
    contents = ['<h1>Overview</h1>', '<p>You can claim Gizmo Allowance if you are over 60.</p>',
                '<p>You must live in Wales.</p>', '<h1>How to claim</h1>',
                '<p>Call the Gizmo line within 30 days.</p>', '<li>Bring &amp; show your card.</li>']
    documents_json = [{'title': 'Gizmo Allowance', 'url': 'https://example.test/gizmo',
                       'contents': contents}]
    rows = [
        {'id': 'train-0', 'url': 'https://example.test/gizmo', 'scenario': 'I am 70 and live in Cardiff.',
         'question': 'Can I claim?', 'not_answerable': False,
         'answers': [['yes', ['You must live in Wales.']]],
         'evidences': [contents[1], contents[2]]},
        {'id': 'train-1', 'url': 'https://example.test/gizmo', 'scenario': 'I qualify.',
         'question': 'How long do I have to call?', 'not_answerable': False,
         'answers': [['30 days', []], ['within a month', []]], 'evidences': [contents[4]]},
        {'id': 'train-2', 'url': 'https://example.test/gizmo', 'scenario': 'I am 20.',
         'question': 'What colour is the card?', 'not_answerable': True, 'answers': [],
         'evidences': []},
        {'id': 'train-3', 'url': 'https://example.test/gizmo',
         'scenario': ' '.join(['word'] * 200), 'question': 'Can I claim?',
         'not_answerable': False, 'answers': [['no', []]], 'evidences': [contents[1]]},
    ]
    (root / 'documents.json').write_text(json.dumps(documents_json))
    (root / 'train.json').write_text(json.dumps(rows))
    (root / 'dev.json').write_text(json.dumps([dict(rows[1], id='dev-1')]))
    return root


def test_conditionalqa_scenario_query_answers_and_gold(tmp_path):
    raw = _conditionalqa(tmp_path / 'raw')
    summary = documents.build_conditionalqa(raw, tmp_path / 'out', tokenizer=WordTokenizer())
    assert summary['episodes'] == {'train': 2, 'validation': 1}
    assert summary['filtered']['train'] == {'unanswerable': 1, 'query_tokens': 1}
    rows = {row['episode_id']: row for row in _read(tmp_path / 'out/episodes-train.jsonl')}
    sources = {row['record_id']: row for row in _read(tmp_path / 'out/sources.jsonl')}
    eligible = rows['conditionalqa-train-0']
    assert eligible['query'] == PROMPT + 'I am 70 and live in Cardiff. Can I claim?'
    assert eligible['answer'] == 'yes'
    assert eligible['provenance']['conditions'] == ['You must live in Wales.']
    [gold] = eligible['required_ids']
    assert 'over 60' in sources[gold]['text'] and 'live in Wales' in sources[gold]['text']
    assert sources[gold]['text'].startswith('Title: Gizmo Allowance — Overview\n')
    deadline = rows['conditionalqa-train-1']
    assert deadline['answer'] == '30 days; within a month'
    assert 'Bring & show your card.' in sources[deadline['required_ids'][0]]['text']
    assert '<' not in ''.join(row['text'] for row in sources.values())
