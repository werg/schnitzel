"""Citance-recall generator (``schnitz.citances``) on a project-authored fixture."""
from __future__ import annotations

import random

from schnitz import citances

WORDS = ('graph network kernel sparse attention encoder latent spectral convex robust '
         'gradient bound estimator sampling residual matrix prior tensor variational '
         'manifold policy reward channel signal protein lattice quantum field').split()


def _sentence(rng: random.Random, n: int = 14) -> str:
    return ' '.join(rng.choice(WORDS) for _ in range(n)).capitalize() + '.'


def _abstract(rng: random.Random) -> str:
    return ' '.join(_sentence(rng, 20) for _ in range(12))


def fixture(n_cited: int = 30, n_citing: int = 400, seed: int = 1):
    rng = random.Random(seed)
    cited = {f'https://openalex.org/W{i}': {'cited': f'https://openalex.org/W{i}',
                                              'arxiv_id': f'2001.{i:05d}',
                                              'title': f'Cited work number {i}',
                                              'abstract': _abstract(rng)}
             for i in range(n_cited)}
    citing = {f'1901.{j:05d}': {'arxiv_id': f'1901.{j:05d}', 'title': f'Citing paper {j}',
                                'license': 'http://creativecommons.org/licenses/by/4.0/'}
              for j in range(n_citing)}
    rows, sample = [], {}
    for i, cid in enumerate(cited):
        for j in rng.sample(sorted(citing), 6 + i % 30):
            before, after = _sentence(rng), _sentence(rng)
            core = _sentence(rng)[:-1]
            text = f'{before} {core} [1]}}, as shown before [2]}}. {after}\n'
            if i == 20 and j[-1] in '37':   # citances quoting the target abstract
                quote = cited[cid]['abstract'].split('.')[0]
                text = f'{before} {quote} [1]}}. {after}\n'
            sid = f's-{i}-{j}'
            sample[sid] = j
            rows.append({'_id': sid, 'text': text, 'marker': '[1]',
                         'marker_offsets': [[text.index('[1]}'), text.index('[1]}') + 3]],
                         'label': cid})
    return cited, citing, rows, sample


def test_markers_are_stripped_and_context_is_three_sentences():
    text = ('We start here. Our model follows the residual design [1]} and the gated '
            'variant [2]}, [3]} of earlier work. Results follow in Sec. 4 below.')
    ctx = citances.context(text, text.index('[1]}'))
    assert ctx['sentence'] == ('Our model follows the residual design and the gated variant '
                               'of earlier work.')
    assert ctx['before'] == 'We start here.'
    assert ctx['after'] == 'Results follow in Sec. 4 below.'
    assert citances.strip_markers('as in ([4]}, [5]}).') == 'as in.'
    assert citances.context('Too short [1]}.', 10) is None


def test_abstract_target_length():
    rng = random.Random(0)
    abstract = _abstract(rng)
    target = citances.abstract_target(abstract, citances.approx_tokens)
    assert 128 <= citances.approx_tokens(target) <= 256
    assert abstract.startswith(target)
    assert citances.abstract_target('Short abstract.', citances.approx_tokens) is None


def _build(**kw):
    cited, citing, rows, sample = fixture()
    wanted = {c: set(citing) for c in cited}
    contexts, _ = citances.collect(rows, sample, wanted)
    args = {'min_citances': 8, 'max_citances': 20, 'validation': 5, 'heldout_rate': 0.15}
    args.update(kw)
    return citances.build(contexts, cited, citing, **args), cited


def test_target_abstract_never_in_kb_and_redundancy():
    (recs, episodes, summary), cited = _build()
    kb_text = [r['text'] for r in recs]
    kb_grams = set().union(*(citances.ngrams(t) for t in kb_text))
    abstract_eps = [e for rows in episodes.values() for e in rows
                    if e['task_family'] == 'public_citance_abstract']
    assert abstract_eps
    for e in abstract_eps:
        assert not any(e['answer'] in t for t in kb_text)
        assert not citances.ngrams(e['answer']) & kb_grams
        full = cited[e['provenance']['cited']]['abstract']
        assert not any(full[:80] in t for t in kb_text)
    assert summary['build_filters'].get('citance_copies_abstract', 0) >= 1
    ids = {r['record_id'] for r in recs}
    for rows in episodes.values():
        for e in rows:
            assert e['redundancy'] == len(e['required_ids']) >= 8
            assert e['sufficient_groups'] == [[r] for r in e['required_ids']]
            assert e['alternatives'] == [e['required_ids']]
            assert set(e['required_ids']) <= ids
            assert e['provenance']['cited_title'] not in ' '.join(
                s['text'] for s in e['supports'])


def test_split_disjoint_by_cited_paper_and_heldout_citers():
    (recs, episodes, _), _ = _build()
    papers = {s: {e['provenance']['cited'] for e in rows} for s, rows in episodes.items()}
    assert len(papers['validation']) == 5
    assert papers['train'] and not papers['train'] & papers['validation']
    stored_citers = {r['provenance']['citing_arxiv'] for r in recs}
    assert not any(citances.heldout_citer(c, 0.15) for c in stored_citers)
    kb_text = ' '.join(r['text'] for r in recs)
    descriptions = [e for rows in episodes.values() for e in rows
                    if e['task_family'] == 'public_citance_description']
    assert descriptions
    kb_grams = set().union(*(citances.ngrams(r['text']) for r in recs))
    for e in descriptions:
        assert e['provenance']['citing_arxiv'] not in stored_citers
        assert e['answer'] not in kb_text
        assert not citances.ngrams(e['answer']) & kb_grams
        assert e['answer'] not in e['query']


def test_build_is_deterministic():
    (a, ea, _), _ = _build()
    (b, eb, _), _ = _build()
    assert [r['record_id'] for r in a] == [r['record_id'] for r in b]
    assert [e['episode_id'] for e in ea['train']] == [e['episode_id'] for e in eb['train']]


def test_titles_agree():
    assert citances.titles_agree('Planck 2018 results. VIII. Gravitational lensing',
                                 'Planck 2018 results')
    assert citances.titles_agree('Deep Residual Learning for Image Recognition',
                                 'Deep residual learning for image recognition.')
    assert not citances.titles_agree('RoBERTa: A Robustly Optimized BERT Pretraining Approach',
                                     'Historiae, history of socio-cultural transformation')


def test_description_target_reused_elsewhere_is_dropped():
    """A held-out citing sentence that another (stored) paper repeats verbatim, e.g. as
    a neighbouring sentence, is not a target."""
    cited, citing, rows, sample = fixture()
    (_, episodes, _), _ = _build()
    target = next(e for rows_ in episodes.values() for e in rows_
                  if e['task_family'] == 'public_citance_description')
    cid = target['provenance']['cited']
    for row in rows:        # every stored context of that cited paper repeats the sentence
        if row['label'] == cid and sample[row['_id']] != target['provenance']['citing_arxiv']:
            head = row['text'].split('[2]}. ')[0]
            row['text'] = f"{head}[2]}}. {target['answer']}\n"
    contexts, _ = citances.collect(rows, sample, {c: set(citing) for c in cited})
    _, again, summary = citances.build(contexts, cited, citing, min_citances=8,
                                       max_citances=20, validation=5, heldout_rate=0.15)
    assert target['episode_id'] not in {e['episode_id'] for r in again.values() for e in r}
    assert summary['build_filters']['description_target_in_kb'] >= 1
