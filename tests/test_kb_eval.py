import json
import math

import pytest

from schnitz.kb_eval import (apply_edit, apply_removal, captured_fraction, content_nats,
                             counterfactual_edits, dependence_report, domain_removal,
                             effective_count, follows, holdout_insertion, insertion_report,
                             nll_summary, removal_report, retention, source_composition,
                             superposition_report, synthetic_facts, synthetic_insertion)
from schnitz.task_verifiers import check_episode


def _record(rid, text, domain='d', kind='passage', **provenance):
    return {'record_id': rid, 'text': text, 'domain': domain, 'kind': kind, 'created_at': 1,
            'provenance': {'dataset': domain, **provenance}}


def _episode(eid, query, answer, required, records, verify=None, **provenance):
    episode = {'episode_id': eid, 'query': query, 'answer': answer, 'required_ids': required,
               'sufficient_groups': [required], 'query_time': 2,
               'supports': [{'record_id': r, 'text': records[r]['text'], 'created_at': 1,
                             'kind': records[r]['kind']} for r in required],
               'provenance': {'domain': records[required[0]]['domain'] if required else 'd',
                              **provenance}}
    if verify is not None:
        episode['verify'] = verify
    return episode


# -- superposition ---------------------------------------------------------------
def test_source_composition_follows_rewrite_lineage():
    lineage = {'i1': ['s1', 's2'], 'i2': {'s2': 1, 's3': 3}, 'r1': ['i1', 'i2']}
    comp = source_composition(lineage)
    assert comp['i1'] == {'s1': 0.5, 's2': 0.5}
    assert comp['r1'] == pytest.approx({'s1': 0.25, 's2': 0.375, 's3': 0.375})
    assert sum(comp['r1'].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        source_composition({'a': ['b'], 'b': ['a']})


def test_effective_count_entropy_and_participation():
    assert effective_count([5.0]) == {'entropy': 1.0, 'participation': 1.0, 'top1': 1.0}
    four = effective_count([0.2] * 4)
    assert four['entropy'] == pytest.approx(4) and four['participation'] == pytest.approx(4)
    skewed = effective_count([3, 1, 0])
    assert skewed['participation'] == pytest.approx(1.6)
    assert skewed['entropy'] == pytest.approx(math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25))))
    # scaling all gates together changes only the total mass, not the effective count
    assert effective_count([30, 10]) == pytest.approx(effective_count([3, 1]))
    assert effective_count([0, 0])['entropy'] == 0.0
    with pytest.raises(ValueError):
        effective_count([1, -1])


def test_superposition_report_separates_localized_from_superposed_items():
    localized = source_composition({f'i{k}': [f's{k}'] for k in range(4)})
    superposed = source_composition({f'i{k}': [f's{k}', f's{(k + 1) % 4}'] for k in range(4)})
    loc = superposition_report(localized, reads=[{'i0': 1.0}, [0.5, 0.5, 0.0]])
    sup = superposition_report(superposed, reads=[[1, 1, 1, 1]])
    assert loc['sources_per_item']['mean'] == 1 and sup['sources_per_item']['mean'] == 2
    assert loc['items_per_source']['mean'] == 1 and sup['items_per_source']['mean'] == 2
    assert sup['effective_sources_per_item']['p50'] == pytest.approx(2)
    assert loc['effective_items_per_read']['entropy']['max'] == pytest.approx(2)
    assert sup['effective_items_per_read']['participation']['mean'] == pytest.approx(4)
    # a trace share below the threshold does not count as a served source
    traced = source_composition({'i0': {'s0': 0.99, 's1': 0.01}})
    assert superposition_report(traced, min_share=0.05)['sources_per_item']['max'] == 1


def test_retention_counts_forgotten_and_gained():
    report = retention({'a': 1, 'b': 1, 'c': 0, 'x': 1}, {'a': 1, 'b': 0, 'c': 1, 'y': 0})
    assert report['n'] == 3 and report['forgotten'] == 1 and report['gained'] == 1
    assert report['retained'] == 1.0 and report['delta'] == 0
    nll = retention({'a': 2.0}, {'a': 2.5}, higher_is_better=False)
    assert nll['delta'] == 0.5 and 'retained' not in nll


# -- counterfactual edits ----------------------------------------------------------
def _tool_kb():
    doc = ('Tool: translate\nDescription: Translate text with translate.\n'
           'Parameters: {"text": {"type": "str"}, "target": {"type": "str"}}')
    records = {'t1': _record('t1', doc, 'xlam', 'tool_doc', tool='translate'),
               't2': _record('t2', 'Tool: weather\nDescription: Weather.\nParameters: {}', 'xlam',
                             'tool_doc', tool='weather')}
    gold = [{'name': 'translate', 'arguments': {'text': 'Ciao', 'target': 'en'}}]
    ep = _episode('x1', 'Request: What does Ciao mean in English?', json.dumps(gold), ['t1'], records,
                  {'type': 'calls', 'gold': gold})
    leak = _episode('x2', 'Request: use translate on Hola', '[]', ['t1'], records,
                    {'type': 'calls', 'gold': gold})
    return records, ep, leak


def test_tool_rename_edit_expects_the_new_name_and_scores_kb_vs_prior():
    records, ep, leak = _tool_kb()
    edits, coverage = counterfactual_edits(records, [ep, leak])
    assert coverage['edited'] == 1 and coverage['skipped'] == {'tool_rename:name_in_query': 1}
    edit = edits[0]
    new_name = edit.replacement
    assert new_name not in json.dumps(records) and edit.original == 'translate'
    new_record = edit.records['t1']
    assert new_record['record_id'] != 't1' and new_record['provenance']['counterfactual_of'] == 't1'
    assert 'translate' not in new_record['text'] and f'Tool: {new_name}\n' in new_record['text']
    kb = apply_edit(records, edit)
    assert 't1' not in kb and new_record['record_id'] in kb and 't2' in kb
    probe = edit.episode
    assert probe['required_ids'] == [new_record['record_id']]
    assert probe['supports'][0]['text'] == new_record['text']
    assert records['t1']['text'].startswith('Tool: translate')  # the original is untouched
    new_call = f'[{{"name": "{new_name}", "arguments": {{"text": "Ciao", "target": "en"}}}}]'
    old_call = '[{"name": "translate", "arguments": {"text": "Ciao", "target": "en"}}]'
    labels = [follows(new_call, edit), follows(old_call, edit), follows('no idea', edit)]
    assert labels == ['kb', 'prior', 'neither']
    assert dependence_report(labels)['follows_kb'] == pytest.approx(1 / 3, abs=1e-4)


def test_parameter_rename_changes_the_argument_key():
    records, ep, _ = _tool_kb()
    edits, _ = counterfactual_edits(records, [ep], rules=('parameter_rename',))
    edit = edits[0]
    assert edit.original == 'text'
    assert f'"{edit.replacement}"' in edit.records['t1']['text']
    assert edit.episode['verify']['gold'][0]['arguments'] == {edit.replacement: 'Ciao', 'target': 'en'}


def test_value_rename_in_stored_tables_is_conservative():
    rows = _record('r1', "Database music, table singer (all rows; columns id, name, country):\n"
                   "(1, 'Joe Sharp', 'Netherlands')\n(2, 'Ann Lee', 'France')", 'spider_memory',
                   'table_rows', db='music')
    values = _record('r2', "Database music, values of singer.country:\n'Netherlands'\n'France'",
                     'spider_memory', 'column_values', db='music')
    other = _record('r3', "Database geo, table c:\n(1, 'Netherlands')", 'spider_memory',
                    'table_rows', db='geo')
    records = {r['record_id']: r for r in (rows, values, other)}
    ask = _episode('m1', 'Question: What is the country of Joe Sharp?', 'Netherlands', ['r1'],
                   records, {'type': 'values', 'rows': [['Netherlands']]}, db_id='music')
    ordered = _episode('m2', 'Question: List countries in alphabetical order.', 'France; Netherlands',
                       ['r1'], records, {'type': 'values', 'rows': [['France'], ['Netherlands']]},
                       db_id='music')
    numeric = _episode('m3', 'Question: How many singers?', '2', ['r1'], records,
                       {'type': 'values', 'rows': [[2]]}, db_id='music')
    edits, coverage = counterfactual_edits(records, [ask, ordered, numeric])
    assert coverage['edited'] == 1
    assert coverage['skipped'] == {'value_rename:spelling_sensitive_question': 1,
                                   'value_rename:no_string_cell': 1}
    edit = edits[0]
    assert set(edit.records) == {'r1', 'r2'}  # every record of the database, not other databases
    new = edit.replacement
    assert new[0].isupper() and 'Netherlands' not in edit.records['r1']['text']
    assert "'Joe Sharp'" in edit.records['r1']['text']  # other cells unchanged
    assert edit.episode['verify'] == {'type': 'values', 'rows': [[new]]}
    assert follows(f'{new}', edit) == 'kb' and follows('Netherlands', edit) == 'prior'


def test_answer_span_edit_for_short_qa_and_coverage_of_other_tasks(tmp_path):
    passage = _record('p1', 'Title: Fighter\nPassage: "Fighter" was recorded by Christina Aguilera '
                      'in 2002.', 'research')
    unrelated = _record('p2', 'Christina Aguilera also sang Beautiful.', 'research')
    elsewhere = _record('p3', 'Christina Aguilera in another KB.', 'other')
    records = {r['record_id']: r for r in (passage, unrelated, elsewhere)}
    qa = _episode('q1', 'Who recorded the song Fighter?', 'Christina Aguilera', ['p1'], records)
    yes = _episode('q2', 'Was Fighter recorded in 2002?', 'yes', ['p1'], records)
    sql = _episode('s1', 'Write SQL.', 'SELECT 1', ['p1'], records,
                   {'type': 'sql', 'db': str(tmp_path / 'x.sqlite'), 'gold': 'SELECT 1'})
    edits, coverage = counterfactual_edits(records, [qa, yes, sql])
    assert coverage['edited'] == 1 and coverage['coverage'] == pytest.approx(1 / 3, abs=1e-4)
    assert coverage['skipped'] == {'answer_span:answer_not_a_span': 1, 'not_derivable': 1}
    edit = edits[0]
    assert set(edit.records) == {'p1', 'p2'}  # consistently within the domain only
    assert len(edit.replacement.split()) == 2
    assert check_episode(f'It was {edit.replacement}.', edit.episode['verify'])
    assert follows('Christina Aguilera', edit) == 'prior'


def test_edits_are_deterministic_per_seed():
    records, ep, _ = _tool_kb()
    first = counterfactual_edits(records, [ep], seed=3)[0][0].replacement
    assert counterfactual_edits(records, [ep], seed=3)[0][0].replacement == first
    assert counterfactual_edits(records, [ep], seed=4)[0][0].replacement != first


# -- removal and insertion ---------------------------------------------------------
def _two_domains():
    records = {'a1': _record('a1', 'alpha one', 'a'), 'a2': _record('a2', 'alpha two', 'a'),
               'b1': _record('b1', 'beta one', 'b')}
    episodes = [_episode('ea', 'q', 'x', ['a1'], records), _episode('eb', 'q', 'y', ['b1'], records),
                _episode('eab', 'q', 'z', ['a2', 'b1'], records)]
    return records, episodes


def test_domain_removal_probes_and_controls():
    records, episodes = _two_domains()
    test = domain_removal(records, episodes, domain='a')
    assert test.removed_ids == {'a1', 'a2'} and test.probes == ['ea', 'eab'] and test.controls == ['eb']
    assert set(apply_removal(records, test)) == {'b1'}
    report = removal_report(test, before={'ea': 1, 'eab': 1, 'eb': 1}, after={'ea': 0, 'eab': 1, 'eb': 1},
                            closed_book={'ea': 0, 'eab': 0, 'eb': 0})
    assert report['probes']['after'] == 0.5 and report['controls']['after'] == 1
    assert report['surviving_gain'] == 0.5
    with pytest.raises(ValueError):
        domain_removal(records, episodes)


def test_holdout_insertion_splits_probes():
    records, episodes = _two_domains()
    test = holdout_insertion(records, episodes, fraction=1 / 3, seed=0)
    assert len(test.new_ids) == 1 and test.base_ids | test.new_ids == set(records)
    for eid in test.new_probes:
        need = set(next(e for e in episodes if e['episode_id'] == eid)['required_ids'])
        assert need & test.new_ids
    for eid in test.old_probes:
        need = set(next(e for e in episodes if e['episode_id'] == eid)['required_ids'])
        assert need <= test.base_ids
    assert len(test.new_probes) + len(test.old_probes) == 3


def test_synthetic_facts_are_novel_and_verifiable():
    records, episodes = _two_domains()
    test = synthetic_insertion(records, episodes, count=8, seed=1)
    assert len(test.extra_records) == 8 and len(test.new_probes) == 8
    assert test.old_probes == ['ea', 'eb', 'eab']
    words = {w for r in records.values() for w in r['text'].split()}
    for episode in test.extra_episodes:
        (rid,) = episode['required_ids']
        text = test.extra_records[rid]['text']
        assert episode['answer'] in text and not set(text.split()) & words
        assert episode['answer'] not in episode['query']
        assert check_episode(f'{episode["answer"]}', episode['verify'])
    again = synthetic_facts(8, seed=1, avoid=(r['text'] for r in records.values()))
    assert again[0] == test.extra_records
    report = insertion_report(test, old_before={'ea': 1, 'eb': 1, 'eab': 0},
                              old_after={'ea': 1, 'eb': 0, 'eab': 0},
                              new_after={k: 1 for k in test.new_probes},
                              new_before={k: 0 for k in test.new_probes})
    assert report['retention']['forgotten'] == 1 and report['new'] == {'n': 8, 'after': 1.0,
                                                                        'before': 0.0}


# -- content over shuffled controls ----------------------------------------------
def test_nll_summary_matches_the_k1_report():
    tokens = 10
    sums = {'noctx': 30.0, 'full': 10.0, 'span': 15.0, 'span_shuffled': 28.0, 'stack': 20.0,
            'stack_shuffled': 22.0}
    report = nll_summary(sums, tokens, ['span', 'stack'],
                         {'span': 'span_shuffled', 'stack': 'stack_shuffled'})
    assert report['gain'] == 2.0
    assert report['captured'] == {'span': 0.75, 'stack': 0.5}
    assert report['content_nats'] == {'span': 1.3, 'stack': 0.2}
    assert content_nats(1.5, 2.8) == pytest.approx(1.3)
    assert captured_fraction(2.0, 2.0, 2.0) == 0.0  # no gain: floored denominator, no division error


def test_nll_summary_reproduces_the_k1_evaluation_record_exactly():
    """``nll_summary(..., gain=False)`` equals the record ``train_kb_codecs.evaluate``
    computes inline (copied below) on fake summed NLLs, including no or negative
    full-text gain, with the same keys in the same order."""
    import random
    spaces = 'ABCD'
    arms = ['span', 'stack', 'stack_shuffled', 'span_shuffled'] + \
        [f'without_{s}' for s in spaces] + [f'only_{s}' for s in spaces]
    rng = random.Random(0)
    for tokens, full_gain in ((4096, 0.7), (777, 0.0), (1000, -0.2), (3, 1e-12)):
        sums = {'noctx': 3.5 * tokens, 'full': (3.5 - full_gain) * tokens}
        sums.update({name: rng.uniform(1.0, 4.0) * tokens for name in arms})

        # train_kb_codecs.evaluate, after its accumulation loop
        nll = {name: value / tokens for name, value in sums.items()}
        gain = max(nll['noctx'] - nll['full'], 1e-9)
        legacy = {'nll': {k: round(v, 4) for k, v in nll.items()},
                  'captured': {k: round((nll['noctx'] - nll[k]) / gain, 4) for k in arms},
                  'content_nats': {'span': round(nll['span_shuffled'] - nll['span'], 4),
                                   'stack': round(nll['stack_shuffled'] - nll['stack'], 4)}}
        report = nll_summary(sums, tokens, arms, {'span': 'span_shuffled', 'stack': 'stack_shuffled'},
                             gain=False)
        assert json.dumps(report) == json.dumps(legacy)
        with_gain = nll_summary(sums, tokens, arms, {'span': 'span_shuffled'})
        assert with_gain['gain'] == round(nll['noctx'] - nll['full'], 4)
