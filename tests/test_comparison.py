from copy import deepcopy

import pytest

from schnitz.comparison import compare_transfer_results


def report():
    return {'rows': [dict(episode=e, environment='world', task_family='action',
                         condition=c, choice_correct=False, answer='yes',
                         choice_sequence_nll={'yes': 1., 'no': 2.})
                     for e in ('a', 'b') for c in ('all', 'zero_values')]}


def test_paired_gain_uses_worlds_and_subtracts_control_improvement():
    old = report()
    new = deepcopy(old)
    for row in new['rows']:
        row['choice_correct'] = True
    result = compare_transfer_results(old, new)['by_family']['action']
    assert result['accuracy_gain']['all']['mean_gain'] == 1.
    assert result['accuracy_gain']['all']['n_worlds'] == 1
    assert result['memory_advantage_gain']['zero_values']['mean_gain'] == 0.
    assert result['memory_advantage_gain']['zero_values']['ci95'] == [0., 0.]


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'answer', 'environment', 'choices'])
def test_reject_mismatched_evaluations(mutation):
    old = report()
    new = deepcopy(old)
    if mutation == 'duplicate':
        new['rows'].append(deepcopy(new['rows'][0]))
    elif mutation == 'missing':
        new['rows'].pop()
    elif mutation == 'choices':
        new['rows'][0]['choice_sequence_nll']['maybe'] = 3.
    else:
        new['rows'][0][mutation] = 'changed'
    with pytest.raises(ValueError):
        compare_transfer_results(old, new)


def test_counterfactual_requires_both_original_and_changed_answer_correct():
    old = report()
    new = deepcopy(old)
    for data in (old, new):
        row = deepcopy(data['rows'][0])
        row.update(condition='cf_permission', counterfactual_should_change=True, answer='no')
        data['rows'].append(row)
    new['rows'][-1]['choice_correct'] = True
    result = compare_transfer_results(old, new)['overall']
    assert result['counterfactual_both_correct_gain']['cf_permission']['mean_gain'] == 0.
    new['rows'][0]['choice_correct'] = True
    result = compare_transfer_results(old, new)['overall']
    assert result['counterfactual_both_correct_gain']['cf_permission']['mean_gain'] == 1.


def test_unchanged_counterfactual_delta_counts_spurious_prediction_changes():
    old = report()
    for row in old['rows']:
        row['predicted_action'] = 'no'
    row = deepcopy(old['rows'][0])
    row.update(condition='cf_permission', counterfactual_should_change=False)
    old['rows'].append(row)
    new = deepcopy(old)
    new['rows'][-1]['predicted_action'] = 'yes'
    result = compare_transfer_results(old, new)['overall']
    delta = result['counterfactual_false_change_rate_delta']['cf_permission']
    assert delta['mean_gain'] == 1.
    assert delta['ci95'] == [1., 1.]
    assert delta['n_queries'] == 1
    assert 'cf_permission' not in result['counterfactual_both_correct_gain']


def generation_report():
    return {'inputs': {'episodes_sha256': 'same-corpus', 'max_new_tokens': 24},
            'generation_rows': [dict(episode=r['episode'], environment=r['environment'],
                task_family=r['task_family'], condition=r['condition'], answer='yes',
                prediction='no', exact_match=False) for r in report()['rows']]}


def test_free_generation_comparison_keeps_world_and_control_pairing():
    from schnitz.comparison import compare_generation_results
    old = generation_report()
    new = deepcopy(old)
    for row in new['generation_rows']:
        row.update(prediction='yes', exact_match=True)
    result = compare_generation_results(old, new)
    assert result['metric'] == 'free_generation_exact_match'
    assert result['overall']['accuracy_gain']['all']['mean_gain'] == 1.
    assert result['overall']['accuracy_gain']['all']['n_worlds'] == 1
    assert result['overall']['memory_advantage_gain']['zero_values']['mean_gain'] == 0.


@pytest.mark.parametrize('mutation', ['corpus', 'budget', 'incorrect_flag', 'duplicate', 'missing'])
def test_free_generation_comparison_rejects_mismatched_evidence(mutation):
    from schnitz.comparison import compare_generation_results
    old = generation_report()
    new = deepcopy(old)
    if mutation == 'corpus':
        new['inputs']['episodes_sha256'] = 'different'
    elif mutation == 'budget':
        new['inputs']['max_new_tokens'] = 64
    elif mutation == 'incorrect_flag':
        new['generation_rows'][0]['exact_match'] = True
    elif mutation == 'duplicate':
        new['generation_rows'].append(deepcopy(new['generation_rows'][0]))
    else:
        new['generation_rows'].pop()
    with pytest.raises(ValueError):
        compare_generation_results(old, new)
