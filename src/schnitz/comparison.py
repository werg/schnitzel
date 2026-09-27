"""Strictly paired transfer diagnostics for two frozen evaluations."""
from __future__ import annotations

from .metrics import paired_world_bootstrap


def compare_transfer_results(baseline: dict, candidate: dict) -> dict:
    """Compare identical question/condition grids, resampling whole worlds.

    Row metadata detect mismatches, but cannot prove equal source/query text.
    Callers must also retain the common episode file's checksum.
    """
    return _compare_results(baseline, candidate, row_key='rows', outcome='choice_correct',
                            prediction='predicted_action', compare_choices=True)


def compare_generation_results(baseline: dict, candidate: dict) -> dict:
    """Compare free generation, retaining corpus/budget identity and world pairing."""
    for field in ('episodes_sha256', 'max_new_tokens'):
        value = baseline.get('inputs', {}).get(field)
        if value is None or value != candidate.get('inputs', {}).get(field):
            raise ValueError(f'Generation input identity differs or is missing: {field}')
    for report in (baseline, candidate):
        for row in report['generation_rows']:
            if (not isinstance(row.get('prediction'), str) or
                    row.get('exact_match') != (row['prediction'] == row['answer'])):
                raise ValueError('Generation exact-match outcome differs from the actual string')
    return _compare_results(baseline, candidate, row_key='generation_rows', outcome='exact_match',
                            prediction='prediction', compare_choices=False) | {
        'metric': 'free_generation_exact_match',
        'episodes_sha256': baseline['inputs']['episodes_sha256'],
        'max_new_tokens': baseline['inputs']['max_new_tokens']}


def _compare_results(baseline, candidate, *, row_key, outcome, prediction, compare_choices):
    def index(report):
        result = {}
        for row in report[row_key]:
            key = (row['episode'], row['condition'])
            if key in result:
                raise ValueError(f'Duplicate evaluation row: {key}')
            if not isinstance(row.get(outcome), bool):
                raise ValueError('Paired comparison requires boolean outcomes')
            result[key] = row
        if not result:
            raise ValueError('Empty evaluation')
        return result

    old, new = index(baseline), index(candidate)
    if old.keys() != new.keys():
        raise ValueError('Evaluations have different episode/condition grids')
    fields = ('environment', 'task_family', 'answer', 'counterfactual_should_change')
    for key in old:
        if any(old[key].get(f) != new[key].get(f) for f in fields):
            raise ValueError(f'Evaluation identity differs: {key}')
        if compare_choices and set(old[key]['choice_sequence_nll']) != set(new[key]['choice_sequence_nll']):
            raise ValueError(f'Candidate answers differ: {key}')

    def interval(pairs):
        rows = []
        for episode, environment, a, b in pairs:
            for condition, value in (('candidate', a), ('baseline', b)):
                rows.append(dict(episode=episode, environment=environment,
                                 condition=condition, choice_correct=value))
        return paired_world_bootstrap(rows, 'candidate', 'baseline')

    def summarize(keys):
        keys = sorted(keys)
        conditions = sorted({condition for _, condition in keys})
        gains = {}
        for condition in conditions:
            selected = [k for k in keys if k[1] == condition]
            gains[condition] = interval([
                (k[0], old[k]['environment'], new[k][outcome], old[k][outcome])
                for k in selected])
        memory = {}
        for control in ('none', 'zero_values'):
            pairs = [(e, old[(e, 'all')]['environment'],
                      int(new[(e, 'all')][outcome]) - int(new[(e, control)][outcome]),
                      int(old[(e, 'all')][outcome]) - int(old[(e, control)][outcome]))
                     for e, c in keys if c == 'all' and (e, control) in keys]
            if pairs:
                memory[control] = interval(pairs)
        cf = {}
        false_changes = {}
        for condition in conditions:
            selected = [k for k in keys if k[1] == condition and
                        old[k].get('counterfactual_should_change')]
            if selected:
                cf[condition] = interval([
                    (e, old[(e, c)]['environment'],
                     new[(e, c)][outcome] and new[(e, 'all')][outcome],
                     old[(e, c)][outcome] and old[(e, 'all')][outcome])
                    for e, c in selected])
            unchanged = [k for k in keys if k[1] == condition and
                         old[k].get('counterfactual_should_change') is False]
            if unchanged:
                false_changes[condition] = interval([
                    (e, old[(e, c)]['environment'],
                     new[(e, c)][prediction] != new[(e, 'all')][prediction],
                     old[(e, c)][prediction] != old[(e, 'all')][prediction])
                    for e, c in unchanged])
        return dict(accuracy_gain=gains, memory_advantage_gain=memory,
                    counterfactual_both_correct_gain=cf,
                    counterfactual_false_change_rate_delta=false_changes)

    families = sorted({r['task_family'] for r in old.values()})
    return {
        'direction': 'candidate minus baseline',
        'false_change_direction': 'Positive false-change rate delta means more spurious changes (worse).',
        'overall': summarize(set(old)),
        'by_family': {f: summarize({k for k, r in old.items() if r['task_family'] == f})
                      for f in families},
        'notice': 'Paired diagnostic intervals resample worlds, not training seeds. '
                  'Row identities must be accompanied by common episode-file provenance. '
                  'Intervals are not corrected for multiple comparisons.',
    }
