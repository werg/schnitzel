"""Task-level and world-clustered diagnostic metrics; no independence assumption across uses."""
from __future__ import annotations

import math
import random
from collections import defaultdict


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    if count == 0:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def paired_world_bootstrap(rows: list[dict], a: str = 'all', b: str = 'none', *,
                           draws: int = 1000, seed: int = 0) -> dict:
    """Average paired accuracy gain, bootstrap entire worlds including all queries."""
    if draws < 2:
        raise ValueError('At least two bootstrap draws required')
    by_episode = defaultdict(dict)
    for row in rows:
        if row.get('choice_correct') is not None:
            by_episode[row['episode']][row['condition']] = row
    worlds = defaultdict(list)
    for variants in by_episode.values():
        if a in variants and b in variants:
            world = variants[a].get('environment', variants[a]['episode'])
            worlds[world].append(float(variants[a]['choice_correct']) - float(variants[b]['choice_correct']))
    if not worlds:
        return {'n_worlds': 0, 'mean_gain': None, 'ci95': None}
    items = list(worlds.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(items) for _ in items]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    mean = sum(map(sum, items)) / sum(map(len, items))
    return {'n_worlds': len(items), 'n_queries': sum(map(len, items)), 'mean_gain': mean,
            'ci95': [samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]],
            'warning': 'Intervals resample worlds, not training seeds; small world counts remain unreliable.'}


def summarize_rows(rows: list[dict]) -> dict:
    summary = {}
    for condition in dict.fromkeys(row['condition'] for row in rows):
        group = [r for r in rows if r['condition'] == condition]
        choices = [r for r in group if r.get('choice_correct') is not None]
        successes = sum(r['choice_correct'] for r in choices)
        summary[condition] = {
            'n': len(group), 'choice_accuracy': successes / len(choices) if choices else None,
            'mean_target_nll': sum(r['target_mean_nll'] for r in group) / len(group),
            'complete_support_recall': sum(r['complete_support'] for r in group) / len(group),
        }
    return summary


def counterfactual_metrics(rows: list[dict], names=('cf_restoration', 'cf_permission')) -> dict:
    original = {r['episode']: r for r in rows if r['condition'] == 'all'}
    output = {}
    for name in names:
        group = [r for r in rows if r['condition'] == name]
        changed = [r for r in group if r['counterfactual_should_change']]
        unchanged = [r for r in group if not r['counterfactual_should_change']]
        output[name] = {
            'change_pairs': len(changed),
            'both_correct_on_change': (sum(r['choice_correct'] and original[r['episode']]['choice_correct']
                                           for r in changed) / len(changed)) if changed else None,
            'appropriate_prediction_change': (sum(r['predicted_action'] != original[r['episode']]['predicted_action']
                                                  for r in changed) / len(changed)) if changed else None,
            'unchanged_pairs': len(unchanged),
            'false_change_rate': (sum(r['predicted_action'] != original[r['episode']]['predicted_action']
                                     for r in unchanged) / len(unchanged)) if unchanged else None,
        }
    return output
