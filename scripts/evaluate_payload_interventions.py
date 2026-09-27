"""Oracle-delivered payload dependence: correct, zeroed and swapped stored values.

Each episode's verified supports are written by the checkpoint's own writer,
serialized at storage precision, and read at the episode's causal query. The
zeroed condition keeps the plan but zeroes payloads; the swapped condition reads
another episode's supports under the same query. Teacher NLL only, not recall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import evidence_ids, load_episodes
from schnitz.operations import atomic_json
from schnitz.training import autocast_context, config_from_run, stored_channel
from schnitz.trajectories import file_sha256


def evaluate(run: Path, episodes_file: Path, output: Path, *, count: int = 128) -> dict:
    if output.exists():
        raise ValueError('Intervention output must be fresh')
    checkpoint = resolve_checkpoint(run, verify=True)
    config = config_from_run(run)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    episodes = load_episodes(episodes_file)[:count]
    if len(episodes) < 2:
        raise ValueError('Interventions need at least two episodes')
    rows = []
    with torch.no_grad(), autocast_context(config):
        records = []
        for episode in episodes:
            selected = set(evidence_ids(episode, config.train.evidence_scope))
            records.append([stored_channel(agent, agent.produce(
                agent.text_ids(source.text, source=True)))
                for source in episode.supports if source.record_id in selected])
        for index, episode in enumerate(episodes):
            prompt, target = agent.prompt_ids(episode.query), agent.target_ids(episode.answer)
            swapped = records[(index + 1) % len(episodes)]
            zeroed = [tuple(value if i % 2 == 0 else torch.zeros_like(value)
                            for i, value in enumerate(record)) for record in records[index]]
            nll = {}
            for name, chosen in (('correct', records[index]), ('zeroed', zeroed),
                                 ('swapped', swapped)):
                result = agent.forward_loop_memory_batch(
                    [prompt], [target], [chosen], [list(range(len(chosen)))])
                nll[name] = float(result.nll)
            rows.append({'episode_id': episode.episode_id, **nll})
    mean = {name: sum(row[name] for row in rows) / len(rows)
            for name in ('correct', 'zeroed', 'swapped')}
    result = {
        'protocol': 'Oracle-supplied supports; teacher-forced answer NLL; not retrieval.',
        'run': str(run), 'checkpoint': checkpoint.name,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'payload_layout': config.memory.payload_layout,
        'episodes_sha256': file_sha256(episodes_file), 'episodes': len(rows),
        'mean_nll': mean,
        'zeroed_minus_correct': mean['zeroed'] - mean['correct'],
        'swapped_minus_correct': mean['swapped'] - mean['correct'],
        'correct_better_than_swapped': sum(row['correct'] < row['swapped']
                                           for row in rows) / len(rows),
    }
    output.mkdir(parents=True)
    atomic_json(output / 'interventions.json', result | {'rows': rows})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.episodes, args.output, count=args.count),
                     indent=2))
