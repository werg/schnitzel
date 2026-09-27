"""Evaluate stored-only exact reconstruction from a frozen short-passage writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation import build_shared_bank, stored_transfer_evaluation
from schnitz.operations import atomic_json
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def evaluate(run: Path, episodes_file: Path, output: Path, *,
             max_episodes: int = 16, generate_episodes: int | None = None,
             max_new_tokens: int = 65) -> dict:
    if min(max_episodes, max_new_tokens) < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive budgets and a fresh output parent required')
    if generate_episodes is None:
        generate_episodes = max_episodes
    if not 0 <= generate_episodes <= max_episodes:
        raise ValueError('Generated episode count must fit the scored subset')
    config = config_from_run(run)
    if config.train.arm != 'memory' or config.train.retrieval != 'oracle':
        raise ValueError('Short reconstruction needs the oracle memory interface')
    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    checkpoint = resolve_checkpoint(run, verify=True)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes or any(e.task_family not in {'passage_reconstruction', 'passage_span'}
                           or len(e.required_ids) != 1 for e in episodes):
        raise ValueError('Evaluation needs one-source passage reconstruction episodes')
    if any(agent.target_ids(e.answer).shape[-1] > max_new_tokens
           for e in episodes[:generate_episodes]):
        raise ValueError('Generation token budget cannot fit every exact target')
    output.mkdir()
    writer_sha = file_sha256(checkpoint / 'model.safetensors')
    generation = writer_sha[:20]
    store = DiskStore(output / 'bank.sqlite')
    with torch.no_grad():
        writes = build_shared_bank(agent, store, episodes, namespace='short_eval',
                                   generation=generation, writer_identity=writer_sha)
    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError('Stored-only inference must not re-encode a source')
    agent.produce = forbidden_writer
    store = DiskStore(output / 'bank.sqlite')
    with torch.no_grad():
        report = stored_transfer_evaluation(agent, store, episodes,
            namespace='short_eval', generation=generation, drop_supports=True)
        rows = []
        with autocast_context(config):
            for episode in episodes[:generate_episodes]:
                prompt = agent.prompt_ids(episode.query)
                correct = read_session(agent, store, prompt, namespace='short_eval',
                    generation=generation, query_time=episode.query_time,
                    oracle_ids=episode.required_ids)
                zero = read_session(agent, store, prompt, namespace='short_eval',
                    generation=generation, query_time=episode.query_time,
                    fixed_plans=correct.plans, fixed_gate_weights=correct.gate_weights,
                    ablate_values=True)
                predictions = {'all': agent.generate_from_memory(prompt, correct.memory,
                    max_new_tokens=max_new_tokens),
                    'zero_values': agent.generate_from_memory(prompt, zero.memory,
                    max_new_tokens=max_new_tokens)}
                rows.append({'episode': episode.episode_id, 'answer': episode.answer,
                             'predictions': predictions,
                             'exact_match': {arm: pred == episode.answer for arm, pred
                                             in predictions.items()}})
    report['generation'] = {'protocol': 'Greedy from reopened stored payloads; '
        'writer disabled and target withheld.', 'max_new_tokens': max_new_tokens,
        'episodes': len(rows),
        'exact_match': {arm: sum(row['exact_match'][arm] for row in rows)
                        for arm in ('all', 'zero_values')}, 'rows': rows}
    report['write_phase'] = writes
    atomic_json(output / 'inputs.json', {'run_checkpoint': checkpoint.name,
        'model_sha256': writer_sha, 'episodes_sha256': file_sha256(episodes_file),
        'max_episodes': max_episodes, 'generate_episodes': generate_episodes,
        'max_new_tokens': max_new_tokens})
    atomic_json(output / 'results.json', report)
    return {'output': str(output), 'episodes': len(episodes),
            'teacher_nll': {key: round(value['mean_target_nll'], 6)
                            for key, value in report['summary'].items()},
            'generation_exact': report['generation']['exact_match']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=16)
    parser.add_argument('--generate-episodes', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=65)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.episodes, args.output,
                              max_episodes=args.max_episodes,
                              generate_episodes=args.generate_episodes,
                              max_new_tokens=args.max_new_tokens), indent=2))
