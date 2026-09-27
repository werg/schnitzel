"""Greedy generation from an existing frozen bank, without candidate answers.

This is a synthetic diagnostic, not an agent execution benchmark. The checkpoint,
episodes and bank must be the matching artifacts from a stored-transfer evaluation.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes, evidence_ids
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


@torch.no_grad()
def evaluate(run, bank, episodes_path, worlds, max_new_tokens, *, learned_world=False, read_budget=2,
             routing_probe=None, independent_routing_query=False, read_count_policy=None):
    if worlds < 1 or max_new_tokens < 1:
        raise ValueError('World and generation budgets must be positive')
    checkpoint = resolve_checkpoint(run, verify=True)
    config = config_from_run(run)
    if config.train.arm != 'memory' or (not learned_world and config.train.retrieval != 'oracle'):
        raise ValueError('This diagnostic requires the controlled oracle-memory arm')
    if learned_world:
        if len(config.memory.payload_dims) != 1 or config.memory.read_steps != 1 or read_budget < 1:
            raise ValueError('Learned world generation requires one space, one read and positive budget')
        config.memory.neighbors = [read_budget]
    if not bank.is_file():
        raise FileNotFoundError('Supply an existing frozen bank')
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    adapter = None
    if routing_probe is not None:
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=routing_probe,
                                          independent_routing_query=independent_routing_query)
    else:
        if independent_routing_query:
            raise ValueError('Supply a routing probe for the independent query override')
        agent = SchnitzelAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    count_policy = (attach_read_count_policy(agent, checkpoint, read_count_policy)
                    if read_count_policy is not None else None)

    bank_report_path = bank.parent / 'results.json'
    if adapter is not None or bank_report_path.exists():
        bank_report = json.loads(bank_report_path.read_text())
        if (bank_report.get('routing_probe') != adapter
                or bank_report.get('checkpoint_manifest_sha256') != file_sha256(checkpoint / 'manifest.json')):
            raise ValueError('Stored bank routing adapter identity differs')

    def forbidden_writer(*args, **kwargs):
        raise AssertionError('Generation must consume stored payloads without calling the writer')
    agent.produce = forbidden_writer
    store = DiskStore(bank)
    episodes = load_episodes(episodes_path)
    universe = frozenset(s.record_id for e in episodes for s in e.supports)
    selected_worlds = list(dict.fromkeys(e.environment for e in episodes))[:worlds]
    rows = []
    with autocast_context(config):
        for episode in episodes:
            if episode.environment not in selected_worlds:
                continue
            prompt = agent.prompt_ids(episode.query)
            plans = None
            gate_weights = None
            for condition in ('all', 'zero_values', 'none'):
                memory = None
                selected_ids = []
                if condition != 'none':
                    session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                           query_time=episode.query_time,
                                           oracle_ids=None if learned_world else tuple(evidence_ids(episode, config.train.evidence_scope)),
                                           exclude_ids=universe - {s.record_id for s in episode.supports} if learned_world else frozenset(),
                                           ablate_values=condition == 'zero_values', fixed_plans=plans,
                                           fixed_gate_weights=gate_weights if condition == 'zero_values' else None)
                    memory = session.memory
                    selected_ids = list(dict.fromkeys(rid for ids in session.selected_ids for rid in ids))
                    if condition == 'all':
                        plans = session.plans
                        gate_weights = session.gate_weights
                prediction = agent.generate_from_memory(prompt, memory, max_new_tokens=max_new_tokens)
                rows.append(dict(episode=episode.episode_id, environment=episode.environment,
                                 task_family=episode.task_family, condition=condition,
                                 answer=episode.answer, prediction=prediction, selected_ids=selected_ids,
                                 exact_match=prediction == episode.answer))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task_family'], row['condition'])].append(row['exact_match'])
    families = sorted({r['task_family'] for r in rows})
    return {
        'protocol': 'Greedy stored-only generation; no candidate answers supplied; writer disabled.',
        'notice': 'First requested validation worlds, chosen by file order. Exact strings after '
                  'decoder whitespace stripping; no agent execution or capacity-substitution claim.',
        'routing': ({'mode': 'world-scoped exact learned ranking', 'read_budget': read_budget,
                     'eligibility': 'supplied world membership; not global retrieval or authorization'}
                    if learned_world else {'mode': 'oracle', 'evidence_scope': config.train.evidence_scope}),
        'routing_probe': adapter,
        'read_count_policy': count_policy,
        'max_new_tokens': max_new_tokens,
        'worlds': len(selected_worlds),
        'inputs': {'checkpoint': str(checkpoint),
                   'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                   'bank': str(bank), 'bank_sha256': file_sha256(bank),
                   'episodes': str(episodes_path), 'episodes_sha256': file_sha256(episodes_path)},
        'by_family': {f: {c: {'n': len(groups[(f, c)]),
                              'exact_match': sum(groups[(f, c)]) / len(groups[(f, c)])}
                          for c in ('all', 'zero_values', 'none')} for f in families},
        'rows': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--learned-world', action='store_true')
    parser.add_argument('--read-count-policy', type=Path)
    parser.add_argument('--routing-probe', type=Path)
    parser.add_argument('--independent-routing-query', action='store_true')
    parser.add_argument('--read-budget', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.run, args.bank, args.episodes, args.worlds, args.max_new_tokens,
                      learned_world=args.learned_world, read_budget=args.read_budget,
                     routing_probe=args.routing_probe, independent_routing_query=args.independent_routing_query,
                     read_count_policy=args.read_count_policy)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
