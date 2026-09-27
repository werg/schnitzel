"""Expand competing worlds around a fixed stored bank and frozen address/count policies."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.frozen_scoring import FrozenScorer
from schnitz.metrics import summarize_rows
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def pool_worlds(worlds, target, size):
    """Nested diagnostic pools; only the full pool needs no target-world oracle."""
    if target not in worlds or not 1 <= size <= len(worlds) or len(worlds) != len(set(worlds)):
        raise ValueError('Distinct worlds, known target and valid pool size required')
    others = sorted((w for w in worlds if w != target),
                    key=lambda w: hashlib.sha256(f'{target}:{w}'.encode()).digest())
    return frozenset([target, *others[:size - 1]])


def query_subset(episodes, query_worlds):
    """Select evaluation questions without removing any candidate-bank worlds."""
    worlds = list(dict.fromkeys(e.environment for e in episodes))
    count = len(worlds) if query_worlds is None else query_worlds
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(worlds):
        raise ValueError('Query-world count must select a nonempty prefix of the corpus')
    selected = frozenset(worlds[:count])
    return [e for e in episodes if e.environment in selected], selected


@torch.no_grad()
def run(source, bank_path, episodes_file, routing_probe, count_policy, output, pools=(1, 4, 16, 32), generation_worlds=8, query_worlds=None):
    if generation_worlds < 1:
        raise ValueError('Positive generation world count required')
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        if len(config.memory.payload_dims) != 1 or config.memory.read_steps != 1:
            raise ValueError('Requires one-space, one-read frozen reference')
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=routing_probe,
                                          independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        agent.requires_grad_(False)
        reference_path = bank_path.parent / 'bank-manifest.json'
        if not reference_path.exists():
            reference_path = bank_path.parent / 'results.json'
        reference = json.loads(reference_path.read_text())
        identity = {'source_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_file), 'bank_sha256': file_sha256(bank_path),
                    'reference_sha256': file_sha256(reference_path), 'routing_probe': adapter,
                    'read_count_policy': count, 'pool_sizes': list(pools), 'generation_worlds': generation_worlds,
                    'max_new_tokens': 24, 'script_sha256': file_sha256(__file__)}
        if (reference['routing_probe'] != adapter or reference['read_count_policy'] != count or
                reference['checkpoint_manifest_sha256'] != identity['source_manifest_sha256'] or
                reference['episodes_sha256'] != identity['episodes_sha256']):
            raise ValueError('Stored-bank source/data/address/count identity differs')
        episodes = load_episodes(episodes_file)
        worlds = list(dict.fromkeys(e.environment for e in episodes))
        evaluated, query_names = query_subset(episodes, query_worlds)
        identity['query_worlds'] = len(query_names)
        identity['generation_worlds'] = min(generation_worlds, len(query_names))
        world_ids = {w: frozenset(s.record_id for e in episodes if e.environment == w for s in e.supports) for w in worlds}
        universe = frozenset().union(*world_ids.values())
        if len(universe) != sum(map(len, world_ids.values())):
            raise ValueError('Diagnostic worlds share source IDs')
        if not pools or len(pools) != len(set(pools)):
            raise ValueError('Distinct nonempty pool sizes required')
        for size in pools:
            pool_worlds(worlds, worlds[0], size)
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Evaluation identity changed')
        atomic_json(output / 'inputs.json', identity)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Global evaluation re-encoded a source')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        store, scorer = DiskStore(bank_path), FrozenScorer(agent)
        with store.connect() as db:
            records = db.execute("SELECT record_id FROM records WHERE namespace='global' AND space='s0' "
                                 "AND generation='frozen-v1' AND deleted=0").fetchall()
        if {r[0] for r in records} != universe:
            raise ValueError('Stored-bank source universe differs from episodes')
        for size in pools:
            destination = output / f'pool-{size}.json'
            if destination.exists():
                if json.loads(destination.read_text())['inputs'] != identity:
                    raise ValueError('Completed pool identity differs')
                continue
            rows, generations = [], []
            with autocast_context(config):
                for index, e in enumerate(evaluated, 1):
                    if stop_requested(output):
                        raise RuntimeError('Stopped; completed pool results remain reusable')
                    eligible_worlds = pool_worlds(worlds, e.environment, size)
                    candidates = frozenset().union(*(world_ids[w] for w in eligible_worlds))
                    prompt = agent.prompt_ids(e.query)
                    plans = None
                    gate_weights = None
                    for condition in ('all', 'selected_pair', 'none', 'zero_values'):
                        memory, selected = None, []
                        if condition != 'none':
                            session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                query_time=e.query_time, oracle_ids=e.required_ids if condition == 'selected_pair' else None,
                                exclude_ids=universe - candidates if condition != 'selected_pair' else frozenset(),
                                ablate_values=condition == 'zero_values',
                                fixed_plans=plans if condition == 'zero_values' else None,
                                fixed_gate_weights=gate_weights if condition == 'zero_values' else None)
                            memory = session.memory
                            selected = list(dict.fromkeys(r for ids in session.selected_ids for r in ids))
                            if condition == 'all':
                                plans = session.plans
                                gate_weights = session.gate_weights
                        groups = e.sufficient_groups or (e.required_ids,)
                        rows.append({'episode': e.episode_id, 'environment': e.environment, 'task_family': e.task_family,
                            'condition': condition, 'answer': e.answer, 'selected_ids': selected,
                            'selected_record_count': len(selected), 'candidate_records': len(candidates),
                            'complete_support': any(set(g) <= set(selected) for g in groups),
                            'all_required': set(e.required_ids) <= set(selected),
                            'only_target_world': bool(selected) and set(selected) <= world_ids[e.environment],
                            **scorer.score(prompt, memory, e.answer, e.choices)})
                        if e.environment in worlds[:generation_worlds]:
                            prediction = scorer.generate(prompt, memory, max_new_tokens=24)
                            generations.append({'episode': e.episode_id, 'environment': e.environment,
                                'task_family': e.task_family, 'condition': condition, 'answer': e.answer,
                                'prediction': prediction, 'exact_match': prediction == e.answer})
                    if index % 32 == 0 or index == len(evaluated):
                        print(json.dumps({'pool_worlds': size, 'completed_queries': index}), flush=True)
            families = sorted({e.task_family for e in evaluated})
            if file_sha256(bank_path) != identity['bank_sha256']:
                raise ValueError('Stored input bank changed during evaluation')
            atomic_json(destination, {'inputs': identity, 'pool_worlds': size,
                'global_bank': size == len(worlds), 'by_family': {
                    f: summarize_rows([r for r in rows if r['task_family'] == f]) for f in families},
                'rows': rows, 'generation_rows': generations, 'decoder_reuse': scorer.report(),
                'notice': 'Smaller pools include supplied target-world membership. Full pool uses every world. '
                          'Exact authorized/time-filtered stored-key scan, not ANN or an agent task; no cold-disk timing claim.'})
            print(json.dumps({'completed_pool': size}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'bank', 'episodes', 'routing-probe', 'count-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pools', nargs='+', type=int, default=[1, 4, 16, 32])
    parser.add_argument('--generation-worlds', type=int, default=8)
    parser.add_argument('--query-worlds', type=int, help='Score first N worlds while retaining the entire candidate corpus')
    args = parser.parse_args()
    run(args.source, args.bank, args.episodes, args.routing_probe, args.count_policy, args.output, args.pools, args.generation_worlds, args.query_worlds)
