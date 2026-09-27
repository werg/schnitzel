"""Inspect entity-pair distinctions in an existing frozen evaluation bank."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.sessions import read_session
from schnitz.store import DiskStore, Selection
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def difference(a, b):
    exact = torch.equal(a, b)
    a, b = a.float().flatten(), b.float().flatten()
    delta = a - b
    return {'exactly_equal': exact, 'max_abs_difference': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / max(float(a.norm()), float(b.norm()), 1e-30))}


@torch.no_grad()
def inspect(run, episodes_path, evaluation):
    checkpoint = resolve_checkpoint(run, verify=True)
    manifest_hash, episode_hash = file_sha256(checkpoint / 'manifest.json'), file_sha256(episodes_path)
    report = json.loads((evaluation / 'summary.json').read_text())
    if (report['checkpoint_manifest_sha256'] != manifest_hash or
            report['episodes_sha256'] != episode_hash):
        raise ValueError('Evaluation bank provenance does not match the requested inputs')
    config = config_from_run(checkpoint)
    if (len(config.memory.payload_dims) != 1 or config.memory.read_steps != 1 or
            config.memory.read_timing != 'loop_boundary' or config.train.arm != 'memory'):
        raise ValueError('This diagnostic requires one space and one in-loop memory read boundary')
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)

    def forbidden(*args, **kwargs):
        raise AssertionError('Writer invoked during stored-bank diagnosis')

    agent.produce = forbidden
    bank = evaluation / 'bank.sqlite'
    store = DiskStore(bank)
    pairs = defaultdict(list)
    with autocast_context(config):
        for episode in load_episodes(episodes_path):
            if episode.task_family not in {'multiuse/permission', 'multiuse/restoration'}:
                continue
            prompt = agent.prompt_ids(episode.query)
            session = read_session(agent, store, prompt, namespace='global',
                generation='frozen-v1', query_time=episode.query_time,
                oracle_ids=tuple(s.record_id for s in episode.supports))
            if len(session.query_keys) != 1 or len(episode.required_ids) != 1:
                raise ValueError('Expected one query and one required fact record')
            plan = replace(session.plans[0][0],
                           selections=(Selection(episode.required_ids[0], 0.),))
            payload = store.fetch(plan)[0]
            tokens = session.memory.events[0]
            pairs[(episode.task_family, episode.environment)].append({
                'answer': episode.answer, 'query': session.query_keys[0],
                'payload': payload.cpu(), 'read': tokens.detach().cpu(),
                'selected_ids': tuple(s.record_id for s in episode.supports)})
    groups = defaultdict(list)
    for (family, _), pair in pairs.items():
        if len(pair) != 2:
            raise ValueError('Expected exactly two entity fact questions per world')
        a, b = pair
        if a['selected_ids'] != b['selected_ids']:
            raise ValueError('Entity-pair reads must use the identical ordered candidate set')
        group = 'same_rule' if a['answer'] == b['answer'] else 'opposed_rule'
        groups[(family, group)].append({name: difference(a[name], b[name])
                                       for name in ('query', 'payload', 'read')})
    result = {}
    for (family, group), values in groups.items():
        summary = {'worlds': len(values)}
        for name in ('query', 'payload', 'read'):
            distances = [v[name]['relative_l2'] for v in values]
            summary[name] = {
                'exact_collision_pairs': sum(v[name]['exactly_equal'] for v in values),
                'relative_l2_min': min(distances), 'relative_l2_median': statistics.median(distances),
                'relative_l2_max': max(distances),
                'max_abs_difference': max(v[name]['max_abs_difference'] for v in values),
            }
        result.setdefault(family, {})[group] = summary
    return {'inputs': {'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': manifest_hash,
                      'episodes_sha256': episode_hash, 'bank_sha256': file_sha256(bank)},
            'by_family': result,
            'notice': 'Stored-only geometric diagnostic, not a probe or proof that information is absent. '
                      'Small differences can still carry information; capability requires behavioral tests.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = inspect(args.run, args.episodes, args.evaluation)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
