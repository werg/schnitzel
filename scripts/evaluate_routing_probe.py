"""Retain paired per-world routing evidence from frozen-feature probe endpoints."""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn

from probe_routing_features import AddressProbe
from schnitz.data import load_episodes
from schnitz.operations import atomic_json
from schnitz.training import autocast_context
from schnitz.trajectories import file_sha256
from schnitz.config import Config


def evaluate(root, output):
    identity = json.loads((root / 'inputs.json').read_text())
    episodes = load_episodes(root / 'heldout.jsonl')
    if file_sha256(root / 'heldout.jsonl') != identity['heldout_sha256']:
        raise ValueError('Held-out data identity changed')
    config = Config()
    for key, value in identity['train_config'].items():
        setattr(config.train, key, value)
    torch.set_num_threads(config.train.threads)
    features = {k: v.to(config.train.device) for k, v in load_file(str(root / 'heldout-features.safetensors')).items()}
    rows, hashes = {}, {}
    for path in sorted(root.glob('*-resume.pt')):
        state = torch.load(path, weights_only=True, map_location=config.train.device)
        if state['identity'] != identity or state['step'] != identity['steps']:
            raise ValueError('Endpoint identity or update budget differs')
        weights = state['model']
        def linear(name):
            shape = weights[name + '.weight'].shape
            return nn.Linear(shape[1], shape[0], bias=False)
        agent = SimpleNamespace(key_head=linear('key'), address_maps=[linear('address')],
                                query_maps=[linear('query_map')], query_head=linear('query_head'))
        model = AddressProbe(agent, 'full_state_query' in path.name).to(config.train.device).eval()
        model.load_state_dict(weights)
        with torch.no_grad(), autocast_context(config):
            ranks = model(features).argsort(descending=True, stable=True).cpu().tolist()
        arm = path.name.removesuffix('-resume.pt')
        rows[arm] = []
        hashes[arm] = file_sha256(path)
        for e, rank in zip(episodes, ranks, strict=True):
            ids = [s.record_id for s in e.supports]
            selected = [ids[i] for i in rank[:2]]
            rows[arm].append({'episode': e.episode_id, 'world': e.environment, 'family': e.task_family,
                              'selected': selected, 'required': list(e.required_ids),
                              'full_required': int(set(e.required_ids) <= set(selected)),
                              'sufficient': int(any(set(g) <= set(selected) for g in e.sufficient_groups or (e.required_ids,))),
                              'top_one_required': int(selected[0] in e.required_ids)})
    paired = []
    for a, b in itertools.combinations(rows, 2):
        for family in sorted({e.task_family for e in episodes}):
            worlds = {}
            for ra, rb in zip(rows[a], rows[b], strict=True):
                if ra['episode'] != rb['episode']:
                    raise ValueError('Episode order differs')
                if ra['family'] == family:
                    worlds.setdefault(ra['world'], []).append([rb[k] - ra[k] for k in
                        ['full_required', 'sufficient', 'top_one_required']])
            values = np.array([np.mean(v, axis=0) for v in worlds.values()])
            rng = np.random.default_rng(43)
            samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
            for i, metric in enumerate(['full_required', 'sufficient', 'top_one_required']):
                paired.append({'a': a, 'b': b, 'family': family, 'metric': metric,
                               'b_minus_a': float(values[:, i].mean()),
                               'world_bootstrap_95': np.quantile(samples[:, i], [.025, .975]).tolist(),
                               'worlds': len(worlds)})
    atomic_json(output, {'identity': identity, 'endpoint_sha256': hashes,
                        'features_sha256': file_sha256(root / 'heldout-features.safetensors'),
                        'rows': rows, 'paired': paired,
                        'notice': 'Feature-space routing only; one training seed, world bootstrap, no downstream capability claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluate(args.probe, args.output)
