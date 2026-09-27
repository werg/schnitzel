"""Matched frozen-feature continuation with an optional learned STOP candidate."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import random
from types import SimpleNamespace

from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from probe_routing_features import AddressProbe, pair_loss
from schnitz.data import load_episodes
from schnitz.operations import atomic_json, control_dir, run_lock, stop_requested
from schnitz.optimizers import MuonAdamW, optimizer_report
from schnitz.runtime import compute_watchdog, available_host_memory, configure_memory, memory_metrics
from schnitz.tracking import Tracking
from schnitz.training import autocast_context, config_from_run
from schnitz.probe_state import restore_probe_state, save_probe_state
from schnitz.checkpoints import resolve_checkpoint
from schnitz.trajectories import file_sha256


def stop_pair_loss(scores, stop, required, lengths):
    """PL probability of all required records, in any order, followed by STOP."""
    full = torch.cat((scores, stop[:, None]), -1)
    first, second = required.unbind(-1)
    def prefix(a, b):
        mask_a = F.one_hot(a, full.shape[-1]).bool()
        mask_b = F.one_hot(b, full.shape[-1]).bool()
        start = full.gather(1, a[:, None])[:, 0] - full.logsumexp(-1)
        after_a = full.masked_fill(mask_a, -torch.inf).logsumexp(-1)
        single = start + stop - after_a
        pair = (start + full.gather(1, b[:, None])[:, 0] - after_a
                + stop - full.masked_fill(mask_a | mask_b, -torch.inf).logsumexp(-1))
        return single, pair
    one, ab = prefix(first, second)
    _, ba = prefix(second, first)
    return -torch.where(lengths == 1, one, torch.logaddexp(ab, ba)).mean()


def select_records(scores, stop, budget):
    ranks = scores.argsort(descending=True, stable=True).cpu().tolist()
    scores = scores.detach().cpu()
    stop = stop.detach().cpu() if stop is not None else None
    # STOP follows tied record scores, matching an appended sentinel in stable sort.
    return [[i for i in row[:budget] if stop is None or scores[n, i] >= stop[n]]
            for n, row in enumerate(ranks)]


class StopProbe(AddressProbe):
    def __init__(self, weights, enabled):
        def linear(name):
            shape = weights[name + '.weight'].shape
            return nn.Linear(shape[1], shape[0], bias=False)
        agent = SimpleNamespace(key_head=linear('key'), address_maps=[linear('address')],
                                query_maps=[linear('query_map')], query_head=linear('query_head'))
        super().__init__(agent, True)
        self.load_state_dict(weights)
        self.stop_head = nn.Linear(self.query_map.out_features, 1) if enabled else None
        if enabled:
            nn.init.zeros_(self.stop_head.weight)
            nn.init.zeros_(self.stop_head.bias)

    def forward(self, features):
        scores = super().forward(features)
        stop = None
        if self.stop_head is not None:
            query = F.normalize(self.query_head(features['raw_query']), dim=-1)
            query = F.normalize(self.query_map(query).float(), dim=-1)
            stop = self.stop_head(query).float()[:, 0]
        return scores, stop


@torch.no_grad()
def assess(model, features, episodes, config):
    with autocast_context(config):
        scores, stop = model(features)
        loss = (pair_loss(scores, features['required'], features['lengths']) if stop is None else
                stop_pair_loss(scores, stop, features['required'], features['lengths'])).item()
    selections = select_records(scores, stop, 2)
    families, rows = {}, []
    for e, selected in zip(episodes, selections, strict=True):
        ids = [e.supports[i].record_id for i in selected]
        row = {'episode': e.episode_id, 'world': e.environment, 'family': e.task_family,
               'selected': ids, 'count': len(ids), 'full_required': int(set(e.required_ids) <= set(ids)),
               'exact_required': int(set(e.required_ids) == set(ids)),
               'sufficient': int(any(set(g) <= set(ids) for g in e.sufficient_groups or (e.required_ids,)))}
        rows.append(row)
        group = families.setdefault(e.task_family, {'n': 0, 'full_required': 0, 'exact_required': 0,
                                                    'sufficient': 0, 'counts': {'0': 0, '1': 0, '2': 0}})
        group['n'] += 1
        group['counts'][str(len(ids))] += 1
        for key in ('full_required', 'exact_required', 'sufficient'):
            group[key] += row[key]
    return {'objective_loss': loss, 'by_family': families, 'rows': rows}


def run(source, features_root, output, steps):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = copy.deepcopy(config_from_run(checkpoint))
        config.train.optimizer = 'muon'
        inputs = json.loads((features_root / 'inputs.json').read_text())
        path = features_root / 'full_state_query-resume.pt'
        initial = torch.load(path, weights_only=True, map_location='cpu')
        if (initial['identity'] != inputs or initial['step'] != inputs['steps']
                or inputs['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json')):
            raise ValueError('Feature endpoint/source identity differs')
        config.train.steps = steps
        config.train.seed = 47
        config.train.gradient_accumulation = 1
        config.train.train_worlds = inputs['train_worlds']
        config.train.episodes_file = str(features_root / 'train.jsonl')
        config.train.optimization_scope = 'routing'
        config.train.retrieval = 'learned'
        config.train.routing_weight = 1.
        config.train.routing_warmup = 0
        config.train.oracle_anchor_weight = 0.
        config.model.freeze_backbone = True
        config.train.loop_counts = []
        config.train.wandb_group = 'routing-stop-feature-probe'
        config.memory.independent_routing_query = True
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        random.seed(47)
        torch.manual_seed(47)
        identity = {'source_endpoint_sha256': file_sha256(path), 'source_identity': inputs,
                    'steps': steps, 'batch_size': 1280, 'seed': 47, 'optimizer_reset': True,
                    'script_sha256': file_sha256(__file__), 'config': asdict(config),
                    'features_sha256': {name: file_sha256(features_root / f'{name}-features.safetensors') for name in ('train', 'heldout')},
                    'notice': 'Feature-space STOP diagnostic; no learned stopping in production inference yet.'}
        features, episodes = {}, {}
        for name in ('train', 'heldout'):
            episode_path = features_root / f'{name}.jsonl'
            if file_sha256(episode_path) != inputs[name + '_sha256']:
                raise ValueError('Feature episode identity differs')
            features[name] = {k: v.to(config.train.device) for k, v in load_file(str(features_root / f'{name}-features.safetensors')).items()}
            episodes[name] = load_episodes(episode_path)
        calibration = StopProbe(initial['model'], False).to(config.train.device)
        with torch.no_grad(), autocast_context(config):
            initial_scores, _ = calibration(features['train'])
        required = features['train']['required']
        weakest = initial_scores.gather(1, required).min(-1).values
        required_mask = F.one_hot(required, initial_scores.shape[-1]).any(1).bool()
        strongest_other = initial_scores.masked_fill(required_mask, -torch.inf).max(-1).values
        stop_bias = float(((weakest + strongest_other) / 2).median())
        identity['stop_bias_initial'] = stop_bias
        identity['stop_initialization'] = 'Training-only median midpoint of weakest required and strongest irrelevant score'
        del calibration
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Continuation identity changed')
        atomic_json(output / 'inputs.json', identity)
        for name, enabled in [('fixed_budget', False), ('learned_stop', True)]:
            root = output / name
            root.mkdir(exist_ok=True)
            model = StopProbe(initial['model'], enabled).to(config.train.device)
            if enabled:
                with torch.no_grad():
                    model.stop_head.bias.fill_(stop_bias)
            named = dict(model.named_parameters())
            matrix = [p for p in named.values() if p.ndim == 2]
            other = [p for p in named.values() if p.ndim != 2]
            optimizer = MuonAdamW([{'params': matrix, 'lr': config.train.learning_rate}],
                [{'params': other, 'lr': config.train.learning_rate}] if other else [], config.train)
            generator = torch.Generator().manual_seed(47)
            state_path = root / 'resume.pt'
            start = restore_probe_state(state_path, model, optimizer, generator, identity, extra={'arm': name})
            def save(step):
                save_probe_state(state_path, model, optimizer, generator, identity, step,
                                 reserve_bytes=config.train.min_free_disk_bytes, extra={'arm': name})
            if not state_path.exists():
                save(0)
            with Tracking(config, root) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'feature_probe': identity, 'arm': name, 'learned_stop': enabled})
                atomic_json(root / 'optimizer.json', {'kind': 'muon_adamw', 'groups': optimizer_report(optimizer),
                            'parameter_names': list(named), 'parameters': sum(p.numel() for p in named.values())})
                for step in range(start, steps):
                    available = available_host_memory()
                    if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Feature continuation stopped with full optimizer/sampling state')
                    indices = torch.randint(len(episodes['train']), (1280,), generator=generator).to(config.train.device)
                    batch = {k: v[indices] for k, v in features['train'].items()}
                    with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            scores, stop = model(batch)
                            loss = (pair_loss(scores, batch['required'], batch['lengths']) if stop is None else
                                    stop_pair_loss(scores, stop, batch['required'], batch['lengths']))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                    if (step + 1) % 20 == 0:
                        row = {'step': step + 1, 'loss': loss.item(), 'arm': name, 'resume_attempt': tracking.attempt, **memory_metrics(config.train.device)}
                        with (root / 'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row) + '\n')
                        tracking.log(row)
                        print(json.dumps(row), flush=True)
                save(steps)
                for split in ('train', 'heldout'):
                    report = assess(model, features[split], episodes[split], config)
                    atomic_json(root / f'{split}.json', report)
                atomic_json(root / 'completed.json', {'steps': steps, 'identity': identity, 'arm': name})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive')
    def request_stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run(args.source, args.features, args.output, args.steps)
