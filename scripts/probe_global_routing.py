"""Matched Muon continuations with within-world or full-bank training negatives."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal

from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from probe_routing_features import pair_loss
from probe_routing_stop import StopProbe
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.operations import atomic_json, control_dir, run_lock, stop_requested
from schnitz.optimizers import optimizer_report
from schnitz.probe_state import restore_probe_state, save_probe_state, parameter_names
from schnitz.runtime import configure_memory, available_host_memory, memory_metrics, compute_watchdog
from schnitz.tracking import Tracking
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def global_features(features, episodes):
    """Deduplicate frozen source features by immutable identity, never text matching."""
    if len(features['key']) != len(episodes) or len(features['raw_query']) != len(episodes):
        raise ValueError('Feature and episode lengths differ')
    sources, vectors = {}, {}
    for i, e in enumerate(episodes):
        if len(e.supports) != features['key'].shape[1] or len(e.required_ids) not in (1, 2):
            raise ValueError('Candidate/required feature layout differs')
        for j, source in enumerate(e.supports):
            if source.created_at >= e.query_time:
                raise ValueError('Future support in frozen features')
            if source.record_id in sources and (sources[source.record_id] != source or
                    not torch.equal(vectors[source.record_id], features['key'][i, j])):
                raise ValueError('Source identity or repeated frozen feature changed')
            sources[source.record_id] = source
            vectors[source.record_id] = features['key'][i, j]
        local = [e.supports[int(j)].record_id for j in features['required'][i][:len(e.required_ids)]]
        if local != list(e.required_ids) or int(features['lengths'][i]) != len(e.required_ids):
            raise ValueError('Training support labels differ from frozen feature annotations')
    ids = sorted(sources)
    index = {rid: i for i, rid in enumerate(ids)}
    return {'key': torch.stack([vectors[rid] for rid in ids]), 'raw_query': features['raw_query'],
            'required': torch.tensor([(list(map(index.__getitem__, e.required_ids)) * 2)[:2] for e in episodes]),
            'lengths': features['lengths'],
            'support_indices': torch.tensor([[index[s.record_id] for s in e.supports] for e in episodes]),
            'created_at': torch.tensor([sources[rid].created_at for rid in ids]),
            'query_time': torch.tensor([e.query_time for e in episodes])}, ids


def address_scores(model, data, indices, scope):
    if scope not in {'world', 'global'}:
        raise ValueError('Unknown training candidate scope')
    # The frozen writer normalizes in its autocast dtype before FP32 key storage.
    encoded_keys = F.normalize(model.address(F.normalize(model.key(data['key']), dim=-1)), dim=-1)
    keys = F.normalize(encoded_keys.float(), dim=-1)
    query = F.normalize(model.query_map(F.normalize(model.query_head(data['raw_query'][indices]), dim=-1)).float(), dim=-1)
    # Search consumes serialized keys and normalizes/scores in FP32.
    with torch.autocast(query.device.type, enabled=False):
        scores = (query @ keys.T) / .1
    eligible = data['created_at'][None] < data['query_time'][indices, None]
    if scope == 'world':
        local = torch.zeros_like(eligible).scatter(1, data['support_indices'][indices], True)
        eligible = eligible & local
    return scores.masked_fill(~eligible, -torch.inf)


@torch.no_grad()
def assess(model, data, episodes, ids, config):
    report = {}
    for scope in ('world', 'global'):
        rows, total = [], 0.
        for start in range(0, len(episodes), 128):
            indices = torch.arange(start, min(start + 128, len(episodes)), device=config.train.device)
            with autocast_context(config):
                scores = address_scores(model, data, indices, scope)
                loss = pair_loss(scores, data['required'][indices], data['lengths'][indices])
            total += loss.item() * len(indices)
            ranks = scores.argsort(descending=True, stable=True)[:, :2].cpu().tolist()
            for e, rank in zip(episodes[start:start + len(indices)], ranks, strict=True):
                selected = [ids[i] for i in rank]
                rows.append({'episode': e.episode_id, 'world': e.environment, 'family': e.task_family,
                    'selected': selected, 'required': list(e.required_ids),
                    'full_required': set(e.required_ids) <= set(selected),
                    'top_one_required': selected[0] in e.required_ids,
                    'only_target_world': set(selected) <= {s.record_id for s in e.supports}})
        families = {}
        for family in sorted({e.task_family for e in episodes}):
            selected = [r for r in rows if r['family'] == family]
            families[family] = {'n': len(selected), **{k: sum(r[k] for r in selected)
                for k in ('full_required', 'top_one_required', 'only_target_world')}}
        report[scope] = {'objective_loss': total / len(episodes), 'by_family': families, 'rows': rows}
    return report


def run(source, features_root, output, steps=800, batch_size=128):
    if steps < 1 or batch_size < 1:
        raise ValueError('Positive update and query batch counts required')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        inputs = json.loads((features_root / 'inputs.json').read_text())
        endpoint = features_root / 'full_state_query-resume.pt'
        previous = torch.load(endpoint, weights_only=True, map_location='cpu')
        if (previous['identity'] != inputs or previous['step'] != inputs['steps'] or
                inputs['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json')):
            raise ValueError('Frozen feature/source/endpoint identity differs')
        config = copy.deepcopy(config_from_run(checkpoint))
        config.train.optimizer = 'muon'
        config.train.steps, config.train.seed = steps, 67
        config.train.gradient_accumulation = 1
        config.train.loop_counts = []
        config.train.wandb_group = 'global-routing-feature-continuation'
        config.train.train_worlds = inputs['train_worlds']
        config.train.episodes_file = str(features_root / 'train.jsonl')
        config.model.freeze_backbone = True
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        torch.set_float32_matmul_precision('highest')
        episodes, data, ids, hashes = {}, {}, {}, {}
        for split in ('train', 'heldout'):
            episode_path = features_root / f'{split}.jsonl'
            if file_sha256(episode_path) != inputs[f'{split}_sha256']:
                raise ValueError('Frozen source episodes changed')
            episodes[split] = load_episodes(episode_path)
            path = features_root / f'{split}-features.safetensors'
            hashes[split] = file_sha256(path)
            values, ids[split] = global_features(load_file(str(path)), episodes[split])
            data[split] = {k: v.to(config.train.device) for k, v in values.items()}
        if set(ids['train']) & set(ids['heldout']):
            raise ValueError('Training and evaluation source IDs overlap')
        identity = {'checkpoint_manifest_sha256': inputs['checkpoint_manifest_sha256'],
                    'source_endpoint_sha256': file_sha256(endpoint), 'source_identity': inputs,
                    'features_sha256': hashes, 'config': asdict(config), 'steps': steps, 'batch_size': batch_size,
                    'seed': 67, 'script_sha256': file_sha256(__file__),
                    'objective': 'unordered required-set PL likelihood, temperature 0.1',
                    'cosine_precision': 'FP32/highest after writer-autocast key normalization and FP32 storage',
                    'candidate_counts': {s: len(i) for s, i in ids.items()},
                    'notice': 'Frozen-feature routing diagnostic, not stored-reader capability evidence.'}
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Continuation identity changed')
        atomic_json(output / 'inputs.json', identity)
        for scope in ('world', 'global'):
            root = output / scope
            root.mkdir(exist_ok=True)
            random.seed(67)
            torch.manual_seed(67)
            model = StopProbe(previous['model'], False).to(config.train.device)
            optimizer = torch.optim.Muon(model.parameters(), lr=config.train.learning_rate,
                momentum=config.train.muon_momentum, ns_steps=config.train.muon_ns_steps,
                weight_decay=config.train.weight_decay, adjust_lr_fn='match_rms_adamw')
            sampler = torch.Generator().manual_seed(67)
            path = root / 'resume.pt'
            start = restore_probe_state(path, model, optimizer, sampler, identity, extra={'arm': scope})
            def save(step):
                save_probe_state(path, model, optimizer, sampler, identity, step,
                    reserve_bytes=config.train.min_free_disk_bytes, extra={'arm': scope})
            atomic_json(root / 'optimizer.json', {'kind': 'native_muon', 'groups': optimizer_report(optimizer),
                                                 'parameter_names': parameter_names(model, optimizer)})
            if not path.exists():
                save(0)
                atomic_json(root / 'initial-heldout.json', assess(model, data['heldout'], episodes['heldout'], ids['heldout'], config))
            with Tracking(config, root) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'feature_probe': identity, 'candidate_scope': scope})
                for step in range(start, steps):
                    available = available_host_memory()
                    if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Routing continuation stopped at complete optimizer boundary')
                    indices = torch.randint(len(episodes['train']), (batch_size,), generator=sampler).to(config.train.device)
                    with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            scores = address_scores(model, data['train'], indices, scope)
                            loss = pair_loss(scores, data['train']['required'][indices], data['train']['lengths'][indices])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                    if (step + 1) % 20 == 0:
                        row = {'step': step + 1, 'loss': loss.item(), 'arm': scope,
                               'resume_attempt': tracking.attempt, **memory_metrics(config.train.device)}
                        with (root / 'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row) + '\n')
                        tracking.log(row)
                        print(json.dumps(row), flush=True)
                save(steps)
            for split in ('train', 'heldout'):
                atomic_json(root / f'{split}.json', assess(model, data[split], episodes[split], ids[split], config))
            atomic_json(root / 'completed.json', {'steps': steps, 'arm': scope, 'identity': identity})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'features', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    def stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(args.source, args.features, args.output, args.steps, args.batch_size)
