"""Frozen-feature address learnability probe; never a stored-memory capability score."""
import argparse
import copy
import json
import signal
import random
from pathlib import Path

from safetensors.torch import load_file, load_model, save_file
import torch
from torch import nn
from torch.nn import functional as F

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes, make_multiuse_world, save_episodes
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.runtime import available_host_memory, configure_memory, memory_metrics, compute_watchdog
from schnitz.probe_state import restore_probe_state, save_probe_state, parameter_names
from schnitz.optimizers import optimizer_report
from schnitz.tracking import Tracking
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def pair_loss(scores, required, lengths):
    """Batched PL objective for one required item or an unordered required pair."""
    first, second = required.unbind(-1)
    logp = scores.log_softmax(-1)
    a, b = logp.gather(1, first[:, None])[:, 0], logp.gather(1, second[:, None])[:, 0]
    mask_a = F.one_hot(first, scores.shape[-1]).bool()
    mask_b = F.one_hot(second, scores.shape[-1]).bool()
    ab = a + scores.gather(1, second[:, None])[:, 0] - scores.masked_fill(mask_a, -torch.inf).logsumexp(-1)
    ba = b + scores.gather(1, first[:, None])[:, 0] - scores.masked_fill(mask_b, -torch.inf).logsumexp(-1)
    return -torch.where(lengths == 1, a, torch.logaddexp(ab, ba)).mean()


class AddressProbe(nn.Module):
    def __init__(self, agent, raw_query):
        super().__init__()
        self.raw_query = raw_query
        self.key = copy.deepcopy(agent.key_head)
        self.address = copy.deepcopy(agent.address_maps[0])
        self.query_map = copy.deepcopy(agent.query_maps[0])
        self.query_head = copy.deepcopy(agent.query_head)
        self.query_head.requires_grad_(raw_query)

    def forward(self, features):
        key = F.normalize(self.key(features['key']), dim=-1)
        key = F.normalize(self.address(key), dim=-1)
        query = F.normalize(self.query_head(features['raw_query']), dim=-1)
        query = self.query_map(query)
        return (F.normalize(query.float(), dim=-1)[:, None] *
                F.normalize(key.float(), dim=-1)).sum(-1) / .1


@torch.no_grad()
def extract(agent, episodes, config, root):
    result = {'key': [], 'raw_query': [], 'query': [], 'required': [], 'lengths': []}
    sources, captured = {}, []
    def capture(_module, args):
        captured.append(args[0].detach().clone())
    hook = agent.key_head.register_forward_pre_hook(capture)
    try:
        for e in episodes:
            for source in e.supports:
                if source.created_at >= e.query_time:
                    raise ValueError('Future source')
                if source.record_id in sources:
                    if sources[source.record_id][0] != source.text:
                        raise ValueError('Source identity changed')
                    continue
                captured.clear()
                with autocast_context(config):
                    agent.produce(agent.text_ids(source.text, source=True))
                if len(captured) != 1:
                    raise ValueError('Expected one writer key')
                sources[source.record_id] = (source.text, captured[0][0].cpu())
                if len(sources) % 256 == 0:
                    print(json.dumps({'extracted_sources': len(sources)}), flush=True)
            if stop_requested(root):
                raise RuntimeError('Probe stopped during reproducible feature extraction')
    finally:
        hook.remove()
    hook = agent.query_head.register_forward_pre_hook(capture)
    try:
        for i, e in enumerate(episodes):
            captured.clear()
            queries = []
            def provider(completed, query):
                if completed == 1:
                    queries.append(query.detach().cpu()[0])
                return None
            with autocast_context(config):
                agent.plan_loop_memory(agent.prompt_ids(e.query), provider)
            if not queries or not captured:
                raise ValueError('Missing causal first query')
            ids = [s.record_id for s in e.supports]
            required = [ids.index(rid) for rid in e.required_ids]
            if len(ids) != 4 or len(required) not in {1, 2}:
                raise ValueError('Probe requires four candidates and one/two required records')
            result['key'].append(torch.stack([sources[rid][1] for rid in ids]))
            result['raw_query'].append(captured[0][0].cpu())
            result['query'].append(queries[0])
            result['required'].append(torch.tensor((required * 2)[:2]))
            result['lengths'].append(torch.tensor(len(required)))
            if i % 128 == 0:
                print(json.dumps({'extracted_queries': i, 'total': len(episodes)}), flush=True)
            if stop_requested(root):
                raise RuntimeError('Probe stopped during reproducible feature extraction')
    finally:
        hook.remove()
    return {name: torch.stack(values) for name, values in result.items()}


@torch.no_grad()
def score(model, features, episodes, config):
    with autocast_context(config):
        scores = model(features)
        loss = pair_loss(scores, features['required'], features['lengths']).item()
    ranks = scores.argsort(descending=True, stable=True).cpu().tolist()
    families = {}
    for rank, e in zip(ranks, episodes, strict=True):
        ids = [s.record_id for s in e.supports]
        selected = {ids[i] for i in rank[:2]}
        group = families.setdefault(e.task_family, {'count': 0, 'full_required': 0, 'sufficient': 0, 'top_one_required': 0})
        group['count'] += 1
        group['full_required'] += set(e.required_ids) <= selected
        group['sufficient'] += any(set(g) <= selected for g in e.sufficient_groups or (e.required_ids,))
        group['top_one_required'] += ids[rank[0]] in e.required_ids
    return {'loss': loss, 'by_family': families}


def run(checkpoint, root, steps, train_worlds=0, batch_size=0):
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root, clear_stop=False):
        checkpoint = resolve_checkpoint(checkpoint, verify=True)
        config = config_from_run(checkpoint)
        config.train.optimizer = 'muon'
        config.train.seed, config.train.steps = 43, steps
        config.train.wandb_group = 'routing-feature-probe'
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        random.seed(43)
        torch.manual_seed(43)
        agent = SchnitzelAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        if (config.memory.read_steps != 1 or len(config.memory.payload_dims) != 1
                or config.memory.independent_routing_query):
            raise ValueError('Probe requires a first-read single-space shared-query source')
        if train_worlds:
            train = [e for i in range(train_worlds) for e in make_multiuse_world(
                i, split='routing-feature-breadth-train-20260919', bindings=2)]
            train_path = root / 'train.jsonl'
            if train_path.exists():
                if load_episodes(train_path) != train:
                    raise ValueError('Synthetic training data changed')
            else:
                save_episodes(train_path, train)
        else:
            train_path = Path(config.train.episodes_file)
            train = load_episodes(train_path)
        heldout = [e for i in range(32) for e in make_multiuse_world(i, split=('routing-feature-breadth-heldout-20260919' if train_worlds
                                                        else 'routing-feature-probe-20260919'), bindings=2)]
        if (root / 'heldout.jsonl').exists():
            if load_episodes(root / 'heldout.jsonl') != heldout:
                raise ValueError('Held-out data changed')
        else:
            save_episodes(root / 'heldout.jsonl', heldout)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'train_sha256': file_sha256(train_path),
                    'heldout_sha256': file_sha256(root / 'heldout.jsonl'), 'steps': steps,
                    'train_config': vars(config.train), 'seed': 43, 'learning_rate': config.train.learning_rate,
                    'optimizer': 'native Muon', 'batch': batch_size or 'all training queries per update', 'train_worlds': train_worlds,
                    'script_sha256': file_sha256(__file__),
                    'notice': 'Frozen-feature learnability probe, unequal parameter counts; no downstream capability claim.'}
        identity_path = root / 'inputs.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Probe identity changed')
        atomic_json(identity_path, identity)
        features = {}
        for split, episodes in [('train', train), ('heldout', heldout)]:
            path = root / f'{split}-features.safetensors'
            if not path.exists():
                data = extract(agent, episodes, config, root)
                temporary = path.with_suffix('.tmp')
                save_file(data, str(temporary))
                temporary.replace(path)
            features[split] = {k: v.to(config.train.device) for k, v in load_file(str(path)).items()}
        models = {name: AddressProbe(agent, raw).to(config.train.device)
                  for name, raw in [('compressed_query', False), ('full_state_query', True)]}
        if train_worlds > 128:
            models.update({f'narrow128_{name}': AddressProbe(agent, raw).to(config.train.device)
                           for name, raw in [('compressed_query', False), ('full_state_query', True)]})
        with torch.no_grad(), autocast_context(config):
            a, b = [models[name](features['train']) for name in ['compressed_query', 'full_state_query']]
        if not torch.equal(a, b):
            raise ValueError(f'Initial scores differ: {(a-b).abs().max().item()}')
        with torch.no_grad(), autocast_context(config):
            reconstructed = F.normalize(models['compressed_query'].query_head(features['train']['raw_query']), dim=-1)
        drift = (reconstructed - features['train']['query']).abs().max().item()
        del agent
        report = {'identity': identity, 'initial_scores_exact': True,
                  'batched_vs_single_query_max_abs': drift, 'arms': {}}
        for name, model in models.items():
            count = 1280 if name.startswith('narrow128_') else len(train)
            train_features = {k: v[:count] for k, v in features['train'].items()}
            generator = torch.Generator().manual_seed(43)
            optimizer = torch.optim.Muon([p for p in model.parameters() if p.requires_grad], lr=config.train.learning_rate,
                momentum=config.train.muon_momentum, ns_steps=config.train.muon_ns_steps,
                weight_decay=config.train.weight_decay, adjust_lr_fn='match_rms_adamw')
            state_path = root / f'{name}-resume.pt'
            telemetry = root / name
            telemetry.mkdir(exist_ok=True)
            start = restore_probe_state(state_path, model, optimizer, generator, identity, metrics_root=telemetry)
            def save(step):
                save_probe_state(state_path, model, optimizer, generator, identity, step,
                                 reserve_bytes=config.train.min_free_disk_bytes)
            if not state_path.exists():
                save(0)
            atomic_json(telemetry / 'optimizer.json', {'kind': 'native_muon', 'groups': optimizer_report(optimizer),
                        'parameter_names': parameter_names(model, optimizer)})
            initial = score(model, train_features, train[:count], config) if start == 0 else None
            with Tracking(config, telemetry) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'feature_probe': identity, 'arm': name, 'frozen_backbone': True})
                for step in range(start, steps):
                    available = available_host_memory()
                    if stop_requested(root) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Probe stopped; address optimizer state saved')
                    if batch_size:
                        indices = torch.randint(count, (batch_size,), generator=generator, device='cpu').to(config.train.device)
                        batch = {k: v[indices] for k, v in train_features.items()}
                    else:
                        batch = train_features
                    with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            loss = pair_loss(model(batch), batch['required'], batch['lengths'])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                    if (step + 1) % 100 == 0:
                        row = {'arm': name, 'step': step + 1, 'loss': loss.item(), 'resume_attempt': tracking.attempt,
                               **memory_metrics(config.train.device)}
                        with (telemetry / 'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row) + '\n')
                        tracking.log(row)
                        print(json.dumps(row), flush=True)
                save(steps)
            report['arms'][name] = {'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'initial_train': initial, 'training_queries': count,
                'train': score(model, train_features, train[:count], config),
                'heldout': score(model, features['heldout'], heldout, config)}
            atomic_json(root / 'results.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--train-worlds', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=0)
    args = parser.parse_args()
    if args.train_worlds < 0 or args.batch_size < 0:
        parser.error('World and batch counts must be nonnegative')
    if args.steps < 1:
        parser.error('--steps must be positive')
    from schnitz.operations import control_dir
    def request_stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run(args.checkpoint, args.output, args.steps, args.train_worlds, args.batch_size)
