"""Supervised character readout from frozen payloads or reader tokens; not decoder capability."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import signal

import torch
from torch import nn
from torch.nn import functional as F

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.operations import atomic_json, control_dir, run_lock, stop_requested
from schnitz.probe_state import restore_probe_state, save_probe_state
from schnitz.runtime import configure_memory, available_host_memory, compute_watchdog, memory_metrics
from schnitz.store import DiskStore, lookup_record
from schnitz.tracking import Tracking
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def split_identifiers(episodes, heldout_worlds):
    worlds = list(dict.fromkeys(e.environment for e in episodes))
    if not 1 <= heldout_worlds < len(worlds):
        raise ValueError('Need nonempty disjoint train and held-out worlds')
    heldout = set(worlds[:heldout_worlds])
    groups = {'train': [], 'heldout': []}
    seen = set()
    for e in episodes:
        if e.task_family != 'multiuse/identifier':
            continue
        if len(e.required_ids) != 1 or re.fullmatch(r'api_[0-9a-f]{6}', e.answer) is None:
            raise ValueError('Expected one source and generated six-hex endpoint')
        rid = e.required_ids[0]
        if rid in seen:
            raise ValueError('Identifier source repeated across examples')
        seen.add(rid)
        source = next(s for s in e.supports if s.record_id == rid)
        if source.created_at >= e.query_time or f'Its endpoint is {e.answer}.' not in source.text:
            raise ValueError('Endpoint label/source/causal boundary differs')
        groups['heldout' if e.environment in heldout else 'train'].append(e)
    if any(not v for v in groups.values()):
        raise ValueError('Both splits need identifier examples')
    return groups


class Readout(nn.Module):
    def __init__(self, training, hidden):
        super().__init__()
        self.register_buffer('center', training.mean(0))
        self.register_buffer('scale', training.std(0, unbiased=False).clamp_min(1e-6))
        width = training.shape[1]
        self.net = (nn.Sequential(nn.Linear(width, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, 96, bias=False))
                    if hidden else nn.Linear(width, 96, bias=False))

    def forward(self, values):
        return self.net((values-self.center)/self.scale).reshape(-1, 6, 16)


@torch.no_grad()
def assess(model, values, labels, config):
    with autocast_context(config):
        prediction = model(values).argmax(-1)
        shuffled = model(values.roll(1, dims=0)).argmax(-1)
    return {name: {'n': len(labels), 'exact_six': int((p == labels).all(-1).sum()),
                   'characters_correct': int((p == labels).sum()), 'characters': labels.numel(),
                   'characters_correct_by_position': (p == labels).sum(0).tolist()}
            for name, p in [('payload', prediction), ('shifted_payload', shuffled)]}


@torch.no_grad()
def reader_feature(agent, store, episode, *, zero_values=False, representation='reader', metadata=None):
    """First native reader output, before bridge injection; targets are never inputs."""
    from schnitz.recurrence import LoopMemory
    from schnitz.sessions import read_session
    if (agent.training or agent.config.memory.read_timing != 'loop_boundary'
            or agent.config.memory.read_steps != 1):
        raise ValueError('Reader feature requires a frozen, single-read recurrent agent')
    if representation not in {'reader', 'reader_inputs', 'reader_state'}:
        raise ValueError('Unknown reader feature boundary')
    captures, hooks = [], []
    if representation != 'reader':
        from schnitz.readers import SetReader
        if not isinstance(agent.reader, SetReader) or len(episode.required_ids) != 1:
            raise ValueError('Intermediate features require one selected record and one reader space')
        if representation == 'reader_inputs':
            if agent.reader.kind != 'mlp':
                raise ValueError('Input-projection diagnostic currently covers the MLP reader')
            for block in agent.reader.blocks:
                hooks.append(block.input.register_forward_hook(
                    lambda _module, _args, result: captures.append(result.detach())))
        else:
            hooks.append(agent.reader.output.register_forward_pre_hook(
                lambda _module, args: captures.append(args[0].detach())))
    try:
        with compute_watchdog(agent.config.train.stall_timeout_seconds, device=agent.device), autocast_context(agent.config):
            session = read_session(agent, store, agent.prompt_ids(episode.query), namespace='global',
                generation='frozen-v1', query_time=episode.query_time, oracle_ids=episode.required_ids,
                ablate_values=zero_values)
    finally:
        for hook in hooks:
            hook.remove()
    memory = session.memory
    if not isinstance(memory, LoopMemory) or not memory.events or memory.events[0] is None:
        raise ValueError('Missing first-boundary reader output')
    if representation == 'reader_inputs':
        if len(captures) != agent.reader.rounds or any(x.shape[:2] != (1, 1) for x in captures):
            raise ValueError('Expected one actual input projection per complete reader round')
        feature = torch.cat(captures, dim=-1)
    elif representation == 'reader_state':
        if len(captures) != 1:
            raise ValueError('Expected exactly one final reader state')
        feature = captures[0]
    else:
        feature = memory.events[0]
    if metadata is not None:
        metadata['captured_dtype'] = str(feature.dtype)
    return feature.detach().float().flatten().cpu()


def run(source, episodes_path, bank_path, output, steps=1600, heldout_worlds=32, representation='payload'):
    if steps < 1:
        raise ValueError('Positive optimizer budget required')
    if representation not in {'payload', 'reader', 'zero_reader', 'reader_inputs', 'reader_state'}:
        raise ValueError('Unknown frozen feature representation')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.train.optimizer = 'muon'
        config.train.steps, config.train.seed, config.train.learning_rate = steps, 83, .001
        config.train.gradient_accumulation = 1
        config.train.loop_counts = []
        config.train.wandb_group = 'frozen-identifier-readout-'+representation
        config.train.episodes_file = str(episodes_path)
        config.model.freeze_backbone = True
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        reference = json.loads((bank_path.parent/'bank-manifest.json').read_text())
        manifest_hash, data_hash = file_sha256(checkpoint/'manifest.json'), file_sha256(episodes_path)
        if reference['checkpoint_manifest_sha256'] != manifest_hash or reference['episodes_sha256'] != data_hash:
            raise ValueError('Stored bank writer/source corpus differs')
        groups = split_identifiers(load_episodes(episodes_path), heldout_worlds)
        config.train.train_worlds = len({e.environment for e in groups['train']})
        store, values, labels = DiskStore(bank_path), {}, {}
        agent = None
        if representation != 'payload':
            from schnitz.evaluation_adapter import load_frozen_agent
            agent, _ = load_frozen_agent(config, checkpoint)
            agent.requires_grad_(False)
            def forbidden(*_args, **_kwargs):
                raise AssertionError('Reader feature extraction invoked writer or compactor')
            agent.produce = forbidden
            if agent.compactor is not None:
                agent.compactor.forward = forbidden
        capture_dtypes = {}
        for split, episodes in groups.items():
            payloads = []
            dtypes = set()
            for e in episodes:
                record = lookup_record(store, e.required_ids[0], namespace='global', space='s0',
                                       generation='frozen-v1', query_time=e.query_time)
                source_record = next(s for s in e.supports if s.record_id == e.required_ids[0])
                if (record.source_id != e.required_ids[0] or record.created_at != source_record.created_at
                        or record.payload.numel() != config.memory.payload_dims[0]):
                    raise ValueError('Stored source provenance differs')
                if stop_requested(output):
                    raise RuntimeError('Stopped during reproducible frozen feature reads')
                available = available_host_memory()
                if available is not None and available < config.train.min_system_available_bytes:
                    raise RuntimeError('Host memory reserve reached during frozen feature reads')
                metadata = {}
                if agent is None:
                    feature = record.payload.float().flatten()
                    metadata['captured_dtype'] = str(record.payload.dtype)
                else:
                    feature = reader_feature(agent, store, e, zero_values=representation == 'zero_reader',
                        representation=(representation if representation in {'reader_inputs', 'reader_state'} else 'reader'),
                        metadata=metadata)
                payloads.append(feature)
                dtypes.add(metadata['captured_dtype'])
                if len(payloads) % 128 == 0:
                    print(json.dumps({'feature_split': split, 'completed': len(payloads),
                                      'total': len(episodes), 'representation': representation}), flush=True)
            values[split] = torch.stack(payloads).to(config.train.device)
            capture_dtypes[split] = sorted(dtypes)
            labels[split] = torch.tensor([[int(c, 16) for c in e.answer[4:]] for e in episodes], device=config.train.device)
        del agent
        feature_constancy = {split: bool(torch.equal(value, value[:1].expand_as(value)))
                             for split, value in values.items()}
        feature_constancy['heldout_first_matches_training_first'] = bool(
            torch.equal(values['heldout'][0], values['train'][0]))
        identity = {'source_manifest_sha256': manifest_hash, 'episodes_sha256': data_hash,
                    'bank_sha256': file_sha256(bank_path), 'bank_identity': reference, 'config': asdict(config),
                    'steps': steps, 'heldout_worlds': heldout_worlds, 'batch': 128,
                    'representation': representation, 'input_dimension': values['train'].shape[1],
                    'feature_constancy': feature_constancy,
                    'feature_capture_dtypes': capture_dtypes,
                    'script_sha256': file_sha256(__file__), 'split_episodes': {k: [e.episode_id for e in v] for k, v in groups.items()},
                    'notice': 'Six supervised hexadecimal classifiers from declared frozen features. Reader features include the causal query. Wider features change readout parameter counts; zero_reader controls query-only information. Negative results do not prove information absent.'}
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Readout experiment identity changed')
        atomic_json(output/'inputs.json', identity)
        for name, hidden in [('linear', 0), ('mlp256', 256)]:
            root = output/name
            root.mkdir(exist_ok=True)
            random.seed(83)
            torch.manual_seed(83)
            model = Readout(values['train'], hidden).to(config.train.device)
            optimizer = torch.optim.Muon(model.parameters(), lr=.001, momentum=config.train.muon_momentum,
                ns_steps=config.train.muon_ns_steps, weight_decay=config.train.weight_decay, adjust_lr_fn='match_rms_adamw')
            sampler = torch.Generator().manual_seed(83)
            arm_identity = identity | {'arm': name, 'hidden': hidden, 'parameters': sum(p.numel() for p in model.parameters())}
            path = root/'resume.pt'
            start = restore_probe_state(path, model, optimizer, sampler, arm_identity)
            def save(step):
                save_probe_state(path, model, optimizer, sampler, arm_identity, step, reserve_bytes=config.train.min_free_disk_bytes)
            if not path.exists():
                save(0)
            with Tracking(config, root) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'readout': {'arm': name, 'hidden': hidden, 'parameters': arm_identity['parameters']}})
                for step in range(start, steps):
                    available = available_host_memory()
                    if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Readout stopped at complete optimizer boundary')
                    index = torch.randint(len(values['train']), (128,), generator=sampler).to(config.train.device)
                    with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            loss = F.cross_entropy(model(values['train'][index]).flatten(0, 1), labels['train'][index].flatten())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                    if (step+1) % 20 == 0:
                        row = {'step': step+1, 'loss': loss.item(), 'resume_attempt': tracking.attempt, **memory_metrics(config.train.device)}
                        with (root/'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row)+'\n')
                        tracking.log(row)
                        print(json.dumps(row), flush=True)
                save(steps)
            report = {split: assess(model, values[split], labels[split], config) for split in groups}
            atomic_json(root/'results.json', {'identity': arm_identity, 'results': report})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'bank', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--heldout-worlds', type=int, default=32)
    parser.add_argument('--representation', choices=['payload', 'reader', 'zero_reader', 'reader_inputs', 'reader_state'], default='payload')
    args = parser.parse_args()
    def stop(_signal, _frame):
        (control_dir(args.output)/'STOP').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(args.source, args.episodes, args.bank, args.output, args.steps, args.heldout_worlds, args.representation)
