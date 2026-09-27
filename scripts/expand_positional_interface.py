#!/usr/bin/env python3
"""Expand an eight-position interface to 32 positions with slice-frozen warmup."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import (reconcile_metrics, resolve_checkpoint, restore_checkpoint,
                              save_checkpoint, stop_on_signal)
from schnitz.data import evidence_ids
from schnitz.episode_index import EpisodeIndex
from schnitz.interface_migration import expand_positional_state, new_slice_masks
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.optimizers import make_optimizer
from schnitz.store import DiskStore
from schnitz.training import config_from_run, stored_channel
from schnitz.trajectories import file_sha256


def _autocast(config):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if config.train.device == 'cuda' and config.train.precision == 'bf16'
            else nullcontext())


def _read_tokens(reader, values, query, weights):
    result = reader(values if hasattr(reader, 'local') else values[0], query,
                    weights if hasattr(reader, 'local') else weights[0])
    return result if isinstance(result, torch.Tensor) else result.tokens


def _expanded_loss(source, target, episode, *, old_weight: float,
                   new_weight: float, task_weight: float):
    selected = set(evidence_ids(episode, target.config.train.evidence_scope))
    sources = [item for item in episode.supports if item.record_id in selected]
    if not sources:
        raise ValueError('Expansion episode has no selected causal evidence')
    with torch.no_grad(), _autocast(source.config):
        old_records = [stored_channel(source, source.produce(
            source.text_ids(item.text, source=True))) for item in sources]
        query = source.query(source.prompt_ids(episode.query))
    with _autocast(target.config):
        new_records = [stored_channel(target, target.produce(
            target.text_ids(item.text, source=True))) for item in sources]
    old_slots, new_slots = (source.config.memory.write_slots,
                            target.config.memory.write_slots)
    old_payload, new_payload = query.sum() * 0, query.sum() * 0
    old_values, new_values = [], []
    joint = target.config.memory.payload_layout == 'joint_tokens'
    for space, (old_dim, new_dim) in enumerate(zip(
            source.config.memory.payload_dims, target.config.memory.payload_dims,
            strict=True)):
        if joint:
            # Joint tokens keep their layout; new writer slots only enter through
            # the codec's attention, so the whole record anchors old behavior.
            old = torch.cat([record[2 * space + 1] for record in old_records], 0)
            new = torch.cat([record[2 * space + 1] for record in new_records], 0)
            old_payload = old_payload + F.mse_loss(new.float(), old.float())
            old_values.append(old[None])
            new_values.append(new[None])
            continue
        channels = old_dim // old_slots
        old = torch.cat([record[2 * space + 1] for record in old_records], 0)
        new = torch.cat([record[2 * space + 1] for record in new_records], 0)
        old = old.reshape(len(sources), old_slots, channels)
        new = new.reshape(len(sources), new_slots, channels)
        old_payload = old_payload + F.mse_loss(new[:, :old_slots].float(), old.float())
        repeated = old[:, torch.arange(old_slots, new_slots, device=old.device) % old_slots]
        new_payload = new_payload + F.mse_loss(new[:, old_slots:].float(), repeated.float())
        old_values.append(old.reshape(1, len(sources), old_dim))
        new_values.append(new.reshape(1, len(sources), new_dim))
    spaces = len(old_values)
    old_payload, new_payload = old_payload / spaces, new_payload / spaces
    weights = [query.new_ones(1, len(sources)) for _ in range(spaces)]
    with torch.no_grad(), _autocast(source.config):
        old_tokens = _read_tokens(source.reader, old_values, query, weights)
    with _autocast(target.config):
        new_tokens = _read_tokens(target.reader, new_values, query, weights)
    old_token = F.mse_loss(new_tokens[:, :source.config.memory.read_slots].float(),
                           old_tokens.float())
    indices = torch.arange(source.config.memory.read_slots,
                           target.config.memory.read_slots, device=query.device)
    repeated_tokens = old_tokens[:, indices % source.config.memory.read_slots]
    new_token = F.mse_loss(new_tokens[:, source.config.memory.read_slots:].float(),
                           repeated_tokens.float())
    task = new_token * 0
    if task_weight:
        result = target.forward_loop_memory_batch(
            [target.prompt_ids(episode.query)], [target.target_ids(episode.answer)],
            [new_records], [list(range(len(sources)))])
        task = result.nll
    loss = (old_weight * (old_payload + old_token)
            + new_weight * (new_payload + new_token) + task_weight * task)
    return loss, {
        'old_payload_loss': float(old_payload.detach()),
        'new_payload_loss': float(new_payload.detach()),
        'old_token_loss': float(old_token.detach()),
        'new_token_loss': float(new_token.detach()),
        'task_nll': float(task.detach()),
    }


def run(source_run: Path, episodes_file: Path, output: Path, *, steps: int,
        write_slots: int = 32, read_slots: int = 32, checkpoint_every: int = 250,
        old_weight: float = 1., new_weight: float = .1,
        task_weight: float = .1, seed: int = 1701, resume: bool = False) -> dict:
    if min(steps, write_slots, read_slots, checkpoint_every) < 1:
        raise ValueError('Expansion counts must be positive')
    if min(old_weight, new_weight, task_weight) < 0:
        raise ValueError('Expansion loss weights must be nonnegative')
    source_checkpoint = resolve_checkpoint(source_run, verify=True)
    source_config = config_from_run(source_run)
    layout = source_config.memory.payload_layout
    if layout not in {'positional', 'joint_tokens'}:
        raise ValueError('Expansion source must be a structured eight-slot checkpoint')
    index = EpisodeIndex(episodes_file)
    channels = ([dim // source_config.memory.write_slots
                 for dim in source_config.memory.payload_dims]
                if layout == 'positional' else list(source_config.memory.space_tokens))
    identity = {
        'format': 1,
        'kind': ('positional-slot-expansion' if layout == 'positional'
                 else 'joint-token-slot-expansion'),
        'source_checkpoint': str(source_checkpoint),
        'source_manifest_sha256': file_sha256(source_checkpoint / 'manifest.json'),
        'episodes': str(episodes_file.resolve()),
        'episodes_sha256': index.sha256,
        'source_write_slots': source_config.memory.write_slots,
        'source_read_slots': source_config.memory.read_slots,
        'write_slots': write_slots,
        'read_slots': read_slots,
        'channels': channels,
        'steps': steps,
        'seed': seed,
        'weights': {'old': old_weight, 'new': new_weight, 'task': task_weight},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        if json.loads((output / 'migration-inputs.json').read_text()) != identity:
            raise ValueError('Expansion inputs or budget changed')
        target_config = config_from_run(output)
    else:
        if output.exists():
            raise FileExistsError(output)
        target_config = config_from_run(source_run)
        target_config.memory.write_slots = write_slots
        target_config.memory.read_slots = read_slots
        if layout == 'positional':
            target_config.memory.payload_dims = [write_slots * width for width in channels]
        target_config.train.steps = steps
        target_config.train.checkpoint_every = checkpoint_every
        # This local optimizer updates slices of mixed old/new tensors. Disable
        # decoupled decay so zero-masked old slices remain bitwise fixed.
        target_config.train.optimizer = 'adamw'
        target_config.train.weight_decay = 0.
        target_config.train.wandb_group = 'positional-interface-expansion'
        target_config.validate()
        output.mkdir(parents=True)
        atomic_json(output / 'migration-inputs.json', identity)
    source = SchnitzelAgent(source_config).to(source_config.train.device).eval()
    load_model(source, str(source_checkpoint / 'model.safetensors'),
               device=source_config.train.device)
    source.requires_grad_(False)
    target = SchnitzelAgent(target_config).to(target_config.train.device)
    migration = expand_positional_state(source, target, seed=seed)
    masks = new_slice_masks(source, target)
    named = dict(target.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in masks)
        if name in masks:
            mask = masks[name]
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
    target.train()
    optimizer = make_optimizer(target)
    rng = random.Random(target_config.train.seed)
    cache = DiskStore(output / 'training_cache.sqlite')
    if resume:
        start = restore_checkpoint(target, optimizer, output, rng, fingerprint)
        reconcile_metrics(output, start)
    else:
        start = 0
        atomic_json(output / 'initialization.json', asdict(migration) | identity | {
            'trainable_slices': {name: int(mask.sum().item()) for name, mask in masks.items()}})
        atomic_json(output / 'run-identity.json', identity | {
            'payload_layout': 'positional', 'slice_frozen_warmup': True})
        save_checkpoint(target, optimizer, output, 0, rng, cache, fingerprint,
                        keep=target_config.train.keep_checkpoints)
    metrics = output / 'metrics.jsonl'
    completed = start
    with run_lock(output, clear_stop=not resume), stop_on_signal() as signal:
        for step in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            episode = index[rng.randrange(len(index))]
            loss, row = _expanded_loss(source, target, episode,
                                       old_weight=old_weight, new_weight=new_weight,
                                       task_weight=task_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in target.parameters() if p.requires_grad],
                                           target_config.train.clip_grad_norm)
            optimizer.step()
            completed = step + 1
            row = {'step': completed, 'loss': float(loss.detach()), **row}
            with metrics.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            stopping = signal['signal'] is not None or stop_requested(output)
            if completed % checkpoint_every == 0 or completed == steps or stopping:
                save_checkpoint(target, optimizer, output, completed, rng, cache, fingerprint,
                                keep=target_config.train.keep_checkpoints)
            if stopping:
                break
    result = {'steps': completed, 'requested_steps': steps,
              'stopped_early': completed < steps,
              'checkpoint': str(resolve_checkpoint(output, verify=True))}
    atomic_json(output / 'training_summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--write-slots', type=int, default=32)
    parser.add_argument('--read-slots', type=int, default=32)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--old-weight', type=float, default=1.)
    parser.add_argument('--new-weight', type=float, default=.1)
    parser.add_argument('--task-weight', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.episodes, args.output, steps=args.steps,
                         write_slots=args.write_slots, read_slots=args.read_slots,
                         checkpoint_every=args.checkpoint_every,
                         old_weight=args.old_weight, new_weight=args.new_weight,
                         task_weight=args.task_weight, seed=args.seed,
                         resume=args.resume), indent=2))


if __name__ == '__main__':
    main()
