#!/usr/bin/env python3
"""Distill a flat Phase 1 checkpoint into an eight-position operator interface."""
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
from schnitz.interface_migration import initialize_positional_student
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.optimizers import make_optimizer
from schnitz.store import DiskStore
from schnitz.training import config_from_run, stored_channel
from schnitz.trajectories import file_sha256


def _autocast(config):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if config.train.device == 'cuda' and config.train.precision == 'bf16'
            else nullcontext())


def _payloads(records, spaces):
    return [torch.cat([record[2 * space + 1] for record in records], 0)[None]
            for space in range(spaces)]


def _tokens(reader, values, query, weights):
    result = reader(values if hasattr(reader, 'local') else values[0], query,
                    weights if hasattr(reader, 'local') else weights[0])
    return result if isinstance(result, torch.Tensor) else result.tokens


def _distillation_loss(teacher, student, episode, *, payload_weight: float,
                       state_weight: float, token_weight: float,
                       downstream_weight: float,
                       task_weight: float) -> tuple[torch.Tensor, dict]:
    selected = set(evidence_ids(episode, student.config.train.evidence_scope))
    sources = [source for source in episode.supports if source.record_id in selected]
    if not sources:
        raise ValueError('Distillation episode has no selected causal evidence')
    with torch.no_grad(), _autocast(teacher.config):
        teacher_records = [stored_channel(teacher, teacher.produce(
            teacher.text_ids(source.text, source=True))) for source in sources]
        prompt = teacher.prompt_ids(episode.query)
        query = teacher.query(prompt)
    with _autocast(student.config):
        student_records = [stored_channel(student, student.produce(
            student.text_ids(source.text, source=True))) for source in sources]
    spaces = len(student.config.memory.payload_dims)
    teacher_values, student_values = (_payloads(records, spaces)
                                      for records in (teacher_records, student_records))
    weights = [query.new_ones(1, len(sources)) for _ in range(spaces)]
    # A joint-token student stores a different layout; payloads are not comparable.
    payload = (sum(F.mse_loss(actual.float(), expected.float())
                   for actual, expected in zip(student_values, teacher_values, strict=True))
               / spaces if payload_weight else query.new_zeros(()))
    with torch.no_grad(), _autocast(teacher.config):
        expected_tokens = _tokens(teacher.reader, teacher_values, query, weights)
    with _autocast(student.config):
        actual_tokens = _tokens(student.reader, student_values, query, weights)
    token = F.mse_loss(actual_tokens.float(), expected_tokens.float())
    state = token * 0
    teacher_local = teacher.reader.local if hasattr(teacher.reader, 'local') else [teacher.reader]
    student_local = student.reader.local if hasattr(student.reader, 'local') else [student.reader]
    for old_reader, new_reader, old_values, new_values, weight in zip(
            teacher_local, student_local, teacher_values, student_values, weights, strict=True):
        with torch.no_grad(), _autocast(teacher.config):
            old_result = old_reader(old_values, query, weight, diagnostics=True)
        with _autocast(student.config):
            new_result = new_reader(new_values, query, weight, diagnostics=True)
        for old_state, new_state, old_stats, new_stats in zip(
                old_result.states, new_result.states,
                old_result.statistics, new_result.statistics, strict=True):
            state = state + F.mse_loss(new_state.float(), old_state.float())
            state = state + F.mse_loss(new_stats.mean().float(), old_stats.mean().float())
            state = state + F.mse_loss(F.softplus(new_stats.log_mass()).float(),
                                       F.softplus(old_stats.log_mass()).float())
    state = state / (spaces * student.config.memory.reader_rounds * 3)
    task, downstream = token * 0, token * 0
    if task_weight or downstream_weight:
        target = student.target_ids(episode.answer)
        prompt_student = student.prompt_ids(episode.query)
        result = student.forward_loop_memory_batch(
            [prompt_student], [target], [student_records], [list(range(len(sources)))])
        task = result.nll
        if downstream_weight:
            with torch.no_grad(), _autocast(teacher.config):
                expected = teacher.forward_loop_memory_batch(
                    [teacher.prompt_ids(episode.query)], [teacher.target_ids(episode.answer)],
                    [teacher_records], [list(range(len(sources)))])
                expected_logits = teacher.backbone.logits(expected.answer_states).float()
            actual_logits = student.backbone.logits(result.answer_states).float()
            hidden = F.mse_loss(result.answer_states.float(), expected.answer_states.float())
            distribution = F.kl_div(actual_logits.log_softmax(-1),
                                    expected_logits.softmax(-1), reduction='batchmean')
            downstream = hidden + distribution / actual_logits.shape[1]
    loss = (payload_weight * payload + state_weight * state
            + token_weight * token + downstream_weight * downstream
            + task_weight * task)
    return loss, {'payload_loss': float(payload.detach()),
                  'state_loss': float(state.detach()),
                  'token_loss': float(token.detach()),
                  'downstream_loss': float(downstream.detach()),
                  'task_nll': float(task.detach())}


def run(teacher_run: Path, episodes_file: Path, output: Path, *, steps: int,
        checkpoint_every: int = 250, payload_weight: float = 1.,
        state_weight: float = 1., token_weight: float = 1.,
        downstream_weight: float = .1, task_weight: float = .1,
        resume: bool = False, layout: str = 'positional',
        space_tokens: tuple[int, ...] = ()) -> dict:
    if min(steps, checkpoint_every) < 1 or min(payload_weight, state_weight,
                                               token_weight, downstream_weight,
                                               task_weight) < 0:
        raise ValueError('Invalid distillation budget or weight')
    teacher_checkpoint = resolve_checkpoint(teacher_run, verify=True)
    teacher_config = config_from_run(teacher_run)
    if teacher_config.memory.payload_layout != 'flat':
        raise ValueError('Teacher checkpoint is not a flat Phase 1 interface')
    if teacher_config.memory.compaction != 'none':
        raise ValueError('Distill the raw interface before the positional compactor')
    index = EpisodeIndex(episodes_file)
    identity = {
        'format': 1,
        'kind': ('flat-to-eight-position-distillation' if layout == 'positional'
                 else 'flat-to-joint-token-distillation'),
        'layout': layout, 'space_tokens': list(space_tokens),
        'teacher_checkpoint': str(teacher_checkpoint),
        'teacher_manifest_sha256': file_sha256(teacher_checkpoint / 'manifest.json'),
        'episodes': str(episodes_file.resolve()),
        'episodes_sha256': index.sha256,
        'steps': steps,
        'weights': {'payload': payload_weight, 'state': state_weight,
                    'tokens': token_weight, 'downstream': downstream_weight,
                    'task': task_weight},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    teacher = SchnitzelAgent(teacher_config).to(teacher_config.train.device).eval()
    load_model(teacher, str(teacher_checkpoint / 'model.safetensors'),
               device=teacher_config.train.device)
    teacher.requires_grad_(False)
    if resume:
        if json.loads((output / 'migration-inputs.json').read_text()) != identity:
            raise ValueError('Distillation inputs or budget changed')
        student_config = config_from_run(output)
    else:
        if output.exists():
            raise FileExistsError(output)
        student_config = config_from_run(teacher_run)
        student_config.memory.payload_layout = layout
        if layout == 'joint_tokens':
            if payload_weight:
                raise ValueError('Joint token students cannot regress flat teacher payloads')
            width = teacher.width
            student_config.memory.space_tokens = list(space_tokens)
            student_config.memory.payload_dims = [tokens * width for tokens in space_tokens]
        student_config.train.steps = steps
        student_config.train.checkpoint_every = checkpoint_every
        student_config.train.wandb_group = 'positional-interface-distillation'
        student_config.validate()
        output.mkdir(parents=True)
        atomic_json(output / 'migration-inputs.json', identity)
    student = SchnitzelAgent(student_config).to(student_config.train.device)
    migration = initialize_positional_student(teacher, student)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(('codecs.', 'reader.')))
    student.train()
    optimizer = make_optimizer(student)
    rng = random.Random(student_config.train.seed)
    cache = DiskStore(output / 'training_cache.sqlite')
    if resume:
        start = restore_checkpoint(student, optimizer, output, rng, fingerprint)
        reconcile_metrics(output, start)
    else:
        start = 0
        atomic_json(output / 'initialization.json', asdict(migration) | identity)
        atomic_json(output / 'run-identity.json', identity | {
            'payload_layout': layout,
            'space_tokens': list(student_config.memory.space_tokens),
            'payload_dims': list(student_config.memory.payload_dims),
        })
        save_checkpoint(student, optimizer, output, 0, rng, cache, fingerprint,
                        keep=student_config.train.keep_checkpoints)
    metrics = output / 'metrics.jsonl'
    completed = start
    with run_lock(output, clear_stop=not resume), stop_on_signal() as signal:
        for step in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            episode = index[rng.randrange(len(index))]
            loss, row = _distillation_loss(
                teacher, student, episode, payload_weight=payload_weight,
                state_weight=state_weight, token_weight=token_weight,
                downstream_weight=downstream_weight,
                task_weight=task_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],
                                           student_config.train.clip_grad_norm)
            optimizer.step()
            completed = step + 1
            row = {'step': completed, 'loss': float(loss.detach()), **row}
            with metrics.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            stopping = signal['signal'] is not None or stop_requested(output)
            if completed % checkpoint_every == 0 or completed == steps or stopping:
                save_checkpoint(student, optimizer, output, completed, rng, cache, fingerprint,
                                keep=student_config.train.keep_checkpoints)
            if stopping:
                break
    result = {'steps': completed, 'requested_steps': steps,
              'stopped_early': completed < steps,
              'checkpoint': str(resolve_checkpoint(output, verify=True))}
    atomic_json(output / 'training_summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--payload-weight', type=float, default=1.)
    parser.add_argument('--state-weight', type=float, default=1.)
    parser.add_argument('--token-weight', type=float, default=1.)
    parser.add_argument('--downstream-weight', type=float, default=.1)
    parser.add_argument('--task-weight', type=float, default=.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--layout', choices=('positional', 'joint_tokens'), default='positional')
    parser.add_argument('--space-tokens', nargs='+', type=int, default=[])
    args = parser.parse_args()
    print(json.dumps(run(args.teacher, args.episodes, args.output, steps=args.steps,
                         checkpoint_every=args.checkpoint_every,
                         payload_weight=args.payload_weight,
                         state_weight=args.state_weight,
                         token_weight=args.token_weight,
                         downstream_weight=args.downstream_weight,
                         task_weight=args.task_weight, resume=args.resume,
                         layout=args.layout, space_tokens=tuple(args.space_tokens)), indent=2))


if __name__ == '__main__':
    main()
