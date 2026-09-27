"""R3 keyspace pretraining: train the writer to fill the key slot.

The student starts from a frozen-layout checkpoint (R2). Its key path is
converted to per-space direct fp32 heads of the configured widths, and the key
slot moves after the value slots. The recurrent core, writer slot embeddings,
query norm and direct heads train; everything else is frozen. A frozen copy of
the starting checkpoint supplies payload-preservation targets, so value
semantics stay put while the key slot learns content.

Per step, sampled queries (Hotpot level-1 search sites and public-corpus
searches) and their positive, hard-negative and random same-domain sources are
written live. Losses:

* teacher alignment of live source keys and live query addresses with each
  space's frozen projected teacher vectors;
* in-batch query-to-source score distillation from those teacher projections;
* union support contrast over the in-batch eligible field;
* serialized-precision payload preservation against the frozen start.

Queries use only their causal prefix; answers never enter a query. Eligibility
is per domain namespace and strict creation time.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_file, load_model
import torch
from torch.nn import functional as F

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import (resolve_checkpoint, restore_checkpoint, save_checkpoint,
                              stop_on_signal)
from schnitz.document_ingestion import (grouped_ingestion_prefixes, source_ingestion_groups,
                                     writer_prefix_ids)
from schnitz.keyspace_distillation import (convert_to_direct, field_kl, flat_positives,
                                        support_ranks, union_field_loss)
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.runtime import available_host_memory, reclaim_cuda_cache
from schnitz.spatial_data import SYSTEM_PROMPT, SpatialTrajectoryIndex, _call, _tokens
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


class Targets:
    """Frozen per-space teacher projections of keyed source and query embeddings."""

    def __init__(self, directory: Path, spaces: list[tuple[str, str, str]],
                 projections: Path | None) -> None:
        ids = json.loads((directory / 'ids.json').read_text())
        self.source_index = {record_id: i for i, record_id in enumerate(ids['sources'])}
        self.query_index = {episode_id: i for i, episode_id in enumerate(ids['queries'])}
        self.sources, self.queries = [], []
        for teacher, source_view, query_view in spaces:
            self.sources.append(load_file(str(
                directory / teacher / f'source-{source_view}.safetensors'))['embeddings'])
            self.queries.append(load_file(str(
                directory / teacher / f'query-{query_view}.safetensors'))['embeddings'])
        self.projections = (load_file(str(projections)) if projections is not None else None)

    def project(self, space: int, vectors: torch.Tensor) -> torch.Tensor:
        vectors = vectors.float()
        if self.projections is not None:
            vectors = vectors @ self.projections[f's{space}'].to(vectors.device).T
        return F.normalize(vectors, dim=-1)

    def source(self, space: int, record_ids: list[str], device) -> torch.Tensor:
        rows = torch.tensor([self.source_index[r] for r in record_ids])
        return self.project(space, self.sources[space][rows].to(device))

    def query(self, space: int, episode_ids: list[str], device) -> torch.Tensor:
        rows = torch.tensor([self.query_index[e] for e in episode_ids])
        return self.project(space, self.queries[space][rows].to(device))


def _restore(student, optimizer, output: Path, rng: random.Random, fingerprint: str) -> int:
    """Resume; early checkpoints lack optimizer names, so check group shapes instead."""
    checkpoint = resolve_checkpoint(output, verify=True)
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
    if state.get('optimizer_parameter_names') is not None:
        return restore_checkpoint(student, optimizer, output, rng, fingerprint)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if manifest['dataset_sha256'] != fingerprint:
        raise ValueError('Settings changed since checkpoint')
    saved = state['optimizer']
    shapes = [[tuple(p.shape) for p in g['params']] for g in optimizer.param_groups]
    saved_shapes = [[tuple(saved['state'][i]['exp_avg'].shape) for i in g['params']
                     if i in saved['state']] for g in saved['param_groups']]
    if saved_shapes != shapes:
        raise ValueError('Optimizer groups changed; refusing mismatched momentum')
    load_model(student, str(checkpoint / 'model.safetensors'), device=student.config.train.device)
    optimizer.load_state_dict(saved)
    rng.setstate(state['python_rng'])
    random.setstate(state['global_python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return int(state['step'])


def fit_projections(targets_dir: Path, spaces: list[tuple[str, str, str]],
                    pairs: list[tuple[str, tuple[str, ...]]], key_dims: list[int],
                    output: Path, *, steps: int, device: str, seed: int = 1701) -> dict:
    """Contrastive teacher-to-key projections on training pairs only, then frozen."""
    targets = Targets(targets_dir, spaces, None)
    rng = np.random.default_rng(seed)
    tensors, report = {}, {}
    for space, width in enumerate(key_dims):
        documents = targets.sources[space].float().to(device)
        queries = targets.queries[space].float().to(device)
        generator = torch.Generator().manual_seed(seed + space)
        weight = torch.empty(width, documents.shape[1])
        torch.nn.init.orthogonal_(weight, generator=generator)
        weight = weight.to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([weight], lr=1e-3)
        for _ in range(steps):
            batch = [pairs[i] for i in rng.choice(len(pairs), 1024, replace=False)]
            query_rows = torch.tensor([targets.query_index[e] for e, _ in batch], device=device)
            positive = [targets.source_index[p[0]] for _, p in batch]
            field_rows = torch.tensor(sorted(set(positive)), device=device)
            position = {row: i for i, row in enumerate(field_rows.tolist())}
            keys = F.normalize(documents[field_rows] @ weight.T, dim=-1)
            address = F.normalize(queries[query_rows] @ weight.T, dim=-1)
            logits = address @ keys.T / .05
            labels = torch.tensor([position[row] for row in positive], device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        tensors[f's{space}'] = weight.detach().cpu().contiguous()
        report[f's{space}'] = float(loss.detach())
    from safetensors.torch import save_file
    save_file(tensors, str(output))
    return report


def _writer_tokens_factory(tokenizer, rows: dict[str, dict], generation: str):
    groups = source_ingestion_groups(list(rows.values()))

    @lru_cache(maxsize=4096)
    def prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(document_id, parts, generation=generation,
                                          scope={'domain': domain}, mode=mode)

    @lru_cache(maxsize=65536)
    def tokens(record_id: str) -> tuple[int, ...]:
        row = rows[record_id]
        document_id, parts, mode = groups[record_id]
        return tuple(writer_prefix_ids(tokenizer, prefixes(
            document_id, parts, mode, row.get('domain', 'research'))[record_id]))
    return tokens


def _query_prefix_ids(tokenizer, query: str, environment: str) -> tuple[int, ...]:
    """Causal prefix through a visible memory.search call (no result, no answer)."""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': query},
                _call('memory-search-0', SimpleNamespace(query=query, environment=environment))]
    return tuple(_tokens(tokenizer, messages)['input_ids'])


def _prefix_features(agent: SchnitzelAgent, prefixes: list[tuple[int, ...]]) -> torch.Tensor:
    """First-core-pass routing features at each prefix's final call token."""
    lengths = torch.tensor([len(p) for p in prefixes], device=agent.device)
    ids = torch.full((len(prefixes), int(lengths.max())), agent.tokenizer.pad_token_id or 0,
                     dtype=torch.long, device=agent.device)
    for row, prefix in enumerate(prefixes):
        ids[row, :len(prefix)] = torch.tensor(prefix, device=agent.device)
    mask = (torch.arange(ids.shape[1], device=agent.device)[None] < lengths[:, None]).long()
    state = agent.backbone.begin(agent.backbone.embed(ids), mask).state
    positions = lengths - 1
    return agent.loop_query_norm(state[torch.arange(len(prefixes), device=agent.device),
                                       positions])


def _site_features(agent: SchnitzelAgent, rows: list[dict]) -> tuple[torch.Tensor, list[dict]]:
    """Exact level-1 routing features from packed trajectories (first core pass)."""
    from cache_keyspace_states import _spatial_inputs
    ids, attention, sites = _spatial_inputs(agent, rows)
    execution = agent.begin_spatial_recurrent(ids, attention, sites)
    pending = agent.spatial_recurrent_query(execution)
    if pending is None:
        raise ValueError('Packed rows have no level-1 search sites')
    active, _query, features = pending
    metadata = [rows[r]['sites'][s] for s, template in enumerate(rows[0]['sites'])
                if template['level'] == 1 for r in range(len(rows))]
    return features, metadata


def train(args) -> dict:
    output = args.output
    spaces = [tuple(item.split(':')) for item in args.space]
    source_rows = {}
    for path in args.sources:
        with path.open(encoding='utf-8') as handle:
            for row in map(json.loads, handle):
                source_rows[row['record_id']] = row
    by_domain: dict[str, list[str]] = {}
    for record_id, row in source_rows.items():
        by_domain.setdefault(row.get('domain', 'research'), []).append(record_id)
    for domain in by_domain:
        by_domain[domain].sort()
    public = [row for row in map(json.loads, args.public_queries.open(encoding='utf-8'))
              if row['split'] == 'train']
    hotpot = SpatialTrajectoryIndex(args.trajectories)
    init = resolve_checkpoint(args.init, verify=True)
    reference_config = config_from_run(args.init)
    if reference_config.memory.key_interface != 'shared_maps':
        raise ValueError('R3 converts a shared-map start into direct heads')
    config = deepcopy(reference_config)
    config.memory.key_interface = 'direct'
    config.memory.key_dims = list(args.key_dims)
    config.memory.key_slot_position = 'last'
    config.model.freeze_backbone = False
    config.model.backbone_train_scope = 'recurrent_core'
    config.model.gradient_checkpointing = True
    # Packed trajectories alternate level-1 and level-2 sites; the writer's own
    # depth (writer_loops) is independent of this consumer depth.
    config.model.loops = max(config.model.loops, 3)
    config.memory.read_steps = config.model.loops - 1
    config.train.steps = args.steps
    config.train.optimizer = 'adamw'
    config.validate()
    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value
    # Resuming is an execution choice, not part of the scientific identity.
    settings = {key: plain(value) for key, value in vars(args).items()
                if key not in {'func', 'resume', 'min_host_available_gib',
                                   'host_pressure_wait_seconds', 'eval_ks',
                                   'cuda_memory_fraction'}}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if args.cuda_memory_fraction and str(args.device).startswith('cuda'):
        # Unified memory: cached-but-free CUDA blocks are host RAM a co-tenant needs.
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction,
                                                   torch.device(args.device).index or 0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    reference = SchnitzelAgent(reference_config).to(args.device).eval()
    load_model(reference, str(init / 'model.safetensors'), device=args.device)
    reference.requires_grad_(False)
    student = SchnitzelAgent(config).to(args.device)
    if not args.resume:
        if output.exists():
            raise FileExistsError(output)
        state = convert_to_direct({k: v.cpu() for k, v in reference.state_dict().items()},
                                  len(config.memory.payload_dims), tuple(args.key_dims),
                                  seed=args.seed)
        student.load_state_dict(state, strict=True)
    tokenizer = student.tokenizer
    writer_tokens = _writer_tokens_factory(tokenizer, source_rows, args.generation)
    prefix_ids = lru_cache(maxsize=None)(
        lambda query, environment: _query_prefix_ids(tokenizer, query, environment))

    def write(agent, record_ids, *, grad: bool):
        ordered = sorted(range(len(record_ids)),
                         key=lambda i: len(writer_tokens(record_ids[i])))
        chunks = []
        with torch.set_grad_enabled(grad), autocast_context(config):
            for start in range(0, len(ordered), args.writer_chunk):
                rows = ordered[start:start + args.writer_chunk]
                chunks.append((rows, agent.produce_batch([
                    torch.tensor([writer_tokens(record_ids[i])], device=args.device)
                    for i in rows])))
        spaces_out = len(config.memory.payload_dims)
        result = []
        for index in range(2 * spaces_out):
            parts = torch.cat([outputs[index] for _, outputs in chunks], 0)
            order = torch.tensor([i for rows, _ in chunks for i in rows], device=args.device)
            restored = torch.empty_like(parts)
            restored[order] = parts
            result.append(restored)
        return result

    if not args.resume:
        output.mkdir(parents=True)
        # Fixed standardization from the start's own states (exact reparametrization).
        sample = sorted(random.Random(args.seed).sample(by_domain['research'], 2048))
        captured = []
        hook = student.writer_key_heads[0].register_forward_pre_hook(
            lambda _m, inputs: captured.append(inputs[0].detach()))
        student.set_key_state_statistics('writer', torch.zeros(student.width),
                                         torch.ones(student.width))
        with torch.no_grad():
            write(student, sample, grad=False)
        hook.remove()
        writer_states = torch.cat(captured).float()
        student.set_key_state_statistics('writer', writer_states.mean(0),
                                         writer_states.std(0))
        with torch.no_grad(), autocast_context(config):
            features = torch.cat([_prefix_features(student, [
                prefix_ids(row['query'], row['domain']) for row in public[i:i + 64]]).float()
                for i in range(0, 1024, 64)])
        student.set_key_state_statistics('query', features.mean(0), features.std(0))
        atomic_json(output / 'r3-inputs.json', settings | {
            'fingerprint': fingerprint, 'init_checkpoint': str(init),
            'init_sha256': file_sha256(init / 'model.safetensors'),
            'config': asdict(config)})
    projections = output / 'teacher-projections.safetensors'
    if not projections.exists():
        pairs = [(row['episode_id'], tuple(row['required_ids'])) for row in public]
        pairs += [(site['episode_id'], tuple(site['required_ids']))
                  for index in range(len(hotpot)) for site in hotpot[index]['sites']]
        report = fit_projections(args.teacher_targets, spaces, pairs, list(args.key_dims),
                                 projections, steps=args.projection_steps, device=args.device)
        atomic_json(output / 'teacher-projections.json', report)
    targets = Targets(args.teacher_targets, spaces, projections)
    trainable = []
    for name, parameter in student.named_parameters():
        train_it = (name.startswith(('writer_key_heads.', 'query_key_heads.', 'write_slots',
                                     'loop_query_norm.'))
                    or (parameter.requires_grad and name.startswith('backbone.base.')))
        parameter.requires_grad_(train_it)
        if train_it:
            trainable.append((name, parameter))
    # Training-only full-field logit scales; ranking and gates use raw cosine.
    initial_scales = [20.0] * len(spaces)
    if args.resume and (output / 'metrics.jsonl').exists():
        logged = [json.loads(line) for line in (output / 'metrics.jsonl').open()]
        logged = [row for row in logged if 'logit_scales' in row]
        if logged:
            initial_scales = logged[-1]['logit_scales']
    log_scales = torch.nn.Parameter(torch.tensor(np.log(initial_scales), dtype=torch.float32,
                                                 device=args.device))
    groups = [
        {'params': [p for n, p in trainable if n.startswith('backbone.')],
         'lr': args.core_learning_rate},
        {'params': [p for n, p in trainable if n.startswith(('write_slots', 'loop_query_norm.'))],
         'lr': args.slot_learning_rate},
        {'params': [p for n, p in trainable if n.startswith(('writer_key_heads.',
                                                             'query_key_heads.'))],
         'lr': args.head_learning_rate},
        {'params': [log_scales], 'lr': 1e-2},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    names = {id(p): n for n, p in trainable} | {id(log_scales): 'logit_scales'}
    optimizer._schnitz_parameter_names = [[names[id(p)] for p in g['params']]
                                       for g in optimizer.param_groups]
    cache = DiskStore(output / 'training_cache.sqlite')
    rng = random.Random(args.seed)
    start = (_restore(student, optimizer, output, rng, fingerprint)
             if args.resume else 0)
    if not args.resume:
        save_checkpoint(student, optimizer, output, 0, rng, cache, fingerprint, keep=4)
    eval_rows = [SpatialTrajectoryIndex(args.validation_trajectories)[i]
                 for i in range(min(args.eval_trajectories,
                                    len(SpatialTrajectoryIndex(args.validation_trajectories))))]
    eval_field = sorted({r for row in eval_rows for site in row['sites']
                         for r in site['required_ids']}
                        | set(random.Random(7).sample(by_domain['research'], args.eval_field)))

    def evaluate() -> dict:
        student.eval()
        with torch.no_grad():
            outputs = write(student, eval_field, grad=False)
            features, metadata = [], []
            for i in range(0, len(eval_rows), 4):
                with autocast_context(config):
                    chunk_features, chunk_meta = _site_features(student, eval_rows[i:i + 4])
                features.append(chunk_features.float())
                metadata.extend(chunk_meta)
            features = torch.cat(features)
            position = {r: i for i, r in enumerate(eval_field)}
            positives = [tuple(position[r] for r in m['required_ids']) for m in metadata]
            summary = {}
            for space in range(len(spaces)):
                keys = outputs[2 * space]
                address = F.normalize(student.routing_address(features, space), dim=-1)
                scores = address @ keys.T
                ranks = support_ranks(scores, positives, torch.ones_like(scores, dtype=torch.bool))
                limit = args.limits[space]
                summary[f's{space}'] = {
                    'recall_at_limit': float(np.mean([min(r) <= limit for r in ranks])),
                    'all_at_limit': float(np.mean([max(r) <= limit for r in ranks])),
                    'recall_at_256': float(np.mean([min(r) <= 256 for r in ranks])),
                    'median_best_rank': float(np.median([min(r) for r in ranks])),
                    # Read-count curve: any / every support within k records.
                    'curve': {str(k): [round(float(np.mean([min(r) <= k for r in ranks])), 4),
                                       round(float(np.mean([max(r) <= k for r in ranks])), 4)]
                              for k in args.eval_ks}}
            summary['field'] = len(eval_field)
            summary['sites'] = len(metadata)
        student.train()
        del outputs, features
        torch.cuda.empty_cache()
        return summary

    metrics = output / 'metrics.jsonl'
    completed = start
    began = time.perf_counter()
    min_host = int(args.min_host_available_gib * 1024 ** 3)
    student.train()
    with run_lock(output, clear_stop=False), stop_on_signal() as signal, ExitStack():
        if start == 0:
            row = {'step': 0, 'eval': evaluate()}
            with metrics.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        for step in range(start, args.steps):
            # A peer's transient host-memory peak should not end the run: release
            # cached blocks and wait; stop with a checkpoint only if it persists.
            pressure = False
            if min_host and (available_host_memory() or min_host) < min_host:
                reclaim_cuda_cache(args.device, 0, force=True)
                waited = 0.0
                while (waited < args.host_pressure_wait_seconds
                       and (available_host_memory() or min_host) < min_host
                       and signal['signal'] is None and not stop_requested(output)):
                    time.sleep(15)
                    waited += 15
                pressure = (available_host_memory() or min_host) < min_host
                print(json.dumps({'host_pressure_wait_seconds': waited, 'step': step,
                                  'stopping': pressure}), flush=True)
            if signal['signal'] is not None or stop_requested(output) or pressure:
                save_checkpoint(student, optimizer, output, completed, rng, cache,
                                fingerprint, keep=4)
                break
            tick = time.perf_counter()
            # Queries: Hotpot level-1 sites plus public searches.
            hotpot_rows = [hotpot[rng.randrange(len(hotpot))]
                           for _ in range(args.hotpot_trajectories)]
            public_rows = rng.sample(public, args.public_queries_per_step)
            with autocast_context(config):
                site_features, site_meta = _site_features(student, hotpot_rows)
                public_features = _prefix_features(student, [
                    prefix_ids(row['query'], row['domain']) for row in public_rows])
            features = torch.cat((site_features.float(), public_features.float()))
            queries = ([{'episode_id': m['episode_id'], 'domain': m['domain'],
                         'required_ids': list(m['required_ids']), 'hard_negative_ids': []}
                        for m in site_meta] + public_rows)
            # Sources: positives, hard negatives, then random same-domain negatives.
            field: list[str] = []
            for row in queries:
                field.extend(row['required_ids'])
                field.extend(row['hard_negative_ids'][:args.hard_negatives])
            field = list(dict.fromkeys(field))
            domains = sorted({row['domain'] for row in queries})
            while len(field) < args.sources_per_step:
                domain = domains[rng.randrange(len(domains))]
                candidate = by_domain[domain][rng.randrange(len(by_domain[domain]))]
                if candidate not in field:
                    field.append(candidate)
            outputs = write(student, field, grad=True)
            # Payload preservation on a subset (positives first) bounds the frozen
            # reference writer's cost; every source still gets a live key.
            preserved = field[:args.payload_sources]
            with torch.no_grad():
                reference_outputs = write(reference, preserved, grad=False)
            position = {r: i for i, r in enumerate(field)}
            field_domains = [source_rows[r].get('domain', 'research') for r in field]
            eligible = torch.tensor([[d == row['domain'] for d in field_domains]
                                     for row in queries], device=args.device)
            positives = [tuple(position[r] for r in row['required_ids']) for row in queries]
            episode_ids = [row['episode_id'] for row in queries]
            align = distill = features.new_zeros(())
            logits = []
            scales = log_scales.clamp(max=float(np.log(100))).exp()
            for space in range(len(spaces)):
                keys = outputs[2 * space]
                address = F.normalize(student.routing_address(features, space), dim=-1)
                teacher_keys = targets.source(space, field, args.device)
                teacher_queries = targets.query(space, episode_ids, args.device)
                align = align + (1 - (keys * teacher_keys).sum(-1)).mean() \
                    + (1 - (address * teacher_queries).sum(-1)).mean()
                student_logits = scales[space] * (address @ keys.T)
                teacher_logits = (teacher_queries @ teacher_keys.T) / args.teacher_temperature
                distill = distill + field_kl(teacher_logits, student_logits, eligible)
                logits.append(student_logits)
            align, distill = align / len(spaces), distill / len(spaces)
            support = union_field_loss(logits, positives, eligible)
            payload = features.new_zeros(())
            for space in range(len(spaces)):
                target = reference_outputs[2 * space + 1].float()
                actual = outputs[2 * space + 1][:len(preserved)].float()
                payload = payload + F.mse_loss(actual, target) / target.var().clamp_min(1e-6)
            payload = payload / len(spaces)
            loss = (args.align_weight * align + args.distill_weight * distill
                    + args.support_weight * support + args.payload_weight * payload)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable] + [log_scales], 1.0)
            optimizer.step()
            completed = step + 1
            with torch.no_grad():
                rows, columns = flat_positives(positives, args.device)
                batch_rank = []
                for space_logits in logits:
                    masked = space_logits.detach().masked_fill(~eligible, float('-inf'))
                    values = masked[rows, columns]
                    batch_rank.append(float(((masked[rows] >= values[:, None]).sum(-1))
                                            .float().median()))
            row = {'step': completed, 'loss': float(loss.detach()),
                   'align': float(align.detach()), 'distill': float(distill.detach()),
                   'support': float(support.detach()), 'payload': float(payload.detach()),
                   'gradient_norm': float(gradient_norm), 'field': len(field),
                   'queries': len(queries), 'batch_median_rank': batch_rank,
                   'logit_scales': [round(float(v), 2) for v in scales.detach()],
                   'step_seconds': time.perf_counter() - tick}
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                row['cuda_peak_allocated_gib'] = round(
                    torch.cuda.max_memory_allocated(args.device) / 1024 ** 3, 2)
                row['cuda_reserved_gib'] = round(
                    torch.cuda.memory_reserved(args.device) / 1024 ** 3, 2)
            if completed % args.eval_every == 0 or completed == args.steps:
                row['eval'] = evaluate()
            if completed % args.log_every == 0 or 'eval' in row:
                with metrics.open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            if completed % args.checkpoint_every == 0 or completed == args.steps:
                save_checkpoint(student, optimizer, output, completed, rng, cache,
                                fingerprint, keep=4)
    summary = {'completed_steps': completed, 'requested_steps': args.steps,
               'elapsed_seconds': time.perf_counter() - began}
    atomic_json(output / 'training_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--validation-trajectories', type=Path, required=True)
    parser.add_argument('--public-queries', type=Path, required=True)
    parser.add_argument('--teacher-targets', type=Path, required=True)
    parser.add_argument('--space', action='append', required=True)
    parser.add_argument('--key-dims', type=int, nargs='+', default=[256, 256, 256, 256])
    parser.add_argument('--limits', type=int, nargs='+', default=[16, 8, 4, 4])
    parser.add_argument('--generation', default='restart-joint-20260923')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--hotpot-trajectories', type=int, default=2)
    parser.add_argument('--public-queries-per-step', type=int, default=24)
    parser.add_argument('--sources-per-step', type=int, default=160)
    parser.add_argument('--hard-negatives', type=int, default=3)
    parser.add_argument('--writer-chunk', type=int, default=32)
    parser.add_argument('--payload-sources', type=int, default=64)
    parser.add_argument('--core-learning-rate', type=float, default=1e-5)
    parser.add_argument('--slot-learning-rate', type=float, default=1e-4)
    parser.add_argument('--head-learning-rate', type=float, default=3e-4)
    parser.add_argument('--align-weight', type=float, default=1.0)
    parser.add_argument('--distill-weight', type=float, default=1.0)
    parser.add_argument('--support-weight', type=float, default=1.0)
    parser.add_argument('--payload-weight', type=float, default=1.0)
    parser.add_argument('--teacher-temperature', type=float, default=.05)
    parser.add_argument('--projection-steps', type=int, default=1500)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-trajectories', type=int, default=250)
    parser.add_argument('--eval-field', type=int, default=10000)
    parser.add_argument('--eval-ks', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--min-host-available-gib', type=float, default=12.0)
    parser.add_argument('--host-pressure-wait-seconds', type=float, default=900.0)
    parser.add_argument('--cuda-memory-fraction', type=float, default=0.0,
                        help='cap the CUDA caching allocator (unified memory is shared)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(train(arguments), indent=2))
