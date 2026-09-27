"""Cache frozen writer key states, payloads and causal query states for key warmup.

With every producer frozen, a per-space key is an exact function of the cached
writer key-slot state, and a direct query address is an exact function of the
cached query-state features. Level-2 query states depend on the level-1 reads
made by this checkpoint (verified supports first, then its own search results),
so they are recorded under that checkpoint's actual selections. Level-1 states
need only the first core pass; ``--full-rows`` bounds the costly full forward
and later trajectories contribute exact level-1 sites only.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

from safetensors import safe_open
from safetensors.torch import load_model, save_file
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.document_ingestion import (grouped_ingestion_prefixes,
                                     source_ingestion_groups, writer_prefix_ids)
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import canonical_json, publish_offline_generation
from schnitz.operations import atomic_json
from schnitz.recurrence import SpatialReadSite
from schnitz.spatial_data import SpatialTrajectoryIndex
from schnitz.spatial_training import spatial_bank_forward
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.training_bank import TrainingBank
from schnitz.trajectories import file_sha256


def cache_states(run: Path, bank_dir: Path, sources: Path, data_path: Path,
                 output: Path, *, limits: tuple[int, ...], routing_candidates: int,
                 writer_batch_size: int = 32, trajectory_batch_size: int = 4,
                 full_rows: int | None = None, level1_rows: int | None = None) -> dict:
    if (output / 'manifest.json').exists() or (output / 'queries.safetensors').exists():
        raise ValueError('State cache query outputs must be fresh')
    checkpoint = resolve_checkpoint(run, verify=True)
    bank_state = json.loads((checkpoint / 'bank-state.json').read_text())
    journal = DiskStore(run / 'training_cache.sqlite')
    if journal.mutable_bank_state() != bank_state:
        raise ValueError('Run journal moved beyond its committed checkpoint')
    manifest_path = bank_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    base = DiskStore(bank_dir / 'bank.sqlite')
    verified = publish_offline_generation(
        base, identity=manifest['identity'], namespace=manifest['namespace'],
        generation=manifest['generation'], spaces=tuple(manifest['spaces']),
        shard_ids=tuple(manifest['shards']), source_count=manifest['sources'],
        verify_only=True)
    if canonical_json(verified) != canonical_json({key: manifest[key] for key in verified}):
        raise ValueError('Published bank failed verification')
    if file_sha256(sources) != manifest['identity']['source_manifest_sha256']:
        raise ValueError('Source manifest differs from bank creation')
    config = config_from_run(run)
    if config.memory.key_interface != 'shared_maps':
        raise ValueError('State caching converts from the shared-map interface')
    torch.manual_seed(config.train.seed)
    agent = SchnitzelAgent(config).to(config.train.device)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    agent.eval()
    index = PublishedKeyIndex(base, namespace=manifest['namespace'],
                              generation=manifest['generation'],
                              spaces=tuple(manifest['spaces']),
                              expected_sources=manifest['sources'])
    bank = TrainingBank(base, journal, index)
    rows = [json.loads(line) for line in sources.open(encoding='utf-8')]
    source_rows = {row['record_id']: row for row in rows}
    groups = source_ingestion_groups(rows)

    @lru_cache(maxsize=256)
    def prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(document_id, parts, generation=manifest['generation'],
                                          scope={'domain': domain}, mode=mode)

    def writer_tokens(record_id):
        row = source_rows[record_id]
        document_id, parts, mode = groups[record_id]
        messages = prefixes(document_id, parts, mode, row.get('domain', 'research'))[record_id]
        return writer_prefix_ids(agent.tokenizer, messages)

    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    reuse = (output / 'writer.safetensors').exists() and (output / 'writer-ids.json').exists()
    # Writer pass: capture the exact key-slot state entering the key projection.
    if reuse:
        writer_check = _verify_writer_cache(agent, config, output, writer_tokens)
        writer_seconds = 0.0
        state_dtype = writer_check['state_dtype']
    else:
        writer_seconds, state_dtype = _writer_pass(
            agent, config, rows, output, writer_tokens, writer_batch_size, began)
        writer_check = _verify_writer_cache(agent, config, output, writer_tokens)
    print(json.dumps({'writer_check': writer_check}), flush=True)
    data = SpatialTrajectoryIndex(data_path)
    total = len(data) if level1_rows is None else min(len(data), level1_rows)
    count = total if full_rows is None else min(full_rows, total)
    features: list[torch.Tensor] = []
    hook = agent.loop_query_norm.register_forward_hook(
        lambda _module, _inputs, result: features.append(result.detach()))
    site_rows, feature_rows = [], []
    query_began = time.perf_counter()

    def record(batch, start, level, level_features):
        expected = len(batch) * sum(site['level'] == level for site in batch[0]['sites'])
        if level_features.shape[0] != expected:
            raise ValueError('Query-state rows differ from site metadata')
        position = 0
        for site_index, template in enumerate(batch[0]['sites']):
            if template['level'] != level:
                continue
            for row_index, row in enumerate(batch):
                site = row['sites'][site_index]
                site_rows.append({
                    'trajectory_index': start + row_index,
                    'trajectory_id': row['trajectory_id'],
                    'call_id': site['call_id'], 'episode_id': site['episode_id'],
                    'level': site['level'], 'required_ids': site['required_ids'],
                    'domain': site['domain'], 'query_time': site['query_time'],
                    'query_position': site['query_position']})
                feature_rows.append(level_features[position].cpu())
                position += 1

    with torch.no_grad(), autocast_context(config):
        # Full causal forward: supplied supports first, then this checkpoint's reads.
        for start in range(0, count, trajectory_batch_size):
            batch = [data[index] for index in range(start, min(start + trajectory_batch_size, count))]
            levels = sorted({site['level'] for site in batch[0]['sites']})
            spatial_bank_forward(agent, bank, index, batch, limits=limits,
                                 routing_candidates=routing_candidates,
                                 pad_token_id=agent.tokenizer.pad_token_id or 0)
            if len(features) != len(levels):
                raise ValueError('Query-state capture is misaligned with recurrence levels')
            for level, level_features in zip(levels, features, strict=True):
                record(batch, start, level, level_features)
            features.clear()
            if (start // trajectory_batch_size + 1) % 25 == 0:
                print(json.dumps({'full_trajectories': start + len(batch),
                                  'seconds': time.perf_counter() - query_began}), flush=True)
        # Exact level-1 sites need only the first core pass: no read has happened yet.
        for start in range(count, total, trajectory_batch_size):
            batch = [data[index] for index in range(start, min(start + trajectory_batch_size, total))]
            input_ids, attention, sites = _spatial_inputs(agent, batch)
            execution = agent.begin_spatial_recurrent(input_ids, attention, sites)
            if execution.recurrent.completed != 1:
                raise ValueError('Level-1 states must follow exactly one core pass')
            agent.spatial_recurrent_query(execution)
            if len(features) != 1:
                raise ValueError('Level-1 capture is misaligned')
            record(batch, start, 1, features[0])
            features.clear()
            if (start // trajectory_batch_size + 1) % 250 == 0:
                print(json.dumps({'level1_trajectories': start + len(batch),
                                  'seconds': time.perf_counter() - query_began}), flush=True)
    hook.remove()
    save_file({'features': torch.stack(feature_rows)}, str(output / 'queries.safetensors'))
    with (output / 'query-sites.jsonl').open('w', encoding='utf-8') as handle:
        for row in site_rows:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    result = {
        'format': 2, 'run': str(run.resolve()), 'checkpoint': checkpoint.name,
        'writer_checkpoint_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'bank_manifest_sha256': file_sha256(manifest_path),
        'source_manifest_sha256': manifest['identity']['source_manifest_sha256'],
        'journal_state': bank_state, 'data': str(data_path), 'data_sha256': data.sha256,
        'full_forward_trajectories': count, 'level1_only_trajectories': total - count,
        'query_sites': len(site_rows),
        'level2_sites': sum(row['level'] == 2 for row in site_rows),
        'sources': len(json.loads((output / 'writer-ids.json').read_text())),
        'state_dtype': state_dtype, 'storage_dtype': config.memory.storage_dtype,
        'limits': list(limits), 'routing_candidates': routing_candidates,
        'level2_features': 'recorded under this checkpoint with supplied supports first',
        'writer_reused': reuse, 'writer_check': writer_check,
        'writer_seconds': writer_seconds,
        'query_seconds': time.perf_counter() - query_began,
    }
    atomic_json(output / 'manifest.json', result)
    return result


def _spatial_inputs(agent, batch):
    maximum = max(len(row['input_ids']) for row in batch)
    input_ids = torch.full((len(batch), maximum), agent.tokenizer.pad_token_id or 0,
                           dtype=torch.long, device=agent.device)
    attention = torch.zeros_like(input_ids)
    for row_index, row in enumerate(batch):
        input_ids[row_index, :len(row['input_ids'])] = torch.tensor(row['input_ids'],
                                                                    device=agent.device)
        attention[row_index, :len(row['input_ids'])] = 1
    levels = tuple(site['level'] for site in batch[0]['sites'])
    if any(tuple(site['level'] for site in row['sites']) != levels for row in batch):
        raise ValueError('A batch needs one shared site layout')
    sites = tuple(SpatialReadSite(
        torch.tensor([row['sites'][s]['query_position'] for row in batch], device=agent.device),
        torch.tensor([row['sites'][s]['workspace_start'] for row in batch], device=agent.device),
        levels[s]) for s in range(len(levels)))
    return input_ids, attention, sites


def _verify_writer_cache(agent, config, output, writer_tokens, probes: int = 48) -> dict:
    """Recompute sampled sources with the checkpoint writer; cached rows must match."""
    ids = json.loads((output / 'writer-ids.json').read_text())
    step = max(1, len(ids) // probes)
    chosen = list(range(0, len(ids), step))[:probes]
    captured: list[torch.Tensor] = []
    hook = agent.key_head.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad(), autocast_context(config):
        outputs = agent.produce_batch([
            torch.tensor([writer_tokens(ids[row])], dtype=torch.long, device=agent.device)
            for row in chosen])
    hook.remove()
    worst = {}
    with safe_open(str(output / 'writer.safetensors'), framework='pt') as handle:
        states = handle.get_slice('states')
        cached = torch.stack([states[row] for row in chosen])
        # Hidden states are large; compare their direction per row.
        worst['state_one_minus_cosine'] = float(1 - torch.nn.functional.cosine_similarity(
            captured[0].float().cpu(), cached.float(), dim=-1).min())
        for space in range(len(config.memory.payload_dims)):
            keys = handle.get_slice(f'shared_key_s{space}')
            worst[f'key_s{space}'] = float((outputs[2 * space].float().cpu() - torch.stack(
                [keys[row] for row in chosen]).float()).abs().max())
        state_dtype = str(cached.dtype)
    # Batch composition changes padding, so allow bf16-level differences only.
    if max(worst.values()) > 2e-2:
        raise ValueError(f'Cached writer rows differ from the checkpoint writer: {worst}')
    return {'probes': len(chosen), 'max_abs_difference': worst, 'state_dtype': state_dtype}


def _writer_pass(agent, config, rows, output, writer_tokens, writer_batch_size, began):
    captured: list[torch.Tensor] = []
    hook = agent.key_head.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach()))
    tokens = {row['record_id']: writer_tokens(row['record_id']) for row in rows}
    order = sorted(tokens, key=lambda record_id: (len(tokens[record_id]), record_id))
    dtype = getattr(torch, config.memory.storage_dtype)
    states, payloads, keys = [], [[] for _ in config.memory.payload_dims], \
        [[] for _ in config.memory.payload_dims]
    with torch.no_grad(), autocast_context(config):
        for start in range(0, len(order), writer_batch_size):
            batch = order[start:start + writer_batch_size]
            outputs = agent.produce_batch([
                torch.tensor([tokens[record_id]], dtype=torch.long, device=agent.device)
                for record_id in batch])
            state = captured.pop()
            if captured or state.shape[0] != len(batch):
                raise ValueError('Writer key-state capture is misaligned')
            states.append(state.cpu())
            for space in range(len(config.memory.payload_dims)):
                keys[space].append(outputs[2 * space].float().cpu())
                payloads[space].append(outputs[2 * space + 1].to(dtype).cpu())
            if (start // writer_batch_size + 1) % 200 == 0:
                print(json.dumps({'writer_sources': start + len(batch),
                                  'seconds': time.perf_counter() - began}), flush=True)
    hook.remove()
    writer_seconds = time.perf_counter() - began
    tensors = {'states': torch.cat(states)}
    for space in range(len(config.memory.payload_dims)):
        tensors[f'payload_s{space}'] = torch.cat(payloads[space])
        tensors[f'shared_key_s{space}'] = torch.cat(keys[space])
    save_file(tensors, str(output / 'writer.safetensors'))
    atomic_json(output / 'writer-ids.json', order)
    return writer_seconds, str(tensors['states'].dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limits', nargs='+', type=int, default=[16, 8, 4, 4])
    parser.add_argument('--routing-candidates', type=int, default=256)
    parser.add_argument('--writer-batch-size', type=int, default=32)
    parser.add_argument('--trajectory-batch-size', type=int, default=4)
    parser.add_argument('--full-rows', type=int)
    parser.add_argument('--level1-rows', type=int)
    args = parser.parse_args()
    print(json.dumps(cache_states(
        args.run, args.bank, args.sources, args.data, args.output,
        limits=tuple(args.limits), routing_candidates=args.routing_candidates,
        writer_batch_size=args.writer_batch_size,
        trajectory_batch_size=args.trajectory_batch_size,
        full_rows=args.full_rows, level1_rows=args.level1_rows), indent=2))
