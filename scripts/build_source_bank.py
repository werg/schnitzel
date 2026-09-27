"""Resumably encode a standalone source manifest with one frozen writer snapshot."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.archiving import ensure_free
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import Source
from schnitz.document_ingestion import (grouped_ingestion_prefixes,
                                     source_ingestion_groups, writer_prefix_ids)
from schnitz.offline_bank import canonical_json, ensure_offline_shard, publish_offline_generation
from schnitz.operations import atomic_json
from schnitz.store import DiskStore, lookup_record
from schnitz.training import autocast_context, config_from_run, output_records, stored_channel
from schnitz.trajectories import file_sha256


def build(run: Path, sources: Path, output: Path, *, max_sources: int,
          shard_size: int = 64, writer_batch_size: int = 16,
          writer_prompt_generation: str | None = None, verify_sources: int = 0) -> dict:
    if min(max_sources, shard_size, writer_batch_size) < 1 or not sources.is_file() or not output.parent.is_dir():
        raise ValueError('Positive budgets, source file and existing output parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Bank output must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Bank build requires a 10 GiB disk reserve')
    checkpoint = resolve_checkpoint(run, verify=True)
    config = config_from_run(run)
    if config.train.arm != 'memory':
        raise ValueError('Bank writer checkpoint must use the memory interface')
    rows = []
    with sources.open(encoding='utf-8') as handle:
        for line in handle:
            if len(rows) >= max_sources:
                break
            rows.append(json.loads(line))
    if len(rows) != max_sources or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Source manifest has too few or duplicate records')
    model_sha = file_sha256(checkpoint / 'model.safetensors')
    identity = {'format': 2, 'writer_checkpoint_sha256': model_sha,
                'writer_checkpoint_step': json.loads((checkpoint / 'manifest.json').read_text())['step'],
                'source_manifest_sha256': file_sha256(sources), 'source_count': max_sources,
                'source_order_sha256': hashlib.sha256(canonical_json(
                    [row['record_id'] for row in rows]).encode()).hexdigest(),
                'model': asdict(config.model), 'memory': asdict(config.memory),
                'compute_precision': config.train.precision,
                'max_source_tokens': config.train.max_source_tokens,
                'writer_input_policy': 'grouped-agentic-holistic-and-streaming-v2'}
    if writer_prompt_generation is not None:
        # Keep the writer's ingestion prompts identical to the ones it was trained on.
        identity['writer_prompt_generation'] = writer_prompt_generation
    generation = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:24]
    prompt_generation = writer_prompt_generation or generation
    lock = output / 'identity.json'
    output.mkdir(exist_ok=True)
    if lock.exists():
        if json.loads(lock.read_text()) != identity:
            raise ValueError('Existing bank identity changed')
    else:
        atomic_json(lock, identity)
    ensure_free(output, reserve_bytes=10 * 1024**3)
    store = DiskStore(output / 'bank.sqlite')
    spaces = tuple(f's{i}' for i in range(len(config.memory.payload_dims)))
    total_shards = (len(rows) + shard_size - 1) // shard_size
    shard_ids = [f'{index:06d}' for index in range(total_shards)]
    progress_path = output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    completed_hint = (progress.get('generation') == generation
                      and progress.get('completed_shards') == total_shards
                      and progress.get('total_shards') == total_shards)
    newly_encoded = 0
    agent = None
    source_groups = source_ingestion_groups(rows)

    @lru_cache(maxsize=256)
    def ingestion_prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(
            document_id, parts, generation=prompt_generation,
            scope={'domain': domain}, mode=mode)

    def writer_ids(row):
        document_id, parts, mode = source_groups[row['record_id']]
        messages = ingestion_prefixes(document_id, parts, mode,
                                      row.get('domain', 'research'))[row['record_id']]
        return torch.tensor([writer_prefix_ids(agent.tokenizer, messages)],
                            dtype=torch.long, device=agent.device)

    def load_writer():
        torch.set_num_threads(config.train.threads)
        writer = SchnitzelAgent(config).to(config.train.device)
        if writer.resolved_revision != config.model.revision:
            raise ValueError('Resolved writer model revision changed')
        load_model(writer, str(checkpoint / 'model.safetensors'), device=config.train.device)
        return writer.eval()

    if not completed_hint:
        agent = load_writer()
        completed_ids = []
        for start in range(0, len(rows), shard_size):
            chunk = rows[start:start + shard_size]
            shard_id = f'{start // shard_size:06d}'
            completed_ids.append(shard_id)
            def records():
                with torch.no_grad(), autocast_context(config):
                    for offset in range(0, len(chunk), writer_batch_size):
                        mini = chunk[offset:offset + writer_batch_size]
                        source_batch = [Source(**{key: row[key] for key in
                                                  ('record_id', 'text', 'created_at', 'kind')})
                                        for row in mini]
                        writer_rows = [writer_ids(row) for row in mini]
                        encoded = agent.produce_batch(writer_rows)
                        payloads = stored_channel(agent, encoded)
                        for index, (row, source) in enumerate(zip(mini, source_batch, strict=True)):
                            outputs = tuple(value[index:index + 1] for value in payloads)
                            for record in output_records(agent, source, outputs, 'corpus', generation):
                                yield replace(record, domain=row.get('domain', 'research'))
            made = ensure_offline_shard(store, records, identity=identity, namespace='corpus',
                                        generation=generation, spaces=spaces, shard_id=shard_id,
                                        source_ids=tuple(row['record_id'] for row in chunk))
            newly_encoded += len(chunk) if made else 0
            atomic_json(progress_path, {'generation': generation,
                        'completed_shards': len(completed_ids), 'total_shards': total_shards,
                        'newly_encoded_sources_this_attempt': newly_encoded})
    manifest = publish_offline_generation(store, identity=identity, namespace='corpus',
                                          generation=generation, spaces=spaces,
                                          shard_ids=tuple(shard_ids), source_count=len(rows))
    atomic_json(output / 'manifest.json', manifest | {'sizes': store.sizes()})
    verification = None
    if verify_sources:
        if agent is None:
            agent = load_writer()
        verification = verify_reproduction(agent, store, rows, writer_ids, generation,
                                           count=verify_sources)
        atomic_json(output / 'verification.json', verification)
    return manifest | {'newly_encoded_sources_this_attempt': newly_encoded,
                       'sizes': store.sizes(), 'verification': verification}


def verify_reproduction(agent, store, rows, writer_ids, generation, *, count: int,
                        seed: int = 1701) -> dict:
    """Re-encode sampled sources alone (batch size 1) and compare with stored records.

    Stored records were written in padded batches, so this measures batch-shape
    reproduction noise at storage precision, per space.
    """
    sample = random.Random(seed).sample(rows, min(count, len(rows)))
    spaces = len(agent.config.memory.payload_dims)
    key_cosines = [[] for _ in range(spaces)]
    payload_errors = [[] for _ in range(spaces)]
    with torch.no_grad(), autocast_context(agent.config):
        for row in sample:
            outputs = stored_channel(agent, agent.produce_batch([writer_ids(row)]))
            for space in range(spaces):
                record = lookup_record(store, row['record_id'], namespace='corpus',
                                       space=f's{space}', generation=generation,
                                       domain=row.get('domain', 'research'))
                key = outputs[2 * space][0].float().cpu()
                payload = outputs[2 * space + 1][0].to(record.payload.dtype).float().cpu()
                stored = record.payload.float()
                key_cosines[space].append(float(torch.nn.functional.cosine_similarity(
                    key, record.key.float(), dim=0)))
                payload_errors[space].append(float(
                    (payload - stored).norm() / stored.norm().clamp_min(1e-12)))

    def summary(values):
        ordered = sorted(values)
        return {'min': ordered[0], 'median': ordered[len(ordered) // 2],
                'max': ordered[-1]}
    result = {'sources': len(sample), 'seed': seed, 'reference_batch_size': 1,
              'key_cosine': [summary(v) for v in key_cosines],
              'payload_relative_error': [summary(v) for v in payload_errors]}
    if min(item['min'] for item in result['key_cosine']) < 0.999:
        result['warning'] = 'Stored keys differ from single-source re-encoding'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-sources', type=int, required=True)
    parser.add_argument('--shard-size', type=int, default=64)
    parser.add_argument('--writer-batch-size', type=int, default=16)
    parser.add_argument('--writer-prompt-generation',
                        help='generation string the writer was trained with in its prompts')
    parser.add_argument('--verify-sources', type=int, default=0,
                        help='re-encode this many sources alone and compare with storage')
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.sources, args.output,
                           max_sources=args.max_sources, shard_size=args.shard_size,
                           writer_batch_size=args.writer_batch_size,
                           writer_prompt_generation=args.writer_prompt_generation,
                           verify_sources=args.verify_sources), indent=2))
