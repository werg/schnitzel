"""Refresh every logical record from one stopped writer without changing its IDs.

The output is a staged mutable journal. Publication requires all space views for
every source, so a trainer cannot observe a partially refreshed keyspace.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.archiving import ensure_free
from schnitz.bank_replay import BankWriterReplay
from schnitz.bank_coherence import verify_refresh_coverage
from schnitz.checkpoints import resolve_checkpoint, stop_on_signal
from schnitz.document_ingestion import (grouped_ingestion_prefixes,
                                     source_ingestion_groups, writer_prefix_ids)
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import canonical_json, publish_offline_generation, writer_prompt_generation
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.training_bank import TrainingBank
from schnitz.trajectories import file_sha256


def refresh(run: Path, bank_dir: Path, sources: Path, output: Path, *,
            chunk_size: int = 64, writer_batch_size: int = 16) -> dict:
    if min(chunk_size, writer_batch_size) < 1 or writer_batch_size > chunk_size:
        raise ValueError('Invalid refresh batch sizes')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Refresh journal must live on external storage')
    checkpoint = resolve_checkpoint(run, verify=True)
    checkpoint_step = json.loads((checkpoint / 'manifest.json').read_text())['step']
    parent_state = json.loads((checkpoint / 'bank-state.json').read_text())
    parent_journal = DiskStore(run / 'training_cache.sqlite')
    if parent_journal.mutable_bank_state() != parent_state:
        raise ValueError('Parent bank moved beyond its committed model checkpoint')
    bank_manifest_path = bank_dir / 'manifest.json'
    bank_manifest = json.loads(bank_manifest_path.read_text())
    base = DiskStore(bank_dir / 'bank.sqlite')
    publish_offline_generation(
        base, identity=bank_manifest['identity'],
        namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
        spaces=tuple(bank_manifest['spaces']), shard_ids=tuple(bank_manifest['shards']),
        source_count=bank_manifest['sources'], verify_only=True)
    source_sha = file_sha256(sources)
    if source_sha != bank_manifest['identity']['source_manifest_sha256']:
        raise ValueError('Refresh sources differ from the original bank')
    rows = [json.loads(line) for line in sources.open(encoding='utf-8')]
    if len(rows) != bank_manifest['sources'] or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Refresh needs every unique source exactly once')
    identity = {
        'format': 1, 'parent_run': str(run.resolve()),
        'parent_checkpoint': checkpoint.name,
        'writer_checkpoint_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'bank_manifest_sha256': file_sha256(bank_manifest_path),
        'source_manifest_sha256': source_sha,
        'parent_bank_state': parent_state,
        'source_order_sha256': hashlib.sha256(canonical_json(
            [row['record_id'] for row in rows]).encode()).hexdigest(),
        'chunk_size': chunk_size, 'writer_batch_size': writer_batch_size,
    }
    output.mkdir(exist_ok=True)
    ensure_free(output, 5 * 1024**3)
    identity_path, progress_path = output / 'identity.json', output / 'progress.json'
    journal_path = output / 'training_cache.sqlite'
    with run_lock(output, clear_stop=False), stop_on_signal() as signals:
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Refresh identity changed')
            if not journal_path.is_file() or not progress_path.is_file():
                raise ValueError('Refresh journal or progress is incomplete')
        else:
            if journal_path.exists() or progress_path.exists():
                raise ValueError('Unidentified refresh output exists')
            pending = output / 'training_cache.pending.sqlite'
            with parent_journal.connect() as source, sqlite3.connect(pending) as target:
                source.backup(target)
            os.replace(pending, journal_path)
            atomic_json(progress_path, {
                'completed_sources': 0, 'cursor': parent_state['cursor']})
            atomic_json(identity_path, identity)
        journal = DiskStore(journal_path)
        progress = json.loads(progress_path.read_text())
        if not 0 <= progress['completed_sources'] <= len(rows):
            raise ValueError('Refresh progress is invalid')
        if (journal.mutable_bank_state()['cursor'] < progress['cursor']
                or journal.mutable_bank_state()['cursor'] > progress['cursor'] + 1):
            raise ValueError('Refresh journal has unexpected unrecorded writes')
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            complete = json.loads(manifest_path.read_text())
            if complete['journal_state'] != journal.mutable_bank_state():
                raise ValueError('Published refresh journal changed')
            return complete
        config = config_from_run(run)
        torch.set_num_threads(config.train.threads)
        agent = SchnitzelAgent(config).to(config.train.device)
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        agent.eval()
        index = PublishedKeyIndex(
            base, namespace=bank_manifest['namespace'],
            generation=bank_manifest['generation'],
            spaces=tuple(bank_manifest['spaces']),
            expected_sources=bank_manifest['sources'])
        mutable = TrainingBank(base, journal, index)
        source_rows = {row['record_id']: row for row in rows}
        groups = source_ingestion_groups(rows)

        @lru_cache(maxsize=256)
        def prefixes(document_id, parts, mode, domain):
            return grouped_ingestion_prefixes(
                document_id, parts, generation=writer_prompt_generation(bank_manifest),
                scope={'domain': domain}, mode=mode)

        @lru_cache(maxsize=2048)
        def writer_tokens(record_id):
            row = source_rows[record_id]
            document_id, parts, mode = groups[record_id]
            messages = prefixes(document_id, parts, mode,
                                row.get('domain', 'research'))[record_id]
            return tuple(writer_prefix_ids(agent.tokenizer, messages))

        class WriterInputs(dict):
            def __getitem__(self, record_id):
                return torch.tensor([writer_tokens(record_id)], dtype=torch.long,
                                    device=agent.device)

            def keys(self):
                return source_rows.keys()

        replay = BankWriterReplay(agent, WriterInputs())
        for start in range(progress['completed_sources'], len(rows), chunk_size):
            if signals['signal'] is not None or stop_requested(output):
                break
            ids = tuple(row['record_id'] for row in rows[start:start + chunk_size])
            with torch.no_grad(), autocast_context(config):
                replay.refresh(mutable, additional_ids=ids,
                               batch_size=writer_batch_size,
                               optimizer_step=checkpoint_step)
            progress = {'completed_sources': start + len(ids),
                        'cursor': mutable.cursor}
            atomic_json(progress_path, progress)
            if (start // chunk_size + 1) % 25 == 0:
                print(json.dumps(progress), flush=True)
        if progress['completed_sources'] != len(rows):
            return {'complete': False, **progress}
        verify_refresh_coverage(
            journal_path, namespace=index.namespace, spaces=tuple(index.spaces),
            source_count=len(rows), parent_cursor=parent_state['cursor'])
        manifest = identity | {
            'complete': True, 'refreshed_sources': len(rows),
            'journal_state': journal.mutable_bank_state(),
            'checkpoint_step': checkpoint_step,
        }
        atomic_json(manifest_path, manifest)
        return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--writer-batch-size', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(refresh(args.run, args.bank, args.sources, args.output,
                             chunk_size=args.chunk_size,
                             writer_batch_size=args.writer_batch_size), indent=2))
