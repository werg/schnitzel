"""Measure verified-support ranks in a published bank without reading payloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.bank_coherence import verify_refresh_coverage
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import (assert_bank_reader_compatible, canonical_json,
                               publish_offline_generation, stored_memory_identity)
from schnitz.operations import atomic_json
from schnitz.store import DiskStore
from schnitz.training_bank import TrainingBank
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def evaluate(run: Path, bank_dir: Path, episodes_file: Path, output: Path, *,
             max_episodes: int = 64, journal: Path | None = None) -> dict:
    if max_episodes < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive episode count and fresh output path required')
    config = config_from_run(run)
    bank_path = bank_dir / 'manifest.json'
    manifest = json.loads(bank_path.read_text())
    store = DiskStore(bank_dir / 'bank.sqlite')
    verified = publish_offline_generation(store, identity=manifest['identity'],
        namespace=manifest['namespace'], generation=manifest['generation'],
        spaces=tuple(manifest['spaces']), shard_ids=tuple(manifest['shards']),
        source_count=manifest['sources'], verify_only=True)
    if canonical_json(verified) != canonical_json({key: manifest[key] for key in verified}):
        raise ValueError('Published bank failed byte verification')
    index = PublishedKeyIndex(store, namespace=manifest['namespace'],
                              generation=manifest['generation'],
                              spaces=tuple(manifest['spaces']),
                              expected_sources=manifest['sources'])
    journal_state = None
    if journal is not None:
        staged_manifest = json.loads((journal.parent / 'manifest.json').read_text())
        if (not staged_manifest.get('complete')
                or staged_manifest['bank_manifest_sha256'] != file_sha256(bank_path)
                or Path(staged_manifest['parent_run']).resolve() != run.resolve()):
            raise ValueError('Rank evaluation needs a complete compatible journal')
        mutable = DiskStore(journal)
        if mutable.mutable_bank_state() != staged_manifest['journal_state']:
            raise ValueError('Rank journal changed since publication')
        verify_refresh_coverage(
            journal, namespace=manifest['namespace'],
            spaces=tuple(manifest['spaces']), source_count=manifest['sources'],
            parent_cursor=staged_manifest['parent_bank_state']['cursor'])
        TrainingBank(store, mutable, index)
        journal_state = staged_manifest['journal_state']
    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    checkpoint = resolve_checkpoint(run, verify=True)
    if journal is None:
        compatibility = assert_bank_reader_compatible(
            run, checkpoint, bank_dir, manifest, config)
    else:
        if (staged_manifest['writer_checkpoint_sha256']
                != file_sha256(checkpoint / 'model.safetensors')
                or stored_memory_identity(asdict(config.memory))
                != stored_memory_identity(dict(manifest['identity']['memory']))
                or any(manifest['identity']['model'][name] != getattr(config.model, name)
                       for name in ('model_id', 'revision'))):
            raise ValueError('Complete journal and reader checkpoint are incompatible')
        compatibility = {'relationship': 'fully_refreshed_mutable_bank',
                         'checkpoint_is_journal_writer_snapshot': True}
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError('Routing evaluation must not re-encode sources')
    agent.produce = forbidden_writer
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes:
        raise ValueError('Routing evaluation needs episodes')
    rows = []
    with torch.no_grad(), autocast_context(config):
        for episode in episodes:
            if episode.support_annotation != 'verified':
                raise ValueError('Only verified support labels can measure routing rank')
            ranks = []
            def provider(completed, _query, routing_query):
                if completed != 1:
                    return None
                for space in range(len(config.memory.payload_dims)):
                    address = agent.routing_address(routing_query, space)[0]
                    plan = index.search(address, top_k=manifest['sources'],
                                        namespace=manifest['namespace'], space=f's{space}',
                                        generation=manifest['generation'],
                                        domain=episode.provenance.get('domain', 'research'),
                                        query_time=episode.query_time)
                    positions = {item.record_id: i + 1 for i, item in
                                 enumerate(plan.selections)}
                    ranks.append(max(positions.get(record_id, manifest['sources'] + 1)
                                     for record_id in episode.required_ids))
                return None
            agent.plan_loop_memory(agent.prompt_ids(episode.query), provider,
                                   include_routing_query=True)
            if len(ranks) != len(config.memory.payload_dims):
                raise ValueError('Causal native read was not reached')
            rows.append({'episode_id': episode.episode_id, 'support_ranks': ranks})
    summary = []
    for space in range(len(config.memory.payload_dims)):
        values = [row['support_ranks'][space] for row in rows]
        summary.append({'space': f's{space}', 'median_rank': statistics.median(values),
                        'mean_reciprocal_rank': sum(1 / value for value in values) / len(values),
                        'recall_at_1': sum(value <= 1 for value in values) / len(values),
                        'recall_at_8': sum(value <= 8 for value in values) / len(values),
                        'recall_at_128': sum(value <= 128 for value in values) / len(values)})
    result = {'protocol': 'Causal prefix, exact stored-key rank; no payload fetch or writer call.',
              'episodes': len(rows), 'source_count': manifest['sources'],
              'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
              'bank_manifest_sha256': file_sha256(bank_path),
              'episodes_sha256': file_sha256(episodes_file),
              'bank_compatibility': compatibility,
              'mutable_journal_state': journal_state,
              'summary': summary, 'rows': rows}
    output.mkdir()
    atomic_json(output / 'results.json', result)
    return {key: value for key, value in result.items() if key != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=64)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.bank, args.episodes, args.output,
                              max_episodes=args.max_episodes,
                              journal=args.journal), indent=2))
