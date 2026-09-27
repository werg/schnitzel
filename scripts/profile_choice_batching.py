"""Measure opt-in candidate batching against serial native BF16 scoring."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.choice_scoring import candidate_nll
from schnitz.data import load_episodes
from schnitz.evaluation_adapter import load_frozen_agent
from schnitz.operations import atomic_json
from schnitz.runtime import configure_memory, memory_metrics
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, bank, episodes_path, output, worlds=4, batch_size=4):
    if worlds < 1 or batch_size < 2:
        raise ValueError('Positive world count and batching of at least two candidates required')
    if output.exists():
        raise FileExistsError(output)
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(checkpoint)
    configure_memory(config.train)
    torch.set_num_threads(config.train.threads)
    agent, _ = load_frozen_agent(config, checkpoint)
    episodes = load_episodes(episodes_path)
    selected = set(list(dict.fromkeys(e.environment for e in episodes))[:worlds])
    episodes = [e for e in episodes if e.environment in selected]
    store = DiskStore(bank)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Candidate scoring re-encoded source data')
    agent.produce = forbidden
    def synchronize():
        if agent.device.type == 'cuda':
            torch.cuda.synchronize()
    rows = []
    with autocast_context(config):
        for index, episode in enumerate(episodes):
            prompt = agent.prompt_ids(episode.query)
            session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                   query_time=episode.query_time, oracle_ids=episode.required_ids)
            targets = [agent.target_ids(answer) for answer in episode.choices]
            # Warm both shapes for this question; exclude reads and warmup from timing.
            for size in (1, batch_size):
                candidate_nll(agent, prompt, session.memory, targets, max_batch=size)
            scores, elapsed = {}, {}
            for size in ((1, batch_size) if index % 2 == 0 else (batch_size, 1)):
                synchronize()
                start = time.perf_counter()
                scores[size] = candidate_nll(agent, prompt, session.memory, targets, max_batch=size)
                synchronize()
                elapsed[size] = time.perf_counter() - start
            serial, batched = scores[1], scores[batch_size]
            rows.append({'episode': episode.episode_id, 'environment': episode.environment,
                         'task_family': episode.task_family, 'serial_nll': serial, 'batched_nll': batched,
                         'maximum_absolute_nll_error': max(abs(a-b) for a, b in zip(serial, batched, strict=True)),
                         'same_choice': min(range(len(serial)), key=serial.__getitem__) == min(range(len(batched)), key=batched.__getitem__),
                         'serial_seconds': elapsed[1], 'batched_seconds': elapsed[batch_size]})
            if (index + 1) % 10 == 0:
                print(json.dumps({'profiled_queries': index + 1, 'total': len(episodes)}), flush=True)
    result = {'source_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
              'episodes_sha256': file_sha256(episodes_path), 'bank_sha256': file_sha256(bank),
              'batch_size': batch_size, 'queries': len(rows), 'rows': rows,
              'changed_choices': sum(not r['same_choice'] for r in rows),
              'maximum_absolute_nll_error': max(r['maximum_absolute_nll_error'] for r in rows),
              'median_serial_seconds': statistics.median(r['serial_seconds'] for r in rows),
              'median_batched_seconds': statistics.median(r['batched_seconds'] for r in rows),
              'resources': memory_metrics(config.train.device),
              'notice': 'Warm per-question shapes and captured memory; alternating timing order on a contended GPU. '
                        'No cold storage, global throughput, bit-exact BF16 or automatic adoption claim.'}
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    run(args.source, args.bank, args.episodes, args.output, args.worlds, args.batch_size)
