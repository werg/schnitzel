"""Offline frozen source encoding without routing or read-count overlays."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from schnitz.archiving import ensure_free
from schnitz.checkpoints import resolve_checkpoint, stop_on_signal
from schnitz.data import load_episodes
from schnitz.evaluation import build_shared_bank
from schnitz.evaluation_adapter import load_frozen_agent
from schnitz.offline_bank import canonical_json
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.runtime import available_host_memory, compute_watchdog
from schnitz.store import DiskStore
from schnitz.training import config_from_run
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, episodes_file, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        def check_stop():
            if stop['signal'] is not None or stop_requested(output):
                raise RuntimeError('Offline bank stop requested')
        check_stop()
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        if config.train.arm not in {'memory', 'direct_latent'}:
            raise ValueError('Frozen bank writing requires a memory-producing source')
        ensure_free(output, 0, config.train.min_free_disk_bytes)
        torch.set_num_threads(config.train.threads)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_file),
                    'routing_probe': None, 'read_count_policy': None}
        writer_identity = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        manifest_path = output/'bank-manifest.json'
        expected = identity | {'offline_writer_identity': writer_identity}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError('Published bank identity differs')
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        produce = agent.produce
        def guarded(ids):
            check_stop()
            available = available_host_memory()
            if available is not None and available < config.train.min_system_available_bytes:
                raise RuntimeError('Host memory reserve reached during offline writing')
            ensure_free(output, 0, config.train.min_free_disk_bytes)
            # The guard ends before SQLite writes/fsync, so slow storage is not
            # classified as a stalled model forward.
            with compute_watchdog(config.train.stall_timeout_seconds):
                result = produce(ids)
                if agent.device.type == 'cuda':
                    torch.cuda.synchronize(agent.device)
                return result
        agent.produce = guarded
        try:
            writes = build_shared_bank(agent, DiskStore(output/'bank.sqlite'), load_episodes(episodes_file),
                                       writer_identity=writer_identity)
        finally:
            agent.produce = produce
        if writes['writer_calls'] or not manifest_path.exists():
            atomic_json(manifest_path, expected | {'script_sha256': file_sha256(__file__),
                'notice': 'Separate offline encoding by a frozen writer. No inference or capability result.'})
        check_stop()
        print(canonical_json({'offline_bank': str(output), 'writes': writes}), flush=True)
        return writes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.episodes, args.output)
