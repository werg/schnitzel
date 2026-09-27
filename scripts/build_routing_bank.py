"""Offline frozen writer for a source-bound routing adapter and existing episodes."""
import argparse
import hashlib
from pathlib import Path

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation import build_shared_bank
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.offline_bank import canonical_json
from schnitz.operations import atomic_json, run_lock
from schnitz.store import DiskStore
from schnitz.training import config_from_run
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, episodes_file, routing_probe, count_policy, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=routing_probe,
                                          independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        agent.requires_grad_(False)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_file), 'routing_probe': adapter,
                    'read_count_policy': count}
        writer_identity = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        writes = build_shared_bank(agent, DiskStore(output / 'bank.sqlite'), load_episodes(episodes_file),
                                   writer_identity=writer_identity)
        # The in-database manifest is the atomic bank creation/recovery authority.
        atomic_json(output / 'bank-manifest.json', identity | {'offline_writer_identity': writer_identity,
            'script_sha256': file_sha256(__file__),
            'notice': 'Offline source encoding only; no inference or capability result.'})
        print(canonical_json({'offline_bank': str(output), 'writes': writes}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'routing-probe', 'count-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.episodes, args.routing_probe, args.count_policy, args.output)
