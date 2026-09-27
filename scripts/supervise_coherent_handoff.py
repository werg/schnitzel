"""Wait for a coherent staged bank, measure ranks, then launch one GPU trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from schnitz.operations import run_status, stop_requested


def complete(run: Path) -> bool:
    path = run / 'training_summary.json'
    return path.exists() and bool(json.loads(path.read_text()).get('complete'))


def execute(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        handle.write(json.dumps({'time': time.time(), 'command': command}) + '\n')
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main(args) -> None:
    refresh_manifest = args.refresh / 'manifest.json'
    refresh_log = args.refresh.parent / 'coherent-handoff.log'
    while True:
        if stop_requested(args.refresh):
            return
        if refresh_manifest.exists():
            manifest = json.loads(refresh_manifest.read_text())
            if manifest.get('complete') and not run_status(args.refresh)['running']:
                break
        if not run_status(args.refresh)['running']:
            execute([args.python, 'scripts/refresh_training_bank.py',
                     '--run', str(args.parent), '--bank', str(args.bank),
                     '--sources', str(args.sources), '--output', str(args.refresh),
                     '--chunk-size', '64', '--writer-batch-size', '16'], refresh_log)
        else:
            time.sleep(args.poll_seconds)
    if not (args.ranks / 'results.json').exists():
        args.ranks.parent.mkdir(parents=True, exist_ok=True)
        execute([args.python, 'scripts/evaluate_bank_ranks.py',
                 '--run', str(args.parent), '--bank', str(args.bank),
                 '--episodes', str(args.validation), '--output', str(args.ranks),
                 '--max-episodes', '128', '--journal',
                 str(args.refresh / 'training_cache.sqlite')], refresh_log)
    while not complete(args.output):
        if stop_requested(args.output):
            return
        if run_status(args.output)['running']:
            time.sleep(args.poll_seconds)
            continue
        command = [args.python, 'scripts/train_spatial_bank.py',
                   '--config', str(args.config), '--data', str(args.data),
                   '--bank', str(args.bank), '--output', str(args.output),
                   '--sources', str(args.sources),
                   '--routing-episodes', str(args.episodes),
                   '--routing-weight', '0.2',
                   '--key-stability-weight', '0.2',
                   '--routing-hard-ramp-steps', '1000',
                   '--writer-key-learning-rate', '0.000001',
                   '--steps', '1000', '--batch-size', '4', '--microbatch-size', '2',
                   '--inflight', '2', '--loops', '3', '--limits', '16', '8', '4', '4',
                   '--routing-candidates', '256', '--checkpoint-every', '500',
                   '--maintenance-records-per-step', '1',
                   '--train-recurrent-core', '--gradient-checkpointing',
                   '--retain-writer-replay-activations', '--max-unused-cuda-gib', '28',
                   '--cache-reclaim-host-reserve-gib', '16', '--profile-steps', '3']
        if (args.output / 'CURRENT').exists():
            command.append('--resume')
        elif args.output.exists():
            raise ValueError('Incomplete output has no committed recovery checkpoint')
        else:
            command.extend(('--init-from', str(args.parent), '--bank-journal',
                            str(args.refresh / 'training_cache.sqlite')))
        execute(command, refresh_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--refresh', type=Path, required=True)
    parser.add_argument('--ranks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error('--poll-seconds must be at least 30')
    main(args)
