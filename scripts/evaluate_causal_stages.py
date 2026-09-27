#!/usr/bin/env python3
"""Compare completed curriculum stages on the same stored-transfer questions.

Run after training. Each stage writes its own frozen bank; this compares trained
systems, not inference depth at a fixed bank or a resource-matched frontier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from schnitz.checkpoints import resolve_checkpoint, stop_on_signal
from schnitz.config import load_config
from schnitz.evaluation import evaluate_transfer_run
from schnitz.launch import file_sha256
from schnitz.operations import atomic_json, run_lock, run_status, stop_requested


def wait_for_completion(run, *, startup_timeout=60):
    if startup_timeout < 0:
        raise ValueError('Startup timeout must be nonnegative')
    print('Waiting for the managed curriculum to complete', flush=True)
    deadline = time.monotonic() + startup_timeout
    registered = False
    while True:
        state = run_status(run)
        if state['stop_requested']:
            raise RuntimeError(f"Curriculum did not complete: {state['status']}")
        if state['status'] == 'complete' and not state['running']:
            return
        if state['status'] == 'not_managed' and not registered:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Curriculum did not register within the startup timeout')
            time.sleep(min(10, remaining))
            continue
        registered = True
        if state['status'] not in {'running', 'starting', 'complete'}:
            raise RuntimeError(f"Curriculum did not complete: {state['status']}")
        time.sleep(10)


def compare(run: Path, episodes: Path | None = None):
    run = run.resolve()
    manifest = json.loads((run / 'launch.json').read_text())
    protocol = manifest['recipe']['protocol']
    if protocol not in {'causal', 'binding'}:
        raise ValueError('This comparison requires a causal or binding curriculum')
    episodes = (episodes or run / ('fresh-causal.jsonl' if protocol == 'causal' else 'fresh-multiuse.jsonl')).resolve()
    episode_hash = file_sha256(episodes)
    output = run / 'stage-transfer-comparison'
    with run_lock(run), stop_on_signal() as stop:
        checkpoints = {}
        for stage in manifest['stages']:
            checkpoint = resolve_checkpoint(stage['run'], verify=True)
            saved = json.loads((checkpoint / 'manifest.json').read_text())
            if saved['step'] != load_config(stage['config']).train.steps:
                raise ValueError(f"Stage {stage['name']} has not completed its prescribed budget")
            checkpoints[stage['name']] = checkpoint
        output.mkdir(exist_ok=True)
        results = {}
        for stage in manifest['stages']:
            if stop['signal'] is not None or stop_requested(run):
                return {'status': 'stopped', 'completed_stages': list(results)}
            name = stage['name']
            identity = {'episodes_sha256': episode_hash,
                        'checkpoint_manifest_sha256': file_sha256(checkpoints[name] / 'manifest.json')}
            path = output / f'{name}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['identity'] != identity:
                    raise ValueError(f'Comparison inputs changed for {name}; retain the old report')
            else:
                print(f'Evaluating {name}', flush=True)
                report = evaluate_transfer_run(stage['run'], episodes,
                    drop_supports=True, boolean_counterfactuals=protocol == 'causal',
                    binding_counterfactuals=protocol == 'binding')
                record = {'identity': identity,
                          'report': {k: v for k, v in report.items() if k != 'rows'}}
                atomic_json(path, record)
            results[name] = record
        result = {'status': 'complete', 'episodes': str(episodes), 'stages': results,
                  'notice': 'One seed; separate frozen bank per trained stage. '
                            'Equal source information, not equal compute or parameter substitution.'}
        atomic_json(output / 'comparison.json', result)
        return {'status': 'complete', 'output': str(output / 'comparison.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path)
    parser.add_argument('--wait', action='store_true', help='Wait for a managed run; exit on stop or failure')
    args = parser.parse_args()
    if args.wait:
        wait_for_completion(args.run)
    print(json.dumps(compare(args.run, args.episodes), indent=2))
