#!/usr/bin/env python3
"""Run the CPU bootstrap/compaction diagnostic synchronously, including failed controls.

All datasets are generated from separate train/development/confirmation namespaces.
--max-steps is a runtime smoke override, not a reproduction of reported accuracy.
No pretrained model, paid service, or background process is involved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import yaml

from schnitz.config import load_config
from schnitz.data import make_boolean_world, save_episodes


def run(command: list[str], log: Path) -> dict:
    print('+ ' + ' '.join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    # CLI train/eval emit one final pretty JSON object after any metric lines.
    start = result.stdout.find('{\n')
    return json.loads(result.stdout[start:])


def reproduce(output: Path, *, resume: bool = False, max_steps: int | None = None, smoke: bool = False) -> None:
    if smoke:
        max_steps = 1
    if max_steps is not None and max_steps < 1:
        raise ValueError('--max-steps must be positive')
    output.mkdir(parents=True, exist_ok=resume)
    root = Path(__file__).resolve().parents[1]
    sets = {
        'xor-train': (256, 'boolean-train-v2', ('xor',)),
        'mixed-train': (256, 'boolean-train-v2', ('a', 'b', 'xor')),
        'xor-dev': (64, 'boolean-test-v2', ('xor',)),
        'mixed-dev': (64, 'boolean-test-v2', ('a', 'b', 'xor')),
        'confirmation': (256, 'untouched-confirm-v2', ('a', 'b', 'xor')),
    }
    for name, (count, split, operations) in sets.items():
        if smoke:
            count = 4
        save_episodes(output / f'{name}.jsonl', [e for i in range(count) for e in
                     make_boolean_world(i, split=split, operations=operations)])
    stages = [
        ('raw', 'tiny_boolean_cpu.yaml', 'xor-train', 'xor-dev', None),
        ('mixed', 'tiny_boolean_mixed_cpu.yaml', 'mixed-train', 'mixed-dev', None),
        ('oracle-text', 'tiny_boolean_oracle_cpu.yaml', 'mixed-train', 'mixed-dev', None),
        ('warm', 'tiny_boolean_warm_cpu.yaml', 'mixed-train', 'confirmation', 'oracle-text'),
        ('compact', 'tiny_boolean_compactor_cpu.yaml', 'xor-train', 'confirmation', 'warm'),
    ]
    reports = {'schema_version': 1, 'max_steps_override': max_steps, 'smoke': smoke, 'stages': {},
               'notice': 'One training seed; synthetic prototype, not a large-model or resource-frontier result.'}
    for name, filename, dataset, test, init in stages:
        config = load_config(root / 'configs' / filename)
        config.train.episodes_file = str(output / f'{dataset}.jsonl')
        if max_steps is not None:
            config.train.steps = min(config.train.steps, max_steps)
        path = output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        directory = output / name
        command = [sys.executable, '-m', 'schnitz.cli', 'train', '--config', str(path), '--output', str(directory)]
        if (directory / 'CURRENT').exists():
            command.append('--resume')
        elif init:
            command.extend(['--init-from', str(output / init)])
        trained = run(command, output / f'{name}-training.txt')
        command = [sys.executable, '-m', 'schnitz.cli', 'evaluate-transfer', '--run', str(directory),
                   '--episodes', str(output / f'{test}.jsonl')]
        if name == 'warm':
            command.extend(['--drop-supports', '--boolean-counterfactuals'])
        if name == 'compact':
            command.extend(['--compact', '--persistent-compact'])
        evaluated = run(command, output / f'{name}-evaluation.txt')
        reports['stages'][name] = {'training': trained, 'evaluation': evaluated}
        (output / 'study.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f'Results: {output / "study.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--smoke', action='store_true', help='Four worlds and one step per stage; runtime only')
    args = parser.parse_args()
    reproduce(args.output, resume=args.resume, max_steps=args.max_steps, smoke=args.smoke)
