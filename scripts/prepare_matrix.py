#!/usr/bin/env python3
"""Prepare, but do not launch, an information-matched pretrained experiment matrix."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import shlex

import yaml

from schnitz.config import load_config
from schnitz.data import make_multiuse_world, save_episodes


def prepare(config_path, output, *, steps=200, worlds=64, eval_worlds=32, bindings=2, seeds=(17,)):
    base = load_config(config_path)
    if base.model.backend == 'hf' and base.model.revision in {'main', 'master', ''}:
        raise ValueError('Pin the pretrained revision before preparing the matrix')
    if min(steps, worlds, eval_worlds, bindings) < 1 or not seeds:
        raise ValueError('Positive experiment counts required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', '# Run from the repository root inside the Spark container.']
    arms = ('oracle_text', 'no_memory', 'shared_compute', 'memory', 'attention', 'one_round', 'two_reads')
    runs = []
    for seed in seeds:
        train_path, test_path = output / f'train-{seed}.jsonl', output / f'test-{seed}.jsonl'
        save_episodes(train_path, [e for i in range(worlds) for e in make_multiuse_world(i, split=f'matrix-train-{seed}', bindings=bindings)])
        save_episodes(test_path, [e for i in range(eval_worlds) for e in make_multiuse_world(i, split=f'matrix-test-{seed}', bindings=bindings)])
        for arm in arms:
            config = deepcopy(base)
            config.train.seed = seed
            config.train.steps = steps
            config.train.episodes_file = str(train_path)
            config.train.optimization_scope = 'all'
            config.train.arm = arm if arm in {'oracle_text','no_memory','shared_compute'} else 'memory'
            config.train.retrieval = 'oracle'
            config.memory.compaction = 'none'
            config.memory.merge_loss_weight = 0.
            config.memory.reader = 'attention' if arm == 'attention' else 'mlp'
            config.memory.reader_rounds = 1 if arm == 'one_round' else base.memory.reader_rounds
            config.memory.read_steps = 2 if arm == 'two_reads' else 1
            config.memory.read_top_k = 1
            config.memory.stream_reads = False
            config.validate()
            name = f'{arm}-seed{seed}'
            path, run = output / f'{name}.yaml', output / name
            path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
            train = ['schnitz','train','--config',str(path),'--output',str(run)]
            evaluate = ['schnitz','evaluate-transfer','--run',str(run),'--episodes',str(test_path)]
            commands.extend([f'echo {shlex.quote(name)}',
                             f'if [[ -f {shlex.quote(str(run / "CURRENT"))} ]]; then',
                             '  ' + shlex.join(train + ['--resume']), 'else',
                             '  ' + shlex.join(train), 'fi', shlex.join(evaluate)])
            runs.append({'arm': arm, 'seed': seed, 'config': str(path), 'run': str(run), 'test': str(test_path)})
    script = output / 'run_all.sh'
    script.write_text('\n'.join(commands) + '\n')
    script.chmod(0o755)
    manifest = {'base_config': str(config_path), 'model_revision': base.model.revision, 'runs': runs,
                'notice': 'Oracle evidence and common data; configurations are not claimed perfectly compute matched.'}
    (output / 'matrix.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--eval-worlds', type=int, default=32)
    parser.add_argument('--bindings', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17])
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.output, steps=args.steps, worlds=args.worlds,
                             eval_worlds=args.eval_worlds, bindings=args.bindings, seeds=tuple(args.seeds)), indent=2))
