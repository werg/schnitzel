"""Evaluate a training run's checkpoints periodically, with memory-use interventions.

Polls the run's checkpoints and evaluates every step that is a multiple of
``--every`` (and not yet evaluated) with ``evaluate_key_table.py``, one at a
time, newest first. Each evaluation writes ``<output>/step-<N>/`` and appends
one summary line to ``<output>/summary.jsonl``. Stop with ``<output>/STOP``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def _steps(run: Path) -> list[int]:
    steps = []
    for path in (run / 'checkpoints').glob('step-*'):
        manifest = path / 'manifest.json'
        if path.is_dir() and manifest.is_file():
            steps.append(json.loads(manifest.read_text())['step'])
    return sorted(steps)


def _summary(step: int, result: dict) -> dict:
    line = {'step': step, 'rows': result['rows'],
            'union_recall': result['union_any_support_recall'],
            'recall': [space['any_support_recall'] for space in result['spaces']],
            'all_recall': [space['all_support_recall'] for space in result['spaces']]}
    for name, entry in result.get('conditions', {}).items():
        line[f'{name}_answer_nll'] = entry['site_answer_nll']
    for name, entry in result.get('memory_use', {}).items():
        if isinstance(entry, dict):
            line[f'{name}_delta'] = entry['nll_minus_normal']
            if 'gold_delivered_nll_minus_normal' in entry:
                line[f'{name}_delta_gold_delivered'] = entry['gold_delivered_nll_minus_normal']
        else:
            line[name] = entry
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--every', type=int, default=500)
    parser.add_argument('--poll-seconds', type=float, default=300)
    parser.add_argument('--once', action='store_true', help='evaluate what exists, then exit')
    parser.add_argument('evaluate_args', nargs=argparse.REMAINDER,
                        help='after --: arguments for evaluate_key_table.py '
                             '(--bank, --data, --conditions, ...)')
    args = parser.parse_args()
    extra = [a for a in args.evaluate_args if a != '--']
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('evaluate_key_table.py')
    failed: set[int] = set()
    while not (args.output / 'STOP').exists():
        pending = [step for step in _steps(args.run)
                   if step % args.every == 0 and step not in failed
                   and not (args.output / f'step-{step:09d}' / 'eval.json').is_file()]
        if pending:
            step = pending[-1]
            target = args.output / f'step-{step:09d}'
            started = time.time()
            completed = subprocess.run(
                [sys.executable, str(script), '--run', str(args.run), '--step', str(step),
                 '--output', str(target), *extra],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if completed.returncode:
                failed.add(step)
                print(json.dumps({'step': step, 'error': completed.stderr[-2000:]}), flush=True)
                continue
            result = json.loads((target / 'eval.json').read_text())
            line = _summary(step, result) | {'seconds': round(time.time() - started)}
            with (args.output / 'summary.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(line) + '\n')
            print(json.dumps(line), flush=True)
            continue
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
