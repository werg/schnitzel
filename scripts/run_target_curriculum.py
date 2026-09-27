"""Run the sustained two-generation four-space target curriculum."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import yaml

from schnitz.checkpoints import resolve_checkpoint
from schnitz.config import load_config
from schnitz.operations import atomic_json
from schnitz.training import train
from schnitz.trajectories import file_sha256

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_source_bank import build as build_bank  # noqa: E402
from evaluate_published_bank import evaluate as evaluate_bank  # noqa: E402


def _episode_count(path: Path) -> int:
    with path.open(encoding='utf-8') as handle:
        return sum(bool(line.strip()) for line in handle)


def _write_config(path: Path, config) -> None:
    rendered = yaml.safe_dump(asdict(config), sort_keys=False)
    if path.exists() and path.read_text() != rendered:
        old, new = yaml.safe_load(path.read_text()), yaml.safe_load(rendered)
        allowed = {('model', 'gradient_checkpointing'), ('memory', 'checkpoint_chunks'),
                   ('train', 'gradient_accumulation'), ('train', 'batch_size'),
                   ('train', 'tokenized_episodes_file')}
        differences = {(section, key) for section in new for key in new[section]
                       if old.get(section, {}).get(key) != new[section][key]}
        if differences - allowed:
            raise ValueError(f'Existing stage configuration changed outside execution policy: {path}')
    path.write_text(rendered)


def _live_config(base, episodes: Path, *, steps: int, seed: int):
    config = load_config(base)
    config.model.freeze_backbone = True
    config.train.seed = seed
    config.train.steps = steps
    config.train.optimizer = 'muon'
    config.model.gradient_checkpointing = False
    config.memory.checkpoint_chunks = False
    config.train.gradient_accumulation = 1
    config.train.batch_size = 2
    config.train.sampling_policy = 'shuffled_passes'
    config.train.live_fraction = 1.0
    config.train.selected_producers_only = False
    config.train.arm = 'memory'
    config.train.retrieval = 'learned'
    config.train.routing_warmup = steps
    config.train.routing_weight = 1.0
    config.train.bank_dir = None
    config.train.bank_read_limits = []
    config.train.bank_payload_contrast_weight = 0.0
    config.train.episodes_file = str(episodes)
    config.train.tokenized_episodes_file = str(episodes.with_suffix('.tokens.jsonl'))
    config.train.max_source_tokens = 65
    config.train.max_prompt_tokens = 128
    config.train.max_target_tokens = 65
    config.train.checkpoint_every = 10_000
    config.train.keep_checkpoints = 2
    config.train.min_free_disk_bytes = 20 * 1024**3
    config.train.wandb_mode = 'offline'
    config.train.wandb_group = 'four-space-target-live'
    config.validate()
    return config


def _bank_config(live, bank: Path, *, steps: int, seed: int):
    config = load_config(live)
    config.train.seed = seed
    config.train.steps = steps
    config.train.routing_warmup = 0
    config.train.live_fraction = 0.0
    config.train.bank_dir = str(bank)
    config.train.bank_read_limits = [16, 8, 4, 4]
    config.train.bank_routing_candidates = 256
    config.train.bank_payload_contrast_weight = 0.0
    config.train.wandb_group = 'four-space-target-bank'
    config.validate()
    return config


def _run_stage(config, run: Path, *, init_from: Path) -> bool:
    resumed = (run / 'CURRENT').exists()
    if resumed:
        completed = json.loads((resolve_checkpoint(run, verify=True) /
                                'manifest.json').read_text())['step']
        if completed == config.train.steps:
            return True
        if completed > config.train.steps:
            raise ValueError('Stage checkpoint exceeds sealed curriculum budget')
    result = train(config, run, resume=resumed,
                   init_from=None if resumed else init_from)
    return result['steps'] == config.train.steps and not result['stopped_early']


def run(corpus: Path, output: Path, *, initial: Path,
        base_config: Path, live_passes: int = 2, bank_passes: int = 2) -> dict:
    if min(live_passes, bank_passes) < 1 or not corpus.is_dir() or not initial.is_dir():
        raise ValueError('Existing corpus/initial run and positive pass counts required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Target curriculum must run on the external disk')
    manifest_path = corpus / 'manifest.json'
    source_path, episodes, validation = (corpus / name for name in
        ('sources-all.jsonl', 'train.jsonl', 'validation.jsonl'))
    manifest = json.loads(manifest_path.read_text())
    expected = {name: manifest['files'][name]['sha256'] for name in
                ('sources-all.jsonl', 'train.jsonl', 'validation.jsonl')}
    for path in (source_path, episodes, validation):
        if file_sha256(path) != expected[path.name]:
            raise ValueError(f'Curriculum input changed: {path.name}')
    count = _episode_count(episodes)
    microbatches_per_pass = count
    examples_per_update = 2
    live_steps = math.ceil(microbatches_per_pass * live_passes / examples_per_update)
    bank_steps = math.ceil(microbatches_per_pass * bank_passes / examples_per_update)
    output.mkdir(exist_ok=True)
    (output / 'banks').mkdir(exist_ok=True)
    plan = {'format': 2, 'corpus_manifest_sha256': file_sha256(manifest_path),
            'initial_checkpoint': str(resolve_checkpoint(initial, verify=True)),
            'base_config_sha256': file_sha256(base_config), 'episodes': count,
            'gradient_accumulation': 1, 'batch_size': 2,
            'examples_per_update': examples_per_update,
            'execution_upgrade': 'padded-batch-v1',
            'live_passes_per_generation': live_passes,
            'bank_passes_per_generation': bank_passes, 'live_steps': live_steps,
            'bank_steps': bank_steps, 'generations': 2,
            'bank_sources': manifest['files']['sources-all.jsonl']['rows'],
            'read_limits': [16, 8, 4, 4], 'routing_candidates': 256}
    plan_path = output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        previous = json.loads(plan_path.read_text())
        immutable = ('corpus_manifest_sha256', 'initial_checkpoint', 'episodes',
                     'live_passes_per_generation', 'bank_passes_per_generation',
                     'live_steps', 'bank_steps', 'generations', 'bank_sources',
                     'read_limits', 'routing_candidates')
        if any(previous.get(key) != plan.get(key) for key in immutable):
            raise ValueError('Target curriculum data or scientific budget changed')
    atomic_json(plan_path, plan)

    parent = initial
    completed = []
    for generation in (1, 2):
        live_run = output / f'g{generation}-live'
        live_path = output / f'g{generation}-live.yaml'
        live = _live_config(base_config, episodes, steps=live_steps,
                            seed=100 + generation)
        _write_config(live_path, live)
        if not _run_stage(live, live_run, init_from=parent):
            return {'status': 'checkpointed', 'stage': f'g{generation}-live'}
        completed.append(f'g{generation}-live')

        bank = output / 'banks' / f'g{generation}-100k'
        if not (bank / 'manifest.json').exists():
            build_bank(live_run, source_path, bank,
                       max_sources=plan['bank_sources'], shard_size=64)
        completed.append(f'g{generation}-bank-build')
        handoff = output / 'HANDOFF_TO_SPATIAL.json'
        if generation == 1 and handoff.exists():
            return {'status': 'spatial_handoff', 'completed': completed,
                    'bank': str(bank), 'parent': str(live_run),
                    'request': json.loads(handoff.read_text())}

        bank_run = output / f'g{generation}-bank-train'
        bank_path = output / f'g{generation}-bank.yaml'
        bank_config = _bank_config(live_path, bank, steps=bank_steps,
                                   seed=200 + generation)
        _write_config(bank_path, bank_config)
        if not _run_stage(bank_config, bank_run, init_from=live_run):
            return {'status': 'checkpointed', 'stage': f'g{generation}-bank-train'}
        completed.append(f'g{generation}-bank-train')

        evaluation = output / f'g{generation}-evaluation'
        if not evaluation.exists():
            evaluate_bank(bank_run, bank, validation, evaluation,
                          max_episodes=min(1024, _episode_count(validation)),
                          limits=(16, 8, 4, 4), selection='learned',
                          generate_episodes=min(64, _episode_count(validation)),
                          generate_max_tokens=65)
        completed.append(f'g{generation}-evaluation')
        parent = bank_run
    result = {'status': 'complete', 'completed': completed,
              'final_run': str(parent), 'plan': plan}
    atomic_json(output / 'complete.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--base-config', type=Path,
                        default=Path('configs/lfm25_230m_four_space_short_local_routing_spark.yaml'))
    parser.add_argument('--live-passes', type=int, default=2)
    parser.add_argument('--bank-passes', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.output, initial=args.initial,
                         base_config=args.base_config, live_passes=args.live_passes,
                         bank_passes=args.bank_passes), indent=2))
