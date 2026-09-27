"""Representative binding-update profile, with optional bounded Torch trace.

Run checkpointing on/off sequentially against the same frozen warm-start.
Measurements cover this synthetic shape distribution and existing GPU contention.
They do not predict long-trajectory memory use or cold-storage performance.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import json
import statistics

import torch

from schnitz import optimizers
from schnitz.training import train, config_from_run
from schnitz.probes import model_probe


def profile(source, output, checkpointing, updates, trace):
    if updates < 100:
        raise ValueError('Use at least 100 updates to cover representative sampled shapes')
    output.mkdir(parents=True, exist_ok=False)
    config = config_from_run(source)
    config.model.freeze_backbone = False
    config.model.backbone_train_scope = 'recurrent_core'
    config.model.gradient_checkpointing = checkpointing
    config.memory.checkpoint_chunks = checkpointing
    config.train.optimizer = 'muon'
    config.train.steps = updates
    config.train.loop_counts = [2, 3]
    config.train.gradient_accumulation = 4
    config.train.live_fraction = 1.
    config.train.checkpoint_every = 1000
    config.train.keep_checkpoints = 1
    config.train.archive_dir = None
    config.train.wandb_mode = 'disabled'
    config.train.cuda_memory_fraction = .35
    config.train.min_system_available_bytes = 8 * 1024 ** 3
    config.train.stall_timeout_seconds = 300
    config.train.oracle_anchor_weight = .1
    config.train.oracle_anchor_loops = 1
    (output / 'preflight.json').write_text(json.dumps(model_probe(config), indent=2) + '\n')
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=20, warmup=1, active=3, repeat=1),
        record_shapes=True) if trace else None
    original_factory = optimizers.make_optimizer
    def make_optimizer(agent):
        optimizer = original_factory(agent)
        original_step = optimizer.step
        def step():
            result = original_step()
            profiler.step()
            return result
        if profiler is not None:
            optimizer.step = step
        return optimizer
    optimizers.make_optimizer = make_optimizer
    try:
        with profiler if profiler is not None else nullcontext():
            summary = train(config, output / 'training', init_from=source)
    finally:
        optimizers.make_optimizer = original_factory
    if profiler is not None:
        profiler.export_chrome_trace(str(output / 'trace.json'))
        (output / 'operators.txt').write_text(profiler.key_averages().table(
            sort_by='self_device_time_total', row_limit=30))
    rows = [json.loads(line) for line in (output / 'training/metrics.jsonl').read_text().splitlines()]
    # Exclude startup, warmup and all traced steps from the throughput window.
    times = [b['elapsed_seconds'] - a['elapsed_seconds'] for a, b in zip(rows, rows[1:]) if b['step'] > 30]
    result = dict(checkpointing=checkpointing, optimizer='muon', source=str(source), updates=updates,
                  warmed_updates=len(times), median_update_seconds=statistics.median(times),
                  p95_update_seconds=float(torch.tensor(times).quantile(.95)),
                  resources=summary['resources'], trace=trace,
                  notice='Synthetic binding distribution, shared GPU, warm model/cache; not a general throughput claim.')
    (output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpointing', choices=['on', 'off'], required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    profile(args.source, args.output, args.checkpointing == 'on', args.updates, args.trace)
