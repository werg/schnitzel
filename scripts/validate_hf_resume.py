"""Bounded actual-model Muon emergency-resume check; artifacts stay outside Git."""
from pathlib import Path
import argparse
import json

import torch
from safetensors.torch import load_file

from schnitz.checkpoints import resolve_checkpoint
from schnitz.operations import request_stop
from schnitz.replay import ReplayTape
from schnitz.training import train, config_from_run


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            equal_state(left, right)
    else:
        assert a == b


def validate(source, output, *, alignment_weight=0.):
    output.mkdir(parents=True, exist_ok=False)
    config = config_from_run(source)
    config.model.freeze_backbone = False
    config.model.backbone_train_scope = 'recurrent_core'
    config.train.optimizer = 'muon'
    config.train.oracle_alignment_weight = alignment_weight
    if alignment_weight:
        config.train.oracle_anchor_weight = .1
        config.train.oracle_anchor_loops = 1
    config.train.steps = 2
    config.train.loop_counts = [2, 3]
    config.train.gradient_accumulation = 2
    config.train.live_fraction = .5
    config.train.checkpoint_every = 1000
    config.train.keep_checkpoints = 1
    config.train.archive_dir = None
    config.train.wandb_mode = 'disabled'
    config.train.cuda_memory_fraction = .35
    config.train.min_system_available_bytes = 8 * 1024 ** 3
    config.train.stall_timeout_seconds = 300
    config.memory.noise_std = .02
    full, stopped = output / 'full', output / 'stopped'
    train(config, full, init_from=source)
    original = ReplayTape.backward
    def interrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        request_stop(stopped)
        return result
    ReplayTape.backward = interrupt
    try:
        partial = train(config, stopped, init_from=source)
    finally:
        ReplayTape.backward = original
    assert partial['steps'] == 0 and partial['saved_microbatches'] == 1
    resumed = train(config, stopped, resume=True)
    a, b = resolve_checkpoint(full, verify=True), resolve_checkpoint(stopped, verify=True)
    equal_state(load_file(str(a / 'model.safetensors')), load_file(str(b / 'model.safetensors')))
    equal_state(torch.load(a / 'training_state.pt', weights_only=True),
                torch.load(b / 'training_state.pt', weights_only=True))
    report = dict(source=str(source), optimizer='muon', device=config.train.device,
                  oracle_alignment_weight=alignment_weight,
                  precision=config.train.precision, completed_steps=resumed['steps'],
                  interrupted_microbatches=partial['saved_microbatches'],
                  exact_model_and_optimizer_rng_state=True, full_checkpoint=str(a), resumed_checkpoint=str(b),
                  note='No optimizer update is taken from partial accumulation. Not a capability experiment.')
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alignment-weight', type=float, default=0.)
    args = parser.parse_args()
    validate(args.source, args.output, alignment_weight=args.alignment_weight)
