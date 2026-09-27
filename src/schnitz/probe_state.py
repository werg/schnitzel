"""Durable optimizer-boundary recovery for small standalone feature probes.

Callers own the run lock, complete each optimizer update, and supply immutable
source/data/config identity. No pending gradient accumulation is supported here;
the main trainer has its separate microbatch/replay checkpoint contract.
"""
from pathlib import Path
import random

import torch

from .checkpoints import _fsync, _fsync_dir, reconcile_metrics
from .archiving import ensure_free


def parameter_names(model, optimizer):
    named = {id(p): name for name, p in model.named_parameters()}
    owned = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(owned) != len(set(owned)) or set(owned) != expected:
        raise ValueError('Probe optimizer ownership must cover each trainable parameter exactly once')
    return [[named[id(p)] for p in group['params']] for group in optimizer.param_groups]


def save_probe_state(path, model, optimizer, sampler, identity, step, *, reserve_bytes,
                     model_key='model', extra=None):
    path = Path(path)
    if step < 0 or model_key not in {'model', 'head', 'compactor'}:
        raise ValueError('Nonnegative completed step and supported model key required')
    state = {'format': 'sdkb-probe-v1', 'identity': identity, model_key: model.state_dict(),
             'optimizer': optimizer.state_dict(), 'parameter_names': parameter_names(model, optimizer),
             'step': step, 'sampling_rng': sampler.get_state(), 'python_rng': random.getstate(),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if extra:
        if set(extra) & set(state):
            raise ValueError('Extra probe metadata shadows recovery state')
        state.update(extra)
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(map(tensor_bytes, value))
        return 0
    ensure_free(path.parent, tensor_bytes(state) + 1024 * 1024, reserve_bytes)
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(state, temporary)
        _fsync(temporary)
        temporary.replace(path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def restore_probe_state(path, model, optimizer, sampler, identity, *, model_key='model', extra=None, metrics_root=None):
    path = Path(path)
    step = 0
    if path.exists():
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state.get('format') != 'sdkb-probe-v1' or state['identity'] != identity:
            raise ValueError('Probe resume format/identity changed')
        if state['parameter_names'] != parameter_names(model, optimizer):
            raise ValueError('Probe optimizer ownership changed')
        if any(state.get(k) != v for k, v in (extra or {}).items()):
            raise ValueError('Probe arm identity changed')
        if not isinstance(state['step'], int) or state['step'] < 0:
            raise ValueError('Invalid completed probe step')
        model.load_state_dict(state[model_key])
        optimizer.load_state_dict(state['optimizer'])
        sampler.set_state(state['sampling_rng'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        step = state['step']
    reconcile_metrics(Path(metrics_root) if metrics_root is not None else path.parent, step)
    return step
