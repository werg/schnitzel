"""Explicit flat-to-positional distillation and positional slot expansion helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class StateMigration:
    copied: tuple[str, ...]
    translated: tuple[str, ...]
    initialized: tuple[str, ...]
    expanded: tuple[str, ...] = ()


def initialize_positional_student(teacher, student) -> StateMigration:
    """Copy the stable system and useful reader maps into an eight-slot student."""
    if teacher.config.memory.payload_layout != 'flat':
        raise ValueError('The distillation teacher must use the legacy flat layout')
    if student.config.memory.payload_layout not in {'positional', 'joint_tokens'}:
        raise ValueError('The distillation student must use a structured layout')
    old, new = teacher.config.memory, student.config.memory
    if (old.write_slots, old.read_slots) != (new.write_slots, new.read_slots):
        raise ValueError('Eight-slot distillation preserves writer and read slot counts')
    if new.payload_layout == 'positional' and old.payload_dims != new.payload_dims:
        raise ValueError('Positional distillation preserves payload widths')
    teacher_state, student_state = teacher.state_dict(), student.state_dict()
    copied, translated = [], []
    excluded = ('codecs.', 'reader.', 'compactor.')
    with torch.no_grad():
        for name, target in student_state.items():
            source = teacher_state.get(name)
            if (not name.startswith(excluded) and source is not None
                    and source.shape == target.shape):
                target.copy_(source)
                copied.append(name)
        # Reader residual/update/output maps have identical meanings. Operator
        # input/relative/source-position maps remain newly initialized.
        for name, target in student_state.items():
            source_name = name
            if '.blocks.' in name and name.endswith('.message.weight'):
                source_name = name.removesuffix('.message.weight') + '.output.weight'
            elif '.blocks.' in name and name.endswith('.target_position'):
                source_name = name.removesuffix('.target_position') + '.slot'
            source = teacher_state.get(source_name)
            if source is not None and source.shape == target.shape:
                target.copy_(source)
                translated.append(f'{source_name}->{name}' if source_name != name else name)
        student.load_state_dict(student_state)
    touched = set(copied) | {item.rsplit('->', 1)[-1] for item in translated}
    initialized = tuple(sorted(set(student_state) - touched))
    return StateMigration(tuple(sorted(copied)), tuple(sorted(translated)), initialized)


def _expanded_rows(source: Tensor, target: Tensor, old_rows: int,
                   generator: torch.Generator) -> None:
    target[:old_rows].copy_(source)
    if target.shape[0] == old_rows:
        return
    base = source.float().mean(0, keepdim=True)
    scale = source.float().std(0, unbiased=False, keepdim=True).clamp_min(1e-4)
    noise = torch.randn((target.shape[0] - old_rows, *target.shape[1:]),
                        generator=generator, device='cpu', dtype=torch.float32)
    value = base.cpu() + 0.05 * scale.cpu() * noise
    target[old_rows:].copy_(value.to(device=target.device, dtype=target.dtype))


def expand_positional_state(source, target, *, seed: int = 1701) -> StateMigration:
    """Expand an eight-position student while preserving every shared parameter."""
    old, new = source.config.memory, target.config.memory
    if (old.payload_layout != new.payload_layout
            or old.payload_layout not in {'positional', 'joint_tokens'}):
        raise ValueError('Slot expansion requires matching structured interfaces')
    if new.write_slots <= old.write_slots or new.read_slots <= old.read_slots:
        raise ValueError('Target slot counts must increase')
    if old.payload_layout == 'joint_tokens':
        # Joint codecs and stored token tables do not depend on writer slots.
        if (old.payload_dims, old.space_tokens) != (new.payload_dims, new.space_tokens):
            raise ValueError('Joint token expansion must preserve stored token layouts')
    else:
        old_channels = [d // old.write_slots for d in old.payload_dims]
        new_channels = [d // new.write_slots for d in new.payload_dims]
        if old_channels != new_channels or len(old_channels) != len(new_channels):
            raise ValueError('Positional expansion must preserve per-space channel widths')
    if source.config.memory.compaction != 'none' or target.config.memory.compaction != 'none':
        raise ValueError('Distill the positional compactor after expanding the raw interface')
    source_state, target_state = source.state_dict(), target.state_dict()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    copied, expanded = [], []
    row_names: dict[str, int] = {
        'write_slots': old.write_slots + 1,
        'loop_workspace': old.read_slots,
    }
    for name in target_state:
        if name.endswith(('.initial_slots', '.null_tokens', '.target_position')):
            row_names[name] = old.read_slots
        if name.endswith('.source_position') and old.payload_layout == 'positional':
            row_names[name] = old.write_slots
    with torch.no_grad():
        for name, target_tensor in target_state.items():
            source_tensor = source_state.get(name)
            if source_tensor is None:
                continue
            if source_tensor.shape == target_tensor.shape:
                target_tensor.copy_(source_tensor)
                copied.append(name)
                continue
            if name in row_names and source_tensor.shape[1:] == target_tensor.shape[1:]:
                _expanded_rows(source_tensor, target_tensor, row_names[name], generator)
                expanded.append(name)
                continue
            if name.endswith('.slot_mix.0.weight'):
                # Kept for defensive compatibility with alternative naming.
                pass
            if '.slot_mix.' in name and source_tensor.ndim == target_tensor.ndim == 2:
                if source_tensor.shape[0] != source_tensor.shape[1]:
                    raise ValueError(f'Expected square slot mixer: {name}')
                target_tensor.zero_()
                count = source_tensor.shape[0]
                target_tensor[:count, :count].copy_(source_tensor)
                target_tensor[count:, count:].copy_(torch.eye(
                    target_tensor.shape[0] - count, device=target_tensor.device,
                    dtype=target_tensor.dtype))
                expanded.append(name)
                continue
            raise ValueError(f'Unrecognized parameter shape change during expansion: {name}: '
                             f'{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}')
        target.load_state_dict(target_state)
    touched = set(copied) | set(expanded)
    initialized = tuple(sorted(set(target_state) - touched))
    if initialized:
        raise ValueError(f'Expansion left parameters without explicit provenance: {initialized}')
    return StateMigration(tuple(sorted(copied)), (), initialized, tuple(sorted(expanded)))


def new_slice_masks(source, target) -> dict[str, Tensor]:
    """Return masks that let a local warmup update only newly added tensor slices."""
    old = source.config.memory
    masks = {}
    for name, parameter in target.named_parameters():
        source_parameter = dict(source.named_parameters()).get(name)
        if source_parameter is None or parameter.shape == source_parameter.shape:
            continue
        mask = torch.zeros_like(parameter)
        if name == 'write_slots':
            mask[old.write_slots + 1:] = 1
        elif name == 'loop_workspace':
            mask[old.read_slots:] = 1
        elif name.endswith('.source_position'):
            mask[old.write_slots:] = 1
        elif name.endswith(('.initial_slots', '.null_tokens', '.target_position')):
            mask[old.read_slots:] = 1
        elif '.slot_mix.' in name:
            mask[old.read_slots:, :] = 1
            mask[:, old.read_slots:] = 1
        else:
            raise ValueError(f'No new-slice mask for expanded parameter: {name}')
        masks[name] = mask
    return masks
