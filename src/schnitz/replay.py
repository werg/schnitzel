"""Exact first-order producer VJPs for a specified mixed cached/live forward pass.

No optimizer step may occur between capture and replay. Source computations must
be pure apart from torch RNG. Mutable buffers, changed training modes and changed
parameters are rejected. Autocast dtype, enabled state and weight-cache policy are
restored. No differentiation through historical environment actions.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn


@dataclass
class _Record:
    module: nn.Module
    producer: Callable[[], tuple[Tensor, ...]]
    leaves: tuple[Tensor, ...]
    cpu_rng: Tensor
    cuda_rng: list[Tensor]
    parameter_versions: tuple[tuple[nn.Parameter, int], ...]
    buffer_values: tuple[tuple[Tensor, Tensor], ...]
    modes: tuple[tuple[nn.Module, bool], ...]
    autocast: dict[str, tuple[bool, torch.dtype]]
    autocast_cache_enabled: bool


@contextmanager
def _record_context(record: _Record) -> Iterator[None]:
    devices = list(range(len(record.cuda_rng)))
    with torch.random.fork_rng(devices=devices), ExitStack() as stack:
        torch.set_rng_state(record.cpu_rng)
        for device, state in enumerate(record.cuda_rng):
            torch.cuda.set_rng_state(state, device)
        for device, (enabled, dtype) in record.autocast.items():
            stack.enter_context(torch.autocast(device, enabled=enabled, dtype=dtype,
                                              cache_enabled=record.autocast_cache_enabled))
        yield


class ReplayTape:
    """One optimizer step's producer tape; leaf uses accumulate cotangents naturally.

    Leaves can be reused many times by the consumer. Cached entries are ordinary
    detached tensors and are NOT registered here. Higher-order gradients, changing
    read selections and distributed sharded parameters are outside this reference.
    """
    def __init__(self, verify_outputs: bool = False) -> None:
        self.records: list[_Record] = []
        self.verify_outputs = verify_outputs
        self.used = False

    def capture(self, module: nn.Module,
                producer: Callable[[], tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        if self.used:
            raise RuntimeError("A replay tape is single-use")
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        versions = tuple((p, p._version) for p in module.parameters())
        buffers = tuple((b, b.detach().clone()) for b in module.buffers())
        modes = tuple((m, m.training) for m in module.modules())
        devices = ["cpu"] + (["cuda"] if cuda_rng else [])
        autocast = {d: (torch.is_autocast_enabled(d), torch.get_autocast_dtype(d)) for d in devices}
        cache_enabled = torch.is_autocast_cache_enabled()
        with torch.no_grad():
            outputs = producer()
        if not isinstance(outputs, tuple) or not outputs or not all(
            isinstance(x, Tensor) and x.is_floating_point() for x in outputs
        ):
            raise TypeError("Producer must return a nonempty tuple of floating tensors")
        if any(not torch.equal(b, old) for b, old in buffers):
            with torch.no_grad():
                for b, old in buffers:
                    b.copy_(old)
            raise RuntimeError("Producer mutated a buffer; replay requires a stateless producer")
        if any(p._version != version for p, version in versions):
            raise RuntimeError("Producer mutated parameters")
        leaves = tuple(x.detach().requires_grad_(True) for x in outputs)
        self.records.append(_Record(module, producer, leaves, cpu_rng, cuda_rng,
                                    versions, buffers, modes, autocast, cache_enabled))
        return leaves

    def backward(self) -> None:
        if self.used:
            raise RuntimeError("Replay was already performed")
        self.used = True
        for record in self.records:
            if any(p._version != version for p, version in record.parameter_versions):
                raise RuntimeError("Parameters changed before replay; optimizer step is too early")
            if any(m.training != mode for m, mode in record.modes):
                raise RuntimeError("Module training mode changed before replay")
            if any(not torch.equal(b, old) for b, old in record.buffer_values):
                raise RuntimeError("Mutable buffer changed before replay")
            if not any(leaf.grad is not None for leaf in record.leaves):
                continue
            with _record_context(record), torch.enable_grad():
                outputs = record.producer()
                if len(outputs) != len(record.leaves):
                    raise RuntimeError("Producer output arity changed")
                active_outputs, gradients = [], []
                for output, leaf in zip(outputs, record.leaves, strict=True):
                    if self.verify_outputs:
                        torch.testing.assert_close(output.detach(), leaf.detach(), rtol=1e-5, atol=1e-6)
                    if leaf.grad is not None and output.requires_grad:
                        active_outputs.append(output)
                        gradients.append(leaf.grad.detach())
                if active_outputs:
                    torch.autograd.backward(active_outputs, gradients)
            if any(not torch.equal(b, old) for b, old in record.buffer_values):
                raise RuntimeError("Producer mutated buffers during replay")
        self.records.clear()
