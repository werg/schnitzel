"""Selective writer replay and post-update refresh for mutable training banks."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import Tensor

from .agent import SchnitzelAgent
from .replay import ReplayTape
from .store import StoredRecord
from .training_bank import TrainingBank


class BankWriterReplay:
    def __init__(self, agent: SchnitzelAgent, writer_inputs: Mapping[str, Tensor], *,
                 verify_outputs: bool = True, checkpoint_backward: bool = True,
                 capture_batch_size: int = 16) -> None:
        if capture_batch_size < 1:
            raise ValueError('Writer replay capture batch size must be positive')
        self.agent = agent
        self.writer_inputs = writer_inputs
        self.tape = ReplayTape(verify_outputs=verify_outputs)
        self.record_ids: list[str] = []
        self._leaves: dict[str, tuple[Tensor, ...]] = {}
        self.checkpoint_backward = checkpoint_backward
        self.capture_batch_size = capture_batch_size

    def _produce(self, record_ids: tuple[str, ...]) -> tuple[Tensor, ...]:
        outputs = self.agent.produce_batch([self.writer_inputs[record_id]
                                            for record_id in record_ids])
        dtype = getattr(torch, self.agent.config.memory.storage_dtype)
        serialized = []
        for space in range(len(self.agent.config.memory.payload_dims)):
            serialized.extend((outputs[2 * space].float(),
                               outputs[2 * space + 1].to(dtype).float()))
        return tuple(serialized)

    def capture(self, record_ids: tuple[str, ...]) -> dict[str, tuple[Tensor, ...]]:
        if not record_ids or len(record_ids) != len(set(record_ids)):
            raise ValueError('Writer replay needs distinct record IDs')
        missing = set(record_ids) - self.writer_inputs.keys()
        if missing:
            raise KeyError(f'Writer inputs missing for {sorted(missing)[:3]}')
        unseen = tuple(record_id for record_id in record_ids if record_id not in self._leaves)
        if unseen:
            ordered = sorted(unseen, key=lambda record_id: self.writer_inputs[record_id].shape[1])
            for start in range(0, len(ordered), self.capture_batch_size):
                batch = tuple(ordered[start:start + self.capture_batch_size])
                leaves = self.tape.capture(self.agent, lambda batch=batch: self._produce(batch))
                self.record_ids.extend(batch)
                self._leaves.update({
                    record_id: tuple(output[row:row + 1] for output in leaves)
                    for row, record_id in enumerate(batch)
                })
        return {record_id: self._leaves[record_id] for record_id in record_ids}

    def backward(self) -> None:
        with _checkpointing(self.agent, enabled=self.checkpoint_backward):
            self.tape.backward()

    @torch.no_grad()
    def refresh(self, bank: TrainingBank, *, batch_size: int = 32,
                additional_ids: tuple[str, ...] = (),
                optimizer_step: int | None = None) -> int:
        """Regenerate touched views after the optimizer step and commit atomically."""
        all_ids = (*self.record_ids, *additional_ids)
        ids = tuple(sorted(dict.fromkeys(all_ids),
                           key=lambda record_id: self.writer_inputs[record_id].shape[1]))
        records = []
        dtype = getattr(torch, self.agent.config.memory.storage_dtype)
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            outputs = self.agent.produce_batch([self.writer_inputs[record_id]
                                                for record_id in batch])
            for row, record_id in enumerate(batch):
                for space in range(len(self.agent.config.memory.payload_dims)):
                    records.append(StoredRecord(
                        record_id, outputs[2 * space][row].detach(),
                        outputs[2 * space + 1][row].to(dtype).detach(),
                        namespace=bank.index.namespace, space=f's{space}',
                        generation=bank.index.generation))
        return bank.update(records, optimizer_step=optimizer_step)


@contextmanager
def _checkpointing(module, *, enabled: bool):
    flags = [(child, child.gradient_checkpointing) for child in module.modules()
             if hasattr(child, 'gradient_checkpointing')]
    if not enabled:
        for child, _ in flags:
            child.gradient_checkpointing = False
    try:
        yield
    finally:
        for child, value in flags:
            child.gradient_checkpointing = value
