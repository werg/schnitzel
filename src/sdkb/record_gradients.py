"""Error-driven writer learning against a large mutable bank.

Three cooperating pieces keep stored keys coherent while the writer trains:

``KeyStateCache``
    One writer key-slot state per record. Stored keys are the direct key heads
    applied to these states, so every retrieved candidate's key is computed
    live with exact head gradients, and after every optimizer step the whole
    index is recomputed from the cache. Head updates never leave keys stale.

``RecordGradients``
    Per-record cotangents with respect to the writer outputs (the key-slot
    state and each space's payload), accumulated across steps with per-step
    decay and a last-updated step. The expensive writer backward runs only for
    a generous close neighbourhood of each step's retrievals, plus a small
    budget of the largest accumulated gradients elsewhere, using everything
    accumulated for those records.

``GradientSink``
    Collects this step's cotangents: consumer reads use leaf tensors for cached
    states and stored payloads, and their gradients are harvested after the
    consumer backward.

``KeyTable``
    Optional terminal keys: one trainable key per record and space, used by
    search and gates in place of head-derived keys. Retrieval gradients update
    only touched rows (per-row Adam); decoder drift never moves the index. The
    decoder learns to predict each row whenever it encodes that record
    (``writer_pass``), and a small commitment pull moves rows toward the
    prediction so keys stay reproducible for records the table never saw.

Accumulated cotangents are applied at the writer's current parameters, so a
record's older contributions are delayed gradients. This is a deliberate,
owner-approved estimator, not the exact full-graph replay reference; with a
decay of zero and a neighbourhood covering every read it reduces to that
reference for payload and key-state paths (tested).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import hashlib
import time

import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F


class KeyStateCache:
    """Writer key-slot states for every bank record, in index order."""

    def __init__(self, record_ids: Sequence[str], states: Tensor, versions: Tensor) -> None:
        if (states.ndim != 2 or len(record_ids) != len(states)
                or versions.shape != (len(states),)):
            raise ValueError('Key-state cache needs one state and version per record')
        self.ids = [str(record_id) for record_id in record_ids]
        if self.ids != sorted(self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError('Key-state cache IDs must be sorted and unique')
        self.position = {record_id: i for i, record_id in enumerate(self.ids)}
        self.states = states.float()
        self.versions = versions.long().cpu()

    def ids_digest(self) -> str:
        return hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()

    def positions(self, record_ids: Iterable[str]) -> Tensor:
        return torch.tensor([self.position[record_id] for record_id in record_ids],
                            dtype=torch.long)

    def keys(self, agent, positions: Tensor | None = None) -> tuple[Tensor, ...]:
        states = self.states if positions is None else self.states[positions.to(
            self.states.device)]
        return agent.writer_space_keys(states.to(agent.device))

    @torch.no_grad()
    def sync_index(self, agent, index, *, chunk: int = 65536) -> None:
        """Recompute every stored key from cached states with the current heads."""
        for space, array in enumerate(index.spaces.values()):
            if array.ids.tolist() != self.ids:
                raise ValueError('Key-state cache and index record order differ')
            parts = []
            for start in range(0, len(self.ids), chunk):
                states = self.states[start:start + chunk].to(agent.device)
                head = agent.writer_key_heads[space]
                parts.append(F.normalize(agent._fp32_head(head, states, 'writer'),
                                         dim=-1).float().cpu())
            array.keys[:] = torch.cat(parts).numpy()
        index.invalidate()

    def update(self, record_ids: Sequence[str], states: Tensor, step: int) -> Tensor:
        """Replace states; return each record's age in steps before the update."""
        positions = self.positions(record_ids)
        ages = step - self.versions[positions]
        self.states[positions.to(self.states.device)] = states.float().to(self.states.device)
        self.versions[positions] = step
        return ages

    def stalest(self, count: int, exclude: Iterable[str] = (),
                active: frozenset[str] | None = None) -> list[str]:
        if count <= 0:
            return []
        eligible = torch.ones(len(self.ids), dtype=torch.bool)
        if active is not None:
            if getattr(self, '_active_key', None) is not active:
                self._active_mask = torch.zeros(len(self.ids), dtype=torch.bool)
                self._active_mask[self.positions(sorted(active))] = True
                self._active_key = active
            eligible &= self._active_mask
        skip = [record_id for record_id in exclude if record_id in self.position]
        if skip:
            eligible[self.positions(skip)] = False
        positions = eligible.nonzero().squeeze(1)
        # Oldest first; ties keep index (record-ID) order, as the stable sort did.
        order = torch.argsort(self.versions[positions], stable=True)[:count]
        return [self.ids[i] for i in positions[order].tolist()]

    def state_dict(self) -> dict:
        return {'ids_sha256': self.ids_digest(), 'states': self.states.cpu(),
                'versions': self.versions.clone()}

    @classmethod
    def from_state_dict(cls, record_ids: Sequence[str], state: dict,
                        device: torch.device | str = 'cpu') -> 'KeyStateCache':
        cache = cls(record_ids, state['states'].to(device), state['versions'])
        if cache.ids_digest() != state['ids_sha256']:
            raise ValueError('Key-state cache belongs to another bank')
        return cache


class RecordGradients:
    """Decayed per-record cotangents for the writer's key-slot state and payloads."""

    def __init__(self, decay: float, *, capacity: int = 16384,
                 payload_dtype: torch.dtype = torch.bfloat16) -> None:
        if not 0 <= decay < 1 or capacity < 1:
            raise ValueError('Invalid record-gradient decay or capacity')
        self.decay, self.capacity, self.payload_dtype = decay, capacity, payload_dtype
        self.state: dict[str, Tensor] = {}
        self.payload: dict[str, list[Tensor | None]] = {}
        self.updated: dict[str, int] = {}
        self.norm_sq: dict[str, float] = {}  # at the record's last update step

    def __len__(self) -> int:
        return len(self.updated)

    def _decayed(self, record_id: str, step: int) -> float:
        return self.decay ** (step - self.updated[record_id])

    def _bring_to(self, record_id: str, step: int) -> None:
        factor = self._decayed(record_id, step)
        if factor != 1.0:
            self.state[record_id] = self.state[record_id] * factor
            self.payload[record_id] = [None if value is None else
                                       (value.float() * factor).to(self.payload_dtype)
                                       for value in self.payload[record_id]]
            self.norm_sq[record_id] *= factor * factor
        self.updated[record_id] = step

    def add(self, step: int, record_id: str, state: Tensor | None,
            payloads: Sequence[Tensor | None]) -> None:
        if record_id in self.updated:
            self._bring_to(record_id, step)
        else:
            self.state[record_id] = torch.zeros(0)
            self.payload[record_id] = [None] * len(payloads)
            self.updated[record_id] = step
            self.norm_sq[record_id] = 0.0
        if state is not None:
            current = self.state[record_id]
            state = state.detach().float().cpu()
            self.state[record_id] = state if not current.numel() else current + state
        merged = []
        for old, new in zip(self.payload[record_id], payloads, strict=True):
            if new is None:
                merged.append(old)
            else:
                new = new.detach().float().cpu()
                merged.append((new if old is None else old.float() + new).to(self.payload_dtype))
        self.payload[record_id] = merged
        total = self.state[record_id].pow(2).sum() if self.state[record_id].numel() else 0.0
        for value in merged:
            if value is not None:
                total = total + value.float().pow(2).sum()
        self.norm_sq[record_id] = float(total)

    def norm(self, record_id: str, step: int) -> float:
        return self.norm_sq[record_id] ** 0.5 * self._decayed(record_id, step)

    def evict(self, step: int) -> int:
        excess = len(self.updated) - self.capacity
        if excess <= 0:
            return 0
        ranked = sorted(self.updated, key=lambda record_id: self.norm(record_id, step))
        for record_id in ranked[:excess]:
            self.pop(record_id)
        return excess

    def select(self, step: int, neighborhood: Iterable[str], *, budget: int,
               extra: int = 0) -> list[str]:
        """Records to flush: the neighbourhood by accumulated norm, then the largest
        accumulated gradients elsewhere."""
        near = [record_id for record_id in dict.fromkeys(neighborhood)
                if record_id in self.updated]
        near.sort(key=lambda record_id: -self.norm(record_id, step))
        chosen = near[:budget]
        if extra > 0:
            taken = set(chosen)
            rest = sorted((record_id for record_id in self.updated if record_id not in taken),
                          key=lambda record_id: -self.norm(record_id, step))
            chosen.extend(rest[:extra])
        return chosen

    def pop(self, record_id: str, step: int | None = None
            ) -> tuple[Tensor | None, list[Tensor | None]]:
        if step is not None:
            self._bring_to(record_id, step)
        state = self.state.pop(record_id)
        payloads = self.payload.pop(record_id)
        self.updated.pop(record_id)
        self.norm_sq.pop(record_id)
        return (state if state.numel() else None), payloads

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'capacity': self.capacity,
                'ids': list(self.updated), 'updated': [self.updated[i] for i in self.updated],
                'norm_sq': [self.norm_sq[i] for i in self.updated],
                'state': [self.state[i] for i in self.updated],
                'payload': [self.payload[i] for i in self.updated]}

    def load_state_dict(self, state: dict) -> None:
        if state['decay'] != self.decay:
            raise ValueError('Record-gradient decay changed')
        self.state = dict(zip(state['ids'], state['state'], strict=True))
        self.payload = dict(zip(state['ids'], state['payload'], strict=True))
        self.updated = dict(zip(state['ids'], state['updated'], strict=True))
        self.norm_sq = dict(zip(state['ids'], state['norm_sq'], strict=True))


class KeyTable:
    """Trainable unit keys per record and space, in index order, updated sparsely.

    ``sphere_adam`` (default): gradients are projected onto each row's tangent
    plane (only movement along the sphere changes cosine scores), the first
    moment is kept per row and coordinate, and the second moment is one running
    scalar per space (mean squared tangent gradient of touched rows). A common
    scale keeps relative gradient strength: a faintly pulled row moves less than
    a strongly pulled one, and each row keeps its gradient direction.
    ``optimizer='adam'`` is per-coordinate Adam, which moves every touched row
    by about the learning rate regardless of its gradient.
    """

    def __init__(self, record_ids: Sequence[str], keys: Sequence[Tensor], *,
                 learning_rate: float, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, optimizer: str = 'sphere_adam',
                 max_step: float = 0.05) -> None:
        if optimizer not in {'sphere_adam', 'adam'}:
            raise ValueError(f'Unknown key-table optimizer: {optimizer}')
        self.optimizer = optimizer
        self.ids = [str(record_id) for record_id in record_ids]
        if (self.ids != sorted(self.ids) or len(set(self.ids)) != len(self.ids)
                or any(tuple(k.shape[:1]) != (len(self.ids),) for k in keys)):
            raise ValueError('Key table needs sorted unique IDs and one key per record')
        self.position = {record_id: i for i, record_id in enumerate(self.ids)}
        self.keys = [F.normalize(k.float(), dim=-1) for k in keys]
        self.exp_avg = [torch.zeros_like(k) for k in self.keys]
        self.exp_avg_sq = [torch.zeros_like(k) if optimizer == 'adam'
                           else k.new_zeros(()) for k in self.keys]
        self.space_steps = [0] * len(self.keys)
        self.counts = torch.zeros(len(self.keys), len(self.ids), dtype=torch.long,
                                  device=self.keys[0].device)
        self.updated = torch.full((len(self.ids),), -1, dtype=torch.long)
        self.learning_rate, self.betas, self.eps = learning_rate, betas, eps
        # Trust region: no row moves more than about ``max_step`` radians per step.
        self.max_step = max_step
        self.leaves: dict[tuple[int, str], Tensor] = {}
        # Rows changed since the last index sync; None forces a full sync.
        self.dirty: list[set[int]] | None = None

    def ids_digest(self) -> str:
        return hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()

    def _mark(self, space: int, positions: Tensor) -> None:
        if self.dirty is not None:
            self.dirty[space].update(positions.tolist())

    def begin_step(self) -> None:
        self.leaves = {}

    def rows(self, space: int, record_ids: Sequence[str], device=None) -> Tensor:
        """Differentiable rows; one leaf per record and space for the whole step."""
        missing = [r for r in dict.fromkeys(record_ids) if (space, r) not in self.leaves]
        if missing:
            positions = torch.tensor([self.position[r] for r in missing],
                                     device=self.keys[space].device)
            for record_id, row in zip(missing, self.keys[space][positions], strict=True):
                self.leaves[(space, record_id)] = row.detach().clone().requires_grad_(True)
        rows = torch.stack([self.leaves[(space, r)] for r in record_ids])
        return rows if device is None else rows.to(device)

    @torch.no_grad()
    def step(self, step: int) -> dict[str, float | int]:
        """Adam on touched rows only, then renormalize; report how far rows moved."""
        beta1, beta2 = self.betas
        moved, touched = [], set()
        for space in range(len(self.keys)):
            items = [(r, leaf.grad) for (s, r), leaf in self.leaves.items()
                     if s == space and leaf.grad is not None]
            if not items:
                continue
            positions = torch.tensor([self.position[r] for r, _ in items],
                                     device=self.keys[space].device)
            grads = torch.stack([g for _, g in items]).float().to(self.keys[space].device)
            old = self.keys[space][positions]
            if self.optimizer == 'sphere_adam':
                grads = grads - (grads * old).sum(-1, keepdim=True) * old
            self.counts[space, positions] += 1
            count = self.counts[space, positions].float()[:, None]
            m = self.exp_avg[space][positions].mul_(beta1).add_(grads, alpha=1 - beta1)
            if self.optimizer == 'adam':
                v = self.exp_avg_sq[space][positions].mul_(beta2).addcmul_(
                    grads, grads, value=1 - beta2)
                self.exp_avg_sq[space][positions] = v
                v = v / (1 - beta2 ** count)
            else:
                self.space_steps[space] += 1
                self.exp_avg_sq[space] = (beta2 * self.exp_avg_sq[space]
                                          + (1 - beta2) * (grads * grads).mean())
                v = self.exp_avg_sq[space] / (1 - beta2 ** self.space_steps[space])
            self.exp_avg[space][positions] = m
            update = (m / (1 - beta1 ** count)) / (v.sqrt() + self.eps)
            step_vector = self.learning_rate * update
            norms = step_vector.norm(dim=-1, keepdim=True)
            step_vector = step_vector * (self.max_step / norms.clamp_min(self.max_step))
            new = F.normalize(old - step_vector, dim=-1)
            self.keys[space][positions] = new
            self._mark(space, positions)
            # Angle from the chord, exact for small steps where arccos of a float32
            # cosine rounds to zero.
            moved.append(2 * torch.arcsin(((new - old).norm(dim=-1) / 2).clamp(max=1)))
            touched.update(r for r, _ in items)
        if touched:
            self.updated[torch.tensor([self.position[r] for r in touched])] = step
        self.leaves = {}
        if not moved:
            return {'table_rows': 0}
        degrees = torch.rad2deg(torch.cat(moved))
        moved = torch.cos(torch.deg2rad(degrees))
        quantiles = degrees.quantile(torch.tensor([0.5, 0.9, 0.99], device=degrees.device))
        return {'table_rows': int(moved.numel()), 'table_records': len(touched),
                'table_step_cosine_mean': float(moved.mean()),
                'table_step_cosine_min': float(moved.min()),
                'table_step_degrees_p50': float(quantiles[0]),
                'table_step_degrees_p90': float(quantiles[1]),
                'table_step_degrees_p99': float(quantiles[2]),
                'table_step_degrees_max': float(degrees.max())}

    @torch.no_grad()
    def pull(self, space: int, record_ids: Sequence[str], targets: Tensor,
             weight: float) -> None:
        """Commitment: move rows a fixed fraction toward the decoder's prediction.

        An interpolation rather than a gradient, so its size does not depend on
        the adaptive step of rows that retrieval never touches.
        """
        if not weight:
            return
        positions = torch.tensor([self.position[r] for r in record_ids],
                                 device=self.keys[space].device)
        rows = self.keys[space][positions]
        self.keys[space][positions] = F.normalize(
            rows + weight * (F.normalize(targets.float(), dim=-1).to(rows.device) - rows),
            dim=-1)
        self._mark(space, positions)

    @torch.no_grad()
    def sync_index(self, index) -> int:
        """Write rows changed since the last sync into the index; return the count.

        The first sync (and any after loading state) writes every row and rebuilds
        device mirrors; later syncs patch only changed rows in place.
        """
        if self.dirty is None:
            for space, array in enumerate(index.spaces.values()):
                if array.ids.tolist() != self.ids:
                    raise ValueError('Key table and index record order differ')
                array.keys[:] = self.keys[space].float().cpu().numpy()
            index.invalidate()
            self.dirty = [set() for _ in self.keys]
            return len(self.ids) * len(self.keys)
        written = 0
        for space, name in enumerate(index.spaces):
            if len(index.spaces[name].ids) != len(self.ids):
                raise ValueError('Key table and index record order differ')
            if not self.dirty[space]:
                continue
            positions = torch.tensor(sorted(self.dirty[space]), device=self.keys[space].device)
            index.patch_keys(name, positions.cpu().numpy(),
                             self.keys[space][positions].float().cpu().numpy())
            written += len(positions)
            self.dirty[space] = set()
        return written

    def state_dict(self) -> dict:
        return {'ids_sha256': self.ids_digest(), 'optimizer': self.optimizer,
                'keys': [k.cpu() for k in self.keys],
                'exp_avg': [m.cpu() for m in self.exp_avg],
                'exp_avg_sq': [v.cpu() for v in self.exp_avg_sq],
                'counts': self.counts.cpu(), 'updated': self.updated.clone(),
                'space_steps': list(self.space_steps)}

    def load_state_dict(self, state: dict) -> None:
        if state['ids_sha256'] != self.ids_digest():
            raise ValueError('Key table belongs to another bank')
        if state.get('optimizer', 'adam') != self.optimizer:
            raise ValueError('Key table optimizer state differs')
        device = self.keys[0].device
        self.keys = [k.to(device) for k in state['keys']]
        self.exp_avg = [m.to(device) for m in state['exp_avg']]
        self.exp_avg_sq = [v.to(device) for v in state['exp_avg_sq']]
        self.counts = state['counts'].to(device)
        self.updated = state['updated'].clone()
        self.space_steps = list(state.get('space_steps', [0] * len(self.keys)))
        self.dirty = None

    def adopt(self, state: dict, parent_ids: Sequence[str]) -> int:
        """Warm-start shared rows from a parent table over another bank.

        ``parent_ids`` is the parent bank's record order, checked against the
        state's digest. Keys, moments and touch counts of records present in both
        banks are copied; the rest keep their published keys. Returns the count.
        """
        parent_ids = [str(record_id) for record_id in parent_ids]
        if hashlib.sha256('\n'.join(parent_ids).encode()).hexdigest() != state['ids_sha256']:
            raise ValueError('Parent record order does not match the parent key table')
        if state.get('optimizer', 'adam') != self.optimizer or len(state['keys']) != len(self.keys):
            raise ValueError('Parent key table optimizer or spaces differ')
        pairs = [(i, self.position[record_id]) for i, record_id in enumerate(parent_ids)
                 if record_id in self.position]
        if not pairs:
            return 0
        device = self.keys[0].device
        source = torch.tensor([i for i, _ in pairs], dtype=torch.long)
        target = torch.tensor([j for _, j in pairs], dtype=torch.long, device=device)
        for space in range(len(self.keys)):
            if state['keys'][space].shape[1] != self.keys[space].shape[1]:
                raise ValueError('Parent key width differs')
            self.keys[space][target] = state['keys'][space][source].to(device)
            self.exp_avg[space][target] = state['exp_avg'][space][source].to(device)
            if self.optimizer == 'adam':
                self.exp_avg_sq[space][target] = state['exp_avg_sq'][space][source].to(device)
        if self.optimizer != 'adam':
            # One second moment per space: keep the parent's scale.
            self.exp_avg_sq = [v.to(device).clone() for v in state['exp_avg_sq']]
        self.space_steps = list(state.get('space_steps', self.space_steps))
        self.counts[:, target.to(self.counts.device)] = state['counts'][:, source].to(
            self.counts.device)
        self.updated[target.cpu()] = state['updated'][source]
        self.dirty = None
        return len(pairs)


class GradientSink:
    """Leaf tensors for one step's consumer reads, harvested after backward."""

    def __init__(self, agent, cache: KeyStateCache, table: KeyTable | None = None) -> None:
        self.agent, self.cache, self.table = agent, cache, table
        self.state_leaves: dict[str, Tensor] = {}
        self.payload_leaves: dict[tuple[str, int], list[Tensor]] = {}
        self.neighborhood: list[str] = []

    def candidate_keys(self, space: int, record_ids: Sequence[str]) -> Tensor:
        """Live keys for candidates from cached states; exact head and state gradients."""
        if self.table is not None:
            return self.table.rows(space, record_ids, self.agent.device)
        missing = [record_id for record_id in dict.fromkeys(record_ids)
                   if record_id not in self.state_leaves]
        if missing:
            states = self.cache.states[self.cache.positions(missing).to(
                self.cache.states.device)].to(self.agent.device)
            for record_id, state in zip(missing, states, strict=True):
                self.state_leaves[record_id] = state.detach().clone().requires_grad_(True)
        states = torch.stack([self.state_leaves[record_id] for record_id in record_ids])
        head = self.agent.writer_key_heads[space]
        return F.normalize(self.agent._fp32_head(head, states, 'writer'), dim=-1)

    def payload(self, space: int, record_id: str, value: Tensor) -> Tensor:
        leaf = value.detach().float().to(self.agent.device).requires_grad_(True)
        self.payload_leaves.setdefault((record_id, space), []).append(leaf)
        return leaf

    def near(self, record_ids: Iterable[str]) -> None:
        self.neighborhood.extend(record_ids)

    def harvest(self, gradients: RecordGradients, step: int, spaces: int) -> int:
        """Move this step's cotangents to the accumulator in a few batched transfers."""
        state_ids = [r for r, leaf in self.state_leaves.items() if leaf.grad is not None]
        state_grads = (dict(zip(state_ids, torch.stack(
            [self.state_leaves[r].grad for r in state_ids]).float().cpu(), strict=True))
                       if state_ids else {})
        payload_grads: dict[tuple[str, int], Tensor] = {}
        for space in range(spaces):
            keys = [key for key, leaves in self.payload_leaves.items()
                    if key[1] == space and any(leaf.grad is not None for leaf in leaves)]
            if not keys:
                continue
            summed = torch.stack([torch.stack([leaf.grad for leaf in self.payload_leaves[key]
                                               if leaf.grad is not None]).sum(0)
                                  for key in keys])
            payload_grads.update(zip(keys, summed.float().cpu(), strict=True))
        records = set(state_grads) | {record_id for record_id, _ in payload_grads}
        for record_id in records:
            gradients.add(step, record_id, state_grads.get(record_id),
                          [payload_grads.get((record_id, space)) for space in range(spaces)])
        return len(records)


def writer_backward(agent, writer_inputs: Callable[[str], Tensor], record_ids: Sequence[str],
                    cotangents: Sequence[tuple[Tensor | None, list[Tensor | None]]], *,
                    batch_size: int = 16, scale: float = 1.0) -> None:
    """Run the writer for these records and push accumulated cotangents into it."""
    spaces = len(agent.config.memory.payload_dims)
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    for start in range(0, len(record_ids), batch_size):
        batch = record_ids[start:start + batch_size]
        cots = cotangents[start:start + batch_size]
        outputs = agent.produce_batch([writer_inputs(record_id) for record_id in batch],
                                      with_key_state=True)
        targets, grads = [], []
        key_state = outputs[-1]
        state_rows = [index for index, (state, _) in enumerate(cots) if state is not None]
        if state_rows:
            rows = torch.tensor(state_rows, device=agent.device)
            targets.append(key_state.float()[rows])
            grads.append(torch.stack([cots[i][0] for i in state_rows]).to(agent.device) * scale)
        for space in range(spaces):
            rows_s = [index for index, (_, payloads) in enumerate(cots)
                      if payloads[space] is not None]
            if not rows_s:
                continue
            rows = torch.tensor(rows_s, device=agent.device)
            # The stored payload passes through storage precision, as consumer reads did.
            payload = outputs[2 * space + 1].to(dtype).float()[rows]
            targets.append(payload)
            grads.append(torch.stack([cots[i][1][space].float() for i in rows_s])
                         .to(agent.device) * scale)
        if targets:
            torch.autograd.backward(targets, grads)



@torch.no_grad()
def refresh_records(agent, writer_inputs: Callable[[str], Tensor], record_ids: Sequence[str],
                    bank, cache: KeyStateCache, step: int, *, batch_size: int = 32,
                    ) -> dict[str, float | int]:
    """Re-encode records with the current writer, forward only, and publish them.

    Updates cached key-slot states and versions, commits keys and payloads to the
    mutable bank journal, and reports drift: the cosine between each record's
    previous stored s0 key (under the current heads) and its new one.
    """
    from .store import StoredRecord
    if not record_ids:
        return {'refreshed': 0}
    spaces = len(agent.config.memory.payload_dims)
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    ordered = sorted(record_ids, key=lambda record_id: writer_inputs(record_id).shape[1])
    positions = cache.positions(ordered)
    old_keys = cache.keys(agent, positions)[0].float()
    records, states = [], []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        outputs = agent.produce_batch([writer_inputs(record_id) for record_id in batch],
                                      with_key_state=True)
        states.append(outputs[-1].float())
        for row, record_id in enumerate(batch):
            for space in range(spaces):
                records.append(StoredRecord(
                    record_id, outputs[2 * space][row].detach().float(),
                    outputs[2 * space + 1][row].to(dtype).detach(),
                    namespace=bank.index.namespace, space=f's{space}',
                    generation=bank.index.generation))
    states = torch.cat(states)
    new_keys = agent.writer_space_keys(states)[0].float()
    cosine = F.cosine_similarity(old_keys, new_keys, dim=-1).cpu()
    ages = cache.update(ordered, states, step)
    views = bank.update(records, optimizer_step=step)
    return {'refreshed': len(ordered), 'refreshed_views': views,
            'drift_cosine_mean': float(cosine.mean()), 'drift_cosine_min': float(cosine.min()),
            'refreshed_age_max': int(ages.max()), 'refreshed_age_mean': float(ages.float().mean())}


def stored_payloads(bank, space: int, record_ids: Sequence[str]) -> list[Tensor]:
    """Currently published payloads of one space, read through the mutable bank."""
    from .store import ReadPlan, Selection
    array = bank.index.spaces[f's{space}']
    plans = []
    for record_id in record_ids:
        position = int(np.flatnonzero(array.ids == record_id)[0])
        plans.append(ReadPlan(bank.index.namespace, f's{space}', bank.index.generation,
                              str(array.domains[position]), int(array.times[position]) + 1,
                              (Selection(record_id, 0.0),)))
    return [values[0] for values in bank.fetch_many(plans)]


def writer_pass(agent, writer_inputs: Callable[[str], Tensor], record_ids: Sequence[str],
                cotangents: Sequence[tuple[Tensor | None, list[Tensor | None]]] | None,
                table: KeyTable, bank, cache: KeyStateCache, step: int, *,
                prediction_weight: float, commitment_weight: float, batch_size: int = 16,
                drift_sample: int = 0, backward: bool = True,
                checkpointing: bool = True) -> dict[str, float | int]:
    """Encode records with gradient: key prediction, payload cotangents, publish.

    For each record the decoder's predicted keys regress onto the (fixed) table
    rows, and the rows move a small fraction toward the prediction (commitment).
    Accumulated payload cotangents, when given, flow into the writer in the same
    backward. With ``backward=False`` (a refresh without cotangents) the pass is
    forward-only: it republishes payloads and applies the commitment pull, and
    reports key-prediction agreement without training it. ``checkpointing=False``
    retains writer activations instead of recomputing them in backward (short
    source documents fit easily). The encoded payloads are then published with the table keys, so a
    flush and a refresh share one forward. Parameter gradients accumulate for the
    caller's optimizer step; table-row gradients for ``table.step``.
    """
    from .store import StoredRecord
    if not record_ids:
        return {'encoded': 0}
    if not backward and cotangents is not None:
        raise ValueError('A forward-only writer pass cannot apply cotangents')
    spaces = len(agent.config.memory.payload_dims)
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    from .bank_replay import _checkpointing
    grad_mode = torch.enable_grad() if backward else torch.no_grad()
    order = sorted(range(len(record_ids)),
                   key=lambda i: writer_inputs(record_ids[i]).shape[1])
    ordered = [record_ids[i] for i in order]
    cots = None if cotangents is None else [cotangents[i] for i in order]
    timer = time.perf_counter()
    drift_ids = set(ordered[:: max(1, len(ordered) // drift_sample)][:drift_sample]
                    if drift_sample else ())
    old_values = (dict(zip(sorted(drift_ids), stored_payloads(bank, 0, sorted(drift_ids)),
                           strict=True)) if drift_ids else {})
    seconds = {'drift_read': time.perf_counter() - timer}
    timer = time.perf_counter()
    scale = 1 / len(ordered)
    records, states, cosines, drift = [], [], [], []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        with grad_mode, _checkpointing(agent, enabled=checkpointing):
            outputs = agent.produce_batch([writer_inputs(record_id) for record_id in batch],
                                          with_key_state=True)
            key_state = outputs[-1]
            predicted = agent.writer_space_keys(key_state)
        loss = key_state.new_zeros((), dtype=torch.float32)
        published_keys = []
        for space in range(spaces):
            positions = torch.tensor([table.position[r] for r in batch],
                                     device=table.keys[space].device)
            rows = table.keys[space][positions].to(agent.device)
            pred = predicted[space].float()
            cosine = F.cosine_similarity(pred, rows, dim=-1)
            loss = loss + scale / spaces * prediction_weight * (1 - cosine).sum()
            cosines.append(cosine.detach())
            table.pull(space, batch, pred.detach(), commitment_weight)
            published_keys.append(table.keys[space][positions].detach().to(agent.device))
        targets, grads = [loss], [torch.ones_like(loss)]
        if cots is not None:
            batch_cots = cots[start:start + batch_size]
            for space in range(spaces):
                rows_s = [i for i, (_, payloads) in enumerate(batch_cots)
                          if payloads[space] is not None]
                if not rows_s:
                    continue
                index = torch.tensor(rows_s, device=agent.device)
                targets.append(outputs[2 * space + 1].to(dtype).float()[index])
                grads.append(torch.stack([batch_cots[i][1][space].float() for i in rows_s])
                             .to(agent.device))
        if backward:
            torch.autograd.backward(targets, grads)
        states.append(key_state.detach().float())
        # One device-to-host copy per space and batch; per-record copies each wait
        # for the device stream, which dominated publishing on a shared GPU.
        host_keys = [keys.float().cpu() for keys in published_keys]
        host_payloads = [outputs[2 * space + 1].detach().to(dtype).cpu()
                         for space in range(spaces)]
        for row, record_id in enumerate(batch):
            if record_id in old_values:
                drift.append(F.cosine_similarity(
                    host_payloads[0][row].float().flatten(),
                    old_values[record_id].float().flatten(), dim=0))
            for space in range(spaces):
                records.append(StoredRecord(
                    record_id, host_keys[space][row], host_payloads[space][row],
                    namespace=bank.index.namespace, space=f's{space}',
                    generation=bank.index.generation))
    if torch.cuda.is_available() and agent.device.type == 'cuda':
        torch.cuda.synchronize(agent.device)
    seconds['encode'] = time.perf_counter() - timer
    timer = time.perf_counter()
    ages = cache.update(ordered, torch.cat(states), step)
    views = bank.update(records, optimizer_step=step)
    seconds['publish'] = time.perf_counter() - timer
    cosines = torch.cat(cosines)
    report = {'encoded': len(ordered), 'views': views,
              'key_prediction_cosine_mean': float(cosines.mean()),
              'key_prediction_cosine_min': float(cosines.min()),
              'age_max': int(ages.max()), 'age_mean': float(ages.float().mean()),
              **{f'{name}_seconds': value for name, value in seconds.items()}}
    if drift:
        drift = torch.stack(drift)
        report.update(value_drift_cosine_mean=float(drift.mean()),
                      value_drift_cosine_min=float(drift.min()))
    return report
