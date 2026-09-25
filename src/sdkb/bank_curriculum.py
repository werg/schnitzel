"""Grow the active training bank in nested stages once the key space is stable.

Each stage activates a subset of the bank: the gold records of a prefix of a
seeded trajectory order (about ``gold_share`` of the stage) plus a seeded
random fill of other records. Both prefixes only grow, so every stage contains
the previous one, and the final stage (size 0) is the full bank with every
trajectory. Training rows are drawn only from trajectories whose gold records
are all active. Search, exploration proposals and the rolling refresh are
restricted to the active subset; this is a curriculum, not an authorization
boundary.

A stage advances after ``min_steps`` once the key space is stable:
- the decoder reproduces its table keys (key-prediction cosine EMA at or above
  ``prediction_cosine``);
- published payloads barely move between encodes (value-drift cosine EMA at or
  above ``value_cosine``);
- unassisted retrieval has plateaued (the recall EMA changed by less than
  ``recall_plateau`` over the last ``window`` steps).
It advances regardless after ``max_steps``.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import random

from .training import EpisodeSampler


class BankCurriculum:
    def __init__(self, sizes: Sequence[int], row_required: Sequence[Sequence[str]],
                 record_ids: Sequence[str], *, seed: int, gold_share: float = 0.5,
                 min_steps: int = 300, max_steps: int = 3000, window: int = 100,
                 prediction_cosine: float = 0.98, value_cosine: float = 0.99,
                 recall_plateau: float = 0.01, decay: float = 0.98) -> None:
        if (not sizes or any(size < 0 for size in sizes) or sizes[-1] != 0
                or any(0 < b <= a for a, b in zip(sizes, sizes[1:]) if b)
                or not 0 < gold_share <= 1 or not 1 <= min_steps <= max_steps
                or window < 1 or not 0 < decay < 1):
            raise ValueError('Curriculum sizes must increase and end with 0 (full bank)')
        self.sizes, self.seed = tuple(sizes), seed
        self.min_steps, self.max_steps, self.window = min_steps, max_steps, window
        self.thresholds = (prediction_cosine, value_cosine, recall_plateau)
        self.decay = decay
        known = set(map(str, record_ids))
        rows = list(range(len(row_required)))
        random.Random(f'sdkb-curriculum-rows:{seed}').shuffle(rows)
        fill = sorted(known)
        random.Random(f'sdkb-curriculum-fill:{seed}').shuffle(fill)
        self.stages: list[tuple[frozenset[str] | None, tuple[int, ...]]] = []
        for size in self.sizes:
            if size == 0 or size >= len(known):
                self.stages.append((None, tuple(range(len(row_required)))))
                continue
            golds: set[str] = set()
            chosen = []
            for row in rows:
                required = set(map(str, row_required[row]))
                if not required <= known:
                    continue
                if chosen and len(golds | required) > gold_share * size:
                    break
                golds |= required
                chosen.append(row)
            active = set(golds)
            for record_id in fill:
                if len(active) >= size:
                    break
                active.add(record_id)
            self.stages.append((frozenset(active), tuple(sorted(chosen))))
        for (a, _), (b, _) in zip(self.stages, self.stages[1:]):
            if a is not None and b is not None and not a <= b:
                raise AssertionError('Curriculum stages must be nested')
        self.stage, self.stage_start = 0, 0
        self.ema: dict[str, float | None] = {'prediction': None, 'value': None, 'recall': None}
        self.recall_history: deque[float] = deque(maxlen=window + 1)
        self._sampler: tuple[int, EpisodeSampler] | None = None

    @property
    def active(self) -> frozenset[str] | None:
        return self.stages[self.stage][0]

    @property
    def rows(self) -> tuple[int, ...]:
        return self.stages[self.stage][1]

    def row(self, step: int, offset: int, batch_size: int) -> int:
        """Deterministic shuffled passes over the current stage's trajectories."""
        if self._sampler is None or self._sampler[0] != self.stage:
            self._sampler = (self.stage, EpisodeSampler(
                len(self.rows), seed=self.seed * 1000 + self.stage))
        position = (step - self.stage_start) * batch_size + offset
        return self.rows[self._sampler[1].index(position)]

    def _mix(self, name: str, value: float | None) -> None:
        if value is None:
            return
        old = self.ema[name]
        self.ema[name] = value if old is None else self.decay * old + (1 - self.decay) * value

    def observe(self, step: int, *, prediction_cosine: float | None,
                value_cosine: float | None, recall: float | None) -> bool:
        """Record one completed step; advance and return True when the stage is done."""
        self._mix('prediction', prediction_cosine)
        self._mix('value', value_cosine)
        self._mix('recall', recall)
        if self.ema['recall'] is not None:
            self.recall_history.append(self.ema['recall'])
        if self.stage == len(self.stages) - 1:
            return False
        elapsed = step - self.stage_start
        if elapsed >= self.max_steps or (elapsed >= self.min_steps and self.stable()):
            self.stage += 1
            self.stage_start = step
            self.recall_history.clear()
            return True
        return False

    def stable(self) -> bool:
        prediction, value, plateau = self.thresholds
        if None in self.ema.values() or len(self.recall_history) <= self.window:
            return False
        return (self.ema['prediction'] >= prediction and self.ema['value'] >= value
                and abs(self.recall_history[-1] - self.recall_history[0]) < plateau)

    def report(self) -> dict:
        active = self.active
        return {'curriculum_stage': self.stage, 'curriculum_stage_start': self.stage_start,
                'curriculum_active_records': None if active is None else len(active),
                'curriculum_rows': len(self.rows),
                **{f'curriculum_ema_{name}': value for name, value in self.ema.items()}}

    def state_dict(self) -> dict:
        return {'sizes': self.sizes, 'stage': self.stage, 'stage_start': self.stage_start,
                'ema': dict(self.ema), 'recall_history': list(self.recall_history)}

    def load_state_dict(self, state: dict) -> None:
        if tuple(state['sizes']) != self.sizes:
            raise ValueError('Curriculum stages changed')
        self.stage, self.stage_start = state['stage'], state['stage_start']
        self.ema = dict(state['ema'])
        self.recall_history = deque(state['recall_history'], maxlen=self.window + 1)
        self._sampler = None
