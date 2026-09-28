"""Training-loop plumbing shared by every stage: the output directory (config,
metrics log, atomic state files), the warmup optimizer, and metric windows."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch


class Run:
    """An output directory: ``config.json``, ``metrics.jsonl`` (appended across
    resumes) and state files saved atomically (``.pending`` then rename)."""

    def __init__(self, output: Path, config: dict | None = None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (self.output / 'config.json').write_text(
                json.dumps(config, indent=2, default=str) + '\n')
        self._metrics = (self.output / 'metrics.jsonl').open('a', encoding='utf-8')
        self.started = time.time()

    def write_config(self, config: dict) -> None:
        (self.output / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')

    def log(self, record: dict) -> None:
        line = json.dumps(record)
        self._metrics.write(line + '\n')
        self._metrics.flush()
        print(line, flush=True)

    def elapsed(self) -> int:
        return round(time.time() - self.started)

    def path(self, name: str) -> Path:
        return self.output / name

    def save(self, name: str, state: dict) -> None:
        path = self.path(name)
        torch.save(state, path.with_suffix('.pending'))
        path.with_suffix('.pending').replace(path)

    def load(self, name: str, device=None) -> dict | None:
        path = self.path(name)
        return torch.load(path, map_location=device) if path.exists() else None


def warmup_optimizer(params, warmup: int, start: int = 0, lr: float | None = None,
                     weight_decay: float = 0.01):
    """AdamW with a linear warmup over ``warmup`` steps, ``start`` steps already done."""
    optimizer = (torch.optim.AdamW(params, weight_decay=weight_decay) if lr is None
                 else torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + start + 1) / warmup))
    return optimizer, schedule


class Window:
    """Running means of per-step results between log lines, per key (a key absent in
    a step does not count that step)."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, result: dict, prefix: str = '') -> None:
        for key, value in result.items():
            name = f'{prefix}/{key}' if prefix else key
            self.sums[name] = self.sums.get(name, 0.0) + float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def means(self, digits: int = 4) -> dict[str, float]:
        out = {k: round(v / self.counts[k], digits) for k, v in self.sums.items()}
        self.sums, self.counts = {}, {}
        return out
