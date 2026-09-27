"""Optional scalar-only W&B logging; JSONL/checkpoints remain recovery authority."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys
import uuid

from .operations import atomic_json


def run_identity(output):
    path = Path(output) / 'run-identity.json'
    if not path.exists():
        atomic_json(path, {'id': uuid.uuid4().hex, 'format': 1})
    return json.loads(path.read_text())['id']


class Tracking:
    def __init__(self, config, output):
        self.config, self.output, self.run = config, Path(output), None
        self.attempt = uuid.uuid4().hex[:12]
        self.errors = []
        self.logging_disabled = False

    def __enter__(self):
        identity = run_identity(self.output)
        t = self.config.train
        if t.wandb_mode != 'disabled':
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError('W&B requested: install schnitz[tracking] or disable wandb_mode') from exc
            directory = self.output / 'tracking'
            directory.mkdir(exist_ok=True)
            self.run = wandb.init(project=t.wandb_project, entity=t.wandb_entity,
                group=t.wandb_group, name=f'{self.output.parent.name}/{self.output.name}', id=identity, resume='allow',
                mode=t.wandb_mode, dir=str(directory), config=asdict(self.config), save_code=False,
                allow_val_change=True, settings={'console': 'off', 'disable_git': True})
            self.run.define_metric('optimizer_step')
            self.run.define_metric('*', step_metric='optimizer_step')
        return self

    def log(self, row):
        if self.run is not None and not self.logging_disabled:
            def scalars(values, prefix=''):
                result = {}
                for name, value in values.items():
                    key = prefix + name
                    if isinstance(value, (int, float)):
                        result[key] = value
                    elif isinstance(value, dict):
                        result.update(scalars(value, key + '/'))
                return result
            scalar = scalars(row)
            scalar.update(optimizer_step=row['step'], resume_attempt=self.attempt)
            try:
                self.run.log(scalar)
            except Exception as exc:
                self.logging_disabled = True
                self._record_failure('log', exc, row['step'])

    def _record_failure(self, operation, error, step=None):
        # SDK exception text can include private remote diagnostics. Record only
        # the error type; SDK-owned debug files remain in the external run directory.
        event = {'operation': operation, 'error_type': type(error).__name__,
                 'optimizer_step': step}
        self.errors.append(event)
        record = {'attempt': self.attempt, 'errors': self.errors,
                  'notice': 'W&B telemetry failed; JSONL and checkpoints remain authoritative.'}
        try:
            atomic_json(self.output/'tracking'/f'failure-{self.attempt}.json', record)
        except OSError:
            # Failure to record optional telemetry must not lose optimizer work.
            pass
        try:
            print(json.dumps({'event': 'tracking_failure', **event,
                              'attempt': self.attempt, 'logging_disabled': self.logging_disabled}),
                  file=sys.stderr, flush=True)
        except OSError:
            pass

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            try:
                self.run.finish(exit_code=1 if exc_type or self.errors else 0)
            except Exception as error:
                self._record_failure('finish', error)
