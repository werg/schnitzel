"""Step-consistent local checkpoints, including the intentionally stale memory bank.

CURRENT is the only commit point. All files live in a new immutable directory;
model, optimizer/RNG, data identity, and SQLite snapshot are published together.
Root model/state links are convenience aliases, never the recovery authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import signal
import sqlite3
import threading
import time
import uuid

def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name == 'nt':
        return  # Windows does not expose POSIX directory fsync.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def reconcile_metrics(run: Path, step: int) -> None:
    """Keep only complete metric rows covered by the authoritative saved state."""
    log = Path(run) / 'metrics.jsonl'
    if log.exists():
        rows = []
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['step'] <= step:
                rows.append(json.dumps(row))
        _atomic_text(log, ''.join(row + '\n' for row in rows))


def _checkpoint_event(event):
    # A disconnected console must not prevent an emergency recovery save.
    try:
        print(json.dumps(event), flush=True)
    except OSError:
        pass


def resolve_checkpoint(run: str | Path, *, verify: bool = False) -> Path:
    run = Path(run)
    if not (run / "CURRENT").exists():
        if (run / "model.safetensors").exists():
            if not (run / 'manifest.json').exists():
                return run  # Read compatibility with the unmanifested 0.1 handoff.
            path = run  # Direct immutable checkpoint paths retain full verification.
        else:
            raise FileNotFoundError("No committed checkpoint")
    else:
        name = (run / "CURRENT").read_text().strip()
        if not name.startswith("step-") or Path(name).name != name:
            raise ValueError("Invalid checkpoint pointer")
        path = run / "checkpoints" / name
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    for filename, expected in manifest["sha256"].items():
        if Path(filename).name != filename or not (path / filename).is_file():
            raise ValueError("Incomplete checkpoint")
        if verify and _digest(path / filename) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
    return path


def save_checkpoint(agent, optimizer, run: Path, step: int, rng: random.Random,
                    cache, fingerprint: str, *, keep: int = 2, archiver=None,
                    accumulation: dict | None = None) -> Path:
    import torch
    from safetensors.torch import save_model
    if step < 0 or keep < 1:
        raise ValueError("Invalid checkpoint step/retention")
    from .archiving import ensure_free
    weights_bytes = sum(p.numel() * p.element_size() for p in agent.parameters())
    optimizer_bytes = 2 * sum(p.numel() * p.element_size() for p in agent.parameters() if p.requires_grad)
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in agent.parameters()
                         if p.grad is not None) if accumulation else 0
    bank_state = cache.mutable_bank_state()
    cache_copy_bytes = (0 if bank_state is not None
                        else cache.sizes()['physical_sqlite_bytes'])
    ensure_free(root, weights_bytes + optimizer_bytes + gradient_bytes +
                cache_copy_bytes + 1024 ** 2, agent.config.train.min_free_disk_bytes)
    token = uuid.uuid4().hex[:12]
    pending = root / (".pending-" + token)
    pending.mkdir()
    final = root / f"step-{step:09d}-{token}"
    started = time.perf_counter()
    _checkpoint_event({'event': 'checkpoint_start', 'step': step, 'path': str(final),
                       'accumulated_microbatches': (accumulation or {}).get('microbatches', 0)})
    try:
        save_model(agent, str(pending / "model.safetensors"))
        state = {"optimizer": optimizer.state_dict(), "step": step,
                 "optimizer_type": agent.config.train.optimizer,
                 "optimizer_parameter_names": getattr(optimizer, '_sdkb_parameter_names', None),
                 "python_rng": rng.getstate(), "global_python_rng": random.getstate(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        if accumulation:
            state['accumulation'] = accumulation
            state['gradients'] = {n: p.grad.detach().cpu() for n, p in agent.named_parameters()
                                  if p.grad is not None}
        torch.save(state, pending / "training_state.pt")
        if bank_state is None:
            # Legacy episode caches remain small and self-contained.
            with cache.connect() as source, sqlite3.connect(
                    pending / "training_cache.sqlite") as dest:
                source.backup(dest)
        else:
            (pending / 'bank-state.json').write_text(
                json.dumps(bank_state, indent=2, sort_keys=True) + '\n')
        (pending / "config.json").write_text(json.dumps(asdict(agent.config), indent=2) + "\n")
        if (run / 'run-identity.json').exists():
            shutil.copyfile(run / 'run-identity.json', pending / 'run-identity.json')
        files = [p for p in pending.iterdir() if p.is_file()]
        for path in files:
            _fsync(path)
        manifest = {"format": 1, "step": step, "dataset_sha256": fingerprint,
                    "resolved_model_revision": agent.resolved_revision,
                    "sha256": {p.name: _digest(p) for p in files}}
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _fsync(pending / "manifest.json")
        _fsync_dir(pending)
        os.replace(pending, final)
        _fsync_dir(root)
        if bank_state is not None:
            cache.pin_mutable_bank(final.name, bank_state['cursor'])
        _atomic_text(run / "CURRENT", final.name + "\n")
        _checkpoint_event({'event': 'checkpoint_committed', 'step': step, 'path': str(final),
                           'elapsed_seconds': time.perf_counter() - started,
                           'bytes': sum(p.stat().st_size for p in final.iterdir() if p.is_file())})
    except BaseException:
        if bank_state is not None:
            cache.unpin_mutable_bank(final.name)
        shutil.rmtree(pending, ignore_errors=True)
        raise
    # Convenient legacy filenames; readers in this version always resolve CURRENT.
    for name in ("model.safetensors", "training_state.pt"):
        if os.name == 'nt':
            continue  # CURRENT is authoritative; Windows symlinks may require privilege.
        alias = run / (name + ".link-tmp")
        alias.unlink(missing_ok=True)
        alias.symlink_to(Path("checkpoints") / final.name / name)
        os.replace(alias, run / name)
    committed = sorted((p for p in root.glob("step-*") if p.is_dir()),
                       key=lambda p: (json.loads((p / "manifest.json").read_text())["step"],
                                      p.stat().st_mtime_ns), reverse=True)
    if archiver is not None:
        archiver.submit(final)
    protected = archiver.protected_names() if archiver is not None else set()
    for old in committed[keep:]:
        if old != final and old.name not in protected:
            if bank_state is not None:
                cache.unpin_mutable_bank(old.name)
            shutil.rmtree(old)
    if bank_state is not None:
        try:
            cache.gc_mutable_bank(cache.mutable_bank_gc_ceiling())
        except (OSError, sqlite3.Error) as error:
            _checkpoint_event({'event': 'checkpoint_bank_gc_deferred', 'step': step,
                               'error': str(error)})
    return final


def restore_checkpoint(agent, optimizer, run: Path, rng: random.Random,
                       fingerprint: str, *, progress: dict | None = None,
                       discard_accumulation: bool = False) -> int:
    import torch
    from safetensors.torch import load_model
    run = Path(run)
    path = resolve_checkpoint(run, verify=True)
    if not (path / 'manifest.json').is_file():
        raise ValueError('Checkpoint lacks complete resume state and manifest; '
                         'use its original checkout or an explicit warm-start')
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["dataset_sha256"] != fingerprint:
        raise ValueError("Episode contents changed since checkpoint")
    if manifest["resolved_model_revision"] != agent.resolved_revision:
        raise ValueError("Base model revision changed; use the pinned original revision")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    required = {'optimizer', 'step', 'optimizer_type', 'optimizer_parameter_names',
                'python_rng', 'global_python_rng', 'torch_rng', 'cuda_rng'}
    if state.get('accumulation') and not discard_accumulation:
        required.add('gradients')
    if required - state.keys() or state.get('optimizer_parameter_names') is None:
        raise ValueError('Checkpoint lacks complete resume state; '
                         'use its original checkout or an explicit warm-start')
    if state['optimizer_type'] != agent.config.train.optimizer:
        raise ValueError('Optimizer changed; use an explicit warm-start with a new run identity')
    if state['optimizer_parameter_names'] != getattr(optimizer, '_sdkb_parameter_names', None):
        raise ValueError('Optimizer parameter names/order changed; refusing mismatched momentum')
    if state.get('accumulation') and not discard_accumulation:
        accumulation_keys = {'microbatches', 'loops', 'totals', 'anchor_total'}
        if agent.config.train.oracle_alignment_weight:
            accumulation_keys.add('alignment_total')
        if agent.config.train.oracle_distillation_weight:
            accumulation_keys.add('distillation_total')
        if accumulation_keys - state['accumulation'].keys():
            raise ValueError('Checkpoint lacks complete accumulation state; use an explicit warm-start')
        if progress is None:
            raise ValueError('Checkpoint has an incomplete optimizer update; restore its accumulation state')
    load_model(agent, str(path / "model.safetensors"), device=agent.config.train.device)
    optimizer.load_state_dict(state["optimizer"])
    if state.get('accumulation') and not discard_accumulation:
        progress.update(state['accumulation'])
        parameters = dict(agent.named_parameters())
        for name, gradient in state['gradients'].items():
            if name not in parameters or not parameters[name].requires_grad:
                raise ValueError(f'Accumulated gradient has no trainable parameter: {name}')
            parameter = parameters[name]
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)
    rng.setstate(state["python_rng"])
    random.setstate(state["global_python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if path != run:
        from .store import DiskStore
        destination = run / "training_cache.sqlite"
        if (path / 'bank-state.json').is_file():
            if not destination.is_file():
                raise FileNotFoundError(
                    'Mutable-bank journal is missing; checkpoints store only its cursor')
            DiskStore(destination).restore_mutable_bank(
                json.loads((path / 'bank-state.json').read_text()))
        else:
            # Discard uncheckpointed legacy cache writes.
            source_cache = path / "training_cache.sqlite"
            source_digest = _digest(source_cache)
            migrated = None
            if destination.is_file():
                try:
                    with sqlite3.connect(destination) as db:
                        migrated = dict(db.execute('''SELECT key,value
                            FROM mutable_bank_meta WHERE key IN
                            ('legacy_source_sha256','legacy_checkpoint_cursor')'''))
                except sqlite3.Error:
                    migrated = None
            if (migrated is not None
                    and migrated.get('legacy_source_sha256') == source_digest
                    and 'legacy_checkpoint_cursor' in migrated):
                store = DiskStore(destination)
                target = int(migrated['legacy_checkpoint_cursor'])
                with store.connect() as db:
                    meta = dict(db.execute('SELECT key,value FROM mutable_bank_meta'))
                    commit = db.execute('''SELECT content_sha256,optimizer_step
                        FROM mutable_bank_commits WHERE cursor=?''', (target,)).fetchone()
                store.restore_mutable_bank({
                    'format': int(meta['format']), 'store_uuid': meta['store_uuid'],
                    'cursor': target, 'floor_cursor': int(meta['floor_cursor']),
                    'maintenance_position': 0,
                    'commit_sha256': commit[0], 'optimizer_step': commit[1],
                })
            else:
                for suffix in ("-wal", "-shm"):
                    Path(str(destination) + suffix).unlink(missing_ok=True)
                temp = run / "cache-restore.tmp"
                shutil.copyfile(source_cache, temp)
                os.replace(temp, destination)
                with sqlite3.connect(destination) as db:
                    db.execute('''CREATE TABLE IF NOT EXISTS mutable_bank_migration_source
                        (sha256 TEXT NOT NULL)''')
                    db.execute('DELETE FROM mutable_bank_migration_source')
                    db.execute('INSERT INTO mutable_bank_migration_source VALUES (?)',
                               (source_digest,))
        # Reconcile mutable run logs, never write into a directly loaded immutable set.
        reconcile_metrics(run, state['step'])
    return int(state["step"])


@contextmanager
def stop_on_signal():
    """Request a checkpoint at the next complete optimizer step, not mid-backward."""
    requested = {"signal": None}
    old = {}
    if threading.current_thread() is threading.main_thread():
        def handle(signum, _frame):
            requested["signal"] = signum
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, handle)
    try:
        yield requested
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)
