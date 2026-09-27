"""Bounded background checkpoint copies to optional external storage.

Active checkpoints stay on the run filesystem. A slow archive coalesces pending
copies to the newest checkpoint; it never creates an unbounded copy queue.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import threading
import time
import uuid

from .checkpoints import _atomic_text, _fsync_dir, resolve_checkpoint


def ensure_free(path, required_bytes=0, reserve_bytes=1024 ** 3):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < required_bytes + reserve_bytes:
        raise OSError(f'Insufficient free disk at {path}: {free} bytes; '
                      f'need {required_bytes} plus {reserve_bytes} reserve. Last committed checkpoint retained.')


def copy_file(source, destination, expected=None):
    """Periodic flush bounds dirty pages on systems with shared CPU/GPU memory."""
    digest, unsynced = hashlib.sha256(), 0
    with Path(source).open('rb') as src, Path(destination).open('wb') as dst:
        while chunk := src.read(8 * 1024 ** 2):
            dst.write(chunk)
            digest.update(chunk)
            unsynced += len(chunk)
            if unsynced >= 64 * 1024 ** 2:
                dst.flush()
                os.fsync(dst.fileno())
                unsynced = 0
        dst.flush()
        os.fsync(dst.fileno())
        if hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    if expected is not None and digest.hexdigest() != expected:
        raise ValueError(f'Archive checksum mismatch: {source}')


def archive_checkpoint(checkpoint, archive_run, *, keep=3, reserve_bytes=1024 ** 3):
    checkpoint, archive_run = Path(checkpoint), Path(archive_run)
    # The configured root must exist: do not silently replace an absent mount.
    if not archive_run.is_dir():
        raise FileNotFoundError(f'Archive run directory unavailable: {archive_run}')
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    current = archive_run / 'CURRENT'
    if current.exists():
        previous = resolve_checkpoint(archive_run)
        identity = checkpoint / 'run-identity.json'
        prior_identity = previous / 'run-identity.json'
        if not identity.exists() or not prior_identity.exists() or identity.read_bytes() != prior_identity.read_bytes():
            raise ValueError('Archive destination belongs to another or unidentified run; use a separate directory')
    files = manifest['sha256']
    if any(Path(name).name != name or (checkpoint / name).is_symlink() for name in files):
        raise ValueError('Invalid archive checkpoint filenames')
    required = sum((checkpoint / name).stat().st_size for name in files)
    ensure_free(archive_run, required, reserve_bytes)
    root = archive_run / 'checkpoints'
    root.mkdir(exist_ok=True)
    destination = root / checkpoint.name
    pending = root / ('.pending-' + uuid.uuid4().hex)
    if not destination.exists():
        pending.mkdir()
        try:
            for name, expected in files.items():
                copy_file(checkpoint / name, pending / name, expected)
            copy_file(checkpoint / 'manifest.json', pending / 'manifest.json')
            _fsync_dir(pending)
            os.replace(pending, destination)
            _fsync_dir(root)
        except BaseException:
            shutil.rmtree(pending, ignore_errors=True)
            raise
    else:
        from .checkpoints import _digest
        for name, expected in files.items():
            if _digest(destination / name) != expected:
                raise ValueError(f'Existing archive checksum mismatch: {name}')
    if not current.exists() or json.loads((resolve_checkpoint(archive_run) / 'manifest.json').read_text())['step'] <= manifest['step']:
        _atomic_text(current, checkpoint.name + '\n')
    committed = sorted(root.glob('step-*'), key=lambda p: (
        json.loads((p / 'manifest.json').read_text())['step'], p.stat().st_mtime_ns), reverse=True)
    current_name = current.read_text().strip()
    for old in committed[keep:]:
        if old.name != current_name:
            shutil.rmtree(old)
    return destination


class CheckpointArchiver:
    def __init__(self, archive_run, *, keep=3, reserve_bytes=1024 ** 3):
        self.destination = Path(archive_run)
        from .operations import file_lock
        self.ownership = file_lock(self.destination / '.archive.lock')
        self.ownership.__enter__()
        self.keep, self.reserve_bytes = keep, reserve_bytes
        self.condition = threading.Condition()
        self.pending = self.active = None
        self.closing = False
        self.error = None
        self.last_completed = None
        self.thread = threading.Thread(target=self._work, name='schnitz-archive', daemon=True)
        self.thread.start()

    def submit(self, checkpoint):
        with self.condition:
            if self.error:
                raise RuntimeError('Checkpoint archive failed; local checkpoint remains available') from self.error
            self.pending = Path(checkpoint)
            self._status()
            self.condition.notify()

    def _status(self):
        from .operations import atomic_json
        atomic_json(self.destination / 'archive-status.json', dict(
            updated_at=time.time(), active=str(self.active) if self.active else None,
            pending=str(self.pending) if self.pending else None,
            last_completed=self.last_completed, error=str(self.error) if self.error else None))

    def protected_names(self):
        with self.condition:
            return {p.name for p in (self.pending, self.active) if p is not None}

    def _work(self):
        try:
            self._copy_loop()
        except Exception as exc:
            # Includes failures writing status metadata, not only payload copies.
            with self.condition:
                self.error = exc
        finally:
            self.ownership.__exit__(None, None, None)

    def _copy_loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closing)
                if self.pending is None:
                    return
                self.active, self.pending = self.pending, None
                self._status()
            try:
                archive_checkpoint(self.active, self.destination, keep=self.keep, reserve_bytes=self.reserve_bytes)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self._status()
                return
            with self.condition:
                self.last_completed = self.active.name
                self.active = None
                self._status()

    def close(self, *, wait=False):
        with self.condition:
            self.closing = True
            self.condition.notify()
        if wait:
            self.thread.join()
        if self.error:
            raise RuntimeError('Checkpoint archive failed; local checkpoint remains available') from self.error


def restore_archive(archive_run, output):
    """Restore a verified checkpoint into a new run directory, never overwrite."""
    checkpoint = resolve_checkpoint(archive_run, verify=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Recovery copies the immutable set. Original prepared input must still be
    # available at the path in its config; resume verifies its content hash.
    restored = archive_checkpoint(checkpoint, output, reserve_bytes=0)
    return {'checkpoint': str(restored), 'output': str(output),
            'note': 'Resume requires the original prepared dataset specified by the checkpoint config.'}


def relocate_checkpoints(run, destination, *, archive_hint=None):
    """Move a stopped stage's checkpoint storage, retaining its public run paths.

    Destination is an existing, dedicated directory on the intended filesystem.
    All retained sets are verified before replacing the local directory with a
    symlink. An optional verified archive supplies same-filesystem hardlinks;
    deleting an archive name cannot delete the retained link. No shared cleanup.
    """
    from .checkpoints import _digest
    from .operations import run_lock
    run, destination = Path(run), Path(destination).absolute()
    with run_lock(run, clear_stop=False):
        source = run / 'checkpoints'
        if not destination.is_dir():
            raise FileNotFoundError('Destination must exist on the mounted disk')
        if source.is_symlink():
            if source.resolve() != destination.resolve():
                raise ValueError('Checkpoints already relocated to another destination')
            resolve_checkpoint(run, verify=True)
            return dict(destination=str(destination), local_bytes_removed=0)
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError('Destination must be outside local checkpoint storage')
        ensure_free(destination)
        resolve_checkpoint(run, verify=True)
        checkpoints = sorted(source.iterdir())
        if any(not p.is_dir() or not p.name.startswith('step-') for p in checkpoints):
            raise ValueError('Inspect pending/unknown checkpoint entries before relocating')
        local_bytes = 0
        for checkpoint in checkpoints:
            manifest = json.loads((checkpoint / 'manifest.json').read_text())
            files = manifest['sha256']
            if any(Path(n).name != n for n in files):
                raise ValueError('Invalid checkpoint filenames')
            target = destination / checkpoint.name
            hint = Path(archive_hint) / checkpoint.name if archive_hint else None
            if not target.exists():
                pending = destination / ('.pending-' + uuid.uuid4().hex)
                pending.mkdir()
                try:
                    for name, expected in files.items():
                        if hint is not None and (hint / name).is_file():
                            if _digest(hint / name) != expected:
                                raise ValueError(f'Archive checksum mismatch: {hint / name}')
                            try:
                                os.link(hint / name, pending / name)
                            except OSError:
                                copy_file(hint / name, pending / name, expected)
                        else:
                            copy_file(checkpoint / name, pending / name, expected)
                    copy_file(checkpoint / 'manifest.json', pending / 'manifest.json')
                    _fsync_dir(pending)
                    os.replace(pending, target)
                    _fsync_dir(destination)
                except BaseException:
                    shutil.rmtree(pending, ignore_errors=True)
                    raise
            else:
                if (target / 'manifest.json').read_bytes() != (checkpoint / 'manifest.json').read_bytes():
                    raise ValueError('Destination checkpoint manifest mismatch')
                for name, expected in files.items():
                    if _digest(target / name) != expected:
                        raise ValueError(f'Destination checksum mismatch: {name}')
            local_bytes += sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file())
        # Create the symlink first: unsupported platforms fail without moving data.
        token = uuid.uuid4().hex
        link, backup = run / ('.checkpoint-link-' + token), run / ('.relocated-' + token)
        link.symlink_to(destination, target_is_directory=True)
        os.replace(source, backup)
        try:
            os.replace(link, source)
            _fsync_dir(run)
        except BaseException:
            os.replace(backup, source)
            link.unlink(missing_ok=True)
            raise
        shutil.rmtree(backup)
        _fsync_dir(run)
        return dict(destination=str(destination), local_bytes_removed=local_bytes)
