"""Local run ownership and cooperative stops, independent of GPU/container runtime."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with pending.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def control_dir(run):
    run = Path(run).resolve()
    # Same control directory through host/container bind mounts of the run root.
    key = hashlib.sha256(run.name.encode()).hexdigest()[:24]
    path = run.parent / '.sdkb-control' / key
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def file_lock(path):
    """OS lock released on process death; an old file is never a stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                handle.write(b'\0')
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Run is already running: {path}') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def run_lock(run, *, clear_stop=True):
    directory = control_dir(run)
    with file_lock(directory / 'owner.lock'):
        if clear_stop:
            (directory / 'STOP').unlink(missing_ok=True)
        yield


def stop_requested(run):
    return (control_dir(run) / 'STOP').exists()


CHECKPOINT_POLICY_FIELDS = {'checkpoint_every', 'keep_checkpoints', 'archive_keep_checkpoints',
                            'archive_dir', 'min_free_disk_bytes'}


def configure_checkpoints(run, **settings):
    """Explicit operational overrides; immutable scientific launch inputs stay intact."""
    settings = {k: v for k, v in settings.items() if v is not ...}
    if not settings or set(settings) - CHECKPOINT_POLICY_FIELDS:
        raise ValueError('Supply supported checkpoint policy settings')
    for name, value in settings.items():
        if name != 'archive_dir' and (not isinstance(value, int) or isinstance(value, bool)
                                     or value < (0 if name == 'min_free_disk_bytes' else 1)):
            raise ValueError(f'Invalid checkpoint policy: {name}')
    if settings.get('archive_dir') is not None and not Path(settings['archive_dir']).is_dir():
        raise FileNotFoundError('Archive directory must exist on the mounted disk')
    run = Path(run)
    if not run.is_dir():
        raise FileNotFoundError(run)
    with run_lock(run, clear_stop=False):
        path = run / 'checkpoint-policy.json'
        old = json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path, old | settings)
    return {'policy': str(path), 'settings': old | settings}


def apply_checkpoint_policy(config, run, launch=None):
    for root in (launch, run):
        if root is None:
            continue
        path = Path(root) / 'checkpoint-policy.json'
        if path.exists():
            settings = json.loads(path.read_text())
            if set(settings) - CHECKPOINT_POLICY_FIELDS:
                raise ValueError('Unsupported checkpoint policy setting')
            for name, value in settings.items():
                setattr(config.train, name, value)
    config.validate()


def request_stop(run):
    path = control_dir(run) / 'STOP'
    atomic_json(path, {'requested_at': time.time()})
    return {'status': 'stop_requested', 'output': str(Path(run).resolve()),
            'behavior': 'Finish the current microbatch/replay and save gradients plus full resume state; evaluations stop at the next stage boundary.'}


def run_status(run):
    directory = control_dir(run)
    path = directory / 'process.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'not_managed'}
    try:
        with file_lock(directory / 'owner.lock'):
            running = False
    except RuntimeError:
        running = True
    if not running and state['status'] == 'running':
        state['status'] = 'starting' if time.time() - state.get('started_at', 0) < 60 else 'interrupted'
    if not running and state['status'] == 'starting':
        pid_file = directory / 'pid.json'
        if pid_file.exists() and os.name != 'nt':
            try:
                os.kill(json.loads(pid_file.read_text())['pid'], 0)
            except ProcessLookupError:
                state['status'] = 'interrupted'
            except PermissionError:
                pass
        if time.time() - state.get('started_at', 0) > 60:
            state['status'] = 'interrupted'
    return state | {'running': running, 'stop_requested': stop_requested(run),
                    'log': str(directory / 'console.log'), 'checkpoints': checkpoint_status(run)}


def latest_logged_step(path):
    """Read a bounded tail, tolerating a concurrent incomplete JSONL write."""
    try:
        with Path(path).open('rb') as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - 65536)
            handle.seek(start)
            tail = handle.read(size - start)
    except FileNotFoundError:
        return None
    if start:
        tail = tail.partition(b'\n')[2]
    for line in reversed(tail.splitlines()):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        step = row.get('step') if isinstance(row, dict) else None
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
            return step
    return None


def checkpoint_status(run):
    import shutil
    root = Path(run)
    stages = [root] if (root / 'CURRENT').exists() else sorted(p.parent for p in root.glob('*/CURRENT'))
    result = []
    from .checkpoints import resolve_checkpoint
    for stage in stages:
        checkpoint = resolve_checkpoint(stage)
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        config = json.loads((checkpoint / 'config.json').read_text())
        report = dict(stage=str(stage), step=manifest['step'], checkpoint=str(checkpoint.resolve()),
                      checkpoint_step=manifest['step'], latest_logged_step=latest_logged_step(stage / 'metrics.jsonl'),
                      externalized=(stage / 'checkpoints').is_symlink(),
                      checkpoint_directory_is_symlink=(stage / 'checkpoints').is_symlink(),
                      filesystem_device_id=checkpoint.stat().st_dev,
                      same_filesystem_as_code=checkpoint.stat().st_dev == Path(__file__).stat().st_dev,
                      filesystem_free_bytes=shutil.disk_usage(checkpoint).free,
                      checkpoint_every=config['train']['checkpoint_every'])
        identity = checkpoint / 'run-identity.json'
        archive = config['train'].get('archive_dir')
        if archive and identity.exists():
            status = Path(archive) / json.loads(identity.read_text())['id'] / 'archive-status.json'
            report['archive'] = json.loads(status.read_text()) if status.exists() else {'status': 'unreported'}
        result.append(report)
    return result


def start_run(recipe, run, *, resume=False):
    """Start one detached local worker; use foreground launch inside a container."""
    run = Path(run).resolve()
    recipe = Path(recipe).resolve()
    if not recipe.is_file():
        raise FileNotFoundError(recipe)
    directory = control_dir(run)
    with file_lock(directory / 'start.lock'):
        state = run_status(run)
        if state['running'] or state['status'] == 'starting':
            raise RuntimeError('Run is already running or starting')
        if run.exists() and not resume:
            raise FileExistsError('Output exists; use --resume')
        atomic_json(directory / 'process.json', {'status': 'starting', 'recipe': str(recipe),
                                               'output': str(run), 'started_at': time.time()})
        # Only this explicit start acknowledges an earlier stop, before spawn.
        (directory / 'STOP').unlink(missing_ok=True)
        command = [sys.executable, '-m', 'schnitz.operations', str(recipe), str(run)]
        if resume:
            command.append('--resume')
        try:
            with (directory / 'console.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=os.name != 'nt',
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                    env=os.environ | {'PYTHONUNBUFFERED': '1'})
        except BaseException:
            atomic_json(directory / 'process.json', {'status': 'failed_to_start'})
            raise
        atomic_json(directory / 'pid.json', {'pid': child.pid})
    return {'status': 'starting', 'pid': child.pid, 'log': str(directory / 'console.log')}


def worker(recipe, run, resume=False):
    from .launch import launch
    directory = control_dir(run)
    state = {'status': 'running', 'pid': os.getpid(), 'recipe': str(recipe), 'output': str(run),
             'started_at': time.time()}
    atomic_json(directory / 'process.json', state)
    try:
        result = launch(recipe, run, resume=resume, preserve_stop=True)
    except BaseException:
        atomic_json(directory / 'process.json', state | {'status': 'failed', 'finished_at': time.time()})
        raise
    atomic_json(directory / 'process.json', state | {'status': result['status'], 'finished_at': time.time()})


if __name__ == '__main__':
    worker(sys.argv[1], sys.argv[2], '--resume' in sys.argv[3:])
