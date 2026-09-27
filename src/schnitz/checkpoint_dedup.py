"""Verified hard-link deduplication of immutable, inactive checkpoint weights."""
from contextlib import ExitStack
from collections import defaultdict
import json
import os
from pathlib import Path
import stat
import uuid

from .operations import run_lock
from .trajectories import file_sha256


def _identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def deduplicate_weights(checkpoints, *, apply=False):
    paths = sorted({Path(p).absolute() for p in checkpoints})
    groups = defaultdict(list)
    planned = {}
    for checkpoint in paths:
        model = checkpoint / 'model.safetensors'
        if (checkpoint != checkpoint.resolve() or checkpoint.parent.name != 'checkpoints'
                or not checkpoint.name.startswith('step-') or model.is_symlink()
                or not stat.S_ISREG(model.stat().st_mode)):
            raise ValueError('Expected canonical checkpoint directories with regular weight files')
        digest = json.loads((checkpoint / 'manifest.json').read_text())['sha256']['model.safetensors']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid weight checksum')
        s = model.stat()
        attrs = tuple((name, os.getxattr(model, name)) for name in sorted(os.listxattr(model))) if hasattr(os, 'listxattr') else ()
        planned[model] = (_identity(model), (s.st_uid, s.st_gid, s.st_mode), attrs)
        groups[(s.st_dev, s.st_size, digest, s.st_uid, s.st_gid, s.st_mode, attrs)].append(model)
    replacements = []
    for key, models in groups.items():
        # Keep an already shared inode as the source. Choosing a newer unshared
        # copy by path order would retain it unnecessarily because shared targets
        # must remain intact (their other links may be outside this inventory).
        source = max(models, key=lambda model: model.stat().st_nlink)
        for target in models:
            # Existing links may belong to another retained archive: leave them intact.
            if _identity(source)[:2] != _identity(target)[:2] and target.stat().st_nlink == 1:
                replacements.append((source, target, key[2], key[1]))
    report = {'applied': apply, 'potential_bytes': sum(r[3] for r in replacements),
              'released_bytes': 0, 'replacements': []}
    if not apply:
        report['replacements'] = [{'source': str(a), 'target': str(b), 'sha256': h, 'bytes': n}
                                  for a, b, h, n in replacements]
        return report
    files = {p for a, b, _, _ in replacements for p in (a, b)}
    owners = {owner for p in files for owner in (p.parent.parent.parent, p.parent.parent.parent.parent)}
    with ExitStack() as stack:
        for owner in sorted(owners):
            stack.enter_context(run_lock(owner, clear_stop=False))
        identities = {p: planned[p][0] for p in files}
        for path in files:
            s = path.stat()
            attrs = tuple((name, os.getxattr(path, name)) for name in sorted(os.listxattr(path))) if hasattr(os, 'listxattr') else ()
            if (path.is_symlink() or _identity(path) != identities[path]
                    or (s.st_uid, s.st_gid, s.st_mode) != planned[path][1] or attrs != planned[path][2]):
                raise ValueError('Checkpoint changed before deduplication')
        verified = set()
        # Verify every candidate before changing any directory entry.
        for source, target, digest, _ in replacements:
            for path in (source, target):
                if path in verified:
                    continue
                if file_sha256(path) != digest or _identity(path) != identities[path]:
                    raise ValueError(f'Weight checksum or identity changed: {path}')
                verified.add(path)
        for source, target, digest, size in replacements:
            if (_identity(source) != identities[source] or _identity(target) != identities[target]
                    or target.stat().st_nlink != 1):
                raise ValueError('Checkpoint changed during deduplication')
            pending = target.with_name('.model-dedup-' + uuid.uuid4().hex)
            try:
                os.link(source, pending, follow_symlinks=False)
                if pending.is_symlink() or _identity(pending) != identities[source]:
                    raise ValueError('Source changed while creating verified link')
                os.replace(pending, target)
                if os.name != 'nt':
                    fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            finally:
                pending.unlink(missing_ok=True)
            report['released_bytes'] += size
            report['replacements'].append({'source': str(source), 'target': str(target),
                                           'sha256': digest, 'bytes': size})
    report['notice'] = 'Checkpoint contents and manifests are unchanged. Byte counts are logical file lengths; open readers may delay physical release. Never edit shared checkpoint files in place.'
    return report
