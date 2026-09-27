import hashlib
import json
from pathlib import Path

import pytest

from schnitz.operations import run_lock


def checkpoint(root, arm, step, data=b'weights'):
    path = root / arm / 'checkpoints' / f'step-{step:09d}-test'
    path.mkdir(parents=True)
    (path / 'model.safetensors').write_bytes(data)
    (path / 'manifest.json').write_text(json.dumps({'sha256': {
        'model.safetensors': hashlib.sha256(data).hexdigest()}}))
    return path


def test_dedup_is_dry_by_default_then_atomic_and_idempotent(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    paths = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b', 'c']]
    before = [(p / 'manifest.json').read_bytes() for p in paths]
    plan = deduplicate_weights(paths)
    assert plan['potential_bytes'] == 14
    assert len({(p / 'model.safetensors').stat().st_ino for p in paths}) == 3
    result = deduplicate_weights(paths, apply=True)
    assert result['released_bytes'] == 14
    assert len({(p / 'model.safetensors').stat().st_ino for p in paths}) == 1
    assert [(p / 'manifest.json').read_bytes() for p in paths] == before
    assert all((p / 'model.safetensors').read_bytes() == b'weights' for p in paths)
    assert deduplicate_weights(paths, apply=True)['released_bytes'] == 0


def test_corrupt_duplicate_aborts_without_replacement(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    a, b = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b']]
    (b / 'model.safetensors').write_bytes(b'corrupt')
    inodes = [(p / 'model.safetensors').stat().st_ino for p in [a, b]]
    with pytest.raises(ValueError, match='checksum'):
        deduplicate_weights([a, b], apply=True)
    assert inodes == [(p / 'model.safetensors').stat().st_ino for p in [a, b]]
    assert (b / 'model.safetensors').read_bytes() == b'corrupt'


def test_dedup_refuses_active_run_or_parent(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    a, b = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b']]
    for owner in [tmp_path, a.parent.parent]:
        with run_lock(owner), pytest.raises(RuntimeError, match='already running'):
            deduplicate_weights([a, b], apply=True)
    assert (a / 'model.safetensors').stat().st_ino != (b / 'model.safetensors').stat().st_ino


def test_dedup_preserves_distinct_permissions_and_existing_links(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    a, b, c = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b', 'c']]
    (b / 'model.safetensors').chmod(0o600)
    Path(tmp_path / 'other-copy').hardlink_to(c / 'model.safetensors')
    existing_inode = (c / 'model.safetensors').stat().st_ino
    result = deduplicate_weights([a, b, c], apply=True)
    assert result['released_bytes'] == 7
    assert (c / 'model.safetensors').stat().st_ino == existing_inode
    assert (tmp_path / 'other-copy').stat().st_ino == existing_inode
    assert (a / 'model.safetensors').stat().st_ino == existing_inode
    assert (b / 'model.safetensors').stat().st_mode & 0o777 == 0o600


def test_dedup_reuses_existing_shared_copy_for_new_warm_starts(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    a, b, source = [checkpoint(tmp_path, arm, 0) for arm in ['a-new', 'b-new', 'z-source']]
    alias = tmp_path/'retained-archive-copy'
    alias.hardlink_to(source/'model.safetensors')
    inode = alias.stat().st_ino
    plan = deduplicate_weights([a, b, source])
    assert plan['potential_bytes'] == 14
    assert all(r['source'] == str(source/'model.safetensors') for r in plan['replacements'])
    result = deduplicate_weights([a, b, source], apply=True)
    assert result['released_bytes'] == 14
    assert all((p/'model.safetensors').stat().st_ino == inode for p in (a, b, source))
    assert alias.stat().st_ino == inode and alias.read_bytes() == b'weights'


def test_dedup_rejects_symlinked_payload(tmp_path):
    from schnitz.checkpoint_dedup import deduplicate_weights
    a, b = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b']]
    (b / 'model.safetensors').unlink()
    (b / 'model.safetensors').symlink_to(a / 'model.safetensors')
    with pytest.raises(ValueError, match='regular'):
        deduplicate_weights([a, b], apply=True)


def test_failed_publication_preserves_destination_and_removes_temporary_link(tmp_path, monkeypatch):
    from schnitz.checkpoint_dedup import deduplicate_weights
    import schnitz.checkpoint_dedup as module
    a, b = [checkpoint(tmp_path, arm, 0) for arm in ['a', 'b']]
    before = (b / 'model.safetensors').stat().st_ino

    def fail(*args):
        raise OSError('simulated publication failure')

    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError, match='publication failure'):
        deduplicate_weights([a, b], apply=True)
    assert (b / 'model.safetensors').stat().st_ino == before
    assert (b / 'model.safetensors').read_bytes() == b'weights'
    assert not list(b.glob('.model-dedup-*'))
