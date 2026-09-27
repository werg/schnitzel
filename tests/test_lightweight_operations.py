"""Host run controls inspect/copy checkpoint bytes without loading ML libraries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BOOTSTRAP = """
import importlib.abc
import sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'safetensors'}:
            raise AssertionError('ML dependency imported by host operation: '+fullname)
sys.meta_path.insert(0, NoML())
from schnitz.cli import main
main(sys.argv[1:])
"""


def cli(*arguments):
    result = subprocess.run([sys.executable, '-c', BOOTSTRAP, *map(str, arguments)],
                            env=os.environ | {'PYTHONPATH': str(Path(__file__).parents[1]/'src')},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def checkpoint(run):
    path = run/'checkpoints'/'step-000000007-fixture'
    path.mkdir(parents=True)
    # Storage commands copy opaque bytes; this is deliberately not a model load.
    (path/'model.safetensors').write_bytes(b'opaque model fixture')
    (path/'training_state.pt').write_bytes(b'opaque optimizer fixture')
    (path/'config.json').write_text(json.dumps({'train': {'checkpoint_every': 10000}}))
    (path/'run-identity.json').write_text(json.dumps({'id': 'fixture', 'format': 1}))
    manifest = {'format': 1, 'step': 7,
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}}
    (path/'manifest.json').write_text(json.dumps(manifest))
    (run/'CURRENT').write_text(path.name+'\n')
    (run/'metrics.jsonl').write_text('{"step": 11}\n')
    return path


def test_status_and_stop_do_not_import_ml_runtime(tmp_path):
    run = tmp_path/'run'
    checkpoint(run)
    status = cli('runs', 'status', '--output', run)
    assert status['checkpoints'][0]['checkpoint_step'] == 7
    assert status['checkpoints'][0]['latest_logged_step'] == 11
    assert cli('runs', 'stop', '--output', run)['status'] == 'stop_requested'
    assert cli('runs', 'status', '--output', run)['stop_requested']


def test_archive_restore_and_relocation_do_not_import_ml_runtime(tmp_path):
    run, archive, restored, relocated = (tmp_path/name for name in ('run', 'archive', 'restored', 'relocated'))
    original = checkpoint(run)
    archive.mkdir()
    cli('archive', '--run', run, '--destination', archive)
    cli('restore', '--archive', archive, '--output', restored)
    recovered = cli('runs', 'status', '--output', restored)['checkpoints'][0]['checkpoint']
    assert (Path(recovered)/'model.safetensors').read_bytes() == (original/'model.safetensors').read_bytes()
    relocated.mkdir()
    cli('relocate-checkpoints', '--run', run, '--destination', relocated)
    status = cli('runs', 'status', '--output', run)
    assert status['checkpoints'][0]['checkpoint_directory_is_symlink']
