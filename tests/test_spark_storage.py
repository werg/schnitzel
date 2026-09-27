"""The Docker wrapper must not silently fall back from configured storage."""
from pathlib import Path
import shutil
import subprocess
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Linux/macOS Bash wrapper')


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / '.schnitz').mkdir()
    script = root / 'scripts/spark.sh'
    shutil.copyfile(Path(__file__).parents[1] / 'scripts/spark.sh', script)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    docker = bin_dir / 'docker'
    docker.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    docker.chmod(0o755)
    capture = tmp_path / 'arguments'
    monkeypatch.setenv('CAPTURE', str(capture))
    monkeypatch.setenv('PATH', str(bin_dir) + ':/usr/bin:/bin')
    monkeypatch.setenv('SCHNITZELJAGD_CACHE_DIR', str(tmp_path / 'cache'))
    (tmp_path / 'cache').mkdir()
    monkeypatch.delenv('SCHNITZELJAGD_RUNS_DIR', raising=False)
    monkeypatch.delenv('SCHNITZELJAGD_ARCHIVE_DIR', raising=False)
    return root, script, capture


def test_machine_cache_default_and_environment_override(wrapper, tmp_path, monkeypatch):
    root, script, capture = wrapper
    external = tmp_path/'external cache'
    external.mkdir()
    (root/'.schnitz/cache-dir').write_text(str(external)+'\n')
    monkeypatch.delenv('SCHNITZELJAGD_CACHE_DIR')
    args = ['bash', str(script), 'run', 'schnitz', 'doctor']
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={external},dst=/cache' in capture.read_text().splitlines()
    override = tmp_path/'override-cache'
    override.mkdir()
    monkeypatch.setenv('SCHNITZELJAGD_CACHE_DIR', str(override))
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={override},dst=/cache' in capture.read_text().splitlines()


def test_runtime_and_temporary_caches_share_configured_storage(wrapper, tmp_path):
    _, script, capture = wrapper
    subprocess.run(['bash', str(script), 'run', 'schnitz', 'doctor'], check=True, capture_output=True)
    args = capture.read_text().splitlines()
    for setting in ['XDG_CACHE_HOME=/cache/xdg', 'TRITON_CACHE_DIR=/cache/triton',
                    'CUDA_CACHE_PATH=/cache/cuda', 'TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor',
                    'TORCH_HOME=/cache/torch', 'WANDB_CACHE_DIR=/cache/wandb', 'TMPDIR=/cache/tmp']:
        assert setting in args
    assert (tmp_path/'cache/tmp').is_dir()


@pytest.mark.parametrize('environment', [False, True])
def test_missing_configured_cache_is_not_created(wrapper, tmp_path, monkeypatch, environment):
    root, script, capture = wrapper
    absent = tmp_path/'unmounted'/'cache'
    if environment:
        monkeypatch.setenv('SCHNITZELJAGD_CACHE_DIR', str(absent))
    else:
        monkeypatch.delenv('SCHNITZELJAGD_CACHE_DIR')
        (root/'.schnitz/cache-dir').write_text(str(absent)+'\n')
    result = subprocess.run(['bash', str(script), 'run', 'schnitz', 'doctor'], capture_output=True)
    assert result.returncode != 0
    assert not absent.exists() and not capture.exists()


def test_machine_storage_default_and_environment_override(wrapper, tmp_path, monkeypatch):
    root, script, capture = wrapper
    external = tmp_path / 'external runs'
    external.mkdir()
    (root / '.schnitz/runs-dir').write_text(str(external) + '\n')
    args = ['bash', str(script), 'start', '--recipe', 'recipe.yaml', '--output', '/runs/test']
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={external},dst=/runs' in capture.read_text().splitlines()
    assert not (root / 'runs').exists()
    override = tmp_path / 'override'
    override.mkdir()
    monkeypatch.setenv('SCHNITZELJAGD_RUNS_DIR', str(override))
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={override},dst=/runs' in capture.read_text().splitlines()


def test_missing_configured_storage_fails_without_creating_it(wrapper, tmp_path):
    root, script, capture = wrapper
    absent = tmp_path / 'unmounted' / 'runs'
    (root / '.schnitz/runs-dir').write_text(str(absent) + '\n')
    result = subprocess.run(['bash', str(script), 'run', 'schnitz', 'doctor'], capture_output=True)
    assert result.returncode != 0
    assert not absent.exists()
    assert not capture.exists()


@pytest.mark.parametrize('command', [['start'], ['run', 'schnitz', 'launch'],
                                     ['run', 'schnitz', 'train'], ['run', 'schnitz', 'runs', 'start']])
@pytest.mark.parametrize('output', [['--output', 'runs/accidental-local'],
                                   ['--output=runs/accidental-local'], ['--output', '']])
def test_relative_run_output_cannot_bypass_the_storage_mount(wrapper, tmp_path, command, output):
    root, script, capture = wrapper
    external = tmp_path / 'external'
    external.mkdir()
    (root / '.schnitz/runs-dir').write_text(str(external) + '\n')
    result = subprocess.run(['bash', str(script), *command, *output], capture_output=True)
    assert result.returncode != 0
    assert b'/runs/' in result.stderr
    assert not capture.exists()
