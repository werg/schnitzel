import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from schnitz.operations import request_stop
from schnitz.trajectories import file_sha256


def queue_fixture(root, monkeypatch, arms):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  'experiments/binding-compact-aware-20260920/run_confirmation.py'))
    main = namespace['main']
    root.mkdir()
    (root/'heldout.jsonl').write_text('')
    inputs = dict(heldout_sha256=file_sha256(root/'heldout.jsonl'), steps=400, arms={})
    for name in arms:
        (root/(name+'.yaml')).write_text('config')
        inputs['arms'][name] = dict(config_sha256=file_sha256(root/(name+'.yaml')))
        (root/name).mkdir()
        (root/name/'manifest.json').write_text(json.dumps(dict(step=400)))
    (root/'inputs.json').write_text(json.dumps(inputs))
    monkeypatch.setitem(main.__globals__, 'resolve_checkpoint', lambda stage, **_: stage)
    monkeypatch.setitem(main.__globals__, 'time', SimpleNamespace(sleep=lambda _: None))
    return main


@pytest.mark.parametrize('target', ['root', 'queue'])
def test_confirmation_stop_before_launch(tmp_path, monkeypatch, target):
    root = tmp_path/'study'
    main = queue_fixture(root, monkeypatch, [])
    request_stop(root if target == 'root' else root/'confirmation-queue')
    with pytest.raises(RuntimeError, match='Confirmation stop requested'):
        main(root)


def test_confirmation_stop_terminates_children_without_launching_pending(tmp_path, monkeypatch):
    root = tmp_path/'study'
    main = queue_fixture(root, monkeypatch, ['arm'])
    children = []
    class Child:
        def __init__(self, *args, **kwargs):
            self.pid = len(children) + 1
            self.terminated = self.waited = False
            self.polls = 0
            children.append(self)

        def poll(self):
            self.polls += 1
            request_stop(root/'confirmation-queue')
            return None if self.polls < 3 else 0

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return 0
    monkeypatch.setitem(main.__globals__, 'subprocess', SimpleNamespace(Popen=Child, STDOUT=-2))
    with pytest.raises(RuntimeError, match='Confirmation stop requested'):
        main(root)
    assert len(children) == 2
    assert all(c.terminated and c.waited for c in children)


def test_confirmation_selected_entrypoint_retains_shared_stop_control(tmp_path, monkeypatch):
    root = tmp_path/'study'
    main = queue_fixture(root, monkeypatch, [])
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=1, poll=lambda: 0)
    monkeypatch.setitem(main.__globals__, 'subprocess', SimpleNamespace(Popen=child, STDOUT=-2))
    run_jobs = main.__globals__['run_jobs']
    run_jobs(root, [('text', ['--output', str(root/'text')])], script='/frozen/text_control.py')
    assert calls[0][1:] == ['/frozen/text_control.py', '--output', str(root/'text')]
    request_stop(root)
    with pytest.raises(RuntimeError, match='Confirmation stop requested'):
        run_jobs(root, [('pending', [])], script='/frozen/text_control.py')
    assert len(calls) == 1
