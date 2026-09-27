from dataclasses import replace
from pathlib import Path
import sys

import pytest

from schnitz.data import make_multiuse_world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from evaluate_exact_detail import selected_text
finally:
    sys.path.pop(0)


def episode():
    return next(e for e in make_multiuse_world(12, bindings=2)
                if e.task_family == 'multiuse/identifier')


def test_text_control_has_only_the_oracle_selected_causal_source():
    e = episode()
    text = selected_text(e)
    assert text == next(s.text for s in e.supports if s.record_id in e.required_ids)
    assert e.answer in text
    assert all(s.text not in text for s in e.supports if s.record_id not in e.required_ids)


@pytest.mark.parametrize('corruption', ['future', 'missing', 'duplicate'])
def test_text_control_rejects_invalid_source_boundaries(corruption):
    e = episode()
    if corruption == 'future':
        e = replace(e, query_time=0)
    elif corruption == 'missing':
        e = replace(e, required_ids=('absent',))
    else:
        e = replace(e, supports=e.supports + (e.supports[0],))
    with pytest.raises(ValueError):
        selected_text(e)
