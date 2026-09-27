from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from schnitz.data import make_multiuse_world


def helpers():
    root = Path(__file__).parents[1]
    return (runpy.run_path(str(root/'experiments/binding-freshness-readout-20260920/prepare.py')),
            runpy.run_path(str(root/'scripts/make_identifier_character_control.py')))


def test_readout_data_preserves_heldout_and_rejects_prior_vocabulary():
    helper, questions = helpers()
    heldout = [questions['full_episode'](e) if e.task_family == 'multiuse/identifier' else e
               for e in make_multiuse_world(0, split='readout-test-heldout', bindings=2)]
    blocked = {e.answer for e in make_multiuse_world(0, split=helper['TRAIN_SPLIT'], bindings=2)
               if e.task_family == 'multiuse/identifier'}
    episodes, report = helper['make_examples'](heldout, blocked, set(), training_worlds=3)
    assert episodes[:len(heldout)] == heldout
    assert report['accepted_seeds'] == [1, 2, 3]
    assert report['rejected_seeds'] == [0]
    training = episodes[len(heldout):]
    heldout_targets = {e.answer for e in heldout if e.task_family == 'multiuse/identifier'}
    assert not {e.answer for e in training if e.task_family == 'multiuse/identifier'} & (blocked | heldout_targets)
    assert not {s.record_id for e in training for s in e.supports} & {
        s.record_id for e in heldout for s in e.supports}
    assert {e.query for e in episodes if e.task_family == 'multiuse/identifier'} == {questions['FULL_QUERY']}


@pytest.mark.parametrize('corruption', ['future', 'query'])
def test_readout_data_rejects_broken_causal_or_constant_query_contract(corruption):
    helper, questions = helpers()
    episode = questions['full_episode'](next(e for e in make_multiuse_world(0, bindings=2)
                                            if e.task_family == 'multiuse/identifier'))
    if corruption == 'future':
        episode = replace(episode, query_time=0)
    else:
        episode = replace(episode, query='A different entity-specific query')
    with pytest.raises(ValueError):
        helper['make_examples']([episode], set(), set(), training_worlds=1)
