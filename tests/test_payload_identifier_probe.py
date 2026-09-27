from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

from schnitz.data import make_multiuse_world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from probe_payload_identifiers import split_identifiers, Readout
finally:
    sys.path.pop(0)


def test_identifier_probe_splits_complete_worlds_without_source_overlap():
    episodes = [e for i in range(4) for e in make_multiuse_world(i, bindings=2)]
    groups = split_identifiers(episodes, 1)
    assert len(groups['heldout']) == 2 and len(groups['train']) == 6
    assert not {e.environment for e in groups['train']} & {e.environment for e in groups['heldout']}
    assert not {e.required_ids[0] for e in groups['train']} & {e.required_ids[0] for e in groups['heldout']}


@pytest.mark.parametrize('corruption', ['future', 'label', 'duplicate'])
def test_identifier_probe_rejects_invalid_examples(corruption):
    episodes = [e for i in range(2) for e in make_multiuse_world(i, bindings=2)]
    index = next(i for i, e in enumerate(episodes) if e.task_family == 'multiuse/identifier')
    if corruption == 'future':
        episodes[index] = replace(episodes[index], query_time=0)
    elif corruption == 'label':
        episodes[index] = replace(episodes[index], answer='api_zzzzzz')
    else:
        episodes.append(episodes[index])
    with pytest.raises(ValueError):
        split_identifiers(episodes, 1)


def test_readout_normalization_is_fixed_from_training_values():
    values = torch.arange(24.).reshape(4, 6)
    model = Readout(values, 8)
    before = {k: v.clone() for k, v in model.named_buffers()}
    model(torch.full((2, 6), 1e3)).sum().backward()
    assert all(torch.equal(dict(model.named_buffers())[k], v) for k, v in before.items())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('representation', ['reader', 'reader_inputs', 'reader_state'])
def test_reader_feature_uses_only_causal_stored_memory(tmp_path, tiny_config, monkeypatch, representation):
    from schnitz.agent import SchnitzelAgent
    from schnitz.evaluation import build_shared_bank
    from schnitz.store import DiskStore
    from probe_payload_identifiers import reader_feature
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing, c.memory.read_steps = 'loop_boundary', 1
    c.train.max_source_tokens = c.train.max_prompt_tokens = 512
    c.validate()
    agent = SchnitzelAgent(c).eval().requires_grad_(False)
    episode = next(e for e in make_multiuse_world(1, bindings=1) if e.task_family == 'multiuse/identifier')
    other = replace(next(e for e in make_multiuse_world(2, bindings=1)
                         if e.task_family == 'multiuse/identifier'), query=episode.query)
    store = DiskStore(tmp_path/'bank.sqlite')
    build_shared_bank(agent, store, [episode, other])
    def forbidden(*args, **kwargs):
        raise AssertionError('Feature extraction used source encoding or target tokens')
    monkeypatch.setattr(agent, 'produce', forbidden)
    monkeypatch.setattr(agent, 'target_ids', forbidden)
    options = {} if representation == 'reader' else {'representation': representation}
    full = reader_feature(agent, store, episode, **options)
    changed_target = reader_feature(agent, store, replace(episode, answer='UNSEEN FUTURE'), **options)
    zero = reader_feature(agent, store, episode, zero_values=True, **options)
    width = {'reader': c.memory.read_slots*agent.width,
             'reader_inputs': c.memory.reader_rounds*c.memory.reader_width,
             'reader_state': c.memory.read_slots*c.memory.reader_width}[representation]
    assert full.shape == (width,)
    torch.testing.assert_close(full, changed_target, rtol=0, atol=0)
    assert not torch.equal(full, zero)
    torch.testing.assert_close(zero, reader_feature(agent, store, other, zero_values=True, **options), rtol=0, atol=0)
    assert not full.requires_grad and full.device.type == 'cpu'
