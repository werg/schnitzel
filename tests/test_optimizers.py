import copy

import pytest
import torch
from safetensors.torch import load_file

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.optimizers import make_optimizer
from schnitz.training import train


def test_writer_key_learning_rate_keeps_all_parameters_owned(tiny_config):
    config = copy.deepcopy(tiny_config)
    config.train.writer_key_learning_rate = 1e-6
    agent = SchnitzelAgent(config)
    optimizer = make_optimizer(agent)
    rates = {name: group['lr'] for group, names in zip(
        optimizer.param_groups, optimizer._sdkb_parameter_names, strict=True)
        for name in names}
    assert rates['key_head.weight'] == 1e-6
    assert rates['address_maps.0.weight'] == 1e-6
    assert rates['query_maps.0.weight'] == config.train.learning_rate
    assert len(rates) == sum(p.requires_grad for p in agent.parameters())


@pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='Native Muon unavailable in this Torch')
@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_muon_excludes_reader_fallback_token_tables(tiny_config, kind):
    config = copy.deepcopy(tiny_config)
    config.train.optimizer = 'muon'
    config.memory.reader = kind
    agent = SchnitzelAgent(config)
    optimizer = make_optimizer(agent)
    muon_ids = {id(p) for g in optimizer.optimizers['muon'].param_groups for p in g['params']}
    adam_ids = {id(p) for g in optimizer.optimizers['adamw'].param_groups for p in g['params']}
    from schnitz.readers import SetReader
    readers = [m for m in agent.modules() if isinstance(m, SetReader)]
    assert readers
    for reader in readers:
        assert id(reader.null_tokens) not in muon_ids
        assert id(reader.null_tokens) in adam_ids
        assert id(reader.initial_slots) in adam_ids
        assert id(reader.output[-1].weight) in muon_ids
        query = torch.randn(1, reader.query_dim)
        empty = torch.empty(1, 0, reader.input_dim)
        reader(empty, query).tokens.sum().backward()
        assert torch.count_nonzero(reader.null_tokens.grad) == reader.null_tokens.numel()


@pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='Native Muon unavailable in this Torch')
def test_legacy_muon_slot_ownership_is_rejected_before_loading_weights(tmp_path, tiny_config, monkeypatch):
    import random
    from schnitz import checkpoints
    from schnitz.optimizers import MuonAdamW
    from schnitz.store import DiskStore
    config = copy.deepcopy(tiny_config)
    config.train.optimizer = 'muon'
    agent = SchnitzelAgent(config)
    optimizer = make_optimizer(agent)
    # Construct the previous ownership topology: fallback tables in memory Muon.
    groups = {name: [dict(params=list(g['params']), lr=g['lr']) for g in opt.param_groups]
              for name, opt in optimizer.optimizers.items()}
    tables = {id(p) for name, p in agent.named_parameters() if name.endswith('.null_tokens')}
    moved = [p for g in groups['adamw'] for p in g['params'] if id(p) in tables]
    assert moved
    for group in groups['adamw']:
        group['params'] = [p for p in group['params'] if id(p) not in tables]
    groups['muon'][-1]['params'].extend(moved)
    order = {id(p): i for i, p in enumerate(agent.parameters())}
    groups['muon'][-1]['params'].sort(key=lambda p: order[id(p)])
    legacy = MuonAdamW(groups['muon'], groups['adamw'], config.train)
    names = {id(p): name for name, p in agent.named_parameters()}
    legacy._sdkb_parameter_names = [[names[id(p)] for p in g['params']] for g in legacy.param_groups]
    cache = DiskStore(tmp_path/'cache.sqlite')
    saved = checkpoints.save_checkpoint(agent, legacy, tmp_path, 0, random.Random(1), cache, 'fixture')
    def forbidden(*args, **kwargs):
        raise AssertionError('Ownership mismatch must fail before model mutation')
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, 'load_model', forbidden)
    with pytest.raises(ValueError, match='names/order changed'):
        checkpoints.restore_checkpoint(agent, optimizer, saved, random.Random(1), 'fixture')


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_state_equal(left, right)
    else:
        assert a == b


@pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='Native Muon unavailable in this Torch')
@pytest.mark.parametrize('selected_only', [False, True])
def test_muon_ownership_and_complete_resume(tmp_path, tiny_config, monkeypatch, selected_only):
    from schnitz.operations import request_stop
    from schnitz.replay import ReplayTape
    config = copy.deepcopy(tiny_config)
    config.train.optimizer = 'muon'
    config.train.steps = 3
    config.train.gradient_accumulation = 2
    config.train.live_fraction = 1. if selected_only else .5
    config.train.selected_producers_only = selected_only
    config.train.checkpoint_every = 1000
    config.memory.noise_std = .02
    agent = SchnitzelAgent(config)
    optimizer = make_optimizer(agent)
    owned = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(p) for p in agent.parameters() if p.requires_grad}
    muon_ids = {id(p) for g in optimizer.optimizers['muon'].param_groups for p in g['params']}
    assert muon_ids
    assert id(agent.backbone.base.embedding.weight) not in muon_ids
    assert id(agent.write_slots) not in muon_ids
    full, partial = tmp_path / 'full', tmp_path / 'partial'
    train(config, full)
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        request_stop(partial)
        return result
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, partial)['saved_microbatches'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, partial, resume=True)
    a, b = resolve_checkpoint(full), resolve_checkpoint(partial)
    assert_state_equal(load_file(str(a / 'model.safetensors')), load_file(str(b / 'model.safetensors')))
    a_state = torch.load(a / 'training_state.pt', weights_only=True)
    b_state = torch.load(b / 'training_state.pt', weights_only=True)
    assert_state_equal(a_state, b_state)


def test_resume_rejects_optimizer_and_learning_rate_changes(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    run = tmp_path / 'run'
    train(config, run, stop_after=1)
    config.train.optimizer = 'muon'
    with pytest.raises(ValueError, match='hyperparameters'):
        train(config, run, resume=True)
    config.train.optimizer = 'adamw'
    config.train.learning_rate *= 2
    with pytest.raises(ValueError, match='hyperparameters'):
        train(config, run, resume=True)


def test_operational_policy_preserves_training_config(tmp_path, tiny_config):
    from schnitz.operations import configure_checkpoints
    run = tmp_path / 'run'
    train(tiny_config, run, stop_after=1)
    configure_checkpoints(run, checkpoint_every=1000, archive_dir=None)
    train(tiny_config, run, resume=True)
    from schnitz.training import config_from_run
    saved = config_from_run(run)
    assert saved.train.checkpoint_every == 1000
    assert saved.train.learning_rate == tiny_config.train.learning_rate
