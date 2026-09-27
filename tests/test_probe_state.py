import random

import pytest
import torch

from schnitz.probe_state import restore_probe_state, save_probe_state


def setup(kind="adamw"):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.3), torch.nn.Linear(3, 2))
    if kind == 'muon':
        from schnitz.config import TrainConfig
        from schnitz.optimizers import MuonAdamW
        optimizer = MuonAdamW([{'params': [p for p in model.parameters() if p.ndim == 2], 'lr': .02}],
                             [{'params': [p for p in model.parameters() if p.ndim != 2], 'lr': .02}], TrainConfig())
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    sampler = torch.Generator().manual_seed(9)
    return model, optimizer, sampler


def update(model, optimizer, sampler):
    optimizer.zero_grad()
    values = torch.randn(4, 3, generator=sampler) * random.random()
    model(values).square().sum().backward()
    optimizer.step()


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_probe_resume_restores_ownership_hyperparameters_rng_and_metrics(tmp_path, kind):
    path = tmp_path / 'resume.pt'
    a, opt, rng = setup(kind)
    update(a, opt, rng)
    save_probe_state(path, a, opt, rng, {'seed': 9}, 1, reserve_bytes=0)
    update(a, opt, rng)
    expected = {k:v.clone() for k,v in a.state_dict().items()}
    (tmp_path / 'metrics.jsonl').write_text('{"step": 1}\n{"step": 2}\n{"step":')
    b, other, sampler = setup(kind)
    other.param_groups[0]['lr'] = .7
    assert restore_probe_state(path, b, other, sampler, {'seed': 9}) == 1
    assert other.param_groups[0]['lr'] == .02
    assert (tmp_path / 'metrics.jsonl').read_text() == '{"step": 1}\n'
    update(b, other, sampler)
    for key, tensor in b.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)


def test_probe_rejects_reordered_optimizer_before_mutation(tmp_path):
    a, optimizer, sampler = setup()
    path = tmp_path / 'resume.pt'
    save_probe_state(path, a, optimizer, sampler, {}, 0, reserve_bytes=0)
    other = torch.optim.AdamW(list(reversed(list(a.parameters()))))
    with pytest.raises(ValueError, match='ownership'):
        restore_probe_state(path, a, other, sampler, {})
    with pytest.raises(ValueError, match='identity'):
        restore_probe_state(path, a, optimizer, sampler, {'changed': True})


def test_failed_probe_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    a, opt, sampler = setup()
    path = tmp_path / 'resume.pt'
    save_probe_state(path, a, opt, sampler, {}, 0, reserve_bytes=0)
    previous = path.read_bytes()
    def fail(*_args):
        raise OSError('disk failure')
    monkeypatch.setattr('schnitz.probe_state._fsync', fail)
    with pytest.raises(OSError, match='disk failure'):
        save_probe_state(path, a, opt, sampler, {}, 1, reserve_bytes=0)
    assert path.read_bytes() == previous
    assert not path.with_suffix('.tmp').exists()
