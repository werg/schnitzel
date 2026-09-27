import importlib.util
from pathlib import Path

from safetensors.torch import load, save
import torch

from schnitz.compaction import SyntheticCompactor, contribution_loss
from schnitz.readers import SetReader


def test_compactor_training_uses_serialized_precision_and_frozen_reader():
    path = Path(__file__).parents[1] / 'scripts/probe_stored_compaction.py'
    spec = importlib.util.spec_from_file_location('stored_compaction_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compactor = SyntheticCompactor(24, 24, 1)
    reader = SetReader(24, 12, 32, width=24, slots=2, rounds=2).eval().requires_grad_(False)
    raw, query = torch.randn(3, 2, 24).bfloat16().float(), torch.randn(3, 12)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        values, weights = module.serialized_codes(compactor, raw)
        stored = load(save({'values': values.detach().bfloat16(), 'weights': weights.detach()}))
        live = reader(values, query, weights).tokens
        persisted = reader(stored['values'].float(), query, stored['weights']).tokens
        torch.testing.assert_close(live, persisted, rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(1), torch.full((3,), 2.), rtol=0, atol=0)
        loss = contribution_loss(reader, raw, torch.ones(3, 2), values, weights, query)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in reader.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in compactor.parameters())


def test_cached_single_read_task_gradient_matches_causal_prefix_plan(tiny_config):
    from schnitz.agent import SchnitzelAgent
    path = Path(__file__).parents[1] / 'scripts/probe_stored_compaction.py'
    spec = importlib.util.spec_from_file_location('task_compaction_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing, c.memory.read_steps = 'loop_boundary', 1
    c.validate()
    agent = SchnitzelAgent(c).eval().requires_grad_(False)
    compactor = SyntheticCompactor(24, 24, 1)
    prompt, target = agent.prompt_ids('Use the stored rules.'), agent.target_ids('RETRY')
    raw = torch.randn(1, 2, 24).bfloat16().float()
    captured, tokens = [], None
    def provider(completed, query):
        nonlocal tokens
        if completed == 1:
            captured.append(query.detach().clone())
            values, weights = module.serialized_codes(compactor, raw)
            tokens = agent.reader(values, query, weights).tokens
        return tokens
    planned = agent.plan_loop_memory(prompt, provider)
    reference = agent.conditioned_nll(prompt, target, planned)
    expected = torch.autograd.grad(reference, tuple(compactor.parameters()))
    values, weights = module.serialized_codes(compactor, raw)
    tokens = agent.reader(values, captured[0], weights).tokens
    cached = module.single_read_memory(agent, prompt, tokens)
    actual = agent.conditioned_nll(prompt, target, cached)
    gradients = torch.autograd.grad(actual, tuple(compactor.parameters()))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in agent.parameters())


def test_compactor_probe_emergency_resume_preserves_complete_state(tmp_path, tiny_config, monkeypatch):
    import pytest
    from schnitz.data import make_multiuse_world, save_episodes
    from schnitz.training import train
    path = Path(__file__).parents[1] / 'scripts/probe_stored_compaction.py'
    spec = importlib.util.spec_from_file_location('recover_compaction_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing, c.memory.read_steps = 'loop_boundary', 1
    c.train.live_fraction = 1.
    c.train.max_source_tokens = 512
    c.train.max_prompt_tokens = 512
    source = tmp_path/'source'
    train(c, source)
    data = tmp_path/'episodes.jsonl'
    save_episodes(data, make_multiuse_world(0, split='compactor-recovery-train', bindings=2))
    # Keep real bank creation, feature extraction, optimization and stored evaluation,
    # using one held-out world for a small download-free recovery regression.
    monkeypatch.setattr(module, 'make_multiuse_world', lambda i, **kw: make_multiuse_world(i, **kw) if i == 0 else [])
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    module.run(source, data, full, steps=2, batch_size=2)
    original = module.MuonAdamW.step
    updates = []
    def update(optimizer):
        original(optimizer)
        updates.append(1)
    monkeypatch.setattr(module.MuonAdamW, 'step', update)
    monkeypatch.setattr(module, 'stop_requested', lambda _: bool(updates))
    with pytest.raises(RuntimeError, match='Stopped'):
        module.run(source, data, resumed, steps=2, batch_size=2)
    state = torch.load(resumed/'resume.pt', weights_only=True)
    assert state['step'] == 1
    assert state['format'] == 'sdkb-probe-v1'
    assert 'python_rng' in state
    monkeypatch.setattr(module, 'stop_requested', lambda _: False)
    module.run(source, data, resumed, steps=2, batch_size=2)
    expected = torch.load(full/'resume.pt', weights_only=True)
    actual = torch.load(resumed/'resume.pt', weights_only=True)
    def same(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                same(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                same(x, y)
        else:
            assert a == b
    same(actual, expected)
