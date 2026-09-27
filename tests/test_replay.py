import copy
import pytest
import torch
from torch import nn

from schnitz.replay import ReplayTape


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.dropout = nn.Dropout(0.3)

    def producer(self, x):
        h = self.dropout(self.linear(x))
        return h.sin(), h.cos()

    def consumer(self, values, x):
        key, value = values
        # Includes shared parameters, repeated use, key and value gradient paths.
        return (self.linear(x) * key).sum() + value.square().sum() + value.sum() * 0.13


@pytest.mark.parametrize("live", [(True, True), (True, False), (False, True)])
def test_exact_full_graph_parity(live):
    ref = SharedModel().double()
    replay = copy.deepcopy(ref)
    xs = [torch.randn(3, 5).double(), torch.randn(3, 5).double()]
    state = torch.get_rng_state()
    outputs = [ref.producer(x) for x in xs]
    outputs = [v if active else tuple(t.detach() for t in v) for v, active in zip(outputs, live, strict=True)]
    loss = sum(ref.consumer(v, x) for v, x in zip(outputs, xs, strict=True))
    loss.backward()
    torch.set_rng_state(state)
    tape = ReplayTape(verify_outputs=True)
    values = []
    for x, active in zip(xs, live, strict=True):
        if active:
            values.append(tape.capture(replay, lambda x=x: replay.producer(x)))
        else:
            with torch.no_grad():
                values.append(replay.producer(x))
    loss_replayed = sum(replay.consumer(v, x) for v, x in zip(values, xs, strict=True))
    loss_replayed.backward()
    rng_before = torch.get_rng_state().clone()
    tape.backward()
    assert torch.equal(rng_before, torch.get_rng_state())
    torch.testing.assert_close(loss, loss_replayed)
    for p, rp in zip(ref.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(p.grad, rp.grad, atol=1e-10, rtol=1e-10)


def test_optimizer_step_before_replay_rejected():
    model = SharedModel()
    tape = ReplayTape()
    values = tape.capture(model, lambda: model.producer(torch.randn(1, 5)))
    sum(x.sum() for x in values).backward()
    with torch.no_grad():
        model.linear.weight.add_(1)
    with pytest.raises(RuntimeError, match="Parameters changed"):
        tape.backward()


def test_mode_change_rejected():
    model = SharedModel()
    tape = ReplayTape()
    values = tape.capture(model, lambda: model.producer(torch.randn(1, 5)))
    sum(x.sum() for x in values).backward()
    model.eval()
    with pytest.raises(RuntimeError, match="training mode"):
        tape.backward()


def test_single_use():
    tape = ReplayTape()
    tape.backward()
    with pytest.raises(RuntimeError):
        tape.backward()


def test_mutable_buffer_rejected_and_restored():
    model = nn.BatchNorm1d(5)
    old = model.running_mean.clone()
    tape = ReplayTape()
    with pytest.raises(RuntimeError, match="buffer"):
        tape.capture(model, lambda: (model(torch.randn(3, 5)),))
    assert torch.equal(old, model.running_mean)


def test_cpu_autocast_replayed_under_original_precision():
    model = SharedModel()
    x = torch.randn(3, 5)
    tape = ReplayTape(verify_outputs=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = tape.capture(model, lambda: model.producer(x))
        loss = sum(x.float().square().mean() for x in values)
    loss.backward()
    tape.backward()
    assert model.linear.weight.grad is not None


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_bf16_capture_and_shared_consumer_inside_the_same_autocast_context(cache_enabled):
    reference = nn.Linear(16, 16)
    replayed = copy.deepcopy(reference)
    x = torch.randn(4, 16)
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache_enabled):
        values = reference(x).sin()
        expected = (reference(x*1.7)*values).float().square().mean()
    expected.backward()
    tape = ReplayTape(verify_outputs=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache_enabled):
        (values,) = tape.capture(replayed, lambda: (replayed(x).sin(),))
        actual = (replayed(x*1.7)*values).float().square().mean()
    actual.backward()
    rng_before = torch.get_rng_state().clone()
    tape.backward()
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for full, replay in zip(reference.parameters(), replayed.parameters(), strict=True):
        # Separate BF16 cast-node accumulation can round differently. This bound
        # detects a missing shared consumer path without claiming bitwise gradients.
        torch.testing.assert_close(replay.grad, full.grad, rtol=.02, atol=.002)


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_replay_restores_autocast_cache_policy_for_shared_parameter_paths(cache_enabled):
    reference = nn.Linear(16, 16)
    replayed = copy.deepcopy(reference)
    x = torch.randn(4, 16)
    def produce(model):
        return (model(x).sin() + model(x * 1.7).cos(),)
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache_enabled):
        output = produce(reference)[0]
    output.float().square().mean().backward()
    tape = ReplayTape(verify_outputs=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache_enabled):
        captured = tape.capture(replayed, lambda: produce(replayed))
    captured[0].float().square().mean().backward()
    # The caller may now be outside its original autocast context or have changed
    # cache policy. Replay must restore the producer's original accumulation graph.
    with torch.autocast('cpu', enabled=False, cache_enabled=not cache_enabled):
        tape.backward()
        assert torch.is_autocast_cache_enabled() == (not cache_enabled)
    for a, b in zip(reference.parameters(), replayed.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
