import copy
import pytest
import torch

from schnitz.readers import SetReader, MultiSpaceReader, merge_statistics


def reader(kind="mlp", **kwargs):
    return SetReader(7, 5, 9, width=12, slots=3, rounds=3, kind=kind, **kwargs).double()


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_permutation_invariance(kind):
    r = reader(kind)
    x, q, weights = torch.randn(2, 11, 7).double(), torch.randn(2, 5).double(), torch.rand(2, 11).double()
    p = torch.randperm(11)
    torch.testing.assert_close(r(x, q, weights).tokens, r(x[:, p], q, weights[:, p]).tokens)


@pytest.mark.parametrize("kind", ["mlp", "attention"])
@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_chunked_forward_and_backward_parity(kind, chunk_size):
    raw = reader(kind, chunk_size=99)
    small = copy.deepcopy(raw)
    small.chunk_size = chunk_size
    small.checkpoint_chunks = True
    x = torch.randn(2, 9, 7, dtype=torch.float64, requires_grad=True)
    q = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    w = torch.rand(2, 9, dtype=torch.float64, requires_grad=True)
    xs, qs, ws = [t.detach().clone().requires_grad_() for t in (x, q, w)]
    a = raw(x, q, w).tokens
    b = small(xs, qs, ws).tokens
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    a.square().sum().backward()
    b.square().sum().backward()
    for left, right in zip((x, q, w), (xs, qs, ws), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-9, rtol=1e-9)
    for p, ps in zip(raw.parameters(), small.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, ps.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_cluster_statistics_merge_before_normalization(kind):
    r = reader(kind)
    x, q = torch.randn(2, 8, 7).double(), torch.randn(2, 5).double()
    state, w = r.initial_state(q), torch.rand(2, 8).double()
    full = r.aggregate(0, x, q, state, w)
    a = r.aggregate(0, x[:, :3], q, state, w[:, :3])
    b = r.aggregate(0, x[:, 3:], q, state, w[:, 3:])
    merged = merge_statistics([a, b])
    torch.testing.assert_close(full.mean(), merged.mean())
    torch.testing.assert_close(full.log_mass(), merged.log_mass())


@pytest.mark.parametrize("kind", ["mlp", "attention"])
@pytest.mark.parametrize("n", [0, 5])
def test_empty_or_zero_weight_neighborhood(kind, n):
    r = reader(kind)
    x, q, w = torch.randn(2, n, 7).double(), torch.randn(2, 5).double(), torch.zeros(2, n).double()
    out = r(x, q, w).tokens
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, r.null_tokens[None].expand(2, -1, -1))
    out.sum().backward()


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_zero_weight_irrelevant_record_does_not_change_read(kind):
    r = reader(kind)
    x, q = torch.randn(1, 3, 7).double(), torch.randn(1, 5).double()
    extended = torch.cat((x, torch.randn(1, 1, 7).double()), 1)
    torch.testing.assert_close(r(x, q).tokens,
                               r(extended, q, torch.tensor([[1, 1, 1, 0.]]).double()).tokens)


def test_negative_weights_rejected():
    r = reader()
    with pytest.raises(ValueError):
        r(torch.randn(1, 1, 7).double(), torch.randn(1, 5).double(), -torch.ones(1, 1).double())


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_multiscale_shared_feedback_gradients(kind):
    r = MultiSpaceReader([4, 8], 5, 12, width=12, slots=3, rounds=2, kind=kind)
    x = [torch.randn(2, n, d, requires_grad=True) for n, d in [(8, 4), (4, 8)]]
    q = torch.randn(2, 5, requires_grad=True)
    out = r(x, q, [torch.ones(2, 8), torch.ones(2, 4)])
    assert out.shape == (2, 3, 12)
    out.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in x + [q])


def test_rounds_have_a_real_effect():
    r = reader()
    x, q = torch.randn(1, 6, 7).double(), torch.randn(1, 5).double()
    out = r(x, q, diagnostics=True)
    assert not torch.allclose(out.states[0], out.states[1])


def test_bf16_autocast_finite():
    r = SetReader(8, 4, 8, width=16, slots=2, rounds=2, checkpoint_chunks=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = r(torch.randn(2, 5, 8), torch.randn(2, 4)).tokens
        loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_attention_zero_multiplicity_gradient_matches_explicit_formula():
    from schnitz.readers import AttentionRound, Statistics
    import math
    block = AttentionRound(4, 3, 6, 2).double()
    x = torch.randn(1, 3, 4, dtype=torch.double)
    query = torch.randn(1, 3, dtype=torch.double)
    state = torch.randn(1, 2, 6, dtype=torch.double)
    weights = torch.tensor([[1., 0., 2.]], dtype=torch.double, requires_grad=True)
    stats = Statistics(*block(x, query, state, weights))
    actual = torch.autograd.grad(stats.mean().sum(), weights)[0]
    q = block.state(state) + block.query(query)[:, None] + block.slot[None]
    scores = torch.einsum('bmd,bnd->bmn', q, block.key(x)) / math.sqrt(6)
    coeff = scores.exp() * weights[:, None]
    expected_value = torch.einsum('bmn,bnd->bmd', coeff, block.value(x)) / coeff.sum(-1, keepdim=True)
    expected = torch.autograd.grad(expected_value.sum(), weights)[0]
    torch.testing.assert_close(actual, expected)


def test_zero_mass_statistics_backward_is_finite():
    from schnitz.readers import Statistics
    numerator = torch.zeros(1, 2, 8, requires_grad=True)
    mass = torch.zeros(1, 2, 1, requires_grad=True)
    stats = Statistics(numerator, mass, torch.zeros_like(mass))
    (8 * stats.mean().sum()).backward()
    assert torch.isfinite(numerator.grad).all()
    assert torch.isfinite(mass.grad).all()


@pytest.mark.parametrize('autocast', [False, True])
def test_saturated_mlp_gate_has_finite_backward(autocast):
    from contextlib import nullcontext
    r = SetReader(8, 4, 8, width=16, slots=2, rounds=2, checkpoint_chunks=True)
    for block in r.blocks:
        with torch.no_grad():
            block.gate.weight.zero_()
            block.gate.bias.fill_(-1000)
    x = torch.randn(1, 2, 8, requires_grad=True)
    q = torch.randn(1, 4, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16) if autocast else nullcontext():
        out = r(x, q).tokens
        loss = out.float().square().sum() * 100
    loss.backward()
    assert torch.isfinite(out).all()
    for name, p in [('x', x), ('q', q), *r.named_parameters()]:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_tiny_positive_gate_preserves_conditional_mean_mass_and_gradients():
    from schnitz.readers import MLPRound, Statistics
    torch.manual_seed(3)
    block = MLPRound(4, 3, 8, 2)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.bias.fill_(-90)
    reference = copy.deepcopy(block).double()
    x, q, state = torch.randn(1, 3, 4), torch.randn(1, 3), torch.randn(1, 2, 8)
    weight = torch.tensor([[1., 0., 2.]])
    stats = Statistics(*block(x, q, state, weight))
    # Independent unscaled FP64 formula: no low-precision/log-scale helper.
    h = torch.nn.functional.silu(reference.input(x.double())[:, :, None] +
        reference.state(state.double())[:, None] + reference.query(q.double())[:, None, None] +
        reference.slot[None, None])
    g = reference.gate(h).sigmoid() * weight.double()[:, :, None, None]
    numerator = reference.output((g * h).sum(1))
    mass = g.sum(1)
    torch.testing.assert_close(stats.mean().double(), numerator / mass, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(stats.log_mass().double(), mass.log(), atol=1e-5, rtol=1e-6)
    stats.mean().square().sum().backward()
    (numerator / mass).square().sum().backward()
    for name, parameter in block.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad.double(), expected, atol=2e-6, rtol=2e-4)


def test_extreme_gates_merge_before_normalization_with_full_gradients():
    raw = reader(chunk_size=99)
    for block in raw.blocks:
        with torch.no_grad():
            block.gate.bias.fill_(-90)
    chunked = copy.deepcopy(raw)
    chunked.chunk_size = 2
    chunked.checkpoint_chunks = True
    inputs = [torch.randn(1, 7, 7, dtype=torch.double, requires_grad=True),
              torch.randn(1, 5, dtype=torch.double, requires_grad=True),
              torch.rand(1, 7, dtype=torch.double, requires_grad=True)]
    copies = [t.detach().clone().requires_grad_() for t in inputs]
    a, b = raw(*inputs).tokens, chunked(*copies).tokens
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    a.square().sum().backward()
    b.square().sum().backward()
    for x, y in zip(inputs + list(raw.parameters()), copies + list(chunked.parameters()), strict=True):
        if x.grad is not None:
            assert torch.isfinite(x.grad).all()
            torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
