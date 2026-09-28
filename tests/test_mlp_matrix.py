import torch

from schnitz.mlp_matrix import MLPMatrix


def _operator(**kwargs):
    torch.manual_seed(0)
    return MLPMatrix({'a': 6, 'b': 10}, 8, state=16, hidden=12, layers=2, frequencies=3, **kwargs)


def test_any_input_and_output_size():
    op = _operator()
    for n_a, n_b, m in ((1, 1, 1), (7, 3, 5), (2, 9, 13)):
        out, mass = op([('a', torch.randn(n_a, 6), 1.0), ('b', torch.randn(n_b, 10), 0.5)], m)
        assert out.shape == (m, 8)
        assert torch.isclose(mass, torch.tensor(1.5))


def test_gate_zero_removes_an_item_exactly():
    op = _operator()
    a, b = torch.randn(4, 6), torch.randn(3, 10)
    alone, _ = op([('a', a, 1.0)], 5)
    gated, _ = op([('a', a, 1.0), ('b', b, 0.0)], 5)
    assert torch.allclose(alone, gated, atol=1e-6)


def test_gates_only_modulate_mass():
    op = _operator()
    items = [('a', torch.randn(4, 6), 1.0), ('b', torch.randn(3, 10), 0.4)]
    out, mass = op(items, 5)
    scaled, scaled_mass = op([(k, x, 3.0 * g) for k, x, g in items], 5)
    assert torch.allclose(out, scaled, atol=1e-5)
    assert torch.isclose(scaled_mass, 3.0 * mass)
    other, _ = op([items[0], ('b', items[1][1], 0.9)], 5)
    assert not torch.allclose(out, other, atol=1e-4)


def test_item_order_does_not_matter():
    op = _operator()
    a1, a2 = torch.randn(4, 6), torch.randn(2, 6)
    forward, _ = op([('a', a1, 1.0), ('a', a2, 1.0)], 3)
    backward, _ = op([('a', a2, 1.0), ('a', a1, 1.0)], 3)
    assert torch.allclose(forward, backward, atol=1e-5)


def test_each_item_weighs_its_gate_whatever_its_length():
    op = _operator()
    seen = {}
    layer = op.layers[0]
    original = layer.forward

    def spy(h, sources, source_pos, relative, weights, target_pos, cond, locality=None,
            extra=None):
        seen['weights'] = weights
        return original(h, sources, source_pos, relative, weights, target_pos, cond, locality,
                        extra)

    layer.forward = spy
    op([('a', torch.randn(8, 6), 1.0), ('b', torch.randn(2, 10), 0.5)], 3)
    w = seen['weights']
    assert torch.isclose(w[:8].sum(), torch.tensor(1.0)) and torch.isclose(w[8:].sum(), torch.tensor(0.5))


def test_condition_changes_the_output_and_gradients_reach_every_part():
    op = _operator(cond=4, out_norm=2.0)
    items = [('a', torch.randn(4, 6, requires_grad=True), 1.0), ('b', torch.randn(3, 10), 1.0)]
    c1, c2 = torch.randn(4), torch.randn(4)
    out1, _ = op(items, 5, cond=c1)
    out2, _ = op(items, 5, cond=c2)
    assert not torch.allclose(out1, out2, atol=1e-4)
    assert torch.allclose(out1.norm(dim=-1), torch.full((5,), 2.0 / 1.05), atol=0.2)
    out1.pow(2).sum().backward()
    assert items[0][1].grad is not None and items[0][1].grad.abs().sum() > 0
    missing = [n for n, p in op.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert all(n.split('.')[3] == 'b' for n in missing if '.source.' in n), missing


def test_checkpointed_layers_give_identical_gradients():
    plain = _operator()
    wrapped = _operator(checkpoint_layers=True)
    wrapped.load_state_dict(plain.state_dict())
    x = torch.randn(5, 6)
    grads = []
    for op in (plain, wrapped):
        out, _ = op([('a', x, 1.0)], 4)
        out.sum().backward()
        grads.append([p.grad.clone() for p in op.parameters() if p.grad is not None])
    for g1, g2 in zip(*grads):
        assert torch.allclose(g1, g2, atol=1e-6)


def test_locality_kernel_lets_a_target_read_its_aligned_source():
    torch.manual_seed(0)
    op = MLPMatrix({'a': 4}, 4, state=8, hidden=8, layers=1, frequencies=2)
    x = torch.randn(10, 4)
    base, _ = op([('a', x, 1.0)], 10)
    x2 = x.clone()
    x2[0] += 5.0   # change the first source position only
    changed, _ = op([('a', x2, 1.0)], 10)
    effect = (changed - base).norm(dim=-1)
    assert effect[0] > 5 * effect[-1]   # the aligned target moves, the far one barely
    for p in op.parameters():
        p.requires_grad_(False)
    flat = MLPMatrix({'a': 4}, 4, state=8, hidden=8, layers=1, frequencies=2, relative=False)
    assert flat.layers[0].bandwidth is None


def test_content_is_not_drowned_by_position_at_small_input_scale():
    """Decoder-space spans have a per-dimension RMS near 0.025; at initialization
    the output must still depend on the content about as much as on its scale-free
    form (the input normalization makes the operator scale invariant)."""
    torch.manual_seed(0)
    op = MLPMatrix({'span': 64}, 16, state=32, hidden=16, layers=2)
    x = torch.randn(10, 64)
    small, unit = op([('span', 0.025 * x, 1.0)], 5)[0], op([('span', x, 1.0)], 5)[0]
    assert (small - unit).norm() < 0.05 * unit.norm()  # up to the LayerNorm epsilon
    other = op([('span', 0.025 * torch.randn(10, 64), 1.0)], 5)[0]
    assert (small - other).norm() > 0.1 * small.norm()
