import torch

from sdkb.bgkit_span import RatioEmbedding, SpanWriter, interface_rms, span_targets


def test_ratio_embedding_starts_at_zero_and_is_smooth():
    emb = RatioEmbedding(16)
    factors = torch.tensor([4.0, 16.0, 128.0])
    assert torch.equal(emb(factors), torch.zeros(3, 16))
    torch.nn.init.normal_(emb.net[-1].weight)
    near = emb(torch.tensor([16.0, 16.5, 64.0]))
    assert (near[0] - near[1]).norm() < (near[0] - near[2]).norm()


def test_marker_is_init_vector_for_every_ratio_at_step_zero():
    init = torch.randn(16)
    writer = SpanWriter(16, target_norm=0.8, marker_init=init)
    out = writer.marker_embedding(torch.tensor([4.0, 32.0]))
    assert torch.allclose(out, init.expand(2, -1))


def test_rep_head_matches_interface_norm():
    writer = SpanWriter(16, target_norm=0.8)
    reps = writer.rep(torch.randn(5, 16) * 10)
    assert torch.allclose(reps.norm(dim=-1), torch.full((5,), 0.8), atol=0.05)
    x = torch.randn(3, 16)
    assert torch.allclose(interface_rms(x, 0.8), x / (x.norm(dim=-1, keepdim=True) + 0.04) * 0.8)


def test_span_targets_emit_then_stop():
    assert span_targets([2, 0, 1]).tolist() == [0, 0, 1, 1, 0, 1]


def test_write_adapter_changes_only_span_positions():
    from sdkb.bgkit_span import attach_write_adapter, span_mask

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.other = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.other(self.q_proj(x))

    layers = torch.nn.ModuleList([Block(), Block()])
    gate, adapters = attach_write_adapter(layers, ('q_proj',), rank=2, alpha=4)
    assert set(adapters) == {'0__q_proj', '1__q_proj'}
    for adapter in adapters.values():
        torch.nn.init.normal_(adapter.up.weight)
    x = torch.randn(2, 5, 8)
    run = lambda: layers[1](layers[0](x))  # noqa: E731
    base = run()
    mask = span_mask([3, 1], [2, 1], 5)
    with gate.active(mask):
        adapted = run()
    assert gate.mask is None
    changed = (adapted - base).abs().sum(-1) > 1e-6
    assert changed.tolist() == [[False, False, False, True, True],
                                [False, True, False, False, False]]
    assert torch.equal(run(), base)


def test_open_adapter_merges_exactly():
    from sdkb.bgkit_span import attach_write_adapter

    layers = torch.nn.ModuleList([torch.nn.Sequential()])
    layers[0].add_module('w1', torch.nn.Linear(6, 4))
    gate, adapters = attach_write_adapter(layers, ('w1',), rank=2, alpha=4)
    assert all(not name.startswith('target') for name, _ in adapters.named_parameters())
    torch.nn.init.normal_(adapters['0__w1'].up.weight)
    x = torch.randn(3, 6)
    with gate.active(torch.ones(3, 1)):
        opened = layers[0](x)
    half = torch.full((3, 1), 0.5)
    with gate.active(half):
        partial = layers[0](x)
    base = layers[0](x)
    assert torch.allclose(partial, (base + opened) / 2, atol=1e-6)
    adapters['0__w1'].merge()
    with gate.active(torch.ones(3, 1)):
        merged = layers[0](x)  # hook removed: the gate no longer adds anything
    assert torch.allclose(merged, opened, atol=1e-5)
