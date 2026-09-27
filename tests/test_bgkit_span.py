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


def test_space_codec_is_size_agnostic_and_small():
    from sdkb.bgkit_span import SpaceCodec

    codec = SpaceCodec(1024, target_norm=0.8)
    assert sum(p.numel() for p in codec.parameters()) < 5_000_000
    for n, m in ((13, 7), (5, 1), (200, 25)):
        out = codec(torch.randn(n, 1024), m)
        assert out.shape == (m, 1024)
        assert torch.allclose(out.norm(dim=-1), torch.full((m,), 0.8), atol=0.05)


def test_space_codec_starts_as_attention_pooling():
    from sdkb.bgkit_span import SpaceCodec, interface_rms

    codec = SpaceCodec(16, target_norm=1.0, inner=8, heads=2)
    same = torch.randn(1, 16).expand(6, -1)
    out = codec(same, 3)
    assert torch.allclose(out, interface_rms(same[:3], 1.0), atol=1e-5)


def test_combiner_gate_zero_removes_a_record_exactly():
    from sdkb.bgkit_span import SpaceCodec

    codec = SpaceCodec(32, target_norm=1.0, inner=16, heads=2)
    a, b = torch.randn(5, 32), torch.randn(4, 32)
    assert torch.allclose(codec.combine([a], 3), codec(a, 3))
    assert torch.allclose(codec.combine([a, b], 3, torch.tensor([1.0, 0.0])), codec(a, 3),
                          atol=1e-5)
    assert not torch.allclose(codec.combine([a, b], 3), codec(a, 3), atol=1e-4)
    torch.nn.init.normal_(codec.key_record.weight)  # record order now matters
    assert not torch.allclose(codec.combine([a, b], 3), codec.combine([b, a], 3), atol=1e-4)


def test_combiner_gates_scale_mass():
    from sdkb.bgkit_span import SpaceCodec

    codec = SpaceCodec(8, target_norm=1.0, inner=8, heads=1, rounds=1)
    for module in (codec.key_content[1], codec.key_position, codec.query0[2]):
        torch.nn.init.zeros_(module.weight)
        torch.nn.init.zeros_(module.bias)
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    out = codec.combine([a, b], 1, torch.tensor([3.0, 1.0]))
    from sdkb.bgkit_span import interface_rms
    assert torch.allclose(out, interface_rms((3 * a + b) / 4, 1.0), atol=1e-5)
