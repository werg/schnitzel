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
