import torch
from torch import nn

from schnitz.span_protocol import ProtocolTokens, untie
from schnitz.span_tokens import SPAN_TOKENS


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.lm_head = nn.Linear(8, 64, bias=False)
        self.lm_head.weight = self.embed.weight   # tied
        self.config = type('C', (), {'tie_word_embeddings': True})()

    def get_input_embeddings(self):
        return self.embed


def _setup():
    torch.manual_seed(0)
    lm = TinyLM()
    before = lm.lm_head(lm.embed(torch.arange(40))).detach()
    untie(lm)
    ratio = nn.Linear(1, 8)
    proto = ProtocolTokens(lm.embed, lm.lm_head, lambda f: ratio(f.reshape(1)).squeeze(0))
    proto.install(lm.embed, lm.lm_head)
    return lm, proto, before


def test_untie_and_install_keep_the_function():
    lm, proto, before = _setup()
    assert lm.lm_head.weight.data_ptr() != lm.embed.weight.data_ptr()
    assert torch.allclose(lm.lm_head(lm.embed(torch.arange(40))), before, atol=1e-6)


def test_protocol_rows_are_their_own_parameters():
    lm, proto, _ = _setup()
    mem = SPAN_TOKENS['mem'][1]
    with torch.no_grad():
        proto.inputs[proto.index('mem')] = 3.0
        proto.outputs[proto.index('mem')] = 0.0
    assert torch.allclose(lm.embed(torch.tensor([mem, 5]))[0], torch.full((8,), 3.0))
    logits = lm.lm_head(torch.randn(2, 8))
    assert torch.all(logits[:, mem] == 0)
    # gradients reach the protocol tables, not the frozen matrix rows
    lm.embed.weight.requires_grad_(False)
    lm.lm_head.weight.requires_grad_(False)
    lm.lm_head(lm.embed(torch.tensor([mem, SPAN_TOKENS['bg'][1]]))).sum().backward()
    assert proto.inputs.grad.abs().sum() > 0 and proto.outputs.grad.abs().sum() > 0


def test_span_logits_and_marker():
    _, proto, _ = _setup()
    h = torch.randn(3, 8)
    out = proto.span_logits(h)
    assert out.shape == (3, 2)
    m1, m2 = proto.marker(torch.tensor(4.0)), proto.marker(torch.tensor(16.0))
    assert m1.shape == (8,) and not torch.allclose(m1, m2)
