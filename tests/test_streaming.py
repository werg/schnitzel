import pytest
import torch

from schnitz.readers import SetReader
from schnitz.streaming import read_stream
from schnitz.store import DiskStore, StoredRecord, ReadPlan, Selection


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
@pytest.mark.parametrize('n', [0, 1, 11])
def test_stream_matches_materialized_without_retaining_all_payloads(kind, n):
    reader = SetReader(8, 4, 6, width=8, slots=3, rounds=3, kind=kind, chunk_size=3).double().eval()
    x = torch.randn(1, n, 8, dtype=torch.double)
    w = torch.rand(1, n, dtype=torch.double)
    q = torch.randn(1, 4, dtype=torch.double)
    calls = 0
    def chunks():
        nonlocal calls
        calls += 1
        for i in range(0, n, 3):
            yield x[:, i:i+3], w[:, i:i+3]
    expected = reader(x, q, w).tokens
    actual = read_stream(reader, q, chunks).tokens
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert calls == reader.rounds


def test_plan_mutation_and_visibility_fail_closed(tmp_path):
    reader = SetReader(8, 4, 8, width=8, slots=2, rounds=2).eval()
    store = DiskStore(tmp_path / 'store.sqlite')
    for i in range(6):
        store.put(StoredRecord(str(i), torch.ones(4), torch.randn(8)))
    plan = ReadPlan('default', 's0', 'v0', 'research', 10, tuple(Selection(str(i), 0.) for i in range(6)))
    calls = 0
    def chunks():
        nonlocal calls
        calls += 1
        if calls == 2:
            store.delete('default', '0')
        yield from store.iter_fetch(plan, 2)
    with pytest.raises(KeyError):
        read_stream(reader, torch.randn(1,4), chunks)


def test_streamed_session_matches_loaded_session(tmp_path, tiny_config):
    from schnitz.agent import SchnitzelAgent
    from schnitz.data import make_episode
    from schnitz.evaluation import build_shared_bank
    from schnitz.sessions import read_session
    agent = SchnitzelAgent(tiny_config).eval()
    e = make_episode(0, distractors=0)
    store = DiskStore(tmp_path / 'store.sqlite')
    build_shared_bank(agent, store, [e])
    kw = dict(namespace='global', generation='frozen-v1', query_time=e.query_time, oracle_ids=e.required_ids)
    expected = read_session(agent, store, agent.prompt_ids(e.query), **kw).memory
    tiny_config.memory.stream_reads = True
    result = read_session(agent, store, agent.prompt_ids(e.query), **kw).memory
    torch.testing.assert_close(result, expected)


def test_stream_null_policy_uses_multiplicity_not_underflowed_gate():
    reader = SetReader(8, 4, 6, width=8, slots=2, rounds=2).eval()
    with torch.no_grad():
        for block in reader.blocks:
            block.gate.weight.zero_()
            block.gate.bias.fill_(-1000.)
    x, q, w = torch.randn(1, 3, 8), torch.randn(1, 4), torch.ones(1, 3)
    expected = reader(x, q, w).tokens
    actual = read_stream(reader, q, lambda: [(x, w)]).tokens
    torch.testing.assert_close(actual, expected)
