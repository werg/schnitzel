import copy
import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.compaction import compact_view
from schnitz.readers import SetReader
from schnitz.replay import ReplayTape
from schnitz.training import stored_channel


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
@pytest.mark.parametrize('grouping', ['whole', 'random', 'local', 'overlap'])
def test_grouped_compaction_mass_and_gradients(kind, grouping):
    reader = SetReader(8, 4, 8, width=8, slots=2, rounds=2, kind=kind).double()
    x = torch.randn(1, 7, 8, dtype=torch.double, requires_grad=True)
    weights = torch.rand(1, 7, dtype=torch.double, requires_grad=True)
    q = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
    view = compact_view(x, weights, grouping=grouping, group_size=3)
    torch.testing.assert_close(view.weights.sum(), weights.sum())
    reader(view.values, q, view.weights).tokens.square().sum().backward()
    assert x.grad is not None and weights.grad is not None and q.grad is not None
    assert torch.isfinite(x.grad).all()


def test_overlap_identical_records_preserve_exact_reader_response():
    reader = SetReader(8, 4, 8, width=8, slots=2, rounds=2).double()
    x = torch.randn(1, 1, 8, dtype=torch.double).repeat(1, 9, 1)
    weights = torch.ones(1, 9, dtype=torch.double)
    q = torch.randn(1, 4, dtype=torch.double)
    view = compact_view(x, weights, grouping='overlap', group_size=3)
    torch.testing.assert_close(reader(view.values, q, view.weights).tokens,
                               reader(x, q, weights).tokens, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('grouping', ['random', 'overlap'])
def test_paired_compaction_replay_gradient_parity(tiny_config, grouping):
    config = copy.deepcopy(tiny_config)
    config.memory.compaction = 'synthetic'
    config.memory.compact_records = 1
    config.memory.compaction_objective = 'paired'
    config.memory.compaction_grouping = grouping
    config.memory.compaction_group_size = 2
    config.memory.behavior_kl_weight = 0.2
    config.memory.noise_std = 0.01
    full = SchnitzelAgent(config)
    replay = copy.deepcopy(full)
    state = torch.get_rng_state()
    def run(agent, use_tape):
        torch.set_rng_state(state)
        tape = ReplayTape(verify_outputs=True)
        records = []
        for text in ['A permits retry.', 'B restores state.', 'C has a timeout.']:
            ids = agent.text_ids(text, source=True)
            def producer(ids=ids):
                return stored_channel(agent, agent.produce(ids))
            records.append(tape.capture(agent, producer) if use_tape else producer())
        result = agent(agent.prompt_ids('How to retry?'), agent.target_ids('retry'),
                       records, [0, 1, 2], compact=True)
        result.loss.backward()
        if use_tape:
            tape.backward()
        return result
    a, b = run(full, False), run(replay, True)
    torch.testing.assert_close(a.loss, b.loss)
    assert a.raw_nll is not None and a.compact_nll is not None
    for (_, p), (_, r) in zip(full.named_parameters(), replay.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, r.grad, rtol=1e-4, atol=1e-6)


def test_multiread_replay_and_causal_query_updates(tiny_config):
    config = copy.deepcopy(tiny_config)
    config.memory.read_steps = 3
    config.memory.read_top_k = 1
    config.validate()
    full, replay = SchnitzelAgent(config), None
    replay = copy.deepcopy(full)
    queries = []
    original = full.query
    def record(prompt, memory=None):
        result = original(prompt, memory)
        queries.append((memory is not None, result.detach()))
        return result
    full.query = record
    def run(agent, use_tape):
        tape = ReplayTape(verify_outputs=True)
        records = []
        for text in ['Rule A.', 'Rule B.']:
            ids = agent.text_ids(text, source=True)
            def produce(ids=ids):
                return stored_channel(agent, agent.produce(ids))
            records.append(tape.capture(agent, produce) if use_tape else produce())
        result = agent(agent.prompt_ids('Task'), agent.target_ids('yes'), records, [0, 1])
        result.loss.backward()
        if use_tape:
            tape.backward()
        return result
    a, b = run(full, False), run(replay, True)
    assert a.read_count == 2 and a.selected == [[0, 1]]
    assert queries[0][0] is False and queries[1][0] is True
    assert not torch.equal(queries[0][1], queries[1][1])
    torch.testing.assert_close(a.loss, b.loss)
    for (_, p), (_, r) in zip(full.named_parameters(), replay.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, r.grad, rtol=1e-4, atol=1e-6)
