import copy

import numpy as np
import pytest
import torch

from sdkb.agent import SDKBAgent
from sdkb.data import make_episode
from sdkb.key_index import PublishedKeyIndex
from sdkb.record_gradients import (GradientSink, KeyStateCache, RecordGradients,
                                   refresh_records, writer_backward)
from sdkb.spatial_data import pack_spatial_trajectory
from sdkb.spatial_training import spatial_bank_pipeline_forward
from sdkb.store import DiskStore, StoredRecord
from sdkb.training_bank import TrainingBank

from test_spatial_training import StableChatTokenizer


def test_record_gradients_decay_select_pop_and_evict():
    grads = RecordGradients(0.5, capacity=3)
    grads.add(0, 'a', torch.ones(4), [torch.ones(2), None])
    grads.add(2, 'a', torch.ones(4), [None, torch.ones(3)])
    state, payloads = grads.pop('a', 2)
    assert torch.allclose(state, torch.full((4,), 1.25))  # 1 * 0.5^2 + 1
    assert torch.allclose(payloads[0].float(), torch.full((2,), 0.25))
    assert torch.allclose(payloads[1].float(), torch.ones(3))
    grads.add(0, 'b', torch.full((4,), 3.0), [None, None])
    grads.add(0, 'c', torch.ones(4), [None, None])
    grads.add(0, 'd', torch.full((4,), 2.0), [None, None])
    assert grads.norm('b', 1) == pytest.approx(6 * 0.5)
    # The neighbourhood is ranked by accumulated norm; extras take the largest elsewhere.
    assert grads.select(1, ['c', 'd', 'x'], budget=1, extra=1) == ['d', 'b']
    grads.add(1, 'e', torch.full((4,), 0.1), [None, None])
    assert grads.evict(1) == 1 and 'e' not in grads.updated
    restored = RecordGradients(0.5, capacity=3)
    restored.load_state_dict(grads.state_dict())
    assert restored.norm('b', 1) == grads.norm('b', 1)


def _direct_agent(tiny_config):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.key_interface = 'direct'
    tiny_config.memory.distance_gating = True
    tiny_config.memory.neighbors = [3]
    return tiny_config


def _bank(agent, tmp_path, count=3):
    episodes = [make_episode(index, distractors=3) for index in range(count)]
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    ids = sorted(sources)
    writer_inputs = {record_id: agent.text_ids(sources[record_id].text, source=True)
                     for record_id in ids}
    with torch.no_grad():
        outputs = agent.produce_batch([writer_inputs[i] for i in ids], with_key_state=True)
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put_many(StoredRecord(
        record_id, outputs[0][position], outputs[1][position].bfloat16(),
        namespace='corpus', space='s0', generation='g1', created_at=sources[record_id].created_at,
    ) for position, record_id in enumerate(ids))
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1', spaces=('s0',),
                              expected_sources=len(ids))
    cache = KeyStateCache(ids, outputs[-1].detach().float(), torch.zeros(len(ids), dtype=torch.long))
    return episodes, writer_inputs, store, index, cache


def test_key_state_cache_reproduces_index_and_tracks_versions(tiny_config, tmp_path):
    agent = SDKBAgent(_direct_agent(tiny_config))
    _, writer_inputs, store, index, cache = _bank(agent, tmp_path)
    stored = index.spaces['s0'].keys.copy()
    cache.sync_index(agent, index)
    assert np.allclose(index.spaces['s0'].keys, stored, atol=1e-5)
    with torch.no_grad():
        agent.writer_key_heads[0].bias.add_(0.3)
    cache.sync_index(agent, index)
    assert not np.allclose(index.spaces['s0'].keys, stored, atol=1e-3)
    first = cache.stalest(2)
    bank = TrainingBank(store, DiskStore(tmp_path / 'cache.sqlite'), index)
    report = refresh_records(agent, lambda r: writer_inputs[r], first, bank, cache, 5)
    assert report['refreshed'] == 2 and report['refreshed_age_max'] == 5
    assert report['drift_cosine_min'] > 0.999  # same writer, heads applied to fresh states
    assert set(cache.stalest(2)).isdisjoint(first)


def test_writer_backward_with_one_step_cotangents_matches_direct_backprop(tiny_config):
    agent = SDKBAgent(_direct_agent(tiny_config))
    episodes = [make_episode(index, distractors=1) for index in range(2)]
    texts = [source.text for episode in episodes for source in episode.supports][:3]
    inputs = [agent.text_ids(text, source=True) for text in texts]
    weights = [torch.randn(agent.width), torch.randn(tiny_config.memory.payload_dims[0])]

    def objective(outputs):
        payload = outputs[1].to(torch.bfloat16).float()
        return (outputs[-1].float() @ weights[0]).sum() + (payload @ weights[1]).sum()

    agent.zero_grad(set_to_none=True)
    objective(agent.produce_batch(inputs, with_key_state=True)).backward()
    reference = {n: p.grad.clone() for n, p in agent.named_parameters() if p.grad is not None}
    agent.zero_grad(set_to_none=True)
    cotangents = [(weights[0].clone(), [weights[1].clone()]) for _ in inputs]
    writer_backward(agent, lambda i: inputs[i], [0, 1, 2], cotangents)
    actual = {n: p.grad for n, p in agent.named_parameters() if p.grad is not None}
    assert reference.keys() == actual.keys()
    for name, value in reference.items():
        assert torch.allclose(actual[name], value, atol=1e-5, rtol=1e-4), name


def test_device_search_matches_numpy_reference(tiny_config, tmp_path):
    agent = SDKBAgent(_direct_agent(tiny_config))
    _, _, _, index, _ = _bank(agent, tmp_path, count=4)
    queries = torch.randn(3, index.spaces['s0'].keys.shape[1])
    kwargs = dict(top_k=5, namespace='corpus', space='s0', generation='g1',
                  domains=('research', 'research', 'other'), query_times=(10, 10, 10))
    reference = index.search_batch(queries, **kwargs)
    index.use_device('cpu')
    device = index.search_batch(queries, **kwargs)
    for a, b in zip(reference, device, strict=True):
        assert [s.record_id for s in a.selections] == [s.record_id for s in b.selections]
        assert np.allclose([s.score for s in a.selections], [s.score for s in b.selections],
                           atol=1e-5)
    excluded = index.search_batch(queries[:1], top_k=5, namespace='corpus', space='s0',
                                  generation='g1', domains=('research',), query_times=(10,),
                                  exclude_ids=(frozenset({reference[0].selections[0].record_id}),))
    assert reference[0].selections[0].record_id not in {
        s.record_id for s in excluded[0].selections}


def test_pipeline_sink_collects_key_state_and_payload_cotangents(tiny_config, tmp_path):
    agent = SDKBAgent(_direct_agent(tiny_config))
    episodes, writer_inputs, store, index, cache = _bank(agent, tmp_path)
    cache.sync_index(agent, index)
    rows = [pack_spatial_trajectory(StableChatTokenizer(), [episode], read_slots=2,
                                    generation='g1') for episode in episodes]
    bank = TrainingBank(store, DiskStore(tmp_path / 'cache.sqlite'), index)
    sink = GradientSink(agent, cache)
    result = spatial_bank_pipeline_forward(
        agent, bank, index, copy.deepcopy(rows), limits=(2,), routing_candidates=3,
        microbatch_size=1, inflight=2, sink=sink)
    agent.zero_grad(set_to_none=True)
    result.loss.backward()
    # Key heads now learn from every candidate through the cached states.
    assert agent.writer_key_heads[0].weight.grad.abs().sum() > 0
    grads = RecordGradients(0.9)
    assert sink.harvest(grads, 0, 1) > 0 and len(grads) > 0
    assert any(grads.state[r].numel() for r in grads.updated)
    assert any(grads.payload[r][0] is not None for r in grads.updated)
    assert set(sink.neighborhood) and set(sink.neighborhood) <= set(cache.ids)
    flush = grads.select(0, sink.neighborhood, budget=4, extra=1)
    before = agent.write_slots.grad
    writer_backward(agent, lambda r: writer_inputs[r], flush,
                    [grads.pop(r, 0) for r in flush])
    assert agent.write_slots.grad is not None
    assert before is None or not torch.equal(before, agent.write_slots.grad)
