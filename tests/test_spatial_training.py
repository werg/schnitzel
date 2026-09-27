import copy
import json
import threading

import torch

from schnitz.agent import SchnitzelAgent
from schnitz.bank_replay import BankWriterReplay
from schnitz.data import make_episode
from schnitz.key_index import PublishedKeyIndex
from schnitz.routing_curriculum import RoutingCandidateIndex
from schnitz.spatial_data import pack_spatial_trajectory
from schnitz.spatial_training import spatial_bank_forward, spatial_bank_pipeline_forward
from schnitz.store import DiskStore, StoredRecord
from schnitz.training_bank import TrainingBank


class StableChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text, mask = "B", [0]
        for message in messages:
            rendered = f"<{message['role']}>" + str(message.get("content", ""))
            if message.get("tool_calls"):
                rendered += json.dumps(message["tool_calls"], sort_keys=True)
            rendered += "</>"
            text += rendered
            mask.extend([int(message["role"] == "assistant")] * len(rendered))
        return {"input_ids": [ord(char) for char in text], "assistant_masks": mask}


def test_spatial_bank_forward_reads_all_sites_without_writer(tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = "middle_block"
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = "loop_boundary"
    tiny_config.memory.neighbors = [3]
    tiny_config.train.routing_weight = .1
    agent = SchnitzelAgent(tiny_config)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    row = pack_spatial_trajectory(StableChatTokenizer(), episodes, read_slots=2,
                                  generation="g1", levels=(1, 1))
    store = DiskStore(tmp_path / "bank.sqlite")
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    store.put_many(StoredRecord(
        record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace="corpus", space="s0", generation="g1", domain="research",
        created_at=source.created_at, source_id=record_id,
    ) for record_id, source in sources.items())
    index = PublishedKeyIndex(store, namespace="corpus", generation="g1",
                              spaces=("s0",), expected_sources=len(sources))
    agent.produce = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("spatial stored reads must not call the writer"))

    observed = []
    result = spatial_bank_forward(
        agent, store, index, [row], limits=(3,), routing_candidates=3,
        plan_observer=lambda call_id, space, plan: observed.append(
            (call_id, space, tuple(item.record_id for item in plan.selections))),
    )
    assert result.loss.isfinite() and result.metrics["read_sites"] == 2
    assert result.metrics["selected_counts"] == [3]
    assert len(observed) == 2
    assert {item[0] for item in observed} == {site["call_id"] for site in row["sites"]}
    assert all(item[1] == "s0" and len(item[2]) == 3 for item in observed)
    result.loss.backward()
    assert agent.query_maps[0].weight.grad is not None
    assert agent.query_maps[0].weight.grad.abs().sum() > 0
    assert next(agent.reader.parameters()).grad is not None


def test_routing_curriculum_keeps_reader_selection_and_trains_address(tiny_config,
                                                                     tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.neighbors = [2]
    agent = SchnitzelAgent(tiny_config)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    row = pack_spatial_trajectory(StableChatTokenizer(), episodes, read_slots=2,
                                  generation='g1', levels=(1, 1))
    row['routing_step'] = 0
    for site, episode in zip(row['sites'], episodes, strict=True):
        site['routing_query_text'] = episode.query
    sources = {source.record_id: source for episode in episodes
               for source in episode.supports}
    manifest = tmp_path / 'sources.jsonl'
    manifest.write_text(''.join(json.dumps({
        'record_id': source.record_id, 'text': source.text,
        'created_at': source.created_at, 'domain': 'research'}) + '\n'
        for source in sources.values()))
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put_many(StoredRecord(
        record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace='corpus', space='s0', generation='g1', domain='research',
        created_at=source.created_at, source_id=record_id,
    ) for record_id, source in sources.items())
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1',
                              spaces=('s0',), expected_sources=len(sources))
    baseline = spatial_bank_pipeline_forward(
        agent, store, index, [row], limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2)
    corrected = spatial_bank_pipeline_forward(
        agent, store, index, [row], limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2,
        routing_teacher=RoutingCandidateIndex(manifest))
    assert corrected.metrics['selected_counts'] == baseline.metrics['selected_counts']
    assert corrected.metrics['lexical_alignment'] >= 0
    assert corrected.metrics['routing_easy_weight'] == 1.0
    assert corrected.metrics['routing_hard_weight'] == 0.0
    assert corrected.metrics['routing_easy_loss'] >= 0
    assert corrected.metrics['routing_hard_loss'] >= 0
    assert corrected.routing.isfinite()
    corrected.loss.backward()
    assert agent.query_maps[0].weight.grad.abs().sum() > 0


def test_stored_routing_is_stable_when_live_writer_keys_drift(tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.neighbors = [2]
    tiny_config.train.key_stability_weight = .2
    tiny_config.train.writer_replay_records_per_site = 2
    agent = SchnitzelAgent(tiny_config)
    episode = make_episode(0, distractors=0)
    row = pack_spatial_trajectory(StableChatTokenizer(), [episode],
                                  read_slots=2, generation='g1')
    store = DiskStore(tmp_path / 'bank.sqlite')
    sources = {source.record_id: source for source in episode.supports}
    store.put_many(StoredRecord(
        rid, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace='corpus', space='s0', generation='g1', domain='research',
        created_at=source.created_at, source_id=rid,
    ) for rid, source in sources.items())
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1',
                              spaces=('s0',), expected_sources=len(sources))
    manifest = tmp_path / 'sources.jsonl'
    manifest.write_text(''.join(json.dumps({
        'record_id': source.record_id, 'text': source.text,
        'created_at': source.created_at, 'domain': 'research'}) + '\n'
        for source in sources.values()))
    row['sites'][0]['routing_query_text'] = episode.query
    teacher = RoutingCandidateIndex(manifest)
    stored = spatial_bank_pipeline_forward(
        agent, store, index, [row], limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2, routing_teacher=teacher)
    leaves = {}

    class FakeReplay:
        def capture(self, record_ids):
            result = {}
            for record_id in record_ids:
                old = index.keys_for_ids('s0', (record_id,), domain='research',
                                         query_time=episode.query_time)
                key = (-old if record_id == next(iter(sources)) else old).requires_grad_()
                leaves[record_id] = key
                result[record_id] = (key, torch.zeros(
                    1, tiny_config.memory.payload_dims[0]))
            return result

    live = spatial_bank_pipeline_forward(
        agent, store, index, [row], limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2, routing_teacher=teacher,
        writer_replay=FakeReplay())
    assert abs(live.metrics['routing_stored_loss']
               - stored.metrics['routing_stored_loss']) < 1e-6
    assert live.metrics['key_stability_loss'] > .9
    live.loss.backward()
    assert any(key.grad is not None and key.grad.abs().sum() > 0
               for key in leaves.values())

    # The auxiliary objective must train replayed writer keys even when neither
    # key stability nor distance gates can supply that gradient.
    agent.zero_grad(set_to_none=True)
    agent.config.train.key_stability_weight = 0
    agent.config.train.routing_live_weight = .5
    agent.config.memory.distance_gating = False
    leaves.clear()
    live_aux = spatial_bank_pipeline_forward(
        agent, store, index, [row], limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2, routing_teacher=teacher,
        writer_replay=FakeReplay())
    assert abs(live_aux.metrics['routing_stored_loss']
               - stored.metrics['routing_stored_loss']) < 1e-6
    assert abs(live_aux.metrics['routing_live_loss']
               - live_aux.metrics['routing_stored_loss']) > 1e-4
    live_aux.loss.backward()
    assert any(key.grad is not None and key.grad.abs().sum() > 0
               for key in leaves.values())


def test_spatial_forward_projects_only_supervised_positions(tiny_config, tmp_path, monkeypatch):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = "middle_block"
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = "loop_boundary"
    agent = SchnitzelAgent(tiny_config)
    episode = make_episode(0, distractors=0)
    row = pack_spatial_trajectory(StableChatTokenizer(), [episode], read_slots=2,
                                  generation="g1")
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put_many(StoredRecord(
        source.record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace="corpus", space="s0", generation="g1", created_at=source.created_at,
    ) for source in episode.supports)
    index = PublishedKeyIndex(store, namespace="corpus", generation="g1", spaces=("s0",),
                              expected_sources=len(episode.supports))
    original, projected = agent.backbone.logits, []

    def observe(hidden):
        projected.append(hidden.shape)
        return original(hidden)

    monkeypatch.setattr(agent.backbone, "logits", observe)
    result = spatial_bank_forward(agent, store, index, [row], limits=(2,),
                                  routing_candidates=2)
    assert projected == [(row["supervised_tokens"], agent.width)]
    assert result.metrics["supervised_tokens"] == row["supervised_tokens"]


def test_integrated_write_states_depend_on_earlier_read_results(tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.writer_loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    agent = SchnitzelAgent(tiny_config)
    episode = make_episode(0, distractors=0)
    row = pack_spatial_trajectory(
        StableChatTokenizer(), [episode], read_slots=2, generation='g1',
        include_writes=True, write_generation='g2',
        write_slots=tiny_config.memory.write_slots)
    stores = []
    key_rows = {source.record_id: torch.randn(tiny_config.memory.key_dim)
                for source in episode.supports}
    for variant in range(2):
        store = DiskStore(tmp_path / f'bank-{variant}.sqlite')
        store.put_many(StoredRecord(
            source.record_id, key_rows[source.record_id],
            torch.zeros(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16)
            if variant == 0 else torch.randn(tiny_config.memory.payload_dims[0]).bfloat16(),
            namespace='corpus', space='s0', generation='g1',
            created_at=source.created_at,
        ) for source in episode.supports)
        index = PublishedKeyIndex(store, namespace='corpus', generation='g1',
                                  spaces=('s0',), expected_sources=len(episode.supports))
        stores.append(spatial_bank_forward(
            agent, store, index, [row], limits=(2,), routing_candidates=2).write_outputs)
    assert stores[0] is not None and stores[0][0].shape == (1, tiny_config.memory.key_dim)
    assert stores[0][1].shape == (1, tiny_config.memory.payload_dims[0])
    assert not torch.equal(stores[0][1], stores[1][1])


def test_spatial_distance_gates_use_density_and_receive_task_gradients(tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.neighbors = [3]
    tiny_config.memory.distance_gating = True
    tiny_config.memory.gate_density_k = 2
    tiny_config.train.support_gate_floor = .1
    agent = SchnitzelAgent(tiny_config)
    episode = make_episode(0, distractors=1)
    row = pack_spatial_trajectory(StableChatTokenizer(), [episode], read_slots=2,
                                  generation='g1')
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put_many(StoredRecord(
        source.record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace='corpus', space='s0', generation='g1', created_at=source.created_at,
    ) for source in episode.supports)
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1', spaces=('s0',),
                              expected_sources=len(episode.supports))
    result = spatial_bank_forward(agent, store, index, [row], limits=(3,),
                                  routing_candidates=3)
    assert result.metrics['gate_mass'][0] > 0
    assert 0 < result.metrics['gate_effective_records'][0] <= 3
    result.loss.backward()
    assert agent.distance_gates[0].adjust.weight.grad.abs().sum() > 0


def test_inflight_microbatches_match_reference_gradients_and_overlap_reads(
        tiny_config, tmp_path, monkeypatch):
    torch.manual_seed(7)
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = "middle_block"
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = "loop_boundary"
    tiny_config.memory.neighbors = [4]
    tiny_config.train.routing_weight = .1
    reference = SchnitzelAgent(tiny_config)
    pipelined = copy.deepcopy(reference)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    rows = [pack_spatial_trajectory(
        StableChatTokenizer(), [episode], read_slots=2, generation="g1",
    ) for episode in episodes]
    store = DiskStore(tmp_path / "bank.sqlite")
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    store.put_many(StoredRecord(
        record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace="corpus", space="s0", generation="g1", domain="research",
        created_at=source.created_at, source_id=record_id,
    ) for record_id, source in sources.items())
    index = PublishedKeyIndex(store, namespace="corpus", generation="g1",
                              spaces=("s0",), expected_sources=len(sources))

    expected = spatial_bank_forward(
        reference, store, index, rows, limits=(4,), routing_candidates=4)
    expected.loss.backward()

    original_fetch = store.fetch_many
    barrier = threading.Barrier(2)

    def synchronized_fetch(plans):
        barrier.wait(timeout=3)
        return original_fetch(plans)

    monkeypatch.setattr(store, "fetch_many", synchronized_fetch)
    actual = spatial_bank_pipeline_forward(
        pipelined, store, index, rows, limits=(4,), routing_candidates=4,
        microbatch_size=1, inflight=2)
    actual.loss.backward()
    torch.testing.assert_close(expected.loss, actual.loss, atol=2e-5, rtol=2e-4)
    assert actual.metrics["pipeline_microbatches"] == 2
    assert actual.metrics["pipeline_inflight"] == 2
    assert actual.metrics["pipeline_max_pending"] == 2
    for name in ("pipeline_retrieval_seconds_p50", "pipeline_retrieval_seconds_p95",
                 "pipeline_retrieval_seconds_p99", "pipeline_ready_queue_seconds_p95",
                 "pipeline_blocked_seconds"):
        assert actual.metrics[name] >= 0
    for (name, first), (other, second) in zip(
            reference.named_parameters(), pipelined.named_parameters(), strict=True):
        assert name == other
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-5, rtol=2e-4,
                                       msg=name)


def test_spatial_pipeline_replays_selected_writer_keys_and_payloads(
        tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.neighbors = [2]
    tiny_config.memory.distance_gating = True
    tiny_config.train.writer_replay_records_per_site = 2
    agent = SchnitzelAgent(tiny_config)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    rows = [pack_spatial_trajectory(
        StableChatTokenizer(), [episode], read_slots=2, generation='g1')
        for episode in episodes]
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    writer_inputs = {record_id: agent.text_ids(source.text, source=True)
                     for record_id, source in sources.items()}
    with torch.no_grad():
        outputs = agent.produce_batch(list(writer_inputs.values()))
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put_many(StoredRecord(
        record_id, outputs[0][position], outputs[1][position].bfloat16(),
        namespace='corpus', space='s0', generation='g1', created_at=source.created_at,
    ) for position, (record_id, source) in enumerate(sources.items()))
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1', spaces=('s0',),
                              expected_sources=len(sources))
    training_bank = TrainingBank(store, DiskStore(tmp_path / 'cache.sqlite'), index)
    replay = BankWriterReplay(agent, writer_inputs)
    result = spatial_bank_pipeline_forward(
        agent, training_bank, index, rows, limits=(2,), routing_candidates=2,
        microbatch_size=1, inflight=2, writer_replay=replay)
    assert len(replay.tape.records) == 1
    assert len(replay.record_ids) == len(set(replay.record_ids))
    result.loss.backward()
    replay.backward()
    assert replay.record_ids
    assert agent.key_head.weight.grad is not None
    assert agent.value_head[1].weight.grad is not None
    assert agent.distance_gates[0].adjust.weight.grad is not None


def test_pipeline_forced_gold_exploration_and_spreading_terms(tiny_config, tmp_path):
    from schnitz.key_geometry import BankLoad
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.neighbors = [3]
    tiny_config.memory.distance_gating = True
    tiny_config.train.writer_replay_records_per_site = 3
    tiny_config.train.routing_live_weight = 1.0
    agent = SchnitzelAgent(tiny_config)
    episodes = [make_episode(index, distractors=3) for index in range(3)]
    rows = [pack_spatial_trajectory(
        StableChatTokenizer(), [episode], read_slots=2, generation='g1')
        for episode in episodes]
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    writer_inputs = {record_id: agent.text_ids(source.text, source=True)
                     for record_id, source in sources.items()}
    with torch.no_grad():
        outputs = agent.produce_batch(list(writer_inputs.values()))
    store = DiskStore(tmp_path / 'bank.sqlite')
    store.put_many(StoredRecord(
        record_id, outputs[0][position], outputs[1][position].bfloat16(),
        namespace='corpus', space='s0', generation='g1', created_at=source.created_at,
    ) for position, (record_id, source) in enumerate(sources.items()))
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1', spaces=('s0',),
                              expected_sources=len(sources))

    def run(probability, microbatch=1, **train):
        for name, value in train.items():
            setattr(agent.config.train, name, value)
        batch = copy.deepcopy(rows)
        for row in batch:
            row['gold_force_probability'] = probability
            row['training_step'] = 3
        bank = TrainingBank(store, DiskStore(tmp_path / f'cache-{probability}.sqlite'), index)
        load = BankLoad(next(iter(index.spaces.values())).ids, 1, decay=.5, threshold=1.5)
        load.begin_step()
        replay = BankWriterReplay(agent, writer_inputs)
        result = spatial_bank_pipeline_forward(
            agent, bank, index, batch, limits=(2,), routing_candidates=3,
            microbatch_size=microbatch, inflight=2, writer_replay=replay, load=load)
        return result, load, replay

    never, load, _ = run(0.0)
    assert never.metrics['gold_forced_fraction'] == 0
    always, _, _ = run(1.0)
    assert always.metrics['gold_forced_fraction'] == 1
    # Forced reads always contain every support; unassisted recall is unchanged.
    assert always.metrics['learned_positive_recall'] == never.metrics['learned_positive_recall']
    assert always.metrics['gate_support_share'][0] >= never.metrics['gate_support_share'][0]
    for group in ('forced_gold', 'retrieved_gold', 'other'):
        assert f'gate_slope_{group}' in always.metrics
    assert never.metrics['gate_weight_forced_gold'] == [None]
    assert load.pending[0] and not load.loads[0].counts.any()
    load.commit()
    assert load.loads[0].counts.sum() > 0

    spread, spread_load, replay = run(
        1.0, microbatch=2, spread_variance_weight=.1, spread_covariance_weight=.1,
        koleo_weight=.1, load_penalty_weight=.1, exploration_fraction=.5)
    assert spread.geometry is not None and spread.geometry.isfinite()
    assert spread.metrics['koleo'] != 0 and spread.metrics['spread_variance'] > 0
    agent.zero_grad(set_to_none=True)
    spread.geometry.backward(retain_graph=True)
    replay.backward()
    # Query-address spreading reaches the query side; KoLeo reaches live writer keys.
    assert agent.query_maps[0].weight.grad.abs().sum() > 0
    assert agent.key_head.weight.grad.abs().sum() > 0
