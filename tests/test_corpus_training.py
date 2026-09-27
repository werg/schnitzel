import torch
from dataclasses import asdict
import json
from safetensors.torch import load_file, load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import Episode, Source, save_episodes
from schnitz.offline_bank import ensure_offline_shard, publish_offline_generation
from schnitz.store import DiskStore, lookup_record
from schnitz.training import output_records, stored_channel, train
from schnitz.trajectories import file_sha256


def test_stored_corpus_forward_uses_only_prefix_and_frozen_payloads(tmp_path, tiny_config):
    from schnitz.corpus_training import stored_corpus_forward, stored_corpus_forward_batch
    tiny_config.model.tiny_layers = 3
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 1
    tiny_config.model.recurrent_end = 2
    tiny_config.model.loops = 2
    tiny_config.model.writer_loops = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.payload_dims = [24, 48]
    tiny_config.memory.neighbors = [2, 1]
    tiny_config.memory.read_steps = 1
    tiny_config.train.retrieval = 'learned'
    tiny_config.train.routing_warmup = 0
    tiny_config.train.routing_weight = .1
    tiny_config.train.bank_routing_candidates = 3
    agent = SchnitzelAgent(tiny_config)
    store = DiskStore(tmp_path / 'bank.sqlite')
    sources = [Source(f's{i}', f'Passage {i}: the private marker is {i}x.', 1, 'passage')
               for i in range(3)]
    with torch.no_grad():
        for source in sources:
            values = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
            store.put_many(output_records(agent, source, values, 'corpus', 'g1'))
    episode = Episode('query', 'corpus', (sources[0],), 'What is the marker?', '0x',
                      ('s0',), False, 0, 0, 2, 'passage_qa', (), (('s0',),), 'verified')
    agent.produce = lambda *_: (_ for _ in ()).throw(AssertionError('writer called on read path'))
    searched = []
    class Searcher:
        def search(self, *args, **kwargs):
            searched.append(kwargs['top_k'])
            return store.search(*args, **kwargs)
    first, first_plan = stored_corpus_forward(agent, store, episode, generation='g1',
                                               limits=(2, 1), searcher=Searcher())
    second, second_plan = stored_corpus_forward(agent, store, episode.__class__(
        **(episode.__dict__ | {'answer': 'different target'})), generation='g1', limits=(2, 1))
    other = episode.__class__(**(episode.__dict__ | {
        'episode_id': 'query-two', 'supports': (sources[1],),
        'required_ids': ('s1',), 'sufficient_groups': (('s1',),), 'answer': '1x'}))
    other_result, _ = stored_corpus_forward(agent, store, other,
                                             generation='g1', limits=(2, 1))
    batched, batched_info = stored_corpus_forward_batch(
        agent, store, [episode, other], generation='g1', limits=(2, 1))
    torch.testing.assert_close(batched.nll, torch.stack((first.nll, other_result.nll)).mean(),
                               atol=3e-6, rtol=3e-5)
    assert batched_info['selected_counts'] == [2, 1]
    assert first_plan['selected_ids'] == second_plan['selected_ids']
    assert all('s0' in ids for ids in first_plan['selected_ids'])
    assert first_plan['selected_counts'] == [2, 1]
    assert searched == [3, 3]
    assert torch.isfinite(first.loss) and torch.isfinite(second.loss)
    first.loss.backward()
    assert agent.reader.local[0].blocks[0].input.weight.grad.norm() > 0
    assert agent.reader.local[1].blocks[0].input.weight.grad.norm() > 0
    assert agent.query_maps[0].weight.grad.norm() > 0
    assert agent.query_maps[1].weight.grad.norm() > 0
    agent.zero_grad(set_to_none=True)
    tiny_config.train.bank_payload_contrast_weight = 0.5
    contrasted, contrast_info = stored_corpus_forward(agent, store, episode,
                                                       generation='g1', limits=(2, 1))
    _, contrast_other = stored_corpus_forward(agent, store, episode.__class__(
        **(episode.__dict__ | {'answer': 'different target'})),
        generation='g1', limits=(2, 1))
    assert contrasted.selected == first.selected
    assert contrast_info['selected_ids'] == contrast_other['selected_ids']
    assert contrast_info['swapped_ids'] == contrast_other['swapped_ids']
    assert all(source_id not in ids for source_id, ids in
               zip(contrast_info['swapped_ids'], contrast_info['selected_ids'], strict=True))
    assert torch.isfinite(contrast_info['contrast_loss'])
    assert torch.isfinite(contrast_info['swapped_source_nll'])
    assert contrast_info['contrast_loss'] > 0
    assert contrast_info['swapped_payload_bytes'] == 2 * (24 + 48)
    (contrasted.loss + 0.5 * contrast_info['contrast_loss']).backward()
    assert agent.reader.local[0].blocks[0].input.weight.grad.norm() > 0
    assert agent.reader.local[1].blocks[0].input.weight.grad.norm() > 0
    tiny_config.train.bank_payload_contrast_weight = 0.0
    agent.zero_grad(set_to_none=True)
    stored_reference, _ = stored_corpus_forward(agent, store, episode,
                                                generation='g1', limits=(1, 1))
    stored_reference.nll.backward()
    stored_grad = agent.reader.local[0].blocks[0].input.weight.grad.clone()
    agent.zero_grad(set_to_none=True)
    agent.config.train.retrieval = 'oracle'
    record = []
    for space in ('s0', 's1'):
        entry = lookup_record(store, 's0', namespace='corpus', space=space, generation='g1')
        record.extend((entry.key[None], entry.payload[None].float()))
    direct = agent(agent.prompt_ids(episode.query), agent.target_ids(episode.answer),
                   [tuple(record)], [0])
    direct.nll.backward()
    torch.testing.assert_close(stored_reference.nll, direct.nll, rtol=0, atol=1e-6)
    torch.testing.assert_close(stored_grad, agent.reader.local[0].blocks[0].input.weight.grad,
                               rtol=0, atol=1e-6)


def test_bank_training_warmstarts_exact_writer_and_resumes(tmp_path, tiny_config):
    config = tiny_config
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start = 1
    config.model.recurrent_end = 2
    config.model.loops = 2
    config.model.writer_loops = 1
    config.model.freeze_backbone = True
    config.memory.read_timing = 'loop_boundary'
    config.memory.payload_dims = [24, 48]
    config.memory.neighbors = [2, 1]
    config.train.live_fraction = 1.0
    config.train.steps = 1
    source = Source('source-a', 'Passage: the marker is aqua.', 1, 'passage')
    decoy = Source('source-b', 'Passage: the marker is copper.', 1, 'passage')
    second_decoy = Source('source-c', 'Passage: the marker is silver.', 1, 'passage')
    episodes = [Episode('q-a', 'test', (source,), 'What is the marker?', 'aqua',
                        ('source-a',), False, 0, 0, 2, 'passage_qa', (),
                        (('source-a',),), 'verified')]
    data = tmp_path / 'episodes.jsonl'
    save_episodes(data, episodes)
    config.train.episodes_file = str(data)
    base = tmp_path / 'base'
    train(config, base)
    checkpoint = resolve_checkpoint(base)
    bank_dir = tmp_path / 'bank'
    bank_dir.mkdir()
    store = DiskStore(bank_dir / 'bank.sqlite')
    agent = SchnitzelAgent(config).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'), device='cpu')
    identity = {'writer_checkpoint_sha256': file_sha256(checkpoint / 'model.safetensors'),
                'model': asdict(config.model), 'memory': asdict(config.memory),
                'compute_precision': config.train.precision,
                'max_source_tokens': config.train.max_source_tokens}

    def records():
        with torch.no_grad():
            for item in (source, decoy, second_decoy):
                values = stored_channel(agent, agent.produce(agent.text_ids(item.text, source=True)))
                yield from output_records(agent, item, values, 'corpus', 'g1')

    ensure_offline_shard(store, records, identity=identity, namespace='corpus',
                         generation='g1', spaces=('s0', 's1'), shard_id='000',
                         source_ids=('source-a', 'source-b', 'source-c'))
    manifest = publish_offline_generation(store, identity=identity, namespace='corpus',
                                          generation='g1', spaces=('s0', 's1'),
                                          shard_ids=('000',), source_count=3)
    (bank_dir / 'manifest.json').write_text(json.dumps(manifest))
    config.train.bank_dir = str(bank_dir)
    config.train.bank_read_limits = [2, 1]
    config.train.retrieval = 'learned'
    config.train.live_fraction = 0.0
    config.train.bank_payload_contrast_weight = 0.5
    config.train.sampling_policy = 'shuffled_passes'
    stage = tmp_path / 'bank-stage'
    result = train(config, stage, init_from=base)
    assert result['steps'] == 1
    first_metric = json.loads((stage / 'metrics.jsonl').read_text().splitlines()[0])
    assert first_metric['payload_contrast_loss'] > 0
    assert first_metric['bank']['swapped_payload_bytes'] == 2 * (24 + 48)
    assert abs(first_metric['optimization_loss'] - (
        first_metric['loss'] + 0.5 * first_metric['payload_contrast_loss'])) < 1e-6
    assert DiskStore(stage / 'training_cache.sqlite').sizes()['records'] == 0
    config.train.steps = 2
    resumed = train(config, stage, resume=True)
    assert resumed['steps'] == 2
    uninterrupted = tmp_path / 'uninterrupted'
    train(config, uninterrupted, init_from=base)
    resumed_weights = load_file(str(resolve_checkpoint(stage) / 'model.safetensors'))
    full_weights = load_file(str(resolve_checkpoint(uninterrupted) / 'model.safetensors'))
    for name in full_weights:
        torch.testing.assert_close(resumed_weights[name], full_weights[name], rtol=0, atol=0)
