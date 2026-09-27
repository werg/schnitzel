import copy
import json

import numpy as np
import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_episode
from schnitz.key_index import PublishedKeyIndex
from schnitz.keyspace_distillation import (LexicalField, convert_to_direct, field_kl,
                                        lexical_field_loss, recall_summary,
                                        source_disjoint_split, support_ranks,
                                        union_field_loss)
from schnitz.keyspace_views import causal_prefix_text, query_view, source_view
from schnitz.offline_bank import stored_memory_identity
from schnitz.routing_curriculum import RoutingCandidateIndex
from schnitz.spatial_data import pack_spatial_trajectory
from schnitz.spatial_training import spatial_bank_forward
from schnitz.store import DiskStore, StoredRecord
from schnitz.teacher_encoders import pool


class StableChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text, mask = 'B', [0]
        for message in messages:
            rendered = f"<{message['role']}>" + str(message.get('content', ''))
            if message.get('tool_calls'):
                rendered += json.dumps(message['tool_calls'], sort_keys=True)
            rendered += '</>'
            text += rendered
            mask.extend([int(message['role'] == 'assistant')] * len(rendered))
        return {'input_ids': [ord(char) for char in text], 'assistant_masks': mask}


def _spatial(config):
    config.model.loops = 2
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start = 0
    config.model.recurrent_end = 1
    config.memory.read_timing = 'loop_boundary'
    config.memory.payload_dims = [24, 24]
    config.memory.neighbors = [3, 2]
    config.train.routing_weight = .1
    return config


def _pair(config):
    shared = SchnitzelAgent(config)
    direct_config = copy.deepcopy(config)
    direct_config.memory.key_interface = 'direct'
    direct = SchnitzelAgent(direct_config)
    direct.load_state_dict(convert_to_direct(shared.state_dict(), 2), strict=True)
    return shared, direct


def test_direct_heads_reproduce_shared_keys_and_spatial_reads(tiny_config, tmp_path):
    shared, direct = _pair(_spatial(tiny_config))
    ids = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6]])]
    with torch.no_grad():
        before, after = shared.produce_batch(ids), direct.produce_batch(ids)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    row = pack_spatial_trajectory(StableChatTokenizer(), episodes, read_slots=2,
                                  generation='g1', levels=(1, 1))
    store = DiskStore(tmp_path / 'bank.sqlite')
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    for space in ('s0', 's1'):
        store.put_many(StoredRecord(
            record_id, torch.randn(tiny_config.memory.key_dim),
            torch.randn(24, dtype=torch.bfloat16), namespace='corpus', space=space,
            generation='g1', created_at=source.created_at, source_id=record_id)
            for record_id, source in sources.items())
    index = PublishedKeyIndex(store, namespace='corpus', generation='g1',
                              spaces=('s0', 's1'), expected_sources=len(sources))
    results, plans = [], []
    for agent in (shared, direct):
        observed = []
        results.append(spatial_bank_forward(
            agent, store, index, [row], limits=(3, 2), routing_candidates=3,
            plan_observer=lambda call, space, plan, observed=observed: observed.append(
                (call, space, tuple(item.record_id for item in plan.selections)))))
        plans.append(observed)
    assert plans[0] == plans[1]
    torch.testing.assert_close(results[0].loss, results[1].loss, rtol=1e-5, atol=1e-5)
    results[1].loss.backward()
    assert direct.query_key_heads[0].weight.grad.abs().sum() > 0
    assert direct.query_head.weight.grad is not None  # the reader keeps its query
    assert direct.key_head is None and direct.query_maps is None


def test_direct_writer_heads_receive_replayed_key_gradients(tiny_config):
    _, direct = _pair(_spatial(tiny_config))
    outputs = direct.produce_batch([torch.tensor([[1, 2, 3]])])
    (outputs[0].sum() + outputs[2].square().sum()).backward()
    assert direct.writer_key_heads[0].weight.grad.abs().sum() > 0


def test_routing_address_rejects_the_wrong_interface_input(tiny_config):
    shared, direct = _pair(_spatial(tiny_config))
    with pytest.raises(ValueError, match='routing features'):
        direct.routing_address(torch.randn(1, tiny_config.memory.key_dim), 0)
    with pytest.raises(ValueError, match='normalized routing query'):
        shared.routing_address(torch.randn(1, shared.width), 0)


def test_legacy_bank_identity_defaults_to_shared_maps():
    assert stored_memory_identity({'key_dim': 64})['key_interface'] == 'shared_maps'


def test_source_disjoint_split_keeps_shared_sources_together():
    sites = [('a', 'b'), ('b',), ('c',), ('d', 'c'), ('e',)] * 3
    held = source_disjoint_split(sites, fraction=.5, seed='x')
    for left, supports in zip(held, sites, strict=True):
        for right, others in zip(held, sites, strict=True):
            if set(supports) & set(others):
                assert left == right


def test_field_losses_and_ranks_respect_the_eligible_field():
    eligible = torch.tensor([[True, True, False, True]])
    logits = torch.tensor([[2.0, 1.0, 100.0, 0.0]], requires_grad=True)
    assert float(field_kl(logits.detach(), logits, eligible).detach()) == pytest.approx(0, abs=1e-6)
    teacher = torch.tensor([[0.0, 3.0, -50.0, 0.0]])
    assert float(field_kl(teacher, logits, eligible)) > 0
    assert support_ranks(logits.detach(), [(1,)], eligible) == [(2,)]
    loss = union_field_loss([logits, logits * 0], [(0,)], eligible)
    loss.backward()
    assert logits.grad[0, 2] == 0
    with pytest.raises(ValueError, match='outside the causal field'):
        union_field_loss([logits], [(2,)], eligible)
    lexical = torch.tensor([[0.0, 0.9, 5.0, 0.1]])
    assert float(lexical_field_loss([logits], lexical, eligible)) > 0
    summary = recall_summary([[(1,), (9,)], [(4,), (2,)]], (1, 2))
    assert summary['spaces'][0]['any_support_recall'] == .5
    assert summary['union_any_support_recall'] == 1.0


def test_lexical_field_matches_the_curriculum_teacher(tmp_path):
    texts = ['Title: Alpha\nPassage: red apples grow', 'Title: Beta\nPassage: blue sky',
             'Title: Gamma\nPassage: red sky apples apples']
    path = tmp_path / 'sources.jsonl'
    path.write_text(''.join(json.dumps({'record_id': f'r{i}', 'text': text, 'created_at': 1})
                            + '\n' for i, text in enumerate(texts)))
    reference = RoutingCandidateIndex(path)
    field = LexicalField(texts)
    query = 'Which red apples?'
    np.testing.assert_allclose(field.scores([query])[0],
                               reference.lexical_similarities(query, ('r0', 'r1', 'r2')),
                               rtol=1e-5, atol=1e-6)


def test_query_windows_are_causal_and_strip_identifiers():
    class Tokenizer:
        def decode(self, ids):
            return ' '.join(f't{token}' if token < 90 else '0123456789abcdef0123456789abcdef'
                            for token in ids)
    ids = [1, 2, 95, -1, 3, 4, 5]
    text = causal_prefix_text(Tokenizer(), ids, 4, 512)
    assert text.split() == ['t1', 't2', 't3']
    arguments = ('Use the previously stored passages. Give only the short response.\n'
                 'Question: Who wrote it?')
    assert query_view('content', arguments) == 'Question: Who wrote it?'
    rows = {'a': {'text': 'Title: T\nPassage: One. Two.'},
            'b': {'text': 'Title: T\nPassage: Three.'}}
    groups = {'a': ('T#0', (('a', ''), ('b', '')), 'streaming'),
              'b': ('T#0', (('a', ''), ('b', '')), 'streaming')}
    assert source_view('title_lead', 'a', rows, groups) == 'Title: T\nOne.'
    assert source_view('chunk_neighbor', 'b', rows, groups).endswith('\nOne. Two.')


def test_last_token_pooling_handles_left_and_right_padding():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    right = torch.tensor([[1, 1, 0], [1, 1, 1]])
    left = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert pool(hidden, right, 'last')[0].tolist() == hidden[0, 1].tolist()
    assert pool(hidden, left, 'last')[0].tolist() == hidden[0, 2].tolist()
    assert pool(hidden, right, 'mean')[0].tolist() == hidden[0, :2].mean(0).tolist()


def test_standardized_head_is_exact_at_init_and_folds_back():
    from schnitz.keyspace_distillation import StandardizedHead
    states = torch.randn(20, 8) * .1 + 5.0  # nearly collinear, like writer key slots
    weight, bias = torch.randn(3, 8), torch.zeros(3)
    head = StandardizedHead(weight, bias, states.mean(0), states.std(0))
    torch.testing.assert_close(head(head.standardize(states)), states @ weight.T + bias,
                               rtol=1e-4, atol=1e-4)
    with torch.no_grad():
        head.weight.add_(torch.randn_like(head.weight))
        head.bias.add_(1.0)
    folded_weight, folded_bias = head.folded()
    torch.testing.assert_close(states @ folded_weight.T + folded_bias,
                               head(head.standardize(states)), rtol=1e-4, atol=1e-3)


def test_direct_heads_compute_in_fp32_under_autocast(tiny_config):
    _, direct = _pair(_spatial(tiny_config))
    states = torch.randn(2, direct.width)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        keys = direct.writer_space_keys(states)
        address = direct.routing_address(states, 0)
    assert keys[0].dtype == torch.float32 and address.dtype == torch.float32


def test_key_slot_last_isolates_payloads_from_key_training(tiny_config):
    config = _spatial(tiny_config)
    config.memory.key_interface = 'direct'
    config.memory.key_slot_position = 'last'
    config.memory.key_dims = [16, 8]
    config.memory.distance_gating = True
    config.validate()
    agent = SchnitzelAgent(config).eval()
    ids = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6]])]
    with torch.no_grad():
        before = agent.produce_batch(ids)
        agent.write_slots[0].add_(1.0)  # the key-slot embedding
        after = agent.produce_batch(ids)
    assert before[0].shape[-1] == 16 and before[2].shape[-1] == 8
    for index in (1, 3):  # payloads
        torch.testing.assert_close(before[index], after[index])
    assert not torch.allclose(before[0], after[0])
    assert [gate.adjust.in_features for gate in agent.distance_gates] == [16, 8]
    single = agent.produce(ids[0])
    torch.testing.assert_close(single[1], after[1][:1], rtol=1e-4, atol=1e-4)


def test_key_widths_require_direct_heads(tiny_config):
    config = _spatial(tiny_config)
    config.memory.key_dims = [16, 8]
    with pytest.raises(ValueError, match='Per-space key widths'):
        config.validate()


def test_widened_direct_conversion_keeps_folded_rows(tiny_config):
    config = _spatial(tiny_config)
    config.memory.distance_gating = True
    shared = SchnitzelAgent(config)
    direct_config = copy.deepcopy(config)
    direct_config.memory.key_interface = 'direct'
    direct_config.memory.key_dims = [20, 16]
    direct = SchnitzelAgent(direct_config)
    state = convert_to_direct(shared.state_dict(), 2, (20, 16), seed=3)
    direct.load_state_dict(state, strict=True)
    folded = convert_to_direct(shared.state_dict(), 2)
    key_dim = config.memory.key_dim
    torch.testing.assert_close(direct.writer_key_heads[0].weight[:key_dim],
                               folded['writer_key_heads.0.weight'])
    assert direct.writer_key_heads[1].weight.shape == (16, direct.width)
    gate = direct.distance_gates[0].adjust.weight
    assert gate.shape[1] == 20 and float(gate[:, key_dim:].abs().sum()) == 0


def test_key_state_statistics_preserve_every_key_and_address(tiny_config):
    _, direct = _pair(_spatial(tiny_config))
    states = torch.randn(5, direct.width) * .1 + 3.0
    before_keys = direct.writer_space_keys(states)
    before_address = direct.routing_address(states, 1)
    direct.set_key_state_statistics('writer', states.mean(0), states.std(0))
    direct.set_key_state_statistics('query', states.mean(0) * 0.5, states.std(0) * 2)
    for left, right in zip(before_keys, direct.writer_space_keys(states), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(before_address, direct.routing_address(states, 1),
                               rtol=1e-4, atol=1e-4)
