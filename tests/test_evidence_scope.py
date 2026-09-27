from dataclasses import replace

import pytest

from schnitz.agent import SchnitzelAgent
from schnitz.data import make_multiuse_world, evidence_ids, save_episodes
from schnitz.evaluation import build_shared_bank, stored_transfer_evaluation
from schnitz.store import DiskStore
from schnitz.training import train


def test_available_evidence_is_not_a_sufficiency_label_or_future_access():
    e = next(e for e in make_multiuse_world(3, bindings=2)
             if e.task_family == 'multiuse/action' and e.answer != 'STOP')
    assert len(evidence_ids(e, 'required')) == 2
    assert len(evidence_ids(e, 'available')) == 4
    assert e.sufficient_groups == (e.required_ids,)
    future = replace(e.supports[0], created_at=e.query_time)
    with pytest.raises(ValueError, match='causal'):
        evidence_ids(replace(e, supports=(future,) + e.supports[1:]), 'available')


def test_available_scope_reaches_training_and_stored_controls(tmp_path, tiny_config, monkeypatch):
    tiny_config.train = replace(tiny_config.train, evidence_scope='available', steps=1, gradient_accumulation=1)
    e = next(e for e in make_multiuse_world(3, bindings=2)
             if e.task_family == 'multiuse/action' and e.answer != 'STOP')
    path = tmp_path / 'episodes.jsonl'
    save_episodes(path, [e])
    tiny_config.train.episodes_file = str(path)
    seen, forward = [], SchnitzelAgent.forward
    def capture(self, prompt, target, records, required, **kwargs):
        seen.append(tuple(required))
        return forward(self, prompt, target, records, required, **kwargs)
    monkeypatch.setattr(SchnitzelAgent, 'forward', capture)
    train(tiny_config, tmp_path / 'run')
    assert seen == [(0, 1, 2, 3)]
    agent = SchnitzelAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, [e])
    def forbidden(*args, **kwargs):
        raise AssertionError('Writer called during stored evaluation')
    monkeypatch.setattr(agent, 'produce', forbidden)
    result = stored_transfer_evaluation(agent, store, [e], drop_supports=True)
    rows = {r['condition']: r for r in result['rows']}
    assert len(rows['all']['selected_ids'][0]) == 4
    assert rows['all']['selected_ids'] == rows['zero_values']['selected_ids']
    assert all(len(rows[c]['selected_ids'][0]) == 3 and not rows[c]['complete_support']
               for c in ('drop_0', 'drop_1'))
    prompts, prompt_ids = [], agent.prompt_ids
    def capture_prompt(query, text=''):
        prompts.append(text)
        return prompt_ids(query, text)
    monkeypatch.setattr(agent, 'prompt_ids', capture_prompt)
    agent.config.train.arm = 'oracle_text'
    stored_transfer_evaluation(agent, store, [e])
    assert all(s.text in prompts[0] for s in e.supports)


def test_available_scope_does_not_turn_routing_negatives_into_positives(tiny_config):
    tiny_config.train = replace(tiny_config.train, evidence_scope='available', retrieval='learned')
    with pytest.raises(ValueError, match='oracle'):
        tiny_config.validate()
