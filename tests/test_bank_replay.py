import torch

from schnitz.agent import SchnitzelAgent
from schnitz.bank_replay import BankWriterReplay
from schnitz.key_index import PublishedKeyIndex
from schnitz.store import DiskStore, ReadPlan, Selection, StoredRecord
from schnitz.training_bank import TrainingBank


def test_selected_serialized_key_and_payload_replay_then_refresh(tiny_config, tmp_path):
    tiny_config.memory.storage_dtype = 'bfloat16'
    agent = SchnitzelAgent(tiny_config)
    inputs = {'record': torch.tensor([[1, 2, 3]], dtype=torch.long)}
    with torch.no_grad():
        initial = agent.produce_batch([inputs['record']])
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('record', initial[0][0], initial[1][0].bfloat16(),
                          namespace='corpus', space='s0', generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    replay = BankWriterReplay(agent, inputs)
    leaves = replay.capture(('record',))['record']
    assert leaves[1].dtype == torch.float32
    loss = leaves[0].square().sum() + leaves[1].square().mean()
    loss.backward()
    replay.backward()
    assert agent.key_head.weight.grad is not None
    assert agent.value_head[1].weight.grad is not None
    with torch.no_grad():
        agent.key_head.weight.add_(.01 * agent.key_head.weight.grad)
    assert replay.refresh(bank) == 1
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('record', 0.),))
    refreshed = bank.fetch_many((plan,))[0][0]
    assert refreshed.dtype == torch.bfloat16


def test_replay_reuses_a_record_selected_by_later_read_waves(tiny_config):
    agent = SchnitzelAgent(tiny_config)
    inputs = {
        'a': torch.tensor([[1, 2, 3]], dtype=torch.long),
        'b': torch.tensor([[4, 5]], dtype=torch.long),
    }
    replay = BankWriterReplay(agent, inputs)
    first = replay.capture(('a',))
    second = replay.capture(('a', 'b'))
    assert second['a'][0] is first['a'][0]
    assert replay.record_ids == ['a', 'b']
    assert len(replay.tape.records) == 2
    sum(value.square().mean() for values in second.values() for value in values).backward()
    replay.backward()
    assert agent.key_head.weight.grad is not None


def test_replay_can_retain_activations_and_restores_checkpoint_policy(tiny_config):
    tiny_config.model.gradient_checkpointing = True
    agent = SchnitzelAgent(tiny_config)
    inputs = {'a': torch.tensor([[1, 2, 3]], dtype=torch.long)}
    replay = BankWriterReplay(agent, inputs, checkpoint_backward=False)
    leaves = replay.capture(('a',))['a']
    sum(value.square().mean() for value in leaves).backward()
    replay.backward()
    assert agent.backbone.base.gradient_checkpointing is True
    assert agent.key_head.weight.grad is not None
