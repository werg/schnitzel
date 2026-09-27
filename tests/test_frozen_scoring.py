import pytest
import torch


def test_frozen_cache_requires_exact_inputs_and_does_not_mutate_scores(tiny_config, monkeypatch):
    from schnitz.agent import SchnitzelAgent
    from schnitz.frozen_scoring import FrozenScorer
    agent = SchnitzelAgent(tiny_config).eval().requires_grad_(False)
    cache = FrozenScorer(agent)
    prompt = agent.prompt_ids('Question')
    calls = []
    original = agent.conditioned_nll
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(agent, 'conditioned_nll', counted)
    first = cache.score(prompt, None, '0', ('0', '1'))
    assert len(calls) == 2
    first['choice_sequence_nll']['0'] = -100.
    second = cache.score(prompt.clone(), None, '0', ('0', '1'))
    assert len(calls) == 2 and second['choice_sequence_nll']['0'] >= 0
    cache.score(prompt, None, '1', ('0', '1'))
    assert len(calls) == 4
    memory = torch.zeros(1, tiny_config.memory.read_slots, agent.width)
    cache.score(prompt, memory, '0', ('0', '1'))
    cache.score(prompt, memory.clone(), '0', ('0', '1'))
    assert len(calls) == 6
    memory[0, 0, 0] = 1.
    cache.score(prompt, memory, '0', ('0', '1'))
    assert len(calls) == 8
    generated = []
    def generate(*args, **kwargs):
        generated.append(1)
        return str(kwargs['max_new_tokens'])
    monkeypatch.setattr(agent, 'generate_from_memory', generate)
    assert cache.generate(prompt, memory, max_new_tokens=2) == '2'
    assert cache.generate(prompt, memory.clone(), max_new_tokens=2) == '2'
    assert cache.generate(prompt, memory, max_new_tokens=3) == '3'
    assert len(generated) == 2
    agent.train()
    with pytest.raises(ValueError, match='frozen'):
        cache.score(prompt, memory, '0', ('0', '1'))
    agent.eval()
    with torch.no_grad():
        agent.memory_gate.add_(1.)
    with pytest.raises(ValueError, match='parameters changed'):
        cache.score(prompt, memory, '0', ('0', '1'))


def test_frozen_cache_observes_every_recurrent_event_and_precision(tiny_config):
    from schnitz.agent import SchnitzelAgent
    from schnitz.frozen_scoring import FrozenScorer
    from schnitz.recurrence import LoopMemory
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops = 3
    c.memory.read_timing = 'loop_boundary'
    agent = SchnitzelAgent(c).eval().requires_grad_(False)
    cache = FrozenScorer(agent)
    prompt = agent.prompt_ids('Question')
    first, second = torch.randn(1, 2, 32), torch.randn(1, 2, 32)
    memory = LoopMemory(prompt.shape[1], 2, 3, (first, second))
    before = cache.score(prompt, memory, '0', ('0', '1'))
    second[0, 0, 0] += 10.
    after = cache.score(prompt, memory, '0', ('0', '1'))
    assert before['choice_sequence_nll'] != after['choice_sequence_nll']
    assert cache.report()['unique_score_inputs'] == 2
    cache.score(prompt, memory, '0', ('0', '1'))
    assert cache.report()['hits']['score'] == 1
    with torch.autocast('cpu', dtype=torch.bfloat16):
        cache.score(prompt, memory, '0', ('0', '1'))
    assert cache.report()['unique_score_inputs'] == 3
