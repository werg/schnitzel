import pytest
import torch


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_batched_candidates_match_serial_and_padding_cannot_change_short_answer(tiny_config, timing):
    from schnitz.agent import SchnitzelAgent
    from schnitz.choice_scoring import candidate_nll
    from schnitz.recurrence import LoopMemory
    c = tiny_config
    if timing == 'loop_boundary':
        c.model.tiny_layers = 4
        c.model.recurrence_mode = 'middle_block'
        c.model.recurrent_start, c.model.recurrent_end = 1, 3
        c.model.loops = 3
        c.memory.read_timing = timing
    c.validate()
    agent = SchnitzelAgent(c).eval()
    prompt = agent.prompt_ids('Question')
    memory = torch.randn(1, c.memory.read_slots, agent.width)
    if timing == 'loop_boundary':
        memory = LoopMemory(prompt.shape[1], c.memory.read_slots, 3, (memory, memory * 2))
    targets = [agent.target_ids(text) for text in ('0', 'LONG ANSWER', '1')]
    for view in (None, memory):
        serial = candidate_nll(agent, prompt, view, targets, max_batch=1)
        batched = candidate_nll(agent, prompt, view, targets, max_batch=2)
        torch.testing.assert_close(torch.tensor(batched), torch.tensor(serial), rtol=1e-6, atol=1e-5)
        changed = candidate_nll(agent, prompt, view, [targets[0], agent.target_ids('MUCH LONGER FUTURE TOKENS')], max_batch=2)
        assert changed[0] == pytest.approx(batched[0], abs=1e-5)
    with pytest.raises(ValueError, match='batch'):
        candidate_nll(agent, prompt, memory, targets, max_batch=0)
