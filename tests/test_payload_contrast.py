import copy
import json

import torch
from safetensors.torch import load_file

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import Episode, Source, save_episodes
from schnitz.operations import request_stop
from schnitz.replay import ReplayTape
from schnitz.training import stored_channel, train


def test_source_swap_loss_accumulates_both_writer_paths_before_step(tiny_config):
    from schnitz.training import payload_contrast_loss
    config = copy.deepcopy(tiny_config)
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start = 1
    config.model.recurrent_end = 2
    config.model.loops = 2
    config.model.writer_loops = 1
    config.memory.read_timing = 'loop_boundary'
    config.memory.payload_dims = [24, 48]
    config.memory.neighbors = [2, 1]
    config.train.arm = 'memory'
    config.train.retrieval = 'oracle'
    full = SchnitzelAgent(config)
    replayed = copy.deepcopy(full)
    texts = ['Passage: the marker is aqua.', 'Passage: the marker is copper.']

    def run(agent, selective):
        tape = ReplayTape(verify_outputs=True)
        records = []
        for text in texts:
            ids = agent.text_ids(text, source=True)
            def producer(ids=ids):
                return stored_channel(agent, agent.produce(ids))
            records.append(tape.capture(agent, producer) if selective else producer())
        prompt = agent.prompt_ids('What is the marker?')
        target = agent.target_ids('aqua')
        correct = agent(prompt, target, records, [0]).nll
        swapped = agent(prompt, target, records, [1]).nll
        objective, contrast = payload_contrast_loss(correct, correct, swapped,
                                                    weight=.3, margin=.5)
        objective.backward()
        if selective:
            tape.backward()
        return objective.detach(), contrast.detach()

    full_loss, full_contrast = run(full, False)
    replay_loss, replay_contrast = run(replayed, True)
    torch.testing.assert_close(full_loss, replay_loss, rtol=0, atol=1e-6)
    torch.testing.assert_close(full_contrast, replay_contrast, rtol=0, atol=1e-6)
    for (name, a), (_, b) in zip(full.named_parameters(), replayed.named_parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_source_swap_training_resumes_partial_update_exactly(tmp_path, tiny_config, monkeypatch):
    config = tiny_config
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start = 1
    config.model.recurrent_end = 2
    config.model.loops = 2
    config.model.writer_loops = 1
    config.memory.read_timing = 'loop_boundary'
    config.memory.payload_dims = [24, 48]
    config.memory.neighbors = [2, 1]
    config.train.steps = 2
    config.train.gradient_accumulation = 2
    config.train.live_fraction = 1.0
    config.train.payload_contrast_weight = .3
    config.train.checkpoint_every = 1000
    good = Source('good', 'Passage: the marker is aqua.', 1, 'passage')
    wrong = Source('wrong', 'Passage: the marker is copper.', 1, 'passage')
    episode = Episode('q', 'test', (good, wrong), 'What is the marker?', 'aqua',
                      ('good',), False, 0, 0, 2, 'passage_qa', (),
                      (('good',),), 'verified')
    path = tmp_path / 'episodes.jsonl'
    save_episodes(path, [episode])
    config.train.episodes_file = str(path)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    train(config, full)
    original = ReplayTape.backward
    calls = 0

    def stop_after_first_micro_of_second_update(self):
        nonlocal calls
        result = original(self)
        calls += 1
        if calls == 3:
            request_stop(interrupted)
        return result

    monkeypatch.setattr(ReplayTape, 'backward', stop_after_first_micro_of_second_update)
    partial = train(config, interrupted)
    assert partial['steps'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, interrupted, resume=True)
    a, b = [load_file(str(resolve_checkpoint(run) / 'model.safetensors'))
            for run in (full, interrupted)]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    rows = [json.loads(line) for line in (interrupted / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['payload_contrast_loss'] > 0 and row['swapped_source_nll'] > 0 for row in rows)
