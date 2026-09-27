"""Training-only text-state supervision must preserve causality and recovery."""
from copy import deepcopy
import hashlib
import json
import random

import pytest
import torch
from safetensors.torch import load_file

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import make_multiuse_world, save_episodes
from schnitz.operations import request_stop
from schnitz.replay import ReplayTape
from schnitz.training import train, stored_channel


def native(config):
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start, config.model.recurrent_end = 1, 2
    config.model.loops, config.model.writer_loops = 3, 1
    config.memory.read_timing = 'loop_boundary'
    config.memory.read_steps, config.memory.read_top_k = 2, 1
    config.train.oracle_alignment_weight = 1.
    config.train.oracle_anchor_weight = .1
    config.train.oracle_anchor_loops = 1
    return config


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def test_alignment_detaches_teacher_and_rejects_position_broadcast():
    from schnitz.agent import answer_state_alignment
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8, requires_grad=True)
    loss = answer_state_alignment(student, teacher)
    loss.backward()
    assert teacher.grad is None
    assert student.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='same answer positions'):
        answer_state_alignment(student, teacher[:, :1])


def test_alignment_states_are_next_token_causal_and_reads_ignore_targets(tiny_config):
    agent = SchnitzelAgent(native(tiny_config)).eval()
    prompt = agent.prompt_ids('Which endpoint?')
    target = agent.target_ids('abcdef')
    changed = target.clone()
    changed[:, 2:] = 17
    records = [agent.produce(agent.text_ids('endpoint abcdef', source=True))]
    queries = []
    original = agent.loop_query
    def query(*args):
        result = original(*args)
        queries.append(result[0].detach().clone())
        return result
    agent.loop_query = query
    a = agent(prompt, target, records, [0])
    first = queries.copy()
    queries.clear()
    b = agent(prompt, changed, records, [0])
    assert a.answer_states.shape == (1, target.shape[1], agent.width)
    torch.testing.assert_close(a.answer_states[:, :3], b.answer_states[:, :3], rtol=0, atol=0)
    equal(first, queries)
    text = agent.prompt_ids('Which endpoint?', 'endpoint abcdef')
    teacher = agent.conditioned_states(text, target, None, loops=1)
    other = agent.conditioned_states(text, changed, None, loops=1)
    torch.testing.assert_close(teacher[:, :3], other[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('retrieval', 'learned'), ('arm', 'oracle_text'),
                                      ('oracle_anchor_weight', 0.), ('oracle_alignment_weight', float('nan'))])
def test_alignment_rejects_unmatched_or_unanchored_paths(tiny_config, field, value):
    config = native(tiny_config)
    setattr(config.train, field, value)
    with pytest.raises(ValueError, match='alignment'):
        config.validate()


def test_alignment_rejects_evidence_not_yet_read(tmp_path, tiny_config):
    config = native(tiny_config)
    config.train.evidence_scope = 'available'
    path = tmp_path/'episodes.jsonl'
    save_episodes(path, make_multiuse_world(12, bindings=2))
    config.train.episodes_file = str(path)
    with pytest.raises(ValueError, match='same evidence in completed latent reads'):
        train(config, tmp_path/'run')


def test_alignment_full_graph_replay_gradients(tiny_config):
    from schnitz.agent import answer_state_alignment
    config = native(tiny_config)
    config.memory.noise_std = .02
    reference = SchnitzelAgent(config)
    replay = deepcopy(reference)
    rng = torch.get_rng_state()
    def backward(agent, use_replay):
        tape = ReplayTape(verify_outputs=True)
        records = []
        for text in ['permission allowed', 'restore snapshot']:
            ids = agent.text_ids(text, source=True)
            def producer(ids=ids):
                return stored_channel(agent, agent.produce(ids))
            records.append(tape.capture(agent, producer) if use_replay else producer())
        target = agent.target_ids('retry')
        result = agent(agent.prompt_ids('What action?'), target, records, [0, 1])
        teacher = agent.conditioned_states(agent.prompt_ids('What action?', 'permission allowed\nrestore snapshot'),
                                           target, None, loops=1)
        logits = agent.backbone.logits(teacher).float()
        anchor = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss = result.loss + answer_state_alignment(result.answer_states, teacher) + .1*anchor
        loss.backward()
        before = torch.get_rng_state().clone()
        if use_replay:
            tape.backward()
        assert torch.equal(before, torch.get_rng_state())
        assert result.read_count == 2
        return loss.detach()
    a = backward(reference, False)
    torch.set_rng_state(rng)
    b = backward(replay, True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (name, p), (_, q) in zip(reference.named_parameters(), replay.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-7, msg=name)
    assert reference.value_head[-1].weight.grad.abs().sum() > 0


def test_alignment_muon_partial_resume(tmp_path, tiny_config, monkeypatch):
    config = native(tiny_config)
    config.train.optimizer = 'muon'
    config.train.steps = 2
    config.train.gradient_accumulation = 2
    config.train.live_fraction = 1.
    config.train.selected_producers_only = True
    config.memory.noise_std = .02
    data = tmp_path/'episodes.jsonl'
    save_episodes(data, [e for e in make_multiuse_world(12, bindings=2)
                         if e.task_family == 'multiuse/action'])
    config.train.episodes_file = str(data)
    replay, partial = [tmp_path/n for n in ('replay', 'partial')]
    train(config, replay)
    def weights(path):
        return load_file(str(resolve_checkpoint(path)/'model.safetensors'))
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        request_stop(partial)
        return result
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, partial)['saved_microbatches'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, partial, resume=True)
    equal(weights(replay), weights(partial))
    states = [torch.load(resolve_checkpoint(p)/'training_state.pt', weights_only=True)
              for p in (replay, partial)]
    equal(*states)
    rows = [json.loads(line) for line in (replay/'metrics.jsonl').read_text().splitlines()]
    assert all(r['oracle_alignment_loss'] > 0 and r['read_count'] == 2 for r in rows)
    for row in rows:
        assert row['optimization_loss'] == pytest.approx(
            row['loss'] + .1*row['oracle_anchor_nll'] + row['oracle_alignment_loss'])
    changed = deepcopy(config)
    changed.train.oracle_alignment_weight = .5
    with pytest.raises(ValueError, match='hyperparameters differ'):
        train(changed, partial, resume=True)


@pytest.mark.parametrize('missing', ['alignment_total', 'anchor_total', 'totals', 'loops', 'microbatches'])
def test_partial_alignment_missing_totals_rejected_before_model_mutation(tmp_path, tiny_config, monkeypatch, missing):
    from schnitz.checkpoints import restore_checkpoint
    from schnitz.optimizers import make_optimizer
    import safetensors.torch
    config = native(tiny_config)
    config.train.gradient_accumulation = 2
    output = tmp_path/'run'
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        request_stop(output)
        return result
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, output)['saved_microbatches'] == 1
    checkpoint = resolve_checkpoint(output)
    state_path = checkpoint/'training_state.pt'
    state = torch.load(state_path, weights_only=True)
    del state['accumulation'][missing]
    torch.save(state, state_path)
    manifest_path = checkpoint/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256']['training_state.pt'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid accumulation must fail before loading model weights')
    monkeypatch.setattr(safetensors.torch, 'load_model', forbidden)
    agent = SchnitzelAgent(config)
    with pytest.raises(ValueError, match='accumulation state'):
        restore_checkpoint(agent, make_optimizer(agent), output, random.Random(1),
                           manifest['dataset_sha256'], progress={})
