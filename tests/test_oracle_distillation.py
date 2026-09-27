"""Fixed one-pass text distributions supervise only causal latent answer states."""
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
from schnitz.training import stored_channel, train


def native(config):
    config.model.tiny_layers = 3
    config.model.recurrence_mode = 'middle_block'
    config.model.recurrent_start, config.model.recurrent_end = 1, 2
    config.model.loops, config.model.writer_loops = 2, 1
    config.model.freeze_backbone = True
    config.memory.read_timing = 'loop_boundary'
    config.memory.read_steps = 1
    config.train.oracle_distillation_weight = 1.
    config.train.oracle_anchor_weight = 0.
    config.train.live_fraction = 1.
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


def test_distribution_kl_detaches_teacher_and_checks_answer_positions():
    from schnitz.agent import answer_distribution_kl
    student = torch.randn(1, 4, 17, requires_grad=True)
    teacher = torch.randn(1, 4, 17, requires_grad=True)
    loss = answer_distribution_kl(student, teacher)
    loss.backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None
    with pytest.raises(ValueError, match='same answer positions'):
        answer_distribution_kl(student, teacher[:, :1])


@pytest.mark.parametrize('field,value', [('retrieval', 'learned'), ('arm', 'oracle_text'),
                                        ('oracle_distillation_weight', float('nan')),
                                        ('oracle_anchor_weight', .1)])
def test_distillation_rejects_unmatched_or_moving_teacher(tiny_config, field, value):
    config = native(tiny_config)
    setattr(config.train, field, value)
    with pytest.raises(ValueError, match='distillation'):
        config.validate()


def test_distillation_rejects_trainable_one_pass_teacher(tiny_config):
    config = native(tiny_config)
    config.model.freeze_backbone = False
    with pytest.raises(ValueError, match='distillation'):
        config.validate()


def test_distillation_causal_states_and_full_graph_replay_gradients(tiny_config):
    from schnitz.agent import answer_distribution_kl
    config = native(tiny_config)
    config.memory.noise_std = .02
    reference = SchnitzelAgent(config)
    replay = deepcopy(reference)
    rng = torch.get_rng_state()
    def backward(agent, use_replay):
        tape, records = ReplayTape(verify_outputs=True), []
        sources = ['permission allowed', 'restore snapshot']
        for text in sources:
            ids = agent.text_ids(text, source=True)
            def producer(ids=ids):
                return stored_channel(agent, agent.produce(ids))
            records.append(tape.capture(agent, producer) if use_replay else producer())
        target = agent.target_ids('retry')
        prompt = agent.prompt_ids('What action?')
        queries = []
        original = agent.loop_query
        def query(*args):
            value = original(*args)
            queries.append(value[0].detach().clone())
            return value
        agent.loop_query = query
        forward_rng = torch.get_rng_state()
        result = agent(prompt, target, records, [0, 1])
        changed = target.clone()
        changed[:, -1] = 17
        first_queries = queries.copy()
        queries.clear()
        after_forward_rng = torch.get_rng_state()
        torch.set_rng_state(forward_rng)
        other = agent(prompt, changed, records, [0, 1])
        torch.set_rng_state(after_forward_rng)
        assert len(queries) == len(first_queries)
        for a, b in zip(queries, first_queries, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(result.answer_states[:, :-1], other.answer_states[:, :-1], rtol=0, atol=0)
        with torch.no_grad():
            teacher = agent.conditioned_logits(agent.prompt_ids('What action?', '\n'.join(sources)),
                                               target, None, loops=1)
        student = agent.backbone.logits(result.answer_states).float()
        loss = result.loss + answer_distribution_kl(student, teacher)
        loss.backward()
        before = torch.get_rng_state().clone()
        if use_replay:
            tape.backward()
        assert torch.equal(before, torch.get_rng_state())
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


def test_distillation_muon_partial_resume(tmp_path, tiny_config, monkeypatch):
    config = native(tiny_config)
    config.train.optimizer = 'muon'
    config.train.steps = 2
    config.train.gradient_accumulation = 2
    config.train.selected_producers_only = True
    config.memory.noise_std = .02
    data = tmp_path/'episodes.jsonl'
    save_episodes(data, [e for e in make_multiuse_world(17, bindings=1)
                         if e.task_family == 'multiuse/action'])
    config.train.episodes_file = str(data)
    full, partial = tmp_path/'full', tmp_path/'partial'
    train(config, full)
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        request_stop(partial)
        return value
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, partial)['saved_microbatches'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, partial, resume=True)
    equal(load_file(str(resolve_checkpoint(full)/'model.safetensors')),
          load_file(str(resolve_checkpoint(partial)/'model.safetensors')))
    states = [torch.load(resolve_checkpoint(p)/'training_state.pt', weights_only=True)
              for p in (full, partial)]
    equal(*states)
    rows = [json.loads(line) for line in (full/'metrics.jsonl').read_text().splitlines()]
    assert all(r['oracle_distillation_kl'] > 0 and r['read_count'] == 1 for r in rows)
    for row in rows:
        assert row['optimization_loss'] == pytest.approx(row['loss'] + row['oracle_distillation_kl'])
    changed = deepcopy(config)
    changed.train.oracle_distillation_weight = .5
    with pytest.raises(ValueError, match='hyperparameters differ'):
        train(changed, partial, resume=True)


def test_missing_distillation_progress_rejected_before_weights(tmp_path, tiny_config, monkeypatch):
    from schnitz.checkpoints import restore_checkpoint
    from schnitz.optimizers import make_optimizer
    import safetensors.torch
    config = native(tiny_config)
    config.train.gradient_accumulation = 2
    output = tmp_path/'run'
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        request_stop(output)
        return value
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, output)['saved_microbatches'] == 1
    checkpoint = resolve_checkpoint(output)
    state_path = checkpoint/'training_state.pt'
    state = torch.load(state_path, weights_only=True)
    del state['accumulation']['distillation_total']
    torch.save(state, state_path)
    manifest_path = checkpoint/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256']['training_state.pt'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(safetensors.torch, 'load_model',
                        lambda *a, **k: pytest.fail('Invalid progress loaded model weights'))
    agent = SchnitzelAgent(config)
    with pytest.raises(ValueError, match='accumulation state'):
        restore_checkpoint(agent, make_optimizer(agent), output, random.Random(1),
                           manifest['dataset_sha256'], progress={})
