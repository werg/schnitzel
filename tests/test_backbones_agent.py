import copy
import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.backbones import TinyBackbone, RecurrentBackbone
from schnitz.data import make_episode, counterfactual, save_episodes, load_episodes
from schnitz.replay import ReplayTape


def test_loop_one_matches_original_and_zero_gate_preserves_it():
    base = TinyBackbone(32, 1, 4).eval()
    recurrent = RecurrentBackbone(base, loops=3).eval()
    ids = torch.randint(3, 259, (1, 12))
    embeds, mask = base.embed(ids), torch.ones_like(ids)
    reference = base.core(embeds, mask)
    torch.testing.assert_close(recurrent.hidden(embeds, mask, loops=1), reference)
    torch.testing.assert_close(recurrent.hidden(embeds, mask), reference)


def test_looped_model_is_causal():
    base = TinyBackbone(32, 1, 4).eval()
    recurrent = RecurrentBackbone(base, loops=3).eval()
    recurrent.loop_gate.data.fill_(0.3)
    ids = torch.randint(3, 259, (1, 12))
    changed = ids.clone()
    changed[:, 6:] = torch.randint(3, 259, (1, 6))
    first = recurrent.hidden(base.embed(ids), torch.ones_like(ids))
    second = recurrent.hidden(base.embed(changed), torch.ones_like(changed))
    torch.testing.assert_close(first[:, :6], second[:, :6])
    assert not torch.allclose(first[:, 6:], second[:, 6:])


def test_loop_gate_has_learning_signal():
    recurrent = RecurrentBackbone(TinyBackbone(32, 1, 4), loops=2)
    ids = torch.randint(3, 259, (1, 12))
    hidden = recurrent.hidden(recurrent.embed(ids), torch.ones_like(ids))
    hidden[..., 0].sum().backward()
    assert recurrent.loop_gate.grad is not None
    assert recurrent.loop_gate.grad.abs() > 0


def test_agent_full_graph_vs_selective_replay(tiny_config):
    a = SchnitzelAgent(tiny_config)
    b = copy.deepcopy(a)
    episode = make_episode(2, distractors=0)
    sources = [a.text_ids(s.text, source=True) for s in episode.supports]
    prompt, target = a.prompt_ids(episode.query), a.target_ids(episode.answer)
    full = [a.produce(source) for source in sources]
    reference = a(prompt, target, full, [0, 1])
    reference.loss.backward()
    tape = ReplayTape(verify_outputs=True)
    records = [tape.capture(b, lambda source=source: b.produce(source)) for source in sources]
    replayed = b(prompt, target, records, [0, 1])
    replayed.loss.backward()
    tape.backward()
    torch.testing.assert_close(reference.loss, replayed.loss)
    for (name, p), (name_b, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert name == name_b
        if p.grad is None:
            assert pb.grad is None, name
        else:
            torch.testing.assert_close(p.grad, pb.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_batched_writer_matches_serial_with_variable_lengths(tiny_config):
    agent = SchnitzelAgent(tiny_config).eval()
    ids = [agent.text_ids(text, source=True) for text in ('short', 'a substantially longer source')]
    serial = [agent.produce(item) for item in ids]
    batched = agent.produce_batch(ids)
    for row, expected in enumerate(serial):
        for actual, reference in zip(batched, expected, strict=True):
            torch.testing.assert_close(actual[row:row + 1], reference, atol=2e-6, rtol=2e-5)


def test_batched_native_consumer_matches_mean_serial_objective(tiny_config):
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.model.loops = 2
    tiny_config.model.writer_loops = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.train.retrieval = 'oracle'
    tiny_config.train.live_fraction = 1.0
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = [make_episode(11, distractors=0), make_episode(12, distractors=0)]
    records, prompts, targets, required = [], [], [], []
    for episode in episodes:
        records.append([agent.produce(agent.text_ids(source.text, source=True))
                        for source in episode.supports])
        prompts.append(agent.prompt_ids(episode.query))
        targets.append(agent.target_ids(episode.answer))
        required.append(list(range(len(episode.supports))))
    serial = torch.stack([agent(prompt, target, record, need).loss
                          for prompt, target, record, need in
                          zip(prompts, targets, records, required, strict=True)]).mean()
    batched = agent.forward_loop_memory_batch(prompts, targets, records, required)
    torch.testing.assert_close(batched.loss, serial, atol=3e-6, rtol=3e-5)


def test_query_is_independent_of_teacher_target(tiny_config):
    a = SchnitzelAgent(tiny_config).eval()
    episode = make_episode(2, distractors=0)
    prompt = a.prompt_ids(episode.query)
    q = a.query(prompt)
    _ = a.target_ids("totally different future answer")
    torch.testing.assert_close(q, a.query(prompt))


@pytest.mark.parametrize("arm", ["memory", "no_memory", "oracle_text"])
def test_forward_arms_and_empty_reads(tiny_config, arm):
    a = SchnitzelAgent(tiny_config)
    e = make_episode(0, distractors=0)
    result = a(a.prompt_ids(e.query), a.target_ids(e.answer), [], [], arm=arm)
    assert torch.isfinite(result.loss)
    result.loss.backward()


@pytest.mark.parametrize("compaction", ["mean", "synthetic"])
def test_temporary_compaction_trains(tiny_config, compaction):
    tiny_config.memory.compaction = compaction
    tiny_config.memory.compact_records = 1
    a = SchnitzelAgent(tiny_config)
    e = make_episode(3, distractors=0)
    records = [a.produce(a.text_ids(s.text, source=True)) for s in e.supports]
    result = a(a.prompt_ids(e.query), a.target_ids(e.answer), records, [0, 1], compact=True)
    result.loss.backward()
    assert torch.isfinite(result.compaction_loss)
    assert a.value_head[-1].weight.grad is not None


def test_counterfactual_respects_decision_branch():
    stopped = make_episode(0, distractors=0, restore=False, capability=0, allowed_capability=1)
    assert counterfactual(stopped, "restoration").answer == "STOP"
    assert counterfactual(stopped, "permission").answer == "RETRY"
    retry = make_episode(0, distractors=0, restore=False, capability=0, allowed_capability=0)
    assert counterfactual(retry, "restoration").answer == "RESTORE_RETRY"
    cf = counterfactual(retry, "restoration")
    assert [s.record_id for s in cf.supports] == [s.record_id for s in retry.supports]
    assert [s.created_at for s in cf.supports] == [s.created_at for s in retry.supports]


def test_fresh_worlds_and_json_roundtrip(tmp_path):
    train = make_episode(0, split="train")
    evaluation = make_episode(0, split="test")
    assert train.environment != evaluation.environment
    path = tmp_path / "episodes.jsonl"
    save_episodes(path, [train])
    assert load_episodes(path) == [train]


def test_causal_source_cutoff():
    e = make_episode(0, distractors=5)
    assert all(s.created_at < e.query_time for s in e.supports)


@pytest.mark.integration
def test_random_lfm_public_embedding_path(monkeypatch):
    pytest.importorskip("transformers")
    # This is a CPU structural test even when the host has CUDA-only extensions.
    import inspect
    from transformers.models.lfm2 import modeling_lfm2
    monkeypatch.setattr(modeling_lfm2, 'causal_conv1d_fn', inspect.unwrap(modeling_lfm2.causal_conv1d_fn))
    from transformers import Lfm2Config, Lfm2ForCausalLM
    config = Lfm2Config(vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=["conv", "full_attention"], block_auto_adjust_ff_dim=False,
        block_ff_dim=64, conv_L_cache=3)
    lm = Lfm2ForCausalLM(config)
    embeddings = lm.get_input_embeddings()(torch.randint(3, 259, (1, 12)))
    embeddings.retain_grad()
    output = lm.base_model(inputs_embeds=embeddings, attention_mask=torch.ones(1, 12),
                           use_cache=False, return_dict=True)
    assert output.last_hidden_state.shape == (1, 12, 32)
    output.last_hidden_state[..., 0].sum().backward()
    assert embeddings.grad is not None
