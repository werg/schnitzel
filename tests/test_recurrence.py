"""Native middle-block conversion, in-loop reads, and replay correctness."""
import copy
import json

import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.backbones import TinyBackbone
from schnitz.data import make_episode
from schnitz.evaluation import build_shared_bank
from schnitz.recurrence import (
    LoopMemory,
    LoopWrite,
    LoopWrites,
    MiddleBlockBackbone,
    SpatialReadSite,
)
from schnitz.replay import ReplayTape
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import stored_channel, train


@pytest.fixture
def loop_config(tiny_config):
    c = copy.deepcopy(tiny_config)
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing = 'loop_boundary'
    c.memory.read_steps, c.memory.read_top_k = 2, 1
    c.train.live_fraction = 1.
    c.validate()
    return c


def test_one_core_pass_is_exact_parent_and_has_same_input_gradient():
    base = TinyBackbone(32, 4, 4).eval()
    model = MiddleBlockBackbone(base, start=1, end=3)
    x = torch.randn(1, 12, 32, requires_grad=True)
    mask = torch.ones(1, 12, dtype=torch.long)
    y = base.core(x, mask)
    z = model.hidden(x, mask, loops=1)
    torch.testing.assert_close(y, z, rtol=0, atol=0)
    a = torch.autograd.grad(y[..., 0].sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(z[..., 0].sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_parameter_identity_and_native_layer_visits():
    base = TinyBackbone(32, 4, 4).eval()
    model = MiddleBlockBackbone(base, loops=3, start=1, end=3)
    visits, final_norm = [], []
    handles = [layer.register_forward_hook(lambda mod, args, out, i=i: visits.append(i))
               for i, layer in enumerate(base.layers)]
    handles.append(base.norm.register_forward_hook(lambda *args: final_norm.append(1)))
    before = {name: id(p) for name, p in model.named_parameters()}
    model.hidden(torch.randn(1, 10, 32), torch.ones(1, 10, dtype=torch.long))
    assert visits == [0, 1, 2, 1, 2, 1, 2, 3]
    assert len(final_norm) == 1
    assert model.manifest()['layer_visits'] == 8
    model.loops = 7
    assert before == {name: id(p) for name, p in model.named_parameters()}
    for h in handles:
        h.remove()


def test_extra_passes_are_live_and_bridge_receives_gradients():
    model = MiddleBlockBackbone(TinyBackbone(32, 4, 4), loops=3, start=1, end=3)
    x, mask = torch.randn(1, 12, 32), torch.ones(1, 12, dtype=torch.long)
    trace = []
    y = model.hidden(x, mask, trace=trace)
    assert len(trace) == 3 and not torch.equal(trace[0], trace[1])
    y[..., 0].sum().backward()
    for name in ['input_logit', 'update_logit', 'reentry.weight']:
        grad = dict(model.bridge.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_fixed_writer_is_independent_of_consumer_loop_count(loop_config):
    agent = SchnitzelAgent(loop_config).eval()
    ids = agent.text_ids('Support: foo requires bar.', source=True)
    a = agent.produce(ids)
    agent.backbone.loops = 5
    b = agent.produce(ids)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_recurrent_prefix_is_causal_and_reads_never_see_targets(loop_config, checkpoint):
    loop_config.model.gradient_checkpointing = checkpoint
    agent = SchnitzelAgent(loop_config).train()
    prompt = agent.prompt_ids('Use the recorded rule.')
    records = [agent.produce(agent.text_ids(s, source=True)) for s in ['a: red', 'b: green']]
    queries = []
    hook = agent.query_head.register_forward_hook(lambda m, args, out: queries.append(out.detach().clone()))
    result = agent(prompt, agent.target_ids('FIRST'), records, [0, 1])
    first = queries.copy()
    queries.clear()
    agent(prompt, agent.target_ids('DIFFERENT FUTURE'), records, [0, 1])
    assert len(first) == len(queries) == 2
    for a, b in zip(first, queries, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first[0], first[1])
    result.loss.backward()
    hook.remove()


def test_payloads_only_change_state_after_first_core(loop_config):
    agent = SchnitzelAgent(loop_config).eval()
    prompt = agent.prompt_ids('Question')
    shape = (1, loop_config.memory.read_slots, agent.width)
    zeros, changed = torch.zeros(shape), torch.randn(shape)
    first_queries, changed_queries = [], []
    def provider(z, captures):
        def f(completed, query):
            captures.append(query.clone())
            return z
        return f
    agent.plan_loop_memory(prompt, provider(zeros, first_queries))
    agent.plan_loop_memory(prompt, provider(changed, changed_queries))
    torch.testing.assert_close(first_queries[0], changed_queries[0], atol=0, rtol=0)
    assert not torch.allclose(first_queries[1], changed_queries[1])


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_integrated_training_matches_stored_prefix_plan(loop_config, tmp_path, monkeypatch, kind):
    loop_config.memory.reader = kind
    agent = SchnitzelAgent(loop_config).eval()
    e = make_episode(0, distractors=0)
    prompt, target = agent.prompt_ids(e.query), agent.target_ids(e.answer)
    records = [stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))) for s in e.supports]
    integrated = agent(prompt, target, records, [0, 1])
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, [e])
    monkeypatch.setattr(agent, 'produce', lambda *args, **kwargs: pytest.fail('Inference called writer'))
    session = read_session(agent, DiskStore(store.path), prompt, namespace='global', generation='frozen-v1',
                           query_time=e.query_time, oracle_ids=e.required_ids)
    assert isinstance(session.memory, LoopMemory)
    assert len(session.plans) == integrated.read_count == 2
    torch.testing.assert_close(integrated.nll, agent.conditioned_nll(prompt, target, session.memory),
                               atol=1e-6, rtol=1e-6)
    assert session.selected_ids == [list(e.required_ids)]


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('retrieval', ['oracle', 'learned'])
@pytest.mark.parametrize('independent', [False, True])
def test_native_inloop_selective_replay_gradient_parity(loop_config, checkpoint, retrieval, independent):
    loop_config.memory.independent_routing_query = independent
    loop_config.model.gradient_checkpointing = checkpoint
    loop_config.memory.checkpoint_chunks = checkpoint
    loop_config.train.retrieval = retrieval
    loop_config.train.routing_warmup = 0
    if retrieval == 'learned':
        loop_config.memory.read_steps = 1
    a, e = SchnitzelAgent(loop_config), make_episode(2, distractors=2 if retrieval == 'learned' else 0)
    b = copy.deepcopy(a)
    sources = [a.text_ids(s.text, source=True) for s in e.supports]
    prompt, target = a.prompt_ids(e.query), a.target_ids(e.answer)
    ref = a(prompt, target, [stored_channel(a, a.produce(s)) for s in sources], [0, 1])
    ref.loss.backward()
    tape = ReplayTape(verify_outputs=True)
    records = [tape.capture(b, lambda s=s: stored_channel(b, b.produce(s))) for s in sources]
    test = b(prompt, target, records, [0, 1])
    test.loss.backward()
    tape.backward()
    torch.testing.assert_close(ref.loss, test.loss)
    for (name, x), (other, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert name == other
        if x.grad is None:
            assert y.grad is None, name
        else:
            torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4, msg=name)
    assert b.backbone.bridge.memory_projection.weight.grad.abs().sum() > 0
    assert b.write_slots.grad.abs().sum() > 0

    if retrieval == 'learned':
        assert ref.routing_loss > 0
        assert ref.selected == test.selected
        if independent:
            assert b.routing_query_head.weight.grad.abs().sum() > 0
        for parameter in (b.key_head.weight, b.query_head.weight, b.address_maps[0].weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_frozen_base_can_train_live_bridge_and_memory(loop_config):
    loop_config.model.freeze_backbone = True
    a = SchnitzelAgent(loop_config)
    e = make_episode(0, distractors=0)
    records = [a.produce(a.text_ids(s.text, source=True)) for s in e.supports]
    a(a.prompt_ids(e.query), a.target_ids(e.answer), records, [0, 1]).loss.backward()
    assert all(p.grad is None for p in a.backbone.base.parameters())
    for name in ['reentry.weight', 'memory_projection.weight', 'input_logit', 'update_logit', 'memory_logit']:
        grad = dict(a.backbone.bridge.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_core_only_freezing_is_exact(loop_config):
    loop_config.model.backbone_train_scope = 'recurrent_core'
    a = SchnitzelAgent(loop_config)
    for name, parameter in a.backbone.base.named_parameters():
        assert parameter.requires_grad == name.startswith(('layers.1.', 'layers.2.')), name


def test_checkpoint_chunking_does_not_repeat_retrieval(loop_config):
    loop_config.model.gradient_checkpointing = True
    a = SchnitzelAgent(loop_config).train()
    x = torch.randn(1, 15, a.width, requires_grad=True)
    memory = torch.randn(1, 2, a.width, requires_grad=True)
    calls = []
    def callback(completed, state, anchor):
        calls.append(completed)
        return LoopWrite(5, memory)
    a.backbone.hidden(x, torch.ones(1, 15, dtype=torch.long), boundary=callback)[..., 0].sum().backward()
    assert calls == [1, 2]
    assert memory.grad.abs().sum() > 0


def test_same_level_spatial_results_scatter_before_one_core_update():
    torch.manual_seed(4)
    model = MiddleBlockBackbone(TinyBackbone(32, 4, 4), loops=2, start=1, end=3).eval()
    inputs = torch.randn(1, 14, 32)
    first = torch.randn(1, 2, 32, requires_grad=True)
    second = torch.randn(1, 3, 32, requires_grad=True)

    def boundary(_completed, _state, _anchor):
        return LoopWrites((LoopWrite(3, first), LoopWrite(9, second)))

    output = model.hidden(inputs, torch.ones(1, 14, dtype=torch.long), boundary=boundary)
    changed_second = second.detach().clone().add_(1.0)

    def intervention(_completed, _state, _anchor):
        return LoopWrites((LoopWrite(3, first), LoopWrite(9, changed_second)))

    intervened = model.hidden(inputs, torch.ones(1, 14, dtype=torch.long), boundary=intervention)
    torch.testing.assert_close(output[:, :9], intervened[:, :9], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output[:, 9:], intervened[:, 9:])
    output.sum().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert second.grad is not None and second.grad.abs().sum() > 0


def test_spatial_result_workspaces_must_not_overlap():
    model = MiddleBlockBackbone(TinyBackbone(32, 4, 4), loops=2, start=1, end=3)
    inputs = torch.randn(1, 12, 32)
    memory = torch.randn(1, 3, 32)

    def boundary(_completed, _state, _anchor):
        return LoopWrites((LoopWrite(3, memory), LoopWrite(5, memory)))

    with pytest.raises(ValueError, match="must not overlap"):
        model.hidden(inputs, torch.ones(1, 12, dtype=torch.long), boundary=boundary)


def test_whole_sequence_spatial_queries_are_batched_by_recurrence_level(loop_config):
    loop_config.model.loops = 3
    loop_config.memory.read_steps = 2
    agent = SchnitzelAgent(loop_config)
    ids = torch.randint(3, 100, (2, 28))
    ids[:, 5:7] = -1
    ids[:, 14:16] = -1
    ids[:, 23:25] = -1
    sites = (
        SpatialReadSite(torch.tensor([3, 3]), torch.tensor([5, 5]), 1),
        SpatialReadSite(torch.tensor([12, 12]), torch.tensor([14, 14]), 1),
        SpatialReadSite(torch.tensor([21, 21]), torch.tensor([23, 23]), 2),
    )
    calls = []

    def provider(level, active, query, routing_query):
        calls.append((level, len(active), query.detach().clone(), routing_query.detach().clone()))
        values = torch.nn.functional.pad(query, (0, agent.width - query.shape[1]))
        return values[:, None].expand(-1, loop_config.memory.read_slots, -1) * level

    hidden = agent.spatial_recurrent_hidden(ids, torch.ones_like(ids), sites, provider)
    assert hidden.shape == (2, 28, agent.width)
    assert [(level, count, query.shape[0]) for level, count, query, _ in calls] == [
        (1, 2, 4), (2, 1, 2)]
    hidden.sum().backward()
    assert agent.loop_workspace.grad is not None
    assert agent.query_head.weight.grad is not None and agent.query_head.weight.grad.abs().sum() > 0


def test_later_level_query_depends_on_earlier_spatial_result(loop_config):
    loop_config.model.loops = 3
    loop_config.memory.read_steps = 2
    agent = SchnitzelAgent(loop_config).eval()
    ids = torch.randint(3, 100, (1, 24))
    ids[:, 4:6] = -1
    ids[:, 18:20] = -1
    sites = (
        SpatialReadSite(torch.tensor([2]), torch.tensor([4]), 1),
        SpatialReadSite(torch.tensor([16]), torch.tensor([18]), 2),
    )

    def run(value):
        later = []

        def provider(level, _active, query, _routing_query):
            if level == 2:
                later.append(query.detach().clone())
            pattern = torch.arange(agent.width, device=query.device, dtype=query.dtype)
            pattern = pattern[None, None].expand(
                query.shape[0], loop_config.memory.read_slots, -1)
            return pattern * (value if level == 1 else 0)

        agent.spatial_recurrent_hidden(ids, torch.ones_like(ids), sites, provider)
        return later[0]

    assert not torch.allclose(run(0), run(3))


def test_depth_schedule_checkpoint_resume_matches_exactly(loop_config, tmp_path):
    from safetensors.torch import load_file
    loop_config.memory.read_steps = 1
    loop_config.train.loop_counts = [2, 3]
    loop_config.train.steps = 4
    one, two = tmp_path / 'one', tmp_path / 'two'
    train(copy.deepcopy(loop_config), one)
    train(copy.deepcopy(loop_config), two, stop_after=2)
    train(copy.deepcopy(loop_config), two, resume=True)
    for name, parameter in load_file(str(one / 'model.safetensors')).items():
        torch.testing.assert_close(parameter, load_file(str(two / 'model.safetensors'))[name], atol=0, rtol=0)
    a = [json.loads(s)['loops'] for s in (one / 'metrics.jsonl').read_text().splitlines()]
    b = [json.loads(s)['loops'] for s in (two / 'metrics.jsonl').read_text().splitlines()]
    assert a == b and set(a) <= {2, 3}


def test_explicit_legacy_conversion_preserves_parent_weights(tiny_config, tmp_path):
    from safetensors.torch import load_file
    old = copy.deepcopy(tiny_config)
    old.model.tiny_layers = 4
    old.train.steps = 1
    train(old, tmp_path / 'old')
    new = copy.deepcopy(old)
    new.model.recurrence_mode = 'middle_block'
    new.model.recurrent_start, new.model.recurrent_end = 1, 3
    new.model.writer_loops, new.model.loops = 1, 2
    new.memory.read_timing = 'loop_boundary'
    new.model.freeze_backbone = True
    with pytest.raises(ValueError, match='explicit'):
        train(new, tmp_path / 'rejected', init_from=tmp_path / 'old')
    new.train.allow_recurrence_conversion = True
    train(new, tmp_path / 'new', init_from=tmp_path / 'old')
    a, b = load_file(str(tmp_path / 'old/model.safetensors')), load_file(str(tmp_path / 'new/model.safetensors'))
    for name, tensor in a.items():
        if name.startswith('backbone.base.'):
            torch.testing.assert_close(tensor, b[name], atol=0, rtol=0)
    assert json.loads((tmp_path / 'new/initialization.json').read_text())['recurrence_conversion']


@pytest.mark.parametrize('change', ['zero_gate', 'depth', 'compaction', 'wrong_mode', 'truncated_reads'])
def test_invalid_recurrent_config_fails(loop_config, change):
    if change == 'zero_gate':
        loop_config.model.recurrent_update_mix = 0.
    if change == 'depth':
        loop_config.model.recurrent_end = 7
    if change == 'compaction':
        loop_config.memory.compaction = 'mean'
    if change == 'wrong_mode':
        loop_config.model.recurrence_mode = 'full_stack'
    if change == 'truncated_reads':
        loop_config.train.loop_counts = [2, 3]
    with pytest.raises(ValueError):
        loop_config.validate()


@pytest.mark.integration
def test_random_native_lfm_exact_split_and_causality(monkeypatch):
    pytest.importorskip('transformers')
    import inspect
    from transformers.models.lfm2 import modeling_lfm2
    monkeypatch.setattr(modeling_lfm2, 'causal_conv1d_fn', inspect.unwrap(modeling_lfm2.causal_conv1d_fn))
    from transformers import Lfm2Config, Lfm2ForCausalLM
    from schnitz.backbones import HFBackbone
    from torch import nn
    c = Lfm2Config(vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        layer_types=['conv', 'full_attention', 'conv', 'full_attention'],
        block_auto_adjust_ff_dim=False, block_ff_dim=64, conv_L_cache=3)
    base = HFBackbone.__new__(HFBackbone)
    nn.Module.__init__(base)
    base.lm = Lfm2ForCausalLM(c).eval()
    base.width, base.max_length = 32, 1024
    model = MiddleBlockBackbone(base, loops=3, start=1, end=3).eval()
    x = base.embed(torch.randint(3, 259, (1, 12)))
    mask = torch.ones(1, 12, dtype=torch.long)
    torch.testing.assert_close(base.core(x, mask), model.hidden(x, mask, loops=1), atol=1e-6, rtol=1e-6)
    y = x.clone()
    y[:, 6:] += .1
    a, b = model.hidden(x, mask), model.hidden(y, mask)
    torch.testing.assert_close(a[:, :6], b[:, :6], atol=1e-6, rtol=1e-6)
    a[..., 0].sum().backward()
    assert model.bridge.reentry.weight.grad.abs().sum() > 0


def test_parent_distribution_anchor_never_changes_frozen_parent(loop_config):
    loop_config.model.freeze_backbone = True
    loop_config.memory.read_steps = 1
    loop_config.train.arm = 'oracle_text'
    loop_config.train.parent_kl_weight = .1
    loop_config.validate()
    agent = SchnitzelAgent(loop_config)
    prompt, target = agent.prompt_ids('Earlier fact: A. What fact?'), agent.target_ids('A')
    before = agent.conditioned_logits(prompt, target, None, loops=1).detach().clone()
    result = agent(prompt, target, [], [])
    assert result.parent_kl is not None and result.parent_kl > 0
    result.loss.backward()
    torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=1e-3).step()
    after = agent.conditioned_logits(prompt, target, None, loops=1)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_bfloat16_recurrent_preflight(loop_config):
    from schnitz.probes import model_probe
    loop_config.train.precision = 'bf16'
    result = model_probe(loop_config)
    assert result['one_loop_identity_max_error'] == 0
    assert not result['two_loop_identity_required']
    assert result['gradient_norms']['bridge_memory_projection'] > 0


def test_one_pass_stored_session_performs_no_read(loop_config, tmp_path, monkeypatch):
    agent = SchnitzelAgent(loop_config).eval()
    agent.backbone.loops = 1
    store = DiskStore(tmp_path / 'bank.sqlite')
    monkeypatch.setattr(store, 'search', lambda *a, **kw: pytest.fail('One-pass control read memory'))
    session = read_session(agent, store, agent.prompt_ids('Q'), namespace='global', generation='frozen-v1',
                           query_time=10)
    assert session.memory is None and session.plans == []


def test_four_stage_curriculum_and_same_bank_depth_sweep(tmp_path):
    from pathlib import Path
    import yaml
    from schnitz.launch import launch
    from schnitz.depth_eval import evaluate_depths
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / 'recipes/tiny_looped_smoke.yaml').read_text())
    recipe['base_config'] = str(root / 'configs/tiny_looped_cpu.yaml')
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe))
    out = tmp_path / 'run'
    assert launch(path, out)['status'] == 'complete'
    assert launch(path, out, resume=True)['status'] == 'complete'
    bridge = json.loads((out / 'recurrence_bridge/metrics.jsonl').read_text().splitlines()[0])
    assert bridge['parent_kl'] > 0
    report = evaluate_depths(out / 'recurrent_joint', out / 'fresh-causal.jsonl', tmp_path / 'depths',
                             max_episodes=3)
    assert report['same_serialized_bank'] and report['writer_depth'] == 1
    assert report['write_phase']['writer_calls'] == 2
    assert [r['recurrence']['layer_visits'] for r in report['depths']] == [4, 6, 8, 10]
    assert not report['depths'][0]['memory_enabled']
    assert report['depths'][-1]['depth_exceeds_final_training_max']


@pytest.mark.integration
def test_random_native_llama_split_matches_parent():
    pytest.importorskip('transformers')
    from transformers import LlamaConfig, LlamaForCausalLM
    from schnitz.backbones import HFBackbone
    from torch import nn
    c = LlamaConfig(vocab_size=259, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
    base = HFBackbone.__new__(HFBackbone)
    nn.Module.__init__(base)
    base.lm = LlamaForCausalLM(c).eval()
    base.width, base.max_length = 32, 1024
    model = MiddleBlockBackbone(base, loops=3, start=1, end=3).eval()
    x = base.embed(torch.randint(3, 259, (1, 12)))
    mask = torch.ones(1, 12, dtype=torch.long)
    torch.testing.assert_close(base.core(x, mask), model.hidden(x, mask, loops=1), atol=1e-6, rtol=1e-6)
    model.hidden(x, mask)[..., 0].sum().backward()
    assert model.bridge.reentry.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('independent', [False, True])
def test_routing_only_training_preserves_payloads_and_oracle_outputs(tmp_path, loop_config, independent):
    loop_config.memory.independent_routing_query = independent
    from safetensors.torch import load_model
    from schnitz.checkpoints import resolve_checkpoint
    loop_config.train.optimization_scope = 'routing'
    loop_config.train.retrieval = 'learned'
    loop_config.train.optimizer = 'muon'
    loop_config.train.routing_warmup = 1
    loop_config.memory.read_steps = 1
    loop_config.train.steps = 2
    run = tmp_path / 'routing'
    train(loop_config, run)
    partial = copy.deepcopy(loop_config)
    partial.train.steps = 1
    resumed_run = tmp_path / 'resumed-routing'
    train(partial, resumed_run)
    train(loop_config, resumed_run, resume=True)
    resumed = SchnitzelAgent(copy.deepcopy(loop_config)).eval()
    load_model(resumed, str(resolve_checkpoint(resumed_run, verify=True) / 'model.safetensors'))
    initial = next((run / 'checkpoints').glob('step-000000000-*'))
    a, b = SchnitzelAgent(copy.deepcopy(loop_config)).eval(), SchnitzelAgent(copy.deepcopy(loop_config)).eval()
    load_model(a, str(initial / 'model.safetensors'))
    load_model(b, str(resolve_checkpoint(run, verify=True) / 'model.safetensors'))
    for name, value in b.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0, msg=name)
    prefixes = ('key_head.', 'address_maps.', 'query_maps.', 'routing_query_head.')
    changed = []
    for name, value in a.state_dict().items():
        other = b.state_dict()[name]
        if name.startswith(prefixes):
            if not torch.equal(value, other):
                changed.append(name)
        else:
            torch.testing.assert_close(value, other, rtol=0, atol=0, msg=name)
    assert {'key_head.weight', 'address_maps.0.weight', 'query_maps.0.weight'} <= set(changed)
    e = make_episode(5, distractors=2)
    records = []
    for agent in (a, b):
        records.append([stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True)))
                        for s in e.supports])
        agent.config.train.retrieval = 'oracle'  # Explicit diagnostic override, no training.
    for old, new in zip(*records, strict=True):
        torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    prompt, target = a.prompt_ids(e.query), a.target_ids(e.answer)
    torch.testing.assert_close(a(prompt, target, records[0], [0, 1]).nll,
                               b(prompt, target, records[1], [0, 1]).nll, rtol=0, atol=0)


@pytest.mark.parametrize('setting,value', [('retrieval', 'oracle'), ('routing_weight', 0.0)])
def test_routing_only_scope_requires_a_live_routing_objective(tiny_config, setting, value):
    tiny_config.train.optimization_scope = 'routing'
    tiny_config.train.retrieval = 'learned'
    setattr(tiny_config.train, setting, value)
    with pytest.raises(ValueError, match='Routing-only'):
        tiny_config.validate()


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_independent_routing_head_preserves_reader_query_and_causality(loop_config, timing):
    loop_config.memory.read_timing = timing
    loop_config.memory.independent_routing_query = True
    loop_config.memory.read_steps = 1
    agent = SchnitzelAgent(loop_config).eval()
    e = make_episode(7, distractors=2)
    prompt = agent.prompt_ids(e.query)
    records = [stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))) for s in e.supports]
    agent.config.train.retrieval = 'oracle'
    target = agent.target_ids(e.answer)
    before = agent(prompt, target, records, [0, 1]).nll.detach()
    with torch.no_grad():
        agent.routing_query_head.weight.normal_()
    after = agent(prompt, target, records, [0, 1]).nll.detach()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    queries = []
    hook = agent.routing_query_head.register_forward_hook(lambda m, args, out: queries.append(out.detach().clone()))
    agent(prompt, target, records, [0, 1])
    a = queries.pop()
    agent(prompt, agent.target_ids('a wholly different future'), records, [0, 1])
    b = queries.pop()
    hook.remove()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_warm_start_copies_trained_query_into_independent_router(tmp_path, loop_config):
    from safetensors.torch import load_file
    loop_config.train.steps = 1
    source = tmp_path / 'source'
    train(loop_config, source)
    target_config = copy.deepcopy(loop_config)
    target_config.memory.independent_routing_query = True
    target_config.train.steps = 1
    target = tmp_path / 'independent'
    train(target_config, target, init_from=source)
    initial = next((target / 'checkpoints').glob('step-000000000-*'))
    weights = load_file(str(initial / 'model.safetensors'))
    torch.testing.assert_close(weights['routing_query_head.weight'], weights['query_head.weight'], atol=0, rtol=0)


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
@pytest.mark.parametrize('reads', [1, 2])
def test_independent_learned_routing_matches_stored_sessions(tmp_path, loop_config, monkeypatch, timing, reads):
    loop_config.memory.read_timing = timing
    loop_config.memory.read_steps = reads
    loop_config.memory.independent_routing_query = True
    loop_config.train.retrieval = 'learned'
    loop_config.train.routing_warmup = 0
    agent = SchnitzelAgent(loop_config).eval()
    with torch.no_grad():
        agent.routing_query_head.weight.normal_()
    e = make_episode(4, distractors=2)
    prompt, target = agent.prompt_ids(e.query), agent.target_ids(e.answer)
    records = [stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))) for s in e.supports]
    integrated = agent(prompt, target, records, [0, 1])
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, [e])
    monkeypatch.setattr(agent, 'produce', lambda *args: pytest.fail('Inference called writer'))
    session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1', query_time=e.query_time)
    expected = [[e.supports[i].record_id for i in indices] for indices in integrated.selected]
    assert session.selected_ids == expected
    torch.testing.assert_close(integrated.nll, agent.conditioned_nll(prompt, target, session.memory), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_learned_read_count_matches_stored_and_full_replay(tmp_path, loop_config, timing):
    loop_config.memory.read_timing = timing
    loop_config.memory.read_steps = 1
    loop_config.memory.independent_routing_query = True
    loop_config.train.retrieval = 'learned'
    loop_config.train.routing_warmup = 0
    agent = SchnitzelAgent(loop_config).eval()
    agent.read_count_head = torch.nn.Linear(loop_config.memory.key_dim, 2).eval().requires_grad_(False)
    agent.read_count_choices = (1, 2)
    with torch.no_grad():
        agent.read_count_head.weight.zero_()
        agent.read_count_head.bias.copy_(torch.tensor([2., 0.]))
    other = copy.deepcopy(agent)
    e = make_episode(6, distractors=2)
    prompt, target = agent.prompt_ids(e.query), agent.target_ids(e.answer)
    sources = [agent.text_ids(s.text, source=True) for s in e.supports]
    reference = agent(prompt, target, [stored_channel(agent, agent.produce(s)) for s in sources], [0, 1])
    assert len(reference.selected[0]) == 1
    reference.loss.backward()
    tape = ReplayTape(verify_outputs=True)
    records = [tape.capture(other, lambda s=s: stored_channel(other, other.produce(s))) for s in sources]
    replay = other(prompt, target, records, [0, 1])
    replay.loss.backward()
    tape.backward()
    for (name, a), (_, b) in zip(agent.named_parameters(), other.named_parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4, msg=name)
    store = DiskStore(tmp_path / 'count-bank.sqlite')
    build_shared_bank(agent, store, [e])
    session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1', query_time=e.query_time)
    assert session.selected_ids == [[e.supports[i].record_id for i in reference.selected[0]]]
    torch.testing.assert_close(reference.nll, agent.conditioned_nll(prompt, target, session.memory), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
@pytest.mark.parametrize('method', ['mean', 'synthetic'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('objective', ['interleaved', 'paired'])
@pytest.mark.parametrize('retrieval', ['oracle', 'learned'])
def test_inloop_temporary_compaction_replay_and_causal_queries(loop_config, kind, method, checkpoint, objective, retrieval):
    c = loop_config
    c.memory.read_steps = 1
    c.memory.reader, c.memory.compaction = kind, method
    c.memory.compact_records = 1
    c.train.retrieval, c.train.routing_warmup = retrieval, 0
    c.memory.compaction_loss_weight = .3
    c.memory.compaction_objective = objective
    c.memory.behavior_kl_weight = .2
    c.memory.noise_std = .01
    c.memory.compaction_grouping = 'random'
    c.model.gradient_checkpointing = checkpoint
    c.memory.checkpoint_chunks = checkpoint
    c.validate()
    a = SchnitzelAgent(c).train()
    b = copy.deepcopy(a)
    e = make_episode(2, distractors=2 if retrieval == 'learned' else 0)
    prompt, target = a.prompt_ids(e.query), a.target_ids(e.answer)
    sources = [a.text_ids(s.text, source=True) for s in e.supports]
    rng = torch.get_rng_state()
    reference = a(prompt, target, [stored_channel(a, a.produce(s)) for s in sources], [0, 1], compact=True)
    reference.loss.backward()
    end_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    tape = ReplayTape(verify_outputs=True)
    records = [tape.capture(b, lambda s=s: stored_channel(b, b.produce(s))) for s in sources]
    replayed = b(prompt, target, records, [0, 1], compact=True)
    replayed.loss.backward()
    tape.backward()
    assert reference.selected == replayed.selected
    if retrieval == 'oracle':
        assert reference.selected == [[0, 1]]
    else:
        assert reference.routing_loss > 0 and b.key_head.weight.grad.abs().sum() > 0
    assert reference.read_count == 1 and reference.compaction_loss > 0
    if objective == 'interleaved':
        assert reference.raw_nll is None and reference.compact_nll is reference.nll
    else:
        assert reference.raw_nll is reference.nll and reference.compact_nll is not None
        assert reference.behavior_kl is not None and reference.behavior_kl >= -1e-6
    torch.testing.assert_close(reference.loss, replayed.loss)
    assert torch.equal(torch.get_rng_state(), end_rng)
    for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if x.grad is None:
            assert y.grad is None, name
        else:
            torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4, msg=name)
    assert b.write_slots.grad.abs().sum() > 0
    if method == 'synthetic':
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.compactor.parameters())
    b.eval()
    queries = []
    hook = b.query_head.register_forward_hook(lambda m, args, out: queries.append(out.detach().clone()))
    detached = [tuple(v.detach() for v in record) for record in records]
    b(prompt, target, detached, [0, 1], compact=True)
    b(prompt, b.target_ids('DIFFERENT FUTURE'), detached, [0, 1], compact=True)
    hook.remove()
    assert len(queries) == 2
    torch.testing.assert_close(queries[0], queries[1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
@pytest.mark.parametrize('method', ['mean', 'synthetic'])
def test_inloop_compaction_matches_persisted_codes(loop_config, tmp_path, monkeypatch, kind, method):
    from schnitz.cluster_store import ClusterBank, state_fingerprint
    from schnitz.store import ReadPlan, Selection
    c = loop_config
    c.memory.read_steps = 1
    c.memory.reader, c.memory.compaction = kind, method
    c.memory.compact_records = 1
    c.validate()
    agent = SchnitzelAgent(c).eval()
    e = make_episode(0, distractors=0)
    prompt, target = agent.prompt_ids(e.query), agent.target_ids(e.answer)
    records = [stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))) for s in e.supports]
    integrated = agent(prompt, target, records, [0, 1], compact=True)
    store = DiskStore(tmp_path/'bank.sqlite')
    build_shared_bank(agent, store, [e])
    codes = ClusterBank(store, view='trained', reader_hash=state_fingerprint(agent.reader))
    raw = torch.cat([r[1] for r in records], 0)[None]
    view = agent._compact_values(raw, raw.new_ones(raw.shape[:2]))
    # Verify the training code actually passed through the storage precision cast.
    assert torch.equal(view.values, view.values.bfloat16().float())
    plan = ReadPlan('global', 's0', 'frozen-v1', 'research', e.query_time,
                    tuple(Selection(rid, 0.) for rid in e.required_ids))
    codes.put(plan, view.values[0].detach().bfloat16(), view.weights[0].detach())
    def forbidden(*args, **kwargs):
        pytest.fail('Stored read regenerated a source or compact code')
    monkeypatch.setattr(agent, 'produce', forbidden)
    monkeypatch.setattr(agent, '_compact_values', forbidden)
    if agent.compactor is not None:
        monkeypatch.setattr(agent.compactor, 'forward', forbidden)
    session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                           query_time=e.query_time, oracle_ids=e.required_ids, cluster_bank=codes)
    torch.testing.assert_close(integrated.nll, agent.conditioned_nll(prompt, target, session.memory),
                               atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_inloop_paired_objective_matches_separate_raw_compact_graphs(loop_config, kind):
    c = loop_config
    c.memory.read_steps = 1
    c.memory.reader = kind
    c.memory.compaction = 'synthetic'
    c.memory.compact_records = 1
    c.memory.compaction_objective = 'paired'
    c.memory.compact_task_weight = .7
    c.memory.compaction_loss_weight = .2
    c.memory.behavior_kl_weight = 0.
    c.validate()
    paired = SchnitzelAgent(c).train()
    reference = copy.deepcopy(paired)
    reference.config.memory.compaction_objective = 'interleaved'
    e = make_episode(0, distractors=0)
    sources = [paired.text_ids(s.text, source=True) for s in e.supports]
    prompt, target = paired.prompt_ids(e.query), paired.target_ids(e.answer)
    records = [stored_channel(paired, paired.produce(s)) for s in sources]
    actual = paired(prompt, target, records, [0, 1], compact=True)
    actual.loss.backward()
    other = [stored_channel(reference, reference.produce(s)) for s in sources]
    raw = reference(prompt, target, other, [0, 1], compact=False)
    compact = reference(prompt, target, other, [0, 1], compact=True)
    expected = raw.nll + .7 * compact.nll + .2 * compact.compaction_loss
    expected.backward()
    torch.testing.assert_close(actual.loss, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.raw_nll, raw.nll, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.compact_nll, compact.nll, atol=1e-6, rtol=1e-6)
    assert actual.selected == raw.selected == compact.selected
    for (name, a), (_, b) in zip(paired.named_parameters(), reference.named_parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4, msg=name)


@pytest.mark.parametrize('timing', ['prefix', 'loop_boundary'])
def test_paired_value_ablation_removes_source_values_from_both_paths(loop_config, timing):
    c = loop_config
    c.memory.read_steps, c.memory.read_timing = 1, timing
    c.memory.compaction, c.memory.compact_records = 'synthetic', 1
    c.memory.compaction_objective = 'paired'
    c.validate()
    agent = SchnitzelAgent(c).eval()
    e = make_episode(0, distractors=0)
    prompt, target = agent.prompt_ids(e.query), agent.target_ids(e.answer)
    with torch.no_grad():
        records = [stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))) for s in e.supports]
        changed = [tuple(v if i % 2 == 0 else torch.full_like(v, 123.) for i, v in enumerate(record))
                   for record in records]
        a = agent(prompt, target, records, [0, 1], compact=True, ablate_values=True)
        b = agent(prompt, target, changed, [0, 1], compact=True, ablate_values=True)
    for field in ['loss', 'raw_nll', 'compact_nll', 'behavior_kl']:
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
