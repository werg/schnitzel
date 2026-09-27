import copy
import importlib.util
from pathlib import Path

import torch

from schnitz.agent import SchnitzelAgent
from schnitz.positional import (PositionalCodec, PositionalCompactor,
                             PositionalSetReader)
from schnitz.interface_migration import (expand_positional_state,
                                      initialize_positional_student,
                                      new_slice_masks)
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import make_episode, save_episodes
from schnitz.training import config_from_run, train
from schnitz.offline_bank import stored_memory_identity


def test_positional_codec_preserves_slot_axis_and_shares_projection():
    codec = PositionalCodec(7, 3).double()
    states = torch.randn(2, 5, 7, dtype=torch.double, requires_grad=True)
    result = codec(states)
    expected = codec.projection(states).flatten(1)
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 15)
    result.square().sum().backward()
    assert codec.projection.weight.grad is not None


def test_positional_reader_chunking_preserves_complete_aggregate():
    torch.manual_seed(9)
    full = PositionalSetReader(20, 5, 4, 7, width=8, slots=3,
                               rounds=2, chunk_size=99).double()
    chunked = copy.deepcopy(full)
    chunked.chunk_size = 2
    values = torch.randn(2, 7, 20, dtype=torch.double)
    query = torch.randn(2, 4, dtype=torch.double)
    weights = torch.rand(2, 7, dtype=torch.double)
    torch.testing.assert_close(full(values, query, weights).tokens,
                               chunked(values, query, weights).tokens,
                               rtol=1e-10, atol=1e-10)


def test_positional_reader_updates_every_source_and_target_position():
    reader = PositionalSetReader(24, 4, 5, 9, width=12, slots=3,
                                 rounds=2).double()
    values = torch.randn(1, 3, 24, dtype=torch.double, requires_grad=True)
    query = torch.randn(1, 5, dtype=torch.double, requires_grad=True)
    reader(values, query).tokens.square().sum().backward()
    assert values.grad.reshape(1, 3, 4, 6).abs().sum((0, 1, 3)).gt(0).all()
    assert reader.initial_slots.grad.abs().sum(1).gt(0).all()
    for block in reader.blocks:
        assert block.source_position.grad.abs().sum(1).gt(0).all()
        assert block.target_position.grad.abs().sum(1).gt(0).all()


def test_positional_compactor_emits_structured_records_and_conserves_mass():
    compact = PositionalCompactor(24, 4, width=12, records=2, rounds=2).double()
    values = torch.randn(2, 5, 24, dtype=torch.double, requires_grad=True)
    weights = torch.rand(2, 5, dtype=torch.double)
    codes, mass = compact(values, weights)
    assert codes.shape == (2, 2, 24)
    assert mass.shape == (2, 2)
    torch.testing.assert_close(mass.sum(1), weights.sum(1))
    (codes.square().mean() + mass.square().mean()).backward()
    assert values.grad is not None and values.grad.abs().sum() > 0


def test_agent_positional_interface_round_trip_and_gradients(tiny_config):
    config = copy.deepcopy(tiny_config)
    config.memory.write_slots = 4
    config.memory.read_slots = 4
    config.memory.payload_dims = [24, 40]
    config.memory.neighbors = [4, 2]
    config.memory.payload_layout = 'positional'
    config.memory.reader = 'mlp'
    config.validate()
    agent = SchnitzelAgent(config)
    produced = agent.produce(agent.text_ids('remember this', source=True))
    assert produced[1].shape == (1, 24)
    assert produced[3].shape == (1, 40)
    query = torch.randn(1, config.memory.key_dim)
    values = [produced[1][:, None], produced[3][:, None]]
    weights = [torch.ones(1, 1), torch.ones(1, 1)]
    tokens = agent.reader(values, query, weights)
    assert tokens.shape == (1, 4, agent.width)
    tokens.square().mean().backward()
    assert all(codec.projection.weight.grad is not None for codec in agent.codecs)


def test_positional_payload_dimensions_must_divide_writer_slots(tiny_config):
    config = copy.deepcopy(tiny_config)
    config.memory.payload_layout = 'positional'
    config.memory.payload_dims = [25]
    try:
        config.validate()
    except ValueError as exc:
        assert 'divide into writer slots' in str(exc)
    else:
        raise AssertionError('invalid positional layout was accepted')


def test_legacy_bank_identity_defaults_to_flat_layout():
    old = stored_memory_identity({'write_slots': 8, 'payload_dims': [256]})
    current = stored_memory_identity({'write_slots': 8, 'payload_dims': [256],
                                      'payload_layout': 'flat'})
    assert old == current
    assert stored_memory_identity({'write_slots': 8, 'payload_dims': [256],
                                   'payload_layout': 'positional'}) != old


def test_flat_teacher_initializes_structured_student_without_copying_dense_codec(tiny_config):
    teacher = SchnitzelAgent(copy.deepcopy(tiny_config))
    structured = copy.deepcopy(tiny_config)
    structured.memory.payload_layout = 'positional'
    student = SchnitzelAgent(structured)
    migration = initialize_positional_student(teacher, student)
    torch.testing.assert_close(student.key_head.weight, teacher.key_head.weight)
    assert 'key_head.weight' in migration.copied
    assert any(name.startswith('codecs.') for name in migration.initialized)
    torch.testing.assert_close(student.reader.blocks[0].message.weight,
                               teacher.reader.blocks[0].output.weight)


def test_positional_expansion_preserves_shared_parameters_and_masks_old_slices(tiny_config):
    old_config = copy.deepcopy(tiny_config)
    old_config.memory.payload_layout = 'positional'
    source = SchnitzelAgent(old_config)
    new_config = copy.deepcopy(old_config)
    new_config.memory.write_slots = 8
    new_config.memory.read_slots = 8
    channels = [d // old_config.memory.write_slots for d in old_config.memory.payload_dims]
    new_config.memory.payload_dims = [8 * c for c in channels]
    target = SchnitzelAgent(new_config)
    migration = expand_positional_state(source, target, seed=3)
    assert 'write_slots' in migration.expanded
    torch.testing.assert_close(target.write_slots[:old_config.memory.write_slots + 1],
                               source.write_slots)
    torch.testing.assert_close(target.codecs[0].projection.weight,
                               source.codecs[0].projection.weight)
    masks = new_slice_masks(source, target)
    assert masks['write_slots'][:old_config.memory.write_slots + 1].sum() == 0
    assert masks['write_slots'][old_config.memory.write_slots + 1:].min() == 1
    slot_mix = masks['reader.slot_mix.0.weight']
    old = old_config.memory.read_slots
    assert slot_mix[:old, :old].sum() == 0
    assert slot_mix[old:, :].min() == 1


def test_flat_to_positional_distillation_is_checkpointed_and_resumable(tmp_path, tiny_config):
    teacher_run = tmp_path / 'teacher'
    train(tiny_config, teacher_run)
    episodes = tmp_path / 'episodes.jsonl'
    save_episodes(episodes, [make_episode(31, distractors=1)])
    path = Path(__file__).parents[1] / 'scripts/distill_positional_interface.py'
    spec = importlib.util.spec_from_file_location('positional_distill', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'student'
    first = module.run(teacher_run, episodes, output, steps=1, checkpoint_every=1,
                       downstream_weight=0., task_weight=0.)
    assert first['steps'] == 1
    assert config_from_run(output).memory.payload_layout == 'positional'
    assert resolve_checkpoint(output, verify=True).is_dir()
    resumed = module.run(teacher_run, episodes, output, steps=1, checkpoint_every=1,
                         downstream_weight=0., task_weight=0., resume=True)
    assert resumed['steps'] == 1 and not resumed['stopped_early']
    expansion_path = Path(__file__).parents[1] / 'scripts/expand_positional_interface.py'
    expansion_spec = importlib.util.spec_from_file_location('positional_expand', expansion_path)
    expansion = importlib.util.module_from_spec(expansion_spec)
    expansion_spec.loader.exec_module(expansion)
    widened = tmp_path / 'widened'
    result = expansion.run(output, episodes, widened, steps=1, write_slots=4,
                           read_slots=4, checkpoint_every=1, task_weight=0.)
    assert result['steps'] == 1
    widened_config = config_from_run(widened)
    assert widened_config.memory.write_slots == widened_config.memory.read_slots == 4
    assert widened_config.memory.payload_dims == [48]
    resumed = expansion.run(output, episodes, widened, steps=1, write_slots=4,
                             read_slots=4, checkpoint_every=1, task_weight=0., resume=True)
    assert resumed['steps'] == 1 and not resumed['stopped_early']


def _joint_config(tiny_config, slots=2):
    config = copy.deepcopy(tiny_config)
    config.memory.write_slots = config.memory.read_slots = slots
    config.memory.payload_layout = 'joint_tokens'
    config.memory.space_tokens = [1, 3]
    config.memory.payload_dims = [32, 96]  # tiny backbone width is 32
    config.memory.neighbors = [2, 2]
    config.memory.reader = 'mlp'
    config.validate()
    return config


def test_joint_tokens_mix_every_writer_slot_into_full_width_tokens(tiny_config):
    from schnitz.positional import JointTokenCodec
    codec = JointTokenCodec(32, 3, heads=4)
    states = torch.randn(2, 5, 32, requires_grad=True)
    stored = codec(states)
    assert stored.shape == (2, 96)
    stored[:, :32].sum().backward()
    # Every writer slot influences even the first stored token.
    assert (states.grad.abs().sum(-1) > 0).all()
    agent = SchnitzelAgent(_joint_config(tiny_config))
    produced = agent.produce(agent.text_ids('remember this', source=True))
    assert produced[1].shape == (1, 32) and produced[3].shape == (1, 96)
    values = [produced[1][:, None], produced[3][:, None]]
    tokens = agent.reader(values, torch.randn(1, agent.config.memory.key_dim),
                          [torch.ones(1, 1), torch.ones(1, 1)])
    assert tokens.shape == (1, 2, agent.width)
    assert [reader.source_slots for reader in agent.reader.local] == [1, 3]
    tokens.square().mean().backward()
    assert all(codec.queries.grad is not None for codec in agent.codecs)


def test_joint_tokens_require_full_width_tokens_per_space(tiny_config):
    config = copy.deepcopy(tiny_config)
    config.memory.payload_layout = 'joint_tokens'
    config.memory.space_tokens = [2, 3]
    try:
        config.validate()
    except ValueError as exc:
        assert 'one token count per space' in str(exc)
    else:
        raise AssertionError('mismatched token counts were accepted')
    config = _joint_config(tiny_config)
    config.memory.payload_dims = [16, 48]
    config.memory.space_tokens = [1, 3]
    try:
        SchnitzelAgent(config)
    except ValueError as exc:
        assert 'full decoder-width' in str(exc)
    else:
        raise AssertionError('compressed joint tokens were accepted')


def test_joint_expansion_keeps_codec_and_token_tables(tiny_config):
    source = SchnitzelAgent(_joint_config(tiny_config, slots=2))
    target = SchnitzelAgent(_joint_config(tiny_config, slots=4))
    migration = expand_positional_state(source, target, seed=5)
    assert 'write_slots' in migration.expanded
    for old, new in zip(source.codecs, target.codecs, strict=True):
        torch.testing.assert_close(new.queries, old.queries)
    torch.testing.assert_close(target.reader.local[1].blocks[0].source_position,
                               source.reader.local[1].blocks[0].source_position)


def test_flat_to_joint_distillation_and_expansion_runners(tmp_path, tiny_config):
    teacher_run = tmp_path / 'teacher'
    train(tiny_config, teacher_run)
    episodes = tmp_path / 'episodes.jsonl'
    save_episodes(episodes, [make_episode(31, distractors=1)])
    path = Path(__file__).parents[1] / 'scripts/distill_positional_interface.py'
    spec = importlib.util.spec_from_file_location('joint_distill', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'joint'
    result = module.run(teacher_run, episodes, output, steps=1, checkpoint_every=1,
                        payload_weight=0., downstream_weight=0., task_weight=0.,
                        layout='joint_tokens', space_tokens=(3,))
    assert result['steps'] == 1
    config = config_from_run(output)
    assert config.memory.payload_layout == 'joint_tokens'
    assert config.memory.payload_dims == [96] and config.memory.space_tokens == [3]
    expansion_path = Path(__file__).parents[1] / 'scripts/expand_positional_interface.py'
    expansion_spec = importlib.util.spec_from_file_location('joint_expand', expansion_path)
    expansion = importlib.util.module_from_spec(expansion_spec)
    expansion_spec.loader.exec_module(expansion)
    widened = tmp_path / 'joint-wide'
    expanded = expansion.run(output, episodes, widened, steps=1, write_slots=4,
                             read_slots=4, checkpoint_every=1, task_weight=0.)
    assert expanded['steps'] == 1
    widened_config = config_from_run(widened)
    assert widened_config.memory.write_slots == widened_config.memory.read_slots == 4
    assert widened_config.memory.payload_dims == [96]
