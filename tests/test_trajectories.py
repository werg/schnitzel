from dataclasses import replace
import json
from pathlib import Path
import pytest
from schnitz.backbones import ByteTokenizer
from schnitz.data import load_episodes, make_episode, save_episodes
from schnitz.episode_index import EpisodeIndex
from schnitz.trajectories import (normalize_message, normalize_trajectory, trajectory_episodes,
                               prepare_trajectories, split_for, chunk_message, CATALOG)


def spec(adapter='ultrachat'):
    return dict(path='test/fixture', revision='fixed', adapter=adapter, license='project-authored')


def row(index=0):
    return {'prompt_id': str(index), 'messages': [
        dict(role='user', content=f'Remember serial {index}.'),
        dict(role='assistant', content=f'Serial {index} remembered.'),
        dict(role='user', content='Repeat it.'), dict(role='assistant', content='TARGET_MARKER'),
        dict(role='tool', content='FUTURE_OBSERVATION')]}


def budgets():
    return dict(max_source_tokens=128, max_prompt_tokens=1800, max_target_tokens=64,
                max_supports=4, max_targets_per_trajectory=4, recent_messages=1, recent_tokens=128)


def test_hermes_and_structured_calls():
    data = dict(id='a', conversations=[dict(**{'from': 'human'}, value='lookup x'),
                     dict(**{'from': 'gpt'}, value='<tool_call>{"name":"f"}</tool_call>')], tools='[{"name":"f"}]')
    t = normalize_trajectory(data, spec('hermes'))
    assert t.messages[-1]['role'] == 'assistant' and t.split_group.startswith('tools:')
    m = normalize_message(dict(role='assistant', content=None, tool_calls=[
        dict(id='call-1', function=dict(name='f', arguments='{"key":"x"}'))]))
    assert '"arguments": {"key": "x"}' in m['content'] and '"id": "call-1"' in m['content']


def test_swe_double_encoding_and_success_filter():
    data = dict(instance_id='org__repo.abc.task1', messages=json.dumps(json.dumps(row()['messages'])), resolved=True)
    t = normalize_trajectory(data, spec('swe_smith') | {'success_only': True})
    assert t.split_group == 'repo:org/repo'
    with pytest.raises(ValueError, match='filtered_unsuccessful'):
        normalize_trajectory(data | {'resolved': False}, spec('swe_smith') | {'success_only': True})


def test_openhands_does_not_append_patch_or_outcome():
    data = dict(repo='org/repo', instance_id='issue1', trajectory=row()['messages'], model_patch='FUTURE_PATCH', resolved=1)
    t = normalize_trajectory(data, spec('openhands'))
    assert 'FUTURE_PATCH' not in str(t.messages) and 'resolved' not in str(t.messages)


def test_xlam_schema_and_answer():
    t = normalize_trajectory(dict(id=1, query='q', tools='[{"name":"f"}]', answers='[{"name":"f","arguments":{}}]'), spec('xlam'))
    assert len(t.messages) == 3 and t.messages[0]['role'] == 'system'


def test_prefix_no_future_target_and_complete_answer():
    t = normalize_trajectory(row(), spec())
    episodes, stats = trajectory_episodes([t], ByteTokenizer(), budgets())
    e = next(e for e in episodes if e.answer == 'TARGET_MARKER')
    assert all('TARGET_MARKER' not in s.text and 'FUTURE_OBSERVATION' not in s.text for s in e.supports)
    assert all(i < e.provenance['target_index'] for i in e.provenance['source_turns'].values())
    assert all(s.created_at < e.query_time for s in e.supports)
    assert e.support_annotation == 'provided_context' and not e.sufficient_groups
    assert 'FUTURE_OBSERVATION' not in e.query and stats['episodes'] > 0


def test_long_targets_skipped_not_truncated():
    data = row()
    data['messages'][3]['content'] = 'X' * 100
    es, stats = trajectory_episodes([normalize_trajectory(data, spec())], ByteTokenizer(), budgets())
    assert all(not e.answer.startswith('X') for e in es)
    assert stats['skipped_long_or_empty_target'] == 1


def test_cross_experience_excludes_same_instance_and_self():
    records = [replace(normalize_trajectory(row(i), spec()), transfer_group='repo:x', split_group='repo:x',
                       instance_id='same' if i < 2 else 'other') for i in range(3)]
    es, _ = trajectory_episodes(records, ByteTokenizer(), budgets(), protocol='cross_experience')
    by_id = {t.trajectory_id: t for t in records}
    assert es
    for e in es:
        assert e.provenance['trajectory_id'] not in e.provenance['source_trajectories']
        assert all(by_id[t].instance_id != e.provenance['instance_id'] for t in e.provenance['source_trajectories'])
        assert e.provenance['ordering'].startswith('experimental')


def test_local_prepare_manifest_split_and_index(tmp_path):
    path = Path(__file__).parent / 'fixtures/trajectories.jsonl'
    m = prepare_trajectories([spec() | {'local_file': str(path), 'max_rows': 60}], ByteTokenizer(),
                            budgets(), tmp_path / 'data', validation_fraction=.25)
    train, val = [load_episodes(tmp_path / 'data' / (s + '.jsonl')) for s in ('train', 'validation')]
    assert not {e.provenance['split_group'] for e in train} & {e.provenance['split_group'] for e in val}
    index = EpisodeIndex(tmp_path / 'data/train.jsonl')
    assert index[0] == train[0] and index[-1] == train[-1] and len(index) == len(train)
    assert m['leakage_checks']['overlapping_groups'] == 0
    assert all(e.provenance['target_complete'] for e in train)
    assert len(m['sources'][0]['spec']['revision']) == 64


def test_preparation_skips_pinned_stream_prefix_and_excludes_prior_groups(tmp_path):
    path = Path(__file__).parent / 'fixtures/trajectories.jsonl'
    baseline = prepare_trajectories([spec() | {'local_file': str(path), 'max_rows': 60}],
                                    ByteTokenizer(), budgets(), tmp_path / 'baseline',
                                    validation_fraction=.25)
    baseline_lines = (tmp_path / 'baseline/normalized.jsonl').read_text().splitlines()
    prior = {json.loads(line)['split_group'] for line in baseline_lines[:10] + baseline_lines[20:25]}
    later = prepare_trajectories([spec() | {'local_file': str(path), 'skip_rows': 10,
                                          'max_rows': 50, 'excluded_split_groups': sorted(prior)}],
                                 ByteTokenizer(), budgets(), tmp_path / 'later',
                                 validation_fraction=.25)
    rows = [json.loads(line) for line in (tmp_path / 'later/normalized.jsonl').read_text().splitlines()]
    earlier = [json.loads(line) for line in (tmp_path / 'baseline/normalized.jsonl').read_text().splitlines()]
    assert rows
    assert not {r['split_group'] for r in rows} & prior
    assert {r['trajectory_id'] for r in rows} <= {r['trajectory_id'] for r in earlier[10:]}
    counts = later['sources'][0]['counts']
    assert counts['rows_skipped'] == 10
    assert counts['rows_scanned'] == 50
    assert counts['excluded_prior_group'] >= 1
    assert later['sources'][0]['spec']['skip_rows'] == 10
    assert later['sources'][0]['spec']['excluded_split_groups'] == sorted(prior)
    assert baseline['sources'][0]['spec']['revision'] == later['sources'][0]['spec']['revision']


def test_negative_stream_skip_rejected(tmp_path):
    path = Path(__file__).parent / 'fixtures/trajectories.jsonl'
    with pytest.raises(ValueError, match='skip_rows'):
        prepare_trajectories([spec() | {'local_file': str(path), 'skip_rows': -1}],
                             ByteTokenizer(), budgets(), tmp_path / 'bad')


def test_unknown_nontext_content_rejected():
    with pytest.raises(ValueError, match='Unsupported nontext'):
        normalize_message(dict(role='user', content=[dict(type='image_url', image_url='https://example.invalid')]))
    with pytest.raises(ValueError, match='message object'):
        normalize_message('not a message')
    assert CATALOG['hermes']['config'] == 'func_calling' and CATALOG['swe_smith']['split'] == 'tool'
    assert CATALOG['xlam']['gated']


def test_group_assignment_deterministic():
    assert split_for('repo:r', 17, .1) == split_for('repo:r', 17, .1)


def test_no_future_first_user_enters_an_earlier_assistant_target():
    data = row()
    data['messages'] = [dict(role='system', content='rules'), dict(role='assistant', content='early greeting'),
                        dict(role='user', content='FUTURE_USER'), dict(role='assistant', content='valid answer')]
    es, _ = trajectory_episodes([normalize_trajectory(data, spec())], ByteTokenizer(), budgets())
    assert all(e.provenance['target_index'] >= 3 for e in es)


def test_index_checks_duplicate_ids_and_conflicting_sources(tmp_path):
    a, path = make_episode(0), tmp_path / 'episodes.jsonl'
    save_episodes(path, [a, a])
    with pytest.raises(ValueError, match='Episode IDs'):
        EpisodeIndex(path)
    b = replace(a, episode_id='different', supports=(replace(a.supports[0], text='different'), *a.supports[1:]))
    save_episodes(path, [a, b])
    with pytest.raises(ValueError, match='shared source ID'):
        EpisodeIndex(path)


def test_cross_budget_keeps_multiple_experiences():
    records = [replace(normalize_trajectory(row(i), spec()), trajectory_id=str(i), transfer_group='repo:x',
                       split_group='repo:x') for i in range(4)]
    es, _ = trajectory_episodes(records, ByteTokenizer(), budgets(), protocol='cross_experience')
    last = [e for e in es if e.provenance['trajectory_id'] == '3']
    assert last and all(len(e.provenance['source_trajectories']) == 2 for e in last)


def test_unicode_and_code_whitespace_preserved():
    data = row()
    text = '世界 café\n    identifier_α = 1\n' * 20
    data['messages'][0]['content'] = text
    t = normalize_trajectory(data, spec())
    chunks = chunk_message(ByteTokenizer(), t, 0, 128)
    header = 'Recorded user context:\n'
    assert ''.join(c.text[len(header):] for c in chunks) == text
    assert all(len(ByteTokenizer().encode(c.text, add_special_tokens=True)) <= 128 for c in chunks)


def test_anthropic_tool_links_preserved():
    m = normalize_message(dict(role='assistant', content=[dict(type='tool_use', id='call1', name='f', input={})]))
    n = normalize_message(dict(role='user', content=[dict(type='tool_result', tool_use_id='call1', content='done')]))
    assert 'call1' in m['content'] and 'call1' in n['content']
