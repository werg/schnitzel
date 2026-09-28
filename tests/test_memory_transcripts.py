"""Pure-function tests for scripts/prepare_memory_transcripts.py (WP3, version 2); no
downloads."""
from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_memory_transcripts.py'
spec = importlib.util.spec_from_file_location('prepare_memory_transcripts', SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

TOKENIZER = Path('/runs/hf-models/LFM2.5-350M')


def rec(rid, text, kind, created_at=1, **prov):
    out = {'record_id': rid, 'text': text, 'kind': kind, 'created_at': created_at}
    if prov:
        out['provenance'] = prov
    return out


def index_of(*records, domain='d'):
    return {r['record_id']: (r['created_at'], r['kind'], domain) for r in records}


def builder(records, command_df=None, pool=None, **options):
    return mt.Builder('kb', index_of(*records), mt.Options(**options), per_domain=False,
                      tool_names={'airline': ['cancel_reservation', 'get_user']},
                      command_df=command_df, pool=pool)


def searches(row):
    return [(i, j, tc['function']['arguments'])
            for i, m in enumerate(row['messages']) for j, tc in enumerate(m.get('tool_calls') or [])
            if tc['function']['name'] == 'memory_search']


def slots(row):
    return [m['content']['slot'] for m in row['messages']
            if m['role'] == 'tool' and isinstance(m['content'], dict) and 'slot' in m['content']]


HOP1 = rec('p1', 'Title: Shinjuku Incident\nPassage: Shinjuku Incident is a 2009 Hong Kong film.',
           'passage')
HOP2 = rec('p2', 'Title: British Hong Kong\nPassage: Hong Kong was under British rule from 1842.',
           'passage')
FUTURE = rec('p3', 'Title: Later\nPassage: written after the question.', 'passage', created_at=9)


def qa_episode(**extra):
    episode = {
        'episode_id': 'musique-1', 'query_time': 2, 'task_family': 'public_multihop_qa',
        'query': 'Use the previously stored passages. Give only the short response.\n'
                 'Question: When was the country where the Shinjuku incident occurred taken by '
                 'the British?',
        'answer': '1842', 'required_ids': ['p2', 'p1'], 'sufficient_groups': [['p2', 'p1']],
        'supports': [HOP2, HOP1, FUTURE], 'provenance': {'dataset': 'musique', 'split': 'train'}}
    episode.update(extra)
    return episode


# -- helpers -----------------------------------------------------------------------
def test_preamble_and_need():
    sql = ('Use the stored notes on database farm. Write one SQLite query that answers the '
           'question. Return only the SQL.\nQuestion: How many farms are there?')
    user = mt.strip_preamble(sql, 'spider')
    assert user.startswith('Database farm. Write one SQLite query')
    assert mt.need_text(user, 'spider') == 'How many farms are there?'
    xlam = ('Use the stored tool documentation. Answer with a JSON list of calls [...] and '
            'nothing else.\nRequest: Translate hello')
    assert mt.strip_preamble(xlam, 'xlam') == 'Translate hello'
    apigen = ('Use the stored agent policy and tool documentation. You are the agent; reply to '
              'the customer or call tools.\nCustomer: I want to cancel.')
    assert mt.strip_preamble(apigen, 'apigen_mt') == 'I want to cancel.'
    fever = 'Question: Is this claim supported or refuted? Claim: Plutonium reacts with hydrogen.'
    assert mt.need_text(fever, 'kilt_fever') == 'Plutonium reacts with hydrogen.'


def test_header_fields():
    assert mt.fields_of('schema', 'Database farm, table city (schema):\nCREATE') == \
        {'db': 'farm', 'table': 'city'}
    assert mt.fields_of('tool_doc', 'airline tool get_user: gets a user') == \
        {'area': 'airline', 'tool': 'get_user'}
    assert mt.fields_of('tool_doc', 'Tool: translate\nDescription: x') == {'tool': 'translate'}


def test_action_commands():
    assert mt.commands_of('go to shelf 1') == ['go']
    assert mt.commands_of('click[buy now]') == ['click']
    assert mt.commands_of('get_neighbors(mj, award.x)') == ['get_neighbors']
    assert mt.commands_of('Operation\n```sql\nSELECT COUNT(*) FROM t\n```') == ['select']
    assert mt.commands_of('\n```sql\nDESC weather\n```') == ['desc']
    assert mt.commands_of('bash\n\n```bash\ncat *.conf | grep -v "^$" | awk -F x\n```') == \
        ['cat', 'grep', 'awk']
    turn = 'Thought: I should look.\nAction: take apple 1 from countertop 1'
    assert [c for p in mt.action_parts(turn) for c in mt.commands_of(p)] == ['take']
    assert mt.action_parts('Thought: done.\nFinal Answer: #1') == ['answer']
    example = 'Worked example. Task: heat a mug\nAction: go to x 1\nAction: heat mug 1 with y'
    assert mt.record_commands(example) == ['go', 'heat']
    assert mt.task_type('scienceworld', 'gold:boil:variation:3') == 'boil'
    assert mt.task_type('scienceworld', '9') == 'test-conductivity'
    assert mt.task_type('alfworld', 'gold:boil:variation:3') is None


# -- single-shot transcripts -------------------------------------------------------
def test_multihop_transcript_structure_slots_and_write():
    row, reason, counts = builder([HOP1, HOP2, FUTURE]).build(qa_episode(), 'train')
    assert reason is None
    roles = [m['role'] for m in row['messages']]
    assert roles == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant',
                     'assistant', 'tool']
    assert 'Use the previously stored' not in row['messages'][1]['content']
    assert 'short description' not in row['messages'][0]['content']
    assert 'memory_search()' in row['messages'][0]['content']
    # every search has no argument; hop order: the question grounds p1's title first
    assert [args for _, _, args in searches(row)] == [{}, {}]
    assert [s['record_ids'] for s in slots(row)] == [['p1'], ['p2']]
    assert [s['record_ids'] for s in row['search_sites']] == [['p1'], ['p2']]
    assert all(s['kb'] == 'kb' for s in slots(row))
    assert row['messages'][6] == {'role': 'assistant', 'content': '1842'}
    # a multi-hop answer is a reusable result: written after the answer
    # the write call has no argument; the text is only a teacher target in write_sites
    write = row['messages'][7]
    assert write['tool_calls'][0]['function'] == {'name': 'memory_write', 'arguments': {}}
    assert write['write_span'] == {'kb': 'kb', 'write_site': 0}
    site = row['write_sites'][0]
    assert site['source'] == 'own_result' and site['message'] == 7
    assert site['teacher_text'].endswith('1842')
    assert site['teacher_text'] not in json.dumps(row['messages'])
    assert counts['searches'] == 2 and counts['writes'] == 1
    assert row['format'] == 3
    json.dumps(row)  # serializable


def test_writes_only_where_reusable():
    episode = qa_episode(task_family='public_qa')
    row, _, _ = builder([HOP1, HOP2, FUTURE]).build(episode, 'train')
    assert not row['write_sites']
    assert 'memory_write' not in row['messages'][0]['content']
    row, _, _ = builder([HOP1, HOP2, FUTURE], writes='all').build(episode, 'train')
    assert row['write_sites']
    row, _, _ = builder([HOP1, HOP2, FUTURE], writes='trajectory').build(qa_episode(), 'train')
    assert not row['write_sites']


def test_future_or_missing_required_records_reject():
    episode = qa_episode(required_ids=['p3'], sufficient_groups=[['p3']])
    row, reason, _ = builder([HOP1, HOP2, FUTURE]).build(episode, 'train')
    assert row is None and reason == 'no_valid_sufficient_group'
    sql = {'episode_id': 's-1', 'query_time': 2, 'task_family': 'text_to_sql',
           'query': 'Use the stored notes on database farm. Write SQL.\nQuestion: How many?',
           'answer': 'SELECT count(*) FROM farm', 'required_ids': ['missing'],
           'supports': [rec('missing', 'Database farm, table farm (schema):\nx', 'schema')],
           'provenance': {'dataset': 'spider'}}
    row, reason, _ = builder([HOP1]).build(sql, 'train')
    assert row is None and reason == 'required_record_invalid'


def test_sql_stages_and_write():
    schema = rec('s1', 'Database farm, table city (schema):\nCREATE TABLE city (Status text)',
                 'schema')
    values = rec('v1', "Database farm, values of city.Status:\n'Village'\n'Town'", 'column_values')
    note = rec('e1', 'Database farm, note: village means Status = Village', 'evidence')
    gold = "SELECT count(*) FROM city WHERE Status = 'Village'"
    episode = {'episode_id': 'bird-1', 'query_time': 2, 'task_family': 'text_to_sql',
               'query': 'Use the stored notes on database farm. Write one SQLite query that '
                        'answers the question. Return only the SQL.\nQuestion: How many cities '
                        'are villages?',
               'answer': gold, 'required_ids': ['s1', 'e1'], 'supports': [schema, values, note],
               'verify': {'type': 'sql', 'gold': gold}, 'provenance': {'dataset': 'bird'}}
    row, reason, _ = builder([schema, values, note]).build(episode, 'validation')
    assert reason is None
    assert [s['kind'] for s in row['search_sites']] == ['schema', 'column_values', 'evidence']
    assert all(s['step'] == 0 and s['trigger'] == 'start' for s in row['search_sites'])
    answer = [m for m in row['messages'] if m['role'] == 'assistant' and m['content']]
    assert answer[-1]['content'] == gold
    assert row['messages'][-2]['tool_calls'][0]['function']['arguments'] == {}
    write = row['write_sites'][0]['teacher_text']
    assert write.startswith('Database farm. Question: How many cities are villages?')
    assert write.endswith(gold)
    assert row['verify'] == episode['verify']


def test_function_calling_keeps_real_tools():
    doc = rec('t1', 'Tool: translate\nDescription: Translate text.\nParameters: {}', 'tool_doc')
    unused = rec('t2', 'Tool: weather\nDescription: Weather.\nParameters: {}', 'tool_doc')
    gold = [{'name': 'translate', 'arguments': {'text': 'Bonjour', 'target': 'en'}}]
    episode = {'episode_id': 'xlam-1', 'query_time': 2, 'task_family': 'function_call',
               'query': 'Use the stored tool documentation. Answer with a JSON list of calls '
                        'and nothing else.\nRequest: What does Bonjour mean in English?',
               'answer': json.dumps(gold), 'required_ids': ['t1'], 'supports': [doc, unused],
               'verify': {'type': 'calls', 'gold': gold}, 'provenance': {'dataset': 'xlam'}}
    row, reason, _ = builder([doc, unused], writes='none').build(episode, 'train')
    assert reason is None
    assert [t['name'] for t in row['tools']] == ['memory_search', 'memory_write', 'translate',
                                                   'weather']
    assert [s['record_ids'] for s in slots(row)] == [['t1']]
    final = row['messages'][-1]['tool_calls']
    assert final[0]['function'] == {'name': 'translate', 'arguments': gold[0]['arguments']}
    b = mt.Builder('kb', index_of(doc, unused), mt.Options(distractor_rate=1.0), per_domain=False)
    row, _, counts = b.build(episode, 'train')
    assert counts['distractors'] == 1
    assert any(s.get('distractor') for s in row['search_sites'])


# -- trajectories ------------------------------------------------------------------------
PROTOCOL = rec('pr', 'alfworld protocol:\nOne command per turn: go to X, take X from Y.',
               'protocol')
FLOOR = rec('fp', 'Floorplan 7, where objects were found:\nmug: cabinet 2, shelf 1\n'
            'apple: fridge 1', 'know_how')
HEAT = rec('w-heat', 'Worked example. Task: heat some egg\nAction: go to fridge 1\nAction: take '
           'egg 1 from fridge 1\nAction: go to microwave 1\nAction: heat egg 1 with microwave 1\n'
           'Action: go to countertop 1\nAction: put egg 1 in/on countertop 1', 'worked_example')
LAMP = rec('w-lamp', 'Worked example. Task: look at a book under the desklamp\nAction: go to '
           'desk 1\nAction: take book 1 from desk 1\nAction: use desklamp 1', 'worked_example')
DF = Counter({'go': 50, 'take': 40, 'put': 30, 'heat': 5, 'use': 6})


def alfworld_episode(fail=False):
    actions = ['look', 'go to cabinet 2', 'take mug 1 from cabinet 2', 'go to microwave 1',
               'heat mug 1 with microwave 1', 'go to shelf 1', 'put mug 1 in/on shelf 1']
    observations = ['You see nothing.', 'On the cabinet 2, you see a mug 1.',
                    'You pick up the mug 1.', 'You arrive at microwave 1.',
                    'You heat the mug 1.', 'You arrive at shelf 1.', 'You put the mug 1.']
    if fail:
        observations[1] = 'Nothing happens.'
    turns = []
    for a, o in zip(actions, observations):
        turns += [{'role': 'assistant', 'text': f'Action: {a}'},
                  {'role': 'environment', 'text': f'Observation: {o}'}]
    return {'episode_id': 'alf-7', 'query_time': 2, 'task_family': 'household_agent',
            'query': 'Use the stored protocol, know-how and worked examples.\nTask: put a hot '
                     'mug in shelf.\nYou see a cabinet 2, a microwave 1 and a shelf 1.',
            'answer': '\n'.join(t['text'] for t in turns), 'required_ids': ['pr', 'fp'],
            'supports': [PROTOCOL, FLOOR, HEAT, LAMP], 'turns': turns,
            'provenance': {'dataset': 'alfworld'}}


def placements(row):
    """(record id, step, trigger) per slot record, and the step each search precedes."""
    out = []
    for site in row['search_sites']:
        after = row['messages'][site['message'] + 1:]
        nxt = next(m for m in after if m['role'] == 'assistant' and not mt.is_memory(m))
        for r in site['record_ids']:
            out.append((r, site['step'], site['trigger'], nxt['content']))
    return out


def test_agent_mid_trajectory_placement():
    records = [PROTOCOL, FLOOR, HEAT, LAMP]
    row, reason, counts = builder(records, command_df=DF).build(alfworld_episode(), 'train')
    assert reason is None
    got = placements(row)
    # protocol at the start, before the first action
    assert ('pr', 0, 'start', 'Action: look') in got
    # know-how before the first action naming a place or object it lists
    assert ('fp', 1, 'action_entity', 'Action: go to cabinet 2') in got
    # the heat example before the first heat (its rarest command), re-read before 'put'
    assert ('w-heat', 4, 'action_command', 'Action: heat mug 1 with microwave 1') in got
    assert ('w-heat', 6, 'action_command_reread', 'Action: put mug 1 in/on shelf 1') in got
    # the desklamp example: 'use' never happens, so 'take' and then the exploration
    # command 'go' place it (the earlier read first)
    lamp = [g[1:3] for g in got if g[0] == 'w-lamp']
    assert lamp == [(1, 'action_command'), (2, 'action_command_reread')]
    assert counts['searches_mid'] == 5 and counts['searches_start'] == 1
    # the source turns are unchanged and in order around the inserted calls
    source = [m['content'] for m in row['messages'] if m['role'] in ('assistant', 'user')
              and not mt.is_memory(m)]
    episode = alfworld_episode()
    assert source[1:] == [t['text'] for t in episode['turns']]
    assert all(args == {} for _, _, args in searches(row))
    assert row['write_sites'] and row['write_sites'][0]['source'] == 'own_trajectory'
    # one read per example when --example-reads 1
    row, _, _ = builder(records, command_df=DF, example_reads=1).build(alfworld_episode(), 'train')
    assert not any(t == 'action_command_reread' for _, _, t, _ in placements(row))


def test_failure_rereads_protocol_from_the_prefix():
    records = [PROTOCOL, FLOOR, HEAT, LAMP]
    row, _, counts = builder(records, command_df=DF).build(alfworld_episode(fail=True), 'train')
    rereads = [s for s in row['search_sites'] if s['trigger'] == 'observation_failure']
    assert len(rereads) == 1 and rereads[0]['record_ids'] == ['pr']
    before = row['messages'][rereads[0]['message'] - 1]
    assert before == {'role': 'user', 'content': 'Observation: Nothing happens.'}
    row, _, _ = builder(records, command_df=DF, failure_rereads=False).build(
        alfworld_episode(fail=True), 'train')
    assert not any(s['trigger'] == 'observation_failure' for s in row['search_sites'])


def apigen_episode():
    policy = rec('pol', 'airline agent policy:\n# Airline Agent Policy\n- Verify the user id first.',
                 'policy')
    cancel = rec('pol2', 'airline agent policy:\n- The agent can only cancel the whole trip.',
                 'policy')
    doc = rec('doc', 'airline tool get_user: gets a user\nParameters: {"user_id": "str"}', 'tool_doc')
    cdoc = rec('cdoc', 'airline tool cancel_reservation: cancels\nParameters: {}', 'tool_doc')
    turns = [{'role': 'assistant', 'text': 'Please give me your user id.'},
             {'role': 'environment', 'text': 'Customer: It is amy_1.'},
             {'role': 'assistant', 'text': 'Call: {"name": "get_user", "arguments": {"user_id": "amy_1"}}'},
             {'role': 'environment', 'text': 'Result: {"name": "Amy"}'},
             {'role': 'assistant', 'text': 'Call: {"name": "cancel_reservation", "arguments": {}}'},
             {'role': 'environment', 'text': 'Result: done'},
             {'role': 'assistant', 'text': 'Thanks Amy, your booking is cancelled.'}]
    episode = {'episode_id': 'apigen-1', 'query_time': 2, 'task_family': 'policy_tool_agent',
               'query': 'Use the stored agent policy and tool documentation. You are the agent; '
                        'reply to the customer or call tools.\nCustomer: Cancel my booking.',
               'answer': '\n'.join(t['text'] for t in turns),
               'required_ids': ['pol', 'pol2', 'doc', 'cdoc'],
               'supports': [policy, cancel, doc, cdoc], 'turns': turns,
               'verify': {'type': 'tau_bench', 'area': 'airline'},
               'provenance': {'dataset': 'apigen_mt', 'area': 'airline'}}
    return episode, [policy, cancel, doc, cdoc]


def test_multiturn_tool_docs_and_policy_sections_just_in_time():
    episode, records = apigen_episode()
    row, reason, _ = builder(records).build(episode, 'train')
    assert reason is None
    msgs = row['messages']
    got = {(r, s['step'], s['trigger']) for s in row['search_sites'] for r in s['record_ids']}
    assert got == {('pol', 0, 'start'), ('doc', 1, 'action_tool'), ('cdoc', 2, 'action_tool'),
                   ('pol2', 2, 'action_tool')}
    first = next(i for i, m in enumerate(msgs) if any(
        tc['function']['name'] == 'get_user' for tc in m.get('tool_calls') or []))
    assert slots({'messages': msgs[first - 2:first]})[0]['record_ids'] == ['doc']
    assert msgs[first + 1] == {'role': 'tool', 'name': 'get_user', 'content': '{"name": "Amy"}'}
    assert [t['name'] for t in row['tools']][2:] == ['cancel_reservation', 'get_user']
    write = msgs[-2]['tool_calls'][0]['function']
    assert write == {'name': 'memory_write', 'arguments': {}}
    assert 'get_user -> cancel_reservation' in row['write_sites'][0]['teacher_text']


def test_scienceworld_pools_same_type_examples():
    protocol = rec('sp', 'scienceworld protocol:\nfocus on X commits.', 'protocol')
    own = rec('own', 'Worked example. Task: boil water\nAction: focus on water\nAction: activate '
              'stove', 'worked_example', dataset='scienceworld', group='gold:boil:variation:3')
    other = [rec(f'b{i}', f'Worked example. Task: boil thing {i}\nAction: focus on thing {i}\n'
                 f'Action: activate stove', 'worked_example', dataset='scienceworld',
                 group=f'gold:boil:variation:{10 + i}') for i in range(4)]
    eto = rec('e0', 'Worked example. Task: boil lead\nAction: focus on lead\nAction: activate '
              'stove', 'worked_example', dataset='scienceworld', group='0')
    melt = rec('m0', 'Worked example. Task: melt ice\nAction: focus on ice', 'worked_example',
               dataset='scienceworld', group='gold:melt:variation:1')
    late = rec('late', 'Worked example. Task: boil later\nAction: activate stove', 'worked_example',
               created_at=5, dataset='scienceworld', group='gold:boil:variation:99')
    everything = [protocol, own, eto, melt, late, *other]
    pool = {}
    for r in everything[1:]:
        pool.setdefault(mt.task_type('scienceworld', r['provenance']['group']), []).append(r)
    turns = [{'role': 'assistant', 'text': 'Action: focus on water'},
             {'role': 'environment', 'text': 'Observation: You focus on the water.'},
             {'role': 'assistant', 'text': 'Action: activate stove'},
             {'role': 'environment', 'text': 'Observation: The stove is on.'}]
    episode = {'episode_id': 'scienceworld-gold:boil:variation:3', 'query_time': 2,
               'task_family': 'science_agent', 'query': 'Task: Your task is to boil water.',
               'answer': '\n'.join(t['text'] for t in turns), 'required_ids': ['sp'],
               'supports': [protocol], 'turns': turns,
               'provenance': {'dataset': 'scienceworld', 'group': 'gold:boil:variation:3'}}
    b = builder(everything, command_df=Counter({'focus': 9, 'activate': 4}), pool=pool)
    row, reason, counts = b.build(copy.deepcopy(episode), 'validation')
    assert reason is None and counts['pooled_examples'] == 3
    read = {r for s in row['search_sites'] for r in s['record_ids']}
    pooled = read - {'sp'}
    assert len(pooled) == 3 and pooled <= {'b0', 'b1', 'b2', 'b3', 'e0'}
    assert counts['pool_own_skipped'] == 1          # the episode's own variation
    # examples are placed before the first 'activate' (most specific), re-read at 'focus'
    steps = {(s['step'], s['trigger']) for s in row['search_sites'] if s['kind'] == 'worked_example'}
    assert steps == {(0, 'action_command'), (1, 'action_command_reread')}
    row, _, counts = builder(everything, pool=pool, pool_examples=False).build(
        copy.deepcopy(episode), 'validation')
    assert not counts['pooled_examples']


def test_determinism_and_seed():
    episode, records = apigen_episode()
    first = builder(records).build(copy.deepcopy(episode), 'train')[0]
    again = builder(records).build(copy.deepcopy(episode), 'train')[0]
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    outputs = {json.dumps(builder(records, seed=s).build(copy.deepcopy(episode), 'train')[0],
                          sort_keys=True) for s in range(8)}
    assert len(outputs) > 1


def test_own_gold_records_dropped_and_gold_slots_train_only():
    answer = ('Thought: find the apple\nAction: go to countertop 1\nAction: take apple 1 from '
              'countertop 1\nAction: go to fridge 1\nAction: cool apple 1 with fridge 1')
    copy_rec = rec('w1', 'Worked example. Task: cool an apple\n' + answer, 'worked_example')
    other = rec('w2', 'Worked example. Task: heat a mug\nAction: go to microwave 1\nAction: heat '
                'mug 2 with microwave 1', 'worked_example')
    protocol = rec('pr', 'alfworld protocol:\nOne command per turn.', 'protocol')
    turns = [{'role': 'assistant', 'text': t} for t in answer.split('\nAction: ')]
    episode = {'episode_id': 'alf-1', 'query_time': 2, 'task_family': 'household_agent',
               'query': 'Use the stored protocol, know-how and worked examples.\nTask: cool an apple',
               'answer': answer, 'required_ids': ['pr'], 'supports': [protocol, copy_rec, other],
               'turns': turns, 'provenance': {'dataset': 'alfworld'}}
    b = builder([copy_rec, other, protocol], gold_slots=True)
    row, reason, counts = b.build(copy.deepcopy(episode), 'train')
    assert reason is None and counts['own_gold_dropped'] == 1
    ids = [r for s in slots(row) for r in s['record_ids']]
    assert 'w1' not in ids and 'w2' in ids
    gold = [s for s in slots(row) if s.get('gold')]
    assert len(gold) == 1 and gold[0]['receding_weight'] == 1.0
    b.pending_gold = None
    row, _, _ = b.build(copy.deepcopy(episode), 'validation')
    assert not any(s.get('gold') for s in slots(row))


def test_audit_flags_arguments_foreign_records_and_reordering():
    b = builder([HOP1, HOP2, FUTURE])
    row, _, _ = b.build(qa_episode(), 'train')
    order = [None, -1, None, None, None, None, 0, None, None]
    assert not b.audit(row, ['1842'], 2, order, 1)
    bad = copy.deepcopy(row)
    i, j, _ = searches(bad)[0]
    bad['messages'][i]['tool_calls'][j]['function']['arguments'] = {'query': 'taken in 1842'}
    assert b.audit(bad, ['1842'], 2)['search_has_arguments'] == 1
    bad = copy.deepcopy(row)
    bad['messages'][i + 1 + j]['content']['slot']['record_ids'] = ['p3']
    assert b.audit(bad, ['1842'], 2)['record_after_query'] == 1
    bad['messages'][i + 1 + j]['content']['slot']['record_ids'] = ['nope']
    assert b.audit(bad, ['1842'], 2)['record_not_in_kb'] == 1
    # the answer moved before the searches: source order check
    bad = copy.deepcopy(row)
    bad['messages'].insert(2, bad['messages'].pop(6))
    moved = [order[k] for k in (0, 1, 6, 2, 3, 4, 5, 7, 8)]
    assert not b.audit(bad, ['1842'], 2, moved, 1)      # order kept, only positions changed
    assert b.audit(bad, ['1842'], 2, [None, 0, -1] + moved[3:], 1)['source_order'] == 1
    assert b.audit(row, ['1842'], 2, [None, -1, 0] + order[3:], 1)[
        'generated_message_outside_memory'] == 1
    # a generated system prompt may not copy the answer or later messages
    bad = copy.deepcopy(row)
    alias = 'Hong Kong became British in 1842'
    bad['messages'][0]['content'] += ' Remember: Hong Kong became British in 1842.'
    assert b.audit(bad, ['1842', alias], 2)['system_leak'] == 1
    assert not b.audit(row, ['1842', alias], 2)


@pytest.mark.skipif(not TOKENIZER.exists(), reason='local LFM2.5 tokenizer not available')
def test_render_through_lfm2_template():
    transformers = pytest.importorskip('transformers')
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    episode, records = apigen_episode()
    row, _, _ = builder(records).build(episode, 'train')
    text = mt.render_text(row['messages'], row['tools'], tok)
    assert '<|tool_call_start|>[memory_search()]<|tool_call_end|>' in text
    assert text.count('<|reserved_23|><|reserved_24|>') == len(slots(row))
    # the write: no argument, the model's own span in the same assistant turn
    assert text.count('<|tool_call_start|>[memory_write()]<|tool_call_end|>'
                      '<|reserved_20|><|reserved_22|><|im_end|>') == 1
    assert row['write_sites'][0]['teacher_text'] not in text
    assert not mt.render_check(tok, row)
    row, _, _ = builder([PROTOCOL, FLOOR, HEAT, LAMP], command_df=DF).build(
        alfworld_episode(fail=True), 'train')
    assert not mt.render_check(tok, row)


def test_audit_flags_write_text_and_arguments():
    b = builder([HOP1, HOP2, FUTURE])
    row, _, _ = b.build(qa_episode(), 'train')
    assert not b.audit(row, ['1842'], 2)
    bad = copy.deepcopy(row)
    site = bad['write_sites'][0]
    bad['messages'][site['message']]['tool_calls'][0]['function']['arguments'] = {
        'content': site['teacher_text']}
    got = b.audit(bad, ['1842'], 2)
    assert got['write_site_malformed'] == 1 and got['teacher_text_in_messages'] == 1
    bad = copy.deepcopy(row)
    del bad['messages'][site['message']]['write_span']
    assert b.audit(bad, ['1842'], 2)['write_site_malformed'] == 1
    bad = copy.deepcopy(row)
    bad['write_sites'] = []
    assert b.audit(bad, ['1842'], 2)['write_without_site'] == 1
