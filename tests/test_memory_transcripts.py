"""Pure-function tests for scripts/prepare_memory_transcripts.py (WP3); no downloads."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_memory_transcripts.py'
spec = importlib.util.spec_from_file_location('prepare_memory_transcripts', SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

TOKENIZER = Path('/runs/hf-models/LFM2.5-350M')


def rec(rid, text, kind, created_at=1):
    return {'record_id': rid, 'text': text, 'kind': kind, 'created_at': created_at}


def index_of(*records, domain='d'):
    return {r['record_id']: (r['created_at'], r['kind'], domain) for r in records}


def builder(records, **options):
    return mt.Builder('kb', index_of(*records), mt.Options(**options), per_domain=False,
                      tool_names={'airline': ['get_user', 'update_booking']})


def searches(row):
    return [(i, j, tc['function']['arguments']['query'])
            for i, m in enumerate(row['messages']) for j, tc in enumerate(m.get('tool_calls') or [])
            if tc['function']['name'] == 'memory_search']


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


# -- text helpers ------------------------------------------------------------------
def test_leak_reason_ngrams_answers_and_prefix_exemption():
    prefix = mt.tokens('Question: which team did Chris Burke join in 2002?')
    future = ['Burke returned to Scottish football in September 2016, signing for Ross County.']
    kw = dict(prefix=prefix, future=future, allowed=[], answers=['Rangers'])
    assert mt.leak_reason('team of Chris Burke in 2002', **kw) is None
    assert mt.leak_reason('Chris Burke Rangers', **kw) == 'answer'
    assert mt.leak_reason('football in September 2016 signing', **kw) == 'ngram'
    # the same 5-gram is fine once it occurs in the causal prefix
    kw['prefix'] = prefix + mt.tokens('football in September 2016 signing')
    assert mt.leak_reason('football in September 2016 signing', **kw) is None
    # the answer may be copied from the prefix (e.g. "A or B?" questions)
    assert mt.leak_reason('Rangers or Celtic', prefix=mt.tokens('Rangers or Celtic?'),
                          future=[], allowed=[], answers=['Rangers']) is None
    # record headers are allowed
    assert mt.leak_reason('schema of table frpm in california schools db', prefix=[],
                          future=['schema of table frpm in california schools db ...'],
                          allowed=['schema of table frpm in california schools db'],
                          answers=[]) is None


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
    span = 'Article: Sonic\nStored passage begins: and published by Sega\nReturn exactly words 1'
    assert mt.need_text(span, 'hotpot_qa') == "passage of Sonic that begins 'and published by Sega'"
    fever = 'Question: Is this claim supported or refuted? Claim: Plutonium reacts with hydrogen.'
    assert mt.need_text(fever, 'kilt_fever') == 'Plutonium reacts with hydrogen.'
    assert 'the' not in mt.keyphrase('Where was the treaty of the city signed?').split()


def test_header_fields():
    assert mt.fields_of('schema', 'Database farm, table city (schema):\nCREATE') == \
        {'db': 'farm', 'table': 'city'}
    assert mt.fields_of('column_values', 'Database farm, values of city.Status:\n...') == \
        {'db': 'farm', 'table': 'city', 'column': 'Status'}
    assert mt.fields_of('tool_doc', 'airline tool get_user: gets a user') == \
        {'area': 'airline', 'tool': 'get_user'}
    assert mt.fields_of('tool_doc', 'Tool: translate\nDescription: x') == {'tool': 'translate'}


# -- transcripts -------------------------------------------------------------------
def test_multihop_transcript_structure_and_slots():
    row, reason, counts = builder([HOP1, HOP2, FUTURE]).build(qa_episode(), 'train')
    assert reason is None
    roles = [m['role'] for m in row['messages']]
    assert roles == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
    assert 'Use the previously stored' not in row['messages'][1]['content']
    # hop order: the passage whose title the question grounds comes first
    slots = [m['content']['slot'] for m in row['messages'] if m['role'] == 'tool']
    assert [s['record_ids'] for s in slots] == [['p1'], ['p2']]
    assert all(s['kb'] == 'kb' for s in slots)
    assert row['messages'][-1] == {'role': 'assistant', 'content': '1842'}
    assert all('1842' not in q for _, _, q in searches(row))
    assert row['loss_mask']['no_loss'] == ['system', 'user', 'tool']
    assert counts['searches'] == 2 and not row['write_sites']
    json.dumps(row)  # serializable


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


def test_sql_stages_and_no_gold_sql_in_queries():
    schema = rec('s1', 'Database farm, table city (schema):\nCREATE TABLE city (Status text)',
                 'schema')
    values = rec('v1', "Database farm, values of city.Status:\n'Village'\n'Town'", 'column_values')
    note = rec('e1', 'Database farm, note: village means Status = Village', 'evidence')
    episode = {'episode_id': 'bird-1', 'query_time': 2, 'task_family': 'text_to_sql',
               'query': 'Use the stored notes on database farm. Write one SQLite query that '
                        'answers the question. Return only the SQL.\nQuestion: How many cities '
                        'are villages?',
               'answer': "SELECT count(*) FROM city WHERE Status = 'Village'",
               'required_ids': ['s1', 'e1'], 'supports': [schema, values, note],
               'verify': {'type': 'sql', 'gold': "SELECT count(*) FROM city WHERE Status = 'Village'"},
               'provenance': {'dataset': 'bird'}}
    row, reason, _ = builder([schema, values, note]).build(episode, 'validation')
    assert reason is None
    kinds = [s['kind'] for s in row['search_sites']]
    assert kinds == ['schema', 'column_values', 'evidence']
    assert row['messages'][-1]['content'] == episode['answer']
    assert row['verify'] == episode['verify']
    for _, _, q in searches(row):
        assert 'status = village' not in q.lower()


def test_function_calling_keeps_real_tools():
    doc = rec('t1', 'Tool: translate\nDescription: Translate text.\nParameters: {}', 'tool_doc')
    unused = rec('t2', 'Tool: weather\nDescription: Weather.\nParameters: {}', 'tool_doc')
    gold = [{'name': 'translate', 'arguments': {'text': 'Bonjour', 'target': 'en'}}]
    episode = {'episode_id': 'xlam-1', 'query_time': 2, 'task_family': 'function_call',
               'query': 'Use the stored tool documentation. Answer with a JSON list of calls '
                        'and nothing else.\nRequest: What does Bonjour mean in English?',
               'answer': json.dumps(gold), 'required_ids': ['t1'], 'supports': [doc, unused],
               'verify': {'type': 'calls', 'gold': gold}, 'provenance': {'dataset': 'xlam'}}
    b = builder([doc, unused])
    row, reason, _ = b.build(episode, 'train')
    assert reason is None
    assert [t['name'] for t in row['tools']] == ['memory_search', 'memory_write', 'translate',
                                                   'weather']
    assert [s['record_ids'] for s in (m['content']['slot'] for m in row['messages']
                                      if m['role'] == 'tool')] == [['t1']]
    final = row['messages'][-1]['tool_calls']
    assert final[0]['function'] == {'name': 'translate', 'arguments': gold[0]['arguments']}
    # distractor flag: an unhelpful search over an unused support
    b = mt.Builder('kb', index_of(doc, unused), mt.Options(distractor_rate=1.0), per_domain=False)
    row, _, counts = b.build(episode, 'train')
    assert counts['distractors'] == 1
    assert any(s.get('distractor') for s in row['search_sites'])


def apigen_episode():
    policy = rec('pol', 'airline agent policy:\n- Verify the user id first.', 'policy')
    doc = rec('doc', 'airline tool get_user: gets a user\nParameters: {"user_id": "str"}', 'tool_doc')
    turns = [{'role': 'assistant', 'text': 'Please give me your user id.'},
             {'role': 'environment', 'text': 'Customer: It is amy_1.'},
             {'role': 'assistant', 'text': 'Call: {"name": "get_user", "arguments": {"user_id": "amy_1"}}'},
             {'role': 'environment', 'text': 'Result: {"name": "Amy"}'},
             {'role': 'assistant', 'text': 'Thanks Amy, your booking is cancelled.'}]
    episode = {'episode_id': 'apigen-1', 'query_time': 2, 'task_family': 'policy_tool_agent',
               'query': 'Use the stored agent policy and tool documentation. You are the agent; '
                        'reply to the customer or call tools.\nCustomer: Cancel my booking.',
               'answer': '\n'.join(t['text'] for t in turns), 'required_ids': ['pol', 'doc'],
               'supports': [policy, doc], 'turns': turns,
               'verify': {'type': 'tau_bench', 'area': 'airline'},
               'provenance': {'dataset': 'apigen_mt', 'area': 'airline'}}
    return episode, [policy, doc]


def test_multiturn_just_in_time_docs_and_write_site():
    episode, records = apigen_episode()
    row, reason, _ = builder(records).build(episode, 'train')
    assert reason is None
    msgs = row['messages']
    roles = [m['role'] for m in msgs]
    assert roles == ['system', 'user', 'assistant', 'tool', 'assistant', 'user', 'assistant',
                     'tool', 'assistant', 'tool', 'assistant', 'assistant', 'tool']
    # the tool doc is searched right before the first call of that tool
    assert msgs[6]['tool_calls'][0]['function']['name'] == 'memory_search'
    assert msgs[7]['content']['slot']['record_ids'] == ['doc']
    assert msgs[8]['tool_calls'][0]['function']['name'] == 'get_user'
    assert msgs[9] == {'role': 'tool', 'name': 'get_user', 'content': '{"name": "Amy"}'}
    assert [t['name'] for t in row['tools']][2:] == ['get_user', 'update_booking']
    write = msgs[11]['tool_calls'][0]['function']
    assert write['name'] == 'memory_write' and 'get_user' in write['arguments']['content']
    assert row['write_sites'] == [{'message': 11, 'call': 0, 'site': 'episode_end',
                                   'source': 'own_trajectory'}]
    assert 'write_result' in msgs[12]['content']


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
    ids = [r for m in row['messages'] if m['role'] == 'tool' and 'slot' in m['content']
           for r in m['content']['slot']['record_ids']]
    assert 'w1' not in ids and 'w2' in ids
    gold = [m['content']['slot'] for m in row['messages']
            if m['role'] == 'tool' and m['content'].get('slot', {}).get('gold')]
    assert len(gold) == 1 and gold[0]['receding_weight'] == 1.0
    b.pending_gold = None
    row, _, _ = b.build(copy.deepcopy(episode), 'validation')
    assert not any(m['content'].get('slot', {}).get('gold') for m in row['messages']
                   if m['role'] == 'tool' and isinstance(m['content'], dict))


def test_audit_flags_leaky_query_and_foreign_record():
    b = builder([HOP1, HOP2, FUTURE])
    episode = qa_episode()
    row, _, _ = b.build(episode, 'train')
    texts = {s['record_id']: s['text'] for s in episode['supports']}
    assert not b.audit(row, texts, ['1842'], 2)
    i, j, _ = searches(row)[0]
    row['messages'][i]['tool_calls'][j]['function']['arguments']['query'] = 'taken in 1842'
    assert b.audit(row, texts, ['1842'], 2)['query_leak_answer'] == 1
    row['messages'][i]['tool_calls'][j]['function']['arguments']['query'] = 'Shinjuku'
    row['messages'][i + 1 + j]['content']['slot']['record_ids'] = ['p3']
    assert b.audit(row, texts, ['1842'], 2)['record_after_query'] == 1


@pytest.mark.skipif(not TOKENIZER.exists(), reason='local LFM2.5 tokenizer not available')
def test_render_through_lfm2_template():
    transformers = pytest.importorskip('transformers')
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    episode, records = apigen_episode()
    row, _, _ = builder(records).build(episode, 'train')
    text = mt.render_text(row['messages'], row['tools'], tok)
    assert "<|tool_call_start|>[memory_search(query='" in text
    assert text.count('<|reserved_23|><|reserved_24|>') == 2
    assert not mt.render_check(tok, row)
