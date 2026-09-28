"""The reference-model benchmark's prompts, per-turn agent items and summaries
(scripts/benchmark_models.py; no model is loaded)."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope='module')
def bench():
    path = Path(__file__).parents[1] / 'scripts/benchmark_models.py'
    spec = importlib.util.spec_from_file_location('benchmark_models', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['benchmark_models'] = module  # dataclasses resolve their module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('benchmark_models', None)


def _support(rid, text):
    return {'record_id': rid, 'text': text, 'created_at': 1, 'kind': 'doc'}


def _agent_episode(opening_call=False):
    call_a = {'name': 'get_user', 'arguments': {'user_id': 'u1'}}
    call_b = {'name': 'cancel_order', 'arguments': {'order_id': '#W9', 'reason': 'no longer needed'}}
    turns = [{'role': 'assistant', 'text': 'Call: ' + json.dumps(call_a)} if opening_call else
             {'role': 'assistant', 'text': 'Please give me your user id.'},
             {'role': 'environment', 'text': 'Customer: It is u1.'} if not opening_call else
             {'role': 'environment', 'text': 'Result: {"orders": ["#W9"]}'},
             {'role': 'assistant', 'text': 'Call: ' + json.dumps(call_a)},
             {'role': 'environment', 'text': 'Result: {"orders": ["#W9"]}'},
             {'role': 'assistant', 'text': 'Shall I cancel #W9?'},
             {'role': 'environment', 'text': 'Customer: Yes, SECRET-FUTURE-CONFIRMATION.'},
             {'role': 'assistant', 'text': 'Call: ' + json.dumps(call_b)},
             {'role': 'environment', 'text': 'Result: cancelled'},
             {'role': 'assistant', 'text': 'Done, AFTER-LAST-CALL.'}]
    return {'episode_id': 'ep1', 'query': 'Use the stored agent policy and tool documentation. '
            'You are the agent.\nCustomer: Cancel my order.',
            'answer': 'unused', 'required_ids': ['p', 't'],
            'supports': [_support('p', 'Policy: confirm first.'), _support('t', 'Tool: cancel_order')],
            'turns': turns, 'verify': {'type': 'tau_bench', 'calls': [call_a, call_b]},
            'provenance': {'area': 'retail'}}


def test_every_call_turn_is_scored_on_its_causal_prefix(bench):
    episode = _agent_episode()
    items = bench.build_items('apigen_mt', episode, 'context', 16000)
    assert [item.turn for item in items] == [2, 6]
    assert [item.first_call for item in items] == [True, False]
    assert [item.item_id for item in items] == ['ep1#t2', 'ep1#t6']
    last = items[1]
    assert last.verify == {'type': 'tau_bench', 'calls': [json.loads(episode['turns'][6]['text'][6:])]}
    text = json.dumps(last.messages)
    assert 'Policy: confirm first.' in last.messages[0]['content']
    assert 'Customer: Cancel my order.' in last.messages[0]['content']
    assert 'SECRET-FUTURE-CONFIRMATION' in text  # the customer turn before the call
    assert 'cancel_order", "arguments"' not in text and 'AFTER-LAST-CALL' not in text
    assert 'Result: cancelled' not in text
    # earlier calls are shown in the instructed JSON-list form, roles alternate
    assert last.messages[3] == {'role': 'assistant',
                                'content': '[' + episode['turns'][2]['text'][6:] + ']'}
    roles = [m['role'] for m in last.messages]
    assert all(a != b for a, b in zip(roles, roles[1:])) and roles[-1] == 'user'
    first = items[0]
    assert 'SECRET-FUTURE-CONFIRMATION' not in json.dumps(first.messages)
    assert [m['role'] for m in first.messages] == ['user', 'assistant', 'user']
    closed = bench.build_items('apigen_mt', episode, 'closed_book', 16000)[1]
    assert 'Policy: confirm first.' not in json.dumps(closed.messages)
    opening = bench.build_items('apigen_mt', _agent_episode(opening_call=True), 'context', 16000)
    assert opening[0].turn == 0 and len(opening[0].messages) == 1


def test_turn_summary_keeps_first_and_opening_call_rates(bench):
    rows = [{'episode_id': f'e{i // 2}#t{t}', 'correct': c, 'turn': t, 'first_call': f, 'tool': tool,
             'context_truncated': False, 'thinking_unfinished': False, 'hit_max': False,
             'prompt_tokens': 10, 'new_tokens': 5, 'prompt_tail': '', 'output': ''}
            for i, (t, f, tool, c) in enumerate([(0, True, 'a', True), (2, False, 'think', False),
                                                  (2, True, 'b', False), (4, False, 'c', True)])]
    summary = bench.summarize(rows, 1.0)
    assert summary['accuracy'] == 0.5 and summary['n'] == 4 and summary['episodes'] == 2
    turns = summary['by_turn']
    assert turns['first_call'] == {'accuracy': 0.5, 'n': 2}
    assert turns['opening_call'] == {'accuracy': 1.0, 'n': 1}
    assert turns['later_calls'] == {'accuracy': 0.5, 'n': 2}
    assert turns['without_think'] == {'accuracy': 0.6667, 'n': 3}


def test_required_records_are_never_cut(bench):
    episode = {'required_ids': ['r1', 'r2'],
               'supports': [_support('x', 'related ' * 10), _support('r1', 'a' * 50),
                            _support('r2', 'b' * 50)]}
    texts, truncated = bench.context_records(episode, budget=60)
    assert texts == ['a' * 50, 'b' * 50] and truncated  # over budget, still whole
    texts, truncated = bench.context_records(episode, budget=1000)
    assert len(texts) == 3 and not truncated


def test_task_budgets_and_verifier_arguments(bench, tmp_path):
    for name, spec in bench.TASKS.items():
        assert spec.context_chars >= 8000, name
    episode = {'episode_id': 'rg1', 'query': 'Use the stored worked examples.\nWhat is 1/2 + 1/4?',
               'required_ids': [], 'supports': [_support('w', 'Worked example: 1+1 = 2')],
               'verify': {'type': 'exact', 'answer': '3/4'},
               'provenance': {'task': 'fraction_simplification'}}
    (item,) = bench.build_items('reasoning_gym', episode, 'context', 8000)
    assert item.verify == {'type': 'exact', 'answer': '3/4', 'task': 'fraction_simplification'}
    assert episode['verify'] == {'type': 'exact', 'answer': '3/4'}  # the corpus row is untouched
    assert bench.safe_check('Answer: 0.75', item.verify)
    assert item.messages[0]['content'].startswith('Reference notes:')
    corpus = tmp_path / bench.TASKS['synlogic'].corpus
    corpus.mkdir()
    rows = [{'episode_id': f's{i}', 'provenance': {'puzzle': p}}
            for i, p in enumerate(['sudoku', 'futoshiki', 'cipher'])]
    (corpus / 'episodes-validation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    picked = bench.load_episodes('synlogic', 10, 0, tmp_path)
    assert sorted(r['episode_id'] for r in picked) == ['s0', 's2']  # futoshiki is excluded
