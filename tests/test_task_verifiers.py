import sqlite3

from sdkb.task_verifiers import (call_match, code_match, extract_block, knights_knaves_match,
                                 sql_match, value_match)


def test_call_match_is_an_unordered_multiset():
    gold = [{'name': 'a', 'arguments': {'x': 1, 'y': [2]}}, {'name': 'b', 'arguments': {}}]
    assert call_match('Calls: [{"name": "b", "arguments": {}}, '
                      '{"name": "a", "arguments": {"y": [2], "x": 1}}]', gold)
    assert not call_match('[{"name": "a", "arguments": {"x": 1, "y": [2]}}]', gold)
    assert not call_match('[{"name": "a", "arguments": {"x": 2, "y": [2]}}, '
                          '{"name": "b", "arguments": {}}]', gold)
    assert not call_match('no calls here', gold)
    assert call_match('<|tool_call_start|>[b(), a(x=1, y=[2])]<|tool_call_end|>', gold)
    assert not call_match('[a(1, y=[2]), b()]', gold)  # positional arguments are unnamed
    assert call_match('[{"name": "b", "arguments": "{}"}, '
                      '{"name": "a", "arguments": {"x": 1, "y": [2]}}]', gold)


def test_sql_match_compares_executed_rows(tmp_path):
    db = tmp_path / 'db.sqlite'
    con = sqlite3.connect(db)
    con.execute('create table t (a int, b text)')
    con.executemany('insert into t values (?, ?)', [(1, 'x'), (2, 'y'), (3, 'x')])
    con.commit()
    con.close()
    gold = "SELECT a FROM t WHERE b = 'x'"
    assert sql_match('```sql\nselect a from t where b="x" order by a desc\n```', gold, db)
    assert not sql_match('select a from t', gold, db)
    assert not sql_match('select nonsense', gold, db)
    assert not sql_match('delete from t', gold, db)  # read-only connection


def test_code_match_runs_pytest_and_stdin_cases():
    test = 'from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n'
    assert code_match('```python\ndef add(a, b):\n    return a + b\n```', test, 'instruct')
    assert not code_match('def add(a, b):\n    return a - b\n', test, 'instruct')
    judge = "{'stdin': ['2 3\\n'], 'stdout': ['5']}"
    program = 'a, b = map(int, input().split())\nprint(a + b)\n'
    assert code_match(program, judge, 'online_judge')
    assert not code_match('print(0)', judge, 'online_judge')


def test_knights_knaves_parses_each_role():
    names, solution = ['Ann', 'Bo'], [True, False]
    assert knights_knaves_match('(1) Ann is a knight\n(2) Bo is a knave', names, solution)
    assert not knights_knaves_match('Ann is a knave and Bo is a knave', names, solution)
    assert not knights_knaves_match('Ann is a knight', names, solution)


def test_extract_block_prefers_the_language_fence():
    assert extract_block('x\n```python\nprint(1)\n```\n```sql\nselect 1\n```', 'sql') == 'select 1'
    assert extract_block('select 2', 'sql') == 'select 2'


def test_value_match_needs_every_value_in_a_short_answer():
    rows = [['Paris', 3.0], ['Lyon', 1.25]]
    assert value_match('Paris (3) and Lyon (1.25)', rows)
    assert not value_match('Paris (3)', rows)
    assert not value_match('Paris 3 Lyon 1.25 ' + 'x ' * 80, rows)
    assert value_match('It is 42.', [[42]])
