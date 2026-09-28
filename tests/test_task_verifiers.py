import json
import sqlite3
import sys

import pytest

from schnitz import task_verifiers
from schnitz.task_verifiers import (call_match, check_episode, code_match, exact_answer_match,
                                    extract_block, final_answer, first_call_match,
                                    knights_knaves_match, reasoning_gym_match, run_sandboxed,
                                    sql_match, synlogic_match, value_match)


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
    assert knights_knaves_match('Final: **Ann**: Knight; Bo - knave', names, solution)


def test_extract_block_prefers_the_language_fence():
    assert extract_block('x\n```python\nprint(1)\n```\n```sql\nselect 1\n```', 'sql') == 'select 1'
    assert extract_block('select 2', 'sql') == 'select 2'


def test_value_match_needs_every_value_in_a_short_answer():
    rows = [['Paris', 3.0], ['Lyon', 1.25]]
    assert value_match('Paris (3) and Lyon (1.25)', rows)
    assert not value_match('Paris (3)', rows)
    assert not value_match('Paris 3 Lyon 1.25 ' + 'x ' * 80, rows)
    assert value_match('It is 42.', [[42]])


def test_exact_answer_uses_the_stated_final_answer():
    assert exact_answer_match('Steps...\nTherefore, the final answer is &C &C &C A&.', '&C &C &C A&')
    assert exact_answer_match('so \\boxed{\\frac{1}{2}} is it', '\\frac{1}{2}')
    assert exact_answer_match('Answer: `Empty`', 'empty')
    assert exact_answer_match('Answer: 12.0', '12')
    assert not exact_answer_match('Answer: 13', '12')
    assert not exact_answer_match('12 is wrong; the answer is 13', '12')  # the last answer counts
    assert final_answer('The answers are listed.\nx\n7') == '7'  # no answer marker: last line


def test_first_call_and_dispatch():
    gold = {'name': 'get_user', 'arguments': {'user_id': 'u1'}}
    assert first_call_match('[{"name": "get_user", "arguments": {"user_id": "u1"}}, '
                            '{"name": "other", "arguments": {}}]', gold)
    assert not first_call_match('[{"name": "other", "arguments": {}}]', gold)
    assert not first_call_match('Sure, what is your user id?', gold)
    assert check_episode('[get_user(user_id="u1")]', {'type': 'tau_bench', 'calls': [gold]})
    assert check_episode('It is 6.', {'type': 'values', 'rows': [[6]]})
    assert not check_episode('Answer: 5', {'type': 'exact', 'answer': '6'})


# -- sandbox -----------------------------------------------------------------------
def test_sandbox_limits_time_memory_network_and_directory(tmp_path):
    import time
    start = time.monotonic()
    assert run_sandboxed([sys.executable, '-c', 'while True: pass'], tmp_path, timeout=2) is None
    assert time.monotonic() - start < 10
    # the session is killed when the command ends, children included
    run = run_sandboxed([sys.executable, '-c', 'import subprocess, sys; '
                         'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
                         'print("ok")'], tmp_path, timeout=10)
    assert run is not None and run.stdout.strip() == 'ok'
    big = run_sandboxed([sys.executable, '-c', 'x = bytearray(1 << 31); print(len(x))'], tmp_path,
                        timeout=20, memory_mb=512)
    assert big is not None and big.returncode != 0 and 'MemoryError' in big.stderr
    net = run_sandboxed([sys.executable, '-c', 'import socket\n'
                         'try:\n    socket.create_connection(("1.1.1.1", 80), timeout=2)\n'
                         'except OSError as e:\n    print("blocked" if "disabled" in str(e) else e)'],
                        tmp_path, timeout=20)
    assert net.stdout.strip() == 'blocked'
    local = run_sandboxed([sys.executable, '-c', 'import socket\n'
                           'server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()\n'
                           'client = socket.create_connection(server.getsockname(), timeout=5)\n'
                           'print("local ok")'], tmp_path, timeout=20)
    assert local.stdout.strip() == 'local ok'  # tests may run a loopback server
    where = run_sandboxed([sys.executable, '-c', 'import os; print(os.getcwd(), os.environ["HOME"])'],
                          tmp_path, timeout=20)
    assert where.stdout.split() == [str(tmp_path)] * 2
    echo = run_sandboxed([sys.executable, '-c', 'print(input()[::-1])'], tmp_path, 'abc\n', 20)
    assert echo.stdout.strip() == 'cba'


def test_code_match_times_out_and_runs_in_a_fresh_directory():
    test = 'from solution import f\n\ndef test_f():\n    assert f() == 1\n'
    assert not code_match('def f():\n    while True:\n        pass\n', test, 'instruct', timeout=3)
    judge = "{'stdin': [''], 'stdout': ['[]']}"
    assert code_match('import os\nprint(sorted(p for p in os.listdir(".") if not p.startswith(".")'
                      ' and p != "solution.py"))', judge, 'online_judge')


# -- Reasoning Gym -----------------------------------------------------------------
@pytest.mark.parametrize('task, answer, prediction', [
    ('kinematics', '576/11', 'The range is \\boxed{\\frac{576}{11}}'),
    ('uniform_acceleration', '280', 'Answer: 280 m'),
    ('advanced_geometry', '21.80°', 'Answer: 21.8°'),
    ('advanced_geometry', '2.319', 'Answer: 2.3194'),
    ('mini_sudoku', '3 2 4 1\n1 4 2 3\n4 3 1 2\n2 1 3 4',
     'Solved.\nAnswer:\n```\n3 2 4 1\n1 4 2 3\n4 3 1 2\n2 1 3 4\n```\nDone, all rows valid.'),
    ('futoshiki', '4   2\n     \n2   4', 'Answer:\n4 2\n2 4'),
    ('sorting_traces', '[1, 4, 6, 5]', 'Answer: [1,4,6,5]'),
    ('palindrome_partitioning', '[["n", "x", "x", "n"], ["nxxn"]]',
     'Answer: [["nxxn"], ["n","x","x","n"]]'),
    ('word_sequence_reversal', 'the, wood, Section', 'Answer: the,wood,Section'),
    ('checkers_capture', '(3,5)', 'Answer: (3, 5)'),
    ('rsa_cryptography', 'd=85, m=240', 'Answer: d = 85, m = 240'),
    ('complex_arithmetic', '-16.0 - 4.0i', 'Answer: -16 - 4i'),
    ('prime_factorization', '2 × 2 × 5 × 167', 'Answer: 2*2*5*167'),
    ('minesweeper_deduction', 'safe: (1,2), (1,3)\nmines: (5,1)',
     'Answer:\nsafe: (1,3), (1,2)\nmines: (5, 1)'),
    ('bitwise_arithmetic', '-0x91f5', 'Answer: -0x91F5'),
    ('ransom_note', 'True', 'Answer: <true>'),
    ('calendar_arithmetic', 'Wednesday', 'Answer: **Wednesday**.\n\nAnswer:'),
])
def test_reasoning_gym_normalizations_accept_equivalent_answers(task, answer, prediction):
    assert reasoning_gym_match(prediction, answer, task)
    assert check_episode(prediction, {'type': 'exact', 'answer': answer, 'task': task})


@pytest.mark.parametrize('task, answer, prediction', [
    ('kinematics', '576/11', 'Answer: 52.36'),  # not the exact value
    ('decimal_chain_sum', '-50.33', 'Answer: -50.334'),  # no rounding slack unless asked
    ('advanced_geometry', '2.319', 'Answer: 2.33'),
    ('mini_sudoku', '3 2\n2 3', 'Answer:\n3 2\n2 3\n3 2'),  # a longer grid is another answer
    ('tower_of_hanoi', 'Move disk 1 from Peg 1 to Peg 3\nMove disk 2 from Peg 1 to Peg 2',
     'Answer:\nMove disk 1 from Peg 1 to Peg 3\nMove disk 2 from Peg 1 to Peg 2\n'
     'Move disk 1 from Peg 3 to Peg 2'),
    ('sorting_traces', '[1, 4, 6, 5]', 'Answer: [1, 4, 5, 6]'),
    ('word_sequence_reversal', 'the, wood, Section', 'Answer: wood, the, Section'),
    ('minesweeper_deduction', 'safe: (1,2)\nmines: (5,1)', 'Answer:\nsafe: (1,2), (5,1)\nmines: none'),
    ('intermediate_integration', '-exp(x) + C', 'Answer: exp(x) + C'),
])
def test_reasoning_gym_normalizations_reject_wrong_answers(task, answer, prediction):
    assert not reasoning_gym_match(prediction, answer, task)


def test_reasoning_gym_symbolic_answers():
    pytest.importorskip('sympy')
    assert reasoning_gym_match('Answer: 3*x - 3*x*log(x) + C', '-3*x*log(x) + 3*x + C',
                               'intermediate_integration')
    assert not reasoning_gym_match('Answer: x**99999 + C', 'x + C', 'intermediate_integration')


# -- SynLogic ----------------------------------------------------------------------
def _fake_synlogic(root):
    (root / 'base').mkdir(parents=True)
    (root / 'base' / 'data.py').write_text(
        'import json\nclass Data:\n'
        '    def __init__(self, question, answer, difficulty=1, metadata=None, **kw):\n'
        '        self.question, self.answer, self.metadata = question, answer, metadata\n'
        '    @classmethod\n    def from_json_str(cls, s):\n        return cls(**json.loads(s))\n')
    (root / 'base' / 'verifier.py').write_text('class Verifier:\n    pass\n')
    folder = root / 'games' / 'tasks' / 'sudoku' / 'scripts'
    folder.mkdir(parents=True)
    (folder / 'sudoku_verifier.py').write_text(
        'class SudokuVerifier:\n    def verify(self, data, answer):\n'
        '        print("noise")\n'
        '        if answer == "loop":\n            while True:\n                pass\n'
        '        return 1.0 if answer.strip("[] ") == data.answer else 0.5\n')


def test_synlogic_match_runs_the_family_verifier_in_the_sandbox(tmp_path, monkeypatch):
    root = tmp_path / 'synlogic'
    _fake_synlogic(root)
    game = json.dumps({'question': 'q', 'answer': '12'})
    assert synlogic_match('<think>12?</think> <answer>[[12]]</answer>', 'sudoku', game, root)
    assert not synlogic_match('<answer>[[13]]</answer>', 'sudoku', game, root)  # partial score
    assert not synlogic_match('<answer>loop</answer>', 'sudoku', game, root, timeout=2)
    with pytest.raises(ValueError):
        synlogic_match('x', 'cipher', game, tmp_path / 'missing')
    monkeypatch.setattr(task_verifiers, 'SYNLOGIC_ROOT', root)
    assert check_episode('<answer>12</answer>',
                         {'type': 'synlogic', 'task': 'sudoku', 'game_data': game})


def test_synlogic_families_that_need_math_verify_are_local():
    game = json.dumps({'question': 'q', 'answer': '-1'})
    assert synlogic_match('so \\boxed{-1}', 'operation', game)
    assert not synlogic_match('so \\boxed{1}', 'operation', game)
    tree = json.dumps({'question': 'q', 'answer': 'bread, jacket, top'})
    assert synlogic_match('<answer>\\boxed{top, bread,jacket}</answer>', 'space_reasoning_tree', tree)
    assert not synlogic_match('\\boxed{bread, jacket}', 'space_reasoning_tree', tree)
    space = json.dumps({'question': 'q', 'answer': 'Left'})
    assert synlogic_match('\\boxed{left}', 'space_reasoning', space)
    assert not synlogic_match('left', 'space_reasoning', space)  # the box is required
