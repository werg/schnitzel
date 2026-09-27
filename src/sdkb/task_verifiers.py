"""Outcome checks for verifiable task episodes (restart plan B9).

Pure Python and standard library, so the checks run anywhere episodes are
built or scored. Each ``*_match`` returns ``True`` only for a correct outcome;
parsing failures count as wrong, never as errors.

- function calls: unordered multiset of (name, arguments), arguments compared as
  canonical JSON; calls may be a JSON list or LFM2's native Python-style list
  (``<|tool_call_start|>[f(a=1), g(b="x")]<|tool_call_end|>``);
- SQL: both queries executed read-only on the task's SQLite database with a time
  limit; result rows compared as multisets (order-insensitive, as BIRD scores);
- code: the candidate written to ``solution.py`` beside the task's pytest file,
  or run on stdin/stdout cases, in a subprocess with a time limit;
- Knights and Knaves: every inhabitant's role parsed from the text.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import time


def _first_json_list(text: str):
    start = text.find('[')
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '[':
                depth += 1
            elif text[end] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('[', start + 1)
    return None


def _call_key(call) -> tuple[str, str] | None:
    if not isinstance(call, dict) or 'name' not in call:
        return None
    arguments = call.get('arguments', {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return str(call['name']), json.dumps(arguments, sort_keys=True)


def _python_calls(text: str):
    """Calls from a Python-style list ``[f(a=1), g()]`` (keyword arguments only)."""
    import ast
    found = re.search(r'\[\s*[A-Za-z_][\w.]*\s*\(.*\)\s*\]', text, re.DOTALL)
    if not found:
        return None
    try:
        tree = ast.parse(found.group(0), mode='eval').body
    except SyntaxError:
        return None
    if not isinstance(tree, ast.List):
        return None
    calls = []
    for node in tree.elts:
        if not isinstance(node, ast.Call) or node.args:
            return None
        try:
            arguments = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        except ValueError:
            return None
        calls.append({'name': ast.unparse(node.func), 'arguments': arguments})
    return calls


def parse_calls(text: str):
    """Predicted calls as a list of ``{name, arguments}``, or ``None``."""
    calls = _first_json_list(text)
    if isinstance(calls, list) and all(isinstance(c, dict) for c in calls) and calls:
        return calls
    return _python_calls(text)


def call_match(prediction: str, gold: list[dict]) -> bool:
    """The predicted calls (JSON or Python-style list) equal ``gold`` as a multiset."""
    calls = parse_calls(prediction)
    if not isinstance(calls, list):
        return False
    keys = [_call_key(call) for call in calls]
    if any(key is None for key in keys):
        return False
    return Counter(keys) == Counter(_call_key(call) for call in gold)


def extract_block(text: str, language: str) -> str:
    """The first fenced ``language`` block (or any fenced block), else the text."""
    for pattern in (rf'```{language}\s*\n(.*?)```', r'```[a-zA-Z]*\s*\n(.*?)```'):
        found = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if found:
            return found.group(1).strip()
    return text.strip()


def run_sql(db_path: Path, sql: str, timeout: float = 10.0):
    """Rows of ``sql`` on a read-only connection, or ``None`` on error or timeout."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return con.execute(sql).fetchall()
    except (sqlite3.Error, sqlite3.Warning, ValueError):
        return None
    finally:
        con.close()


def sql_match(prediction: str, gold_sql: str, db_path: Path, timeout: float = 10.0) -> bool:
    """Executing the predicted SQL gives the gold result rows (as a multiset)."""
    predicted = run_sql(db_path, extract_block(prediction, 'sql'), timeout)
    if predicted is None:
        return False
    expected = run_sql(db_path, gold_sql, timeout)
    return expected is not None and Counter(map(repr, predicted)) == Counter(map(repr, expected))


def code_match(prediction: str, test: str, style: str, timeout: float = 20.0) -> bool:
    """The predicted program passes the task's tests.

    ``style`` ``online_judge``: ``test`` is a Python literal ``{'stdin': [...],
    'stdout': [...]}`` and the program's stripped stdout must match each case;
    otherwise ``test`` is a pytest file importing from ``solution``."""
    code = extract_block(prediction, 'python')
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        (root / 'solution.py').write_text(code, encoding='utf-8')
        if style == 'online_judge':
            import ast
            try:
                cases = ast.literal_eval(test)
            except (ValueError, SyntaxError):
                return False
            for given, wanted in zip(cases['stdin'], cases['stdout'], strict=True):
                try:
                    run = subprocess.run([sys.executable, 'solution.py'], input=given, cwd=root,
                                         capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    return False
                if run.returncode != 0 or run.stdout.strip() != str(wanted).strip():
                    return False
            return True
        (root / 'test_solution.py').write_text(test, encoding='utf-8')
        try:
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                                  'test_solution.py'], cwd=root, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return run.returncode == 0


def knights_knaves_match(prediction: str, names: list[str], solution: list[bool]) -> bool:
    """Every inhabitant is named with the right role (``True`` = knight)."""
    for name, knight in zip(names, solution, strict=True):
        found = re.findall(rf'\b{re.escape(name)}\b\s+is\s+an?\s+(knight|knave)', prediction,
                           re.IGNORECASE)
        if not found or (found[-1].lower() == 'knight') != knight:
            return False
    return True
