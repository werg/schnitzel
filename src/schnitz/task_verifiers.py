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
  or run on stdin/stdout cases, in a sandboxed subprocess (``run_sandboxed``: its
  own temporary working directory and session, wall-time, CPU, data-segment (heap) and
  file-size limits, a minimal environment and a best-effort block of Internet
  sockets; not a security boundary: the containers allow no network namespace);
- Knights and Knaves: every inhabitant's role parsed from the text;
- answers from memory: every expected value appears in a short answer;
- exact answers (Reasoning Gym): the final answer (``\\boxed{}``, ``Answer:`` line or
  last line) equals the expected string after light normalization; with the
  generator named (``verify.task``) also per-family normalizations
  (``reasoning_gym_match``: numbers and fractions, grids, lists, sets, complex
  numbers, factorizations, symbolic expressions);
- SynLogic: the family's verifier from the SynLogic repository (a local checkout,
  ``SYNLOGIC_ROOT``), run in the sandbox; the three families whose verifiers import
  ``math_verify`` (not installed) are reimplemented here (``synlogic_match``);
- agent turns (APIGen-MT): the first predicted call equals the gold call.

``check_episode`` dispatches on an episode's ``verify`` spec
(``scripts/prepare_task_corpora.py``).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

SYNLOGIC_ROOT = Path(os.environ.get(
    'SCHNITZ_SYNLOGIC_ROOT', '/archive/raw/worlds-20260927/synlogic/github-SynLogic'))


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
    con.text_factory = lambda raw: raw.decode('utf-8', 'replace')
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


# Run in the child before the command: resource limits, then exec. (``preexec_fn``
# is unsafe with threads, and the benchmark verifies in a thread pool.)
_LAUNCH = """import os, resource, sys
for key, value in zip(('DATA', 'CPU', 'FSIZE', 'CORE'), map(int, sys.argv[1:5])):
    resource.setrlimit(getattr(resource, 'RLIMIT_' + key), (value, value + (key == 'CPU')))
os.execv(sys.argv[5], sys.argv[5:])
"""
# ``sitecustomize`` of every sandboxed Python: Internet sockets connect and names
# resolve only on the loopback interface (tests may run a local server). Best effort
# only: code can bypass it through ``_socket``.
_NO_NETWORK = """import socket
_LOCAL = {None, '', 'localhost', '127.0.0.1', '::1', '0.0.0.0'}
def _local(host):
    return host in _LOCAL or str(host).startswith('127.')
def _guard(method, position):
    def guarded(self, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            address = args[position] if len(args) > position else None
            if not (isinstance(address, tuple) and _local(address[0])):
                raise OSError('network is disabled in the verifier sandbox')
        return method(self, *args)
    return guarded
socket.socket.connect = _guard(socket.socket.connect, 0)
socket.socket.connect_ex = _guard(socket.socket.connect_ex, 0)
socket.socket.sendto = _guard(socket.socket.sendto, -1)
_getaddrinfo = socket.getaddrinfo
def getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        raise OSError('network is disabled in the verifier sandbox')
    return _getaddrinfo(host, *args, **kwargs)
socket.getaddrinfo = getaddrinfo
"""


def run_sandboxed(args: list[str], cwd: Path, stdin: str | None = None, timeout: float = 20.0,
                  memory_mb: int = 2048, file_mb: int = 64) -> subprocess.CompletedProcess | None:
    """Run ``args`` (``args[0]`` an absolute executable) in the directory ``cwd`` with
    limits: wall time ``timeout`` (then the whole process group is killed), CPU time,
    data segment ``memory_mb`` (heap and private anonymous memory; the address space
    is not limited, since importing CUDA-enabled torch maps several GB of libraries),
    size of written files ``file_mb``, no core dumps; a minimal environment (``HOME``
    and ``TMPDIR`` are ``cwd``, one BLAS/OpenMP thread) and Python Internet sockets
    disabled. Returns ``None`` on timeout.

    Not a security boundary: the containers allow no network namespace, so the
    socket block is a Python-level guard, and the files of the host stay readable."""
    cwd = Path(cwd)
    guard = cwd / '.sandbox'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text(_NO_NETWORK, encoding='utf-8')
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(cwd), 'TMPDIR': str(cwd),
           'LANG': 'C.UTF-8', 'PYTHONPATH': str(guard), 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1'}
    limits = [memory_mb << 20, max(1, math.ceil(timeout)), file_mb << 20, 0]
    command = [sys.executable, '-c', _LAUNCH, *map(str, limits), *args]
    # output goes to files (capped by the file-size limit), not unbounded pipes
    with open(guard / 'stdout', 'w+b') as out, open(guard / 'stderr', 'w+b') as err:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=out,
                                stderr=err, start_new_session=True)
        timed_out = False
        try:
            proc.communicate(None if stdin is None else stdin.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:  # the command on timeout, and anything it left running
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if timed_out:
            proc.wait()
            return None
        texts = []
        for handle in (out, err):
            handle.seek(0)
            texts.append(handle.read().decode('utf-8', 'replace'))
    return subprocess.CompletedProcess(command, proc.returncode, *texts)


def code_match(prediction: str, test: str, style: str, timeout: float = 20.0,
               memory_mb: int = 2048) -> bool:
    """The predicted program passes the task's tests, run by ``run_sandboxed`` in a
    fresh temporary directory.

    ``style`` ``online_judge``: ``test`` is a Python literal ``{'stdin': [...],
    'stdout': [...]}`` and the program's stripped stdout must match each case;
    otherwise ``test`` is a pytest file importing from ``solution``."""
    code = extract_block(prediction, 'python')
    with tempfile.TemporaryDirectory(prefix='schnitz-verify-') as work:
        root = Path(work)
        (root / 'solution.py').write_text(code, encoding='utf-8')
        if style == 'online_judge':
            import ast
            try:
                cases = ast.literal_eval(test)
            except (ValueError, SyntaxError):
                return False
            for given, wanted in zip(cases['stdin'], cases['stdout'], strict=True):
                run = run_sandboxed([sys.executable, 'solution.py'], root, given, timeout, memory_mb)
                if run is None or run.returncode != 0 or run.stdout.strip() != str(wanted).strip():
                    return False
            return True
        (root / 'test_solution.py').write_text(test, encoding='utf-8')
        run = run_sandboxed([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                             'test_solution.py'], root, None, timeout, memory_mb)
        return run is not None and run.returncode == 0


def knights_knaves_match(prediction: str, names: list[str], solution: list[bool]) -> bool:
    """Every inhabitant is named with the right role (``True`` = knight)."""
    text = re.sub(r'[*_`#>]', '', prediction)  # markdown emphasis and headings
    for name, knight in zip(names, solution, strict=True):
        found = re.findall(rf'\b{re.escape(name)}\b\s*(?:is\s+(?:an?\s+)?|:\s*|-\s*|=\s*)'
                           r'(knight|knave)', text, re.IGNORECASE)
        if not found or (found[-1].lower() == 'knight') != knight:
            return False
    return True


def _norm(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        return f'{value:.4g}'
    return re.sub(r'\s+', ' ', str(value)).strip().lower()


def value_match(prediction: str, rows: list[list], slack: int = 40) -> bool:
    """Every expected cell appears in the answer, which may be at most about three
    times the length of the expected values (listing everything is not an answer).
    Numbers match numerically (integers exactly, others to 4 significant digits)."""
    cells = [_norm(v) for row in rows for v in row if v is not None]
    if not cells:
        return False
    text = re.sub(r'\s+', ' ', prediction).strip().lower()
    if len(text) > 3 * sum(len(c) for c in cells) + slack:
        return False
    numbers = {_norm(float(n)) for n in re.findall(r'-?\d+(?:\.\d+)?', text)}
    return all(c in text or c in numbers for c in cells)


def first_call_match(prediction: str, gold_call: dict) -> bool:
    """The first predicted call equals ``gold_call`` (name and canonical arguments).
    Meaningful only where the gold trajectory's first action is this call."""
    calls = parse_calls(prediction)
    if not calls:
        return False
    key = _call_key(calls[0])
    return key is not None and key == _call_key(gold_call)


def _boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (nested braces allowed)."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, begin = 0, start + len('\\boxed{')
    for index in range(begin - 1, len(text)):
        depth += {'{': 1, '}': -1}.get(text[index], 0)
        if depth == 0:
            return text[begin:index]
    return None


def final_answer(text: str) -> str:
    """The stated final answer: the last ``\\boxed{}``, else the rest of the line after the
    last ``answer:`` / ``answer is``, else the last non-empty line."""
    boxed = _boxed(text)
    if boxed is not None:
        return boxed
    found = re.findall(r'\banswer\b(?:\s+is)?\s*[:=]?\s*(.+)', text, re.IGNORECASE)
    if found:
        return found[-1]
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ''


def _unwrap(text: str) -> str:
    """Whitespace collapsed; markdown bold, ``$...$``, backticks, quotes, the angle
    brackets of a copied ``<answer>`` placeholder and a trailing full stop removed
    (in any nesting)."""
    text = re.sub(r'\s+', ' ', text).strip()
    for _ in range(3):
        text = text.strip().rstrip('.').strip().strip('*').strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in '$`"\'':
            text = text[1:-1]
        elif len(text) >= 3 and text[0] == '<' and text[-1] == '>' and not re.search(r'[<>]', text[1:-1]):
            text = text[1:-1]
    return text.strip()


def _norm_answer(text: str) -> str:
    return _unwrap(text).lower()


def exact_answer_match(prediction: str, answer: str) -> bool:
    """The final answer (``final_answer``) equals ``answer`` after whitespace, case,
    quote/backtick/``$`` and trailing-period normalization; numbers compare numerically
    (relative tolerance 1e-9)."""
    got, want = _norm_answer(final_answer(prediction)), _norm_answer(str(answer))
    if got == want:
        return True
    try:
        a, b = float(got.replace(',', '')), float(want.replace(',', ''))
    except ValueError:
        return False
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


# -- Reasoning Gym: per-family normalizations ---------------------------------------
# ``reasoning_gym`` (and its ``score_answer``) is not installed in the containers, so
# the families of the task corpus get normalizations instead. Families whose
# ``score_answer`` accepts any valid solution (not only the stored one) stay exact
# match and undercount; ``REASONING_GYM_COVERAGE`` lists them.
_RG_ROUNDED = {'advanced_geometry', 'decimal_arithmetic', 'polynomial_equations'}
_RG_UNORDERED_LISTS = {'palindrome_partitioning'}
_RG_SPACELESS = {'checkers_capture', 'rsa_cryptography', 'time_intervals', 'bitwise_arithmetic'}
_RG_SYMBOLIC = {'intermediate_integration', 'polynomial_multiplication', 'simple_integration'}
REASONING_GYM_COVERAGE = {
    # any valid solution scores in Reasoning Gym; here only the stored one
    'multiple_solutions': {'graph_color', 'jugs', 'boxnet', 'n_queens', 'shortest_path',
                           'kakurasu', 'survo', 'letter_jumble', 'tower_of_hanoi'},
    'normalized': {'numbers': 'integers, decimals, fractions, units and degree signs',
                   'rounded': sorted(_RG_ROUNDED), 'grids': 'whitespace-separated rows',
                   'lists': 'JSON/Python literals, comma-separated items',
                   'unordered': sorted(_RG_UNORDERED_LISTS), 'spaceless': sorted(_RG_SPACELESS),
                   'symbolic': sorted(_RG_SYMBOLIC), 'sets': ['minesweeper_deduction'],
                   'complex': ['complex_arithmetic'], 'factors': ['prime_factorization']},
}
_NUMBER = re.compile(r'[-+−]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?')


def _number(text: str) -> Fraction | None:
    """A number written as an integer, decimal or fraction (``a/b`` or ``\\frac{a}{b}``;
    thousands separators, a leading ``x =`` and a trailing unit such as ``°``, ``%``,
    ``m/s^2`` allowed), else ``None``."""
    text = text.strip().replace('−', '-').replace('\\,', '')
    text = re.sub(r'\\d?frac\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}', r'\1/\2', text)
    text = re.sub(r'^[A-Za-z]\w*\s*=\s*|^\$\s*', '', text)
    text = re.sub(r'\s*(?:°|degrees?|%|[a-zA-Z]+(?:/[a-zA-Z]+(?:\^?\d)?)?)$', '', text).strip()
    if not _NUMBER.fullmatch(text):
        return None
    try:
        if '/' in text:
            num, den = text.split('/')
            return Fraction(Fraction(num.replace(',', '').strip()), Fraction(den.strip()))
        return Fraction(text.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def _decimals(text: str) -> int:
    found = re.search(r'\.(\d+)', text)
    return len(found.group(1)) if found else 0


def _same_number(got: str, want: str, rounded: bool) -> bool:
    a, b = _number(got), _number(want)
    if a is None or b is None:
        return False
    if a == b:
        return True
    places = _decimals(want)
    # an answer rounded to ``places`` decimals: the prediction within half a unit
    return rounded and places > 0 and abs(a - b) <= Fraction(1, 2 * 10 ** places)


def _same_items(got: str, want: str, rounded: bool) -> bool:
    """Comma-separated items, equal item by item (numbers numerically)."""
    a = [x.strip() for x in got.strip().strip('[]()').split(',')]
    b = [x.strip() for x in want.strip().strip('[]()').split(',')]
    return len(a) == len(b) > 1 and all(
        x.lower() == y.lower() or _same_number(x, y, rounded) for x, y in zip(a, b))


def _literal(text: str):
    import ast
    text = text.strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def _canonical(value, unordered: bool = False):
    if isinstance(value, (list, tuple)):
        items = [_canonical(v, unordered) for v in value]
        return sorted(items, key=repr) if unordered else items
    if isinstance(value, dict):
        return {str(k): _canonical(v, unordered) for k, v in value.items()}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return Fraction(value) if math.isfinite(value) else value
    return value


def _grid(text: str) -> list[list[str]]:
    rows = [line for line in text.strip().splitlines() if line.strip() and
            not line.strip().startswith('```')]
    return [line.replace(',', ' ').split() for line in rows]


def _minesweeper(text: str) -> dict | None:
    sets = {}
    for key in ('safe', 'mines'):
        found = re.search(rf'{key}\s*:\s*(.*)', text, re.IGNORECASE)
        if not found:
            return None
        sets[key] = set(re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', found.group(1)))
    return sets


def _complex(text: str) -> complex | None:
    text = re.sub(r'\s+', '', text).replace('−', '-').replace('i', 'j')
    try:
        return complex(text)
    except ValueError:
        return None


def _factors(text: str) -> list[int] | None:
    parts = re.split(r'\s*(?:×|\*|x|·)\s*', text.strip())
    return sorted(int(p) for p in parts) if all(p.isdigit() for p in parts) else None


def _same_expression(got: str, want: str) -> bool:
    """Symbolic equality with sympy (when installed), constants of integration
    dropped. Only short expressions of letters, digits and operators are parsed."""
    def clean(text):
        text = re.sub(r'\+\s*C\b', '', text).replace('^', '**').strip()
        # letters, digits and operators only; no dunder names, no huge powers
        return text if len(text) < 400 and re.fullmatch(r'[\w\s+\-*/().]*', text) \
            and '__' not in text and not re.search(r'\*\*\s*\(?\s*\d{4,}', text) else None
    a, b = clean(got), clean(want)
    if a is None or b is None:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        names = {n: sympy.Symbol(n) for n in set(re.findall(r'[A-Za-z_]\w*', a + ' ' + b))
                 if n not in ('exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'pi', 'E', 'atan',
                              'asin', 'acos', 'sinh', 'cosh', 'tanh')}
        return sympy.simplify(parse_expr(a, local_dict=names) - parse_expr(b, local_dict=names)) == 0
    except Exception:  # noqa: BLE001 - sympy missing or an unparsable answer: not equal
        return False


def _rg_answer(prediction: str, lines: int) -> str:
    """The stated answer spanning ``lines`` lines: the last ``\\boxed{}``, else the
    lines after the last ``answer:`` marker, else the last lines of the text."""
    boxed = _boxed(prediction)
    if boxed is not None:
        return boxed
    text = re.sub(r'[*`]{2,}|^\s*```\w*\s*$', '', prediction, flags=re.MULTILINE)
    markers = [m for m in re.finditer(r'\banswer\b(?:\s+is)?\s*[:=]?', text, re.IGNORECASE)
               if text[m.end():].strip()]
    tail = text[markers[-1].end():] if markers else ''
    rows = [line for line in tail.splitlines() if line.strip()]
    if len(rows) >= lines:
        # trailing commentary is cut, but not more rows of the same shape (a longer
        # grid or move list is a different answer)
        if len(rows) > lines and lines > 1 and _shape(rows[lines]) == _shape(rows[lines - 1]):
            return '\n'.join(rows)
        return '\n'.join(rows[:lines])
    rows = [line for line in text.splitlines() if line.strip()]
    return '\n'.join(rows[-lines:])


def _shape(row: str) -> tuple:
    """Token kinds of a row (number, word, other; words themselves where alphabetic
    words lead), so a grid row and a comment differ but two moves agree."""
    tokens = row.split()
    kinds = tuple('d' if t.lstrip('-').isdigit() else 'w' if t.isalpha() else 'o' for t in tokens)
    return kinds, (tokens[0] if tokens and tokens[0].isalpha() else '')


def reasoning_gym_match(prediction: str, answer: str, task: str | None = None) -> bool:
    """``exact_answer_match``, or the stated answer equals ``answer`` after the
    normalization for its shape and family ``task`` (the Reasoning Gym generator):
    numbers numerically (fractions, decimals, degree signs; within half a unit of the
    last stored decimal for families that ask for rounding), grids row by row and
    token by token, JSON/Python literals structurally (unordered where the family
    says any order), comma-separated items item by item, ``minesweeper_deduction``
    sets, complex numbers, prime factorizations, and symbolic expressions (sympy)."""
    if exact_answer_match(prediction, answer):
        return True
    want = str(answer).strip()
    lines = max(1, len([line for line in want.splitlines() if line.strip()]))
    got = _rg_answer(prediction, lines).strip()
    raw = _unwrap(got) if lines == 1 else got
    single = raw.lower() if lines == 1 else got
    rounded = task in _RG_ROUNDED
    if lines == 1 and single == _norm_answer(want):
        return True
    if lines > 1 or (task in ('arc_1d', 'string_synthesis', 'string_splitting', 'rearc')):
        grid = _grid(single)
        if grid and grid == _grid(want):
            return True
    if task == 'minesweeper_deduction':
        mines = _minesweeper(got)
        return mines is not None and mines == _minesweeper(want)
    if task == 'complex_arithmetic':
        a, b = _complex(single), _complex(want)
        return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))
    if task == 'prime_factorization':
        return _factors(single) is not None and _factors(single) == _factors(want)
    if task in _RG_SYMBOLIC:
        return _same_expression(raw, want)
    if task == 'bitwise_arithmetic':
        try:
            return int(single.replace(' ', ''), 16) == int(want, 16)
        except ValueError:
            return False
    if task in _RG_SPACELESS and re.sub(r'\s+', '', single).lower() == re.sub(r'\s+', '', want).lower():
        return True
    if lines == 1 and _same_number(single, want, rounded):
        return True
    if want[:1] in '[{(' and want[-1:] in ']})':
        a, b = _literal(single), _literal(want)
        unordered = task in _RG_UNORDERED_LISTS
        if a is not None and b is not None and _canonical(a, unordered) == _canonical(b, unordered):
            return True
    return lines == 1 and ',' in want and _same_items(single, want, rounded)


# -- SynLogic -------------------------------------------------------------------------
_SYNLOGIC_RUNNER = """import contextlib, importlib, io, json, sys
request = json.loads(sys.stdin.read())
sys.path.insert(0, request['root'])
module, name = request['verifier']
with contextlib.redirect_stdout(io.StringIO()):
    from base.data import Data
    verifier = getattr(importlib.import_module(module), name)()
    score = verifier.verify(Data.from_json_str(request['game_data']), request['answer'])
print(json.dumps({'score': float(score) if isinstance(score, (bool, int, float)) else None}))
"""
# family -> (module, class) of the verifier, as in the repository's task2verifier.py
SYNLOGIC_VERIFIERS = {
    'arc_agi': ('corpus.misc.tasks.arc_agi.scripts.arc_agi_verifier', 'ArcAGIVerifier'),
    'arrow_maze': ('games.tasks.arrow_maze.scripts.arrow_maze_verifier', 'ArrowMazeVerifier'),
    'boolean_expressions': ('games.tasks.boolean_expressions.scripts.boolean_expressions_verifier',
                            'BooleanExpressionsVerifier'),
    'buggy_tables': ('games.tasks.buggy_tables.scripts.game_of_buggy_tables_verifier',
                     'BuggyTableVerifier'),
    'calcudoko': ('games.tasks.calcudoko.scripts.calcudoko_verifier', 'CalcudokoVerifier'),
    'campsite': ('games.tasks.campsite.scripts.campsite_verifier', 'CampsiteVerifier'),
    'cipher': ('games.tasks.cipher.scripts.cipher_verifier', 'CipherVerifier'),
    'cryptarithm': ('games.tasks.cryptarithm.scripts.cryptarithm_verifier', 'CryptarithmVerifier'),
    'dyck_language': ('games.tasks.dyck_language.scripts.dyck_language_verifier',
                      'DyckLanguageVerifier'),
    'dyck_language_errors': ('games.tasks.dyck_language_errors.scripts.dyck_language_errors_verifier',
                             'DyckLanguageErrorsVerifier'),
    'dyck_language_reasoning_errors': (
        'games.tasks.dyck_language_reasoning_errors.scripts.'
        'dyck_language_reasoning_errors_verifier', 'DyckLanguageReasoningErrorsVerifier'),
    'futoshiki': ('games.tasks.futoshiki.scripts.futoshiki_verifier', 'FutoshikiVerifier'),
    'goods_exchange': ('games.tasks.goods_exchange.scripts.goods_exchange_verifier',
                       'GoodsExchangeVerifier'),
    'kukurasu': ('games.tasks.kukurasu.scripts.kukurasu_verifier', 'KukurasuVerifier'),
    'math_path': ('games.tasks.math_path.scripts.math_path_verifier', 'MathPathVerifier'),
    'mathador': ('games.tasks.game_of_24.scripts.game_of_24_verifier', 'GameOf24Verifier'),
    'minesweeper': ('games.tasks.minesweeper.scripts.minesweeper_verifier', 'MinesweeperVerifier'),
    'norinori': ('games.tasks.norinori.scripts.norinori_verifier', 'NorinoriVerifier'),
    'number_wall': ('games.tasks.number_wall.scripts.number_wall_verifier', 'NumberWallVerifier'),
    'numbrix': ('games.tasks.numbrix.scripts.numbrix_verifier', 'NumbrixVerifier'),
    'object_counting': ('games.tasks.object_counting.scripts.object_counting_verifier',
                        'ObjectCountingVerifier'),
    'object_properties': ('games.tasks.object_properties.scripts.object_properties_verifier',
                          'ObjectPropertiesVerifier'),
    'skyscraper_puzzle': ('games.tasks.skyscraper_puzzle.scripts.skyscraper_puzzle_verifier',
                          'SkyscraperPuzzleVerifier'),
    'star_placement_puzzle': (
        'games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_verifier',
        'StarPlacementPuzzleVerifier'),
    'sudoku': ('games.tasks.sudoku.scripts.sudoku_verifier', 'SudokuVerifier'),
    'survo': ('games.tasks.survo.scripts.survo_verifier', 'SurvoVerifier'),
    'time_sequence': ('games.tasks.time_sequence.scripts.time_sequence_verifier',
                      'TimeSequenceVerifier'),
    'web_of_lies': ('games.tasks.web_of_lies.scripts.web_of_lies_verifier', 'WebOfLiesVerifier'),
    'word_sorting': ('games.tasks.word_sorting.scripts.word_sorting_verifier', 'WordSortingVerifier'),
    'word_sorting_mistake': ('games.tasks.word_sorting_mistake.scripts.word_sorting_mistake_verifier',
                             'WordSortingMistakeVerifier'),
    'wordscapes': ('games.tasks.wordscapes.scripts.wordscapes_verifier', 'WordscapesVerifier'),
    'zebra_puzzle': ('corpus.misc.tasks.zebra_puzzle.scripts.zebra_puzzle_verifier',
                     'ZebraPuzzleVerifier'),
}


def _synlogic_answer(prediction: str) -> str:
    """SynLogic's reward extraction: the text after ``</think>``, the content of its
    first ``<answer>...</answer>`` if there is one."""
    text = prediction.split('</think>', 1)[1] if '</think>' in prediction else prediction
    found = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return found.group(1).strip() if found else text


def _synlogic_local(task: str, answer: str, gold: str) -> bool:
    """The families whose repository verifiers import ``math_verify``: the last
    ``\\boxed{}`` against the stored answer (``space_reasoning`` case-insensitively,
    ``space_reasoning_tree`` as a set of comma-separated items, ``operation``
    numerically, math_verify's role)."""
    boxed = _boxed(answer)
    if task == 'operation':
        got = boxed if boxed is not None else final_answer(answer)
        got = re.sub(r'^\$|\$$', '', got.strip())
        return _norm_answer(got) == _norm_answer(gold) or _same_number(got, gold, False)
    if boxed is None:
        return False
    if task == 'space_reasoning':
        return boxed.strip().lower() == gold.lower()
    items = {x for x in boxed.replace('，', ',').replace(' ', '').split(',')}
    return items == set(gold.replace('，', ',').replace(' ', '').split(','))


SYNLOGIC_LOCAL = ('operation', 'space_reasoning', 'space_reasoning_tree')


def synlogic_match(prediction: str, task: str, game_data: str, root: Path | None = None,
                   timeout: float = 30.0) -> bool:
    """SynLogic's accuracy reward: the answer extracted as in the repository's
    reward example (``_synlogic_answer``), scored by the family's verifier from the
    checkout at ``root`` (default ``SYNLOGIC_ROOT``), which runs in the sandbox (the
    verifiers ``eval`` answers); correct only at full score. The format reward (one
    ``<think>`` block, ending with ``</answer>``) is not required. Raises ``ValueError``
    for a family without a verifier or without the checkout."""
    answer = _synlogic_answer(prediction)
    if task in SYNLOGIC_LOCAL:
        return _synlogic_local(task, answer, str(json.loads(game_data)['answer']).strip())
    root = Path(root or SYNLOGIC_ROOT)
    if task not in SYNLOGIC_VERIFIERS or not (root / 'base' / 'verifier.py').exists():
        raise ValueError(f'No SynLogic verifier for {task!r} under {root}')
    request = json.dumps({'root': str(root.resolve()), 'verifier': SYNLOGIC_VERIFIERS[task],
                          'game_data': game_data, 'answer': answer})
    with tempfile.TemporaryDirectory(prefix='schnitz-synlogic-') as work:
        (Path(work) / 'runner.py').write_text(_SYNLOGIC_RUNNER, encoding='utf-8')
        run = run_sandboxed([sys.executable, 'runner.py'], Path(work), request, timeout)
    if run is None or run.returncode != 0 or not run.stdout.strip():
        return False
    score = json.loads(run.stdout.strip().splitlines()[-1]).get('score')
    return score is not None and score >= 1.0


def check_episode(prediction: str, verify: dict) -> bool:
    """Whether ``prediction`` is a correct outcome for an episode's ``verify`` spec.
    Raises ``ValueError`` for spec types without a verifier (and SynLogic families
    without one)."""
    kind = verify['type']
    if kind == 'calls':
        return call_match(prediction, verify['gold'])
    if kind == 'sql':
        return sql_match(prediction, verify['gold'], Path(verify['db']))
    if kind == 'code':
        return code_match(prediction, verify['test'], verify['style'])
    if kind == 'knights':
        return knights_knaves_match(prediction, verify['names'], verify['solution'])
    if kind == 'values':
        return value_match(prediction, verify['rows'])
    if kind == 'exact':
        if 'task' in verify:  # a Reasoning Gym generator named by the evaluator
            return reasoning_gym_match(prediction, verify['answer'], verify['task'])
        return exact_answer_match(prediction, verify['answer'])
    if kind == 'synlogic':
        return synlogic_match(prediction, verify['task'], verify['game_data'])
    if kind == 'tau_bench':
        return bool(verify['calls']) and first_call_match(prediction, verify['calls'][0])
    raise ValueError(f'No verifier for {kind!r}')
