"""Verifiable task corpora with reference information in the knowledge base (plan B9).

Each dataset becomes a domain with ``sources.jsonl`` (knowledge-base records) and
``episodes-{split}.jsonl``. Reference information goes into the KB, not the
query: tool documentation (xLAM), database schemas with column descriptions,
value lists and table contents (Spider, BIRD) and BIRD's evidence notes,
worked examples and rules (Knights and Knaves). An episode keeps the R6 schema
(``query``, ``answer``, ``required_ids``, ``sufficient_groups``, ``supports``,
``provenance``) with a long teacher ``answer`` (calls, SQL, reasoning) and a
``verify`` spec for ``sdkb.task_verifiers``. ``required_ids`` are the records the
answer depends on; ``supports`` add the causally available related records.

Records are created before every query (``created_at`` 1, queries at 2); none is
built from a teacher answer except worked examples, which come from *other*
(training) tasks and are marked ``kind: worked_example``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_corpus_common import Writer, clean, record_id  # noqa: E402

from sdkb.task_verifiers import run_sql  # noqa: E402

RAW = Path('/archive/raw')
RECORD_CHARS = 1500


def _lit(value):
    """A list field that some releases store as its Python literal string."""
    return __import__('ast').literal_eval(value) if isinstance(value, str) else value


def _lenient(raw: bytes) -> str:
    """Text columns with stray non-UTF-8 bytes (some Spider rows) decode with replacement."""
    return raw.decode('utf-8', 'replace')


def record(domain: str, text: str, kind: str, **provenance) -> dict:
    text = text.strip()
    if not text:
        raise ValueError('Empty record')
    return {'record_id': record_id(domain, text), 'text': text, 'domain': domain,
            'created_at': 1, 'kind': kind, 'provenance': {'dataset': domain, **provenance}}


def pack(header: str, lines: list[str], limit: int = RECORD_CHARS) -> list[str]:
    """Split ``lines`` into texts of at most ``limit`` characters, each starting with ``header``."""
    texts, current = [], []
    for line in lines:
        if current and len(header) + sum(len(x) + 1 for x in current) + len(line) > limit:
            texts.append(header + '\n'.join(current))
            current = []
        current.append(line[:limit - len(header) - 1])
    if current:
        texts.append(header + '\n'.join(current))
    return texts


def task_episode(domain: str, split: str, identifier: str, query: str, answer: str,
                 required: list[dict], supports: list[dict], verify: dict, family: str,
                 **provenance) -> dict:
    everything = {r['record_id']: r for r in (*required, *supports)}
    required_ids = list(dict.fromkeys(r['record_id'] for r in required))
    return {
        'episode_id': f'{domain}-{identifier}', 'environment': f'{domain}-{split}',
        'query': query, 'answer': answer, 'query_time': 2,
        'required_ids': required_ids, 'sufficient_groups': [required_ids],
        'support_annotation': 'verified', 'task_family': family,
        'supports': [{'record_id': k, 'text': r['text'], 'created_at': r['created_at'],
                      'kind': r['kind']} for k, r in everything.items()],
        'verify': verify,
        'provenance': {'dataset': domain, 'domain': domain, 'split': split, **provenance},
    }


# -- function calling ----------------------------------------------------------
def xlam(output: Path, validation: int) -> dict:
    domain = 'xlam'
    rows = json.load((RAW / 'agentic-20260927/xlam-function-calling-60k/'
                      'xlam_function_calling_60k.json').open())
    writer = Writer(output, domain)
    tools_by_name: dict[str, dict] = {}
    for index, row in enumerate(rows):
        split = 'validation' if index >= len(rows) - validation else 'train'
        offered = []
        for tool in json.loads(row['tools']):
            text = (f'Tool: {tool["name"]}\nDescription: {clean(tool.get("description", ""))}\n'
                    f'Parameters: {json.dumps(tool.get("parameters", {}), ensure_ascii=False)}')
            rec = record(domain, text[:RECORD_CHARS * 2], 'tool_doc', tool=tool['name'])
            tools_by_name.setdefault(tool['name'], rec)
            offered.append(rec)
        calls = json.loads(row['answers'])
        used = {call['name'] for call in calls}
        required = [r for r in offered if r['provenance']['tool'] in used]
        if not required:
            writer.filters_for(split).reject('no_used_tool')
            continue
        query = ('Use the stored tool documentation. Answer with a JSON list of calls '
                 '[{"name": ..., "arguments": {...}}] and nothing else.\nRequest: '
                 + row['query'])
        item = task_episode(domain, split, str(row['id']), query, json.dumps(calls),
                            required, offered, {'type': 'calls', 'gold': calls}, 'function_call')
        writer.add(split, item, offered)
    return writer.close({'domain': domain, 'records': 'one per distinct tool documentation',
                         'distinct_tool_names': len(tools_by_name)})


# -- text to SQL ----------------------------------------------------------------
def _describe(db_dir: Path, table: str) -> dict[str, str]:
    """BIRD column descriptions, keyed by lower-case column name."""
    path = db_dir / 'database_description' / f'{table}.csv'
    if not path.exists():
        return {}
    notes = {}
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            with path.open(encoding=encoding, newline='') as handle:
                for row in csv.DictReader(handle):
                    name = (row.get('original_column_name') or '').strip()
                    parts = [clean(row.get(k) or '') for k in
                             ('column_description', 'value_description')]
                    parts = [p for p in parts if p and p.lower() != 'nan']
                    if name and parts:
                        notes[name.lower()] = '; '.join(parts)
            return notes
        except UnicodeDecodeError:
            notes = {}
    return notes


def database_records(domain: str, db_id: str, db_path: Path, *, full_rows: int,
                     sample_rows: int, value_limit: int) -> dict[str, list[dict]]:
    """Records of one database, by lower-case table name: schema (with column
    descriptions), value lists of low-cardinality text columns, and rows (all rows
    of tables up to ``full_rows`` rows, else a ``sample_rows`` sample)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.text_factory = _lenient
    out: dict[str, list[dict]] = {}
    tables = [(n, s) for n, s in con.execute(
        "select name, sql from sqlite_master where type='table' and sql is not null")]
    for table, create in tables:
        notes = _describe(db_path.parent, table)
        columns = [(r[1], r[2]) for r in con.execute(f'pragma table_info("{table}")')]
        described = [f'- {c} ({t}): {notes[c.lower()]}' for c, t in columns if c.lower() in notes]
        text = f'Database {db_id}, table {table}:\n{create.strip()}'
        if described:
            text += '\nColumn notes:\n' + '\n'.join(described)
        recs = [record(domain, part, 'schema', db=db_id, table=table)
                for part in pack(f'Database {db_id}, table {table} (schema):\n',
                                 text.split('\n')[1:])]
        count = con.execute(f'select count(*) from "{table}"').fetchone()[0]
        for column, kind in columns:
            if 'char' not in (kind or '').lower() and 'text' not in (kind or '').lower():
                continue
            values = [v for (v,) in con.execute(
                f'select distinct "{column}" from "{table}" where "{column}" is not null '
                f'limit {value_limit + 1}')]
            if 0 < len(values) <= value_limit:
                recs += [record(domain, part, 'column_values', db=db_id, table=table, column=column)
                         for part in pack(f'Database {db_id}, values of {table}.{column}:\n',
                                          [repr(v) for v in values])]
        names = ', '.join(c for c, _ in columns)
        limit = full_rows if count <= full_rows else sample_rows
        rows = con.execute(f'select * from "{table}" limit {limit}').fetchall()
        label = 'all rows' if count <= full_rows else f'{len(rows)} of {count} rows'
        recs += [record(domain, part, 'table_rows', db=db_id, table=table, rows=count)
                 for part in pack(f'Database {db_id}, table {table} ({label}; columns {names}):\n',
                                  [repr(tuple(r)) for r in rows])]
        out[table.lower()] = recs
    con.close()
    return out


def _tables_in(sql: str, tables: set[str]) -> list[str]:
    """Tables named in ``sql``: quoted identifiers (backticks, double quotes, brackets)
    or bare words, matched case-insensitively."""
    words = {next(g for g in m if g).lower()
             for m in re.findall(r'`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|(\w+)', sql)}
    return [t for t in tables if t in words]


def sql_corpus(output: Path, dataset: str, *, full_rows: int, sample_rows: int,
               value_limit: int) -> dict:
    domain = dataset
    if dataset == 'spider':
        base = RAW / 'agentic-20260927/spider/official/spider_data'
        splits = {'train': [*json.load((base / 'train_spider.json').open()),
                            *json.load((base / 'train_others.json').open())],
                  'validation': json.load((base / 'dev.json').open())}
        locate = {s: (lambda db, b=base: b / 'database' / db / f'{db}.sqlite') for s in splits}
        gold_key = 'query'
    else:
        dev = RAW / 'agentic-20260927/bird/dev_20240627'
        train = RAW / 'agentic-20260927/bird/train'
        filtered = RAW / 'agentic-20260927/bird/hf-bird23-train-filtered/data'
        rows = [json.loads(line) for path in sorted(filtered.glob('*.jsonl'))
                for line in path.open()]  # the cleaner filtered release
        if not rows:
            rows = json.load((train / 'train.json').open())
        splits = {'train': rows, 'validation': json.load((dev / 'dev.json').open())}
        locate = {'train': lambda db: train / 'train_databases' / db / f'{db}.sqlite',
                  'validation': lambda db: dev / 'dev_databases' / db / f'{db}.sqlite'}
        gold_key = 'SQL'
    writer = Writer(output, domain)
    cache: dict[tuple[str, str], dict[str, list[dict]]] = {}
    sizes = {'records': 0, 'chars': 0}
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            gold = row.get(gold_key) or row.get('query') or row.get('SQL')
            db_id = row['db_id']
            path = locate[split](db_id)
            if not path.exists():
                writer.filters_for(split).reject('missing_database')
                continue
            if (split, db_id) not in cache:
                cache[split, db_id] = database_records(domain, db_id, path, full_rows=full_rows,
                                                       sample_rows=sample_rows,
                                                       value_limit=value_limit)
                recs = [r for rs in cache[split, db_id].values() for r in rs]
                sizes['records'] += len(recs)
                sizes['chars'] += sum(len(r['text']) for r in recs)
            db = cache[split, db_id]
            used = _tables_in(gold, set(db))
            if not used:
                writer.filters_for(split).reject('no_table_in_gold')
                continue
            required = [r for t in used for r in db[t] if r['kind'] == 'schema']
            # the KB holds every record of the database; an episode's supports list the
            # used tables' records (schema, values, rows) - the rest stay retrievable
            supports = [r for t in used for r in db[t]]
            everything = [r for rs in db.values() for r in rs]
            evidence = clean(row.get('evidence', ''))
            if evidence:
                note = record(domain, f'Database {db_id}, note: {evidence}', 'evidence', db=db_id)
                required.append(note)
                supports.append(note)
            query = (f'Use the stored notes on database {db_id}. Write one SQLite query that '
                     f'answers the question. Return only the SQL.\nQuestion: {clean(row["question"])}')
            verify = {'type': 'sql', 'db': str(path), 'gold': gold}
            item = task_episode(domain, split, f'{split}-{index}', query, gold.strip(), required,
                                supports, verify, 'text_to_sql', db_id=db_id,
                                difficulty=row.get('difficulty'))
            writer.add(split, item, everything + [r for r in required if r['kind'] == 'evidence'])
    return writer.close({'domain': domain, 'full_rows': full_rows, 'sample_rows': sample_rows,
                         'value_limit': value_limit, 'database_records': sizes})


# -- answers from stored database contents --------------------------------------
def spider_memory(output: Path, *, max_chars: int, max_rows: int, max_cells: int,
                  max_answer: int) -> dict:
    """Small Spider databases stored whole in the KB (schema and all rows); questions
    answered from memory with the values, no SQL. A test of how precisely stored
    facts are recalled. Questions whose gold result is short (up to ``max_rows``
    rows, ``max_cells`` cells, ``max_answer`` characters) and non-empty."""
    domain = 'spider_memory'
    base = RAW / 'agentic-20260927/spider/official/spider_data'
    splits = {'train': [*json.load((base / 'train_spider.json').open()),
                        *json.load((base / 'train_others.json').open())],
              'validation': json.load((base / 'dev.json').open())}
    writer = Writer(output, domain)
    stored: dict[str, dict[str, list[dict]] | None] = {}
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            db_id = row['db_id']
            path = base / 'database' / db_id / f'{db_id}.sqlite'
            if db_id not in stored:
                con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
                con.text_factory = _lenient
                size = sum(len(repr(tuple(r))) for (t,) in con.execute(
                    "select name from sqlite_master where type='table'")
                    for r in con.execute(f'select * from "{t}"'))
                con.close()
                stored[db_id] = (database_records(domain, db_id, path, full_rows=10 ** 9,
                                                  sample_rows=0, value_limit=0)
                                 if size <= max_chars else None)
            db = stored[db_id]
            if db is None:
                writer.filters_for(split).reject('database_too_large')
                continue
            result = run_sql(path, row['query'])
            cells = [v for r in (result or []) for v in r]
            answer = '; '.join(', '.join(str(v) for v in r) for r in (result or []))
            if (not result or len(result) > max_rows or len(cells) > max_cells
                    or any(v is None for v in cells) or len(answer) > max_answer):
                writer.filters_for(split).reject('answer_not_short')
                continue
            used = _tables_in(row['query'], set(db)) or list(db)
            required = [r for t in used for r in db[t]]
            everything = [r for rs in db.values() for r in rs]
            query = (f'Use the stored contents of database {db_id}. Answer the question with '
                     f'the values only.\nQuestion: {clean(row["question"])}')
            item = task_episode(domain, split, f'{split}-{index}', query, answer, required, required,
                                {'type': 'values', 'rows': [list(r) for r in result]},
                                'stored_table_qa', db_id=db_id)
            writer.add(split, item, everything)
    kept = {k: v for k, v in stored.items() if v is not None}
    return writer.close({'domain': domain, 'max_chars': max_chars, 'databases_stored': len(kept),
                         'database_records': sum(len(r) for d in kept.values() for r in d.values())})


# -- knights and knaves ---------------------------------------------------------
RULES = ('Knights and knaves: every inhabitant is either a knight, who always tells the '
         'truth, or a knave, who always lies. To solve a puzzle, assume a role for one '
         'person, derive what their statement implies about the others, and reject any '
         'assumption that leads to a contradiction; the solution assigns every person a '
         'role consistent with all statements.')


def knights(output: Path, examples: int, seed: int) -> dict:
    domain = 'knights'
    base = RAW / 'worlds-20260927/knights-and-knaves'
    rng = random.Random(seed)
    writer = Writer(output, domain)
    rules = record(domain, RULES, 'rules')
    train = {p: [json.loads(line) for line in path.open()] for p, path in
             ((int(re.search(r'people(\d+)', f.name).group(1)), f)
              for f in sorted((base / 'train').glob('people*.jsonl')))}
    worked = []
    for people, rows in train.items():
        for row in rng.sample(rows, min(examples, len(rows))):
            steps = '\n'.join(_lit(row['cot_repeat_steps']))
            worked.append((people, row['index'], record(
                domain, f'Worked example ({people} people):\n{row["quiz"]}\n{row["cot_head"]}\n'
                f'{steps}\n{row["cot_foot"]}\n{row["solution_text_format"]}', 'worked_example',
                people=people)))
    held = {(p, i) for p, i, _ in worked}
    for split, folder in (('train', 'train'), ('validation', 'test')):
        for path in sorted((base / folder).glob('people*.jsonl')):
            people = int(re.search(r'people(\d+)', path.name).group(1))
            for row in (json.loads(line) for line in path.open()):
                if split == 'train' and (people, row['index']) in held:
                    continue
                names, solution = _lit(row['names']), _lit(row['solution'])
                steps = '\n'.join(_lit(row['cot_repeat_steps']))
                answer = f'{row["cot_head"]}\n{steps}\n{row["cot_foot"]}\n{row["solution_text_format"]}'
                examples_here = [w for p, _, w in worked if p == people]
                query = ('Use the stored rules and worked examples. Solve the puzzle and answer '
                         'with one line per person: "(1) Name is a knight/knave".\n' + row['quiz'])
                item = task_episode(domain, split, f'{split}-{people}-{row["index"]}', query, answer,
                                    [rules], [rules, *examples_here],
                                    {'type': 'knights', 'names': names, 'solution': solution},
                                    'logic_puzzle', people=people)
                writer.add(split, item, [rules, *examples_here])
    return writer.close({'domain': domain, 'worked_examples': len(worked)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='dataset', required=True)
    x = sub.add_parser('xlam')
    x.add_argument('--validation', type=int, default=2000)
    for name in ('spider', 'bird'):
        s = sub.add_parser(name)
        s.add_argument('--full-rows', type=int, default=60)
        s.add_argument('--sample-rows', type=int, default=8)
        s.add_argument('--value-limit', type=int, default=40)
    m = sub.add_parser('spider-memory')
    m.add_argument('--max-chars', type=int, default=60000)
    m.add_argument('--max-rows', type=int, default=5)
    m.add_argument('--max-cells', type=int, default=6)
    m.add_argument('--max-answer', type=int, default=160)
    k = sub.add_parser('knights')
    k.add_argument('--examples', type=int, default=20)
    k.add_argument('--seed', type=int, default=0)
    for s in sub.choices.values():
        s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == 'xlam':
        manifest = xlam(args.output, args.validation)
    elif args.dataset == 'spider-memory':
        manifest = spider_memory(args.output, max_chars=args.max_chars, max_rows=args.max_rows,
                                 max_cells=args.max_cells, max_answer=args.max_answer)
    elif args.dataset == 'knights':
        manifest = knights(args.output, args.examples, args.seed)
    else:
        manifest = sql_corpus(args.output, args.dataset, full_rows=args.full_rows,
                              sample_rows=args.sample_rows, value_limit=args.value_limit)
    print(json.dumps(manifest, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
