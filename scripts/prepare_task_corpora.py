"""Verifiable task corpora with reference information in the knowledge base (plan B9).

Each dataset becomes a domain with ``sources.jsonl`` (knowledge-base records) and
``episodes-{split}.jsonl``. Reference information goes into the KB, not the
query: tool documentation (xLAM), database schemas with column descriptions,
value lists and table contents (Spider, BIRD) and BIRD's evidence notes,
worked examples and rules (Knights and Knaves). An episode keeps the R6 schema
(``query``, ``answer``, ``required_ids``, ``sufficient_groups``, ``supports``,
``provenance``) with a long teacher ``answer`` (calls, SQL, reasoning) and a
``verify`` spec for ``schnitz.task_verifiers``. ``required_ids`` are the records the
answer depends on; ``supports`` add the causally available related records.

Records are created before every query (``created_at`` 1, queries at 2); none is
built from a teacher answer except worked examples, which come from *other*
(training) tasks and are marked ``kind: worked_example``.

Spider and BIRD take ``--alias-variants N``: schema-aliased variants of every
database (``schnitz.sql_alias``; docs/bgkit-restart-plan.md, B9 corpora), so the
identifiers a query needs come from the KB rather than from the question.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_corpus_common import Writer, clean, record_id  # noqa: E402

from schnitz.sql_alias import rename_create, rename_text  # noqa: E402
from schnitz.task_verifiers import run_sql  # noqa: E402

RAW = Path(os.environ.get('SCHNITZ_RAW', '/archive/raw'))
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


def database_contents(db_path: Path, *, full_rows: int, sample_rows: int,
                      value_limit: int) -> list[dict]:
    """What the records of one database show, read once: per table its CREATE text,
    columns, BIRD column descriptions, value lists of low-cardinality text columns and
    rows (all rows of tables up to ``full_rows`` rows, else a ``sample_rows`` sample)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.text_factory = _lenient
    out = []
    tables = [(n, s) for n, s in con.execute(
        "select name, sql from sqlite_master where type='table' and sql is not null")]
    for table, create in tables:
        notes = _describe(db_path.parent, table)
        columns = [(r[1], r[2]) for r in con.execute(f'pragma table_info("{table}")')]
        count = con.execute(f'select count(*) from "{table}"').fetchone()[0]
        values = []
        for column, kind in columns:
            if 'char' not in (kind or '').lower() and 'text' not in (kind or '').lower():
                continue
            found = [v for (v,) in con.execute(
                f'select distinct "{column}" from "{table}" where "{column}" is not null '
                f'limit {value_limit + 1}')]
            if 0 < len(found) <= value_limit:
                values.append((column, found))
        limit = full_rows if count <= full_rows else sample_rows
        rows = con.execute(f'select * from "{table}" limit {limit}').fetchall()
        label = 'all rows' if count <= full_rows else f'{len(rows)} of {count} rows'
        out.append({'table': table, 'create': create, 'columns': columns, 'notes': notes,
                    'count': count, 'values': values, 'rows': rows, 'label': label})
    con.close()
    return out


def render_database(domain: str, db_id: str, contents: list[dict],
                    alias: dict | None = None) -> dict[str, list[dict]]:
    """Records of one database, by lower-case (original) table name: schema (with column
    descriptions), value lists and rows. ``alias`` (``--alias-variants``): ``db`` (the
    variant's handle), ``names`` (lower-case original -> fresh name, ``sql_alias``) and
    ``variant``; every identifier is renamed and the provenance keeps the originals."""
    names = alias['names'] if alias else {}

    def name(original: str) -> str:
        return names.get(original.lower(), original)

    handle = alias['db'] if alias else db_id

    def prov(table: str, column: str | None = None, **extra) -> dict:
        out = {'db': handle, 'table': name(table)}
        if column is not None:
            out['column'] = name(column)
        out.update(extra)
        if alias:
            out.update({'original_db': db_id, 'original_table': table,
                        **({'original_column': column} if column is not None else {}),
                        'alias_variant': alias['variant']})
        return out

    out: dict[str, list[dict]] = {}
    for info in contents:
        table, columns, notes = info['table'], info['columns'], info['notes']
        create = info['create'].strip()
        if alias:
            create = rename_create(create, names)
        described = [f'- {name(c)} ({t}): '
                     f'{rename_text(notes[c.lower()], names) if alias else notes[c.lower()]}'
                     for c, t in columns if c.lower() in notes]
        text = f'Database {handle}, table {name(table)}:\n{create}'
        if described:
            text += '\nColumn notes:\n' + '\n'.join(described)
        recs = [record(domain, part, 'schema', **prov(table))
                for part in pack(f'Database {handle}, table {name(table)} (schema):\n',
                                 text.split('\n')[1:])]
        for column, values in info['values']:
            recs += [record(domain, part, 'column_values', **prov(table, column))
                     for part in pack(f'Database {handle}, values of {name(table)}.{name(column)}:\n',
                                      [repr(v) for v in values])]
        header = ', '.join(name(c) for c, _ in columns)
        recs += [record(domain, part, 'table_rows', **prov(table, rows=info['count']))
                 for part in pack(f'Database {handle}, table {name(table)} ({info["label"]}; '
                                  f'columns {header}):\n', [repr(tuple(r)) for r in info['rows']])]
        out[table.lower()] = recs
    return out


def database_records(domain: str, db_id: str, db_path: Path, *, full_rows: int,
                     sample_rows: int, value_limit: int) -> dict[str, list[dict]]:
    """Records of one database, by lower-case table name (``render_database``)."""
    return render_database(domain, db_id, database_contents(
        db_path, full_rows=full_rows, sample_rows=sample_rows, value_limit=value_limit))


def _tables_in(sql: str, tables: Iterable[str]) -> list[str]:
    """Tables named in ``sql``: quoted identifiers (backticks, double quotes, brackets)
    or bare words, matched case-insensitively; in ``tables`` order (deterministic)."""
    words = {next(g for g in m if g).lower()
             for m in re.findall(r'`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|(\w+)', sql)}
    return [t for t in tables if t in words]


def _variants(domain: str, db_id: str, contents: list[dict], count: int, style: str,
              handle: str, seed: int, handles: set[str]) -> list[dict]:
    """``count`` aliased variants of one database (``schnitz.sql_alias``): fresh names
    per variant, the handle, the inverse mapping and the rendered records. ``count`` 0
    with an opaque handle: one variant renaming only the database id."""
    from schnitz.sql_alias import alias_db, alias_names, inverse_of
    tables = {info['table']: [c for c, _ in info['columns']] for info in contents}
    spelled = {n.lower(): n for t, cs in tables.items() for n in (t, *cs)}
    out = []
    for v in range(max(count, 1)):
        rng = random.Random(f'{seed}:{domain}:{db_id}:{v}')
        names = alias_names(tables, rng, style) if count else {}
        db = alias_db(db_id, rng, style, handle, handles, spelled.values())
        alias = {'db': db, 'names': names, 'variant': v}
        out.append({**alias, 'inverse': inverse_of(names, spelled),
                    'records': render_database(domain, db_id, contents, alias)})
    return out


def sql_corpus(output: Path, dataset: str, *, full_rows: int, sample_rows: int,
               value_limit: int, alias_variants: int = 0, alias_style: str = 'semantic',
               db_handle: str = 'name', no_handle_fraction: float = 0.0,
               alias_seed: int = 0) -> dict:
    """Text-to-SQL episodes over the databases' records. With ``alias_variants`` N > 0
    every database gets N schema-aliased variants (``schnitz.sql_alias``); each episode
    uses one (seeded by its id) and the KB holds every variant, so retrieval has to find
    the right one. A ``no_handle_fraction`` of the episodes instead uses the original
    database without naming it in the query (the KB then holds the original records
    too). ``db_handle`` names the database in the query: ``name`` (the variant's
    aliased id), ``opaque`` (a seeded code) or ``none`` (only without aliasing)."""
    from schnitz.sql_alias import AliasError, alias_query, rename_text
    if db_handle == 'none' and alias_variants:
        raise ValueError('--db-handle none needs unaliased databases; use --no-handle-fraction')
    aliasing = alias_variants > 0 or db_handle == 'opaque'
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
    variants: dict[tuple[str, str], list[dict]] = {}
    kb_records: dict[tuple[str, str], list[dict]] = {}
    handles: set[str] = set()
    sizes = {'records': 0, 'chars': 0}
    uses = {'plain': 0, 'no_handle': 0, 'aliased': 0}
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            gold = row.get(gold_key) or row.get('query') or row.get('SQL')
            db_id = row['db_id']
            path = locate[split](db_id)
            if not path.exists():
                writer.filters_for(split).reject('missing_database')
                continue
            if (split, db_id) not in cache:
                contents = database_contents(path, full_rows=full_rows, sample_rows=sample_rows,
                                             value_limit=value_limit)
                cache[split, db_id] = render_database(domain, db_id, contents)
                recs = [r for rs in cache[split, db_id].values() for r in rs]
                if aliasing:
                    variants[split, db_id] = _variants(domain, db_id, contents, alias_variants,
                                                       alias_style, db_handle, alias_seed,
                                                       handles)
                    # the KB: every variant (and the original records if some episodes
                    # use the original database without a handle)
                    recs = [r for v in variants[split, db_id] for rs in v['records'].values()
                            for r in rs] + (recs if no_handle_fraction > 0 else [])
                kb_records[split, db_id] = recs
                sizes['records'] += len(recs)
                sizes['chars'] += sum(len(r['text']) for r in recs)
            db = cache[split, db_id]
            used = _tables_in(gold, db)
            if not used:
                writer.filters_for(split).reject('no_table_in_gold')
                continue
            # the episode's form: plain (the defaults), original without a handle, or a variant
            pick = random.Random(f'{alias_seed}:{domain}-{split}-{index}')
            no_handle = db_handle == 'none' or (no_handle_fraction > 0
                                                and pick.random() < no_handle_fraction)
            variant = None if no_handle or not aliasing else \
                variants[split, db_id][pick.randrange(len(variants[split, db_id]))]
            names = variant['names'] if variant else {}
            handle = variant['db'] if variant else db_id
            answer = gold
            if names:
                try:
                    answer = alias_query(gold, names, variant['inverse'])
                except AliasError as error:
                    writer.filters_for(split).reject(f'alias_{error.reason}')
                    continue
            if variant:
                db = variant['records']
            required = [r for t in used for r in db[t] if r['kind'] == 'schema']
            # the KB holds every record of the database; an episode's supports list the
            # used tables' records (schema, values, rows) - the rest stay retrievable
            supports = [r for t in used for r in db[t]]
            everything = kb_records[split, db_id]
            evidence = clean(row.get('evidence', ''))
            if evidence:
                extra = {'original_db': db_id, 'alias_variant': variant['variant']} \
                    if variant else {}
                text = rename_text(evidence, names) if names else evidence
                note = record(domain, f'Database {handle}, note: {text}', 'evidence',
                              db=handle, **extra)
                required.append(note)
                supports.append(note)
            if no_handle:
                query = ('Use the stored database notes. Write one SQLite query that answers '
                         f'the question. Return only the SQL.\nQuestion: {clean(row["question"])}')
            else:
                query = (f'Use the stored notes on database {handle}. Write one SQLite query that '
                         f'answers the question. Return only the SQL.\nQuestion: '
                         f'{clean(row["question"])}')
            verify = {'type': 'sql', 'db': str(path), 'gold': answer}
            if names:
                verify['alias'] = {'variant': variant['variant'], 'db': handle,
                                   'inverse': variant['inverse']}
            extra = {}
            if aliasing or no_handle:
                extra = {'original_db_id': db_id,
                         'alias_variant': variant['variant'] if variant else None,
                         'alias_style': alias_style if names else None,
                         'db_handle': 'none' if no_handle else db_handle}
                uses['no_handle' if no_handle else 'aliased' if variant else 'plain'] += 1
            item = task_episode(domain, split, f'{split}-{index}', query, answer.strip(), required,
                                supports, verify, 'text_to_sql', db_id=handle,
                                difficulty=row.get('difficulty'), **extra)
            writer.add(split, item, everything + [r for r in required if r['kind'] == 'evidence'])
    manifest = {'domain': domain, 'full_rows': full_rows, 'sample_rows': sample_rows,
                'value_limit': value_limit, 'database_records': sizes}
    if aliasing or db_handle != 'name':
        manifest['aliasing'] = {'alias_variants': alias_variants, 'alias_style': alias_style,
                                'db_handle': db_handle, 'no_handle_fraction': no_handle_fraction,
                                'alias_seed': alias_seed, 'episodes': uses}
    return writer.close(manifest)


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
            used = _tables_in(row['query'], db) or list(db)
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


# -- code -----------------------------------------------------------------------
def _fraction(key: str) -> float:
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def kodcode(output: Path, caps: dict[str, int], pool: float, validation: float,
            seed: int) -> dict:
    """KodCode problems with unit tests; a held-out pool of solved problems as worked
    examples in the KB (Python documentation is added by ``add_background.py``)."""
    import pyarrow.parquet as pq
    domain = 'kodcode'
    columns = ['question_id', 'subset', 'style', 'question', 'solution', 'test', 'gpt_difficulty']
    rows = []
    for path in sorted((RAW / 'agentic-20260927/kodcode-v1/data').glob('train-*.parquet')):
        rows += [r for r in pq.read_table(path, columns=columns).to_pylist()
                 if r['style'] in ('instruct', 'online_judge', 'complete')]
    rng = random.Random(seed)
    by_level: dict[str, list] = {}
    for row in rows:
        by_level.setdefault(row['gpt_difficulty'], []).append(row)
    writer = Writer(output, domain)
    examples: dict[str, list[dict]] = {}
    chosen = []
    for level, cap in caps.items():
        group = by_level.get(level, [])
        rng.shuffle(group)
        for row in group[:int(cap * (1 + pool))]:
            if _fraction('pool:' + row['question_id']) < pool / (1 + pool):
                text = (f'Worked example ({row["subset"]}):\n{row["question"][:700]}\n'
                        f'Solution:\n{row["solution"][:700]}')
                examples.setdefault(row['subset'], []).append(
                    record(domain, text, 'worked_example', subset=row['subset']))
            else:
                chosen.append(row)
    everything = [r for rs in examples.values() for r in rs]
    for row in chosen:
        split = 'validation' if _fraction('val:' + row['question_id']) < validation else 'train'
        how = ('Write a Python program that reads from stdin and writes to stdout.'
               if row['style'] == 'online_judge' else 'Write the Python code.')
        query = (f'Use the stored examples and documentation.\n{row["question"]}\n\n{how} '
                 'Put the code in one ```python block.')
        picks = sorted(examples.get(row['subset'], []), key=lambda r: record_id(
            row['question_id'], r['record_id']))[:3]
        item = task_episode(domain, split, row['question_id'], query,
                            f'```python\n{row["solution"].strip()}\n```', [], picks,
                            {'type': 'code', 'test': row['test'], 'style': row['style']},
                            'code', subset=row['subset'], difficulty=row['gpt_difficulty'])
        writer.add(split, item, everything if not writer.sources else picks)
    return writer.close({'domain': domain, 'caps': caps, 'worked_examples': len(everything)})


# -- multi-turn tool use with a domain policy ------------------------------------
def apigen_mt(output: Path, validation: float) -> dict:
    """APIGen-MT (tau-bench airline/retail): the policy and tool documentation live in
    the KB; the customer's first message is the query; the trajectory interleaves the
    simulated customer, assistant replies, calls and results."""
    domain = 'apigen_mt'
    rows = json.load((RAW / 'agentic-20260927/apigen-mt-5k/apigen-mt_5k.json').open())
    writer = Writer(output, domain)
    policies: dict[str, list[dict]] = {}
    tools: dict[str, dict] = {}
    for index, row in enumerate(rows):
        area = 'airline' if 'airline' in row['system'][:200].lower() else 'retail'
        if row['system'] not in policies:
            lines = [line for line in row['system'].split('\n') if line.strip()]
            policies[row['system']] = [record(domain, part, 'policy', area=area)
                                       for part in pack(f'{area} agent policy:\n', lines)]
        specs = json.loads(row['tools']) if isinstance(row['tools'], str) else row['tools']
        docs = {}
        for tool in specs:
            key = f'{area}:{tool["name"]}'
            if key not in tools:
                tools[key] = record(domain, f'{area} tool {tool["name"]}: {tool.get("description", "")}\n'
                                    f'Parameters: {json.dumps(tool.get("parameters", {}))}'[:3000],
                                    'tool_doc', area=area, tool=tool['name'])
            docs[tool['name']] = tools[key]
        conv = row['conversations']
        if not conv or conv[0]['from'] != 'human':
            writer.filters_for('train').reject('no_opening_message')
            continue
        role = {'human': ('environment', 'Customer: '), 'gpt': ('assistant', ''),
                'function_call': ('assistant', 'Call: '), 'observation': ('environment', 'Result: ')}
        turns = [{'role': role[c['from']][0], 'text': role[c['from']][1] + c['value']}
                 for c in conv[1:] if c['from'] in role]
        calls = [json.loads(c['value']) for c in conv if c['from'] == 'function_call']
        used = [docs[c['name']] for c in calls if c.get('name') in docs]
        split = 'validation' if _fraction(f'apigen:{index}') < validation else 'train'
        required = policies[row['system']] + list({r['record_id']: r for r in used}.values())
        query = ('Use the stored agent policy and tool documentation. You are the agent; '
                 f'reply to the customer or call tools.\nCustomer: {conv[0]["value"]}')
        item = task_episode(domain, split, str(index), query, '\n'.join(t['text'] for t in turns),
                            required, required, {'type': 'tau_bench', 'area': area, 'calls': calls},
                            'policy_tool_agent', area=area)
        item['turns'] = turns
        writer.add(split, item, required + list(tools.values()))
    return writer.close({'domain': domain, 'policies': len(policies), 'tools': len(tools)})


# -- reasoning gym and synlogic ----------------------------------------------------
def reasoning_gym(output: Path, pool: float) -> dict:
    """Reasoning Gym SFT rows whose LLM reasoning reached the oracle answer; per
    generator a pool of worked examples in the KB (one rule set per generator)."""
    import pyarrow.parquet as pq
    domain = 'reasoning_gym'
    base = RAW / 'worlds-20260927/multilingual-reasoning-gym-sft-en/en'
    rows = [r for path in sorted(base.rglob('*.parquet')) for r in pq.read_table(path).to_pylist()]
    writer = Writer(output, domain)
    examples: dict[str, list[dict]] = {}
    kept = []
    for i, row in enumerate(rows):
        if str(row.get('llm_boxed_answer')) != str(row.get('oracle_answer')):
            writer.filters_for('train').reject('llm_answer_wrong')
            continue
        if _fraction(f'rg-pool:{i}') < pool:
            examples.setdefault(row['task_name'], []).append(record(
                domain, f'Worked example ({row["task_name"]}):\n{row["question"][:600]}\n'
                f'Answer: {row["oracle_answer"]}', 'worked_example', task=row['task_name']))
        else:
            kept.append((i, row))
    everything = [r for rs in examples.values() for r in rs]
    for i, row in kept:
        split = 'validation' if _fraction(f'rg-val:{i}') < 0.1 else 'train'
        picks = examples.get(row['task_name'], [])[:3]
        query = f'Use the stored worked examples.\n{row["question"]}'
        item = task_episode(domain, split, str(i), query, row['answer'], [], picks,
                            {'type': 'exact', 'answer': str(row['oracle_answer'])}, 'rule_reasoning',
                            task=row['task_name'], level=row.get('curriculum_level'))
        writer.add(split, item, everything if not writer.sources else picks)
    return writer.close({'domain': domain, 'worked_examples': len(everything)})


def synlogic(output: Path, pool: float, min_ascii: float) -> dict:
    """SynLogic puzzles (English prompts): answers from the generator data, verification
    by the SynLogic verifiers (``verify.task`` names the family); worked examples per
    family in the KB."""
    import pyarrow.parquet as pq
    domain = 'synlogic'
    writer = Writer(output, domain)
    examples: dict[str, list[dict]] = {}
    kept = []
    for level in ('easy', 'hard'):
        for split in ('train', 'validation'):
            path = RAW / f'worlds-20260927/synlogic/synlogic_{level}/{split}.parquet'
            for i, row in enumerate(pq.read_table(path).to_pylist()):
                prompt = row['prompt'][0]['content']
                if sum(ch.isascii() for ch in prompt) / max(len(prompt), 1) < min_ascii:
                    writer.filters_for(split).reject('not_english')
                    continue
                data = json.loads(row['extra_info']['game_data_str'])
                answer = str(data.get('answer', '')).strip()
                if not answer:
                    writer.filters_for(split).reject('no_answer')
                    continue
                family = row['data_source'].split('/')[-1]
                ident = f'{level}-{split}-{i}'
                if split == 'train' and _fraction('sl-pool:' + ident) < pool:
                    examples.setdefault(family, []).append(record(
                        domain, f'Worked example ({family}):\n{prompt[-900:]}\nAnswer: {answer[:400]}',
                        'worked_example', family=family))
                else:
                    kept.append((split, ident, family, prompt, answer, row['extra_info']['game_data_str']))
    everything = [r for rs in examples.values() for r in rs]
    for split, ident, family, prompt, answer, game in kept:
        picks = examples.get(family, [])[:3]
        item = task_episode(domain, split, ident, f'Use the stored worked examples.\n{prompt}',
                            f'<answer>{answer}</answer>', [], picks,
                            {'type': 'synlogic', 'task': family, 'game_data': game},
                            'logic_puzzle', puzzle=family)
        writer.add(split, item, everything if not writer.sources else picks)
    return writer.close({'domain': domain, 'worked_examples': len(everything)})


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
        s.add_argument('--alias-variants', type=int, default=0,
                       help='schema-aliased variants per database (0: the original names)')
        s.add_argument('--alias-style', choices=('semantic', 'random'), default='semantic',
                       help='fresh names: synonyms, abbreviations and conventions, or unrelated '
                            'words')
        s.add_argument('--db-handle', choices=('name', 'opaque', 'none'), default='name',
                       help='the database named in the query: its (aliased) name, an opaque '
                            'code, or none (unaliased databases only)')
        s.add_argument('--no-handle-fraction', type=float, default=0.0,
                       help='share of episodes on the original database without a handle')
        s.add_argument('--alias-seed', type=int, default=0,
                       help='seed of the aliases and episode choices (new aliases per '
                            'regeneration)')
    m = sub.add_parser('spider-memory')
    m.add_argument('--max-chars', type=int, default=60000)
    m.add_argument('--max-rows', type=int, default=5)
    m.add_argument('--max-cells', type=int, default=6)
    m.add_argument('--max-answer', type=int, default=160)
    kc = sub.add_parser('kodcode')
    kc.add_argument('--easy', type=int, default=40000)
    kc.add_argument('--medium', type=int, default=20000)
    kc.add_argument('--hard', type=int, default=10000)
    kc.add_argument('--pool', type=float, default=0.05)
    kc.add_argument('--validation', type=float, default=0.02)
    kc.add_argument('--seed', type=int, default=0)
    am = sub.add_parser('apigen-mt')
    am.add_argument('--validation', type=float, default=0.05)
    rg = sub.add_parser('reasoning-gym')
    rg.add_argument('--pool', type=float, default=0.1)
    sl = sub.add_parser('synlogic')
    sl.add_argument('--pool', type=float, default=0.1)
    sl.add_argument('--min-ascii', type=float, default=0.9)
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
    elif args.dataset == 'kodcode':
        manifest = kodcode(args.output, {'easy': args.easy, 'medium': args.medium, 'hard': args.hard},
                           args.pool, args.validation, args.seed)
    elif args.dataset == 'apigen-mt':
        manifest = apigen_mt(args.output, args.validation)
    elif args.dataset == 'reasoning-gym':
        manifest = reasoning_gym(args.output, args.pool)
    elif args.dataset == 'synlogic':
        manifest = synlogic(args.output, args.pool, args.min_ascii)
    elif args.dataset == 'knights':
        manifest = knights(args.output, args.examples, args.seed)
    else:
        manifest = sql_corpus(args.output, args.dataset, full_rows=args.full_rows,
                              sample_rows=args.sample_rows, value_limit=args.value_limit,
                              alias_variants=args.alias_variants, alias_style=args.alias_style,
                              db_handle=args.db_handle,
                              no_handle_fraction=args.no_handle_fraction,
                              alias_seed=args.alias_seed)
    print(json.dumps(manifest, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
