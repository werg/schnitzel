"""Schema aliasing of text-to-SQL corpora (``schnitz.sql_alias``)."""
from collections import Counter
import random
import sqlite3

import pytest

from schnitz.sql_alias import (SQL_WORDS, AliasError, alias_db, alias_names, alias_query,
                               inverse_of, normalized, rename_create, rename_sql, rename_text,
                               unmapped_originals)
from schnitz.task_verifiers import check_episode

TABLES = {
    'singer': ['Singer_ID', 'Name', 'Country', 'Age', 'Is_male'],
    'concert': ['concert_ID', 'concert_Name', 'Theme', 'Stadium_ID', 'Year'],
    'singer_in_concert': ['concert_ID', 'Singer_ID'],
    'stadium': ['Stadium_ID', 'Location', 'Name', 'Capacity'],
    'frpm': ['CDSCode', 'Free Meal Count (K-12)', 'Enrollment (K-12)'],
    'cast': ['msid', 'role'],
}
SPELLED = {n.lower(): n for t, cs in TABLES.items() for n in (t, *cs)}

QUERIES = [
    'SELECT count(*) FROM singer',
    'SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 '
    'ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id ORDER BY count(*) DESC LIMIT 1',
    'SELECT name FROM singer WHERE singer_id NOT IN (SELECT singer_id FROM singer_in_concert)',
    "SELECT name FROM singer WHERE country = 'name of the Country' AND age > 20",
    'SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm ORDER BY CDSCode',
    'SELECT "Name", [Country] FROM "singer" WHERE Is_male = \'T\'',
    "SELECT strftime('%Y', T1.Year) FROM concert T1 JOIN singer_in_concert T2 "
    'ON T1.concert_ID = T2.concert_ID',
    'SELECT T1.name FROM singer AS T1 WHERE T1.age > (SELECT avg(age) FROM singer)',
    'SELECT name FROM singer INTERSECT SELECT name FROM stadium',
    'SELECT T2.role FROM cast AS T2 WHERE T2.msid = 3',
    'select Name from singer where Country = "France" order by Age',
]


def mapping(seed=0, style='semantic'):
    names = alias_names(TABLES, random.Random(seed), style)
    return names, inverse_of(names, SPELLED)


@pytest.mark.parametrize('style', ['semantic', 'random'])
@pytest.mark.parametrize('seed', range(5))
def test_gold_sql_round_trips(seed, style):
    names, inverse = mapping(seed, style)
    for sql in QUERIES:
        renamed = alias_query(sql, names, inverse)
        assert normalized(rename_sql(renamed, inverse)) == normalized(sql)
        assert not unmapped_originals(renamed, inverse), (sql, renamed)
        assert renamed != sql


def test_literals_and_functions_are_kept():
    names, _ = mapping()
    out = rename_sql("SELECT count(name) FROM singer WHERE country LIKE '%Name%' -- name",
                     names)
    assert "'%Name%'" in out and '-- name' in out and 'count(' in out
    assert f'FROM {names["singer"]}' in out and f'({names["name"]})' in out
    # a keyword table used bare in table position and before a dot
    out = rename_sql('SELECT cast.role FROM cast', names)
    assert out == f'SELECT {names["cast"]}.{names["role"]} FROM {names["cast"]}'
    assert rename_sql('SELECT CAST(age AS REAL) FROM singer', names).startswith('SELECT CAST(')


@pytest.mark.parametrize('sql,reason', [
    ('SELECT name AS age FROM singer', 'alias_collision'),
    ('SELECT * FROM singer name WHERE name.age > 3', 'alias_collision'),
    ('SELECT * FROM singer WHERE name = "Name"', 'quoted_literal_ambiguous'),
    ('SELECT * FROM singer WHERE country IN ("France", "Country")', 'quoted_literal_ambiguous'),
])
def test_ambiguous_rewrites_are_rejected(sql, reason):
    names, inverse = mapping()
    with pytest.raises(AliasError) as caught:
        alias_query(sql, names, inverse)
    assert caught.value.reason == reason


def test_round_trip_failure_is_rejected():
    names, inverse = mapping()
    fresh = names['name']
    # a query alias equal to a fresh name would be renamed back: not invertible
    with pytest.raises(AliasError) as caught:
        alias_query(f'SELECT {fresh}.age FROM singer AS {fresh}', names, inverse)
    assert caught.value.reason == 'alias_roundtrip'


@pytest.mark.parametrize('style', ['semantic', 'random'])
def test_fresh_names_are_unique_new_and_not_keywords(style):
    for seed in range(20):
        names = alias_names(TABLES, random.Random(seed), style)
        assert set(names) == set(SPELLED)
        fresh = [n.lower() for n in names.values()]
        assert len(set(fresh)) == len(fresh)
        squashed = {''.join(ch for ch in o if ch.isalnum()) for o in SPELLED}
        for name in names.values():
            assert name.lower() not in SQL_WORDS
            assert ''.join(ch for ch in name.lower() if ch.isalnum()) not in squashed
            assert name.replace('_', 'a').isalnum() and not name[0].isdigit()
    assert alias_names(TABLES, random.Random(3)) == alias_names(TABLES, random.Random(3))
    assert alias_names(TABLES, random.Random(3)) != alias_names(TABLES, random.Random(4))


def test_database_handles():
    taken: set[str] = set()
    handles = [alias_db('concert_singer', random.Random(v), 'semantic', 'name', taken)
               for v in range(4)]
    assert len(set(handles)) == 4 and 'concert_singer' not in handles
    code = alias_db('concert_singer', random.Random(0), 'semantic', 'opaque', taken)
    assert code.startswith('db_') and len(code) == 9


def test_create_table_is_renamed_positionally():
    names, _ = mapping()
    create = ('CREATE TABLE "singer_in_concert" (\n"concert_ID" int,\nSinger_ID text NOT NULL,\n'
              'PRIMARY KEY ("concert_ID","Singer_ID"),\n'
              'FOREIGN KEY ("concert_ID") REFERENCES "concert"("concert_ID")\n)')
    out = rename_create(create, names)
    c, s = names['concert_id'], names['singer_id']
    assert out == (f'CREATE TABLE "{names["singer_in_concert"]}" (\n"{c}" int,\n{s} text NOT NULL,\n'
                   f'PRIMARY KEY ("{c}","{s}"),\n'
                   f'FOREIGN KEY ("{c}") REFERENCES "{names["concert"]}"("{c}")\n)')
    # type words stay even when a column has the same name
    typed = alias_names({'t': ['text', 'real']}, random.Random(0))
    out = rename_create('CREATE TABLE t (text text, real real)', typed)
    assert out == f'CREATE TABLE {typed["t"]} ({typed["text"]} text, {typed["real"]} real)'


def test_notes_and_evidence_rename_identifier_mentions_only():
    names, _ = mapping()
    text = ('rate = `Free Meal Count (K-12)` / `Enrollment (K-12)`; the name of the singer; '
            "CDSCode is the code; singer.Age > 30; Country = 'France'; age of the singer")
    out = rename_text(text, names)
    assert f'`{names["free meal count (k-12)"]}`' in out
    assert 'the name of the singer' in out and 'age of the singer' in out
    assert f'{names["cdscode"]} is the code' in out
    assert f'{names["singer"]}.{names["age"]} > 30' in out
    assert f"{names['country']} = 'France'" in out


def test_aliased_answer_verifies_on_the_original_database(tmp_path):
    db = tmp_path / 'db.sqlite'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE singer (Singer_ID int, Name text, Country text, Age int)')
    con.executemany('INSERT INTO singer VALUES (?, ?, ?, ?)',
                    [(1, 'A', 'France', 30), (2, 'B', 'Peru', 40), (3, 'C', 'France', 50)])
    con.commit()
    con.close()
    tables = {'singer': ['Singer_ID', 'Name', 'Country', 'Age']}
    names = alias_names(tables, random.Random(1))
    inverse = inverse_of(names, {n.lower(): n for n in ('singer', *tables['singer'])})
    gold = alias_query("SELECT name FROM singer WHERE country = 'France'", names, inverse)
    verify = {'type': 'sql', 'db': str(db), 'gold': gold,
              'alias': {'variant': 0, 'db': 'x', 'inverse': inverse}}
    s, n, a = names['singer'], names['name'], names['age']
    assert check_episode(f'```sql\nSELECT {n} FROM {s} WHERE {a} != 40\n```', verify)
    assert not check_episode(f'SELECT {n} FROM {s}', verify)
    # guessing the original schema does not verify on an aliased task
    assert not check_episode("SELECT Name FROM singer WHERE Country = 'France'", verify)
    assert Counter(unmapped_originals('SELECT Name FROM singer', inverse)) == \
        Counter({'name', 'singer'})


def _corpus_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_task_corpora.py'
    spec = importlib.util.spec_from_file_location('prepare_task_corpora', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spider(root):
    import json
    base = root / 'agentic-20260927/spider/official/spider_data'
    rows = {'train_spider.json': [], 'train_others.json': [], 'dev.json': []}
    for db_id, split in (('concert_singer', 'train_spider.json'), ('pets', 'dev.json')):
        folder = base / 'database' / db_id
        folder.mkdir(parents=True)
        con = sqlite3.connect(folder / f'{db_id}.sqlite')
        con.execute('CREATE TABLE "singer" ("Singer_ID" int, "Name" text, "Country" text, '
                    'PRIMARY KEY ("Singer_ID"))')
        con.execute('CREATE TABLE concert (concert_ID int, Singer_ID int, Year text, '
                    'FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID))')
        con.executemany('INSERT INTO singer VALUES (?, ?, ?)', [(1, 'Ann', 'France'),
                                                                (2, 'Bo', 'Peru')])
        con.executemany('INSERT INTO concert VALUES (?, ?, ?)', [(1, 1, '2014'), (2, 2, '2015')])
        con.commit()
        con.close()
        rows[split] += [{'db_id': db_id, 'question': f'Question {i}?', 'query': q}
                        for i, q in enumerate([
                            "SELECT Name FROM singer WHERE Country = 'France'",
                            'SELECT T1.Name FROM singer AS T1 JOIN concert AS T2 ON '
                            'T1.Singer_ID = T2.Singer_ID WHERE T2.Year = "2015"',
                            'SELECT count(*) FROM concert'] * 4)]
    for name, content in rows.items():
        (base / name).write_text(json.dumps(content))


def test_sql_corpus_aliases_every_database(tmp_path):
    import json
    tc = _corpus_script()
    tc.RAW = tmp_path / 'raw'
    _spider(tc.RAW)
    plain = tc.sql_corpus(tmp_path / 'plain', 'spider', full_rows=60, sample_rows=8,
                          value_limit=40)
    assert 'aliasing' not in plain
    manifest = tc.sql_corpus(tmp_path / 'alias', 'spider', full_rows=60, sample_rows=8,
                             value_limit=40, alias_variants=3, no_handle_fraction=0.3)
    assert manifest['aliasing']['episodes']['aliased'] > 0
    assert manifest['aliasing']['episodes']['no_handle'] > 0
    sources = [json.loads(line) for line in (tmp_path / 'alias/sources.jsonl').open()]
    plain_sources = [json.loads(line) for line in (tmp_path / 'plain/sources.jsonl').open()]
    # the KB: three variants of every database (train and dev) plus the original records
    kb = [s for s in sources if s['kind'] != 'evidence']
    assert len(kb) == 4 * len(plain_sources)
    variants = {(s['provenance'].get('original_db'), s['provenance'].get('alias_variant'))
                for s in kb}
    assert variants == {(db, v) for db in ('concert_singer', 'pets') for v in (0, 1, 2)} | \
        {(None, None)}
    for s in kb:
        prov = s['provenance']
        if 'original_db' in prov:
            assert prov['db'] != prov['original_db'] and prov['table'] != prov['original_table']
            header = s['text'].split('\n')[0]
            assert 'singer' not in header.lower() and 'concert' not in header.lower()
    for split in ('train', 'validation'):
        for line in (tmp_path / f'alias/episodes-{split}.jsonl').open():
            row = json.loads(line)
            prov, verify = row['provenance'], row['verify']
            if prov['db_handle'] == 'none':
                assert prov['original_db_id'] not in row['query']
                assert 'alias' not in verify
                continue
            assert f'database {prov["db_id"]}.' in row['query']
            assert prov['db_id'] != prov['original_db_id']
            texts = [s['text'] for s in row['supports']]
            assert all(t.startswith(f'Database {prov["db_id"]},') for t in texts)
            assert check_episode(row['answer'], verify)
            assert verify['alias']['db'] == prov['db_id']
