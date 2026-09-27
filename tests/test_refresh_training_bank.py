import sqlite3

import pytest

from schnitz.bank_coherence import verify_refresh_coverage


def test_refresh_coverage_requires_new_heads_for_every_space(tmp_path):
    path = tmp_path / 'journal.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE mutable_bank_heads '
                   '(namespace TEXT, record_id TEXT, space TEXT, cursor INTEGER)')
        db.executemany('INSERT INTO mutable_bank_heads VALUES (?,?,?,?)', (
            ('corpus', 'a', 's0', 11), ('corpus', 'a', 's1', 11),
            ('corpus', 'b', 's0', 12), ('corpus', 'b', 's1', 12)))
    verify_refresh_coverage(path, namespace='corpus', spaces=('s0', 's1'),
                            source_count=2, parent_cursor=10)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE mutable_bank_heads SET cursor=10 WHERE record_id='b' "
                   "AND space='s1'")
    with pytest.raises(ValueError, match='complete'):
        verify_refresh_coverage(path, namespace='corpus', spaces=('s0', 's1'),
                                source_count=2, parent_cursor=10)
