"""Inspect or prune mutable-bank revisions below a recovery-safe cursor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from schnitz.store import DiskStore


def inspect(path: Path) -> dict:
    store = DiskStore(path)
    state = store.mutable_bank_state()
    if state is None:
        raise ValueError('The selected SQLite file is not a mutable-bank journal')
    with store.connect() as db:
        pins = [dict(checkpoint=row[0], cursor=row[1]) for row in db.execute(
            'SELECT checkpoint,cursor FROM mutable_bank_checkpoint_pins ORDER BY cursor')]
        revisions, tensor_bytes = db.execute('''SELECT COUNT(*),
            COALESCE(SUM(LENGTH(key)+LENGTH(payload)),0)
            FROM mutable_bank_revisions''').fetchone()
    return state | {'checkpoint_pins': pins, 'revision_views': revisions,
                    'revision_tensor_bytes': tensor_bytes,
                    'physical_sqlite_bytes': store.sizes()['physical_sqlite_bytes']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--before-cursor', type=int)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--vacuum', action='store_true',
                        help='Also rewrite SQLite to return free pages to the filesystem')
    args = parser.parse_args()
    before = inspect(args.journal)
    result = {'before': before, 'applied': False}
    if args.apply:
        if args.before_cursor is None:
            raise ValueError('--apply requires --before-cursor')
        result['garbage_collected'] = DiskStore(args.journal).gc_mutable_bank(
            args.before_cursor)
        if args.vacuum:
            with DiskStore(args.journal).connect() as db:
                db.execute('VACUUM')
        result['after'] = inspect(args.journal)
        result['applied'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
