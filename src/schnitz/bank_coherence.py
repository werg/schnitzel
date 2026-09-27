"""Coverage checks for a fully refreshed mutable key and payload journal."""
from __future__ import annotations

from pathlib import Path
import sqlite3


def verify_refresh_coverage(journal: Path, *, namespace: str,
                            spaces: tuple[str, ...], source_count: int,
                            parent_cursor: int) -> None:
    """All logical records must have every view written after the parent cursor."""
    with sqlite3.connect(f'file:{journal}?mode=ro', uri=True) as db:
        counts = dict(db.execute('''SELECT space,COUNT(*) FROM mutable_bank_heads
            WHERE namespace=? AND cursor>? GROUP BY space''',
            (namespace, parent_cursor)).fetchall())
    if set(counts) != set(spaces) or any(counts[space] != source_count
                                           for space in spaces):
        raise ValueError('Refresh is not complete in every bank space')
