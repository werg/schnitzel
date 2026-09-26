"""Transactional disk payload store and a streaming exact-search reference.

SQLite stores safetensors payloads, not Python pickle. Search scans eligible keys
on CPU in bounded chunks; it is a correctness baseline, not an ANN index or a
claim of cold-disk throughput. No source trajectory is exposed on the read path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor
from safetensors.torch import load, save


@dataclass(frozen=True)
class Selection:
    record_id: str
    score: float


@dataclass(frozen=True)
class ReadPlan:
    namespace: str
    space: str
    generation: str
    domain: str
    query_time: int
    selections: tuple[Selection, ...]
    bank_cursor: int | None = None


@dataclass
class StoredRecord:
    record_id: str
    key: Tensor
    payload: Tensor
    namespace: str = "default"
    space: str = "s0"
    generation: str = "v0"
    domain: str = "research"
    created_at: int = 0
    source_id: str = ""


class DiskStore:
    def __init__(self, path: str | Path, *, cache_mib: int = 32,
                 secure_delete: bool | None = None) -> None:
        self.path = Path(path)
        if cache_mib < 1:
            raise ValueError("SQLite cache budget must be positive")
        self.cache_mib = cache_mib
        # None keeps SQLite's build default (on in Debian/Ubuntu builds). A training
        # journal of our own payload revisions may turn it off: with it on, garbage
        # collection overwrites every freed page with zeros, twice through the WAL.
        self.secure_delete = secure_delete
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS records (
                namespace TEXT NOT NULL, record_id TEXT NOT NULL, space TEXT NOT NULL,
                generation TEXT NOT NULL, domain TEXT NOT NULL, created_at INTEGER NOT NULL,
                key BLOB NOT NULL, key_dim INTEGER NOT NULL, payload BLOB NOT NULL,
                source_id TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(namespace,record_id,space,generation));
            CREATE INDEX IF NOT EXISTS eligible ON records
                (namespace,space,generation,domain,deleted,created_at);
            CREATE TABLE IF NOT EXISTS lineage (
                namespace TEXT NOT NULL, parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
                PRIMARY KEY(namespace,parent_id,child_id));
            CREATE TABLE IF NOT EXISTS tombstones (
                namespace TEXT NOT NULL, record_id TEXT NOT NULL,
                PRIMARY KEY(namespace,record_id));
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute(f"PRAGMA cache_size={-self.cache_mib * 1024}")
        db.execute("PRAGMA mmap_size=0")  # no unaccounted application mmap cache
        if self.secure_delete is not None:
            db.execute(f"PRAGMA secure_delete={'ON' if self.secure_delete else 'OFF'}")
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, record: StoredRecord, *, children: tuple[str, ...] = ()) -> None:
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._put(db, record, children)

    def put_many(self, records: Iterable[StoredRecord]) -> None:
        """Stream raw records through one atomic transaction for offline bank creation.

        No partial batch is visible or retained after an exception. Derived
        compact records still use put(..., children=...) to record their lineage.
        """
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for record in records:
                self._put(db, record, ())

    @staticmethod
    def _event_digest(records: list[StoredRecord]) -> str:
        digest = hashlib.sha256()
        for record in sorted(records, key=lambda item: (item.record_id, item.space)):
            key = record.key.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
            payload = save({"payload": record.payload.detach().cpu().contiguous()})
            metadata = (record.namespace, record.record_id, record.space, record.generation,
                        record.domain, record.created_at, record.source_id)
            digest.update(json.dumps(metadata, separators=(",", ":")).encode())
            digest.update(key.tobytes())
            digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _ensure_event_tables(db) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS event_streams (
                namespace TEXT NOT NULL, stream TEXT NOT NULL, generation TEXT NOT NULL,
                next_position INTEGER NOT NULL, visibility_time INTEGER NOT NULL,
                PRIMARY KEY(namespace,stream,generation));
            CREATE TABLE IF NOT EXISTS event_commits (
                namespace TEXT NOT NULL, stream TEXT NOT NULL, generation TEXT NOT NULL,
                position INTEGER NOT NULL, event_id TEXT NOT NULL, visibility_time INTEGER NOT NULL,
                record_ids TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                PRIMARY KEY(namespace,stream,generation,position),
                UNIQUE(namespace,stream,generation,event_id));
            CREATE TABLE IF NOT EXISTS event_lineage (
                namespace TEXT NOT NULL, generation TEXT NOT NULL,
                record_id TEXT NOT NULL, parent_ref TEXT NOT NULL,
                PRIMARY KEY(namespace,generation,record_id,parent_ref));
            CREATE TABLE IF NOT EXISTS event_record_metadata (
                namespace TEXT NOT NULL, generation TEXT NOT NULL,
                record_id TEXT NOT NULL, metadata TEXT NOT NULL,
                PRIMARY KEY(namespace,generation,record_id));
        """)

    def commit_event(self, records: Iterable[StoredRecord], *, namespace: str, stream: str,
                     generation: str, position: int, event_id: str, visibility_time: int,
                     expected_spaces: tuple[str, ...],
                     lineage: dict[str, tuple[str, ...]] | None = None,
                     record_metadata: dict[str, dict] | None = None) -> bool:
        """Atomically advance a causal stream and publish all views of its writes.

        Returns false for an exact idempotent retry. The caller finishes event
        ``position`` before committing it; later searches use a strictly greater
        ``query_time`` to observe these records.
        """
        rows = list(records)
        if (not namespace or not stream or not generation or not event_id or position < 0
                or visibility_time < 0 or not expected_spaces
                or len(expected_spaces) != len(set(expected_spaces))):
            raise ValueError("Invalid temporal event commit")
        logical: dict[str, set[str]] = {}
        for record in rows:
            if (record.namespace != namespace or record.generation != generation
                    or record.created_at != visibility_time or record.space not in expected_spaces):
                raise ValueError("Event records differ from their visibility scope")
            logical.setdefault(record.record_id, set()).add(record.space)
        if any(spaces != set(expected_spaces) for spaces in logical.values()):
            raise ValueError("Every event record needs every expected space view")
        if len(rows) != len(logical) * len(expected_spaces):
            raise ValueError("Event contains duplicate record-space views")
        lineage = lineage or {record_id: () for record_id in logical}
        record_metadata = record_metadata or {record_id: {} for record_id in logical}
        if (set(lineage) != set(logical) or set(record_metadata) != set(logical)
                or any(not isinstance(parent, str) or not parent
                       for parents in lineage.values() for parent in parents)
                or any(len(parents) != len(set(parents)) for parents in lineage.values())
                or any(not isinstance(metadata, dict) for metadata in record_metadata.values())):
            raise ValueError("Event lineage and metadata must cover every logical record")
        content_hash = self._event_digest(rows)
        annotations = json.dumps({
            record_id: {"lineage": sorted(lineage[record_id]),
                        "metadata": record_metadata[record_id]}
            for record_id in sorted(logical)
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if any(lineage.values()) or any(record_metadata.values()):
            content_hash = hashlib.sha256((content_hash + annotations).encode()).hexdigest()
        record_ids = json.dumps(sorted(logical), separators=(",", ":"))
        with self.connect() as db:
            self._ensure_event_tables(db)
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("""SELECT position,event_id,visibility_time,record_ids,content_sha256
                FROM event_commits WHERE namespace=? AND stream=? AND generation=?
                AND (position=? OR event_id=?)""",
                (namespace, stream, generation, position, event_id)).fetchall()
            expected = (position, event_id, visibility_time, record_ids, content_hash)
            if prior:
                if len(prior) == 1 and tuple(prior[0]) == expected:
                    return False
                raise ValueError("Temporal event position or identity was already committed differently")
            state = db.execute("""SELECT next_position,visibility_time FROM event_streams
                WHERE namespace=? AND stream=? AND generation=?""",
                (namespace, stream, generation)).fetchone()
            if state is None:
                if position != 0:
                    raise ValueError("A temporal stream must begin at position zero")
            elif state[0] != position or visibility_time < state[1]:
                raise ValueError("Temporal events must commit once in monotonic order")
            for record in rows:
                self._put(db, record, ())
            for record_id in sorted(logical):
                for parent_ref in lineage[record_id]:
                    db.execute("INSERT INTO event_lineage VALUES (?,?,?,?)", (
                        namespace, generation, record_id, parent_ref))
                db.execute("INSERT INTO event_record_metadata VALUES (?,?,?,?)", (
                    namespace, generation, record_id,
                    json.dumps(record_metadata[record_id], sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False)))
            db.execute("INSERT INTO event_commits VALUES (?,?,?,?,?,?,?,?)", (
                namespace, stream, generation, position, event_id, visibility_time,
                record_ids, content_hash))
            db.execute("""INSERT INTO event_streams VALUES (?,?,?,?,?)
                ON CONFLICT(namespace,stream,generation) DO UPDATE SET
                next_position=excluded.next_position, visibility_time=excluded.visibility_time""",
                (namespace, stream, generation, position + 1, visibility_time))
        return True

    def event_frontier(self, *, namespace: str, stream: str, generation: str) -> dict:
        with self.connect() as db:
            exists = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='event_streams'""").fetchone()
            if exists is None:
                return {"next_position": 0, "visibility_time": -1}
            row = db.execute("""SELECT next_position,visibility_time FROM event_streams
                WHERE namespace=? AND stream=? AND generation=?""",
                (namespace, stream, generation)).fetchone()
        return {"next_position": 0, "visibility_time": -1} if row is None else {
            "next_position": row[0], "visibility_time": row[1]}

    def generation_summary(self, *, namespace: str, generation: str,
                           spaces: tuple[str, ...]) -> dict:
        """Return a deterministic integrity summary for one complete generation."""
        if not namespace or not generation or not spaces or len(spaces) != len(set(spaces)):
            raise ValueError("Generation summary needs an identity and distinct spaces")
        placeholders = ",".join("?" for _ in spaces)
        digest, rows, logical_ids = hashlib.sha256(), 0, set()
        with self.connect() as db:
            result = db.execute(
                f"""SELECT record_id,space,domain,created_at,key,key_dim,payload,source_id,deleted
                    FROM records WHERE namespace=? AND generation=?
                    AND space IN ({placeholders}) ORDER BY record_id,space""",
                (namespace, generation, *spaces),
            )
            for row in result:
                rows += 1
                logical_ids.add(row[0])
                for value in row:
                    encoded = value if isinstance(value, bytes) else str(value).encode()
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
        return {"logical_records": len(logical_ids), "views": rows,
                "sha256": digest.hexdigest()}

    def _put(self, db, record: StoredRecord, children: tuple[str, ...]) -> None:
        key = record.key.detach().float().cpu().contiguous()
        value = record.payload.detach().cpu().contiguous()
        if key.ndim != 1 or not value.is_floating_point() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("Finite vector key and floating payload required")
        if not record.record_id or not record.namespace or not record.generation:
            raise ValueError("Record identity and version cannot be empty")
        key_blob = key.numpy().astype("<f4", copy=False).tobytes()
        value_blob = save({"payload": value})
        if db.execute("SELECT 1 FROM tombstones WHERE namespace=? AND record_id=?",
                      (record.namespace, record.record_id)).fetchone():
            raise ValueError("Deleted IDs cannot be resurrected; use a new generation ID")
        for child in children:
            rows = db.execute(
                "SELECT domain,deleted FROM records WHERE namespace=? AND record_id=?",
                (record.namespace, child)).fetchall()
            if not rows or any(d != record.domain or deleted for d, deleted in rows):
                raise PermissionError("Compaction requires live authorization-homogeneous children")
            if child == record.record_id:
                raise ValueError("A record cannot compact itself")
        # Immutable versions: overwriting a payload would invalidate captured plans.
        db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,0)", (
            record.namespace, record.record_id, record.space, record.generation,
            record.domain, record.created_at, key_blob, key.numel(), value_blob, record.source_id))
        for child in children:
            db.execute("INSERT OR IGNORE INTO lineage VALUES (?,?,?)",
                       (record.namespace, record.record_id, child))

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = "default",
               space: str = "s0", generation: str = "v0", domain: str = "research",
               query_time: int = 2**62, key_chunk_size: int = 1024,
               exclude_ids: frozenset[str] = frozenset()) -> ReadPlan:
        if query.ndim != 1 or not torch.isfinite(query).all() or top_k < 0 or key_chunk_size < 1:
            raise ValueError("Invalid search parameters")
        if top_k == 0:
            return ReadPlan(namespace, space, generation, domain, query_time, ())
        q = query.detach().float().cpu().numpy()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        best: list[tuple[float, str]] = []
        with self.connect() as db:
            cursor = db.execute("""SELECT record_id,key,key_dim FROM records
                WHERE namespace=? AND space=? AND generation=? AND domain=?
                AND created_at<? AND deleted=0 ORDER BY record_id""",
                (namespace, space, generation, domain, query_time))
            while rows := cursor.fetchmany(key_chunk_size):
                if any(dim != len(q) for _, _, dim in rows):
                    raise ValueError("Key dimensions incompatible with query generation")
                keys = np.stack([np.frombuffer(b, dtype="<f4") for _, b, _ in rows])
                keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
                scores = keys @ q
                best.extend((float(score), row[0]) for row, score in zip(rows, scores, strict=True)
                            if row[0] not in exclude_ids)
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:top_k]
        return ReadPlan(namespace, space, generation, domain, query_time,
                        tuple(Selection(record_id, score) for score, record_id in best))

    def fetch(self, plan: ReadPlan) -> list[Tensor]:
        """Revalidate visibility at use time; deletion invalidates outstanding plans."""
        return self.fetch_many((plan,))[0]

    def lookup(self, record_id: str, *, namespace: str, space: str,
               generation: str, domain: str = "research",
               query_time: int = 2**62) -> StoredRecord:
        """Fetch one compatible stored record through the public backend surface."""
        with self.connect() as db:
            row = db.execute("""SELECT key,payload,created_at,source_id FROM records
                WHERE namespace=? AND record_id=? AND space=? AND generation=? AND domain=?
                AND created_at<? AND deleted=0""",
                (namespace, record_id, space, generation, domain, query_time)).fetchone()
        if row is None:
            raise KeyError(record_id)
        key = torch.from_numpy(np.frombuffer(row[0], dtype="<f4").copy())
        return StoredRecord(record_id, key, load(row[1])["payload"], namespace, space,
                            generation, domain, row[2], row[3])

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[Tensor]]:
        """Revalidate and load several captured plans through one connection."""
        result = []
        with self.connect() as db:
            has_offline_index = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='index' AND name='offline_record_scope'""").fetchone() is not None
            indexed = " INDEXED BY offline_record_scope" if has_offline_index else ""
            for plan in plans:
                ids = [selection.record_id for selection in plan.selections]
                if not ids:
                    result.append([])
                    continue
                placeholders = ",".join("?" for _ in set(ids))
                rows = db.execute(f"""SELECT record_id,payload FROM records{indexed}
                    WHERE namespace=? AND space=? AND generation=? AND domain=?
                    AND created_at<? AND deleted=0 AND record_id IN ({placeholders})""",
                    (plan.namespace, plan.space, plan.generation, plan.domain,
                     plan.query_time, *set(ids))).fetchall()
                available = {record_id: payload for record_id, payload in rows}
                missing = next((record_id for record_id in ids if record_id not in available), None)
                if missing is not None:
                    raise KeyError(f"Unavailable, stale or unauthorized record: {missing}")
                result.append([load(available[record_id])["payload"] for record_id in ids])
        return result

    def iter_fetch(self, plan: ReadPlan, chunk_size: int = 32):
        """Reload and revalidate at most one selected chunk at a time on the host."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for start in range(0, len(plan.selections), chunk_size):
            part = ReadPlan(plan.namespace, plan.space, plan.generation, plan.domain,
                            plan.query_time, plan.selections[start:start + chunk_size],
                            plan.bank_cursor)
            values = self.fetch(part)
            yield torch.stack(values)[None], torch.ones(1, len(values))

    def delete(self, namespace: str, record_id: str) -> set[str]:
        """Tombstone a source and invalidate all transitive compacted derivatives."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("""WITH RECURSIVE affected(id) AS (
                SELECT ? UNION SELECT lineage.parent_id FROM lineage JOIN affected
                ON lineage.child_id=affected.id WHERE lineage.namespace=?)
                SELECT id FROM affected""", (record_id, namespace)).fetchall()
            affected = {row[0] for row in rows}
            for rid in affected:
                db.execute("UPDATE records SET deleted=1 WHERE namespace=? AND record_id=?", (namespace, rid))
                db.execute("INSERT OR IGNORE INTO tombstones VALUES (?,?)", (namespace, rid))
        return affected

    def mutable_bank_state(self) -> dict | None:
        """Return the small recovery token for a mutable-bank journal, if present."""
        with self.connect() as db:
            exists = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='mutable_bank_meta'""").fetchone()
            if exists is None:
                return None
            meta = dict(db.execute('SELECT key,value FROM mutable_bank_meta'))
            cursor = int(meta['cursor'])
            commit = db.execute('''SELECT content_sha256,optimizer_step
                FROM mutable_bank_commits WHERE cursor=?''', (cursor,)).fetchone()
        return {
            'format': int(meta['format']), 'store_uuid': meta['store_uuid'],
            'cursor': cursor, 'floor_cursor': int(meta['floor_cursor']),
            'maintenance_position': int(meta['maintenance_position']),
            'commit_sha256': None if commit is None else commit[0],
            'optimizer_step': None if commit is None else commit[1],
        }

    def restore_mutable_bank(self, state: dict) -> None:
        """Roll a journal back to a checkpoint token without copying bank payloads."""
        required = {'format', 'store_uuid', 'cursor', 'floor_cursor',
                    'maintenance_position', 'commit_sha256', 'optimizer_step'}
        if set(state) != required or state['format'] != 1:
            raise ValueError('Invalid mutable-bank checkpoint state')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            meta = dict(db.execute('SELECT key,value FROM mutable_bank_meta'))
            target, current = int(state['cursor']), int(meta['cursor'])
            if meta.get('store_uuid') != state['store_uuid']:
                raise ValueError('Checkpoint refers to a different mutable bank')
            if (target != 0 and target < int(meta['floor_cursor'])) or target > current:
                raise ValueError('Checkpoint bank cursor is outside retained history')
            commit = db.execute('''SELECT content_sha256,optimizer_step
                FROM mutable_bank_commits WHERE cursor=?''', (target,)).fetchone()
            expected = None if target == 0 else (
                state['commit_sha256'], state['optimizer_step'])
            if commit != expected:
                raise ValueError('Checkpoint bank cursor content differs from its journal')
            self._rollback_mutable_events(db, target)
            db.execute('DELETE FROM mutable_bank_revisions WHERE cursor>?', (target,))
            db.execute('DELETE FROM mutable_bank_commits WHERE cursor>?', (target,))
            db.execute('DELETE FROM mutable_bank_status WHERE cursor>?', (target,))
            db.execute('DELETE FROM mutable_bank_dependencies '
                       'WHERE introduced_cursor>?', (target,))
            db.execute('UPDATE mutable_bank_dependencies SET retired_cursor=NULL '
                       'WHERE retired_cursor>?', (target,))
            db.execute('DELETE FROM mutable_bank_heads')
            if db.execute("""SELECT 1 FROM sqlite_master WHERE type='table'
                    AND name='mutable_bank_rebuild_leases'""").fetchone() is not None:
                db.execute('DELETE FROM mutable_bank_rebuild_leases')
            db.execute('''INSERT INTO mutable_bank_heads
                SELECT namespace,record_id,space,MAX(cursor)
                FROM mutable_bank_revisions WHERE cursor<=?
                GROUP BY namespace,record_id,space''', (target,))
            db.execute("UPDATE mutable_bank_meta SET value=? WHERE key='cursor'",
                       (str(target),))
            if target == 0:
                db.execute("UPDATE mutable_bank_meta SET value='0' "
                           "WHERE key='floor_cursor'")
            db.execute("UPDATE mutable_bank_meta SET value=? "
                       "WHERE key='maintenance_position'",
                       (str(state['maintenance_position']),))

    @staticmethod
    def _rollback_mutable_events(db, cursor: int) -> None:
        """Restore event frontiers whose atomic bank commit is being rolled back."""
        exists = db.execute("""SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='mutable_bank_event_cursors'""").fetchone()
        if exists is None:
            return
        affected = db.execute('''SELECT namespace,stream,generation,position
            FROM mutable_bank_event_cursors WHERE cursor>?''', (cursor,)).fetchall()
        scopes = {(namespace, stream, generation)
                  for namespace, stream, generation, _position in affected}
        for namespace, stream, generation, position in affected:
            row = db.execute('''SELECT record_ids FROM event_commits
                WHERE namespace=? AND stream=? AND generation=? AND position=?''',
                (namespace, stream, generation, position)).fetchone()
            if row is not None:
                for record_id in json.loads(row[0]):
                    db.execute('''DELETE FROM event_lineage WHERE namespace=?
                        AND generation=? AND record_id=?''',
                        (namespace, generation, record_id))
                    db.execute('''DELETE FROM event_record_metadata WHERE namespace=?
                        AND generation=? AND record_id=?''',
                        (namespace, generation, record_id))
            db.execute('''DELETE FROM event_commits WHERE namespace=? AND stream=?
                AND generation=? AND position=?''',
                (namespace, stream, generation, position))
        db.execute('DELETE FROM mutable_bank_event_cursors WHERE cursor>?', (cursor,))
        for namespace, stream, generation in scopes:
            latest = db.execute('''SELECT position,visibility_time FROM event_commits
                WHERE namespace=? AND stream=? AND generation=? ORDER BY position DESC
                LIMIT 1''', (namespace, stream, generation)).fetchone()
            if latest is None:
                db.execute('''DELETE FROM event_streams WHERE namespace=? AND stream=?
                    AND generation=?''', (namespace, stream, generation))
            else:
                db.execute('''UPDATE event_streams SET next_position=?,visibility_time=?
                    WHERE namespace=? AND stream=? AND generation=?''',
                    (latest[0] + 1, latest[1], namespace, stream, generation))

    def pin_mutable_bank(self, checkpoint: str, cursor: int) -> None:
        with self.connect() as db:
            exists = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='mutable_bank_checkpoint_pins'""").fetchone()
            if exists is not None:
                db.execute('INSERT OR REPLACE INTO mutable_bank_checkpoint_pins VALUES (?,?)',
                           (checkpoint, cursor))

    def unpin_mutable_bank(self, checkpoint: str) -> None:
        with self.connect() as db:
            exists = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='mutable_bank_checkpoint_pins'""").fetchone()
            if exists is not None:
                db.execute('DELETE FROM mutable_bank_checkpoint_pins WHERE checkpoint=?',
                           (checkpoint,))

    def gc_mutable_bank(self, before_cursor: int) -> dict[str, int]:
        """Release superseded revisions while retaining every pinned recovery point."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            meta = dict(db.execute('SELECT key,value FROM mutable_bank_meta'))
            current, floor = int(meta['cursor']), int(meta['floor_cursor'])
            pinned = [row[0] for row in db.execute(
                'SELECT cursor FROM mutable_bank_checkpoint_pins WHERE cursor>0').fetchall()]
            ceiling = min(pinned) if pinned else current
            if before_cursor < floor or before_cursor > ceiling:
                raise ValueError('GC cursor would remove checkpoint-pinned history')
            # Keep the newest baseline revision/status no later than the new floor,
            # plus every later event needed for rollback to a retained checkpoint.
            # A row is superseded when a newer one for the same record exists at or
            # before the new floor. EXISTS probes the lookup indexes; the former
            # row-value NOT IN over a grouped subquery took ~12 min per checkpoint.
            db.execute('''CREATE INDEX IF NOT EXISTS mutable_status_lookup
                ON mutable_bank_status(namespace,record_id,cursor)''')
            deleted_revisions = db.execute('''DELETE FROM mutable_bank_revisions AS old
                WHERE old.cursor<? AND EXISTS (
                    SELECT 1 FROM mutable_bank_revisions AS newer
                    WHERE newer.namespace=old.namespace AND newer.record_id=old.record_id
                      AND newer.space=old.space AND newer.cursor>old.cursor
                      AND newer.cursor<=?)''',
                (before_cursor, before_cursor)).rowcount
            deleted_status = db.execute('''DELETE FROM mutable_bank_status AS old
                WHERE old.cursor<? AND EXISTS (
                    SELECT 1 FROM mutable_bank_status AS newer
                    WHERE newer.namespace=old.namespace AND newer.record_id=old.record_id
                      AND newer.cursor>old.cursor AND newer.cursor<=?)''',
                (before_cursor, before_cursor)).rowcount
            deleted_commits = db.execute(
                'DELETE FROM mutable_bank_commits WHERE cursor<?',
                (before_cursor,)).rowcount
            db.execute("UPDATE mutable_bank_meta SET value=? WHERE key='floor_cursor'",
                       (str(before_cursor),))
        return {'revisions': deleted_revisions, 'status': deleted_status,
                'commits': deleted_commits}

    def mutable_bank_gc_ceiling(self) -> int | None:
        """Return the oldest non-base checkpoint cursor safe for journal GC."""
        state = self.mutable_bank_state()
        if state is None:
            return None
        with self.connect() as db:
            pins = [row[0] for row in db.execute('''SELECT cursor
                FROM mutable_bank_checkpoint_pins WHERE cursor>0''').fetchall()]
        return min(pins) if pins else int(state['cursor'])

    def sizes(self) -> dict[str, int]:
        with self.connect() as db:
            count, payload, keys = db.execute("""SELECT count(*),coalesce(sum(length(payload)),0),
                coalesce(sum(length(key)),0) FROM records WHERE deleted=0""").fetchone()
        physical = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())
        return {"records": count, "serialized_payload_bytes": payload,
                "key_bytes": keys, "physical_sqlite_bytes": physical,
                "sqlite_cache_budget_bytes_per_connection": self.cache_mib * 1024**2}


class AsyncRetriever:
    """CPU/disk future API only. No automatic backbone scheduling or latency claim."""
    def __init__(self, store: DiskStore, workers: int = 2) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, query: Tensor, **kwargs) -> Future:
        query = query.detach().float().cpu().clone()
        def read():
            plan = self.store.search(query, **kwargs)
            return plan, self.store.fetch(plan)
        return self.pool.submit(read)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> AsyncRetriever:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def lookup_record(store, record_id: str, *, namespace: str, space: str,
                  generation: str, domain: str = "research", query_time: int = 2**62) -> StoredRecord:
    """Compatibility helper over the public local/network lookup operation."""
    return store.lookup(record_id, namespace=namespace, space=space,
                        generation=generation, domain=domain, query_time=query_time)
