"""Durable revision journal for a continuously learned logical memory bank.

The offline store is a physical bootstrap snapshot. Learned key/payload changes
are appended as revisions and made current by an atomic head update. Reads consume
stored tensors only; source trajectories are used solely by explicit writer jobs.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import threading
import time
import uuid

import numpy as np
import torch
from safetensors.torch import load, save

from .key_index import PublishedKeyIndex
from .store import DiskStore, ReadPlan, StoredRecord


class TrainingBank:
    """One mutable logical catalog backed by append-only physical revisions."""

    FORMAT = 1

    def __init__(self, base: DiskStore, journal: DiskStore,
                 index: PublishedKeyIndex) -> None:
        self.base, self.cache, self.index = base, journal, index
        self._spaces = tuple(index.spaces)
        # Optional write-through payload cache (``enable_payload_cache``).
        self._payloads: OrderedDict[tuple[str, str], tuple[int, torch.Tensor]] = OrderedDict()
        self._payload_bytes = 0
        self._payload_limit = 0
        self._payload_lock = threading.Lock()
        self.payload_cache_hits = self.payload_cache_misses = 0
        self._ensure_schema()
        self._migrate_overlay()
        self._reload_index()

    def enable_payload_cache(self, max_bytes: int) -> None:
        """Keep recently read or published payloads in host memory (LRU, byte cap).

        Every publish writes through, so a cached entry is always its record's
        newest revision; it serves any read whose snapshot cursor is at or after
        the entry's revision. Rollback clears the cache. Reads still check the
        record's authorization scope against the index.
        """
        with self._payload_lock:
            self._payload_limit = max(0, int(max_bytes))
            self._evict_payloads()

    def _evict_payloads(self) -> None:
        while self._payload_bytes > self._payload_limit and self._payloads:
            _, (_, value) = self._payloads.popitem(last=False)
            self._payload_bytes -= value.numel() * value.element_size()

    def _remember_payload(self, space: str, record_id: str, cursor: int,
                          value: torch.Tensor) -> None:
        if not self._payload_limit:
            return
        value = value.detach().cpu()
        with self._payload_lock:
            old = self._payloads.pop((space, record_id), None)
            if old is not None:
                if old[0] > cursor:
                    self._payloads[(space, record_id)] = old
                    return
                self._payload_bytes -= old[1].numel() * old[1].element_size()
            self._payloads[(space, record_id)] = (cursor, value)
            self._payload_bytes += value.numel() * value.element_size()
            self._evict_payloads()

    def _cached_payload(self, space: str, record_id: str, cursor: int) -> torch.Tensor | None:
        with self._payload_lock:
            entry = self._payloads.get((space, record_id))
            if entry is None or entry[0] > cursor:
                return None
            self._payloads.move_to_end((space, record_id))
            return entry[1]

    def _authorized(self, plan: ReadPlan, record_id: str) -> bool:
        array = self.index.spaces[plan.space]
        position = int(np.searchsorted(array.ids, record_id))
        return (position < len(array.ids) and array.ids[position] == record_id
                and array.domains[position] == plan.domain
                and array.times[position] < plan.query_time
                and not array.deleted[position])

    @staticmethod
    def _tables(db) -> set[str]:
        return {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    def _ensure_schema(self) -> None:
        with self.cache.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS mutable_bank_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS mutable_bank_commits (
                    cursor INTEGER PRIMARY KEY, optimizer_step INTEGER,
                    content_sha256 TEXT NOT NULL, logical_records INTEGER NOT NULL,
                    views INTEGER NOT NULL, committed_at_ns INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS mutable_bank_revisions (
                    cursor INTEGER NOT NULL, namespace TEXT NOT NULL,
                    record_id TEXT NOT NULL, space TEXT NOT NULL,
                    generation TEXT NOT NULL, domain TEXT NOT NULL,
                    created_at INTEGER NOT NULL, source_id TEXT NOT NULL,
                    key BLOB NOT NULL, key_dim INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY(cursor,namespace,record_id,space));
                CREATE INDEX IF NOT EXISTS mutable_revision_lookup ON
                    mutable_bank_revisions(namespace,record_id,space,cursor);
                CREATE TABLE IF NOT EXISTS mutable_bank_heads (
                    namespace TEXT NOT NULL, record_id TEXT NOT NULL,
                    space TEXT NOT NULL, cursor INTEGER NOT NULL,
                    PRIMARY KEY(namespace,record_id,space));
                CREATE TABLE IF NOT EXISTS mutable_bank_dependencies (
                    namespace TEXT NOT NULL, parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL, introduced_cursor INTEGER NOT NULL,
                    child_cursor INTEGER NOT NULL, retired_cursor INTEGER,
                    PRIMARY KEY(namespace,parent_id,child_id,introduced_cursor));
                CREATE INDEX IF NOT EXISTS mutable_child_lookup ON
                    mutable_bank_dependencies(namespace,child_id,parent_id);
                CREATE TABLE IF NOT EXISTS mutable_bank_status (
                    cursor INTEGER NOT NULL, namespace TEXT NOT NULL,
                    record_id TEXT NOT NULL, invalidated INTEGER NOT NULL,
                    cause_id TEXT NOT NULL,
                    PRIMARY KEY(cursor,namespace,record_id));
                CREATE TABLE IF NOT EXISTS mutable_bank_checkpoint_pins (
                    checkpoint TEXT PRIMARY KEY, cursor INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS mutable_bank_event_cursors (
                    namespace TEXT NOT NULL, stream TEXT NOT NULL,
                    generation TEXT NOT NULL, position INTEGER NOT NULL,
                    cursor INTEGER NOT NULL,
                    PRIMARY KEY(namespace,stream,generation,position));
                CREATE TABLE IF NOT EXISTS mutable_bank_rebuild_leases (
                    namespace TEXT NOT NULL, record_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL, lease_until_ns INTEGER NOT NULL,
                    invalidated_cursor INTEGER NOT NULL,
                    PRIMARY KEY(namespace,record_id));
            """)
            defaults = {
                'format': str(self.FORMAT), 'cursor': '0', 'floor_cursor': '0',
                'maintenance_position': '0', 'store_uuid': uuid.uuid4().hex,
                'catalog_scope': json.dumps({
                    'namespace': self.index.namespace,
                    'generation': self.index.generation,
                    'spaces': self._spaces,
                }, sort_keys=True, separators=(',', ':')),
            }
            for key, value in defaults.items():
                db.execute('INSERT OR IGNORE INTO mutable_bank_meta VALUES (?,?)',
                           (key, value))
            found = dict(db.execute('SELECT key,value FROM mutable_bank_meta'))
            if int(found['format']) != self.FORMAT:
                raise ValueError('Unsupported mutable-bank journal format')
            if found['catalog_scope'] != defaults['catalog_scope']:
                raise ValueError('Mutable-bank journal belongs to another base catalog')
            if 'mutable_bank_migration_source' in self._tables(db):
                source = db.execute(
                    'SELECT sha256 FROM mutable_bank_migration_source').fetchone()
                if source is None or len(source[0]) != 64:
                    raise ValueError('Legacy migration source identity is invalid')
                db.execute('INSERT OR REPLACE INTO mutable_bank_meta VALUES (?,?)',
                           ('legacy_source_sha256', source[0]))
            if found.get('base_lineage_imported') != '1':
                with self.base.connect() as source:
                    rows = source.execute(
                        'SELECT namespace,parent_id,child_id FROM lineage').fetchall()
                db.executemany('''INSERT OR IGNORE INTO mutable_bank_dependencies
                    VALUES (?,?,?,0,0,NULL)''', rows)
                db.execute("INSERT OR REPLACE INTO mutable_bank_meta VALUES "
                           "('base_lineage_imported','1')")

    @staticmethod
    def _meta(db, key: str) -> str:
        row = db.execute('SELECT value FROM mutable_bank_meta WHERE key=?', (key,)).fetchone()
        if row is None:
            raise ValueError(f'Mutable-bank metadata is missing {key}')
        return str(row[0])

    @property
    def cursor(self) -> int:
        with self.cache.connect() as db:
            return int(self._meta(db, 'cursor'))

    def _migrate_overlay(self) -> None:
        """Convert the former overwrite overlay without losing resume state."""
        with self.cache.connect() as db:
            if 'training_bank_overlay' not in self._tables(db):
                return
        while True:
            with self.cache.connect() as db:
                ids = [row[0] for row in db.execute('''SELECT DISTINCT o.record_id
                    FROM training_bank_overlay o WHERE NOT EXISTS (
                        SELECT 1 FROM mutable_bank_heads h WHERE h.namespace=?
                        AND h.record_id=o.record_id) ORDER BY o.record_id LIMIT 4096''',
                    (self.index.namespace,)).fetchall()]
                if not ids:
                    break
                placeholders = ','.join('?' for _ in ids)
                rows = db.execute(f'''SELECT space,record_id,key,key_dim
                    FROM training_bank_overlay WHERE record_id IN ({placeholders})
                    ORDER BY record_id,space''', ids).fetchall()
            if (len(rows) != len(ids) * len(self._spaces)
                    or any(space not in self._spaces
                           or len(key) != key_dim * 4 for space, _record_id, key,
                           key_dim in rows)):
                raise ValueError('Legacy overlay lacks complete compatible space views')
            self._migrate_batch(ids)
        with self.cache.connect() as db:
            db.execute('DROP TABLE training_bank_overlay')
            source = db.execute("""SELECT value FROM mutable_bank_meta
                WHERE key='legacy_source_sha256'""").fetchone()
            if source is not None:
                cursor = int(self._meta(db, 'cursor'))
                db.execute('INSERT OR REPLACE INTO mutable_bank_meta VALUES (?,?)',
                           ('legacy_checkpoint_cursor', str(cursor)))
                db.execute('''INSERT OR REPLACE INTO mutable_bank_checkpoint_pins
                    VALUES (?,?)''', ('legacy:' + source[0], cursor))

    def _migrate_batch(self, record_ids: list[str]) -> None:
        """Copy already serialized legacy tensors without decoding their payloads."""
        placeholders = ','.join('?' for _ in record_ids)
        with self.cache.connect() as db:
            db.execute('ATTACH DATABASE ? AS bootstrap', (str(self.base.path),))
            db.execute('BEGIN IMMEDIATE')
            cursor = int(self._meta(db, 'cursor')) + 1
            expected = len(record_ids) * len(self._spaces)
            matched = db.execute(f'''SELECT COUNT(*) FROM training_bank_overlay o
                JOIN bootstrap.records b ON b.namespace=? AND b.generation=?
                 AND b.record_id=o.record_id AND b.space=o.space AND b.deleted=0
                WHERE o.record_id IN ({placeholders})''',
                (self.index.namespace, self.index.generation, *record_ids)).fetchone()[0]
            if matched != expected:
                raise ValueError('Legacy overlay differs from its bootstrap catalog')
            digest = hashlib.sha256(json.dumps(
                {'legacy_migration': cursor, 'record_ids': record_ids,
                 'spaces': self._spaces}, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            db.execute('INSERT INTO mutable_bank_commits VALUES (?,?,?,?,?,?)',
                       (cursor, None, digest, len(record_ids), expected, time.time_ns()))
            db.execute(f'''INSERT INTO mutable_bank_revisions
                SELECT ?,b.namespace,o.record_id,o.space,b.generation,b.domain,
                       b.created_at,b.source_id,o.key,o.key_dim,o.payload
                FROM training_bank_overlay o JOIN bootstrap.records b
                  ON b.namespace=? AND b.generation=? AND b.record_id=o.record_id
                 AND b.space=o.space AND b.deleted=0
                WHERE o.record_id IN ({placeholders})''',
                (cursor, self.index.namespace, self.index.generation, *record_ids))
            db.execute(f'''INSERT INTO mutable_bank_heads
                SELECT ?,record_id,space,? FROM training_bank_overlay
                WHERE record_id IN ({placeholders})
                ON CONFLICT(namespace,record_id,space) DO UPDATE SET
                cursor=excluded.cursor''',
                (self.index.namespace, cursor, *record_ids))
            db.executemany('INSERT INTO mutable_bank_status VALUES (?,?,?,?,?)',
                           ((cursor, self.index.namespace, record_id, 0, '')
                            for record_id in record_ids))
            descendants = db.execute(f'''WITH RECURSIVE affected(id,cause) AS (
                SELECT parent_id,child_id FROM mutable_bank_dependencies
                 WHERE namespace=? AND child_id IN ({placeholders})
                   AND introduced_cursor<=? AND
                       (retired_cursor IS NULL OR retired_cursor>?)
                UNION SELECT d.parent_id,a.cause FROM mutable_bank_dependencies d
                 JOIN affected a ON d.child_id=a.id WHERE d.namespace=?
                   AND d.introduced_cursor<=? AND
                       (d.retired_cursor IS NULL OR d.retired_cursor>?))
                SELECT DISTINCT id,cause FROM affected''',
                (self.index.namespace, *record_ids, cursor - 1, cursor - 1,
                 self.index.namespace, cursor - 1, cursor - 1)).fetchall()
            db.executemany('INSERT OR REPLACE INTO mutable_bank_status VALUES (?,?,?,?,?)',
                           ((cursor, self.index.namespace, parent, 1, cause)
                            for parent, cause in descendants
                            if parent not in record_ids))
            db.execute("UPDATE mutable_bank_meta SET value=? WHERE key='cursor'",
                       (str(cursor),))

    def _invalidated(self, db, cursor: int) -> set[str]:
        rows = db.execute('''SELECT s.record_id,s.invalidated FROM mutable_bank_status s
            JOIN (SELECT namespace,record_id,MAX(cursor) cursor FROM mutable_bank_status
                  WHERE namespace=? AND cursor<=? GROUP BY namespace,record_id) latest
              ON latest.namespace=s.namespace AND latest.record_id=s.record_id
             AND latest.cursor=s.cursor WHERE s.namespace=?''',
            (self.index.namespace, cursor, self.index.namespace)).fetchall()
        return {record_id for record_id, invalidated in rows if invalidated}

    def _reload_index(self) -> None:
        with self.cache.connect() as db:
            cursor = int(self._meta(db, 'cursor'))
            rows = db.execute('''SELECT r.record_id,r.space,r.domain,r.created_at,
                r.source_id,r.key,r.key_dim FROM mutable_bank_heads h
                JOIN mutable_bank_revisions r ON r.cursor=h.cursor
                 AND r.namespace=h.namespace AND r.record_id=h.record_id
                 AND r.space=h.space WHERE h.cursor<=?''', (cursor,)).fetchall()
            invalidated = self._invalidated(db, cursor)
        self.index.bank_cursor = cursor
        for space in self._spaces:
            self.index.upsert_many(space, [
                (record_id, key, key_dim, domain, created_at, source_id,
                 record_id in invalidated)
                for record_id, row_space, domain, created_at, source_id, key, key_dim
                in rows if row_space == space
            ])
        self.index.set_deleted(invalidated, True)

    @staticmethod
    def _encode(record: StoredRecord) -> tuple[bytes, int, bytes]:
        key = record.key.detach().float().cpu().contiguous()
        payload = record.payload.detach().cpu().contiguous()
        if (key.ndim != 1 or not torch.isfinite(key).all()
                or not payload.is_floating_point() or not torch.isfinite(payload).all()):
            raise ValueError('Mutable bank updates need finite vector keys and payloads')
        key = torch.nn.functional.normalize(key, dim=-1)
        return (key.numpy().astype('<f4', copy=False).tobytes(), key.numel(),
                save({'payload': payload}))

    def _scope(self, record: StoredRecord) -> tuple[str, int, str]:
        array = self.index.spaces[record.space]
        positions = np.flatnonzero(array.ids == record.record_id)
        if len(positions) == 1:
            position = int(positions[0])
            stored_source = str(array.source_ids[position])
            if record.source_id and stored_source and record.source_id != stored_source:
                raise ValueError('A mutable revision cannot change source identity')
            return (str(array.domains[position]), int(array.times[position]),
                    stored_source or record.source_id)
        if (record.namespace != self.index.namespace
                or record.generation != self.index.generation
                or not record.domain or record.created_at < 0
                or not record.source_id):
            raise ValueError('New records need the active catalog scope')
        return record.domain, record.created_at, record.source_id

    def update(self, records: Iterable[StoredRecord], *, optimizer_step: int | None = None,
               children: Mapping[str, Sequence[str]] | None = None,
               expected_child_cursors: Mapping[str, Mapping[str, int]] | None = None,
               rebuild_claims: Mapping[str, tuple[str, int]] | None = None,
               event: dict | None = None) -> int:
        """Atomically publish complete all-space revisions for logical records."""
        rows = list(records)
        if not rows:
            return 0
        grouped: dict[str, dict[str, StoredRecord]] = {}
        for record in rows:
            if record.space not in self.index.spaces:
                raise KeyError(f'Unknown memory space: {record.space}')
            if record.space in grouped.setdefault(record.record_id, {}):
                raise ValueError('Mutable update contains a duplicate record-space view')
            grouped[record.record_id][record.space] = record
        expected = set(self._spaces)
        if any(set(views) != expected for views in grouped.values()):
            raise ValueError('Every mutable record revision needs every configured space view')
        children = children or {}
        expected_child_cursors = expected_child_cursors or {}
        rebuild_claims = rebuild_claims or {}
        if set(children) - set(grouped):
            raise ValueError('Compaction dependencies refer to an unpublished parent')
        if (set(expected_child_cursors) - set(children)
                or any(set(expected_child_cursors[parent]) != set(children[parent])
                       for parent in expected_child_cursors)):
            raise ValueError('Expected child cursors must cover a declared dependency set')
        if (set(rebuild_claims) - set(expected_child_cursors)
                or any(not worker_id or invalidated_at < 0
                       for worker_id, invalidated_at in rebuild_claims.values())):
            raise ValueError('Rebuild claims need expected children, worker and cursor')
        if event is not None:
            required = {'stream', 'position', 'event_id', 'visibility_time',
                        'lineage', 'record_metadata'}
            if (set(event) != required or event['position'] < 0
                    or event['visibility_time'] < 0 or not event['stream']
                    or not event['event_id']
                    or set(event['lineage']) != set(grouped)
                    or set(event['record_metadata']) != set(grouped)):
                raise ValueError('Invalid mutable-bank event commit')
            if any(view.created_at != event['visibility_time']
                   for views in grouped.values() for view in views.values()):
                raise ValueError('Event record time differs from its visibility frontier')

        encoded = []
        digest = hashlib.sha256()
        for record_id in sorted(grouped):
            for space in self._spaces:
                record = grouped[record_id][space]
                domain, created_at, source_id = self._scope(record)
                key, key_dim, payload = self._encode(record)
                if key_dim != self.index.spaces[space].keys.shape[1]:
                    raise ValueError('Mutable key dimension differs from the active index')
                values = (self.index.namespace, record_id, space,
                          self.index.generation, domain, created_at, source_id,
                          key, key_dim, payload)
                encoded.append(values)
                digest.update(json.dumps(values[:7], separators=(',', ':')).encode())
                digest.update(key)
                digest.update(payload)
        digest.update(json.dumps({
            'children': {record_id: sorted(child_ids)
                         for record_id, child_ids in sorted(children.items())},
            'event': event,
        }, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())

        with self.cache.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if event is not None:
                DiskStore._ensure_event_tables(db)
                record_ids = json.dumps(sorted(grouped), separators=(',', ':'))
                prior = db.execute('''SELECT position,event_id,visibility_time,
                    record_ids,content_sha256 FROM event_commits
                    WHERE namespace=? AND stream=? AND generation=?
                    AND (position=? OR event_id=?)''',
                    (self.index.namespace, event['stream'], self.index.generation,
                     event['position'], event['event_id'])).fetchall()
                expected_event = (event['position'], event['event_id'],
                                  event['visibility_time'], record_ids,
                                  digest.hexdigest())
                if prior:
                    if len(prior) == 1 and tuple(prior[0]) == expected_event:
                        return 0
                    raise ValueError('Event position or identity was committed differently')
                state = db.execute('''SELECT next_position,visibility_time
                    FROM event_streams WHERE namespace=? AND stream=? AND generation=?''',
                    (self.index.namespace, event['stream'],
                     self.index.generation)).fetchone()
                if state is None and event['position'] != 0:
                    raise ValueError('A mutable event stream must begin at position zero')
                if (state is not None and (state[0] != event['position']
                                           or event['visibility_time'] < state[1])):
                    raise ValueError('Mutable events must commit once in monotonic order')
            cursor = int(self._meta(db, 'cursor')) + 1
            for parent, (worker_id, invalidated_at) in rebuild_claims.items():
                lease = db.execute('''SELECT worker_id,lease_until_ns,invalidated_cursor
                    FROM mutable_bank_rebuild_leases
                    WHERE namespace=? AND record_id=?''',
                    (self.index.namespace, parent)).fetchone()
                if (lease is None or lease[0] != worker_id or lease[1] <= time.time_ns()
                        or lease[2] != invalidated_at):
                    raise RuntimeError(
                        f'Rebuild lease is stale or belongs to another worker: {parent}')
            updated_ids = set(grouped)
            missing_rebuild = self._invalidated(db, cursor - 1) & updated_ids - set(children)
            if missing_rebuild:
                raise ValueError(
                    'Invalidated derived records need current child revisions: '
                    f'{sorted(missing_rebuild)[:3]}')
            placeholders = ','.join('?' for _ in updated_ids)
            descendants = db.execute(f'''WITH RECURSIVE affected(id,cause) AS (
                SELECT parent_id,child_id FROM mutable_bank_dependencies
                 WHERE namespace=? AND child_id IN ({placeholders})
                   AND introduced_cursor<=? AND (retired_cursor IS NULL OR retired_cursor>?)
                UNION SELECT d.parent_id,a.cause FROM mutable_bank_dependencies d
                 JOIN affected a ON d.child_id=a.id WHERE d.namespace=?
                   AND d.introduced_cursor<=? AND (d.retired_cursor IS NULL OR d.retired_cursor>?))
                SELECT DISTINCT id,cause FROM affected''',
                (self.index.namespace, *sorted(updated_ids), cursor - 1, cursor - 1,
                 self.index.namespace, cursor - 1, cursor - 1)).fetchall()
            invalidated = {parent: cause for parent, cause in descendants
                           if parent not in updated_ids}
            db.execute('INSERT INTO mutable_bank_commits VALUES (?,?,?,?,?,?)',
                       (cursor, optimizer_step, digest.hexdigest(), len(grouped), len(encoded),
                        time.time_ns()))
            for values in encoded:
                db.execute('''INSERT INTO mutable_bank_revisions VALUES
                    (?,?,?,?,?,?,?,?,?,?,?)''', (cursor, *values))
                db.execute('''INSERT INTO mutable_bank_heads VALUES (?,?,?,?)
                    ON CONFLICT(namespace,record_id,space) DO UPDATE SET
                    cursor=excluded.cursor''', (values[0], values[1], values[2], cursor))
            for record_id in sorted(updated_ids):
                db.execute('INSERT INTO mutable_bank_status VALUES (?,?,?,?,?)',
                           (cursor, self.index.namespace, record_id, 0, ''))
                db.execute('''DELETE FROM mutable_bank_rebuild_leases
                    WHERE namespace=? AND record_id=?''',
                           (self.index.namespace, record_id))
            for parent, cause in sorted(invalidated.items()):
                db.execute('INSERT INTO mutable_bank_status VALUES (?,?,?,?,?)',
                           (cursor, self.index.namespace, parent, 1, cause))
            for parent, child_ids in children.items():
                if len(child_ids) != len(set(child_ids)) or parent in child_ids:
                    raise ValueError('Invalid compact-record dependencies')
                db.execute('''UPDATE mutable_bank_dependencies SET retired_cursor=?
                    WHERE namespace=? AND parent_id=? AND retired_cursor IS NULL''',
                    (cursor, self.index.namespace, parent))
                for child in child_ids:
                    if child in self._invalidated(db, cursor - 1):
                        raise ValueError(f'Cannot compact invalidated child: {child}')
                    child_cursor = db.execute('''SELECT COALESCE(MAX(cursor),0)
                        FROM mutable_bank_heads WHERE namespace=? AND record_id=?''',
                        (self.index.namespace, child)).fetchone()[0]
                    known = any(child in set(map(str, array.ids))
                                for array in self.index.spaces.values())
                    if child_cursor == 0 and not known:
                        raise KeyError(f'Unknown compact child: {child}')
                    expected_cursor = expected_child_cursors.get(parent, {}).get(child)
                    if expected_cursor is not None and child_cursor != expected_cursor:
                        raise RuntimeError(
                            f'Child {child} changed while its compaction was built')
                    db.execute('INSERT INTO mutable_bank_dependencies VALUES (?,?,?,?,?,NULL)',
                               (self.index.namespace, parent, child, cursor, child_cursor))
            db.execute("UPDATE mutable_bank_meta SET value=? WHERE key='cursor'", (str(cursor),))
            if event is not None:
                for record_id in sorted(grouped):
                    for parent_ref in event['lineage'][record_id]:
                        db.execute('INSERT INTO event_lineage VALUES (?,?,?,?)', (
                            self.index.namespace, self.index.generation,
                            record_id, parent_ref))
                    db.execute('INSERT INTO event_record_metadata VALUES (?,?,?,?)', (
                        self.index.namespace, self.index.generation, record_id,
                        json.dumps(event['record_metadata'][record_id], sort_keys=True,
                                   separators=(',', ':'), ensure_ascii=False)))
                db.execute('INSERT INTO event_commits VALUES (?,?,?,?,?,?,?,?)', (
                    self.index.namespace, event['stream'], self.index.generation,
                    event['position'], event['event_id'], event['visibility_time'],
                    json.dumps(sorted(grouped), separators=(',', ':')),
                        digest.hexdigest()))
                db.execute('INSERT INTO mutable_bank_event_cursors VALUES (?,?,?,?,?)', (
                    self.index.namespace, event['stream'], self.index.generation,
                    event['position'], cursor))
                db.execute('''INSERT INTO event_streams VALUES (?,?,?,?,?)
                    ON CONFLICT(namespace,stream,generation) DO UPDATE SET
                    next_position=excluded.next_position,
                    visibility_time=excluded.visibility_time''', (
                        self.index.namespace, event['stream'], self.index.generation,
                        event['position'] + 1, event['visibility_time']))

        # Existing live records keep their scope, so only their keys change and the
        # resident index is patched in place; anything else takes the full upsert.
        patches: dict[str, tuple[list[int], list[np.ndarray]]] = {}
        for values in encoded:
            _namespace, record_id, space, _generation, domain, created_at, \
                _source_id, key, key_dim, _payload = values
            array = self.index.spaces[space]
            position = int(np.searchsorted(array.ids, record_id))
            vector = np.frombuffer(key, dtype='<f4')
            if (position < len(array.ids) and array.ids[position] == record_id
                    and not array.deleted[position] and array.domains[position] == domain
                    and int(array.times[position]) == created_at
                    and key_dim == array.keys.shape[1]
                    and np.isfinite(vector).all() and np.linalg.norm(vector) > 0):
                patches.setdefault(space, ([], []))
                patches[space][0].append(position)
                patches[space][1].append(vector / np.linalg.norm(vector))
            else:
                self.index.upsert(space, record_id, key, key_dim, domain=domain,
                                  created_at=created_at, source_id=_source_id, deleted=False)
        for space, (positions, vectors) in patches.items():
            self.index.patch_keys(space, np.asarray(positions), np.stack(vectors))
        self.index.set_deleted(set(invalidated), True)
        if self._payload_limit:
            for record in rows:
                self._remember_payload(record.space, record.record_id, cursor, record.payload)
        self.index.bank_cursor = cursor
        return len(encoded)

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[torch.Tensor]]:
        plans = tuple(plans)
        requested = {(plan.space, selection.record_id)
                     for plan in plans for selection in plan.selections}
        overlay: dict[tuple[str, str], torch.Tensor] = {}
        plan_cursors = {plan.bank_cursor for plan in plans if plan.bank_cursor is not None}
        if len(plan_cursors) > 1:
            raise ValueError('One fetch batch cannot mix mutable-bank snapshots')
        with self.cache.connect() as db:
            current = int(self._meta(db, 'cursor'))
            floor = int(self._meta(db, 'floor_cursor'))
            cursor = next(iter(plan_cursors), current)
            if (cursor != 0 and cursor < floor) or cursor > current:
                raise ValueError('Read plan names an unavailable mutable-bank cursor')
            invalidated = self._invalidated(db, cursor)
            authorized = {(plan.space, selection.record_id): plan
                          for plan in plans for selection in plan.selections}
            for space, record_id in requested:
                if record_id in invalidated:
                    raise KeyError(f'Invalidated compact record: {record_id}')
                cached = (self._cached_payload(space, record_id, cursor)
                          if self._payload_limit else None)
                if cached is not None and self._authorized(
                        authorized[(space, record_id)], record_id):
                    overlay[(space, record_id)] = cached
                    self.payload_cache_hits += 1
                    continue
                self.payload_cache_misses += 1
                row = db.execute('''SELECT payload,cursor FROM mutable_bank_revisions
                    WHERE namespace=? AND record_id=? AND space=? AND cursor<=?
                    ORDER BY cursor DESC LIMIT 1''',
                    (self.index.namespace, record_id, space, cursor)).fetchone()
                if row is not None:
                    overlay[(space, record_id)] = load(row[0])['payload']
                    if cursor == current:
                        self._remember_payload(space, record_id, int(row[1]),
                                               overlay[(space, record_id)])

        result = []
        for plan in plans:
            missing = tuple(selection for selection in plan.selections
                            if (plan.space, selection.record_id) not in overlay)
            base_values = self.base.fetch(ReadPlan(
                plan.namespace, plan.space, plan.generation, plan.domain,
                plan.query_time, missing, plan.bank_cursor)) if missing else []
            fallback = {selection.record_id: value
                        for selection, value in zip(missing, base_values, strict=True)}
            if cursor == current:
                # No journal revision exists at or before the newest cursor, so the
                # base payload is the record's newest revision.
                for record_id, value in fallback.items():
                    self._remember_payload(plan.space, record_id, 0, value)
            values = [overlay.get((plan.space, selection.record_id),
                                  fallback.get(selection.record_id))
                      for selection in plan.selections]
            if any(value is None for value in values):
                raise KeyError('Read plan refers to an unavailable logical record')
            result.append(values)
        return result

    def maintenance_ids(self, count: int, *, exclude: Iterable[str] = (),
                        eligible: Iterable[str] | None = None) -> tuple[str, ...]:
        """Rotate fairly through eligible IDs without rescanning the journal."""
        if count < 0:
            raise ValueError('Maintenance sample count cannot be negative')
        omitted = set(exclude)
        ids = next(iter(self.index.spaces.values())).ids
        if not len(ids) or count == 0:
            return ()
        allowed = set(eligible) if eligible is not None else None
        with self.cache.connect() as db:
            position = int(self._meta(db, 'maintenance_position')) % len(ids)
            chosen = []
            inspected = 0
            while inspected < len(ids) and len(chosen) < count:
                record_id = str(ids[(position + inspected) % len(ids)])
                inspected += 1
                if record_id not in omitted and (allowed is None or record_id in allowed):
                    chosen.append(record_id)
            db.execute("UPDATE mutable_bank_meta SET value=? "
                       "WHERE key='maintenance_position'",
                       (str((position + inspected) % len(ids)),))
        return tuple(chosen)

    def invalidated_records(self) -> tuple[dict, ...]:
        """Return the durable rebuild queue for derived compact records."""
        with self.cache.connect() as db:
            cursor = int(self._meta(db, 'cursor'))
            invalidated = self._invalidated(db, cursor)
            result = []
            for record_id in sorted(invalidated):
                status = db.execute('''SELECT cause_id,cursor FROM mutable_bank_status
                    WHERE namespace=? AND record_id=? AND cursor<=?
                    ORDER BY cursor DESC LIMIT 1''',
                    (self.index.namespace, record_id, cursor)).fetchone()
                children = tuple(row[0] for row in db.execute('''SELECT child_id
                    FROM mutable_bank_dependencies WHERE namespace=? AND parent_id=?
                    AND introduced_cursor<=? AND
                    (retired_cursor IS NULL OR retired_cursor>?) ORDER BY child_id''',
                    (self.index.namespace, record_id, cursor, cursor)).fetchall())
                result.append({'record_id': record_id, 'cause_id': status[0],
                               'invalidated_at': status[1], 'children': children})
        return tuple(result)

    def claim_rebuilds(self, worker_id: str, *, limit: int = 1,
                       lease_seconds: float = 300,
                       now_ns: int | None = None) -> tuple[dict, ...]:
        """Lease ready compact descendants in dependency order.

        A parent is ready only after all invalidated compact children have been
        rebuilt. Leases coordinate workers; publication remains the all-space
        ``update`` transaction and clears the lease atomically.
        """
        if not worker_id or limit < 1 or lease_seconds <= 0:
            raise ValueError('A rebuild claim needs a worker, positive limit and lease')
        now = time.time_ns() if now_ns is None else now_ns
        if now < 0:
            raise ValueError('Rebuild lease time cannot be negative')
        lease_until = now + int(lease_seconds * 1_000_000_000)
        with self.cache.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cursor = int(self._meta(db, 'cursor'))
            invalidated = self._invalidated(db, cursor)
            db.execute('''DELETE FROM mutable_bank_rebuild_leases
                WHERE namespace=? AND lease_until_ns<=?''',
                       (self.index.namespace, now))
            jobs = []
            for record_id in sorted(invalidated):
                children = tuple(row[0] for row in db.execute('''SELECT child_id
                    FROM mutable_bank_dependencies WHERE namespace=? AND parent_id=?
                    AND introduced_cursor<=? AND
                    (retired_cursor IS NULL OR retired_cursor>?) ORDER BY child_id''',
                    (self.index.namespace, record_id, cursor, cursor)).fetchall())
                if not children or any(child in invalidated for child in children):
                    continue
                child_cursors = {
                    child: db.execute('''SELECT COALESCE(MAX(cursor),0)
                        FROM mutable_bank_heads WHERE namespace=? AND record_id=?''',
                        (self.index.namespace, child)).fetchone()[0]
                    for child in children
                }
                status = db.execute('''SELECT cause_id,cursor FROM mutable_bank_status
                    WHERE namespace=? AND record_id=? AND cursor<=?
                    ORDER BY cursor DESC LIMIT 1''',
                    (self.index.namespace, record_id, cursor)).fetchone()
                lease = db.execute('''SELECT worker_id,lease_until_ns
                    FROM mutable_bank_rebuild_leases
                    WHERE namespace=? AND record_id=?''',
                    (self.index.namespace, record_id)).fetchone()
                if lease is not None:
                    continue
                db.execute('INSERT INTO mutable_bank_rebuild_leases VALUES (?,?,?,?,?)', (
                    self.index.namespace, record_id, worker_id, lease_until, status[1]))
                jobs.append({'record_id': record_id, 'cause_id': status[0],
                             'invalidated_at': status[1], 'children': children,
                             'child_cursors': child_cursors,
                             'worker_id': worker_id, 'lease_until_ns': lease_until})
                if len(jobs) == limit:
                    break
        return tuple(jobs)

    def release_rebuilds(self, worker_id: str, record_ids: Iterable[str]) -> int:
        """Release this worker's jobs after cancellation or failed inference."""
        ids = tuple(dict.fromkeys(record_ids))
        if not worker_id or any(not record_id for record_id in ids):
            raise ValueError('A rebuild release needs valid worker and record IDs')
        if not ids:
            return 0
        placeholders = ','.join('?' for _ in ids)
        with self.cache.connect() as db:
            return db.execute(f'''DELETE FROM mutable_bank_rebuild_leases
                WHERE namespace=? AND worker_id=? AND record_id IN ({placeholders})''',
                (self.index.namespace, worker_id, *ids)).rowcount

    def rollback(self, cursor: int, *, maintenance_position: int | None = None) -> None:
        """Discard a later uncheckpointed branch and restore logical heads."""
        with self.cache.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            floor = int(self._meta(db, 'floor_cursor'))
            current = int(self._meta(db, 'cursor'))
            if (cursor != 0 and cursor < floor) or cursor > current:
                raise ValueError('Mutable-bank cursor is outside retained journal history')
            db.execute('DELETE FROM mutable_bank_revisions WHERE cursor>?', (cursor,))
            with self._payload_lock:
                self._payloads.clear()
                self._payload_bytes = 0
            DiskStore._rollback_mutable_events(db, cursor)
            db.execute('DELETE FROM mutable_bank_commits WHERE cursor>?', (cursor,))
            db.execute('DELETE FROM mutable_bank_status WHERE cursor>?', (cursor,))
            db.execute('DELETE FROM mutable_bank_dependencies WHERE introduced_cursor>?',
                       (cursor,))
            db.execute('UPDATE mutable_bank_dependencies SET retired_cursor=NULL '
                       'WHERE retired_cursor>?', (cursor,))
            db.execute('DELETE FROM mutable_bank_heads')
            db.execute('DELETE FROM mutable_bank_rebuild_leases')
            db.execute('''INSERT INTO mutable_bank_heads
                SELECT namespace,record_id,space,MAX(cursor)
                FROM mutable_bank_revisions WHERE cursor<=?
                GROUP BY namespace,record_id,space''', (cursor,))
            db.execute("UPDATE mutable_bank_meta SET value=? WHERE key='cursor'", (str(cursor),))
            if cursor == 0:
                db.execute("UPDATE mutable_bank_meta SET value='0' "
                           "WHERE key='floor_cursor'")
            if maintenance_position is not None:
                db.execute("UPDATE mutable_bank_meta SET value=? "
                           "WHERE key='maintenance_position'", (str(maintenance_position),))
        self.index.reload(self.base)
        self._reload_index()

    def sizes(self) -> dict[str, int]:
        with self.cache.connect() as db:
            revisions, payload_bytes = db.execute('''SELECT COUNT(*),
                COALESCE(SUM(LENGTH(payload)+LENGTH(key)),0)
                FROM mutable_bank_revisions''').fetchone()
            heads = db.execute('SELECT COUNT(*) FROM mutable_bank_heads').fetchone()[0]
            cursor = int(self._meta(db, 'cursor'))
            invalid = len(self._invalidated(db, cursor))
        return {'views': heads, 'revision_views': revisions,
                'tensor_bytes': payload_bytes, 'cursor': cursor,
                'invalidated_records': invalid}
