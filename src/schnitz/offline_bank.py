"""Atomic identity and integrity checks for resumable frozen offline writes."""
from collections.abc import Callable, Iterable
import hashlib
import json
from pathlib import Path

from .store import DiskStore, StoredRecord


CONSUMER_MEMORY_FIELDS = frozenset({
    # neighbors caps how many records a consumer reads; it does not change what is stored.
    'neighbors', 'read_steps', 'checkpoint_chunks', 'distance_gating', 'gate_density_k', 'gate_min_temperature',
    'gate_max_temperature', 'gate_initial_temperature',
    'gate_max_radius_adjustment', 'gate_density_fraction', 'gate_floor_mode',
})


def stored_memory_identity(memory: dict) -> dict:
    """Fields that determine serialized keys/payloads rather than read behavior."""
    result = {key: value for key, value in memory.items()
              if key not in CONSUMER_MEMORY_FIELDS}
    # Banks published before the positional interface are unambiguously flat.
    # Preserve their compatibility without mutating historical manifests.
    result.setdefault('payload_layout', 'flat')
    # Banks published before per-space direct heads used shared key maps.
    result.setdefault('key_interface', 'shared_maps')
    result.setdefault('space_tokens', [])
    result.setdefault('key_dims', [])
    result.setdefault('key_slot_position', 'first')
    return result


def assert_bank_writer_compatible(run: Path, checkpoint: Path, bank_dir: Path,
                                  manifest: dict, *, training_bank_dir: str | Path | None) -> None:
    """Reject stale payloads if an evaluator's writer differs from bank creation.

    A published-bank training stage may change consumer and query weights while
    its producer is frozen. In that case compare actual serialized producer
    tensors to the exact checkpoint that created the bank.
    """
    from safetensors import safe_open
    import torch

    from .trajectories import file_sha256

    expected = manifest['identity']['writer_checkpoint_sha256']
    current = checkpoint / 'model.safetensors'
    if file_sha256(current) == expected:
        return
    if training_bank_dir is None or Path(training_bank_dir).resolve() != bank_dir.resolve():
        raise ValueError('Evaluator model was not trained against this bank generation')
    initialization = json.loads((run / 'initialization.json').read_text())
    source = Path(initialization['checkpoint']) / 'model.safetensors'
    if file_sha256(source) != expected:
        raise ValueError('Bank-stage initialization differs from the frozen writer checkpoint')
    # writer_loops=1 bypasses the recurrent bridge; only the parent decoder
    # participates in source encoding. The bridge is allowed to learn on reads.
    prefixes = ('backbone.base.', 'write_slots', 'key_head.', 'value_head.',
                'address_maps.', 'writer_key_heads.', 'codecs.')
    with safe_open(source, framework='pt', device='cpu') as origin, \
            safe_open(current, framework='pt', device='cpu') as evaluated:
        keys = {key for key in origin.keys() if key.startswith(prefixes)}
        if not keys or keys != {key for key in evaluated.keys() if key.startswith(prefixes)}:
            raise ValueError('Bank-stage writer parameter set changed')
        for key in sorted(keys):
            if not torch.equal(origin.get_tensor(key), evaluated.get_tensor(key)):
                raise ValueError(f'Bank-stage writer parameters changed: {key}')


def assert_bank_reader_compatible(run: Path, checkpoint: Path, bank_dir: Path,
                                  manifest: dict, config) -> dict:
    """Prove that a checkpoint can consume a bank without claiming writer identity.

    Spatial stages may adapt recurrent decoder weights after a frozen generation
    was written. Their immutable launch record ties the consumer to that bank;
    read evaluation needs that relationship and the stored interface, while a
    fresh generic evaluator still needs exact writer compatibility.
    """
    from dataclasses import asdict

    from .trajectories import file_sha256

    current_memory = stored_memory_identity(asdict(config.memory))
    bank_memory = stored_memory_identity(dict(manifest['identity']['memory']))
    bank_model = manifest['identity']['model']
    if current_memory != bank_memory or any(
            bank_model[name] != getattr(config.model, name)
            for name in ('model_id', 'revision', 'writer_loops')):
        raise ValueError('Bank stored interface differs from the reader checkpoint')
    spatial = run / 'spatial-inputs.json'
    if spatial.is_file():
        inputs = json.loads(spatial.read_text())
        if (Path(inputs['bank']).resolve() != bank_dir.resolve()
                or inputs['bank_manifest_sha256'] != file_sha256(bank_dir / 'manifest.json')
                or inputs['generation'] != manifest['generation']):
            raise ValueError('Spatial consumer was trained against another bank generation')
        return {'relationship': 'trained_spatial_consumer', 'checkpoint_is_bank_writer_snapshot':
                file_sha256(checkpoint / 'model.safetensors') ==
                manifest['identity']['writer_checkpoint_sha256']}
    assert_bank_writer_compatible(
        run, checkpoint, bank_dir, manifest, training_bank_dir=config.train.bank_dir)
    return {'relationship': 'writer_compatible',
            'checkpoint_is_bank_writer_snapshot':
            file_sha256(checkpoint / 'model.safetensors') ==
            manifest['identity']['writer_checkpoint_sha256']}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _contents(db, namespace, generation, spaces):
    digest, count = hashlib.sha256(), 0
    placeholders = ','.join('?' for _ in spaces)
    rows = db.execute(f'''SELECT * FROM records WHERE namespace=? AND generation=?
                         AND space IN ({placeholders}) ORDER BY space,record_id''',
                      (namespace, generation, *spaces))
    for row in rows:
        count += 1
        for value in row:
            encoded = value if isinstance(value, bytes) else canonical_json(value).encode()
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return count, digest.hexdigest()


def ensure_offline_records(store: DiskStore, factory: Callable[[], Iterable[StoredRecord]], *,
                           identity: dict, namespace: str, generation: str,
                           spaces: tuple[str, ...], expected_count: int) -> bool:
    """Return True only when this call writes a new bank.

    Caller supplies verified frozen-writer identity and source/config identity.
    Records and manifest commit together. Resuming validates the raw bytes without
    invoking factory; derived code views are outside this raw-bank digest. Existing
    banks without this manifest fail closed: use a new path for an explicit offline
    rebuild. This helper neither adopts legacy data nor changes immutable inserts.
    """
    if not spaces or len(spaces) != len(set(spaces)) or expected_count < 0:
        raise ValueError('Distinct raw spaces and nonnegative count required')
    serialized = canonical_json(identity)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('''CREATE TABLE IF NOT EXISTS offline_banks (
            namespace TEXT NOT NULL, generation TEXT NOT NULL,
            identity TEXT NOT NULL, records INTEGER NOT NULL, digest TEXT NOT NULL,
            PRIMARY KEY(namespace,generation))''')
        previous = db.execute('SELECT identity,records,digest FROM offline_banks WHERE namespace=? AND generation=?',
                              (namespace, generation)).fetchone()
        if previous is not None and previous[0] != serialized:
            raise ValueError('Offline bank identity changed')
        count, digest = _contents(db, namespace, generation, spaces)
        if previous is not None:
            if (count, digest) != previous[1:] or count != expected_count:
                raise ValueError('Offline bank raw contents changed')
            return False
        if count:
            raise ValueError('Existing bank has no offline manifest; rebuild offline at a new path')
        for record in factory():
            if record.namespace != namespace or record.generation != generation or record.space not in spaces:
                raise ValueError('Offline writer emitted an unexpected record scope')
            store._put(db, record, ())
        count, digest = _contents(db, namespace, generation, spaces)
        if count != expected_count:
            raise ValueError('Offline writer record count differs')
        db.execute('INSERT INTO offline_banks VALUES (?,?,?,?,?)',
                   (namespace, generation, serialized, count, digest))
    return True


def _shard_schema(db):
    # Shard verification filters generation plus bounded record-ID sets. The
    # records primary key orders generation last, while the serving eligibility
    # index orders record ID last; either can devolve into repeated generation
    # scans as a bank grows. Build this before publication and bulk ingestion.
    db.execute('''CREATE INDEX IF NOT EXISTS offline_record_scope ON records
                  (namespace,generation,record_id,space)''')
    db.execute('''CREATE TABLE IF NOT EXISTS offline_shards (
        namespace TEXT NOT NULL, generation TEXT NOT NULL, shard_id TEXT NOT NULL,
        identity TEXT NOT NULL, source_ids TEXT NOT NULL, records INTEGER NOT NULL,
        digest TEXT NOT NULL, PRIMARY KEY(namespace,generation,shard_id))''')
    db.execute('''CREATE TABLE IF NOT EXISTS offline_generations (
        namespace TEXT NOT NULL, generation TEXT NOT NULL, manifest TEXT NOT NULL,
        PRIMARY KEY(namespace,generation))''')


def _shard_contents(db, namespace, generation, spaces, source_ids):
    if not source_ids:
        return 0, hashlib.sha256().hexdigest()
    query = f'''SELECT * FROM records INDEXED BY offline_record_scope
                 WHERE namespace=? AND generation=?
                 AND space IN ({','.join('?' for _ in spaces)})
                 AND record_id IN ({','.join('?' for _ in source_ids)})
                 ORDER BY space,record_id'''
    digest, count = hashlib.sha256(), 0
    for row in db.execute(query, (namespace, generation, *spaces, *source_ids)):
        count += 1
        for value in row:
            encoded = value if isinstance(value, bytes) else canonical_json(value).encode()
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return count, digest.hexdigest()


def ensure_offline_shard(store: DiskStore, factory: Callable[[], Iterable[StoredRecord]], *,
                         identity: dict, namespace: str, generation: str,
                         spaces: tuple[str, ...], shard_id: str,
                         source_ids: tuple[str, ...]) -> bool:
    """Atomically publish every space of a bounded immutable source shard.

    A completed shard is byte-verified and never re-encoded on resume. The caller
    must finish all shards and publish the generation before using it for training.
    """
    if (not shard_id or not spaces or len(spaces) != len(set(spaces)) or
            not source_ids or len(source_ids) != len(set(source_ids))):
        raise ValueError('Distinct spaces and source IDs plus a shard ID required')
    serialized, ids_json = canonical_json(identity), canonical_json(source_ids)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        _shard_schema(db)
        previous = db.execute('''SELECT identity,source_ids,records,digest FROM offline_shards
                                WHERE namespace=? AND generation=? AND shard_id=?''',
                              (namespace, generation, shard_id)).fetchone()
        count, digest = _shard_contents(db, namespace, generation, spaces, source_ids)
        if previous is not None:
            if previous[:2] != (serialized, ids_json):
                raise ValueError('Offline shard identity or sources changed')
            if previous[2:] != (count, digest) or count != len(source_ids) * len(spaces):
                raise ValueError('Offline shard raw contents changed')
            return False
        if count:
            raise ValueError('Offline shard has unmanifested records')
        seen = set()
        for record in factory():
            pair = (record.record_id, record.space)
            if (record.namespace != namespace or record.generation != generation or
                    record.record_id not in source_ids or record.space not in spaces or
                    record.source_id != record.record_id or pair in seen):
                raise ValueError('Offline shard emitted an unexpected source/space identity')
            seen.add(pair)
            store._put(db, record, ())
        expected = {(source_id, space) for source_id in source_ids for space in spaces}
        if seen != expected:
            raise ValueError('Offline shard does not contain every complete source view')
        count, digest = _shard_contents(db, namespace, generation, spaces, source_ids)
        db.execute('INSERT INTO offline_shards VALUES (?,?,?,?,?,?,?)',
                   (namespace, generation, shard_id, serialized, ids_json, count, digest))
    return True


def publish_offline_generation(store: DiskStore, *, identity: dict, namespace: str,
                               generation: str, spaces: tuple[str, ...],
                               shard_ids: tuple[str, ...], source_count: int,
                               verify_only: bool = False) -> dict:
    """Verify every shard; published readers use a shared read transaction."""
    if not shard_ids or len(shard_ids) != len(set(shard_ids)) or source_count < 1:
        raise ValueError('Distinct shards and positive source count required')
    with store.connect() as db:
        db.execute('BEGIN' if verify_only else 'BEGIN IMMEDIATE')
        if not verify_only:
            _shard_schema(db)
        rows = db.execute('''SELECT shard_id,identity,source_ids,records,digest FROM offline_shards
                             WHERE namespace=? AND generation=? ORDER BY shard_id''',
                          (namespace, generation)).fetchall()
        if {row[0] for row in rows} != set(shard_ids):
            raise ValueError('Offline generation shard set is incomplete or changed')
        ids = set()
        for shard_id, frozen, encoded_ids, count, digest in rows:
            members = tuple(json.loads(encoded_ids))
            if frozen != canonical_json(identity) or ids.intersection(members):
                raise ValueError('Offline generation identity or shard overlap changed')
            actual = _shard_contents(db, namespace, generation, spaces, members)
            if actual != (count, digest) or count != len(members) * len(spaces):
                raise ValueError(f'Offline shard raw contents changed: {shard_id}')
            ids.update(members)
        total, digest = _contents(db, namespace, generation, spaces)
        if len(ids) != source_count or total != source_count * len(spaces):
            raise ValueError('Offline generation source or record count changed')
        manifest = {'identity': identity, 'namespace': namespace, 'generation': generation,
                    'spaces': spaces, 'shards': shard_ids, 'sources': source_count,
                    'records': total, 'digest': digest}
        serialized = canonical_json(manifest)
        prior = db.execute('SELECT manifest FROM offline_generations WHERE namespace=? AND generation=?',
                           (namespace, generation)).fetchone()
        if prior is not None and prior[0] != serialized:
            raise ValueError('Offline generation publication changed')
        if prior is None:
            if verify_only:
                raise ValueError('Offline generation has not been published')
            db.execute('INSERT INTO offline_generations VALUES (?,?,?)',
                       (namespace, generation, serialized))
    return manifest


def writer_prompt_generation(manifest: dict) -> str:
    """Generation string the writer saw in its ingestion prompts for this bank.

    A bank may record the generation its writer was trained with, so stored keys
    come from the same prompt distribution as training; otherwise it is the bank's
    own generation.
    """
    return manifest['identity'].get('writer_prompt_generation') or manifest['generation']
