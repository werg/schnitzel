"""Persistent, versioned full-cluster codes with exact raw fallback for partial selections.

Original keys stay in the ordinary index. A code replaces only a complete selected
cluster; no learned decoder is asked to enforce subset membership or authorization.
Raw values are retained for fallback, so this prototype reduces selected read payload,
not total disk usage. Deletion lineage invalidates shared codes structurally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
from torch import Tensor
from safetensors.torch import load, save

from .store import DiskStore, StoredRecord, ReadPlan, Selection, lookup_record


def state_fingerprint(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class CompactRead:
    values: Tensor
    weights: Tensor
    used_clusters: list[str]
    raw_fallback_ids: list[str]
    serialized_value_bytes: int
    logical_tensor_bytes: int


class ClusterBank:
    def __init__(self, store: DiskStore, *, view: str, reader_hash: str):
        if not view or not reader_hash:
            raise ValueError('Explicit compaction view and reader identity required')
        self.store, self.view, self.reader_hash = store, view, reader_hash
        with store.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS cluster_codes (
                namespace TEXT, parent_id TEXT, view TEXT, reader_hash TEXT,
                space TEXT, generation TEXT, domain TEXT, children TEXT, tensors BLOB,
                PRIMARY KEY(namespace,parent_id,view));
            CREATE TABLE IF NOT EXISTS cluster_members (
                namespace TEXT, space TEXT, generation TEXT, view TEXT, child_id TEXT, parent_id TEXT,
                PRIMARY KEY(namespace,space,generation,view,child_id));
            ''')

    def put(self, plan: ReadPlan, values: Tensor, weights: Tensor) -> str:
        children = tuple(s.record_id for s in plan.selections)
        if not children or len(set(children)) != len(children):
            raise ValueError('Distinct, nonempty children required')
        if values.ndim != 2 or weights.shape != (values.shape[0],) or not values.is_floating_point():
            raise ValueError('Values [K,D], multiplicities [K] required')
        if not torch.isfinite(values).all() or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid code tensors')
        if not torch.isclose(weights.float().sum(), weights.new_tensor(float(len(children)), dtype=torch.float32), rtol=1e-5, atol=1e-6):
            raise ValueError('Code must preserve full-cluster multiplicity')
        records = [lookup_record(self.store, rid, namespace=plan.namespace, space=plan.space,
                                  generation=plan.generation, domain=plan.domain, query_time=plan.query_time)
                   for rid in children]
        if any(r.payload.ndim != 1 or r.payload.numel() != values.shape[1] for r in records):
            raise ValueError('Code width differs from child payload interface')
        blob = save({'values': values.detach().cpu().contiguous(), 'weights': weights.detach().float().cpu().contiguous()})
        parent = 'cluster_' + hashlib.sha256((self.view + repr(sorted(children))).encode() + blob).hexdigest()[:24]
        marker = StoredRecord(parent, records[0].key, torch.zeros(1), plan.namespace,
                              '__compact__/' + plan.space, self.view, plan.domain,
                              max(r.created_at for r in records), 'full-cluster')
        # Disjointness, live-child revalidation, lineage and code publication share
        # one writer transaction. A failed publication leaves no inert marker.
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for child in children:
                if db.execute('SELECT 1 FROM cluster_members WHERE namespace=? AND space=? AND generation=? AND view=? AND child_id=?',
                              (plan.namespace, plan.space, plan.generation, self.view, child)).fetchone():
                    raise ValueError('Clusters must be disjoint within a persistent view; use a new view')
            self.store._put(db, marker, children)
            db.execute('INSERT INTO cluster_codes VALUES (?,?,?,?,?,?,?,?,?)',
                       (plan.namespace, parent, self.view, self.reader_hash, plan.space,
                        plan.generation, plan.domain, json.dumps(children), blob))
            db.executemany('INSERT INTO cluster_members VALUES (?,?,?,?,?,?)',
                           [(plan.namespace, plan.space, plan.generation, self.view, child, parent) for child in children])
        return parent

    def fetch(self, plan: ReadPlan) -> CompactRead:
        selected = [s.record_id for s in plan.selections]
        if len(set(selected)) != len(selected):
            raise ValueError('Duplicate selected IDs would duplicate evidence')
        values, weights, clusters, raw_ids = [], [], [], []
        used, serialized = set(), 0
        with self.store.connect() as db:
            for rid in selected:
                if rid in used:
                    continue
                row = db.execute('''SELECT c.parent_id,c.reader_hash,c.domain,c.children,c.tensors
                    FROM cluster_members m JOIN cluster_codes c ON m.namespace=c.namespace
                    AND m.parent_id=c.parent_id AND m.view=c.view
                    WHERE m.namespace=? AND m.space=? AND m.generation=? AND m.view=? AND m.child_id=?''',
                    (plan.namespace, plan.space, plan.generation, self.view, rid)).fetchone()
                if row is not None and set(json.loads(row[3])) <= set(selected):
                    parent, identity, domain, child_json, blob = row
                    if identity != self.reader_hash:
                        raise ValueError('Stale reader/code compatibility')
                    if domain != plan.domain:
                        raise PermissionError('Code belongs to a different authorization domain')
                    marker = ReadPlan(plan.namespace, '__compact__/' + plan.space, self.view,
                                      plan.domain, plan.query_time, (Selection(parent, 0.),))
                    # Includes temporal and transitive deletion checks without fetching child values.
                    self.store.fetch(marker)
                    tensors = load(blob)
                    values.append(tensors['values'])
                    weights.append(tensors['weights'])
                    clusters.append(parent)
                    used.update(json.loads(child_json))
                    serialized += len(blob)
                else:
                    single = ReadPlan(plan.namespace, plan.space, plan.generation, plan.domain,
                                      plan.query_time, (Selection(rid, 0.),))
                    payload = self.store.fetch(single)[0]
                    values.append(payload[None])
                    weights.append(torch.ones(1))
                    raw_ids.append(rid)
                    used.add(rid)
                    serialized += db.execute('''SELECT length(payload) FROM records WHERE namespace=? AND record_id=?
                        AND space=? AND generation=?''', (plan.namespace, rid, plan.space, plan.generation)).fetchone()[0]
        if not values:
            raise ValueError('Use the explicit null-read path for empty selections')
        result = torch.cat(values, 0)
        mass = torch.cat(weights, 0)
        logical = result.numel() * result.element_size() + mass.numel() * mass.element_size()
        return CompactRead(result[None], mass[None], clusters, raw_ids, serialized, logical)

    def sizes(self) -> dict:
        with self.store.connect() as db:
            count, size = db.execute('SELECT count(*),coalesce(sum(length(tensors)),0) FROM cluster_codes WHERE view=?', (self.view,)).fetchone()
        return {'codes': count, 'serialized_code_bytes': size,
                'raw_values_retained': True, 'notice': 'Net disk savings are not claimed; raw subset fallback is retained.'}
