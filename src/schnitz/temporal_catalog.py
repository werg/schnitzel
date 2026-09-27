"""Transitional retrieval over a frozen base and append-only authored records."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .key_index import PublishedKeyIndex
from .store import DiskStore, ReadPlan, Selection, StoredRecord


@dataclass
class _GrowingSpace:
    ids: np.ndarray
    keys: np.ndarray
    domains: np.ndarray
    times: np.ndarray
    deleted: np.ndarray
    count: int
    positions: dict[str, int]


class GrowingCatalogIndex:
    """Merge base-snapshot results with append-only temporal records.

    This preserves the current physical snapshot implementation while the unified
    mutable-record journal is pending. Returned plans use a virtual catalog scope;
    :class:`CatalogStore` resolves every selected opaque ID back to its physical
    revision before fetching bytes.
    """

    def __init__(self, parent: PublishedKeyIndex, authored: DiskStore, *,
                 authored_namespace: str, authored_generation: str,
                 capacity: int, catalog_generation: str) -> None:
        if capacity < 1 or not authored_namespace or not authored_generation:
            raise ValueError("A growing catalog needs positive capacity and generation identity")
        self.parent = parent
        self.authored_store = authored
        self.authored_namespace = authored_namespace
        self.authored_generation = authored_generation
        self.namespace = "catalog"
        self.generation = catalog_generation
        self.spaces: dict[str, _GrowingSpace] = {}
        for name, parent_space in parent.spaces.items():
            key_dim = parent_space.keys.shape[1]
            with authored.connect() as db:
                rows = db.execute(
                    """SELECT record_id,key,key_dim,domain,created_at,deleted FROM records
                       WHERE namespace=? AND generation=? AND space=? ORDER BY record_id""",
                    (authored_namespace, authored_generation, name),
                ).fetchall()
            if len(rows) > capacity or any(row[2] != key_dim for row in rows):
                raise ValueError("Existing authored keys exceed capacity or differ in width")
            ids = np.empty(capacity, dtype=object)
            keys = np.empty((capacity, key_dim), dtype=np.float32)
            domains = np.empty(capacity, dtype=object)
            times = np.empty(capacity, dtype=np.int64)
            deleted = np.zeros(capacity, dtype=bool)
            positions = {}
            for position, row in enumerate(rows):
                record_id, blob, _, domain, created_at, removed = row
                if record_id in positions or record_id in set(parent_space.ids):
                    raise ValueError("Catalog record IDs must be globally unique")
                vector = np.frombuffer(blob, dtype="<f4").copy()
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                ids[position], keys[position] = record_id, vector
                domains[position], times[position], deleted[position] = domain, created_at, bool(removed)
                positions[record_id] = position
            self.spaces[name] = _GrowingSpace(
                ids, keys, domains, times, deleted, len(rows), positions)
        counts = {space.count for space in self.spaces.values()}
        id_sets = {tuple(space.ids[:space.count]) for space in self.spaces.values()}
        if len(counts) != 1 or len(id_sets) != 1:
            raise ValueError("Authored generation is incomplete across spaces")
        self.capacity = capacity
        self.key_bytes = parent.key_bytes + sum(space.keys.nbytes for space in self.spaces.values())
        self._parent_ids = {name: frozenset(map(str, space.ids))
                            for name, space in parent.spaces.items()}

    @property
    def authored_count(self) -> int:
        return next(iter(self.spaces.values())).count

    @property
    def max_created_at(self) -> int:
        parent_max = max(int(space.times.max(initial=-1)) for space in self.parent.spaces.values())
        authored_max = max(int(space.times[:space.count].max(initial=-1))
                           for space in self.spaces.values())
        return max(parent_max, authored_max)

    def origin(self, space: str, record_id: str) -> str:
        if record_id in self._parent_ids[space]:
            return "parent"
        if record_id in self.spaces[space].positions:
            return "authored"
        raise KeyError(f"Unknown catalog record: {record_id}")

    def add_records(self, records: Iterable[StoredRecord]) -> None:
        rows = list(records)
        grouped: dict[str, dict[str, StoredRecord]] = {}
        for record in rows:
            if (record.namespace != self.authored_namespace
                    or record.generation != self.authored_generation
                    or record.space not in self.spaces):
                raise ValueError("Appended record differs from the authored catalog scope")
            grouped.setdefault(record.record_id, {})[record.space] = record
        expected = set(self.spaces)
        if (not grouped or any(set(by_space) != expected for by_space in grouped.values())
                or len(rows) != len(grouped) * len(expected)):
            raise ValueError("Every appended logical record needs one view in every space")
        if self.authored_count + len(grouped) > self.capacity:
            raise MemoryError("Growing catalog capacity exhausted")
        for record_id, by_space in grouped.items():
            if any(record_id in self._parent_ids[space]
                   or record_id in self.spaces[space].positions for space in expected):
                raise ValueError("Catalog record IDs cannot be reused")
            for name in sorted(expected):
                record = by_space[name]
                space = self.spaces[name]
                position = space.count
                vector = record.key.detach().float().cpu().numpy().copy()
                if vector.shape != (space.keys.shape[1],) or not np.isfinite(vector).all():
                    raise ValueError("Appended catalog key has invalid shape or values")
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                space.ids[position], space.keys[position] = record_id, vector
                space.domains[position] = record.domain
                space.times[position] = record.created_at
                space.deleted[position] = False
                space.positions[record_id] = position
                space.count += 1

    def search_batch(self, queries: Tensor, *, top_k: int = 16,
                     namespace: str = "catalog", space: str = "s0",
                     generation: str = "", domains: Sequence[str],
                     query_times: Sequence[int],
                     exclude_ids: Sequence[frozenset[str]] | None = None) -> tuple[ReadPlan, ...]:
        if (namespace != self.namespace or generation != self.generation
                or space not in self.spaces):
            raise ValueError("Catalog search scope differs from its generation")
        excluded = tuple(exclude_ids or (frozenset(),) * queries.shape[0])
        parent_plans = self.parent.search_batch(
            queries, top_k=top_k, namespace=self.parent.namespace, space=space,
            generation=self.parent.generation, domains=domains, query_times=query_times,
            exclude_ids=excluded,
        )
        authored = self.spaces[space]
        q = queries.detach().float().cpu().numpy().copy()
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        scores = q @ authored.keys[:authored.count].T if authored.count else np.empty((len(q), 0))
        plans = []
        for row, (domain, query_time, omitted, parent_plan) in enumerate(zip(
                domains, query_times, excluded, parent_plans, strict=True)):
            candidates = [(selection.score, selection.record_id)
                          for selection in parent_plan.selections]
            if authored.count:
                eligible = ((authored.domains[:authored.count] == domain)
                            & (authored.times[:authored.count] < query_time)
                            & ~authored.deleted[:authored.count])
                if omitted:
                    eligible &= ~np.isin(authored.ids[:authored.count], tuple(omitted))
                indices = np.flatnonzero(eligible)
                order = np.lexsort((authored.ids[indices], -scores[row, indices]))[:top_k]
                candidates.extend((float(scores[row, position]), str(authored.ids[position]))
                                  for position in indices[order])
            candidates.sort(key=lambda pair: (-pair[0], pair[1]))
            plans.append(ReadPlan(
                self.namespace, space, self.generation, domain, query_time,
                tuple(Selection(record_id, score) for score, record_id in candidates[:top_k]),
            ))
        return tuple(plans)

    def keys_for_ids(self, space: str, record_ids: Sequence[str], *,
                     domain: str, query_time: int) -> Tensor:
        if space not in self.spaces or not record_ids:
            raise ValueError("Catalog key lookup needs a known space and record IDs")
        authored = self.spaces[space]
        result = np.empty((len(record_ids), authored.keys.shape[1]), dtype=np.float32)
        parent_ids, parent_positions = [], []
        for output_position, record_id in enumerate(record_ids):
            if record_id in self._parent_ids[space]:
                parent_ids.append(record_id)
                parent_positions.append(output_position)
                continue
            position = authored.positions.get(record_id)
            if position is None:
                raise KeyError("Record ID is absent from the catalog")
            if (authored.domains[position] != domain or authored.times[position] >= query_time
                    or authored.deleted[position]):
                raise PermissionError("Authored key is outside its causal authorization scope")
            result[output_position] = authored.keys[position]
        if parent_ids:
            keys = self.parent.keys_for_ids(
                space, parent_ids, domain=domain, query_time=query_time).numpy()
            result[np.asarray(parent_positions)] = keys
        return torch.from_numpy(result)


class CatalogStore:
    """Resolve virtual catalog read plans to their immutable physical stores."""

    def __init__(self, parent: DiskStore, authored: DiskStore,
                 index: GrowingCatalogIndex) -> None:
        self.parent, self.authored, self.index = parent, authored, index

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[Tensor]]:
        plans = list(plans)
        results: list[list[Tensor | None]] = [
            [None] * len(plan.selections) for plan in plans]
        parent_jobs, authored_jobs = [], []
        parent_maps, authored_maps = [], []
        for plan_index, plan in enumerate(plans):
            if plan.namespace != self.index.namespace or plan.generation != self.index.generation:
                raise ValueError("Read plan is outside the virtual catalog")
            groups = {"parent": [], "authored": []}
            mappings = {"parent": [], "authored": []}
            for selection_index, selection in enumerate(plan.selections):
                origin = self.index.origin(plan.space, selection.record_id)
                groups[origin].append(selection)
                mappings[origin].append((plan_index, selection_index))
            if groups["parent"]:
                parent_jobs.append(ReadPlan(
                    self.index.parent.namespace, plan.space, self.index.parent.generation,
                    plan.domain, plan.query_time, tuple(groups["parent"])))
                parent_maps.append(mappings["parent"])
            if groups["authored"]:
                authored_jobs.append(ReadPlan(
                    self.index.authored_namespace, plan.space,
                    self.index.authored_generation, plan.domain, plan.query_time,
                    tuple(groups["authored"])))
                authored_maps.append(mappings["authored"])
        for jobs, mappings, store in (
                (parent_jobs, parent_maps, self.parent),
                (authored_jobs, authored_maps, self.authored)):
            if not jobs:
                continue
            for values, locations in zip(store.fetch_many(jobs), mappings, strict=True):
                for value, (plan_index, selection_index) in zip(values, locations, strict=True):
                    results[plan_index][selection_index] = value
        if any(value is None for row in results for value in row):
            raise AssertionError("Catalog fetch failed to resolve a selected payload")
        return [[value for value in row if value is not None] for row in results]

    def fetch(self, plan: ReadPlan) -> list[Tensor]:
        return self.fetch_many((plan,))[0]
