"""Resident exact key arrays for a verified physical bank snapshot.

This is a CPU exact scan, not ANN. Stored payloads remain in DiskStore and every
fetch revalidates namespace, domain, physical revision, time and deletion status.
Spatial training may patch these arrays from its checkpointed mutable overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .store import DiskStore, ReadPlan, Selection


@dataclass
class _Space:
    ids: np.ndarray
    keys: np.ndarray
    domains: np.ndarray
    times: np.ndarray
    source_ids: np.ndarray
    deleted: np.ndarray


ELIGIBLE_CACHE_ENTRIES = 256


class PublishedKeyIndex:
    """Load complete raw keys once; search with the SQLite reference semantics."""

    def __init__(self, store: DiskStore, *, namespace: str, generation: str,
                 spaces: tuple[str, ...], expected_sources: int):
        if not namespace or not generation or not spaces or expected_sources < 1:
            raise ValueError('Published index needs a scoped nonempty generation')
        self.namespace, self.generation = namespace, generation
        self.bank_cursor: int | None = None
        self._space_names = spaces
        self._expected_sources = expected_sources
        self.reload(store)

    def reload(self, store: DiskStore) -> None:
        """Restore the physical base view before applying journal revisions."""
        self.spaces = {}
        common_ids = None
        for space in self._space_names:
            with store.connect() as db:
                rows = db.execute('''SELECT record_id,key,key_dim,domain,created_at,source_id,deleted
                                     FROM records WHERE namespace=? AND generation=? AND space=?
                                     ORDER BY record_id''',
                                  (self.namespace, self.generation, space)).fetchall()
            if len(rows) != self._expected_sources:
                raise ValueError('Published bank space is not complete')
            ids = np.asarray([row[0] for row in rows], dtype=object)
            if common_ids is not None and not np.array_equal(ids, common_ids):
                raise ValueError('Published bank spaces have different logical source IDs')
            common_ids = ids
            dimensions = {row[2] for row in rows}
            if len(dimensions) != 1:
                raise ValueError('Published key dimensions differ within one space')
            keys = np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows]).copy()
            if keys.shape[1] != next(iter(dimensions)) or not np.isfinite(keys).all():
                raise ValueError('Published key bytes are incompatible or nonfinite')
            keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
            self.spaces[space] = _Space(ids, keys,
                                        np.asarray([row[3] for row in rows], dtype=object),
                                        np.asarray([row[4] for row in rows], dtype=np.int64),
                                        np.asarray([row[5] for row in rows], dtype=object),
                                        np.asarray([bool(row[6]) for row in rows]))
        if len(self.spaces) != len(self._space_names):
            raise ValueError('Published spaces must be distinct')
        self.key_bytes = sum(array.keys.nbytes for array in self.spaces.values())
        self.invalidate()

    def upsert(self, space: str, record_id: str, key: bytes, key_dim: int, *,
               domain: str, created_at: int, source_id: str = '',
               deleted: bool = False) -> None:
        """Apply one committed journal head to the resident exact index."""
        if space not in self.spaces or not record_id or not domain or created_at < 0:
            raise ValueError('Invalid mutable index revision')
        vector = np.frombuffer(key, dtype='<f4').copy()
        array = self.spaces[space]
        if vector.shape != (key_dim,) or key_dim != array.keys.shape[1]:
            raise ValueError('Mutable key dimension differs from the base index')
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError('Mutable key is nonfinite or zero')
        vector /= norm
        positions = np.flatnonzero(array.ids == record_id)
        if len(positions) > 1:
            raise ValueError('Resident index contains duplicate logical IDs')
        if len(positions) == 1:
            position = int(positions[0])
            array.keys[position] = vector
            array.domains[position] = domain
            array.times[position] = created_at
            if source_id:
                array.source_ids[position] = source_id
            array.deleted[position] = deleted
        else:
            array.ids = np.append(array.ids, record_id)
            array.keys = np.concatenate((array.keys, vector[None]), axis=0)
            array.domains = np.append(array.domains, domain)
            array.times = np.append(array.times, created_at)
            array.source_ids = np.append(array.source_ids, source_id)
            array.deleted = np.append(array.deleted, deleted)
            order = np.argsort(array.ids, kind='stable')
            array.ids, array.keys = array.ids[order], array.keys[order]
            array.domains, array.times = array.domains[order], array.times[order]
            array.source_ids = array.source_ids[order]
            array.deleted = array.deleted[order]
        self.key_bytes = sum(item.keys.nbytes for item in self.spaces.values())
        self.invalidate()

    def upsert_many(self, space: str, rows: Sequence[tuple]) -> None:
        """Apply a journal snapshot with one vectorized existing-ID patch."""
        if space not in self.spaces or not rows:
            return
        array = self.spaces[space]
        ids = np.asarray([row[0] for row in rows], dtype=object)
        if len(set(ids)) != len(ids):
            raise ValueError('Mutable index batch contains duplicate logical IDs')
        dimensions = {row[2] for row in rows}
        if dimensions != {array.keys.shape[1]}:
            raise ValueError('Mutable key dimension differs from the base index')
        keys = np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows]).copy()
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        if not np.isfinite(keys).all() or np.any(norms <= 0):
            raise ValueError('Mutable key is nonfinite or zero')
        keys /= norms
        positions = np.searchsorted(array.ids, ids)
        existing = positions < len(array.ids)
        existing &= np.asarray([
            array.ids[position] == record_id if position < len(array.ids) else False
            for position, record_id in zip(positions, ids, strict=True)
        ])
        old_positions = positions[existing]
        array.keys[old_positions] = keys[existing]
        array.domains[old_positions] = np.asarray(
            [row[3] for row, present in zip(rows, existing, strict=True) if present],
            dtype=object)
        array.times[old_positions] = np.asarray(
            [row[4] for row, present in zip(rows, existing, strict=True) if present],
            dtype=np.int64)
        sources = np.asarray(
            [row[5] for row, present in zip(rows, existing, strict=True) if present],
            dtype=object)
        replace_source = sources != ''
        array.source_ids[old_positions[replace_source]] = sources[replace_source]
        array.deleted[old_positions] = np.asarray(
            [row[6] for row, present in zip(rows, existing, strict=True) if present],
            dtype=bool)
        if not np.all(existing):
            fresh = ~existing
            array.ids = np.concatenate((array.ids, ids[fresh]))
            array.keys = np.concatenate((array.keys, keys[fresh]), axis=0)
            array.domains = np.concatenate((array.domains, np.asarray(
                [row[3] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=object)))
            array.times = np.concatenate((array.times, np.asarray(
                [row[4] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=np.int64)))
            array.source_ids = np.concatenate((array.source_ids, np.asarray(
                [row[5] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=object)))
            array.deleted = np.concatenate((array.deleted, np.asarray(
                [row[6] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=bool)))
            order = np.argsort(array.ids, kind='stable')
            array.ids, array.keys = array.ids[order], array.keys[order]
            array.domains, array.times = array.domains[order], array.times[order]
            array.source_ids, array.deleted = (
                array.source_ids[order], array.deleted[order])
        self.key_bytes = sum(item.keys.nbytes for item in self.spaces.values())
        self.invalidate()

    @staticmethod
    def _positions_of(ids: np.ndarray, record_ids) -> np.ndarray:
        """Positions of the given IDs in a sorted unique ID array (absent IDs skipped).

        ``np.isin`` on object (string) arrays is quadratic-like in practice: 218k
        IDs against a 32k active set took about nine minutes.
        """
        wanted = np.asarray(sorted(record_ids), dtype=object)
        if not len(wanted) or not len(ids):
            return np.zeros(0, dtype=np.int64)
        positions = np.searchsorted(ids, wanted)
        inside = positions < len(ids)
        positions, wanted = positions[inside], wanted[inside]
        return positions[ids[positions] == wanted]

    def _membership(self, ids: np.ndarray, record_ids) -> np.ndarray:
        mask = np.zeros(len(ids), dtype=bool)
        mask[self._positions_of(ids, record_ids)] = True
        return mask

    def set_deleted(self, record_ids: set[str], deleted: bool) -> None:
        if not record_ids:
            return
        for array in self.spaces.values():
            array.deleted[self._positions_of(array.ids, record_ids)] = deleted
        self.invalidate()

    def patch_keys(self, space: str, positions: np.ndarray, keys: np.ndarray) -> None:
        """Replace unit keys of existing records in place, device mirror included.

        Only keys change: identities, scopes and deletion flags stay, so search
        eligibility caches remain valid and no full mirror rebuild is needed.
        """
        if not len(positions):
            return
        array = self.spaces[space]
        positions = np.asarray(positions, dtype=np.int64)
        keys = np.asarray(keys, dtype=np.float32)
        if keys.shape != (len(positions), array.keys.shape[1]):
            raise ValueError('Key patch shape differs from the index')
        array.keys[positions] = keys
        mirror = self._mirrors.get(space)
        if mirror is not None:
            mirror['keys'][torch.from_numpy(positions).to(self._device)] = (
                torch.from_numpy(keys).to(self._device))

    tie_margin = 64

    def use_device(self, device: str | torch.device | None) -> None:
        """Serve exact searches from device-resident mirrors (None: NumPy scan).

        Same scores, eligibility and tie order (score, then record ID) as the
        NumPy scan, which remains the CPU path and the test oracle.
        """
        self._device = None if device is None else torch.device(device)
        self.invalidate()

    def invalidate(self) -> None:
        """Mark device mirrors stale after any change to the resident arrays."""
        self._mirrors = {}
        self._eligible = {}

    def eligible_mask(self, space: str, *, domain: str, query_time: int) -> np.ndarray:
        """Boolean mask, in record order, of records this query may read."""
        cache = self.__dict__.setdefault('_eligible', {})
        key = (space, domain, int(query_time))
        mask = cache.get(key)
        array = self.spaces[space]
        if mask is None or len(mask) != len(array.ids):
            mask = (array.domains == domain) & (array.times < query_time) & ~array.deleted
            active = self._active_mask(space)
            if active is not None:
                mask &= active
            cache[key] = mask
            # Dated corpora have many query times; keep the most recent masks only.
            while len(cache) > ELIGIBLE_CACHE_ENTRIES:
                cache.pop(next(iter(cache)))
        else:
            cache[key] = cache.pop(key)
        return mask

    def restrict(self, record_ids: Iterable[str] | None) -> None:
        """Limit search and proposals to an active training subset (None: all).

        A curriculum restriction, not an authorization boundary: eligibility by
        domain, time and deletion still applies inside the subset.
        """
        self._active = None if record_ids is None else frozenset(map(str, record_ids))
        self._active_masks = {}
        self.invalidate()

    def _active_mask(self, space: str) -> np.ndarray | None:
        active = getattr(self, '_active', None)
        if active is None:
            return None
        array = self.spaces[space]
        masks = self.__dict__.setdefault('_active_masks', {})
        mask = masks.get(space)
        if mask is None or len(mask) != len(array.ids):
            mask = masks[space] = self._membership(array.ids, active)
        return mask

    def _mirror(self, space: str) -> dict:
        lock = self.__dict__.setdefault('_mirror_lock', threading.Lock())
        with lock:
            mirror = self._mirrors.get(space)
            if mirror is None:
                array = self.spaces[space]
                names, codes = np.unique(array.domains.astype(str), return_inverse=True)
                mirror = {
                    'keys': torch.from_numpy(array.keys).to(self._device, torch.float32),
                    'codes': torch.from_numpy(codes.astype(np.int64)).to(self._device),
                    'names': {name: code for code, name in enumerate(names.tolist())},
                    'times': torch.from_numpy(array.times).to(self._device),
                    'deleted': torch.from_numpy(array.deleted.astype(bool)
                                                | (False if self._active_mask(space) is None
                                                   else ~self._active_mask(space))
                                                ).to(self._device),
                    'position': None,
                }
                self._mirrors[space] = mirror
            return mirror

    def _search_device(self, space: str, q: np.ndarray, top_k: int, domains, query_times,
                       excluded) -> list[list[tuple[str, float]]]:
        array, mirror = self.spaces[space], self._mirror(space)
        queries = torch.from_numpy(q).to(self._device)
        scores = queries @ mirror['keys'].T
        codes = torch.tensor([mirror['names'].get(domain, -1) for domain in domains],
                             device=self._device)
        times = torch.tensor(list(query_times), device=self._device)
        eligible = ((mirror['codes'][None] == codes[:, None])
                    & (mirror['times'][None] < times[:, None]) & ~mirror['deleted'][None])
        for row, omitted in enumerate(excluded):
            if omitted:
                positions = self._positions_of(array.ids, omitted)
                eligible[row, torch.from_numpy(positions).to(self._device)] = False
        scores = scores.masked_fill(~eligible, float('-inf'))
        # Take a margin beyond k, then order exactly like the reference: score
        # descending, ties by record ID, so both paths return the same plans.
        k = min(top_k + self.tie_margin, scores.shape[1])
        values, indices = scores.topk(k, dim=1) if k else (scores[:, :0], scores[:, :0].long())
        values, indices = values.cpu().numpy(), indices.cpu().numpy()
        results = []
        for row in range(len(domains)):
            keep = np.isfinite(values[row])
            chosen, found = indices[row][keep], values[row][keep]
            order = np.lexsort((array.ids[chosen], -found))[:top_k]
            results.append([(str(array.ids[i]), float(v))
                            for i, v in zip(chosen[order], found[order], strict=True)])
        return results

    def keys_for_ids(self, space: str, record_ids: Sequence[str], *, domain: str,
                     query_time: int) -> Tensor:
        """Return fixed candidate keys while enforcing the index visibility snapshot."""
        if space not in self.spaces or not record_ids:
            raise ValueError('Published key lookup needs a known space and record IDs')
        array = self.spaces[space]
        positions = np.searchsorted(array.ids, np.asarray(record_ids))
        if any(position >= len(array.ids) or array.ids[position] != record_id
               for position, record_id in zip(positions, record_ids, strict=True)):
            raise KeyError('Record ID is absent from the published generation')
        if any(array.domains[position] != domain or array.times[position] >= query_time
               or array.deleted[position] for position in positions):
            raise PermissionError('Record key is outside its causal authorization scope')
        return torch.from_numpy(array.keys[positions].copy())

    def eligible_ids(self, space: str, record_ids: Sequence[str], *, domain: str,
                     query_time: int) -> tuple[str, ...]:
        """Filter proposals using the current bank authorization snapshot."""
        if space not in self.spaces:
            raise ValueError('Unknown key space')
        array = self.spaces[space]
        active = self._active_mask(space)
        result = []
        for record_id in record_ids:
            position = int(np.searchsorted(array.ids, record_id))
            if (position < len(array.ids) and array.ids[position] == record_id
                    and array.domains[position] == domain
                    and array.times[position] < query_time
                    and not array.deleted[position]
                    and (active is None or active[position])):
                result.append(record_id)
        return tuple(result)

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = 'corpus',
               space: str = 's0', generation: str = '', domain: str = 'research',
               query_time: int = 2**62, exclude_ids: frozenset[str] = frozenset()) -> ReadPlan:
        return self.search_batch(
            query[None], top_k=top_k, namespace=namespace, space=space,
            generation=generation, domains=(domain,), query_times=(query_time,),
            exclude_ids=(exclude_ids,),
        )[0]

    def search_batch(self, queries: Tensor, *, top_k: int = 16,
                     namespace: str = 'corpus', space: str = 's0', generation: str = '',
                     domains: Sequence[str], query_times: Sequence[int],
                     exclude_ids: Sequence[frozenset[str]] | None = None) -> tuple[ReadPlan, ...]:
        """Search several site queries with one dense key matrix operation."""
        if namespace != self.namespace or generation != self.generation or space not in self.spaces:
            raise ValueError('Key index scope differs from its published generation')
        if (queries.ndim != 2 or not torch.isfinite(queries).all() or top_k < 0
                or len(domains) != queries.shape[0] or len(query_times) != queries.shape[0]):
            raise ValueError('Invalid published batched key search request')
        excluded = tuple(exclude_ids or (frozenset(),) * queries.shape[0])
        if len(excluded) != queries.shape[0]:
            raise ValueError('One exclusion set is required per published query')
        array = self.spaces[space]
        q = queries.detach().float().cpu().numpy().copy()
        if q.shape[1] != array.keys.shape[1]:
            raise ValueError('Key dimensions incompatible with query generation')
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        if getattr(self, '_device', None) is not None:
            return tuple(
                ReadPlan(namespace, space, generation, domain, query_time,
                         tuple(Selection(record_id, score) for record_id, score in found),
                         self.bank_cursor)
                for domain, query_time, found in zip(
                    domains, query_times,
                    self._search_device(space, q, top_k, domains, query_times, excluded),
                    strict=True))
        scores = q @ array.keys.T
        plans = []
        for row, (domain, query_time, omitted) in enumerate(
                zip(domains, query_times, excluded, strict=True)):
            eligible = (array.domains == domain) & (array.times < query_time) & ~array.deleted
            if self._active_mask(space) is not None:
                eligible &= self._active_mask(space)
            if omitted:
                eligible &= ~self._membership(array.ids, omitted)
            indices = np.flatnonzero(eligible)
            order = np.lexsort((array.ids[indices], -scores[row, indices]))[:top_k]
            chosen = indices[order]
            plans.append(ReadPlan(
                namespace, space, generation, domain, query_time,
                tuple(Selection(str(array.ids[i]), float(scores[row, i])) for i in chosen),
                self.bank_cursor,
            ))
        return tuple(plans)
