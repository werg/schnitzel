"""Public storage boundaries that a local or network backend must preserve.

The protocols deliberately exclude SQLite connections and source trajectories.
Captured plans carry causal scope and an optional mutable-bank cursor; a backend
must revalidate that scope when payloads are fetched.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor

from .store import ReadPlan, StoredRecord


@runtime_checkable
class StoredReadBackend(Protocol):
    """Stored-only payload API suitable for an RPC client implementation."""

    def fetch(self, plan: ReadPlan) -> list[Tensor]: ...

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[Tensor]]: ...

    def iter_fetch(self, plan: ReadPlan, chunk_size: int = 32) -> Iterator[tuple[Tensor, Tensor]]: ...

    def lookup(self, record_id: str, *, namespace: str, space: str,
               generation: str, domain: str = "research",
               query_time: int = 2**62) -> StoredRecord: ...


@runtime_checkable
class KeySearchBackend(StoredReadBackend, Protocol):
    """Read backend that also owns key search, as a remote service normally will."""

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = "default",
               space: str = "s0", generation: str = "v0", domain: str = "research",
               query_time: int = 2**62, key_chunk_size: int = 1024,
               exclude_ids: frozenset[str] = frozenset()) -> ReadPlan: ...


@runtime_checkable
class BatchKeyIndexBackend(Protocol):
    """Exact or ANN key service used by packed multi-site training waves."""

    namespace: str
    generation: str
    bank_cursor: int | None

    def search_batch(self, queries: Tensor, *, top_k: int, namespace: str,
                     space: str, generation: str, domains: Sequence[str],
                     query_times: Sequence[int],
                     exclude_ids: Sequence[frozenset[str]] | None = None
                     ) -> tuple[ReadPlan, ...]: ...

    def keys_for_ids(self, space: str, record_ids: Sequence[str], *,
                     domain: str, query_time: int) -> Tensor: ...

    def eligible_ids(self, space: str, record_ids: Sequence[str], *,
                     domain: str, query_time: int) -> tuple[str, ...]: ...


@runtime_checkable
class MutableRecoveryBackend(Protocol):
    """Checkpoint/GC control plane required of a network journal backend."""

    def mutable_bank_state(self) -> dict | None: ...

    def restore_mutable_bank(self, state: dict) -> None: ...

    def pin_mutable_bank(self, checkpoint: str, cursor: int) -> None: ...

    def unpin_mutable_bank(self, checkpoint: str) -> None: ...

    def gc_mutable_bank(self, before_cursor: int) -> dict[str, int]: ...

    def mutable_bank_gc_ceiling(self) -> int | None: ...
