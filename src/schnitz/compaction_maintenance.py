"""Leased background rebuilding for invalidated compact records."""
from __future__ import annotations

from collections.abc import Callable, Iterable

from .store import StoredRecord
from .training_bank import TrainingBank


def rebuild_invalidated(bank: TrainingBank, worker_id: str,
                        build: Callable[[dict], Iterable[StoredRecord]], *,
                        limit: int = 1, lease_seconds: float = 300,
                        optimizer_step: int | None = None) -> dict[str, int]:
    """Claim ready dependency layers and publish rebuilt all-space revisions.

    ``build`` performs explicit offline compactor inference. It receives no source
    trajectories and must return stored views for exactly the claimed parent.
    Failed jobs are released for retry; successfully updated parents clear their
    leases in the same journal transaction.
    """
    jobs = bank.claim_rebuilds(worker_id, limit=limit, lease_seconds=lease_seconds)
    rebuilt = 0
    for job in jobs:
        try:
            records = tuple(build(job))
            if not records or {record.record_id for record in records} != {job['record_id']}:
                raise ValueError('A compaction rebuild must return only its claimed parent')
            bank.update(records, optimizer_step=optimizer_step,
                        children={job['record_id']: job['children']},
                        expected_child_cursors={
                            job['record_id']: job['child_cursors']},
                        rebuild_claims={job['record_id']: (
                            worker_id, job['invalidated_at'])})
            rebuilt += 1
        except Exception:
            bank.release_rebuilds(worker_id, (job['record_id'],))
            raise
    return {'claimed': len(jobs), 'rebuilt': rebuilt}
