"""Evaluation-only payload interventions over independently verified stored banks."""
from .store import DiskStore, ReadPlan, lookup_record


class TargetedPayloadStore:
    """Swap only designated selected records, retaining base values everywhere else.

    Both sides must authorize the same immutable identity/provenance and precision.
    Target IDs are experimental intervention annotations, never a routing policy.
    Only materialized reads through previously captured plans are supported.
    """
    def __init__(self, base: DiskStore, alternative: DiskStore, target_ids: frozenset[str]):
        self.base, self.alternative, self.target_ids = base, alternative, target_ids

    def search(self, *_args, **_kwargs):
        raise ValueError('Targeted interventions require fixed captured read plans')

    def fetch(self, plan: ReadPlan):
        # Validate every original selection before admitting any replacement.
        values = self.base.fetch(plan)
        scope = dict(namespace=plan.namespace, space=plan.space, generation=plan.generation,
                     domain=plan.domain, query_time=plan.query_time)
        for i, selection in enumerate(plan.selections):
            if selection.record_id not in self.target_ids:
                continue
            original = lookup_record(self.base, selection.record_id, **scope)
            replacement = lookup_record(self.alternative, selection.record_id, **scope)
            if (original.created_at != replacement.created_at or original.source_id != replacement.source_id or
                    original.payload.dtype != replacement.payload.dtype or original.payload.shape != replacement.payload.shape):
                raise ValueError('Targeted replacement changed provenance or payload interface')
            values[i] = replacement.payload
        return values
