"""Promotion accounting for persistent compact views.

Contribution distillation is a training loss. Promotion additionally requires a
held-out behavioral control and measured serialized savings; this module keeps
those claims separate and machine-checkable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CompactionQuality:
    mean_kl: float
    argmax_disagreement: float
    byte_reduction: float
    record_reduction: float
    nll_increase: float | None
    failures: tuple[str, ...]

    @property
    def promotable(self) -> bool:
        return not self.failures


def assess_compaction(raw_logits: Tensor, compact_logits: Tensor, *,
                      raw_serialized_bytes: int, compact_serialized_bytes: int,
                      raw_records: int, compact_records: int,
                      maximum_kl: float, maximum_disagreement: float,
                      minimum_byte_reduction: float,
                      raw_nll: float | None = None,
                      compact_nll: float | None = None,
                      maximum_nll_increase: float | None = None) -> CompactionQuality:
    """Assess one held-out, information-matched raw/compact comparison."""
    if (raw_logits.shape != compact_logits.shape or raw_logits.ndim < 2
            or not torch.isfinite(raw_logits).all()
            or not torch.isfinite(compact_logits).all()):
        raise ValueError('Raw and compact logits must be finite and shape-compatible')
    if (raw_serialized_bytes < 1 or compact_serialized_bytes < 0
            or raw_records < 1 or compact_records < 0
            or maximum_kl < 0 or not 0 <= maximum_disagreement <= 1
            or not 0 <= minimum_byte_reduction <= 1):
        raise ValueError('Invalid compaction accounting or promotion thresholds')
    supplied_nll = raw_nll is not None or compact_nll is not None
    if supplied_nll and (raw_nll is None or compact_nll is None
                         or maximum_nll_increase is None):
        raise ValueError('NLL promotion control requires both losses and a threshold')
    raw_log = F.log_softmax(raw_logits.float(), dim=-1)
    compact_log = F.log_softmax(compact_logits.float(), dim=-1)
    mean_kl = float(F.kl_div(compact_log, raw_log.exp(), reduction='batchmean'))
    disagreement = float((raw_logits.argmax(-1) != compact_logits.argmax(-1)).float().mean())
    byte_reduction = 1.0 - compact_serialized_bytes / raw_serialized_bytes
    record_reduction = 1.0 - compact_records / raw_records
    nll_increase = None if not supplied_nll else float(compact_nll - raw_nll)
    failures = []
    if mean_kl > maximum_kl:
        failures.append('mean_kl')
    if disagreement > maximum_disagreement:
        failures.append('argmax_disagreement')
    if byte_reduction < minimum_byte_reduction:
        failures.append('byte_reduction')
    if nll_increase is not None and nll_increase > maximum_nll_increase:
        failures.append('nll_increase')
    return CompactionQuality(mean_kl, disagreement, byte_reduction,
                             record_reduction, nll_increase, tuple(failures))
