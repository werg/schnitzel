import pytest
import torch

from schnitz.compaction_quality import assess_compaction


def test_compaction_quality_requires_behavior_and_net_savings():
    raw = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    close = raw + torch.tensor([[0.01, -0.01], [0.0, 0.0]])
    report = assess_compaction(
        raw, close, raw_serialized_bytes=1000, compact_serialized_bytes=300,
        raw_records=8, compact_records=2, maximum_kl=0.01,
        maximum_disagreement=0.0, minimum_byte_reduction=0.5,
    )
    assert report.promotable
    assert report.byte_reduction == pytest.approx(0.7)
    assert report.record_reduction == pytest.approx(0.75)

    no_savings = assess_compaction(
        raw, close, raw_serialized_bytes=1000, compact_serialized_bytes=900,
        raw_records=8, compact_records=2, maximum_kl=0.01,
        maximum_disagreement=0.0, minimum_byte_reduction=0.5,
    )
    assert not no_savings.promotable
    assert 'byte_reduction' in no_savings.failures


def test_compaction_quality_rejects_behavior_change_even_when_small_on_disk():
    raw = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    changed = raw.flip(-1)
    report = assess_compaction(
        raw, changed, raw_serialized_bytes=1000, compact_serialized_bytes=10,
        raw_records=8, compact_records=1, maximum_kl=0.01,
        maximum_disagreement=0.0, minimum_byte_reduction=0.5,
    )
    assert not report.promotable
    assert {'mean_kl', 'argmax_disagreement'} <= set(report.failures)
