"""Bank creation's record-to-span half (``schnitz.kb.bank``); CPU, a fake writer."""
from __future__ import annotations

import json
import math

import pytest
import torch

from schnitz.kb import bank
from schnitz.kb.decoder import length_factors


class FakeWriter:
    """``text_ids`` and ``free_run`` as ``schnitz.kb.decoder.Model`` has them: a span's
    reps are a deterministic function of the text and the rep index."""

    def __init__(self):
        self.calls = 0

    def text_ids(self, text: str) -> torch.Tensor:
        return torch.tensor([ord(c) for c in text])

    def free_run(self, examples, lengths):
        self.calls += 1
        out = []
        for ex, n in zip(examples, lengths):
            assert ex['prompt'] == 'memory'
            base = float(ex['ids'].sum())
            out.append(torch.arange(n)[:, None].float() + base + torch.zeros(n, 4))
        return out, [None] * len(out)


def test_write_spans_follow_the_length_schedule_in_input_order():
    texts = ['a' * 400, 'bb', 'c' * 90]
    spans = bank.write_spans(FakeWriter(), texts, 's1', batch_size=2)
    for text, span in zip(texts, spans):
        n = len(text)
        assert span.shape == (max(1, math.ceil(n / length_factors(n)[1])), 4)
        assert float(span[0, 0]) == sum(map(ord, text))


def test_span_cache_round_trip_resume_and_mismatch(tmp_path):
    writer = FakeWriter()
    records = {f'r{i}': 'x' * (5 + 7 * i) for i in range(7)}
    first = dict(list(records.items())[:4])
    bank.SpanCache.build(tmp_path / 'c', writer, first, 's0', batch_size=3, shard_records=3,
                         meta={'writer': 'w1'})
    calls = writer.calls
    cache = bank.SpanCache.build(tmp_path / 'c', writer, records, 's0', batch_size=3,
                                 shard_records=3, meta={'writer': 'w1'})
    assert len(cache) == 7 and all(r in cache for r in records)
    assert writer.calls - calls == 1               # only the three new records were written
    want = bank.write_spans(FakeWriter(), list(records.values()), 's0')
    reopened = bank.SpanCache(tmp_path / 'c')
    for r, span in zip(records, want):
        torch.testing.assert_close(reopened.get(r).float(), span.to(torch.bfloat16).float())
    with pytest.raises(ValueError):               # another writer or level: refused
        bank.SpanCache.build(tmp_path / 'c', writer, records, 's0', meta={'writer': 'w2'})
    with pytest.raises(ValueError):
        bank.SpanCache.build(tmp_path / 'c', writer, records, 's1', meta={'writer': 'w1'})


def test_record_sources_collects_slot_records_per_kb(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    sources = [{'record_id': f'r{i}', 'text': f'text {i}', 'created_at': i, 'kind': 'passage',
                'domain': 'd1' if i < 4 else 'd2'} for i in range(8)]
    (corpus / 'sources.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sources))
    transcripts = tmp_path / 'memory'
    transcripts.mkdir()
    (transcripts / 'manifest.json').write_text(json.dumps(
        {'input': str(corpus), 'kb': 'c', 'kb_per_domain': True}))

    def row(i, kb, ids):
        return {'episode_id': f'e{i}', 'kb': kb, 'messages': [
            {'role': 'tool', 'content': {'slot': {'kb': kb, 'record_ids': ids}}}]}

    rows = [row(0, 'c:d1', ['r1', 'r2']), row(1, 'c:d2', ['r5']), row(2, 'c:d1', ['r3'])]
    (transcripts / 'transcripts-train.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in rows))
    records = bank.record_sources([transcripts], {'train': 2})
    assert set(records) == {'r1', 'r2', 'r5'}
    assert records['r5'] == {'text': 'text 5', 'kb': 'c:d2', 'created_at': 5}
    with_extra = bank.record_sources([transcripts], {'train': None}, distractors=1)
    assert set(with_extra) == {'r1', 'r2', 'r3', 'r5', 'r0', 'r4'}   # one more per KB
    bad = rows + [row(3, 'c:d1', ['missing'])]
    (transcripts / 'transcripts-train.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in bad))
    with pytest.raises(ValueError):
        bank.record_sources([transcripts], {'train': None})


def test_record_sources_banks_every_alternative_copy(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    sources = [{'record_id': f'r{i}', 'text': f'text {i}', 'created_at': 1, 'kind': 'passage',
                'domain': 'd'} for i in range(6)]
    (corpus / 'sources.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sources))
    transcripts = tmp_path / 'memory'
    transcripts.mkdir()
    (transcripts / 'manifest.json').write_text(json.dumps(
        {'input': str(corpus), 'kb': 'c', 'kb_per_domain': False}))
    row = {'episode_id': 'e0', 'kb': 'c', 'messages': [{'role': 'tool', 'content': {'slot': {
        'kb': 'c', 'record_ids': ['r1'], 'alternatives': ['r1', 'r3', 'r4']}}}]}
    (transcripts / 'transcripts-train.jsonl').write_text(json.dumps(row) + '\n')
    assert set(bank.record_sources([transcripts], {'train': None})) == {'r1', 'r3', 'r4'}
    assert bank.needed_records([{**row, '_dir': 'x'}]) == {
        'c': dict.fromkeys(['r1', 'r3', 'r4'], 'x')}
