"""Indexed span labels stay inside the stored source and outside the query."""
from dataclasses import asdict
from pathlib import Path
import runpy

from schnitz.corpus_data import short_reconstruction
from schnitz.data import Source


def test_indexed_span_preserves_versioned_support_and_word_boundaries():
    indexed_span = runpy.run_path(str(Path(__file__).parents[1] /
                                    'scripts/prepare_short_span_curriculum.py'))['indexed_span']
    source = Source('versioned-source', 'Title: Example\nPassage: ' +
                    ' '.join(f'word{i},' for i in range(28)), 1, 'passage')
    full = short_reconstruction(source, max_words=28)
    span = indexed_span(asdict(full), span_words=8)
    assert span.required_ids == full.required_ids
    assert span.supports == full.supports
    assert span.query_time == full.query_time
    assert span.task_family == 'passage_span'
    assert span.answer not in span.query
    start = span.provenance['span_start_word'] - 1
    assert span.answer == ' '.join(f'word{i},' for i in range(start, start + 8))
