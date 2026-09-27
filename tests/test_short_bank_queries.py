"""A passage prefix can locate the source without disclosing a later span."""
from dataclasses import asdict, replace
from pathlib import Path
import runpy

from schnitz.corpus_data import short_reconstruction
from schnitz.data import Source


def test_located_span_uses_prior_versioned_source_and_disjoint_clue():
    located_span = runpy.run_path(str(Path(__file__).parents[1] /
                                    'scripts/prepare_short_bank_queries.py'))['located_span']
    source = Source('source-version', 'Title: Example\nPassage: ' +
                    ' '.join(f'word{i}' for i in range(28)), 1, 'passage')
    full = replace(short_reconstruction(source, max_words=28),
                   provenance={'article_title': 'Example'})
    located = located_span(asdict(full))
    assert located.required_ids == full.required_ids
    assert located.supports == full.supports
    assert located.query_time == full.query_time
    assert located.query.startswith('Article: Example\nStored passage begins: word0 word1')
    assert located.provenance['span_start_word'] > 6
    assert located.answer not in located.query
    start = located.provenance['span_start_word'] - 1
    assert located.answer == ' '.join(f'word{i}' for i in range(start, start + 8))
