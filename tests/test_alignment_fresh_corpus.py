import json
from pathlib import Path
import runpy

import pytest

from schnitz.data import load_episodes, make_multiuse_world, save_episodes


def test_fresh_alignment_corpus_excludes_sources_and_both_endpoint_versions(tmp_path):
    helper = runpy.run_path(str(Path(__file__).parents[1]/'experiments/binding-text-alignment-20260920/prepare_fresh.py'))
    split = 'fresh-regression'
    excluded = make_multiuse_world(0, split=split, bindings=2)
    source = tmp_path/'excluded.jsonl'
    save_episodes(source, excluded)
    output = tmp_path/'fresh'
    helper['prepare'](output, [source], worlds=2, split=split)
    declaration = json.loads((output/'declaration.json').read_text())
    assert declaration['rejected_seeds'][0] == 0
    rows = load_episodes(output/'episodes.jsonl')
    assert len(rows) == 20
    blocked, selected = [helper['identities'](values) for values in (excluded, rows)]
    assert all(not a & b for a, b in zip(blocked, selected, strict=True))
    identifiers = [e for e in rows if e.task_family == 'multiuse/identifier']
    assert len(identifiers) == 4 and len({e.query for e in identifiers}) == 1
    assert all(all(s.created_at < e.query_time for s in e.supports) for e in rows)
    assert all(e.required_ids for e in rows)
    with pytest.raises(FileExistsError):
        helper['prepare'](output, [source], worlds=2, split=split)
