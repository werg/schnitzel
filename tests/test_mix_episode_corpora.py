import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from mix_episode_corpora import mix  # noqa: E402
from public_corpus_common import Writer, episode, source  # noqa: E402


def _jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _dataset(tmp_path, domain, count):
    writer = Writer(tmp_path / domain, domain)
    for index in range(count):
        gold = source(domain, f'Fact number {index} is blue.', title=f'Doc {index}')
        other = source(domain, f'Unrelated passage {index}.', title=f'Other {index}')
        for split in ('train', 'validation'):
            item = episode(domain=domain, split=split, identifier=f'{split}-{index}',
                           question=f'What colour is fact {index}?', answer='blue',
                           gold=[gold], supports=[other], filters=writer.filters_for(split))
            writer.add(split, item, [gold, other])
    writer.sources[source(domain, 'Never referenced.')['record_id']] = source(
        domain, 'Never referenced.')
    writer.close({'role': 'training'})
    return tmp_path / domain


def test_mix_keeps_base_sources_first_and_only_referenced_new_sources(tmp_path):
    base = [{'record_id': 'b1', 'text': 'x', 'domain': 'research', 'created_at': 1}]
    _jsonl(tmp_path / 'sources.jsonl', base)
    _jsonl(tmp_path / 'train.jsonl', [{'episode_id': 'old-1'}])
    _jsonl(tmp_path / 'validation.jsonl', [{'episode_id': 'old-2'}])
    first, second = _dataset(tmp_path, 'alpha', 5), _dataset(tmp_path, 'beta', 3)
    result = mix(base_train=tmp_path / 'train.jsonl', base_validation=tmp_path / 'validation.jsonl',
                 base_sources=tmp_path / 'sources.jsonl', datasets=[first, second],
                 output=tmp_path / 'out', train_cap=4, validation_cap=2, caps={'beta': 1},
                 seed=3, keep_unreferenced={'beta'})
    assert result['counts']['alpha']['train'] == 4 and result['counts']['beta']['train'] == 1
    rows = [json.loads(line) for line in (tmp_path / 'out/sources.jsonl').read_text().splitlines()]
    assert rows[0] == base[0] and len({row['record_id'] for row in rows}) == len(rows)
    # alpha keeps only referenced passages; beta keeps its whole haystack
    assert sum(row['domain'] == 'beta' for row in rows) == 7
    train = (tmp_path / 'out/train-episodes.jsonl').read_text().splitlines()
    assert len(train) == 1 + 4 + 1
    with pytest.raises(FileExistsError):
        mix(base_train=tmp_path / 'train.jsonl', base_validation=tmp_path / 'validation.jsonl',
            base_sources=tmp_path / 'sources.jsonl', datasets=[first], output=tmp_path / 'out',
            train_cap=1, validation_cap=1, caps={}, seed=3, keep_unreferenced=set())
