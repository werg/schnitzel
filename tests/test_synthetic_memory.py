"""Timed bAbI-style and RULER-style synthetic episodes on project-authored fixtures."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import prepare_synthetic_memory as synthetic
finally:
    sys.path.pop(0)

# Project-authored text in the bAbI file format (not upstream data).
STORIES = """1 Mary went to the kitchen.
2 John went to the hallway.
3 Where is Mary?\tkitchen\t1
4 Mary went to the garden.
5 Where is Mary?\tgarden\t4
1 Mary went to the kitchen.
2 Mary picked up the lamp there.
3 Mary went to the garden.
4 Where is the lamp?\tgarden\t2 3
"""


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _babi_archive(root: Path) -> Path:
    root.mkdir(parents=True)
    with tarfile.open(root / synthetic.BABI_ARCHIVE, 'w:gz') as handle:
        for split in ('train', 'test'):
            data = STORIES.encode()
            info = tarfile.TarInfo(f'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{split}.txt')
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    return root


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_parse_babi_stories():
    stories = synthetic.parse_babi(STORIES)
    assert len(stories) == 2
    assert stories[0][2] == {'line': 3, 'question': 'Where is Mary?', 'answer': 'kitchen',
                             'supports': [1]}
    assert stories[1][3]['supports'] == [2, 3]


def test_babi_time_hides_later_contradicting_fact(tmp_path):
    raw = _babi_archive(tmp_path / 'raw')
    summary = synthetic.build_facts(raw, tmp_path / 'out', tasks=(1,), train=10, validation=10,
                                    tokenizer=WordTokenizer())
    assert summary['episodes'] == {'validation': 3, 'train': 3}
    sources = {row['record_id']: row for row in _read(tmp_path / 'out/sources.jsonl')}
    # Identical sentences in different stories/splits are distinct timed passages.
    kitchen = [row for row in sources.values() if row['text'].endswith('Mary went to the kitchen.')]
    assert len(kitchen) == 4 and len({row['record_id'] for row in kitchen}) == 4
    assert all('provenance' in row and 'article_title' not in row['provenance']
               for row in sources.values())
    rows = _read(tmp_path / 'out/episodes-train.jsonl')
    early = next(row for row in rows if row['answer'] == 'kitchen')
    late = next(row for row in rows if row['query_time'] == 5)
    story = early['provenance']['story']
    assert early['query'].endswith(f'Story {story}: Where is Mary?')
    assert early['query_time'] == 3 and late['query_time'] == 5
    assert [s['created_at'] for s in early['supports']] == [1, 2]
    assert all(s['created_at'] < early['query_time'] for s in early['supports'])
    # The later "garden" fact exists in the bank but is not causally readable at time 3.
    garden = next(row for row in sources.values()
                  if row['text'] == f'Story {story}, line 4: Mary went to the garden.')
    assert garden['created_at'] == 4 and garden['record_id'] not in {
        s['record_id'] for s in early['supports']}
    assert late['required_ids'] == [garden['record_id']]
    assert early['provenance']['hidden_later_lines'] == 1
    lamp = next(row for row in rows if row['answer'] == 'garden' and row['query_time'] == 4)
    assert len(lamp['required_ids']) == 2 and lamp['sufficient_groups'] == [lamp['required_ids']]
    assert early['provenance']['story'] != lamp['provenance']['story']


def test_babilong_padding_remaps_gold_lines():
    story = synthetic.parse_babi(STORIES)[1]
    events, mapping = synthetic.pad_story(story, random.Random(3), ['Unrelated sentence one.',
                                                                    'Unrelated sentence two.'], 2)
    assert [item['position'] for item in events] == [1, 2, 3, 4, 5, 6]
    assert events[-1]['question'] == 'Where is the lamp?'
    assert sum(bool(item.get('distractor')) for item in events) == 2
    for line in (1, 2, 3):
        assert events[mapping[line] - 1]['line'] == line
    rows = synthetic.babi_story_episodes(
        story, task=1, split='train', story_id=12345, story_index=1, rng=random.Random(5),
        pool=['Unrelated sentence one.', 'Unrelated sentence two.'], max_distractors=2,
        distractor_fraction=1.0, max_prior=48, filters=synthetic.Filters())
    [(item, passages)] = rows
    assert item['query_time'] == len(passages) + 1
    gold_text = {s['text'].split(': ', 1)[1] for s in item['supports']
                 if s['record_id'] in item['required_ids']}
    assert gold_text == {'Mary picked up the lamp there.', 'Mary went to the garden.'}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ruler_deterministic_counterfactual_and_temporal_pairs(tmp_path):
    first = synthetic.build_ruler(tmp_path / 'a', train=80, validation=20, source_budget=4000,
                                  validation_source_budget=400, counterfactual_fraction=1.0)
    synthetic.build_ruler(tmp_path / 'b', train=80, validation=20, source_budget=4000,
                          validation_source_budget=400, counterfactual_fraction=1.0)
    assert first['episodes'] == {'validation': 20, 'train': 80}
    for name in ('sources.jsonl', 'episodes-train.jsonl', 'episodes-validation.jsonl'):
        assert _sha(tmp_path / 'a' / name) == _sha(tmp_path / 'b' / name)
    rows = _read(tmp_path / 'a/episodes-train.jsonl')
    families = {row['provenance']['family'] for row in rows}
    assert families == set(synthetic.RULER_FAMILIES)
    for row in rows:
        assert all(s['created_at'] < row['query_time'] for s in row['supports'])
        assert row['answer'].lower() not in row['query'].lower()

    pairs: dict[str, list[dict]] = {}
    for row in rows:
        if 'counterfactual_pair' in row['provenance']:
            pairs.setdefault(row['provenance']['counterfactual_pair'], []).append(row)
    assert pairs and all(len(pair) == 2 for pair in pairs.values())
    for original, twin in pairs.values():
        assert original['provenance']['counterfactual_role'] == 'original'
        assert twin['provenance']['counterfactual_role'] == 'counterfactual'
        changed = original['provenance']['counterfactual_changed']
        assert (original['answer'], twin['answer']) == (changed['original'],
                                                         changed['counterfactual'])
        strip = lambda row: [s['text'].split(': ', 1)[1] for s in row['supports']]  # noqa: E731
        differing = [i for i, (a, b) in enumerate(zip(strip(original), strip(twin))) if a != b]
        assert len(differing) == 1
        a, b = strip(original)[differing[0]], strip(twin)[differing[0]]
        assert a.replace(changed['original'], changed['counterfactual']) == b
        question = lambda row: row['query'].split('Registry ', 1)[1].split(': ', 1)[1]  # noqa: E731
        assert question(original) == question(twin)
        assert original['query_time'] == twin['query_time']

    temporal: dict[str, dict] = {}
    for row in rows:
        if 'temporal_pair' in row['provenance']:
            temporal.setdefault(row['provenance']['temporal_pair'], {})[
                row['provenance']['temporal_role']] = row
    assert temporal
    for pair in temporal.values():
        before, after = pair['before_update'], pair['after_update']
        assert before['query'] == after['query'] and before['answer'] != after['answer']
        assert before['query_time'] < after['query_time']
        assert after['required_ids'][0] not in {s['record_id'] for s in before['supports']}
