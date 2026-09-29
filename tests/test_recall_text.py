"""Verbatim recall from overlapping windows (schnitz.recall_text); no downloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import random

from schnitz import recall_text as rt

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_memory_transcripts.py'
spec = importlib.util.spec_from_file_location('prepare_memory_transcripts_recall', SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

WORDS = ('river castle harbour meadow engine lantern violin orchard glacier copper saddle '
         'thunder marble falcon library compass ribbon canyon pepper kettle').split()


def document(i: int, sentences: int = 40) -> rt.Document:
    rng = random.Random(i)
    parts = []
    for s in range(sentences):
        words = [rng.choice(WORDS) for _ in range(rng.randint(6, 14))]
        parts.append(f'{words[0].capitalize()} {" ".join(words[1:])} number {i}-{s}.')
    text = ' '.join(parts[:5]) + '\n' + ' '.join(parts[5:])
    return rt.Document(f'doc{i}', f'Article {i}', text, 'test')


def build(redundancy=8, docs=12):
    return rt.build([document(i) for i in range(docs)], rt.regex_offsets, window=48,
                    stride=48 // redundancy, validation=0.25, min_tokens=100, max_tokens=5000)


def test_targets_start_at_sentences_with_space_carrying_tokens():
    import re

    def bpe_like(text):          # tokens carry the space before them, as LFM2's do
        return [m.span() for m in re.finditer(r'\s?\w+|\s?[^\w\s]', text)]
    _, eps, _ = rt.build([document(i) for i in range(6)], bpe_like, window=48, stride=6,
                         min_tokens=100, max_tokens=5000)
    for e in [x for rows in eps.values() for x in rows]:
        assert e['answer'][0].isupper(), e['answer'][:40]
        if e['provenance']['recall'] != 'title':
            assert e['query'].rstrip().endswith(('.', 'verbatim.')), e['query'][-60:]


def test_deterministic_and_split_by_document():
    recs, eps, summary = build()
    assert (recs, eps, summary) == build()
    train = {e['provenance']['document'] for e in eps['train']}
    validation = {e['provenance']['document'] for e in eps['validation']}
    assert validation and train and not train & validation
    # validation documents' windows are in the KB
    assert validation <= {r['provenance']['document'] for r in recs}


def test_every_target_token_has_redundancy_copies_and_groups_cover():
    for redundancy in (2, 8):
        recs, eps, summary = build(redundancy)
        assert summary['redundancy'] == redundancy
        by_id = {r['record_id']: r for r in recs}
        for e in [x for rows in eps.values() for x in rows]:
            assert 64 <= e['provenance']['target_tokens'] <= 256
            assert e['answer'] not in e['query']
            supports = {s['record_id'] for s in e['supports']}
            for group in e['sufficient_groups']:
                assert set(group) <= supports
                # a group's records together hold every target token
                start, end = e['provenance']['target_span']
                covered = set()
                for r in group:
                    covered.update(range(*by_id[r]['provenance']['tokens']))
                assert covered >= set(range(start, end))
            # each segment's alternatives hold it whole, at least `redundancy` of them
            # (head and tail windows keep the document's ends as redundant)
            for segment in e['alternatives']:
                assert set(segment) <= supports and len(segment) >= redundancy
            assert set().union(*map(set, e['alternatives'])) == supports


def test_transcript_slots_name_every_copy():
    recs, eps, _ = build()
    index = {r['record_id']: (1, r['kind'], rt.DOMAIN) for r in recs}
    b = mt.Builder('recall-text', index, mt.Options(), per_domain=False)
    e = next(x for x in eps['train'] if x['provenance']['recall'] == 'continuation')
    row, reason, _ = b.build(e, 'train')
    assert reason is None, reason
    slots = [m['content']['slot'] for m in row['messages']
             if isinstance(m.get('content'), dict) and 'slot' in m['content']]
    assert [s['record_ids'][0] for s in slots] == e['sufficient_groups'][0]
    named = {r for s in slots for r in s['alternatives']}
    assert named == {s['record_id'] for s in e['supports']}
    assert row['messages'][-1]['content'] == e['answer']
    assert not b.audit(row, [e['answer']], 2)
    # neutral: every other record of the document, never a slot's own positives
    document = {r['record_id'] for r in recs
                if r['provenance']['document'] == e['provenance']['document']}
    for s in slots:
        assert s['neutral'] and not set(s['neutral']) & {*s['record_ids'], *s['alternatives']}
        assert set(s['neutral']) | set(s['alternatives']) == document
    assert set(e['neutral']) <= set(slots[0]['neutral'])


def test_neutral_is_the_rest_of_the_document():
    recs, eps, summary = build()
    by_doc = {}
    for r in recs:
        by_doc.setdefault(r['provenance']['document'], set()).add(r['record_id'])
    for e in [x for rows in eps.values() for x in rows]:
        supports = {s['record_id'] for s in e['supports']}
        assert set(e['neutral']) == by_doc[e['provenance']['document']] - supports
        assert len(e['neutral']) == len(set(e['neutral']))
    assert summary['neutral_per_episode_median'] > 0
