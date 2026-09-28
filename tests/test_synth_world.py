"""The synthetic people world (schnitz.synth_world) and its transcripts; no downloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from schnitz import synth_world as sw

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_memory_transcripts.py'
spec = importlib.util.spec_from_file_location('prepare_memory_transcripts_synth', SCRIPT)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def small(seed=3, redundancy=4, hops=2):
    return sw.build(seed, 300, redundancy, hops=hops, validation=0.2, two_hop_rate=0.5)


def test_deterministic_by_seed_and_names_unique():
    _, recs, eps, summary = small()
    _, recs2, eps2, summary2 = small()
    assert recs == recs2 and eps == eps2 and summary == summary2
    assert len({r['record_id'] for r in recs}) == len(recs)       # copies are distinct
    _, other, _, _ = small(seed=4)
    assert {r['record_id'] for r in recs} != {r['record_id'] for r in other}
    world = sw.make_world(5, 3000)
    assert len({p.name for p in world.people}) == 3000


def test_every_answer_is_stated_in_at_least_redundancy_records():
    for redundancy in (1, 4, 9):
        world, recs, eps, summary = small(redundancy=redundancy)
        by_id = {r['record_id']: r for r in recs}
        assert set(summary['asked_fact_copies']['copies_per_fact']) == {redundancy}
        for rows in eps.values():
            for e in rows:
                assert len(e['alternatives']) == e['provenance']['hops']
                last = e['alternatives'][-1]
                assert len(last) >= redundancy
                # the answer is literally in every copy of the final hop
                assert all(e['answer'] in by_id[r]['text'] for r in last)
                # every sufficient group has one record per hop, each a support
                supports = {s['record_id'] for s in e['supports']}
                for group in e['sufficient_groups']:
                    assert len(group) == len(e['alternatives'])
                    assert all(r in hop for r, hop in zip(group, e['alternatives']))
                    assert set(group) <= supports
                if e['provenance']['hops'] == 1:
                    assert len(e['sufficient_groups']) == len(last)


def test_validation_people_are_never_asked_in_train():
    world, recs, eps, _ = small()
    train = {e['provenance']['person'] for e in eps['train']}
    validation = {e['provenance']['person'] for e in eps['validation']}
    assert validation and not train & validation
    # their records are in the KB all the same
    stated = {int(f.split(':')[0]) for r in recs for f in r['provenance']['facts']
              if not f.startswith('c')}
    assert validation <= stated


def test_transcript_slot_names_every_copy():
    _, recs, eps, _ = small(redundancy=4)
    index = {r['record_id']: (1, r['kind'], sw.DOMAIN) for r in recs}
    b = mt.Builder('synth-people', index, mt.Options(), per_domain=False)
    for e in [eps['train'][0], next(e for e in eps['train'] if e['provenance']['hops'] == 2)]:
        row, reason, counts = b.build(e, 'train')
        assert reason is None, reason
        slots = [m['content']['slot'] for m in row['messages']
                 if isinstance(m.get('content'), dict) and 'slot' in m['content']]
        assert len(slots) == len(e['alternatives'])
        for slot, hop in zip(slots, e['alternatives']):
            assert len(slot['record_ids']) == 1 and slot['record_ids'][0] in hop
            assert sorted(slot['alternatives']) == sorted(hop)
        assert not b.audit(row, [e['answer']], 2)
        bad = {**row, 'messages': [dict(m) for m in row['messages']]}
        i = next(i for i, m in enumerate(bad['messages'])
                 if isinstance(m.get('content'), dict) and 'slot' in m['content'])
        slot = dict(bad['messages'][i]['content']['slot'], alternatives=['unknown'])
        bad['messages'][i] = {**bad['messages'][i], 'content': {'slot': slot}}
        got = b.audit(bad, [e['answer']], 2)
        assert got['record_not_in_kb'] == 1 and got['slot_not_in_alternatives'] == 1
