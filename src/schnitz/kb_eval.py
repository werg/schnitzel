"""Evaluation utilities for the knowledge-base stack (docs/knowledge-base-stack.md, 5.2, WP4).

Pure Python and model-agnostic: the functions take plain mappings (records,
episodes, lineage, read logs, scores) and return plain data, so the trainer and
evaluators can apply them to any store or model.

1. **Superposition metrics.** Source composition of items through rewrite
   lineage; sources served per item, items per source, the effective number of
   items carrying mass in a read (``exp`` of the entropy of normalized gate masses,
   and the participation ratio), and retention of old knowledge after new items.
2. **KB-dependence tests** (invariant 9: the knowledge must be in the KB).
   Counterfactual edits (a fact changed consistently in the KB, with the answer
   the edited KB implies), domain removal and insertion, as data the evaluator
   applies; scoring says whether an output follows the KB or the decoder's prior.
3. **Content over shuffled controls**: ``content_nats = NLL(shuffled) - NLL(actual)``
   and captured fractions of the full-text gain, as ``scripts/train_kb_codecs.py``
   reports them.

Records follow ``scripts/prepare_task_corpora.py`` (``record_id``, ``text``,
``domain``, ``kind``, ``created_at``, ``provenance``); episodes carry ``episode_id``,
``query``, ``answer``, ``required_ids``, ``supports`` and optionally ``verify``.
Record and episode ids stay opaque: edits create new versions with new ids and
``counterfactual_of`` provenance, never rewrite a record in place.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import re

from schnitz.task_verifiers import check_episode

Weights = Mapping[str, float] | Iterable[str]


# -- 1. superposition metrics ------------------------------------------------------
def _weights(value: Weights) -> dict[str, float]:
    if isinstance(value, Mapping):
        out = {k: float(w) for k, w in value.items()}
    else:
        out = dict.fromkeys(value, 1.0)
    if any(w < 0 for w in out.values()):
        raise ValueError('Negative weight')
    return out


def source_composition(lineage: Mapping[str, Weights]) -> dict[str, dict[str, float]]:
    """Source shares of every item, resolved through rewrite lineage.

    ``lineage[item]`` names what an item was made from, as contributor -> weight (an
    iterable means equal weights). A contributor that is itself a key of ``lineage``
    is an item (e.g. an input of a rewrite by S_s) and contributes its own
    composition; any other contributor is a source (record id). Shares of an item
    sum to one; item and source ids must not collide. Cycles raise ``ValueError``."""
    done: dict[str, dict[str, float]] = {}
    active: set[str] = set()

    def resolve(item: str) -> dict[str, float]:
        if item in done:
            return done[item]
        if item in active:
            raise ValueError(f'Lineage cycle through {item!r}')
        active.add(item)
        weights = _weights(lineage[item])
        total = sum(weights.values())
        shares: dict[str, float] = {}
        for contributor, weight in weights.items():
            if total <= 0 or weight == 0:
                continue
            parts = resolve(contributor) if contributor in lineage else {contributor: 1.0}
            for source, share in parts.items():
                shares[source] = shares.get(source, 0.0) + weight / total * share
        active.discard(item)
        done[item] = shares
        return shares

    return {item: resolve(item) for item in lineage}


def effective_count(weights: Iterable[float]) -> dict[str, float]:
    """Effective number of carriers of nonnegative ``weights`` (gate masses of the
    items in one read, or source shares of one item): ``entropy`` = exp(H(p)) and
    ``participation`` = (sum w)^2 / sum w^2, both 1 for one carrier and n for n equal
    ones; ``top1`` is the largest share. All zero for an empty or massless read."""
    values = [float(w) for w in weights]
    if any(w < 0 for w in values):
        raise ValueError('Negative mass')
    total = sum(values)
    if total <= 0:
        return {'entropy': 0.0, 'participation': 0.0, 'top1': 0.0}
    probs = [w / total for w in values if w > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return {'entropy': math.exp(entropy), 'participation': 1.0 / sum(p * p for p in probs),
            'top1': max(probs)}


def distribution(values: Iterable[float]) -> dict[str, float]:
    """``n``, mean, min, median, 90th percentile and max (nearest rank)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {'n': 0}
    n = len(ordered)

    def rank(q: float) -> float:
        return ordered[min(n - 1, max(0, math.ceil(q * n) - 1))]

    return {'n': n, 'mean': round(sum(ordered) / n, 4), 'min': ordered[0], 'p50': rank(0.5),
            'p90': rank(0.9), 'max': ordered[-1]}


def superposition_report(composition: Mapping[str, Mapping[str, float]],
                         reads: Iterable[Mapping[str, float] | Sequence[float]] = (),
                         min_share: float = 0.0) -> dict:
    """Standing superposition metrics (5.2).

    ``composition``: item -> source shares (``source_composition``). A source counts
    for an item when its share exceeds ``min_share``. ``reads``: the gate masses of
    the items retrieved in each read (item -> mass, or a list of masses).
    ``effective_sources_per_item`` weighs shares (a source with a tiny share counts
    little); ``items_per_source`` counts over the items that hold the source."""
    per_item = {item: sum(1 for s in shares.values() if s > min_share)
                for item, shares in composition.items()}
    holders: Counter[str] = Counter()
    for shares in composition.values():
        holders.update(s for s, share in shares.items() if share > min_share)
    effective = [effective_count(read.values() if isinstance(read, Mapping) else read)
                 for read in reads]
    return {
        'items': len(composition), 'sources': len(holders),
        'sources_per_item': distribution(per_item.values()),
        'effective_sources_per_item': distribution(
            effective_count(shares.values())['entropy'] for shares in composition.values()),
        'items_per_source': distribution(holders.values()),
        'reads': len(effective),
        'effective_items_per_read': {
            'entropy': distribution(e['entropy'] for e in effective),
            'participation': distribution(e['participation'] for e in effective),
            'top1_share': distribution(e['top1'] for e in effective)},
    }


def retention(before: Mapping[str, float], after: Mapping[str, float],
              higher_is_better: bool = True) -> dict:
    """Old knowledge after new items are written in: per-probe scores before and
    after insertion, compared on the probes present in both. ``retained`` is
    mean(after)/mean(before) for scores where higher is better (accuracy); with
    binary scores ``forgotten``/``gained`` count probes that flipped."""
    shared = sorted(before.keys() & after.keys())
    if not shared:
        return {'n': 0}
    b = [float(before[k]) for k in shared]
    a = [float(after[k]) for k in shared]
    mean_b, mean_a = sum(b) / len(b), sum(a) / len(a)
    out = {'n': len(shared), 'before': round(mean_b, 4), 'after': round(mean_a, 4),
           'delta': round(mean_a - mean_b, 4)}
    if higher_is_better and mean_b > 0:
        out['retained'] = round(mean_a / mean_b, 4)
    if all(v in (0.0, 1.0) for v in b + a):
        sign = 1 if higher_is_better else -1
        out['forgotten'] = sum(1 for x, y in zip(b, a) if sign * (y - x) < 0)
        out['gained'] = sum(1 for x, y in zip(b, a) if sign * (y - x) > 0)
    return out


# -- 2. KB-dependence tests --------------------------------------------------------
_SYLLABLES = ('ka', 've', 'lor', 'mi', 'tan', 'qu', 'zel', 'dro', 'fin', 'sa', 'bru', 'nox',
              'pel', 'rhi', 'tov', 'wex', 'gar', 'lum', 'syr', 'dak')
# questions whose answer can depend on a string's spelling or order: renaming the
# answer value would change which value is the answer
_SPELLING = re.compile(r'\b(alphabet\w*|order\w*|sort\w*|ascending|descending|contain\w*|'
                       r'start\w*|begin\w*|end(?:s|ing)?\s+with|like|letters?|substring|'
                       r'characters?|length|longest|shortest)\b', re.IGNORECASE)
_NON_SPANS = {'yes', 'no', 'true', 'false', 'supported', 'not supported', 'refuted', 'none',
              'unknown', 'insufficient information'}


def _digest(*parts: str) -> str:
    return hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()[:32]


class _Nonces:
    """Pseudo-words that occur in no record of the KB (case-insensitive)."""

    def __init__(self, texts: Iterable[str], seed: int):
        self.vocabulary = {w.lower() for t in texts for w in re.findall(r'\w+', t)}
        self.seed = seed

    def word(self, key: str, syllables: int = 3) -> str:
        rng = random.Random(f'{self.seed}:{key}')
        for attempt in range(1000):  # longer words once short ones run out
            word = ''.join(rng.choice(_SYLLABLES) for _ in range(syllables + attempt // 50))
            if word not in self.vocabulary:
                self.vocabulary.add(word)
                return word
        raise RuntimeError('No free nonce word')

    def like(self, key: str, value: str) -> str:
        """A nonce with the same word count and capitalization style as ``value``;
        short lower-case words (of, the, and) are kept."""
        words = value.split(' ')
        out = []
        for i, original in enumerate(words):
            if len(original) <= 3 and original.islower():
                out.append(original)
                continue
            new = self.word(f'{key}:{i}', 2 + (len(original) > 6))
            if original.isupper():
                new = new.upper()
            elif original[:1].isupper():
                new = new.capitalize()
            out.append(new)
        return ' '.join(out)


@dataclass
class Edit:
    """A counterfactual edit for one episode.

    ``records``: original record id -> the edited record (a new version with a new
    id and ``provenance.counterfactual_of``); apply with ``apply_edit``. ``episode``:
    the probe over the edited KB (ids remapped, ``answer`` and ``verify`` giving the
    answer the edited KB implies). ``original_verify`` checks the pre-edit answer,
    i.e. what the decoder's prior would say."""
    edit_id: str
    episode_id: str
    rule: str
    original: str
    replacement: str
    records: dict[str, dict]
    episode: dict
    original_verify: dict


def expected_verify(episode: dict) -> dict:
    """The episode's ``verify`` spec; short-answer episodes without one (R6 QA) are
    checked as a value answer."""
    return episode.get('verify') or {'type': 'values', 'rows': [[episode['answer']]]}


def _edited_record(record: dict, text: str, edit_id: str) -> dict:
    new = copy.deepcopy(record)
    new['record_id'] = _digest(record['record_id'], edit_id)
    new['text'] = text
    new['provenance'] = {**record.get('provenance', {}),
                         'counterfactual_of': record['record_id'], 'edit_id': edit_id}
    return new


def _edited_episode(episode: dict, edit_id: str, changed: dict[str, dict], answer: str,
                    verify: dict) -> dict:
    new = copy.deepcopy(episode)
    remap = {old: rec['record_id'] for old, rec in changed.items()}
    new['episode_id'] = f'{episode["episode_id"]}~{edit_id}'
    new['required_ids'] = [remap.get(r, r) for r in episode.get('required_ids', [])]
    new['sufficient_groups'] = [[remap.get(r, r) for r in group]
                                for group in episode.get('sufficient_groups', [])]
    for support in new.get('supports', []):
        if support['record_id'] in changed:
            support['text'] = changed[support['record_id']]['text']
            support['record_id'] = remap[support['record_id']]
    new['answer'] = answer
    new['verify'] = verify
    new['provenance'] = {**episode.get('provenance', {}),
                         'counterfactual_of': episode['episode_id'], 'edit_id': edit_id}
    return new


def _substitute(records: Iterable[dict], pattern: re.Pattern, replacement: str,
                edit_id: str, needle: str) -> dict[str, dict]:
    changed = {}
    for record in records:
        if needle not in record['text']:
            continue
        text = pattern.sub(lambda _: replacement, record['text'])
        if text != record['text']:
            changed[record['record_id']] = _edited_record(record, text, edit_id)
    return changed


def _word(value: str) -> re.Pattern:
    return re.compile(rf'(?<![\w-]){re.escape(value)}(?![\w-])')


def _rename_tool(episode, verify, records, nonces, edit_id):
    calls = verify.get('gold') if verify['type'] == 'calls' else verify.get('calls')
    if not calls:
        return 'no_calls'
    name = calls[0]['name']
    if re.search(rf'(?<!\w){re.escape(name)}(?!\w)', episode['query']):
        return 'name_in_query'
    docs = [r for r in records if r.get('provenance', {}).get('tool') == name]
    if not docs:
        return 'no_tool_doc'
    new_name = f'{nonces.word(edit_id + ":a", 2)}_{nonces.word(edit_id + ":b", 2)}'
    changed = _substitute(docs, _word(name), new_name, edit_id, name)
    if not changed:
        return 'name_not_in_doc'
    renamed = [{**c, 'name': new_name} if c['name'] == name else c for c in calls]
    key = 'gold' if verify['type'] == 'calls' else 'calls'
    return name, new_name, changed, json.dumps(renamed), {**verify, key: renamed}


def _rename_parameter(episode, verify, records, nonces, edit_id):
    calls = verify.get('gold') if verify['type'] == 'calls' else verify.get('calls')
    if not calls:
        return 'no_calls'
    name = calls[0]['name']
    arguments = calls[0].get('arguments')
    if not isinstance(arguments, dict) or not arguments:
        return 'no_arguments'
    param = next(iter(arguments))
    if re.search(rf'(?<!\w){re.escape(param)}(?!\w)', episode['query']):
        return 'parameter_in_query'
    docs = [r for r in records if r.get('provenance', {}).get('tool') == name]
    new_param = nonces.word(edit_id, 3)
    changed = _substitute(docs, re.compile(re.escape(json.dumps(param))), json.dumps(new_param),
                          edit_id, json.dumps(param))
    if not changed:
        return 'parameter_not_in_doc'
    renamed = [{**c, 'arguments': {(new_param if k == param else k): v
                                   for k, v in c['arguments'].items()}}
               if c['name'] == name and isinstance(c.get('arguments'), dict) else c for c in calls]
    key = 'gold' if verify['type'] == 'calls' else 'calls'
    return param, new_param, changed, json.dumps(renamed), {**verify, key: renamed}


def _rename_value(episode, verify, records, nonces, edit_id):
    """Rename one whole string cell of a stored-table answer (``values`` verify):
    the ``repr`` of the cell is replaced in every record of the same database, so
    selections, joins, grouping and distinct counts are unchanged."""
    if verify['type'] != 'values':
        return 'not_values'
    if _SPELLING.search(episode['query']):
        return 'spelling_sensitive_question'
    cells = [v for row in verify['rows'] for v in row]
    candidates = [v for v in cells if isinstance(v, str) and len(v.strip()) >= 3
                  and not re.fullmatch(r'[\d\s.,:/-]+', v)
                  and v.strip().lower() not in episode['query'].lower()]
    if not candidates:
        return 'no_string_cell'
    value = candidates[0]
    db = episode.get('provenance', {}).get('db_id')
    scope = [r for r in records if r.get('provenance', {}).get('db') == db] if db else records
    lead = value[:len(value) - len(value.lstrip())]
    new_value = lead + nonces.like(edit_id, value.strip())
    changed = _substitute(scope, re.compile(re.escape(repr(value))), repr(new_value), edit_id,
                          repr(value))
    if not changed:
        return 'value_not_stored'
    rows = [[new_value if v == value else v for v in row] for row in verify['rows']]
    answer = episode.get('answer', '').replace(value.strip(), new_value.strip())
    return value, new_value, changed, answer, {'type': 'values', 'rows': rows}


def _rename_span(episode, verify, records, nonces, edit_id):
    """Rename a short answer span (R6 QA) wherever it occurs as a whole phrase in
    the domain's records. Only name-like spans (capitalized, at most four words)
    qualify, so the renamed phrase reads as another entity."""
    answer = str(episode.get('answer', '')).strip()
    if (not 3 <= len(answer) <= 60 or answer.lower() in _NON_SPANS
            or re.fullmatch(r'[\d\s.,:/%$-]+', answer)):
        return 'answer_not_a_span'
    if not answer[0].isupper() or len(answer.split()) > 4:
        return 'answer_not_a_name'
    if answer.lower() in episode['query'].lower():
        return 'answer_in_query'
    required = set(episode.get('required_ids', []))
    pattern = _word(answer)
    if not any(pattern.search(r['text']) for r in records if r['record_id'] in required):
        return 'answer_not_in_required'
    domain = episode.get('provenance', {}).get('domain')
    scope = [r for r in records if domain is None or r.get('domain') == domain]
    new_answer = nonces.like(edit_id, answer)
    changed = _substitute(scope, pattern, new_answer, edit_id, answer)
    return answer, new_answer, changed, new_answer, {'type': 'values', 'rows': [[new_answer]]}


RULES: dict[str, Callable] = {'tool_rename': _rename_tool, 'parameter_rename': _rename_parameter,
                              'value_rename': _rename_value, 'answer_span': _rename_span}


def counterfactual_edits(records: Mapping[str, dict], episodes: Iterable[dict], seed: int = 0,
                         rules: Sequence[str] = ('tool_rename', 'value_rename', 'answer_span'),
                         ) -> tuple[list[Edit], dict]:
    """One conservative, rule-based counterfactual edit per episode where one is
    derivable, and a coverage report (edited count and skip reasons per rule).

    Rules, tried in ``rules`` order:

    - ``tool_rename`` (``calls`` / ``tau_bench``): the first gold call's tool gets a
      new name in its documentation records; expected calls use the new name.
    - ``parameter_rename``: the first gold call's first argument key is renamed in
      the tool's documentation; expected calls use the new key.
    - ``value_rename`` (``values``, stored tables): a string answer cell is renamed
      in every record of the database; expected rows carry the new value. Skipped
      when the question could depend on spelling or order.
    - ``answer_span`` (short QA answers without ``verify``): the answer phrase is
      renamed wherever it occurs in the domain's records.

    SQL, code, puzzle and exact-answer tasks have no derivable edit and are counted
    as ``not_derivable``. Nonces never occur in the KB before the edit."""
    records = dict(records)
    nonces = _Nonces((r['text'] for r in records.values()), seed)
    pool = list(records.values())
    edits: list[Edit] = []
    skipped: Counter[str] = Counter()
    by_rule: Counter[str] = Counter()
    total = 0
    for episode in episodes:
        total += 1
        verify = expected_verify(episode)
        reasons = []
        for rule in rules:
            if rule in ('tool_rename', 'parameter_rename') and verify['type'] not in ('calls', 'tau_bench'):
                continue
            # stored-table answers carry a values spec; short QA answers carry none
            if rule == 'value_rename' and episode.get('verify', {}).get('type') != 'values':
                continue
            if rule == 'answer_span' and 'verify' in episode:
                continue
            edit_id = 'cf' + _digest(str(seed), rule, episode['episode_id'])[:12]
            result = RULES[rule](episode, verify, pool, nonces, edit_id)
            if isinstance(result, str):
                reasons.append(f'{rule}:{result}')
                continue
            original, replacement, changed, answer, new_verify = result
            edits.append(Edit(edit_id, episode['episode_id'], rule, original, replacement, changed,
                              _edited_episode(episode, edit_id, changed, answer, new_verify),
                              verify))
            by_rule[rule] += 1
            break
        else:
            skipped['; '.join(reasons) or 'not_derivable'] += 1
    return edits, {'episodes': total, 'edited': len(edits),
                   'coverage': round(len(edits) / max(total, 1), 4),
                   'by_rule': dict(by_rule), 'skipped': dict(skipped)}


def apply_edit(records: Mapping[str, dict], edit: Edit) -> dict[str, dict]:
    """The KB with ``edit``: each changed record superseded by its edited version."""
    out = {k: v for k, v in records.items() if k not in edit.records}
    out.update({rec['record_id']: rec for rec in edit.records.values()})
    return out


def follows(output: str, edit: Edit, check: Callable[[str, dict], bool] = check_episode) -> str:
    """``kb`` when the output gives the edited KB's answer, ``prior`` when it gives the
    original answer (the decoder's weights), ``both`` or ``neither``."""
    kb, prior = bool(check(output, edit.episode['verify'])), bool(check(output, edit.original_verify))
    return {(True, False): 'kb', (False, True): 'prior', (True, True): 'both'}.get((kb, prior),
                                                                                   'neither')


def dependence_report(labels: Iterable[str]) -> dict:
    """Fractions of ``follows`` labels; ``follows_kb`` is the edit-following rate."""
    counts = Counter(labels)
    n = sum(counts.values())
    return {'n': n, **{f'follows_{k}' if k in ('kb', 'prior') else k:
                       round(counts.get(k, 0) / max(n, 1), 4)
                       for k in ('kb', 'prior', 'both', 'neither')}}


@dataclass
class RemovalTest:
    """Records removed from the KB, episodes that need them (``probes``: the
    capability should go) and episodes that touch none of them (``controls``: it
    should stay)."""
    removed_ids: frozenset[str]
    probes: list[str]
    controls: list[str]


def _needed(episode: dict) -> set[str]:
    return set(episode.get('required_ids', [])) | {s['record_id'] for s in episode.get('supports', [])}


def domain_removal(records: Mapping[str, dict], episodes: Iterable[dict], domain: str | None = None,
                   kind: str | None = None) -> RemovalTest:
    """Remove every record of ``domain`` and/or ``kind``. Probes require a removed
    record; controls neither require nor are supported by one."""
    if domain is None and kind is None:
        raise ValueError('Give a domain or a kind')
    removed = frozenset(k for k, r in records.items()
                        if (domain is None or r.get('domain') == domain)
                        and (kind is None or r.get('kind') == kind))
    probes, controls = [], []
    for episode in episodes:
        if set(episode.get('required_ids', [])) & removed:
            probes.append(episode['episode_id'])
        elif not _needed(episode) & removed:
            controls.append(episode['episode_id'])
    return RemovalTest(removed, probes, controls)


def apply_removal(records: Mapping[str, dict], test: RemovalTest) -> dict[str, dict]:
    return {k: v for k, v in records.items() if k not in test.removed_ids}


def _mean(scores: Mapping[str, float], ids: Iterable[str]) -> float | None:
    values = [float(scores[i]) for i in ids if i in scores]
    return round(sum(values) / len(values), 4) if values else None


def removal_report(test: RemovalTest, before: Mapping[str, float], after: Mapping[str, float],
                   closed_book: Mapping[str, float] | None = None) -> dict:
    """Probe and control scores with the full KB (``before``), after removal and
    optionally without any KB. ``surviving_gain`` = (after - closed) / (before -
    closed) on probes: the share of the KB's gain left after removal (should be
    about 0; above 0 means the capability was not in the removed records)."""
    out = {'removed': len(test.removed_ids)}
    for name, ids in (('probes', test.probes), ('controls', test.controls)):
        out[name] = {'n': len(ids), 'before': _mean(before, ids), 'after': _mean(after, ids)}
        if closed_book is not None:
            out[name]['closed_book'] = _mean(closed_book, ids)
    probes = out['probes']
    if closed_book is not None and None not in (probes['before'], probes['after'],
                                                probes['closed_book']):
        gain = probes['before'] - probes['closed_book']
        out['surviving_gain'] = round((probes['after'] - probes['closed_book']) / gain, 4) \
            if gain > 0 else None
    return out


@dataclass
class InsertionTest:
    """``base_ids`` form the initial KB; ``new_ids`` are written in later.
    ``new_probes`` need a new record (and only base or new ones); ``old_probes``
    need only base records (retention)."""
    base_ids: frozenset[str]
    new_ids: frozenset[str]
    new_probes: list[str]
    old_probes: list[str]
    extra_records: dict[str, dict] = field(default_factory=dict)
    extra_episodes: list[dict] = field(default_factory=list)


def holdout_insertion(records: Mapping[str, dict], episodes: Iterable[dict], fraction: float,
                      seed: int = 0) -> InsertionTest:
    """Hold out ``fraction`` of the records some episode requires as new knowledge."""
    episodes = list(episodes)
    required = sorted({r for e in episodes for r in e.get('required_ids', []) if r in records})
    rng = random.Random(seed)
    new = frozenset(rng.sample(required, round(fraction * len(required))))
    base = frozenset(records) - new
    new_probes, old_probes = [], []
    for episode in episodes:
        need = set(episode.get('required_ids', []))
        if need and need & new and need <= base | new:
            new_probes.append(episode['episode_id'])
        elif need and need <= base:
            old_probes.append(episode['episode_id'])
    return InsertionTest(base, new, new_probes, old_probes)


_FACTS = (
    ('{e} is a town in the province of {v}.', 'In which province is {e}?'),
    ('The river {e} flows into the lake {v}.', 'Into which lake does the river {e} flow?'),
    ('{e} was founded by the engineer {v}.', 'Who founded {e}?'),
    ('The main export of {e} is {v}.', 'What is the main export of {e}?'),
    ('The {e} protocol uses port {n}.', 'Which port does the {e} protocol use?'),
    ('{e} has a population of {n} people.', 'What is the population of {e}?'),
)


def synthetic_facts(count: int, seed: int = 0, domain: str = 'synthetic',
                    avoid: Iterable[str] = ()) -> tuple[dict[str, dict], list[dict]]:
    """``count`` records with facts about nonce entities and one probe question per
    record, answerable only from the KB (no decoder can know them beforehand).
    ``avoid``: texts whose words nonces must not collide with (the existing KB)."""
    nonces = _Nonces(avoid, seed)
    rng = random.Random(seed)
    records, episodes = {}, []
    for i in range(count):
        fact, question = _FACTS[i % len(_FACTS)]
        entity = nonces.word(f'fact{i}:e').capitalize()
        value = str(rng.randrange(1000, 99999)) if '{n}' in fact else \
            nonces.word(f'fact{i}:v').capitalize()
        text = fact.format(e=entity, v=value, n=value)
        rid = _digest(domain, text)
        records[rid] = {'record_id': rid, 'text': text, 'domain': domain, 'created_at': 1,
                        'kind': 'synthetic_fact', 'provenance': {'dataset': domain, 'seed': seed}}
        episodes.append({
            'episode_id': f'{domain}-{seed}-{i}', 'environment': f'{domain}-probe',
            'query': 'Use the stored notes. Give only the short answer.\nQuestion: '
                     + question.format(e=entity),
            'answer': value, 'query_time': 2, 'required_ids': [rid], 'sufficient_groups': [[rid]],
            'support_annotation': 'verified', 'task_family': 'synthetic_fact_qa',
            'supports': [{'record_id': rid, 'text': text, 'created_at': 1, 'kind': 'synthetic_fact'}],
            'verify': {'type': 'values', 'rows': [[value]]},
            'provenance': {'dataset': domain, 'domain': domain, 'split': 'probe'}})
    return records, episodes


def synthetic_insertion(records: Mapping[str, dict], episodes: Iterable[dict], count: int,
                        seed: int = 0) -> InsertionTest:
    """Insert ``count`` synthetic fact records into the whole KB; every existing
    episode whose required records are in the KB is a retention probe."""
    new_records, new_episodes = synthetic_facts(count, seed,
                                                avoid=(r['text'] for r in records.values()))
    base = frozenset(records)
    old = [e['episode_id'] for e in episodes
           if e.get('required_ids') and set(e['required_ids']) <= base]
    return InsertionTest(base, frozenset(new_records), [e['episode_id'] for e in new_episodes],
                         old, new_records, new_episodes)


def insertion_report(test: InsertionTest, old_before: Mapping[str, float],
                     old_after: Mapping[str, float], new_after: Mapping[str, float],
                     new_before: Mapping[str, float] | None = None) -> dict:
    """Retention of old knowledge and acquisition of new knowledge after insertion;
    ``new_before`` (the probes before insertion) is the no-knowledge control."""
    out = {'inserted': len(test.new_ids),
           'retention': retention({k: old_before[k] for k in test.old_probes if k in old_before},
                                  old_after),
           'new': {'n': len(test.new_probes), 'after': _mean(new_after, test.new_probes)}}
    if new_before is not None:
        out['new']['before'] = _mean(new_before, test.new_probes)
    return out


# -- 3. content over shuffled controls -----------------------------------------------
def content_nats(nll_actual: float, nll_shuffled: float) -> float:
    """Nats per token the actual content buys over a shuffled control (another
    example's span or items at the same length): NLL(shuffled) - NLL(actual)."""
    return nll_shuffled - nll_actual


def captured_fraction(nll_arm: float, nll_noctx: float, nll_full: float) -> float:
    """Fraction of the full-text gain an arm captures: (NLL(no context) - NLL(arm)) /
    (NLL(no context) - NLL(full text)), the gain floored at 1e-9 as in K1."""
    return (nll_noctx - nll_arm) / max(nll_noctx - nll_full, 1e-9)


def nll_summary(sums: Mapping[str, float], tokens: int, arms: Iterable[str],
                shuffled: Mapping[str, str]) -> dict:
    """``train_kb_codecs.evaluate``'s report from summed token NLLs per arm: mean
    NLL per arm, ``captured`` per arm (needs ``noctx`` and ``full``), and
    ``content_nats`` per ``shuffled`` pair (arm -> its shuffled control). ``gain`` is
    the full-text gain the captured fractions divide by; a nonpositive gain makes
    them meaningless."""
    nll = {name: value / tokens for name, value in sums.items()}
    return {'nll': {k: round(v, 4) for k, v in nll.items()},
            'gain': round(nll['noctx'] - nll['full'], 4),
            'captured': {k: round(captured_fraction(nll[k], nll['noctx'], nll['full']), 4)
                         for k in arms},
            'content_nats': {k: round(content_nats(nll[k], nll[s]), 4) for k, s in shuffled.items()}}
