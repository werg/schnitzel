"""Timed synthetic memory episodes: bAbI stories (+ BABILong-style distractors) and a
RULER-style generator with counterfactual pairs.

Time matters in both domains. Every story line is its own stored passage whose
``created_at`` is the line position (1, 2, 3, ...); a question's ``query_time`` is
the position of the question, so only earlier lines are causally readable and a
later contradicting line is hidden. Supports list every in-story passage created
before the question; gold is the annotated supporting lines, all required.

Identical sentences recur across bAbI stories, and one passage identity cannot have
two creation times, so every passage carries its story and line
(``Story 48213, line 3: Mary went to the kitchen.``) and the query names the story
(``Story 48213: Where is Mary?``). Synthetic sources get no article title, so the
writer never groups several lines of one story into one holistic write (that would
let an earlier passage's payload see a later line).

``synthetic_facts`` (bAbI en-10k, CC BY 3.0): tasks listed in ``BABI_TASKS``.
Validation comes from the upstream test files. With ``--distractor-text`` (the
QASPER archive), up to two unrelated abstract sentences are inserted into a story
as extra timed lines (BABILong-style), which renumbers later lines; gold line
references are remapped.

``synthetic_ruler`` (generated from a seed): random opaque keys, values and variable
names in registries of timed entries.

* ``kv_single`` -- one needle key queried at a time; queries at random times see
  only earlier entries.
* ``kv_multi`` -- two or three keys in one query; all needles required.
* ``kv_update`` -- a key is reassigned later; the same query text asked before and
  after the update has different answers (``temporal_pair``).
* ``vt_chain`` -- variable tracking: ``VAR b = VAR a`` chains of 2..4 hops plus a
  distractor chain; every hop is required.
* counterfactual pairs -- for a fraction of ``kv_single``/``vt_chain`` stories, a
  twin story identical except for one needle value (and its opaque registry id, so
  both can share a bank); the answer changes accordingly. Both episodes carry
  ``provenance.counterfactual_pair``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import tarfile

from public_corpus_common import (
    Filters, Writer, clean, episode, load_tokenizer, source, split_sentences)

FACTS = 'synthetic_facts'
RULER = 'synthetic_ruler'
BABI_ARCHIVE = 'tasks_1-20_v1-2.tar.gz'
# Counting (7), negation-only yes/no (9), spatial/size yes/no (17, 18), path finding
# (19) and motivations (20) are skipped: they test reasoning more than memory, or
# have near-trivial answers without useful supporting-line structure.
BABI_TASKS = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
YES_NO = {'yes', 'no', 'maybe'}


# bAbI -----------------------------------------------------------------------------

def parse_babi(text: str) -> list[list[dict]]:
    """Stories of ``{'line', 'text'}`` statements and ``{'line', 'question', ...}``."""
    stories, current = [], []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        number, _, rest = raw.partition(' ')
        line = int(number)
        if line == 1 and current:
            stories.append(current)
            current = []
        if '\t' in rest:
            question, answer, *support = rest.split('\t')
            current.append({'line': line, 'question': clean(question), 'answer': clean(answer),
                            'supports': [int(value) for value in ' '.join(support).split()]})
        else:
            current.append({'line': line, 'text': clean(rest)})
    if current:
        stories.append(current)
    return stories


def load_babi(raw: Path, task: int, split: str) -> list[list[dict]]:
    upstream = 'train' if split == 'train' else 'test'
    with tarfile.open(raw / BABI_ARCHIVE) as handle:
        pattern = re.compile(rf'tasks_1-20_v1-2/en-10k/qa{task}_[a-z-]+_{upstream}\.txt$')
        names = [name for name in handle.getnames() if pattern.match(name)]
        if len(names) != 1:
            raise FileNotFoundError(f'bAbI task {task} {upstream}: {names}')
        return parse_babi(handle.extractfile(names[0]).read().decode('utf-8'))


def distractor_pool(qasper_archive: Path, *, limit: int = 20000) -> list[str]:
    """Unrelated sentences (QASPER train abstracts) for BABILong-style padding."""
    with tarfile.open(qasper_archive) as handle:
        papers = json.loads(handle.extractfile('qasper-train-v0.3.json').read())
    pool = []
    for key in sorted(papers):
        for sentence in split_sentences(papers[key].get('abstract') or ''):
            if 40 <= len(sentence) <= 200 and len(pool) < limit:
                pool.append(sentence)
    return pool


def pad_story(story: list[dict], rng: random.Random, pool: list[str],
              count: int) -> tuple[list[dict], dict[int, int]]:
    """Insert ``count`` distractor statements before the last question.

    Returns the renumbered timeline and a map from upstream line to new position.
    """
    events = [dict(item) for item in story]
    last_question = max(i for i, item in enumerate(events) if 'question' in item)
    for sentence in rng.sample(pool, count) if pool and count else []:
        events.insert(rng.randint(0, last_question), {'line': None, 'text': sentence,
                                                     'distractor': True})
        last_question += 1
    mapping = {}
    for position, item in enumerate(events, start=1):
        if item['line'] is not None:
            mapping[item['line']] = position
        item['position'] = position
    return events, mapping


def babi_story_episodes(story: list[dict], *, task: int, split: str, story_id: int,
                        story_index: int, rng: random.Random, pool: list[str],
                        max_distractors: int, distractor_fraction: float, max_prior: int,
                        filters: Filters, tokenizer=None) -> list[tuple[dict, list[dict]]]:
    count = rng.randint(1, max_distractors) if (
        pool and max_distractors and rng.random() < distractor_fraction) else 0
    events, mapping = pad_story(story, rng, pool, count)
    passages: dict[int, dict] = {}
    for item in events:
        if 'question' in item:
            continue
        passages[item['position']] = source(
            FACTS, f'Story {story_id}, line {item["position"]}: {item["text"]}',
            created_at=item['position'],
            provenance={'task': f'qa{task}', 'story': story_id, 'line': item['position'],
                        'role': 'distractor' if item.get('distractor') else 'fact',
                        'upstream_line': item['line'], 'upstream_story_index': story_index,
                        'upstream_split': 'train' if split == 'train' else 'test'})
    result = []
    for item in events:
        if 'question' not in item:
            continue
        query_time = item['position']
        prior = [row for time, row in sorted(passages.items()) if time < query_time]
        if len(prior) > max_prior:
            filters.reject('too_many_prior_lines')
            continue
        gold = [passages[mapping[line]] for line in item['supports']]
        answer = item['answer'].replace(',', ', ')
        built = episode(
            domain=FACTS, split=split, identifier=f'qa{task}-{split}-{story_index}-{item["line"]}',
            question=f'Story {story_id}: {item["question"]}', answer=answer, gold=gold,
            supports=prior, filters=filters, tokenizer=tokenizer, query_time=query_time,
            all_required=True, annotation='verified', task_family='synthetic_timed_facts',
            allow_answer_in_query=answer.lower() in YES_NO,
            provenance={'task': f'qa{task}', 'story': story_id,
                        'upstream_story_index': story_index, 'upstream_line': item['line'],
                        'distractor_lines': count, 'prior_lines': len(prior),
                        'hidden_later_lines': sum(time > query_time for time in passages)})
        if built is not None:
            result.append((built, list(passages.values())))
    return result


def build_facts(raw: Path, output: Path, *, train: int = 10000, validation: int = 500,
                source_budget: int = 20000, validation_source_budget: int = 2000,
                tasks: tuple[int, ...] = BABI_TASKS, distractor_text: Path | None = None,
                max_distractors: int = 2, distractor_fraction: float = 0.5,
                max_prior: int = 48, seed: int = 2609, tokenizer=None) -> dict:
    writer = Writer(output, FACTS)
    pool = distractor_pool(distractor_text) if distractor_text else []
    ids = random.Random(f'{seed}:story-ids')
    used_ids: set[int] = set()
    budgets = {'validation': (validation, validation_source_budget),
               'train': (train, source_budget - validation_source_budget)}
    for split in ('validation', 'train'):
        episodes_total, sources_total = budgets[split]
        filters = writer.filters_for(split)
        split_start = len(writer.sources)
        for task_rank, task in enumerate(tasks):
            remaining = len(tasks) - task_rank
            quota = (episodes_total - len(writer.episodes.get(split, []))) // remaining
            task_start = len(writer.sources)
            source_quota = (sources_total - (task_start - split_start)) // remaining
            stories = load_babi(raw, task, split)
            order = list(range(len(stories)))
            random.Random(f'{seed}:{FACTS}:{split}:{task}').shuffle(order)
            kept = 0
            for story_index in order:
                if kept >= quota:
                    break
                story_id = ids.randrange(10000, 100000)
                while story_id in used_ids:
                    story_id = ids.randrange(10000, 100000)
                rng = random.Random(f'{seed}:{split}:{task}:{story_index}')
                built = babi_story_episodes(
                    stories[story_index], task=task, split=split, story_id=story_id,
                    story_index=story_index, rng=rng, pool=pool,
                    max_distractors=max_distractors, distractor_fraction=distractor_fraction,
                    max_prior=max_prior, filters=filters, tokenizer=tokenizer)
                built = built[:quota - kept]
                if not built:
                    continue
                story_sources = {row['record_id'] for _, rows in built for row in rows}
                if len(writer.sources) - task_start + len(story_sources) > source_quota:
                    filters.reject('source_budget')
                    break
                used_ids.add(story_id)
                for item, rows in built:
                    kept += writer.add(split, item, rows)
    return writer.close({
        'source': 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz',
        'archive_sha256': '84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c',
        'license': 'CC-BY-3.0 (bAbI data); distractor sentences: QASPER abstracts, CC-BY-4.0',
        'upstream': 'en-10k', 'upstream_splits': {'train': 'train', 'validation': 'test'},
        'tasks': [f'qa{task}' for task in tasks], 'seed': seed,
        'limits': {'train': train, 'validation': validation, 'sources': source_budget,
                   'validation_sources': validation_source_budget, 'max_prior': max_prior},
        'distractors': {'max_per_story': max_distractors, 'story_fraction': distractor_fraction,
                        'pool': len(pool)},
        'tokenizer': tokenizer is not None,
        'notice': ('Passages are story lines timed by position; queries at the question '
                   'position see only earlier lines. Story ids are random; all lines of a '
                   'story are stored even when only a prefix is queried.')})


# RULER-style generator ----------------------------------------------------------------

CONSONANTS = 'bdfgklmnprstvzh'
VOWELS = 'aeiou'
LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


class Names:
    """Unique random opaque identifiers."""

    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def _unique(self, make) -> str:
        while True:
            value = make()
            if value not in self.used:
                self.used.add(value)
                return value

    def word(self) -> str:
        return self._unique(lambda: ''.join(
            self.rng.choice(CONSONANTS) + self.rng.choice(VOWELS)
            for _ in range(self.rng.randint(2, 3))) + self.rng.choice(CONSONANTS))

    def code(self) -> str:
        return self._unique(lambda: ''.join(self.rng.choice(LETTERS) for _ in range(3))
                            + '-' + str(self.rng.randint(1000, 9999)))

    def number(self) -> str:
        return self._unique(lambda: str(self.rng.randint(10000, 99999)))

    def registry(self) -> int:
        return int(self._unique(lambda: str(self.rng.randint(100000, 999999))))


def _kv_line(key: str, value: str) -> str:
    return f'The code for {key} is {value}.'


def generate_story(family: str, names: Names, rng: random.Random) -> dict:
    """A timeline of statements (index 0 = time 1) and questions over it."""
    if family in {'kv_single', 'kv_multi'}:
        keys = [names.word() for _ in range(rng.randint(6, 10))]
        values = [names.code() for _ in keys]
        lines = [_kv_line(k, v) for k, v in zip(keys, values)]
        questions = []
        for _ in range(4):
            width = 1 if family == 'kv_single' else rng.randint(2, 3)
            query_time = rng.randint(width + 1, len(lines) + 1)
            chosen = sorted(rng.sample(range(query_time - 1), width))
            if width == 1:
                text = f'what is the code for {keys[chosen[0]]}?'
            else:
                text = ('what are the codes for ' + ', '.join(keys[i] for i in chosen[:-1])
                        + f' and {keys[chosen[-1]]}? Answer in that order.')
            questions.append({'question': text, 'answer': ', '.join(values[i] for i in chosen),
                              'gold': chosen, 'query_time': query_time,
                              'needle': chosen[0], 'key': keys[chosen[0]]})
        return {'lines': lines, 'questions': questions, 'values': values, 'keys': keys}
    if family == 'kv_update':
        keys = [names.word() for _ in range(rng.randint(5, 8))]
        values = [names.code() for _ in keys]
        lines = [_kv_line(k, v) for k, v in zip(keys, values)]
        target = rng.randrange(len(keys) - 1)
        new_value = names.code()
        update_at = rng.randint(target + 1, len(lines))
        lines.insert(update_at, f'The code for {keys[target]} is changed to {new_value}.')
        text = f'what is the code for {keys[target]}?'
        before = rng.randint(target + 2, update_at + 1)
        after = rng.randint(update_at + 2, len(lines) + 1)
        pair = {'line_before': target + 1, 'line_after': update_at + 1}
        return {'lines': lines, 'questions': [
            {'question': text, 'answer': values[target], 'gold': [target],
             'query_time': before, 'temporal': 'before_update', **pair},
            {'question': text, 'answer': new_value, 'gold': [update_at],
             'query_time': after, 'temporal': 'after_update', **pair}]}
    if family == 'vt_chain':
        hops = rng.randint(2, 4)
        chain = [names.word() for _ in range(hops + 1)]
        other = [names.word() for _ in range(rng.randint(2, 3))]
        value, other_value = names.number(), names.number()
        statements = [('chain', 0, f'VAR {chain[0]} = {value}.')]
        statements += [('chain', i, f'VAR {chain[i]} = VAR {chain[i - 1]}.')
                       for i in range(1, len(chain))]
        noise = [('other', 0, f'VAR {other[0]} = {other_value}.')]
        noise += [('other', i, f'VAR {other[i]} = VAR {other[i - 1]}.')
                  for i in range(1, len(other))]
        noise += [('kv', 0, _kv_line(names.word(), names.code()))
                  for _ in range(rng.randint(1, 3))]
        # Merge preserving each chain's internal order.
        merged, a, b = [], list(statements), list(noise)
        while a or b:
            source_list = a if (a and (not b or rng.random() < 0.5)) else b
            merged.append(source_list.pop(0))
        gold = [i for i, (kind, _, _) in enumerate(merged) if kind == 'chain']
        other_gold = [i for i, (kind, _, _) in enumerate(merged) if kind == 'other']
        return {'lines': [text for _, _, text in merged], 'questions': [
            {'question': f'what value does VAR {chain[-1]} hold?', 'answer': value,
             'gold': gold, 'query_time': len(merged) + 1, 'needle': gold[0]},
            {'question': f'what value does VAR {other[-1]} hold?', 'answer': other_value,
             'gold': other_gold, 'query_time': len(merged) + 1, 'needle': other_gold[0]}]}
    raise ValueError(family)


def counterfactual_twin(story: dict, family: str, question: dict, names: Names) -> dict:
    """Same timeline and question with the needle value replaced."""
    needle = question['needle']
    lines = list(story['lines'])
    old = question['answer']
    new = names.number() if family == 'vt_chain' else names.code()
    if old not in lines[needle]:
        raise AssertionError('needle value missing from its line')
    lines[needle] = lines[needle].replace(old, new)
    return {'lines': lines, 'questions': [{**question, 'answer': new}],
            'changed': {'time': needle + 1, 'original': old, 'counterfactual': new}}


def ruler_story_episodes(story: dict, *, family: str, split: str, registry: int,
                         identifier: str, filters: Filters, tokenizer=None,
                         extra: dict | None = None) -> list[tuple[dict, list[dict]]]:
    passages = [source(RULER, f'Registry {registry}, entry {time}: {text}', created_at=time,
                       provenance={'family': family, 'registry': registry, 'entry': time,
                                   'generator_split': split})
                for time, text in enumerate(story['lines'], start=1)]
    result = []
    for index, question in enumerate(story['questions']):
        query_time = question['query_time']
        prior = passages[:query_time - 1]
        gold = [passages[i] for i in question['gold']]
        meta = {'family': family, 'registry': registry, 'prior_entries': len(prior),
                'hidden_later_entries': len(passages) - len(prior), **(extra or {})}
        if 'temporal' in question:
            meta |= {'temporal_pair': f'{RULER}-{identifier}', 'temporal_role': question['temporal']}
        built = episode(
            domain=RULER, split=split, identifier=f'{identifier}-q{index}',
            question=f'Registry {registry}: {question["question"]}', answer=question['answer'],
            gold=gold, supports=prior, filters=filters, tokenizer=tokenizer,
            query_time=query_time, all_required=True, annotation='verified',
            task_family='synthetic_ruler', provenance=meta)
        if built is not None:
            result.append((built, passages))
    return result


RULER_FAMILIES = ('kv_single', 'kv_multi', 'kv_update', 'vt_chain')


def build_ruler(output: Path, *, train: int = 10000, validation: int = 500,
                source_budget: int = 20000, validation_source_budget: int = 1800,
                counterfactual_fraction: float = 0.3, seed: int = 2609,
                tokenizer=None) -> dict:
    writer = Writer(output, RULER)
    names = Names(random.Random(f'{seed}:{RULER}:names'))
    budgets = {'validation': (validation, validation_source_budget),
               'train': (train, source_budget - validation_source_budget)}
    pairs = 0
    for split in ('validation', 'train'):
        wanted, sources_allowed = budgets[split]
        filters = writer.filters_for(split)
        rng = random.Random(f'{seed}:{RULER}:{split}')
        start_sources = len(writer.sources)
        index = 0
        while len(writer.episodes.get(split, [])) < wanted:
            family = RULER_FAMILIES[index % len(RULER_FAMILIES)]
            story = generate_story(family, names, rng)
            identifier = f'{split}-{index:06d}'
            index += 1
            rows = ruler_story_episodes(story, family=family, split=split,
                                        registry=names.registry(), identifier=identifier,
                                        filters=filters, tokenizer=tokenizer)
            remaining = wanted - len(writer.episodes.get(split, []))
            if (family in {'kv_single', 'vt_chain'} and rng.random() < counterfactual_fraction
                    and rows and rows[0][0]['episode_id'].endswith('-q0')):
                twin = counterfactual_twin(story, family, story['questions'][0], names)
                pair = {'counterfactual_pair': f'{RULER}-{identifier}',
                        'counterfactual_changed': twin['changed']}
                twin_rows = ruler_story_episodes(
                    twin, family=family, split=split, registry=names.registry(),
                    identifier=f'{identifier}-cf', filters=filters, tokenizer=tokenizer,
                    extra=pair | {'counterfactual_role': 'counterfactual'})
                if twin_rows and len(rows) + len(twin_rows) <= remaining:
                    rows[0][0]['provenance'] |= pair | {'counterfactual_role': 'original'}
                    rows += twin_rows
                    pairs += 1
                else:
                    filters.reject('counterfactual_unpaired')
            rows = rows[:remaining]
            partners = Counter(item['provenance'].get('temporal_pair') for item, _ in rows)
            rows = [(item, sources) for item, sources in rows  # never keep half a pair
                    if partners[item['provenance'].get('temporal_pair')] != 1
                    or 'temporal_pair' not in item['provenance']]
            new = {row['record_id'] for _, sources in rows for row in sources}
            if len(writer.sources) - start_sources + len(new - writer.sources.keys()) \
                    > sources_allowed:
                filters.reject('source_budget')
                break
            for item, sources in rows:
                writer.add(split, item, sources)
    return writer.close({
        'source': 'generated', 'generator': 'scripts/prepare_synthetic_memory.py',
        'license': 'project-generated', 'seed': seed, 'families': list(RULER_FAMILIES),
        'counterfactual_fraction': counterfactual_fraction, 'counterfactual_pairs': pairs,
        'limits': {'train': train, 'validation': validation, 'sources': source_budget,
                   'validation_sources': validation_source_budget},
        'tokenizer': tokenizer is not None,
        'notice': ('Registry entries are timed by position. kv_update episodes form '
                   'temporal pairs (same query text, different query_time and answer). '
                   'Counterfactual twins differ in one needle value and in the opaque '
                   'registry id only. Names and values are random strings.')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True,
                        help='Directory holding babi/ (and qasper/ for distractor text)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Parent directory; writes <output>/synthetic_facts and '
                             '<output>/synthetic_ruler')
    parser.add_argument('--datasets', nargs='+', default=[FACTS, RULER], choices=[FACTS, RULER])
    parser.add_argument('--no-distractors', action='store_true')
    parser.add_argument('--seed', type=int, default=2609)
    parser.add_argument('--no-tokenizer', action='store_true')
    args = parser.parse_args()
    tokenizer = None if args.no_tokenizer else load_tokenizer()
    for name in args.datasets:
        if name == FACTS:
            text = None if args.no_distractors else args.raw / 'qasper/qasper-train-dev-v0.3.tgz'
            summary = build_facts(args.raw / 'babi', args.output / name, distractor_text=text,
                                  seed=args.seed, tokenizer=tokenizer)
        else:
            summary = build_ruler(args.output / name, seed=args.seed, tokenizer=tokenizer)
        print(json.dumps({key: summary[key] for key in (
            'domain', 'episodes', 'sources', 'gold_per_episode', 'filtered')}, indent=2))
