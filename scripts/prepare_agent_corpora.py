"""Agent-trajectory corpora with protocol, know-how and examples in the KB (plan B9).

Sources: simulator-grounded ALFWorld and ScienceWorld trajectories (cp2107, with
its train/validation/test splits), ETO's ReAct trajectories with thoughts
(ALFWorld, ScienceWorld, WebShop; training games only), THUDM AgentInstruct and
AgentBank conversations. Every trajectory becomes one block list (goal, initial
observation, then assistant turns "Thought: ... / Action: ..." and environment
turns "Observation: ...").

Knowledge base per domain:
- ``protocol``: the environment's interaction instructions (and the in-context
  example the source prompt carried), deduplicated - no longer part of the query;
- ``worked_example``: a held-out pool (``--pool`` of the training trajectories,
  by hash) rendered as goal and actions; pool trajectories are never episodes, so
  no episode can read its own trajectory;
- ``know_how`` (ALFWorld): per floorplan, where objects were seen in the pool's
  observations - the environment regularities a recursive run should accumulate.
Background (ConceptNet, household and science facts) is added afterwards with
``add_background.py``.

Episodes keep the R6 schema; ``turns`` holds the trajectory ({"role":
"assistant"|"environment", "text"}), trained on assistant turns only, and
``answer`` its text. ``verify`` names the environment and game so attempts can be
scored by the simulator where one runs; the stored trajectories all succeed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_task_corpora import RAW, pack, record, task_episode  # noqa: E402
from public_corpus_common import Writer  # noqa: E402

WORLDS = RAW / 'worlds-20260927'
PROTOCOLS = {
    'alfworld': ('ALFWorld household tasks: act with one command per turn - go to X, take X from '
                 'Y, put X in/on Y, open X, close X, toggle X, heat X with Y, cool X with Y, clean '
                 'X with Y, use X, examine X, look, inventory. Receptacles and objects carry '
                 'numbers (countertop 1, apple 2). Reply as "Thought: ...\\nAction: ...".'),
    'scienceworld': ('ScienceWorld tasks: explore rooms, pick up, move, open, activate and focus on '
                     'objects to run the experiment the task describes; "look around" lists a '
                     'room, "teleport to ROOM" moves between rooms, "focus on X" commits to the '
                     'object the task is about. Reply as "Thought: ...\\nAction: ...".'),
}


def _hash_fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _floorplan(source: str) -> str | None:
    found = re.search(r'-(\d+)/trial_', source or '')
    return found.group(1) if found else None


def cp2107(env: str, folder: str):
    """Yield (split, id, goal, blocks, meta) from cp2107 trajectory files."""
    for split in ('train', 'validation', 'test'):
        path = WORLDS / 'cp2107-textworld-trajectories' / folder / f'{split}.jsonl'
        for line in path.open():
            row = json.loads(line)
            yield split, row['trajectory_id'], row['goal'], row['blocks'], {
                'source': row.get('source'), 'floorplan': _floorplan(row.get('source', '')),
                'task_id': row['trajectory_id'] if env == 'scienceworld' else None}


def _turns_from_blocks(blocks: list[dict]) -> tuple[str, list[dict]]:
    """Initial observation and alternating turns from Goal/Think/Action/Observation blocks."""
    initial, turns, thought = '', [], None
    for block in blocks:
        kind, text = block['type'], block['text'].strip()
        if kind == 'Observation' and not turns and thought is None and not initial:
            initial = text
        elif kind == 'Think':
            thought = text
        elif kind == 'Action':
            head = f'Thought: {thought}\n' if thought else ''
            turns.append({'role': 'assistant', 'text': f'{head}Action: {text}'})
            thought = None
        elif kind == 'Observation':
            turns.append({'role': 'environment', 'text': f'Observation: {text}'})
    return initial, turns


def sharegpt(rows, loss_key: str | None = 'loss'):
    """(protocol, task, turns) from ShareGPT conversations. Turns before the task
    (instructions and in-context example turns, ``loss`` False) form the protocol."""
    for row in rows:
        conv = row['conversations']
        if isinstance(conv, str):
            conv = json.loads(conv)
        first_real = next((i for i, c in enumerate(conv) if c['from'] == 'gpt'
                           and c.get(loss_key) is not False and c['value'].strip() != 'OK'), None)
        if first_real is None or first_real < 1:
            continue
        task = conv[first_real - 1]['value']
        protocol = '\n\n'.join(c['value'] for c in conv[:first_real - 1]
                               if c['value'].strip() != 'OK')
        turns = [{'role': 'assistant' if c['from'] == 'gpt' else 'environment',
                  'text': c['value'].strip()} for c in conv[first_real:]]
        yield row, protocol, task, turns


def _render(goal: str, turns: list[dict]) -> str:
    actions = [re.sub(r'^Thought:.*?\n', '', t['text'], flags=re.DOTALL)
               for t in turns if t['role'] == 'assistant']
    return f'Worked example. Task: {goal}\n' + '\n'.join(actions)


SEEN = re.compile(r'(?:On|In) (?:the )?([a-z]+ \d+)(?:[^.,]*)?, you see (.+?)\.')
OPENED = re.compile(r'You open the ([a-z]+ \d+)\.[^.]*\.\s*In it, you see (.+?)\.')


def _sightings(text: str) -> list[tuple[str, str]]:
    found = []
    for pattern in (SEEN, OPENED):
        for place, items in pattern.findall(text):
            for item in re.split(r',\s*(?:and\s+)?|\s+and\s+', items):
                item = re.sub(r'^an?\s+', '', item.strip())
                if item and item != 'nothing':
                    found.append((re.sub(r'\s+\d+$', '', item), place))
    return found


class Domain:
    def __init__(self, output: Path, domain: str, pool: float):
        self.writer, self.domain, self.pool = Writer(output, domain), domain, pool
        self.protocols: dict[str, dict] = {}
        self.examples: dict[str, list[dict]] = defaultdict(list)
        self.know: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.pending: list[tuple] = []
        self.counts = defaultdict(int)

    def protocol(self, text: str, env: str) -> list[dict]:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.protocols:
            lines = [line for line in text.split('\n') if line.strip()]
            self.protocols[key] = [record(self.domain, part, 'protocol', env=env)
                                   for part in pack(f'{env} protocol:\n', lines)]
        return self.protocols[key]

    def add(self, split: str, ident: str, goal: str, initial: str, turns: list[dict],
            protocol: list[dict], verify: dict, *, family: str, group: str = '',
            floorplan: str | None = None):
        if not turns or not any(t['role'] == 'assistant' for t in turns):
            self.counts['empty'] += 1
            return
        if split == 'train' and _hash_fraction(f'{self.domain}:{ident}') < self.pool:
            self.examples[group].append(record(self.domain, _render(goal, turns)[:1500],
                                               'worked_example', group=group))
            if floorplan:
                for turn in turns:
                    if turn['role'] == 'environment':
                        for item, place in _sightings(turn['text']):
                            self.know[floorplan][item].add(place)
            return
        self.pending.append((split, ident, goal, initial, turns, protocol, verify, family, group,
                             floorplan))

    def close(self, meta: dict) -> dict:
        know = {}
        for plan, items in self.know.items():
            lines = [f'{item}: {", ".join(sorted(places))}' for item, places in sorted(items.items())]
            know[plan] = [record(self.domain, part, 'know_how', floorplan=plan)
                          for part in pack(f'Floorplan {plan}, where objects were found:\n', lines)]
        for (split, ident, goal, initial, turns, protocol, verify, family, group,
             plan) in self.pending:
            examples = self.examples.get(group, [])
            picks = sorted(examples, key=lambda r: hashlib.sha256(
                (ident + r['record_id']).encode()).hexdigest())[:3]
            required = protocol + know.get(plan, [])
            query = ('Use the stored protocol, know-how and worked examples.\n'
                     f'Task: {goal}\n' + (f'{initial}\n' if initial else ''))
            item = task_episode(self.domain, split, ident, query.strip(),
                                '\n'.join(t['text'] for t in turns), required, required + picks,
                                verify, family, group=group, floorplan=plan)
            item['turns'] = turns
            everything = [r for rs in self.protocols.values() for r in rs]
            everything += [r for rs in self.examples.values() for r in rs]
            everything += [r for rs in know.values() for r in rs]
            self.writer.add(split, item, everything if not self.writer.sources else required + picks)
        return self.writer.close({'domain': self.domain, 'pool': self.pool,
                                  'protocols': len(self.protocols),
                                  'worked_examples': sum(map(len, self.examples.values())),
                                  'know_how_floorplans': len(know), **meta,
                                  'skipped': dict(self.counts)})


def alfworld(output: Path, pool: float) -> dict:
    dom = Domain(output, 'alfworld', pool)
    base = dom.protocol(PROTOCOLS['alfworld'], 'alfworld')
    held = set()
    for split, ident, goal, blocks, meta in cp2107('alfworld', 'alfworld-rollouts-v3'):
        if split != 'train':
            held.add(re.sub(r'/game\.tw-pddl$', '', meta['source'] or ''))
        initial, turns = _turns_from_blocks(blocks)
        task = re.sub(r'^.*?/(\w+?)-.*$', r'\1', meta['source'] or 'unknown')
        dom.add(split, ident, goal, initial, turns, base,
                {'type': 'trajectory', 'env': 'alfworld', 'game': meta['source']},
                family='household_agent', group=task, floorplan=meta['floorplan'])
    eto = json.load((WORLDS / 'eto-sft-trajectory/data/alfworld_sft.json').open())
    for row, protocol, task_text, turns in sharegpt(eto, loss_key='__none__'):
        game = row['game_file'].replace('data/', '')
        if game in held:
            dom.counts['eto_heldout_game'] += 1
            continue
        goal = re.search(r'Your task is to: (.+)', task_text)
        initial = task_text.split('\n\nYour task is to:')[0]
        task = re.sub(r'^.*?/(\w+?)-.*$', r'\1', game)
        dom.add('train', f'eto-{row["id"]}', goal.group(1) if goal else task_text, initial, turns,
                base + dom.protocol(protocol, 'alfworld'),
                {'type': 'trajectory', 'env': 'alfworld', 'game': game},
                family='household_agent', group=task, floorplan=_floorplan(game + '/trial_'))
    return dom.close({'sources_used': ['cp2107 alfworld-rollouts-v3', 'ETO alfworld']})


def scienceworld(output: Path, pool: float) -> dict:
    dom = Domain(output, 'scienceworld', pool)
    base = dom.protocol(PROTOCOLS['scienceworld'], 'scienceworld')
    held = set()
    for split, ident, goal, blocks, _ in cp2107('scienceworld', 'scienceworld-compact-v2'):
        if split != 'train':
            held.add(ident)
        initial, turns = _turns_from_blocks(blocks)
        dom.add(split, ident, goal, initial, turns, base,
                {'type': 'trajectory', 'env': 'scienceworld', 'task': ident},
                family='science_agent', group=ident.split('_')[0])
    eto = json.load((WORLDS / 'eto-sft-trajectory/data/sciworld_sft.json').open())
    for row, protocol, task_text, turns in sharegpt(eto, loss_key='__none__'):
        if row['id'] in held:
            dom.counts['eto_heldout_task'] += 1
            continue
        dom.add('train', f'eto-{row["id"]}', task_text, '', turns,
                base + dom.protocol(protocol, 'scienceworld'),
                {'type': 'trajectory', 'env': 'scienceworld', 'task': row['id']},
                family='science_agent', group=str(row['id']).split('_')[0])
    return dom.close({'sources_used': ['cp2107 scienceworld-compact-v2', 'ETO sciworld']})


def webshop(output: Path, pool: float, validation: float) -> dict:
    dom = Domain(output, 'webshop', pool)
    eto = json.load((WORLDS / 'eto-sft-trajectory/data/webshop_sft.json').open())
    for row, protocol, task_text, turns in sharegpt(eto, loss_key='__none__'):
        ident = str(row['id'])
        split = 'validation' if _hash_fraction('webshop-val:' + ident) < validation else 'train'
        goal = re.search(r'Instruction: \[SEP\] (.+?) \[SEP\]', task_text)
        dom.add(split, ident, goal.group(1) if goal else task_text, task_text, turns,
                dom.protocol(protocol, 'webshop'),
                {'type': 'trajectory', 'env': 'webshop', 'reward': row.get('reward')},
                family='web_agent', group='webshop')
    return dom.close({'sources_used': ['ETO webshop']})


def conversations(output: Path, domain: str, files: list[Path], pool: float,
                  validation: float) -> dict:
    """AgentInstruct / AgentBank subsets: ShareGPT conversations, protocol into the KB."""
    import pyarrow.parquet as pq
    dom = Domain(output, domain, pool)
    for path in files:
        env = path.parent.name if path.parent.name != 'data' else path.name.split('-')[0]
        for row, protocol, task_text, turns in sharegpt(pq.read_table(path).to_pylist()):
            ident = f'{env}-{row.get("id")}'
            split = 'validation' if _hash_fraction(f'{domain}-val:{ident}') < validation else 'train'
            dom.add(split, ident, task_text.strip()[:2000], '', turns,
                    dom.protocol(protocol, env) if protocol.strip() else [],
                    {'type': 'trajectory', 'env': env}, family=f'{env}_agent', group=env)
    return dom.close({'sources_used': [str(p) for p in files]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('alfworld', 'scienceworld', 'webshop',
                                            'agentinstruct', 'agentbank'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=float, default=0.1)
    parser.add_argument('--validation', type=float, default=0.05)
    parser.add_argument('--subsets', nargs='*', default=[
        'alfred', 'iqa', 'rearrange', 'intercode_bash', 'intercode_sql', 'mind2web', 'webarena'])
    args = parser.parse_args()
    if args.dataset == 'alfworld':
        manifest = alfworld(args.output, args.pool)
    elif args.dataset == 'scienceworld':
        manifest = scienceworld(args.output, args.pool)
    elif args.dataset == 'webshop':
        manifest = webshop(args.output, args.pool, args.validation)
    elif args.dataset == 'agentinstruct':
        files = sorted((WORLDS / 'thudm-agentinstruct/data').glob('*.parquet'))
        files = [f for f in files if f.name.split('-')[0] in ('db', 'os', 'kg', 'mind2web')]
        manifest = conversations(args.output, 'agentinstruct', files, args.pool, args.validation)
    else:
        files = [f for s in args.subsets for f in sorted((WORLDS / 'agentbank' / s).glob('*.parquet'))]
        manifest = conversations(args.output, 'agentbank', files, args.pool, args.validation)
    print(json.dumps({k: manifest[k] for k in ('episodes', 'sources', 'filtered')} | {
        k: manifest.get(k) for k in ('protocols', 'worked_examples', 'know_how_floorplans',
                                     'skipped')}))


if __name__ == '__main__':
    main()
