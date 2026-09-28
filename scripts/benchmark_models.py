"""Reference models on the task corpora (docs/knowledge-base-stack.md, 1.1 and WP4).

Measures the target the small decoder plus the knowledge base has to reach: a
Hugging Face chat model answers validation episodes of the task corpora
(``scripts/prepare_task_corpora.py``), scored by the episode's ``verify`` spec
(``schnitz.task_verifiers.check_episode``). Two conditions per task:

- ``context``: the episode's ``supports`` (its required records first, then the
  related records the corpus attached; tool docs, schemas, evidence, rules,
  worked examples) as text in the prompt. Required records are always included;
  related records while the total stays within the task's character budget
  (``Task.context_chars``, set from the measured lengths so that no validation
  episode is cut; ``--context-chars`` overrides it). This is oracle retrieval as
  plain text, the text-arm upper reference.
- ``closed_book``: the same question without records (SQL tasks keep the
  database name).

The corpus queries start with "Use the stored ..."; that sentence becomes
"Use the ... above." with context and is dropped closed-book. Greedy decoding
(repetition penalty 1) with the model's chat template; ``--chat-kwargs`` passes
template switches such as ``{"enable_thinking": false}``. Reasoning is stripped
before scoring (the text after the last ``</think>``; an unfinished reasoning
block scores wrong and is counted). The same episodes (seeded by task) are used
for every model and condition, so results pair per episode.

Agent trajectories (APIGen-MT) are scored per turn: every assistant turn that is
a tool call is a scored item, given its causal prefix (the gold earlier turns as
chat messages, earlier calls in the instructed JSON-list form, then the tool
results and customer messages) and nothing after it; the first predicted call
must equal the gold call. The summary gives the per-turn accuracy and, in
``by_turn``, the first call of each episode, calls that open an episode (the
earlier first-call metric) and calls other than the free-text ``think`` tool.

Reasoning Gym episodes are scored with the generator's normalizations
(``reasoning_gym_match``, the generator from ``provenance.task``); SynLogic with the
SynLogic verifiers (``synlogic_match``). SynLogic ``futoshiki`` is excluded: its
stored puzzles contradict their own constraints (3 of 20 validation answers pass
the family's verifier and 12 of 20 puzzles have no solution).

Prompt lengths: every prompt with its generation budget has to fit
``--context-window`` tokens (32k, the LFM2.5 window; the other reference models
have 128k); longer prompts are not generated and score wrong (counted as
``prompt_too_long``). Measured with the LFM2.5 and Ling-3.0 tokenizers on all
validation episodes, the longest full-context prompt is 20.6k (LFM) / 25.4k
(Ling) tokens (spider_memory), so no episode is cut or skipped at these budgets.

Output: ``--output`` JSON with per (model, task, condition) accuracy, Wilson 95%
interval (per item; per-turn items of one episode are not independent), n, group
accuracies, truncation counts and a few samples; every item's outcome goes to
``<output>.episodes.jsonl`` as it is scored, and a rerun skips finished items and
keys of the same ``HARNESS`` version (resumable). One model per process. Runs in
``sdkb-bgkit``; Ling-3.0-tiny needs transformers 4.57 (see the WP4 report) and
``--trust-remote-code``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import time

from schnitz.task_verifiers import check_episode

CORPORA = Path('/archive/corpora')
# version of prompts and scoring; results and items of another version are redone
HARNESS = 2


@dataclass(frozen=True)
class Task:
    corpus: str
    max_new: int
    context_chars: int  # related-record budget (covers every validation episode)
    group: str | None = None  # provenance field for per-group accuracy
    suffix: str = ''  # answer-format instruction appended to the question
    turns: bool = False  # APIGen-MT: every tool-call turn is scored given its prefix
    verify_task: str | None = None  # provenance field passed to the verifier as ``task``
    exclude: frozenset[str] = frozenset()  # group values left out (see the docstring)


CALL_SUFFIX = '\n\nTo call a tool, answer with only a JSON list [{"name": ..., "arguments": {...}}].'
# Budgets: the longest validation support set (characters) rounded up; with them the
# longest prompt is 20.6k LFM2.5 tokens (``--measure`` reproduces the numbers).
TASKS = {
    'xlam': Task('tasks-xlam-20260927', 320, 8_000),  # max 5.7k chars
    'spider': Task('tasks-spider-20260927', 256, 20_000),  # max 16.4k
    'bird': Task('tasks-bird-20260927', 320, 48_000, group='difficulty'),  # max 40.4k
    'kodcode': Task('tasks-kodcode-20260927', 768, 8_000, group='difficulty'),  # max 4.3k
    'knights': Task('tasks-knights-20260927', 1200, 80_000, group='people'),  # max 71.9k
    'reasoning_gym': Task('tasks-reasoning-gym-20260927', 2048, 8_000, group='task',
                          verify_task='task',
                          suffix='\n\nEnd with a final line "Answer: <answer>".'),  # max 5.8k
    'apigen_mt': Task('tasks-apigen-mt-20260927', 256, 16_000, group='area', turns=True,
                      suffix=CALL_SUFFIX),  # max 10.6k, all required
    'spider_memory': Task('tasks-spider-memory-20260927', 128, 48_000),  # max 40.6k, all required
    'synlogic': Task('tasks-synlogic-20260927', 2048, 8_000, group='puzzle',
                     exclude=frozenset({'futoshiki'})),  # max 4.0k
}
CONDITIONS = ('context', 'closed_book')
STORED = re.compile(r'^Use the stored ([^.\n]*)\.\s*')


def load_episodes(task: str, per_task: int, seed: int, corpora: Path = CORPORA) -> list[dict]:
    """``per_task`` validation episodes of ``task``, the same for every model."""
    spec = TASKS[task]
    rows = []
    with (corpora / spec.corpus / 'episodes-validation.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if spec.group and row['provenance'].get(spec.group) in spec.exclude:
                continue
            if spec.turns and not call_turns(row):
                continue
            rows.append(row)
    rng = random.Random(f'{seed}:{task}')
    return rng.sample(rows, min(per_task, len(rows)))


def context_records(episode: dict, budget: int) -> tuple[list[str], bool]:
    """Support texts, required records first and always; related records each whole
    while the total stays within ``budget`` characters. The flag says whether a
    related record was left out."""
    required = set(episode['required_ids'])
    ordered = sorted(episode['supports'], key=lambda s: s['record_id'] not in required)
    texts, used, truncated = [], 0, False
    for support in ordered:
        if support['record_id'] not in required and used + len(support['text']) > budget:
            truncated = True
            continue
        texts.append(support['text'])
        used += len(support['text'])
    return texts, truncated


def build_prompt(task: str, episode: dict, condition: str, budget: int) -> tuple[str, bool]:
    """The user message for ``condition`` and whether the context was truncated."""
    query = episode['query']
    found = STORED.match(query)
    question = (query[found.end():] if found else query) + TASKS[task].suffix
    if condition == 'closed_book':
        db = episode['provenance'].get('db_id') \
            if episode['provenance'].get('db_handle') != 'none' else None
        return (f'Database: {db}\n' if db else '') + question, False
    texts, truncated = context_records(episode, budget)
    notes = '\n\n'.join(f'[{i}] {text}' for i, text in enumerate(texts, 1))
    what = found.group(1) if found else 'reference notes'
    return f'Reference notes:\n\n{notes}\n\nUse the {what} above. {question}', truncated


def call_turns(episode: dict) -> list[tuple[int, dict]]:
    """(turn index, gold call) of every assistant turn that is one tool call."""
    out = []
    for index, turn in enumerate(episode.get('turns') or []):
        if turn['role'] == 'assistant' and turn['text'].startswith('Call: '):
            try:
                call = json.loads(turn['text'][len('Call: '):])
            except json.JSONDecodeError:
                continue
            if isinstance(call, dict) and 'name' in call:
                out.append((index, call))
    return out


def prefix_messages(episode: dict, turn: int) -> list[dict]:
    """The gold turns before ``turn`` as chat messages: assistant replies as they
    are, assistant calls as the instructed JSON list, customer messages and tool
    results as user messages (consecutive ones joined)."""
    messages: list[dict] = []
    for item in episode['turns'][:turn]:
        role = 'assistant' if item['role'] == 'assistant' else 'user'
        text = item['text']
        if role == 'assistant' and text.startswith('Call: '):
            text = f'[{text[len("Call: "):]}]'
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'] += '\n\n' + text
        else:
            messages.append({'role': role, 'content': text})
    return messages


@dataclass
class Item:
    """One scored generation: an episode, or one call turn of an agent episode."""
    item_id: str
    episode: dict
    messages: list[dict]
    verify: dict
    truncated: bool
    turn: int | None = None
    first_call: bool = False
    tool: str | None = None


def build_items(task: str, episode: dict, condition: str, budget: int) -> list[Item]:
    spec = TASKS[task]
    prompt, truncated = build_prompt(task, episode, condition, budget)
    first = [{'role': 'user', 'content': prompt}]
    if not spec.turns:
        verify = dict(episode['verify'])
        if spec.verify_task:
            verify['task'] = episode['provenance'][spec.verify_task]
        return [Item(episode['episode_id'], episode, first, verify, truncated)]
    items = []
    for rank, (index, call) in enumerate(call_turns(episode)):
        messages = first + prefix_messages(episode, index)
        if len(messages) > 1 and messages[1]['role'] == 'user':  # transcript opens with the user
            messages[:2] = [{'role': 'user',
                             'content': prompt + '\n\n' + messages[1]['content']}]
        items.append(Item(f'{episode["episode_id"]}#t{index}', episode, messages,
                          {'type': 'tau_bench', 'calls': [call]}, truncated, index, rank == 0,
                          call['name']))
    return items


def strip_thinking(text: str, opened: bool) -> tuple[str, bool]:
    """The answer after reasoning, and whether a reasoning block was left unfinished.
    ``opened``: the generation prompt already opened ``<think>``."""
    if '</think>' in text:
        return text.split('</think>')[-1], False
    if opened or text.lstrip().startswith('<think>'):
        return '', True
    return text, False


def safe_check(answer: str, verify: dict) -> bool:
    """``check_episode``; a verifier crash on a malformed answer or spec scores wrong."""
    try:
        return bool(check_episode(answer, verify))
    except Exception:  # noqa: BLE001 - any failure to verify is a failed outcome
        return False


def wilson(correct: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _rate(values: list[bool]) -> dict:
    return {'accuracy': round(sum(values) / max(len(values), 1), 4), 'n': len(values)}


def summarize(rows: list[dict], seconds: float) -> dict:
    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    groups: dict[str, list[bool]] = {}
    for r in rows:
        if r.get('group') is not None:
            groups.setdefault(str(r['group']), []).append(r['correct'])
    out = {
        'accuracy': round(correct / max(n, 1), 4), 'ci95': wilson(correct, n), 'n': n,
        'harness': HARNESS,
        'by_group': {g: _rate(v) for g, v in sorted(groups.items())},
        'context_truncated': sum(r['context_truncated'] for r in rows),
        'thinking_unfinished': sum(r['thinking_unfinished'] for r in rows),
        'hit_max_new_tokens': sum(r['hit_max'] for r in rows),
        'prompt_too_long': sum(r.get('prompt_too_long', False) for r in rows),
        'generation_failed': sum(r.get('generation_failed', False) for r in rows),
        'mean_prompt_tokens': round(sum(r['prompt_tokens'] for r in rows) / max(n, 1)),
        'max_prompt_tokens': max((r['prompt_tokens'] for r in rows), default=0),
        'mean_new_tokens': round(sum(r['new_tokens'] for r in rows) / max(n, 1)),
        'seconds': round(seconds),
        'samples': [{k: r[k] for k in ('episode_id', 'correct', 'prompt_tail', 'output')}
                    for r in rows[:3]],
    }
    if any(r.get('turn') is not None for r in rows):
        out['episodes'] = len({r['episode_id'].split('#')[0] for r in rows})
        out['by_turn'] = {
            'all_calls': _rate([r['correct'] for r in rows]),
            'first_call': _rate([r['correct'] for r in rows if r['first_call']]),
            'opening_call': _rate([r['correct'] for r in rows if r['turn'] == 0]),
            'later_calls': _rate([r['correct'] for r in rows if not r['first_call']]),
            'without_think': _rate([r['correct'] for r in rows if r['tool'] != 'think']),
        }
    return out


class Generator:
    """Greedy batched chat generation with one Hugging Face model."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map='cuda',
            trust_remote_code=args.trust_remote_code).eval()
        self.chat_kwargs = json.loads(args.chat_kwargs)
        self.system = args.system
        self.config = copy.deepcopy(self.model.generation_config)
        self.config.update(do_sample=False, temperature=None, top_p=None, top_k=None,
                           repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
        self.specials = sorted({t for t in self.tok.all_special_tokens
                                if t not in ('<think>', '</think>')}, key=len, reverse=True)

    def template(self, messages: list[dict]) -> str:
        messages = ([{'role': 'system', 'content': self.system}] if self.system else []) + messages
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **self.chat_kwargs)

    def length(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)['input_ids'])

    def generate_split(self, texts: list[str], max_new: int) -> list[dict | None]:
        """``generate``, halving the batch on CUDA out-of-memory (the memory cap);
        ``None`` for a single prompt that does not fit (scored wrong, counted)."""
        if not texts:
            return []
        try:
            return self.generate(texts, max_new)
        except self.torch.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(texts) == 1:
                return [None]
            half = len(texts) // 2
            return self.generate_split(texts[:half], max_new) + \
                self.generate_split(texts[half:], max_new)

    def generate(self, texts: list[str], max_new: int) -> list[dict]:
        torch = self.torch
        batch = self.tok(texts, return_tensors='pt', padding=True,
                         add_special_tokens=False).to(self.model.device)
        batch.pop('token_type_ids', None)  # some tokenizers return them; decoders take none
        with torch.no_grad():
            config = copy.deepcopy(self.config)
            config.max_new_tokens = max_new
            out = self.model.generate(**batch, generation_config=config)
        new = out[:, batch['input_ids'].shape[1]:]
        results = []
        for text, ids in zip(texts, new.tolist()):
            ids = [i for i in ids if i != self.tok.pad_token_id]
            raw = self.tok.decode(ids, skip_special_tokens=False)
            answer, unfinished = strip_thinking(raw, text.rstrip().endswith('<think>'))
            for special in self.specials:
                answer = answer.replace(special, '')
            results.append({'raw': raw, 'answer': answer.strip(), 'thinking_unfinished': unfinished,
                            'new_tokens': len(ids), 'hit_max': len(ids) >= max_new})
        return results


def _quantiles(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {q: ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]
            for q, p in (('p50', .5), ('p90', .9), ('p99', .99), ('max', 1.0))}


def measure(args) -> dict:
    """Token lengths (``--model``'s tokenizer and chat template) of every validation
    item per task and condition, with the characters of required and of all support
    records and the related records the budget leaves out. No model is loaded."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    chat = json.loads(args.chat_kwargs)
    report = {}
    for task in args.tasks:
        spec = TASKS[task]
        budget = args.context_chars or spec.context_chars
        episodes = load_episodes(task, 10 ** 9, args.seed, args.corpora)
        chars = {'required': [], 'all': []}
        for ep in episodes:
            required = set(ep['required_ids'])
            chars['required'].append(sum(len(s['text']) for s in ep['supports']
                                         if s['record_id'] in required))
            chars['all'].append(sum(len(s['text']) for s in ep['supports']))
        entry = {'episodes': len(episodes), 'context_chars': budget, 'max_new': spec.max_new,
                 'support_chars': {k: _quantiles(v) for k, v in chars.items()}}
        for condition in args.conditions:
            lengths, cut = [], 0
            for ep in episodes:
                for item in build_items(task, ep, condition, budget):
                    text = tok.apply_chat_template(item.messages, tokenize=False,
                                                   add_generation_prompt=True, **chat)
                    lengths.append(len(tok(text, add_special_tokens=False)['input_ids']))
                    cut += item.truncated
            entry[condition] = {'items': len(lengths), 'truncated': cut,
                                'over_window': sum(n + spec.max_new + args.extra_new_tokens
                                                   > args.context_window for n in lengths),
                                'prompt_tokens': _quantiles(lengths)}
        report[task] = entry
        print(json.dumps({task: entry}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='local Hugging Face model directory')
    parser.add_argument('--label', required=True, help='model name in the results')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument('--per-task', type=int, default=200,
                        help='episodes per task (agent tasks score every call turn of them)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--corpora', type=Path, default=CORPORA)
    parser.add_argument('--context-chars', type=int,
                        help='related-record budget for every task (default: per task)')
    parser.add_argument('--context-window', type=int, default=32768,
                        help='prompt plus generation budget must fit; longer prompts are not '
                             'generated and score wrong (counted)')
    parser.add_argument('--extra-new-tokens', type=int, default=0,
                        help='added to every task budget, for reasoning models')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-batch-tokens', type=int, default=32768,
                        help='cap on batch size x (longest prompt + new tokens)')
    parser.add_argument('--chat-kwargs', default='{}', help='JSON passed to apply_chat_template')
    parser.add_argument('--system', default='', help='optional system message')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--cuda-fraction', type=float, default=0.08)
    parser.add_argument('--verify-workers', type=int, default=8)
    parser.add_argument('--measure', action='store_true',
                        help='only report prompt token lengths of all validation items '
                             '(tokenizer of --model) to --output; no generation')
    args = parser.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.measure:
        args.output.write_text(json.dumps(measure(args), indent=2) + '\n')
        return

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    results.setdefault('meta', {})
    results.setdefault('results', {})
    sidecar = args.output.with_name(args.output.name + '.episodes.jsonl')
    done: dict[str, dict[str, dict]] = {}
    if sidecar.exists():
        for line in sidecar.open(encoding='utf-8'):
            row = json.loads(line)
            if row.get('harness') == HARNESS:
                done.setdefault(row['key'], {})[row['episode_id']] = row
    todo = [(t, c) for t in args.tasks for c in args.conditions
            if results['results'].get(f'{args.label}/{t}/{c}', {}).get('harness') != HARNESS]
    if not todo:
        print('nothing to do', flush=True)
        return
    generator = Generator(args)
    results['meta'][args.label] = {
        'model': args.model, 'chat_kwargs': args.chat_kwargs, 'system': args.system,
        'harness': HARNESS, 'context_window': args.context_window,
        'context_chars': {t: args.context_chars or TASKS[t].context_chars for t in args.tasks},
        'extra_new_tokens': args.extra_new_tokens, 'seed': args.seed, 'per_task': args.per_task}
    log = sidecar.open('a', encoding='utf-8')
    verify = ThreadPoolExecutor(args.verify_workers)
    for task, condition in todo:
        key = f'{args.label}/{task}/{condition}'
        spec = TASKS[task]
        max_new = spec.max_new + args.extra_new_tokens
        budget = args.context_chars or spec.context_chars
        episodes = load_episodes(task, args.per_task, args.seed, args.corpora)
        finished = done.setdefault(key, {})
        all_items = [item for ep in episodes for item in build_items(task, ep, condition, budget)]
        pending = []
        for item in all_items:
            if item.item_id in finished:
                continue
            text = generator.template(item.messages)
            pending.append((item, text, generator.length(text)))
        pending.sort(key=lambda entry: -entry[2])
        started = time.time()
        index = 0
        while index < len(pending):
            longest = pending[index][2]
            size = max(1, min(args.batch_size, args.max_batch_tokens // (longest + max_new)))
            chunk = pending[index:index + size]
            index += size
            runnable = [entry for entry in chunk if entry[2] + max_new <= args.context_window]
            outs = generator.generate_split([entry[1] for entry in runnable], max_new)
            by_id = {entry[0].item_id: out for entry, out in zip(runnable, outs) if out is not None}
            checks = verify.map(lambda entry: safe_check(by_id[entry[0].item_id]['answer'],
                                                         entry[0].verify)
                                if entry[0].item_id in by_id else False, chunk)
            for (item, _, length), correct in zip(chunk, checks):
                out = by_id.get(item.item_id, {'raw': '', 'answer': '', 'new_tokens': 0,
                                               'thinking_unfinished': False, 'hit_max': False})
                too_long = length + max_new > args.context_window
                row = {'key': key, 'episode_id': item.item_id, 'harness': HARNESS,
                       'correct': correct,
                       'group': item.episode['provenance'].get(spec.group) if spec.group else None,
                       'context_truncated': item.truncated, 'prompt_tokens': length,
                       'prompt_too_long': too_long,
                       'generation_failed': not too_long and item.item_id not in by_id,
                       'thinking_unfinished': out['thinking_unfinished'], 'hit_max': out['hit_max'],
                       'new_tokens': out['new_tokens'],
                       'prompt_tail': item.messages[-1]['content'][-400:],
                       'answer': out['answer'][:2000], 'output': out['raw'][-4000:]}
                if item.turn is not None:
                    row.update(turn=item.turn, first_call=item.first_call, tool=item.tool)
                finished[item.item_id] = row
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            print(f'{key}: {len(finished)}/{len(all_items)} '
                  f'({time.time() - started:.0f}s)', flush=True)
        rows = [finished[item.item_id] for item in all_items]
        results['results'][key] = summarize(rows, time.time() - started)
        print(json.dumps({key: {k: results['results'][key][k] for k in ('accuracy', 'ci95', 'n')}}),
              flush=True)
        tmp = args.output.with_suffix('.pending')
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(args.output)
    verify.shutdown()


if __name__ == '__main__':
    main()
