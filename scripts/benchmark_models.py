"""Reference models on the task corpora (docs/knowledge-base-stack.md, 1.1 and WP4).

Measures the target the small decoder plus the knowledge base has to reach: a
Hugging Face chat model answers validation episodes of the task corpora
(``scripts/prepare_task_corpora.py``), scored by the episode's ``verify`` spec
(``schnitz.task_verifiers.check_episode``). Two conditions per task:

- ``context``: the episode's ``supports`` (its required records first, then the
  related records the corpus attached; tool docs, schemas, evidence, rules,
  worked examples) as text in the prompt, up to ``--context-chars``. This is
  oracle retrieval as plain text, the text-arm upper reference.
- ``closed_book``: the same question without records (SQL tasks keep the
  database name).

The corpus queries start with "Use the stored ..."; that sentence becomes
"Use the ... above." with context and is dropped closed-book. Greedy decoding
(repetition penalty 1) with the model's chat template; ``--chat-kwargs`` passes
template switches such as ``{"enable_thinking": false}``. Reasoning is stripped
before scoring (the text after the last ``</think>``; an unfinished reasoning
block scores wrong and is counted). The same episodes (seeded by task) are used
for every model and condition, so results pair per episode.

Output: ``--output`` JSON with per (model, task, condition) accuracy, Wilson 95%
interval, n, group accuracies, truncation counts and a few samples; every
episode's outcome goes to ``<output>.episodes.jsonl`` as it is scored, and a
rerun skips finished episodes and keys (resumable). One model per process.
Runs in ``sdkb-bgkit``; Ling-3.0-tiny needs transformers 4.57 (see the WP4
report) and ``--trust-remote-code``.
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


@dataclass(frozen=True)
class Task:
    corpus: str
    max_new: int
    group: str | None = None  # provenance field for per-group accuracy
    suffix: str = ''  # answer-format instruction appended to the question
    first_call_only: bool = False  # APIGen-MT: episodes whose first action is a call


TASKS = {
    'xlam': Task('tasks-xlam-20260927', 320),
    'spider': Task('tasks-spider-20260927', 256),
    'bird': Task('tasks-bird-20260927', 320, group='difficulty'),
    'kodcode': Task('tasks-kodcode-20260927', 768, group='difficulty'),
    'knights': Task('tasks-knights-20260927', 1200, group='people'),
    'reasoning_gym': Task('tasks-reasoning-gym-20260927', 2048,
                          suffix='\n\nEnd with a final line "Answer: <answer>".'),
    'apigen_mt': Task('tasks-apigen-mt-20260927', 256, group='area', first_call_only=True,
                      suffix='\n\nTo call a tool, answer with only a JSON list '
                             '[{"name": ..., "arguments": {...}}].'),
    'spider_memory': Task('tasks-spider-memory-20260927', 128),
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
            if spec.first_call_only and not (row.get('turns') and row['turns'][0]['role'] == 'assistant'
                                             and row['turns'][0]['text'].startswith('Call: ')):
                continue
            rows.append(row)
    rng = random.Random(f'{seed}:{task}')
    return rng.sample(rows, min(per_task, len(rows)))


def context_records(episode: dict, budget: int) -> tuple[list[str], bool]:
    """Support texts, required records first, each included whole while the total
    stays within ``budget`` characters; the flag says whether any was left out."""
    required = set(episode['required_ids'])
    ordered = sorted(episode['supports'], key=lambda s: s['record_id'] not in required)
    texts, used, truncated = [], 0, False
    for support in ordered:
        if used + len(support['text']) > budget:
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
        db = episode['provenance'].get('db_id')
        return (f'Database: {db}\n' if db else '') + question, False
    texts, truncated = context_records(episode, budget)
    notes = '\n\n'.join(f'[{i}] {text}' for i, text in enumerate(texts, 1))
    what = found.group(1) if found else 'reference notes'
    return f'Reference notes:\n\n{notes}\n\nUse the {what} above. {question}', truncated


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


def summarize(rows: list[dict], seconds: float) -> dict:
    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    groups: dict[str, list[bool]] = {}
    for r in rows:
        if r.get('group') is not None:
            groups.setdefault(str(r['group']), []).append(r['correct'])
    return {
        'accuracy': round(correct / max(n, 1), 4), 'ci95': wilson(correct, n), 'n': n,
        'by_group': {g: {'accuracy': round(sum(v) / len(v), 4), 'n': len(v)}
                     for g, v in sorted(groups.items())},
        'context_truncated': sum(r['context_truncated'] for r in rows),
        'thinking_unfinished': sum(r['thinking_unfinished'] for r in rows),
        'hit_max_new_tokens': sum(r['hit_max'] for r in rows),
        'prompt_too_long': sum(r.get('prompt_too_long', False) for r in rows),
        'generation_failed': sum(r.get('generation_failed', False) for r in rows),
        'mean_prompt_tokens': round(sum(r['prompt_tokens'] for r in rows) / max(n, 1)),
        'mean_new_tokens': round(sum(r['new_tokens'] for r in rows) / max(n, 1)),
        'seconds': round(seconds),
        'samples': [{k: r[k] for k in ('episode_id', 'correct', 'prompt_tail', 'output')}
                    for r in rows[:3]],
    }


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

    def template(self, prompt: str) -> str:
        messages = ([{'role': 'system', 'content': self.system}] if self.system else []) + \
            [{'role': 'user', 'content': prompt}]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='local Hugging Face model directory')
    parser.add_argument('--label', required=True, help='model name in the results')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument('--per-task', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--corpora', type=Path, default=CORPORA)
    parser.add_argument('--context-chars', type=int, default=24000)
    parser.add_argument('--max-prompt-tokens', type=int, default=32768,
                        help='longer prompts are not generated and score wrong (counted)')
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
    args = parser.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    results =json.loads(args.output.read_text()) if args.output.exists() else {}
    results.setdefault('meta', {})
    results.setdefault('results', {})
    sidecar = args.output.with_name(args.output.name + '.episodes.jsonl')
    done: dict[str, dict[str, dict]] = {}
    if sidecar.exists():
        for line in sidecar.open(encoding='utf-8'):
            row = json.loads(line)
            done.setdefault(row['key'], {})[row['episode_id']] = row
    todo = [(t, c) for t in args.tasks for c in args.conditions
            if f'{args.label}/{t}/{c}' not in results['results']]
    if not todo:
        print('nothing to do', flush=True)
        return
    generator = Generator(args)
    results['meta'][args.label] = {
        'model': args.model, 'chat_kwargs': args.chat_kwargs, 'system': args.system,
        'context_chars': args.context_chars, 'extra_new_tokens': args.extra_new_tokens,
        'seed': args.seed, 'per_task': args.per_task}
    log = sidecar.open('a', encoding='utf-8')
    verify = ThreadPoolExecutor(args.verify_workers)
    for task, condition in todo:
        key = f'{args.label}/{task}/{condition}'
        spec = TASKS[task]
        max_new = spec.max_new + args.extra_new_tokens
        episodes = load_episodes(task, args.per_task, args.seed, args.corpora)
        finished = done.setdefault(key, {})
        items = []
        for ep in episodes:
            if ep['episode_id'] in finished:
                continue
            prompt, truncated = build_prompt(task, ep, condition, args.context_chars)
            text = generator.template(prompt)
            items.append((ep, prompt, text, truncated, generator.length(text)))
        items.sort(key=lambda item: -item[4])
        started = time.time()
        index = 0
        while index < len(items):
            longest = items[index][4]
            size = max(1, min(args.batch_size, args.max_batch_tokens // (longest + max_new)))
            chunk = items[index:index + size]
            index += size
            runnable = [item for item in chunk if item[4] <= args.max_prompt_tokens]
            outs = generator.generate_split([item[2] for item in runnable], max_new)
            by_id = {item[0]['episode_id']: out for item, out in zip(runnable, outs)
                     if out is not None}
            checks = verify.map(lambda item: safe_check(by_id[item[0]['episode_id']]['answer'],
                                                        item[0]['verify'])
                                if item[0]['episode_id'] in by_id else False, chunk)
            for (ep, prompt, _, truncated, length), correct in zip(chunk, checks):
                out = by_id.get(ep['episode_id'], {'raw': '', 'answer': '', 'new_tokens': 0,
                                                   'thinking_unfinished': False, 'hit_max': False})
                row = {'key': key, 'episode_id': ep['episode_id'], 'correct': correct,
                       'group': ep['provenance'].get(spec.group) if spec.group else None,
                       'context_truncated': truncated, 'prompt_tokens': length,
                       'prompt_too_long': length > args.max_prompt_tokens,
                       'generation_failed': (length <= args.max_prompt_tokens
                                             and ep['episode_id'] not in by_id),
                       'thinking_unfinished': out['thinking_unfinished'], 'hit_max': out['hit_max'],
                       'new_tokens': out['new_tokens'], 'prompt_tail': prompt[-400:],
                       'answer': out['answer'][:2000],
                       'output': out['raw'][-4000:]}
                finished[ep['episode_id']] = row
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            print(f'{key}: {len(finished)}/{len(episodes)} '
                  f'({time.time() - started:.0f}s)', flush=True)
        rows = [finished[ep['episode_id']] for ep in episodes]
        results['results'][key] = summarize(rows, time.time() - started)
        print(json.dumps({key: {k: results['results'][key][k] for k in ('accuracy', 'ci95', 'n')}}),
              flush=True)
        tmp = args.output.with_suffix('.pending')
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(args.output)
    verify.shutdown()


if __name__ == '__main__':
    main()
