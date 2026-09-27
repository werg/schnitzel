"""Single-pass baselines of S2 (and its base LFM2.5-350M) on candidate B9 tasks.

Greedy generation, scored with ``sdkb.task_verifiers``, with and without the
reference information a knowledge base would supply (tool documentation, database
schema, BIRD evidence, a worked example). Picks stretch-goal slices: low but
nonzero without the reference, clearly higher with it. Runs in ``sdkb-bgkit``.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import random
import sqlite3
import time

import torch

from sdkb.task_verifiers import call_match, code_match, knights_knaves_match, sql_match

AGENTIC = Path('/archive/raw/agentic-20260927')
WORLDS = Path('/archive/raw/worlds-20260927')


def _lit(value):
    """A list field that some releases store as its Python literal string."""
    return __import__('ast').literal_eval(value) if isinstance(value, str) else value


def _schema(db: Path, limit: int = 6000) -> str:
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    text = '\n'.join(sql for (sql,) in con.execute(
        "select sql from sqlite_master where type='table' and sql is not null"))
    con.close()
    return text[:limit]


def _sql(out, gold, db, timeout):
    return sql_match(out, gold, db, timeout)


def _knights(out, names, solution):
    return knights_knaves_match(out, names, solution)


def _calls(out, gold):
    return call_match(out, gold)


def _code(out, test, style):
    return code_match(out, test, style)


def xlam(n: int, rng: random.Random):
    rows = json.load((AGENTIC / 'xlam-function-calling-60k/xlam_function_calling_60k.json').open())
    for row in rng.sample(rows, n):
        prompt = (f'You can call these tools:\n{row["tools"]}\n\nAnswer the request with a JSON '
                  'list of calls [{"name": ..., "arguments": {...}}] and nothing else.\n\n'
                  f'Request: {row["query"]}')
        gold = json.loads(row['answers'])
        yield 'with_docs', prompt, partial(_calls, gold=gold)


def spider(n: int, rng: random.Random):
    base = AGENTIC / 'spider/official/spider_data'
    rows = json.load((base / 'dev.json').open())
    for row in rng.sample(rows, n):
        db = base / 'database' / row['db_id'] / f'{row["db_id"]}.sqlite'
        check = partial(_sql, gold=row['query'], db=db, timeout=10)
        ask = f'Write one SQLite query that answers: {row["question"]}\nReturn only the SQL.'
        yield 'schema', f'Database schema:\n{_schema(db)}\n\n{ask}', check
        yield 'no_schema', f'Database: {row["db_id"]}\n\n{ask}', check


def bird(n: int, rng: random.Random):
    base = AGENTIC / 'bird/dev_20240627'
    rows = json.load((base / 'dev.json').open())
    for row in rng.sample(rows, n):
        db = base / 'dev_databases' / row['db_id'] / f'{row["db_id"]}.sqlite'
        check = partial(_sql, gold=row['SQL'], db=db, timeout=20)
        ask = f'Write one SQLite query that answers: {row["question"]}\nReturn only the SQL.'
        schema = f'Database schema:\n{_schema(db)}\n\n'
        yield f'schema_evidence/{row["difficulty"]}', f'{schema}Hint: {row["evidence"]}\n\n{ask}', check
        yield f'schema/{row["difficulty"]}', schema + ask, check


def kodcode(n: int, rng: random.Random):
    import pyarrow.parquet as pq
    table = pq.read_table(AGENTIC / 'kodcode-v1/data/train-00001-of-00015.parquet',
                          columns=['subset', 'style', 'question', 'test', 'gpt_difficulty'])
    rows = [r for r in table.to_pylist() if r['subset'] not in ('Package', 'Docs')
            and r['style'] in ('instruct', 'online_judge')]
    for level in ('easy', 'medium'):
        for row in rng.sample([r for r in rows if r['gpt_difficulty'] == level], n):
            how = ('Write a Python program that reads from stdin and writes to stdout.'
                   if row['style'] == 'online_judge' else 'Write the Python function.')
            prompt = f'{row["question"]}\n\n{how} Put the code in one ```python block.'
            yield level, prompt, partial(_code, test=row['test'], style=row['style'])


def knights(n: int, rng: random.Random):
    for people in (3, 4, 5):
        test = [json.loads(line) for line in
                (WORLDS / f'knights-and-knaves/test/people{people}_num100.jsonl').open()]
        train = [json.loads(line) for line in
                 (WORLDS / f'knights-and-knaves/train/people{people}_num1000.jsonl').open()]
        for row in rng.sample(test, min(n, len(test))):
            names, solution = _lit(row['names']), _lit(row['solution'])
            fmt = 'Answer with one line per person: "(1) Name is a knight/knave".'
            check = partial(_knights, names=names, solution=solution)
            yield f'plain/{people}', f'{row["quiz"]}\n\n{fmt}', check
            ex = rng.choice(train)
            steps = '\n'.join(_lit(ex['cot_repeat_steps']))
            worked = (f'Example puzzle:\n{ex["quiz"]}\n{ex["cot_head"]}\n{steps}\n{ex["cot_foot"]}\n'
                      f'{ex["solution_text_format"]}\n\n')
            yield f'worked_example/{people}', f'{worked}Now solve:\n{row["quiz"]}\n\n{fmt}', check


TASKS = {'xlam': (xlam, 320), 'spider': (spider, 256), 'bird': (bird, 320), 'kodcode': (kodcode, 768),
         'knights': (knights, 640)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--tasks', nargs='+', default=list(TASKS))
    parser.add_argument('--per-task', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--models', nargs='+', default=['s2', 'base'])
    parser.add_argument('--cuda-fraction', type=float, default=0.12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from bgkit_core.host_memory_guard import cap_cuda
    cap_cuda(args.cuda_fraction)
    from bgkit2.data.templates import Templates
    from bgkit2.training.standalone import load_models
    core = load_models(args.experiment, str(args.checkpoint))
    del core.encoder
    torch.cuda.empty_cache()
    tok = core.tok
    Templates.from_tokenizer(tok, style='chat')  # the chat template S2 was trained with
    tok.padding_side = 'left'
    results = json.loads(args.output.read_text()) if args.output.exists() else {}

    def generate(prompts: list[str], max_new: int) -> list[str]:
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        batch = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(core.device)
        with torch.no_grad(), core.autocast():
            out = core.decoder.lm.generate(**batch, max_new_tokens=max_new, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)

    for task in args.tasks:
        build, max_new = TASKS[task]
        items = list(build(args.per_task, random.Random(0)))
        for model in args.models:
            key = f'{task}/{model}'
            if key in results:
                continue
            started, scores, samples = time.time(), {}, {}
            for start in range(0, len(items), args.batch_size):
                chunk = items[start:start + args.batch_size]
                if model == 'base':
                    with core.decoder.disable_adapter():
                        outs = generate([p for _, p, _ in chunk], max_new)
                else:
                    outs = generate([p for _, p, _ in chunk], max_new)
                for (condition, prompt, check), out in zip(chunk, outs):
                    scores.setdefault(condition, []).append(bool(check(out)))
                    samples.setdefault(condition, [prompt[-300:], out[:500]])
            results[key] = {'accuracy': {c: round(sum(v) / len(v), 3) for c, v in scores.items()},
                            'n': {c: len(v) for c, v in scores.items()},
                            'seconds': round(time.time() - started), 'samples': samples}
            print(json.dumps({key: results[key]['accuracy'], 'n': results[key]['n']}), flush=True)
            args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
