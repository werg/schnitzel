"""Codec distillation (restart plan 3.3a): dense span -> coarser spaces.

The coarser spaces s1..s3 are small size-agnostic projections of the densest
span s0 (``schnitz.bgkit_span.SpaceCodec``, one per space). Trained on the B1 cache:
source = the teacher's s0 reps of a bank passage, target = the teacher's reps of
the same passage at that space's ratio (same count). Losses: cosine to the
teacher reps, and functional - the frozen S2 decoder reading the codec output
reconstructs the passage (NLL) and matches its reading of the teacher reps (KL).

Evaluation per space on held-out bank passages: reconstruct NLL with no context,
full text, teacher reps, codec(teacher s0) and a baseline that mean-pools s0 in
m equal chunks, as captured fractions of the full-text gain. Later the codecs are
applied to the writer's own dense spans. Training-only; runs in ``schnitz-bgkit``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bgkit_reps import (Model, TeacherCache, _batches, _heldout,  # noqa: E402
                              _kl)

from schnitz.bgkit_span import SpaceCodec, interface_rms  # noqa: E402

TARGETS = ('s1', 's2', 's3')


def _example(cache: TeacherCache, model: Model, item, tag: str) -> dict:
    shard, row, _, _, source = item
    ids = model.text_ids(cache.texts[source])
    return {'ids': ids, 'target': ids, 'task': 'reconstruct', 'tag': tag,
            'source': cache.reps(shard, row, 's0').to(model.device).float(),
            'teacher': cache.reps(shard, row, tag).to(model.device).float()}


def _pool(source: torch.Tensor, count: int, norm: float) -> torch.Tensor:
    chunks = torch.tensor_split(source, count)
    return interface_rms(torch.stack([c.mean(0) for c in chunks]), norm)


def train_step(model: Model, codecs, examples, weights: dict) -> dict:
    with model.core.autocast():
        outs = [codecs[ex['tag']](ex['source'], ex['teacher'].shape[0]) for ex in examples]
        cos = 1 - F.cosine_similarity(torch.cat(outs), torch.cat(
            [ex['teacher'] for ex in examples]), dim=-1).mean()
        logits, targets = model.read(examples, outs)
        nll = F.cross_entropy(logits, targets)
        with torch.no_grad():
            t_logits, _ = model.read(examples, [ex['teacher'] for ex in examples])
        kl = _kl(logits, t_logits)
    loss = weights['cos'] * cos + weights['nll'] * nll + weights['kl'] * kl
    loss.backward()
    return {'loss': loss.item(), 'cos': cos.item(), 'nll': nll.item(), 'kl': kl.item()}


@torch.no_grad()
def evaluate(model: Model, codecs, cache: TeacherCache, items, batch_size: int) -> dict:
    out = {}
    for tag in TARGETS:
        sums: dict[str, float] = {}
        tokens = 0
        for start in range(0, len(items), batch_size):
            examples = [_example(cache, model, item, tag) for item in items[start:start + batch_size]]
            with model.core.autocast():
                arms = {
                    'noctx': model.read(examples, None),
                    'full': model.read(examples, None, True),
                    'teacher': model.read(examples, [ex['teacher'] for ex in examples]),
                    'dense_s0': model.read(examples, [ex['source'] for ex in examples]),
                    'codec': model.read(examples, [codecs[tag](ex['source'], ex['teacher'].shape[0])
                                                   for ex in examples]),
                    'pool': model.read(examples, [_pool(ex['source'], ex['teacher'].shape[0],
                                                        model.target_norm) for ex in examples])}
            for name, (logits, targets) in arms.items():
                sums[name] = sums.get(name, 0.0) + F.cross_entropy(
                    logits, targets, reduction='sum').item()
            tokens += int(arms['noctx'][1].numel())
        nll = {name: value / tokens for name, value in sums.items()}
        gain = max(nll['noctx'] - nll['full'], 1e-9)
        out[tag] = {'nll': {k: round(v, 4) for k, v in nll.items()},
                    'captured': {k: round((nll['noctx'] - nll[k]) / gain, 4)
                                 for k in ('dense_s0', 'teacher', 'codec', 'pool')}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-tokens', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--weights', default='cos=1,nll=1,kl=1')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-items', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cuda-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cache = TeacherCache(args.cache, args.sources)
    train = [item for item in cache.items if not _heldout(item[2])]
    heldout = sorted((item for item in cache.items if _heldout(item[2]) and item[3] <= 512),
                     key=lambda item: item[2])[:args.eval_items]
    model = Model(argparse.Namespace(cuda_fraction=args.cuda_fraction, experiment=args.experiment,
                                     checkpoint=args.checkpoint, adapter_rank=0,
                                     gate_open_start=-1, merge_at=-1))
    codecs = torch.nn.ModuleDict({tag: SpaceCodec(1024, model.target_norm) for tag in TARGETS})
    codecs.to(model.device)
    optimizer = torch.optim.AdamW(codecs.parameters(), lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    state_path, step = args.output / 'codecs.pt', 0
    if state_path.exists():
        state = torch.load(state_path, map_location=model.device)
        codecs.load_state_dict(state['codecs'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
    start = step
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + start + 1) / args.warmup))
    (args.output / 'config.json').write_text(json.dumps(dict(
        vars(args), params=sum(p.numel() for p in codecs.parameters()),
        heldout_items=len(heldout)), indent=2, default=str) + '\n')
    metrics = (args.output / 'metrics.jsonl').open('a', encoding='utf-8')

    def log(record):
        metrics.write(json.dumps(record) + '\n')
        metrics.flush()
        print(json.dumps(record), flush=True)

    if step == 0:
        log({'step': 0, 'eval': evaluate(model, codecs, cache, heldout, 32)})
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    window: dict[str, float] = {}
    started = time.time()
    while step < args.steps:
        examples = [_example(cache, model, item, rng.choice(TARGETS)) for item in next(batches)]
        optimizer.zero_grad(set_to_none=True)
        result = train_step(model, codecs, examples, weights)
        torch.nn.utils.clip_grad_norm_(codecs.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        for key, value in result.items():
            window[key] = window.get(key, 0.0) + value
        if step % args.log_every == 0:
            log({'step': step, **{k: round(v / args.log_every, 4) for k, v in window.items()},
                 'elapsed_s': round(time.time() - started)})
            window = {}
        if step % args.eval_every == 0 or step == args.steps:
            torch.save({'codecs': codecs.state_dict(), 'optimizer': optimizer.state_dict(),
                        'step': step}, state_path.with_suffix('.pending'))
            state_path.with_suffix('.pending').replace(state_path)
            log({'step': step, 'eval': evaluate(model, codecs, cache, heldout, 32)})


if __name__ == '__main__':
    main()
