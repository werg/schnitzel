"""K1: autoencoding through the knowledge-base spaces (docs/knowledge-base-stack.md, section 5).

A record's span x (n reps in the decoder's input space) goes through one forward
codec per space into items of m_s = ceil(r_s · n) positions of width d_s; the
recombiner reads the items of all spaces and produces an n-rep span. All
codecs and the recombiner are MLP-matrix operators (``schnitz.mlp_matrix``).
The spaces together hold about as many values as the span; each alone is a
bottleneck.

Losses: the frozen decoder reads the recombined span and reconstructs the
record's text (NLL), with a KL to reading x itself, plus a light cosine to x.
Space dropout: each space's item is removed (gate 0, exact) with probability
``--space-dropout``, at least one space kept, so every space carries part of the
content and the recombiner works with spaces missing.

Inputs: the B1 teacher cache (S2 spans at the writer's densest ratio level s0)
until the B3 writer's own spans are generated. The reader is the B3 decoder from
``--reader-state`` (frozen), else S2.

Evaluation on held-out bank passages, as captured fractions of the full-text
gain and content nats over a shuffled control: the span itself, all spaces,
each space dropped, each space alone, and the recombined span of another
passage's items (shuffled). Training-only; runs in ``schnitz-bgkit``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from schnitz.kb_eval import nll_summary
from schnitz.kb.decoder import Model, TeacherCache, frozen_reader, _batches, _heldout
from schnitz.kb.losses import reconstruction_losses
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.stack import SPACES, Stack


def _example(cache: TeacherCache, model: Model, item) -> dict:
    shard, row, _, _, source = item
    ids = model.text_ids(cache.texts[source])
    return {'ids': ids, 'target': ids, 'task': 'reconstruct',
            'span': cache.reps(shard, row, 's0').to(model.device).float()}


def _keep(rng: random.Random, p: float) -> dict[str, float]:
    keep = {name: 0.0 if rng.random() < p else 1.0 for name in SPACES}
    if not any(keep.values()):
        keep[rng.choice(list(SPACES))] = 1.0
    return keep


def train_step(model: Model, stack: Stack, examples, weights: dict, rng: random.Random,
               dropout: float) -> dict:
    with model.core.autocast():
        outs = [stack.decode(stack.encode(ex['span']), _keep(rng, dropout), ex['span'].shape[0])
                for ex in examples]
        loss, parts = reconstruction_losses(model, stack, examples, outs, weights)
    loss.backward()
    return {'loss': loss.item(), **parts}


@torch.no_grad()
def evaluate(model: Model, stack: Stack, cache: TeacherCache, items, batch_size: int) -> dict:
    all_on = {name: 1.0 for name in SPACES}
    arms = ['span', 'stack', 'stack_shuffled', 'span_shuffled'] + \
        [f'without_{s}' for s in SPACES] + [f'only_{s}' for s in SPACES]
    sums: dict[str, float] = {}
    tokens = 0
    for start in range(0, len(items), batch_size):
        examples = [_example(cache, model, item) for item in items[start:start + batch_size]]
        with model.core.autocast():
            encoded = [stack.encode(ex['span']) for ex in examples]
            counts = [ex['span'].shape[0] for ex in examples]
            spans = {'span': [ex['span'] for ex in examples],
                     'stack': [stack.decode(e, all_on, n) for e, n in zip(encoded, counts)],
                     # the next passage's items decoded at this passage's length
                     'stack_shuffled': [stack.decode(e, all_on, n) for e, n in
                                        zip(encoded[1:] + encoded[:1], counts)]}
            spans['span_shuffled'] = spans['span'][1:] + spans['span'][:1]
            for s in SPACES:
                without = {name: float(name != s) for name in SPACES}
                only = {name: float(name == s) for name in SPACES}
                spans[f'without_{s}'] = [stack.decode(e, without, n) for e, n in zip(encoded, counts)]
                spans[f'only_{s}'] = [stack.decode(e, only, n) for e, n in zip(encoded, counts)]
            reads = {'noctx': model.read(examples, None), 'full': model.read(examples, None, True)}
            reads.update({name: model.read(examples, spans[name]) for name in arms})
        for name, (logits, targets) in reads.items():
            sums[name] = sums.get(name, 0.0) + F.cross_entropy(logits, targets, reduction='sum').item()
        tokens += int(reads['noctx'][1].numel())
    return nll_summary(sums, tokens, arms,
                       {'span': 'span_shuffled', 'stack': 'stack_shuffled'}, gain=False)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--reader-state', type=Path,
                        help='B3 writer.pt whose merged decoder reads (frozen); default S2')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-tokens', type=int, default=4096)
    parser.add_argument('--state', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--space-dropout', type=float, default=0.25)
    parser.add_argument('--checkpointing', action='store_true',
                        help='recompute operator layers in backward')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--weights', default='cos=0.1,nll=1,kl=1')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-items', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cuda-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)


def run(args) -> None:
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cache = TeacherCache(args.cache, args.sources)
    train = [item for item in cache.items if not _heldout(item[2])]
    heldout = sorted((item for item in cache.items if _heldout(item[2]) and item[3] <= 512),
                     key=lambda item: item[2])[:args.eval_items]
    model = frozen_reader(args.checkpoint, args.experiment, args.reader_state,
                          args.cuda_fraction)
    stack = Stack(model.target_norm, args.state, args.hidden, args.layers, args.checkpointing)
    stack.to(model.device)
    out = Run(args.output)
    state, step = out.load('stack.pt', model.device), 0
    if state is not None:
        stack.load_state_dict(state['stack'])
        step = state['step']
    else:  # corpus statistics for the stack's standardized space
        sample = random.Random(1).sample(train, min(len(train), 2000))
        stack.set_statistics([cache.reps(item[0], item[1], 's0') for item in sample])
    optimizer, schedule = warmup_optimizer(stack.parameters(), args.warmup, step, lr=args.lr)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    out.write_config(dict(vars(args), spaces=SPACES,
                          params=sum(p.numel() for p in stack.parameters()),
                          heldout_items=len(heldout)))
    log = out.log

    if step == 0:
        log({'step': 0, 'eval': evaluate(model, stack, cache, heldout, 16)})
    window = Window()
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    while step < args.steps:
        examples = [_example(cache, model, item) for item in next(batches)]
        optimizer.zero_grad(set_to_none=True)
        result = train_step(model, stack, examples, weights, rng, args.space_dropout)
        torch.nn.utils.clip_grad_norm_(stack.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        window.add(result)
        if step % args.log_every == 0:
            log({'step': step, **window.means(), 'elapsed_s': out.elapsed()})
        if step % args.eval_every == 0 or step == args.steps:
            out.save('stack.pt', {'stack': stack.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'step': step})
            log({'step': step, 'eval': evaluate(model, stack, cache, heldout, 16)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    run(parser.parse_args())
