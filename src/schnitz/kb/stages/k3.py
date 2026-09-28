"""K3: superposition-operator warm-up (docs/knowledge-base-stack.md, 5.1 step 5).

For a target record, a neighbourhood of related records (``--neighbors``, the
exact-cosine table of ``scripts/bgkit_neighbors.py``) is encoded by the frozen K1
codecs into items per space. Per space, the superposition operator S_s rewrites
the neighbourhood into an item of the target's own size, conditioned on the
target's key; the recombiner R turns the four rewritten items into the target's
span, which the frozen decoder reads to reconstruct the target text (NLL, KL to
reading the target's own span, a light cosine).

- K3a: S_s sees the target key only. K3b (``--neighbour-keys``): it also sees each
  neighbour's key at each of its positions (continues K3a from its state).
- ``--present`` is the probability that the target itself is in its
  neighbourhood (unmarked), as at read time where the query must select it;
  otherwise the target is dropped (drop-one) and S_s has to reconstruct it from
  related items.
- Keys are ``KeyHeads.item_key`` of the target's items. They are frozen unless
  ``--train-keys`` (K2 trains them for retrieval); a key is a function of the
  target, so the key-only control (the neighbourhood of another target with this
  key) measures how much the key alone carries.

- Depth 2 (``--depth 2``, owner): S_s applied recursively. Each neighbour r of the
  target becomes a level-1 item, S_s over r's own neighbourhood (r itself dropped
  with ``--drop-self``) for r's key; the target's item is S_s at level 2 over those
  level-1 items (the target's own level-1 item included with ``--present``). Each
  level has its own operators; a depth-1 state initializes level 1. The target can
  then also reach level 2 only through its neighbours' neighbourhoods: eval arms
  ``deep_present``, ``deep_via_neighbours`` (target absent at level 2) and
  ``deep_absent`` (target removed everywhere); their difference is what
  superposition carries.

Counts follow the pretraining rule: the target's own size per space and in reps.
Trained: S_s per space, and R with ``--train-recombiner``; codecs and the decoder
are frozen. Evaluation arms: the target's span, K1's reconstruction, K3 with the
target present, drop-one, and the key-only control.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from schnitz.kb.decoder import Model, Neighbours, TeacherCache, _batches, _heldout, frozen_reader
from schnitz.kb.losses import reconstruction_losses
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.stack import SPACES, KeyHeads, Stack, SuperpositionOperator, read_count
from schnitz.kb.stages.k1 import _example
from schnitz.kb_eval import nll_summary


class K3(torch.nn.Module):
    def __init__(self, stack: Stack, state: int, hidden: int, layers: int, checkpointing: bool,
                 query_width: int, depth: int = 1):
        super().__init__()
        self.stack = stack

        def operators():
            return torch.nn.ModuleDict({s: SuperpositionOperator(s, state, hidden, layers,
                                                                 checkpointing)
                                        for s in SPACES})
        self.ops = operators()                              # level 1
        self.deep = operators() if depth >= 2 else None     # level 2
        self.keys = KeyHeads(query_width)

    @torch.no_grad()
    def items(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.stack.encode(span)

    def rewrite(self, target: dict[str, torch.Tensor], neighbourhood: list[dict],
                neighbour_keys: bool, key_from: dict[str, torch.Tensor] | None = None,
                ops: torch.nn.ModuleDict | None = None):
        """S_s per space for the target's key (from ``key_from`` when given: the
        key-only control) and the target's own sizes."""
        ops = self.ops if ops is None else ops
        out = {}
        for s in SPACES:
            key = self.keys.item_key(s, (key_from or target)[s])
            nbrs = [(n[s], 1.0, self.keys.item_key(s, n[s])) for n in neighbourhood]
            count = read_count([], [], torch.ones(0), target=target[s].shape[0])
            out[s] = ops[s](nbrs, key, count, neighbour_keys)[0]
        return out

    def deep_rewrite(self, target: dict[str, torch.Tensor], level1: list[tuple[dict, list]],
                     neighbour_keys: bool, key_from: dict[str, torch.Tensor] | None = None):
        """Depth 2: ``level1`` holds (record items, its neighbourhood) per level-1
        item; each becomes S_s over its neighbourhood for its key, then the target's
        item is S_s over those level-1 items."""
        firsts = [self.rewrite(own, hood, neighbour_keys) for own, hood in level1]
        return self.rewrite(target, firsts, neighbour_keys, key_from, ops=self.deep)


def _deep_inputs(encode, neighbours: Neighbours, item, k: int, rng: random.Random | None,
                 drop_self: float, present: bool, absent: bool = False):
    """The level-1 inputs of a depth-2 rewrite for ``item``: its neighbours (and, if
    ``present``, itself), each with its own neighbourhood; the record itself is left
    out of its neighbourhood with probability ``drop_self``; with ``absent`` the
    target is removed from every neighbourhood."""
    firsts = neighbours.of(item, k) + ([item] if present else [])
    level1 = []
    for r in firsts:
        hood = [o for o in neighbours.of(r, k) if not (absent and o[2] == item[2])]
        if rng is None or rng.random() >= drop_self:
            hood = hood + [r]
        level1.append((encode(r), [encode(o) for o in hood] or [encode(r)]))
    return level1


def _build(k3: K3, neighbours: Neighbours, cache: TeacherCache, model: Model, items, k: int,
           rng: random.Random, p_present: float):
    examples = [_example(cache, model, item) for item in items]
    targets = [k3.items(ex['span']) for ex in examples]
    hoods = []
    for item, target in zip(items, targets):
        hood = [k3.items(_example(cache, model, other)['span'])
                for other in neighbours.of(item, k)]
        if rng.random() < p_present or not hood:
            hood.insert(rng.randrange(len(hood) + 1), target)
        hoods.append(hood)
    return examples, targets, hoods


def _build_deep(k3: K3, neighbours: Neighbours, cache: TeacherCache, model: Model, items,
                k: int, rng: random.Random, p_present: float, drop_self: float):
    memo: dict[str, dict] = {}

    def encode(item):
        if item[2] not in memo:
            memo[item[2]] = k3.items(_example(cache, model, item)['span'])
        return memo[item[2]]
    examples = [_example(cache, model, item) for item in items]
    targets = [encode(item) for item in items]
    level1 = [_deep_inputs(encode, neighbours, item, k, rng, drop_self,
                           present=rng.random() < p_present) for item in items]
    return examples, targets, level1


def train_step(model: Model, k3: K3, examples, targets, hoods, weights: dict,
               neighbour_keys: bool, deep: bool = False) -> dict:
    all_on = {s: 1.0 for s in SPACES}
    rewrite = k3.deep_rewrite if deep else k3.rewrite
    with model.core.autocast():
        outs = [k3.stack.decode(rewrite(t, h, neighbour_keys), all_on, ex['span'].shape[0])
                for ex, t, h in zip(examples, targets, hoods)]
        loss, parts = reconstruction_losses(model, k3.stack, examples, outs, weights)
    loss.backward()
    return {'loss': loss.item(), **parts}


@torch.no_grad()
def evaluate(model: Model, k3: K3, neighbours: Neighbours, cache: TeacherCache, items,
             k: int, neighbour_keys: bool, batch_size: int = 8) -> dict:
    all_on = {s: 1.0 for s in SPACES}
    arms = ['span', 'k1', 'present', 'dropone', 'key_only', 'span_shuffled', 'present_shuffled']
    deep = k3.deep is not None
    if deep:
        arms += ['deep_present', 'deep_via_neighbours', 'deep_absent', 'deep_shuffled']
    sums: dict[str, float] = {}
    tokens = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        examples = [_example(cache, model, item) for item in batch]
        targets = [k3.items(ex['span']) for ex in examples]
        related = [[k3.items(_example(cache, model, o)['span']) for o in neighbours.of(item, k)]
                   for item in batch]
        counts = [ex['span'].shape[0] for ex in examples]
        with model.core.autocast():
            def decode(parts):
                return [k3.stack.decode(p, all_on, n) for p, n in zip(parts, counts)]
            spans = {'span': [ex['span'] for ex in examples],
                     'k1': decode(targets),
                     'present': decode([k3.rewrite(t, r + [t], neighbour_keys)
                                        for t, r in zip(targets, related)]),
                     'dropone': decode([k3.rewrite(t, r or [t], neighbour_keys)
                                        for t, r in zip(targets, related)]),
                     # another target's neighbourhood (with that target) under this key
                     'key_only': decode([k3.rewrite(targets[(i + 1) % len(targets)],
                                                    related[(i + 1) % len(targets)]
                                                    + [targets[(i + 1) % len(targets)]],
                                                    neighbour_keys, key_from=t)
                                         for i, t in enumerate(targets)])}
            spans['span_shuffled'] = spans['span'][1:] + spans['span'][:1]
            spans['present_shuffled'] = spans['present'][1:] + spans['present'][:1]
            if deep:
                memo: dict[str, dict] = {}

                def encode(item):
                    if item[2] not in memo:
                        memo[item[2]] = k3.items(_example(cache, model, item)['span'])
                    return memo[item[2]]
                for name, present, absent in (('deep_present', True, False),
                                              ('deep_via_neighbours', False, False),
                                              ('deep_absent', False, True)):
                    spans[name] = decode([k3.deep_rewrite(
                        t, _deep_inputs(encode, neighbours, item, k, None, 0.0, present, absent),
                        neighbour_keys) for t, item in zip(targets, batch)])
                spans['deep_shuffled'] = spans['deep_present'][1:] + spans['deep_present'][:1]
            reads = {'noctx': model.read(examples, None), 'full': model.read(examples, None, True)}
            reads.update({name: model.read(examples, spans[name]) for name in arms})
        for name, (logits, tgt) in reads.items():
            sums[name] = sums.get(name, 0.0) + F.cross_entropy(logits, tgt, reduction='sum').item()
        tokens += int(reads['noctx'][1].numel())
    shuffled = {'span': 'span_shuffled', 'present': 'present_shuffled'}
    if deep:
        shuffled['deep_present'] = 'deep_shuffled'
    out = nll_summary(sums, tokens, arms, shuffled, gain=False)
    if deep:  # what reaches the target only through its neighbours' superposed items
        out['content_nats']['through_superposition'] = round(
            out['nll']['deep_absent'] - out['nll']['deep_via_neighbours'], 4)
    # what the neighbourhood adds beyond the key: another target's neighbourhood
    # under this target's key, against this target's own neighbourhood
    out['content_nats']['neighbourhood'] = round(out['nll']['key_only'] - out['nll']['present'], 4)
    return out


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--reader-state', type=Path, help='writer-stage state whose decoder reads')
    parser.add_argument('--stack-state', type=Path, required=True, help='K1 stack.pt (codecs, R)')
    parser.add_argument('--init-state', type=Path, help='K3a k3.pt to continue from (K3b)')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--neighbors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--neighbourhood', type=int, default=4)
    parser.add_argument('--present', type=float, default=0.5)
    parser.add_argument('--neighbour-keys', action='store_true', help='K3b')
    parser.add_argument('--depth', type=int, default=1, choices=(1, 2),
                        help='2: recursive S_s (level-1 items over neighbourhoods, then level 2)')
    parser.add_argument('--drop-self', type=float, default=0.3,
                        help='depth 2: chance a level-1 record is left out of its own neighbourhood')
    parser.add_argument('--train-keys', action='store_true')
    parser.add_argument('--train-recombiner', action='store_true')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-tokens', type=int, default=2048)
    parser.add_argument('--state', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--weights', default='cos=0.1,nll=1,kl=1')
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-items', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cuda-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)


def run(args) -> None:
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cache = TeacherCache(args.cache, args.sources)
    neighbours = Neighbours(args.neighbors, cache)
    train = [item for item in cache.items if not _heldout(item[2]) and item[2] in neighbours.row]
    heldout = sorted((item for item in cache.items if _heldout(item[2]) and item[3] <= 512
                      and item[2] in neighbours.row), key=lambda item: item[2])[:args.eval_items]
    model = frozen_reader(args.checkpoint, args.experiment, args.reader_state, args.cuda_fraction)
    stack = Stack(model.target_norm, args.state, args.hidden, args.layers, args.checkpointing)
    stack.load_state_dict(torch.load(args.stack_state, map_location='cpu')['stack'])
    k3 = K3(stack, args.state, args.hidden, args.layers, args.checkpointing,
            model.writer.marker.shape[0], args.depth).to(model.device)
    for p in stack.codecs.parameters():
        p.requires_grad_(False)
    for p in stack.recombiner.parameters():
        p.requires_grad_(args.train_recombiner)
    for p in k3.keys.parameters():
        p.requires_grad_(args.train_keys)
    out = Run(args.output)
    state, step = out.load('k3.pt', model.device), 0
    if state is not None:
        k3.load_state_dict(state['k3'])
        step = state['step']
    elif args.init_state:  # a depth-1 state initializes level 1 (and the keys) of depth 2
        missing, unexpected = k3.load_state_dict(
            torch.load(args.init_state, map_location=model.device)['k3'], strict=False)
        if unexpected or any(not name.startswith('deep.') for name in missing):
            raise ValueError(f'--init-state does not fit: missing {missing[:5]}, '
                             f'unexpected {unexpected[:5]}')
    params = [p for p in k3.parameters() if p.requires_grad]
    optimizer, schedule = warmup_optimizer(params, args.warmup, step, lr=args.lr)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    out.write_config(dict(vars(args), spaces=SPACES, heldout_items=len(heldout),
                          params=sum(p.numel() for p in params)))

    def run_eval():
        return evaluate(model, k3, neighbours, cache, heldout, args.neighbourhood,
                        args.neighbour_keys)

    if step == 0:
        out.log({'step': 0, 'eval': run_eval()})
    window = Window()
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    while step < args.steps:
        if args.depth >= 2:
            examples, targets, hoods = _build_deep(k3, neighbours, cache, model, next(batches),
                                                   args.neighbourhood, rng, args.present,
                                                   args.drop_self)
        else:
            examples, targets, hoods = _build(k3, neighbours, cache, model, next(batches),
                                              args.neighbourhood, rng, args.present)
        optimizer.zero_grad(set_to_none=True)
        result = train_step(model, k3, examples, targets, hoods, weights, args.neighbour_keys,
                            deep=args.depth >= 2)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        window.add(result)
        if step % args.log_every == 0:
            out.log({'step': step, **window.means(), 'elapsed_s': out.elapsed()})
        if step % args.eval_every == 0 or step == args.steps:
            out.save('k3.pt', {'k3': k3.state_dict(), 'optimizer': optimizer.state_dict(),
                               'step': step})
            out.log({'step': step, 'eval': run_eval()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    run(parser.parse_args())

