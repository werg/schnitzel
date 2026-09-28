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

from schnitz.kb.decoder import Model, TeacherCache, _batches, _heldout, frozen_reader
from schnitz.kb.losses import reconstruction_losses
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.stack import SPACES, KeyHeads, Stack, SuperpositionOperator, read_count
from schnitz.kb.stages.k1 import _example
from schnitz.kb_eval import nll_summary


class Neighbours:
    def __init__(self, path: Path, cache: TeacherCache):
        data = torch.load(path, weights_only=False)
        self.table, ids = data['neighbors'], data['record_ids']
        index = {record_id: i for i, record_id in enumerate(ids)}
        by_id = {item[2]: item for item in cache.items}
        self.items = [by_id.get(record_id) for record_id in ids]
        self.row = {record_id: index[record_id] for record_id in by_id if record_id in index}

    def of(self, item, k: int) -> list:
        row = self.row.get(item[2])
        if row is None:
            return []
        return [self.items[j] for j in self.table[row, :k].tolist() if self.items[j] is not None]


class K3(torch.nn.Module):
    def __init__(self, stack: Stack, state: int, hidden: int, layers: int, checkpointing: bool,
                 query_width: int):
        super().__init__()
        self.stack = stack
        self.ops = torch.nn.ModuleDict({s: SuperpositionOperator(s, state, hidden, layers,
                                                                 checkpointing)
                                        for s in SPACES})
        self.keys = KeyHeads(query_width)

    @torch.no_grad()
    def items(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.stack.encode(span)

    def rewrite(self, target: dict[str, torch.Tensor], neighbourhood: list[dict],
                neighbour_keys: bool, key_from: dict[str, torch.Tensor] | None = None):
        """S_s per space for the target's key (from ``key_from`` when given: the
        key-only control) and the target's own sizes."""
        out = {}
        for s in SPACES:
            key = self.keys.item_key(s, (key_from or target)[s])
            nbrs = [(n[s], 1.0, self.keys.item_key(s, n[s])) for n in neighbourhood]
            count = read_count([], [], torch.ones(0), target=target[s].shape[0])
            out[s] = self.ops[s](nbrs, key, count, neighbour_keys)[0]
        return out


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


def train_step(model: Model, k3: K3, examples, targets, hoods, weights: dict,
               neighbour_keys: bool) -> dict:
    all_on = {s: 1.0 for s in SPACES}
    with model.core.autocast():
        outs = [k3.stack.decode(k3.rewrite(t, h, neighbour_keys), all_on, ex['span'].shape[0])
                for ex, t, h in zip(examples, targets, hoods)]
        loss, parts = reconstruction_losses(model, k3.stack, examples, outs, weights)
    loss.backward()
    return {'loss': loss.item(), **parts}


@torch.no_grad()
def evaluate(model: Model, k3: K3, neighbours: Neighbours, cache: TeacherCache, items,
             k: int, neighbour_keys: bool, batch_size: int = 8) -> dict:
    all_on = {s: 1.0 for s in SPACES}
    arms = ['span', 'k1', 'present', 'dropone', 'key_only', 'span_shuffled', 'present_shuffled']
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
            reads = {'noctx': model.read(examples, None), 'full': model.read(examples, None, True)}
            reads.update({name: model.read(examples, spans[name]) for name in arms})
        for name, (logits, tgt) in reads.items():
            sums[name] = sums.get(name, 0.0) + F.cross_entropy(logits, tgt, reduction='sum').item()
        tokens += int(reads['noctx'][1].numel())
    out = nll_summary(sums, tokens, arms,
                      {'span': 'span_shuffled', 'present': 'present_shuffled'}, gain=False)
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
            model.writer.marker.shape[0]).to(model.device)
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
    elif args.init_state:
        k3.load_state_dict(torch.load(args.init_state, map_location=model.device)['k3'])
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
        examples, targets, hoods = _build(k3, neighbours, cache, model, next(batches),
                                          args.neighbourhood, rng, args.present)
        optimizer.zero_grad(set_to_none=True)
        result = train_step(model, k3, examples, targets, hoods, weights, args.neighbour_keys)
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

