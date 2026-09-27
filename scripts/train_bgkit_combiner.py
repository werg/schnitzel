"""B7 combiner training: gated reads of several records -> one BGKit span.

One combiner per space (``SpaceCodec.combine``; s1..s3 initialized from the
trained codecs, s0 from the s1 codec). Per R6 episode and a random space s:

- target: the S2 teacher encoding of the episode's gold texts joined in order
  (``cache_bgkit_teacher.py --episodes``), m reps;
- input records (each its B1 teacher reps in space s): the gold records, up to
  ``--related`` semantically related records (``bgkit_neighbors.py``), and the
  target itself as a *feedback* record; golds keep their order (the target
  encodes them in order), related and feedback records go to random positions;
- gates: feedback on the self-record curriculum (owner direction, 27 September):
  1 until ``--feedback-hold``, then linearly to 0 at ``--feedback-end``, so the
  combiner moves from copying the target to building it from the actual
  sources. Gold 1 and related ``--related-gate`` (fudged) until
  ``--learned-gates-from``; then a gate head scores every record against the
  question (frozen S2 decoder state), trained by the task loss and a BCE to the
  gold labels.

Losses: cosine to the target reps; functional through the frozen S2 decoder
reading the combiner span: reconstruction of the joined gold texts and the
episode's answer to its question (NLL), each with KL to reading the teacher span.
The teacher is weak at QA (question-free encodings), so once the feedback record
is gone the distillation terms (cosine and both KLs) decay to ``--distill-floor``
of their weight over ``--distill-decay`` steps and the task NLLs drive the
combiner past its teacher.

``--writer-state`` (a B3 ``writer.pt``, adapter included) replaces the cached
teacher reps of every input record by the writer's own free-running spans
(``--writer-passes`` rollout passes on its own reps); with ``--writer-train`` the
writer trains too, so gradients run writer -> combiner -> reader in one graph,
the writer keeping its cosine and stop losses as an anchor (``writer`` weight).

Evaluation on validation episodes (gates at the current policy, no feedback
record): answer and reconstruction NLL for no context, full gold text, teacher
span, the gold records' teacher spans concatenated, the combiner over golds, over
golds + related, and over related only (gold removed), as captured fractions of
the full-text gain. Training-only; runs in ``sdkb-bgkit``.
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bgkit_reps import (SPACES, Model, TeacherCache, _example, _kl,  # noqa: E402
                              _rollout)

from sdkb.bgkit_span import SpaceCodec  # noqa: E402


class GateHead(nn.Module):
    """Relevance of a record (mean of its reps) to the question state, in (0, 1)."""

    def __init__(self, width: int, inner: int = 256):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner))
        self.record = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, records: list[torch.Tensor]) -> torch.Tensor:
        summary = torch.stack([r.float().mean(0) for r in records])
        q = self.query(query.float())
        return torch.sigmoid(self.record(summary) @ q / q.shape[-1] ** 0.5 + self.bias)


class Episodes:
    def __init__(self, path: Path, cache: Path, sources: TeacherCache):
        self.rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        texts = {record_id: sources.texts[source] for _, _, record_id, _, source in sources.items}
        joined = ['\n\n'.join(texts[r] for r in row['required_ids']) for row in self.rows]
        self.cache = TeacherCache(cache, None, texts=joined)
        self.by_id = {item[2]: item for item in self.cache.items}
        self.rows = [row for row in self.rows if row['episode_id'] in self.by_id]


class Combiner:
    def __init__(self, args, model: Model, bank: TeacherCache):
        self.model, self.bank, self.args = model, bank, args
        index = torch.load(args.neighbors, weights_only=False)
        self.neighbors = index['neighbors']
        self.bank_items = {item[2]: item for item in bank.items}
        self.bank_index = {record_id: i for i, record_id in enumerate(index['record_ids'])}
        self.bank_ids = index['record_ids']
        self.combiners = nn.ModuleDict({tag: SpaceCodec(1024, model.target_norm) for tag in SPACES})
        if args.init_codecs:
            flat = torch.load(args.init_codecs, map_location='cpu')['codecs']
            for tag in SPACES:
                source = tag if any(k.startswith(f'{tag}.') for k in flat) else 's1'
                state = {k[len(source) + 1:]: v for k, v in flat.items()
                         if k.startswith(f'{source}.')}
                missing, unexpected = self.combiners[tag].load_state_dict(state, strict=False)
                if unexpected or any(not k.startswith('key_record') for k in missing):
                    raise ValueError(f'codec init mismatch: {missing} {unexpected}')
        self.gate_head = GateHead(1024)
        self.combiners.to(model.device)
        self.gate_head.to(model.device)
        from bgkit2.data.templates import decoder_sentinel
        self.sentinel = decoder_sentinel(model.tok)

    def parameters(self):
        return list(self.combiners.parameters()) + list(self.gate_head.parameters())

    def question_instr(self, query: str) -> torch.Tensor:
        sent_str, sent_id = self.sentinel
        ids = self.model.tok.apply_chat_template(
            [{'role': 'user', 'content': f'Context:\n{sent_str}\n{query}'}],
            add_generation_prompt=True, tokenize=True)
        if hasattr(ids, 'input_ids'):
            ids = ids['input_ids']
        return torch.tensor(ids[ids.index(sent_id) + 1:])

    def record(self, record_id: str, tag: str) -> torch.Tensor:
        shard, row, *_ = self.bank_items[record_id]
        return self.bank.reps(shard, row, tag).to(self.model.device).float()

    @torch.no_grad()
    def query_state(self, queries: list[str]) -> torch.Tensor:
        dec, states = self.model.decoder, []
        for query in queries:
            ids = torch.tensor(self.model.tok(query, add_special_tokens=False)['input_ids'][:256],
                               device=self.model.device)
            with self.model.core.autocast():
                hidden = dec._hidden(inputs_embeds=dec.embed(ids)[None],
                                     attention_mask=torch.ones(1, ids.shape[0], dtype=torch.long,
                                                               device=ids.device))
            states.append(hidden[0].float().mean(0))
        return torch.stack(states)

    def build(self, episodes: Episodes, row: dict, tag: str, rng: random.Random, *,
              related: int, feedback: float, gold: bool = True) -> dict:
        shard, erow, _, _, index = episodes.by_id[row['episode_id']]
        target = episodes.cache.reps(shard, erow, tag).to(self.model.device).float()
        golds = row['required_ids']
        pool = []
        for record_id in golds:
            pool.extend(self.bank_ids[j] for j in self.neighbors[self.bank_index[record_id]].tolist())
        pool = [r for r in dict.fromkeys(pool) if r not in golds]
        chosen = rng.sample(pool, min(related, len(pool)))
        # golds keep their order (the target encodes them in order); related records
        # and the feedback record go to random positions among them
        entries = [('gold', r) for r in golds] if gold else []
        extra = [('related', r) for r in chosen] + ([('feedback', None)] if feedback > 0 else [])
        for entry in extra:
            entries.insert(rng.randint(0, len(entries)), entry)
        records = [target if kind == 'feedback' else self.record(r, tag) for kind, r in entries]
        record_ids = [r for _, r in entries]
        ids = self.model.text_ids(episodes.cache.texts[index])
        answer = torch.tensor(self.model.tok(' ' + row['answer'].strip(),
                                             add_special_tokens=False)['input_ids'][:64])
        return {'tag': tag, 'kinds': [kind for kind, _ in entries], 'records': records,
                'record_ids': record_ids,
                'target_reps': target, 'feedback': feedback, 'query': row['query'],
                'recon': {'ids': ids, 'target': ids, 'task': 'reconstruct'},
                'qa': {'ids': ids, 'target': answer, 'task': 'reconstruct',
                       'instr': self.question_instr(row['query'])}}

    def own_spans(self, examples, train: bool):
        """Replace every source record by the writer's own free-running span of it."""
        slots = [(i, j) for i, ex in enumerate(examples)
                 for j, record_id in enumerate(ex['record_ids']) if record_id is not None]
        if not slots or not self.args.writer_state:
            return None
        writes = [_example(self.bank, self.model, self.bank_items[examples[i]['record_ids'][j]],
                           examples[i]['tag']) for i, j in slots]
        with torch.set_grad_enabled(train):
            _, preds, cos, stop = _rollout(self.model, writes, self.args.writer_passes, 1.0)
        for (i, j), pred in zip(slots, preds):
            examples[i]['records'][j] = pred if train else pred.detach()
        return cos + 0.2 * stop if train else None

    def gates(self, examples, learned: bool, related_gate: float):
        if learned:
            states = self.query_state([ex['query'] for ex in examples])
        out, bce = [], []
        for i, ex in enumerate(examples):
            fixed = torch.tensor([{'gold': 1.0, 'related': related_gate,
                                   'feedback': ex['feedback']}[k] for k in ex['kinds']],
                                 device=self.model.device)
            if learned:
                sources = [j for j, k in enumerate(ex['kinds']) if k != 'feedback']
                scored = self.gate_head(states[i], [ex['records'][j] for j in sources])
                labels = torch.tensor([ex['kinds'][j] == 'gold' for j in sources],
                                      dtype=torch.float, device=scored.device)
                prob = scored.float().clamp(1e-5, 1 - 1e-5)
                bce.append(-(labels * prob.log() + (1 - labels) * (1 - prob).log()).mean())
                fixed = fixed.clone()
                fixed[sources] = scored.float()
            out.append(fixed)
        return out, (torch.stack(bce).mean() if bce else None)

    def combine(self, ex: dict, gates: torch.Tensor) -> torch.Tensor:
        return self.combiners[ex['tag']].combine(ex['records'], ex['target_reps'].shape[0], gates)


def _feedback(args, step: int) -> float:
    if step < args.feedback_hold:
        return 1.0
    return max(0.0, 1.0 - (step - args.feedback_hold) / max(args.feedback_end - args.feedback_hold, 1))


def _distill(args, step: int) -> float:
    """Weight factor of the distillation terms: 1 until the feedback record is gone,
    then linearly down to ``--distill-floor`` over ``--distill-decay`` steps."""
    if step < args.feedback_end:
        return 1.0
    done = min(1.0, (step - args.feedback_end) / max(args.distill_decay, 1))
    return 1.0 - done * (1.0 - args.distill_floor)


def train_step(comb: Combiner, examples, weights: dict, learned: bool, related_gate: float,
               distill: float = 1.0) -> dict:
    model = comb.model
    with model.core.autocast():
        anchor = comb.own_spans(examples, comb.args.writer_train)
        gates, bce = comb.gates(examples, learned, related_gate)
        outs = [comb.combine(ex, g) for ex, g in zip(examples, gates)]
        teacher = [ex['target_reps'] for ex in examples]
        cos = 1 - F.cosine_similarity(torch.cat(outs), torch.cat(teacher), dim=-1).mean()
        result, loss = {'cos': cos.item()}, distill * weights['cos'] * cos
        for part in ('recon', 'qa'):
            views = [ex[part] for ex in examples]
            logits, targets = model.read(views, outs)
            with torch.no_grad():
                t_logits, _ = model.read(views, teacher)
            nll, kl = F.cross_entropy(logits, targets), _kl(logits, t_logits)
            loss = loss + weights[f'{part}_nll'] * nll + distill * weights[f'{part}_kl'] * kl
            result[f'{part}_nll'], result[f'{part}_kl'] = nll.item(), kl.item()
        if bce is not None:
            loss = loss + weights['gate'] * bce
            result['gate_bce'] = bce.item()
        if anchor is not None:
            loss = loss + weights.get('writer', 0.5) * anchor
            result['writer_anchor'] = anchor.item()
    loss.backward()
    return {'loss': loss.item(), **result}


@torch.no_grad()
def evaluate(comb: Combiner, episodes: Episodes, rows, args, learned: bool) -> dict:
    model, out = comb.model, {}
    rng = random.Random(1234)
    for tag in args.eval_spaces:
        sums: dict[tuple[str, str], float] = {}
        tokens = {'recon': 0, 'qa': 0}
        for start in range(0, len(rows), args.eval_batch):
            batch = rows[start:start + args.eval_batch]
            full = [comb.build(episodes, row, tag, rng, related=args.related, feedback=0.0)
                    for row in batch]
            only = [comb.build(episodes, row, tag, random.Random(7), related=0, feedback=0.0)
                    for row in batch]
            rel = [comb.build(episodes, row, tag, random.Random(7), related=args.related,
                              feedback=0.0, gold=False) for row in batch]
            with model.core.autocast():
                for group in (full, only, rel):
                    comb.own_spans(group, False)
                g_full, _ = comb.gates(full, learned, args.related_gate)
                g_rel, _ = comb.gates(rel, learned, args.related_gate)
                spans = {
                    'teacher': [ex['target_reps'] for ex in full],
                    'gold_spans': [torch.cat(ex['records']) for ex in only],
                    'comb_gold': [comb.combine(ex, torch.ones(len(ex['records']),
                                                              device=model.device)) for ex in only],
                    'comb_gold_related': [comb.combine(ex, g) for ex, g in zip(full, g_full)],
                    'comb_related_only': [comb.combine(ex, g) if ex['records'] else
                                          ex['target_reps'][:0] for ex, g in zip(rel, g_rel)]}
                for part in ('recon', 'qa'):
                    views = [ex[part] for ex in full]
                    arms = {'noctx': model.read(views, None), 'full': model.read(views, None, True)}
                    arms.update({name: model.read(views, reps) for name, reps in spans.items()})
                    for name, (logits, targets) in arms.items():
                        sums[part, name] = sums.get((part, name), 0.0) + F.cross_entropy(
                            logits, targets, reduction='sum').item()
                    tokens[part] += int(arms['noctx'][1].numel())
        result = {}
        for part in ('recon', 'qa'):
            nll = {name: sums[part, name] / tokens[part] for p, name in sums if p == part}
            gain = max(nll['noctx'] - nll['full'], 1e-9)
            result[part] = {'nll': {k: round(v, 4) for k, v in nll.items()},
                            'captured': {k: round((nll['noctx'] - v) / gain, 4)
                                         for k, v in nll.items() if k not in ('noctx', 'full')}}
        out[tag] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--bank-cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--neighbors', type=Path, required=True)
    parser.add_argument('--train-episodes', type=Path, required=True)
    parser.add_argument('--train-cache', type=Path, required=True)
    parser.add_argument('--eval-episodes', type=Path, required=True)
    parser.add_argument('--eval-cache', type=Path, required=True)
    parser.add_argument('--init-codecs', type=Path)
    parser.add_argument('--writer-state', type=Path,
                        help='B3 writer.pt: read the writer\'s own spans instead of teacher reps')
    parser.add_argument('--writer-adapter-rank', type=int, default=16)
    parser.add_argument('--writer-passes', type=int, default=4)
    parser.add_argument('--writer-train', action='store_true')
    parser.add_argument('--writer-lr', type=float, default=1e-4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--related', type=int, default=4)
    parser.add_argument('--related-gate', type=float, default=0.0)
    parser.add_argument('--feedback-hold', type=int, default=500)
    parser.add_argument('--feedback-end', type=int, default=4000)
    parser.add_argument('--learned-gates-from', type=int, default=4000)
    parser.add_argument('--distill-floor', type=float, default=0.1)
    parser.add_argument('--distill-decay', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--weights', default='cos=1,recon_nll=1,recon_kl=1,qa_nll=1,qa_kl=1,gate=0.5')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-items', type=int, default=192)
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--eval-spaces', nargs='+', default=['s1', 's3'])
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cuda-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    bank = TeacherCache(args.bank_cache, args.sources)
    train = Episodes(args.train_episodes, args.train_cache, bank)
    evald = Episodes(args.eval_episodes, args.eval_cache, bank)
    eval_rows = sorted(evald.rows, key=lambda row: row['episode_id'])[:args.eval_items]
    model = Model(argparse.Namespace(
        cuda_fraction=args.cuda_fraction, experiment=args.experiment, checkpoint=args.checkpoint,
        adapter_rank=args.writer_adapter_rank if args.writer_state else 0,
        gate_open_start=-1, merge_at=-1))
    if args.writer_state:
        model.load_trained(torch.load(args.writer_state, map_location=model.device))
    comb = Combiner(args, model, bank)
    groups = [{'params': comb.parameters(), 'lr': args.lr}]
    if args.writer_train:
        writer_args = argparse.Namespace(lr=args.writer_lr, adapter_lr=args.writer_lr,
                                         decoder_lr=args.writer_lr / 10)
        groups += model.param_groups(writer_args)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    state_path, step = args.output / 'combiner.pt', 0
    if state_path.exists():
        state = torch.load(state_path, map_location=model.device)
        comb.combiners.load_state_dict(state['combiners'])
        comb.gate_head.load_state_dict(state['gate_head'])
        if 'writer' in state:
            model.load_trained(state['writer'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
    start = step
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + start + 1) / args.warmup))
    (args.output / 'config.json').write_text(json.dumps(dict(
        vars(args), train_episodes_n=len(train.rows), eval_items_n=len(eval_rows)),
        indent=2, default=str) + '\n')
    metrics = (args.output / 'metrics.jsonl').open('a', encoding='utf-8')

    def log(record):
        metrics.write(json.dumps(record) + '\n')
        metrics.flush()
        print(json.dumps(record), flush=True)

    if step == 0:
        log({'step': 0, 'eval': evaluate(comb, evald, eval_rows, args, False)})
    window: dict[str, float] = {}
    started = time.time()
    while step < args.steps:
        feedback = _feedback(args, step)
        learned = step >= args.learned_gates_from
        examples = [comb.build(train, row, rng.choice(SPACES), rng, related=rng.randint(0, args.related),
                               feedback=feedback)
                    for row in rng.sample(train.rows, args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        distill = _distill(args, step)
        result = train_step(comb, examples, weights, learned, args.related_gate, distill)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                       1.0)
        optimizer.step()
        schedule.step()
        step += 1
        for key, value in result.items():
            window[key] = window.get(key, 0.0) + value
        if step % args.log_every == 0:
            log({'step': step, **{k: round(v / args.log_every, 4) for k, v in window.items()},
                 'feedback_gate': round(feedback, 3), 'learned_gates': learned,
                 'distill': round(distill, 3),
                 'elapsed_s': round(time.time() - started)})
            window = {}
        if step % args.eval_every == 0 or step == args.steps:
            torch.save({'combiners': comb.combiners.state_dict(),
                        'gate_head': comb.gate_head.state_dict(),
                        **({'writer': model.trained_state()} if args.writer_train else {}),
                        'optimizer': optimizer.state_dict(), 'step': step},
                       state_path.with_suffix('.pending'))
            state_path.with_suffix('.pending').replace(state_path)
            log({'step': step, 'eval': evaluate(comb, evald, eval_rows, args,
                                                step >= args.learned_gates_from)})


if __name__ == '__main__':
    main()
