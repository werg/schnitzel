"""Writer stages B2-B4 of the BGKit restart: the frozen S2 decoder learns to generate BGKit spans.

The decoder (with its S2 LoRA) is frozen. Trainable: the span marker, its
continuous ratio code, the rep head and the emit/stop head (``schnitz.bgkit_span``).
Teacher-forced on the B1 cache: the writer sequence is

    chat(user: <write prompt> + passage) <|bg|>(rho) R_1 ... R_k

with teacher reps R as inputs; the state at the marker and at R_1..R_{k-1}
predicts R_1..R_k, and every span position predicts emit (0) or stop (1, after
R_k). Two streams (``--classical-fraction`` of the steps):

- pipeline: bank sources from the B1 cache at their length-scaled space ratios,
  written under the "compact ... with BGKit into a memory record" prompt;
- classical BGKit compression: reconstruct/continue samples from BGKit's own
  corpus (its ``AutoencodeDataset`` over the S2 train stores), at a log-uniform
  ratio x1-x128, written under "summarize this text with BGKit ..."; the teacher
  reps come from the frozen S2 encoder online, with BGKit's own compression prompt.

Losses: cosine to the teacher rep; stop cross-entropy; functional - the frozen
decoder reads the student reps in S2's reconstruct layout and pays NLL on the
passage and KL to its own reading of the teacher reps.

B3 options (all off by default, so the defaults are B2): ``--init-writer`` starts
from a B2 ``writer.pt``; ``--adapter-rank`` adds the span-gated write adapter
(``schnitz.bgkit_span.attach_write_adapter``: active only at span positions of a
write, so reading and ordinary text stay S2); ``--rollout-passes``/``--sample-*``
train on the writer's own reps (parallel passes, each feeding the previous
pass's reps at a ramping fraction of span positions); ``--gate-open-start`` /
``--gate-open-steps`` open the adapter on non-span positions from 0 to 1, so it
becomes a global LoRA; ``--merge-at`` then folds it and S2's LoRA exactly into the
weights and trains the whole decoder. Whenever the adapter acts outside spans or
the decoder is unfrozen, a replay KL to a frozen S2 copy on plain corpus text and
on reading teacher reps keeps S2's behaviour (``replay`` weight).

QA over memory (``--qa-fraction``, owner decision 27 September): per R6 episode
the writer writes each gold record (and ``--qa-related`` semantically related
records) under the memory prompt without seeing the question, and the reader
answers the question from those spans in order. The only loss on the answer is
its NLL - no distillation target, since BGKit's question-free encodings are weak
at QA - so gradients reach the writer, and the reader too once the gate opens.

Evaluation on held-out bank sources at each space ratio and on BGKit's eval
stores at x4/x16/x64: task NLL with no context, full text, teacher reps, student
teacher-forced reps and student free-running reps (fed back, teacher length), the
captured fraction of the full-text gain, and the free-running stop-length error. Training-only; runs in the
``schnitz-bgkit`` container.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from schnitz.bgkit_span import MEMORY_PROMPT, SUMMARIZE_PROMPTS, span_targets
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.decoder import (ADAPTER_TARGETS, LEVELS, Model, TeacherCache, _batches,
                                _heldout, _kl, _subset)

SPACES = LEVELS  # the writer's ratio levels (historical name in this stage)


def _example(cache: TeacherCache, model: Model, item, tag: str) -> dict:
    shard, row, record_id, tokens, source = item
    ids = model.text_ids(cache.texts[source])
    return {'ids': ids, 'target': ids, 'task': 'reconstruct', 'prompt': 'memory',
            'teacher': cache.reps(shard, row, tag), 'factor': cache.factor(shard, row, tag)}


def _classical(model: Model, samples, factors: list[float],
               stated: list[bool] | None = None) -> list[dict]:
    """``stated[i]``: the prompt names the factor (nearest power of two; B4)."""
    teacher = model.teacher(samples, factors)
    stated = stated or [False] * len(samples)
    out = []
    for s, t, f, named in zip(samples, teacher, factors, stated):
        power = 2 ** min(7, max(0, round(math.log2(f))))
        out.append({'ids': s.ctx_ids.long(), 'target': s.target_ids.long(), 'task': s.task,
                    'prompt': f'summarize-{s.task}@{power}' if named else f'summarize-{s.task}',
                    'teacher': t, 'factor': f, 'ratio_stated': named})
    return out


def _classical_dataset(model: Model, stores, weights, ctx_max: int, seed: int, size=None):
    from bgkit2.data.autoencode import AutoencodeDataset
    from bgkit2.data.token_store import TokenStore
    return AutoencodeDataset([TokenStore(p) for p in stores], weights, ctx_min=64,
                             ctx_max=ctx_max, cont_min=32, cont_max=256, p_continue=0.5,
                             seed=seed, epoch_size=size, task_prompt_ids=model.encoder_prompts)


def _rollout(model: Model, examples, passes: int, sample: float, sequential: int = 0):
    """Writer forward on the teacher-fed span with a fraction ``sample`` of inputs
    replaced by the writer's own reps; gradients flow through the final pass.

    With ``sequential`` > 0 the first ``sequential`` reps of each span are generated
    one at a time from the writer's own previous reps (no gradient), so those
    positions are exactly free-running; ``passes`` parallel passes (each feeding the
    previous pass's reps, detached) then cover longer spans. Returns teacher reps,
    predicted reps and the cosine and stop losses."""
    writer = model.writer
    teacher_feed = [ex['teacher'].to(model.device).float() for ex in examples]
    feed = teacher_feed
    # the prompt and source are computed once (with the current weights and gate) and
    # every no-gradient pass runs only span positions
    prefix = model.prefix(examples)
    if sequential and sample > 0:
        limits = [min(sequential, t.shape[0]) for t in teacher_feed]
        own = [t[:0] for t in teacher_feed]
        with torch.no_grad():
            for step in range(max(limits)):
                states = model.write(examples, own, prefix)
                own = [o if step >= n else torch.cat([o, writer.rep(h[-1:])])
                       for o, h, n in zip(own, states, limits)]
        feed = [torch.cat([o, t[o.shape[0]:]]) for o, t in zip(own, teacher_feed)]
    if passes and sample > 0:
        mask = [torch.rand(t.shape[0], 1, device=model.device) < sample for t in teacher_feed]
        start = feed
        for _ in range(passes):
            with torch.no_grad():
                states = model.write(examples, feed, prefix)
                preds = [writer.rep(h[:-1]) for h in states]
            feed = [torch.where(m, p, f) for m, p, f in zip(mask, preds, start)]
    # k+1 states (marker, R_1..R_k): the first k predict R_1..R_k, all k+1 emit/stop;
    # the gradient pass reuses the prefix only while nothing trainable acts on it
    full = model.write(examples, feed, None if model.replaying else prefix)
    counts = [t.shape[0] for t in teacher_feed]
    preds = [writer.rep(h[:-1]) for h in full]
    cos = 1 - F.cosine_similarity(torch.cat(preds), torch.cat(teacher_feed), dim=-1).mean()
    # one stop position per span against k emit positions: ``stop_pos_weight`` balances
    # the classes so the stop logit does not hover at the threshold
    stop = F.cross_entropy(writer.stop(torch.cat(full).float()),
                           span_targets(counts, model.device),
                           weight=torch.tensor([1.0, model.stop_pos_weight], device=model.device))
    return teacher_feed, preds, cos, stop


class QAEpisodes:
    """R6 episodes whose gold (and related) records the writer writes for QA reads."""

    def __init__(self, path: Path, cache: TeacherCache, neighbors: Path | None):
        self.cache = cache
        self.items = {item[2]: item for item in cache.items}
        self.rows = [row for row in (json.loads(line) for line in path.open(encoding='utf-8'))
                     if all(r in self.items for r in row['required_ids'])]
        self.neighbors, self.index, self.ids = None, None, None
        if neighbors is not None:
            data = torch.load(neighbors, weights_only=False)
            self.neighbors, self.ids = data['neighbors'], data['record_ids']
            self.index = {record_id: i for i, record_id in enumerate(self.ids)}

    def build(self, model: Model, row: dict, tag: str, rng: random.Random, related: int,
              port: bool = False):
        golds = list(row['required_ids'])
        entries = golds[:]
        if related and self.neighbors is not None:
            pool = [self.ids[j] for r in golds for j in self.neighbors[self.index[r]].tolist()]
            pool = [r for r in dict.fromkeys(pool) if r not in golds]
            for record_id in rng.sample(pool, min(related, len(pool))):
                entries.insert(rng.randint(0, len(entries)), record_id)
        records = [_example(self.cache, model, self.items[r], tag) for r in entries]
        joined = model.text_ids('\n\n'.join(self.cache.texts[self.items[r][4]] for r in golds))
        answer = torch.tensor(model.tok(' ' + row['answer'].strip(),
                                        add_special_tokens=False)['input_ids'][:64])
        view = {'ids': joined, 'target': answer, 'task': 'reconstruct',
                'instr': model.question_instr(row['query'])}
        if port:  # the question arrives through the soft input port instead of as text
            view['instr'] = model.question_instr('')
            view['port_ids'] = model.text_ids(row['query'], limit=256)
        return records, view


def qa_step(model: Model, episodes: QAEpisodes, rows, weights: dict, rng: random.Random,
            related: int, passes: int, sample: float, sequential: int = 0,
            port_share: float = 0.0) -> dict:
    """Writer writes the records (question-free); the reader answers from them. With
    ``port_share`` a fraction of questions arrives through the soft input port."""
    tag = rng.choice(SPACES)
    built = [episodes.build(model, row, tag, rng, related, port=rng.random() < port_share)
             for row in rows]
    ported = [view for _, view in built if 'port_ids' in view]
    if ported:
        for view, reps in zip(ported, model.port_reps([v['port_ids'] for v in ported])):
            view['port'] = reps
    records = [record for recs, _ in built for record in recs]
    with model.core.autocast():
        _, preds, cos, stop = _rollout(model, records, passes, sample, sequential)
        spans, start = [], 0
        for recs, _ in built:
            spans.append(torch.cat(preds[start:start + len(recs)]))
            start += len(recs)
        logits, targets = model.read([view for _, view in built], spans)
        nll = F.cross_entropy(logits, targets)
        loss = weights['qa'] * nll + weights['cos'] * cos + weights['stop'] * stop
        extra, logged = model.take_protocol_losses(weights)
        if extra is not None:
            loss = loss + extra
    loss.backward()
    return {'loss': loss.item(), 'qa_nll': nll.item(), 'cos': cos.item(), 'stop': stop.item(),
            'records': len(records), 'ported': len(ported), **logged}


@torch.no_grad()
def evaluate_qa(model: Model, episodes: QAEpisodes, rows, related: int,
                batch_size: int = 8) -> dict:
    """Answer NLL reading: nothing, the gold text, the gold records' teacher spans, the
    writer's free-running spans of the golds, and of golds plus related records; and
    the matched controls ``*_shuffled``, the same kind of spans of the *next* episode
    in the batch. A trained span can lower the answer NLL by format alone (the
    no-context arm has no slot at all), so content is the gain over its shuffled
    control (``content_nats``)."""
    out = {}
    for tag in ('s0', 's2'):
        sums: dict[str, float] = {}
        tokens = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            gold = [episodes.build(model, row, tag, random.Random(5), 0) for row in batch]
            mixed = [episodes.build(model, row, tag, random.Random(5), related) for row in batch]
            views = [view for _, view in gold]
            arms = {}
            with model.core.autocast():
                arms['noctx'] = model.read(views, None)
                arms['full'] = model.read(views, None, True)
                teacher = [torch.cat([r['teacher'] for r in recs]) for recs, _ in gold]
                arms['teacher'] = model.read(views, teacher)
                arms['teacher_shuffled'] = model.read(views, teacher[1:] + teacher[:1])
                for name, group in (('student_free', gold), ('student_free_related', mixed)):
                    records = [r for recs, _ in group for r in recs]
                    free, _ = model.free_run(records, [r['teacher'].shape[0] for r in records])
                    spans, pos = [], 0
                    for recs, _ in group:
                        spans.append(torch.cat(free[pos:pos + len(recs)]))
                        pos += len(recs)
                    arms[name] = model.read(views, spans)
                    if name == 'student_free':
                        arms['student_free_shuffled'] = model.read(views, spans[1:] + spans[:1])
                        if model.protocol is not None:  # question through the soft input port
                            ported = [episodes.build(model, row, tag, random.Random(5), 0, port=True)[1]
                                      for row in batch]
                            for view, reps in zip(ported, model.port_reps(
                                    [v['port_ids'] for v in ported])):
                                view['port'] = reps
                            arms['student_free_port'] = model.read(ported, spans)
                            arms['student_free_port_shuffled'] = model.read(
                                ported, spans[1:] + spans[:1])
            for name, (logits, targets) in arms.items():
                sums[name] = sums.get(name, 0.0) + F.cross_entropy(
                    logits, targets, reduction='sum').item()
            tokens += int(arms['noctx'][1].numel())
        nll = {name: value / tokens for name, value in sums.items()}
        gain = max(nll['noctx'] - nll['full'], 1e-9)
        out[f'qa/{tag}'] = {'nll': {k: round(v, 4) for k, v in nll.items()},
                            'captured': {k: round((nll['noctx'] - v) / gain, 4)
                                         for k, v in nll.items() if k not in ('noctx', 'full')},
                            'content_nats': {
                                'teacher': round(nll['teacher_shuffled'] - nll['teacher'], 4),
                                'student_free': round(nll['student_free_shuffled']
                                                      - nll['student_free'], 4),
                                **({'student_free_port': round(nll['student_free_port_shuffled']
                                                               - nll['student_free_port'], 4)}
                                   if 'student_free_port' in nll else {})}}
    return out


def train_step(model: Model, examples, weights: dict, passes: int = 0,
               sample: float = 0.0, replay_tokens: int = 1024, sequential: int = 0) -> dict:
    """One step. ``passes`` > 0 with ``sample`` > 0 trains on the writer's own reps: a
    fixed random fraction ``sample`` of span inputs is replaced by the previous pass's
    predictions (detached); gradients flow through the final pass."""
    with model.core.autocast():
        teacher_feed, preds, cos, stop = _rollout(model, examples, passes, sample, sequential)
        pred = torch.cat(preds)
        logits, targets = model.read(examples, preds)
        nll = F.cross_entropy(logits, targets)
        with torch.no_grad():
            t_logits, _ = model.read(examples, teacher_feed, decoder=model.reference)
        kl = _kl(logits, t_logits)
        loss = (weights['cos'] * cos + weights['stop'] * stop + weights['nll'] * nll
                + weights['kl'] * kl)
        result = {'cos': cos.item(), 'stop': stop.item(), 'nll': nll.item(), 'kl': kl.item()}
        if model.replaying:
            # replay: outside spans the decoder must still read teacher reps and text as S2
            # on a random subset of positions, chosen before the LM head (memory)
            pick = _subset(len(targets), replay_tokens, model.device)
            read_now, _ = model.read(examples, teacher_feed, index=pick)
            n_text = sum(ex['ids'].shape[0] for ex in examples)
            text_pick = _subset(n_text, replay_tokens, model.device)
            text_now = model.text_logits(examples, index=text_pick)
            with torch.no_grad():
                text_ref = model.text_logits(examples, decoder=model.reference, index=text_pick)
            replay = _kl(read_now, t_logits[pick]) + _kl(text_now, text_ref)
            loss = loss + weights.get('replay', 1.0) * replay
            result['replay'] = replay.item()
        extra, logged = model.take_protocol_losses(weights)
        if extra is not None:
            loss = loss + extra
            result.update(logged)
    loss.backward()
    return {'loss': loss.item(), **result, 'reps': len(pred)}


@torch.no_grad()
def _score(model: Model, groups) -> dict:
    sums: dict[str, float] = {}
    tokens, count, length_err, stop_hits, stop_missing = 0, 0, 0.0, 0, 0
    replay, replay_n = 0.0, 0
    for examples in groups:
        counts = [ex['teacher'].shape[0] for ex in examples]
        with model.core.autocast():
            tf = [model.writer.rep(h[:-1]) for h in model.write(
                examples, [ex['teacher'] for ex in examples])]
            free, stops = model.free_run(examples, counts)
            arms = {'noctx': model.read(examples, None), 'full': model.read(examples, None, True),
                    'teacher': model.read(examples, [ex['teacher'] for ex in examples]),
                    'student_tf': model.read(examples, tf),
                    'student_free': model.read(examples, free)}
            if model.reference is not None:
                ref_read, _ = model.read(examples, [ex['teacher'] for ex in examples],
                                         decoder=model.reference)
                text_kl = _kl(model.text_logits(examples),
                              model.text_logits(examples, decoder=model.reference))
                replay += (_kl(arms['teacher'][0], ref_read) + text_kl).item()
                replay_n += 1
        for name, (logits, targets) in arms.items():
            sums[name] = sums.get(name, 0.0) + F.cross_entropy(
                logits, targets, reduction='sum').item()
        tokens += int(arms['noctx'][1].numel())
        count += len(examples)
        for k, stop in zip(counts, stops):
            # no stop within k + 1 positions: an overrun, scored as the full length
            stop_missing += stop is None
            stop_hits += stop == k
            length_err += (1.0 if stop is None else abs(stop - k) / max(k, 1))
    nll = {name: value / tokens for name, value in sums.items()}
    extra = {}
    if model.reference is not None:
        extra = {'replay_kl': round(replay / max(replay_n, 1), 4)}
    gain = max(nll['noctx'] - nll['full'], 1e-9)
    return {'nll': {k: round(v, 4) for k, v in nll.items()},
            'captured': {k: round((nll['noctx'] - nll[k]) / gain, 4)
                         for k in ('teacher', 'student_tf', 'student_free')},
            'stop_exact': round(stop_hits / count, 4),
            'stop_missing': round(stop_missing / count, 4),
            'length_rel_err': round(length_err / count, 4), **extra}


def evaluate(model: Model, cache: TeacherCache, items, classical, batch_size: int) -> dict:
    out = {}
    model.protocol_losses, model.record_protocol = [], model.protocol is not None
    for tag in SPACES:
        out[f'bank/{tag}'] = _score(model, (
            [_example(cache, model, item, tag) for item in items[i:i + batch_size]]
            for i in range(0, len(items), batch_size)))
    for factor in (4.0, 16.0, 64.0):
        out[f'classical/x{int(factor)}'] = _score(model, (
            _classical(model, classical[i:i + batch_size], [factor] * len(classical[i:i + batch_size]))
            for i in range(0, len(classical), batch_size)))
    if model.record_protocol:  # opening and closing a span on held-out teacher-fed writes
        model.record_protocol = False
        _, out['protocol'] = model.take_protocol_losses({})
    return out


def _gate(args, step: int) -> float:
    if args.gate_open_start < 0 or step < args.gate_open_start:
        return 0.0
    return min(1.0, (step - args.gate_open_start + 1) / max(args.gate_open_steps, 1))


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--stop-pos-weight', type=float, default=1.0,
                        help='class weight of the stop position in the emit/stop loss')
    parser.add_argument('--merge-checkpoint', action=argparse.BooleanOptionalAction, default=True,
                        help='after the merge, recompute decoder-layer activations in backward')
    parser.add_argument('--batch-tokens', type=int, default=4096,
                        help='per batch: source tokens (bank) or context + target tokens (classical)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--classical-fraction', type=float, default=0.4)
    parser.add_argument('--classical-batch', type=int, default=16)
    parser.add_argument('--classical-ctx-max', type=int, default=512)
    parser.add_argument('--classical-eval', type=int, default=64)
    parser.add_argument('--weights', default='cos=1,stop=0.2,nll=1,kl=1')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-items', type=int, default=256)
    parser.add_argument('--eval-max-tokens', type=int, default=512)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--cuda-fraction', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-writer', type=Path, help='B3: start from this B2 writer.pt')
    parser.add_argument('--adapter-rank', type=int, default=0)
    parser.add_argument('--adapter-lr', type=float, default=2e-4)
    parser.add_argument('--rollout-passes', type=int, default=0)
    parser.add_argument('--sequential-reps', type=int, default=0,
                        help="generate the first N reps of each span one at a time from "
                        "the writer's own reps before the parallel passes")
    parser.add_argument('--sample-max', type=float, default=1.0)
    parser.add_argument('--sample-ramp', type=int, default=2000,
                        help='steps over which the self-fed fraction rises to --sample-max')
    parser.add_argument('--gate-open-start', type=int, default=-1,
                        help='step at which the write adapter starts opening outside spans')
    parser.add_argument('--gate-open-steps', type=int, default=2000)
    parser.add_argument('--merge-at', type=int, default=-1,
                        help='step at which the open adapter and S2 LoRA are merged and the '
                        'whole decoder trains (needs the gate fully open)')
    parser.add_argument('--decoder-lr', type=float, default=1e-5)
    parser.add_argument('--replay-tokens', type=int, default=1024)
    parser.add_argument('--qa-episodes', type=Path, help='R6 train episodes for QA over memory')
    parser.add_argument('--qa-eval-episodes', type=Path)
    parser.add_argument('--qa-fraction', type=float, default=0.0)
    parser.add_argument('--qa-batch', type=int, default=12)
    parser.add_argument('--qa-related', type=int, default=2)
    parser.add_argument('--qa-eval-items', type=int, default=96)
    parser.add_argument('--neighbors', type=Path)
    parser.add_argument('--protocol', action='store_true',
                        help='B4: span protocol tokens with LM-head rows, turn end after the span, '
                             'memory/port delimiters in reads (restart plan 3.2)')
    parser.add_argument('--init-state', type=Path,
                        help='B4: start from this B3 writer.pt (writer and merged decoder)')
    parser.add_argument('--protocol-lr', type=float, default=1e-3)
    parser.add_argument('--ratio-stated', type=float, default=0.0,
                        help='fraction of classical prompts that state the compression factor')
    parser.add_argument('--port-share', type=float, default=0.0,
                        help='final fraction of QA questions given through the soft input port')
    parser.add_argument('--port-ramp', type=int, default=2000)


def run(args) -> None:
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}
    weights.setdefault('qa', 1.0)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cache = TeacherCache(args.cache, args.sources)
    train = [item for item in cache.items if not _heldout(item[2])]
    heldout = [item for item in cache.items if _heldout(item[2]) and item[3] <= args.eval_max_tokens]
    heldout = sorted(heldout, key=lambda item: item[2])[:args.eval_items]
    model = Model(args)
    data = model.core.cfg2.data
    classical_train = _classical_dataset(model, data.train_stores, data.train_weights,
                                         args.classical_ctx_max, args.seed)
    evald = _classical_dataset(model, data.eval_stores, None, 256, 10_007, args.classical_eval)
    classical_eval = [evald[i] for i in range(len(evald))]
    qa_train = qa_eval = None
    if args.qa_fraction > 0:
        qa_train = QAEpisodes(args.qa_episodes, cache, args.neighbors)
        qa_eval = QAEpisodes(args.qa_eval_episodes, cache, args.neighbors)
        qa_eval_rows = sorted(qa_eval.rows, key=lambda row: row['episode_id'])[:args.qa_eval_items]

    def run_eval():
        result = evaluate(model, cache, heldout, classical_eval, args.classical_batch)
        if qa_eval is not None:
            result.update(evaluate_qa(model, qa_eval, qa_eval_rows, args.qa_related))
        return result
    out = Run(args.output)
    step = 0
    state = out.load('writer.pt', model.device)
    if state is not None:
        step = state['step']
        model.gate_value = _gate(args, step)
        model.load_trained(state)
    elif args.init_state:
        init = torch.load(args.init_state, map_location=model.device)
        model.load_trained({k: init[k] for k in ('writer', 'merged', 'decoder', 'protocol')
                            if k in init})
    elif args.init_writer:
        model.writer.load_state_dict(
            torch.load(args.init_writer, map_location=model.device)['writer'])
    if args.protocol and model.protocol is None:
        model.install_protocol()

    def build_optimizer(start: int):
        return warmup_optimizer(model.param_groups(args), args.warmup, start)

    if 0 <= args.merge_at < args.gate_open_start + args.gate_open_steps - 1 and args.adapter_rank:
        raise ValueError('--merge-at must come after the gate is fully open')
    optimizer, schedule = build_optimizer(step - args.merge_at if model.merged else step)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    out.write_config(dict(vars(args), train_items=len(train), heldout_items=len(heldout),
                  target_norm=model.target_norm, adapter_targets=ADAPTER_TARGETS, memory_prompt=MEMORY_PROMPT,
                  summarize_prompts=SUMMARIZE_PROMPTS))
    log = out.log

    if step == 0:
        log({'step': 0, 'eval': run_eval()})
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    window = Window()
    while step < args.steps:
        draw = rng.random()
        if qa_train is not None and draw >= 1 - args.qa_fraction:
            stream = 'qa'
        elif draw < args.classical_fraction:
            samples, budget = [], args.batch_tokens
            while len(samples) < args.classical_batch:
                sample = classical_train[rng.randrange(len(classical_train))]
                budget -= sample.target_ids.shape[0] + sample.ctx_ids.shape[0]
                if samples and budget < 0:
                    break
                samples.append(sample)
            factors = [2 ** rng.uniform(0, 7) for _ in samples]  # x1 .. x128, log-uniform
            examples = _classical(model, samples, factors,
                                  [rng.random() < args.ratio_stated for _ in samples])
            stream = 'classical'
        else:
            examples = [_example(cache, model, item, rng.choice(SPACES))
                        for item in next(batches)]
            stream = 'bank'
        optimizer.zero_grad(set_to_none=True)
        model.gate_value = _gate(args, step)
        if step == args.merge_at and not model.merged:
            model.merge()
            optimizer, schedule = build_optimizer(0)  # fresh warmup for the whole decoder
            log({'step': step, 'event': 'merged; training the whole decoder'})
        sample = args.sample_max * min(1.0, step / max(args.sample_ramp, 1))
        if stream == 'qa':
            result = qa_step(model, qa_train, rng.sample(qa_train.rows, args.qa_batch), weights,
                             rng, rng.randint(0, args.qa_related), args.rollout_passes, sample,
                             args.sequential_reps,
                             args.port_share * min(1.0, step / max(args.port_ramp, 1)))
        else:
            result = train_step(model, examples, weights, args.rollout_passes, sample,
                                args.replay_tokens, args.sequential_reps)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                       1.0)
        optimizer.step()
        schedule.step()
        step += 1
        window.add(result, stream)
        if step % args.log_every == 0:
            log({'step': step, **window.means(), 'lr': schedule.get_last_lr()[0],
                 'elapsed_s': out.elapsed()})
        if step % args.eval_every == 0 or step == args.steps:
            out.save('writer.pt', {**model.trained_state(), 'optimizer': optimizer.state_dict(),
                                   'step': step})
            log({'step': step, 'eval': run_eval()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    run(parser.parse_args())
