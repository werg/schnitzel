"""B9: learning by experience over rounds (restart plan B9; docs/knowledge-base-stack.md,
5.1 step 9). Building blocks in ``schnitz.kb.experience``.

The model works on verifiable task episodes (memory transcripts v3 of a task corpus:
``verify`` spec, teacher trajectory) over the dataset's KB for ``--rounds`` rounds. One
persistent KB per dataset for the whole run: the L1 banks (``--banks``, from ``train.py l1
build`` over the same transcripts) are copied to ``<output>/kbs`` once, and every
round's write goes into them. Per round t of a training episode:

1. **Attempt.** The frozen decoder generates from the episode's prompt (system and
   user messages, memory tools), greedy by default; each ``memory_search()`` call the
   model emits is executed as an L1 read (query at the call's closing parenthesis from
   the exact prefix, retrieval over the episode's own KB only, S_s, R, span in the tool
   message). The final answer is scored by the episode's verifier
   (``schnitz.task_verifiers``).
2. **SFT.** The teacher trajectory (``l1.run_episode``: its reads retrieved from the KB
   as it is at round t) gives the task NLL on assistant tokens, plus the retrieval loss
   on its slots; gradients accumulate over rounds and episodes, one optimizer step per
   batch. Trained: the L1 reader (query and item-key heads, gate offsets, S_s, R); item
   values in place with ``--item-lr`` > 0 (L1a regime). The decoder, writer and codecs
   are frozen.
3. **Write.** At the round's end the model calls ``memory_write()`` with its attempt in
   context and generates the ``<|bg|>`` span in place (single pass, the frozen writer
   free-running at the length schedule of the attempt's generated tokens); the codecs
   give the items, the item-key heads their keys. The task's own record is appended at
   its first write and superseded (same ids, new version) afterwards, so later rounds and
   later tasks read the newest version; the registry (``experience.Registry``) records how
   it was produced.

Gold records: the first time a training task is met while the gold weight is positive,
its teacher text (the transcript's write-site ``teacher_text``, else question and answer)
is written by the bank writer under the memory prompt (bank creation) and appended as a
``gold`` record. Its gate is multiplied by w = global schedule (``--gold-start``, linear to
0 over ``--gold-anneal`` steps) x ``--gold-decay`` ** (supersedes of the task's own record);
gates scale mass, so w is its exact share (the read's ``weights``); at w = 0 it is
hidden. Records that read a gold-carrying record are ``hinted``.

Held-out evaluation (validation episodes, never trained on, never given gold): rounds
run round-major over the evaluation episodes, each round in three arms: ``normal``,
``removed`` (the task's own prior-round record hidden) and ``swapped`` (its values
replaced by the next evaluation episode's own record; its key kept); accuracy per round
and arm by the verifiers. Evaluation rounds read only training records whose lineage
never saw gold for the evaluation task (``--heldout-lineage task``; ``any``: no gold
lineage at all) and their own records of this evaluation; they write (``split='eval'``)
into the persistent KB, never visible to training or to later evaluations.

Gradient into earlier rounds' writes (the L1b regime, ``--backprop-rounds k``, default
2): the round's write is logged as a write source (``schnitz.kb.producer.WriteLog``: the
write site's token ids, the attempt's read spans, the generated span and items, and per
read its query state and the items it read). In the SFT pass of round t the task's own
record (written at round t-1) is read as its producers' recomputation
(``schnitz.kb.producer.Producers``: the same selective replay as L1b, bit-exact at the
forward), so round t's losses reach the writer's span heads and the codecs through the
write of round t-1; the replay of that write recomputes each of its reads of the task's
own record from the round t-2 write (forward: the logged span exactly, gradient through
the recomputed read: ``L1Reader.reread``), and so on, truncated after k writes. Other
records are read as stored. The producers train at their own rates (``--l1b-codec-lr``,
``--l1b-writer-lr``); with k = 0 the writes are detached and the writer and codecs are
frozen. The log lives for one episode (its rounds share the parameters: one optimizer
step per batch).

Resume: every checkpoint pairs ``b9.pt`` (reader, optimizer, step, RNG, registries) with a
live checkpoint of every KB; a restart restores it and discards later commits. Episodes
whose verifier needs an environment (agent trajectories, APIGen-MT turns) are skipped.
Training-only; runs in ``sdkb-bgkit``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
from pathlib import Path
import random
import time

import torch

from schnitz.kb.bank import Transcripts, slots_of, write_spans as bank_write_spans
from schnitz.kb.experience import (GenProtocol, GoldSchedule, KBView, Registry, generate,
                                   read_items, read_mass, write_source)
from schnitz.kb.producer import (FEEDS, Producers, Writer, WriteLog, producer_params,
                                 span_length)
from schnitz.kb.read import ItemCache
from schnitz.kb_store import NewItem, Provenance
from schnitz.task_verifiers import check_episode

VERIFIABLE = ('sql', 'calls', 'code', 'knights', 'values', 'exact', 'synlogic')
ARMS = ('normal', 'removed', 'swapped')


def verify(answer: str, spec: dict) -> bool:
    """The verifier's outcome; a verifier failure on a malformed answer scores wrong."""
    try:
        return bool(check_episode(answer, spec))
    except Exception:  # noqa: BLE001 - any failure to verify is a failed outcome
        return False


def gold_text(row: dict) -> str:
    """The teacher's record of a task: the last write site's teacher text, else the
    question and the answer."""
    sites = row.get('write_sites') or []
    if sites and sites[-1].get('teacher_text'):
        return sites[-1]['teacher_text']
    question = next((m['content'] for m in row['messages'] if m['role'] == 'user'
                     and isinstance(m.get('content'), str)), '')
    return f'{question}\nAnswer: {row.get("answer", "")}'


def prompt_ids(row: dict, tok) -> list[int]:
    """The episode's prompt: its messages before the first assistant turn, with the
    memory tools, ending in the assistant generation prompt."""
    from schnitz.kb.stages.l1 import chat
    messages, tools = chat(row)
    first = next((i for i, m in enumerate(messages) if m['role'] == 'assistant'), len(messages))
    out = tok.apply_chat_template(messages[:first], tools=tools, add_generation_prompt=True,
                                  tokenize=True, return_dict=True)
    return list(out['input_ids'])


def rounds_loop(task: str, rounds: int, view_for, attempt_fn, sft_fn, write_fn) -> list[dict]:
    """The round loop of one episode: round t builds its view of the KB (``view_for(t)``),
    attempts, trains (``sft_fn`` may be None), and only then writes, so round t reads
    exactly the records written before it."""
    results = []
    for t in range(rounds):
        view = view_for(t)
        attempt = attempt_fn(t, view)
        sft = sft_fn(t, view) if sft_fn is not None else None
        record = write_fn(t, attempt)
        results.append({'round': t, 'attempt': attempt, 'sft': sft, 'record': record})
    return results


class Experience:
    """The model parts, KBs and registries of a B9 run."""

    def __init__(self, args, model, frozen, reader, kbs, ctx, proto, level: str):
        self.args, self.model, self.frozen, self.reader = args, model, frozen, reader
        self.kbs, self.ctx, self.proto, self.level = kbs, ctx, proto, level
        self.registries = {name: Registry(kb.dataset) for name, kb in kbs.items()}
        self.schedule = GoldSchedule(args.gold_start, args.gold_anneal, args.gold_decay)
        self.device = frozen.device
        # the writer's in-place writes and their replay (one write at a time: the replay
        # has the generation's batch composition)
        self.writer = None if model is None else Writer(model, reader.stack, frozen.embed, 1)
        self.log: WriteLog | None = None
        self.producers: Producers | None = None
        self.producer_stats: list[dict] = []
        if args.temperature > 0:
            gen = torch.Generator(device='cpu').manual_seed(args.seed)

            def sample(logits, step):
                p = torch.softmax(logits.float().cpu() / args.temperature, -1)
                return int(torch.multinomial(p, 1, generator=gen))
            self.policy = sample
        else:
            from schnitz.kb.experience import greedy
            self.policy = greedy

    def embed(self, ids) -> torch.Tensor:
        return self.frozen.embed(torch.as_tensor(ids, dtype=torch.long))

    def kb_of(self, row: dict):
        kb = self.kbs.get(row['kb'])
        if kb is None or kb.dataset != row['kb']:
            raise PermissionError(f'{row["episode_id"]}: no KB of dataset {row["kb"]!r}')
        return kb

    def gold_weight(self, dataset: str, step: int):
        reg = self.registries[dataset]
        return lambda task: self.schedule.weight(reg, task, step)

    def view(self, row: dict, split: str, step: int, mode: str = 'normal',
             swap_with: str | None = None) -> tuple[KBView, dict[str, float]]:
        kb = self.kb_of(row)
        hidden, substitute, weights = self.registries[kb.dataset].visibility(
            row['episode_id'], split, gold_weight=self.gold_weight(kb.dataset, step),
            mode=mode, swap_with=swap_with, heldout_lineage=self.args.heldout_lineage)
        return KBView(kb, hidden, substitute), weights

    # -- gradients into earlier rounds' writes ------------------------------------------
    def begin_episode(self, row: dict) -> None:
        """With ``--backprop-rounds`` k > 0: a write log and a producer replay for the
        episode's rounds (the task's own records written in this episode, depth k)."""
        k = getattr(self.args, 'backprop_rounds', 0)
        if k <= 0 or self.writer is None:
            self.log = self.producers = None
            return
        reg = self.registries[self.kb_of(row).dataset]
        self.log = log = WriteLog(None)

        def resolve(dataset: str, space: str, item_id: str):
            rec = reg.record_of(item_id)
            if rec is not None and rec.record_id in log.index:
                return ('write', dataset, rec.record_id)
            return None
        self.producers = Producers(self.writer, resolve=resolve, log=log,
                                   mode=getattr(self.args, 'l1b_replay', 'free'), batch=1,
                                   depth=k, reread=self.reader.reread)

    def end_episode(self) -> dict | None:
        """Backpropagate the episode's accumulated item gradients through its writes."""
        if self.producers is None:
            return None
        stats = self.producers.backward()
        self.producer_stats.append(stats)
        self.log = self.producers = None
        return stats

    def _rereads(self, attempt, dataset: str) -> list:
        """Per read of the attempt: its query state and, per space, the items read (their
        values as read, their recomputable sources, their gate scales)."""
        out = []
        for read in attempt.reads:
            if read is None or read.state is None or attempt.cache is None:
                out.append(None)
                continue
            spaces = {}
            for s, info in read.spaces.items():
                if not info.refs:
                    continue
                spaces[s] = {'keys': [self.producers.source(d, s, i) for d, i in info.refs],
                             'values': [attempt.cache.values[(d, s, i)].detach()
                                        for d, i in info.refs],
                             'scales': list(info.scales)}
            out.append({'state': read.state, 'spaces': spaces})
        return out

    # -- the three parts of a round -----------------------------------------------------
    @torch.no_grad()
    def attempt(self, row: dict, ep, view: KBView, weights: dict):
        cache = ItemCache(self.device, train=False)

        def read(state):
            with self.frozen.autocast():
                return self.reader.read(state, [view], [ep.kb], ep.query_time, cache,
                                        weights=weights)
        policy = self.policy
        if getattr(self.args, 'open_with_search', False):
            # the attempt opens with a memory_search() call (as every teacher transcript
            # does) until the decoder has learned to call on its own (B4)
            forced = list(self.model.tok('<|tool_call_start|>[memory_search()]<|tool_call_end|>',
                                         add_special_tokens=False)['input_ids'])
            policy = lambda logits, step: (forced[step] if step < len(forced)  # noqa: E731
                                           else self.policy(logits, step))
        attempt = generate(self.frozen.lm, self.embed, lambda x: self.frozen.mid(x[None])[0],
                           read, prompt_ids(row, self.model.tok), self.proto,
                           self.args.max_new, policy, self.args.max_reads,
                           self.frozen.autocast)
        attempt.cache = cache
        return attempt, verify(attempt.answer, row['verify'])

    def sft(self, ep, view: KBView, weights: dict, weight: float, task: str) -> dict:
        from schnitz.kb.stages.l1 import run_episode
        cache = ItemCache(self.device, train=self.args.item_lr > 0)
        self.ctx.kbs = {ep.kb: view}
        try:
            nll, n, reads, _ = run_episode(self.ctx, ep, cache, 'retrieve', weights=weights,
                                           producer=self.producers)
            loss = weight * nll / n
            aux = [r.aux for r in reads if r.aux is not None]
            if aux and self.args.retrieval_weight:
                loss = loss + self.args.retrieval_weight * torch.stack(aux).mean()
            loss.backward()
        finally:
            self.ctx.kbs = dict(self.kbs)
        items = cache.apply(self.args.item_lr)['items'] if self.args.item_lr > 0 else 0
        reg = self.registries[view.dataset]
        return {'nll': nll.item() / n, 'tokens': n, 'items_updated': items,
                'aux': float(torch.stack(aux).mean()) if aux else None,
                'sft_mass': read_mass(reads, reg, task)}

    @torch.no_grad()
    def write(self, row: dict, ep, attempt, t: int, step: int, split: str,
              correct: bool | None):
        """The round's single-pass write: the span generated in place after the
        attempt's ``memory_write()`` call, its items and keys, appended or superseding
        the task's own record."""
        kb = self.kb_of(row)
        reg = self.registries[kb.dataset]
        task = row['episode_id']
        site, mems, reads = write_source(attempt, self.proto)
        factor, n = span_length(max(attempt.generated, 1), self.level)
        span, = self.writer.generate([{'inputs': self.writer.inputs(site, mems, reads),
                                       'factor': factor}], [n])
        items = self.writer.encode(span)
        tag = f'eval{reg.generation}' if split == 'eval' else f's{step}'
        record_id = f'b9:{task}:{tag}:r{t}'
        if self.log is not None and split == 'train':
            self.log.add(record_id, kb=kb.dataset, prefix_ids=site, mems=mems, reads=reads,
                         factor=factor, span=span, step=step,
                         items={s: v.bfloat16().float() for s, v in items.items()},
                         rereads=self._rereads(attempt, kb.dataset))
            self.log.flush(step)
        previous = reg.own.get(task)
        ids = {}
        for s, values in items.items():
            key = self.reader.keys.item_key(s, values).float().cpu()
            item = NewItem(values.cpu(), key, Provenance((record_id,), 'codec', step), 1.0,
                           ep.query_time, reg.records[previous].items[s] if previous else None)
            if previous:
                kb.supersede(s, [item])
                ids[s] = item.id
            else:
                ids[s] = kb.append(s, [item])[0]
        self.ctx._rows = {}
        return reg.add_own(task, record_id, ids, round=t, step=step, split=split,
                           read_items=read_items(attempt.reads), correct=correct)

    @torch.no_grad()
    def write_gold(self, row: dict, ep, step: int):
        kb = self.kb_of(row)
        reg = self.registries[kb.dataset]
        task = row['episode_id']
        if task in reg.gold or self.schedule.weight(reg, task, step) <= 0:
            return None
        span = bank_write_spans(self.model, [gold_text(row)], self.level)[0]
        record_id = f'b9gold:{task}'
        items = self.writer.encode(span.float())
        ids = {}
        for s, values in items.items():
            key = self.reader.keys.item_key(s, values).float().cpu()
            ids[s] = kb.append(s, [NewItem(values.cpu(), key, Provenance((record_id,), 'codec',
                                                                          step), 1.0,
                                           ep.query_time)])[0]
        self.ctx._rows = {}
        return reg.add_gold(task, record_id, ids, step)

    # -- episodes -----------------------------------------------------------------------
    def train_episode(self, row: dict, ep, step: int) -> list[dict]:
        weights_by_round = round_weights(self.args.round_weights, self.args.rounds)
        if not self.args.no_gold:
            self.write_gold(row, ep, step)
        reg = self.registries[self.kb_of(row).dataset]
        task = row['episode_id']
        state = {}

        def view_for(t):
            state['view'], state['weights'] = self.view(row, 'train', step)
            return state['view']

        def attempt_fn(t, view):
            attempt, correct = self.attempt(row, ep, view, state['weights'])
            state['correct'] = correct
            return attempt

        def sft_fn(t, view):
            return self.sft(ep, view, state['weights'], weights_by_round[t], task)

        def write_fn(t, attempt):
            return self.write(row, ep, attempt, t, step, 'train', state['correct'])

        out = []
        self.begin_episode(row)
        rounds = rounds_loop(task, self.args.rounds, view_for, attempt_fn, sft_fn, write_fn)
        self.end_episode()
        for r in rounds:
            a = r['attempt']
            out.append({'round': r['round'], 'correct': r['record'].correct,
                        'generated': a.generated, 'reads': len(a.reads), 'stop': a.stop,
                        'kind': r['record'].kind, 'mass': read_mass(a.reads, reg, task),
                        'gold_weight': self.schedule.weight(reg, task, step), **r['sft']})
        return out

    @torch.no_grad()
    def evaluate(self, episodes: list, step: int) -> dict:
        """Held-out rounds, round-major, in the three arms (see the module docstring)."""
        for reg in self.registries.values():
            reg.generation += 1
        acc = {arm: [[] for _ in range(self.args.rounds)] for arm in ARMS}
        mass = defaultdict(list)
        samples = []
        for t in range(self.args.rounds):
            normal = []
            for i, (row, ep) in enumerate(episodes):
                other = episodes[(i + 1) % len(episodes)][0]['episode_id']
                for arm in ARMS:
                    if arm != 'normal' and (t == 0 or (arm == 'swapped' and len(episodes) < 2)):
                        continue
                    view, weights = self.view(row, 'eval', step, arm, other)
                    attempt, correct = self.attempt(row, ep, view, weights)
                    acc[arm][t].append(correct)
                    if arm == 'normal':
                        normal.append((row, ep, attempt, correct))
                        reg = self.registries[row['kb']]
                        for k, v in read_mass(attempt.reads, reg, row['episode_id']).items():
                            mass[f'{k}_r{t}'].append(v)
                        if len(samples) < 3:
                            samples.append({'episode': row['episode_id'], 'round': t,
                                            'answer': attempt.answer[:300], 'correct': correct,
                                            'reads': len(attempt.reads), 'stop': attempt.stop})
            for row, ep, attempt, correct in normal:
                self.write(row, ep, attempt, t, step, 'eval', correct)
        report = {arm: [round(sum(v) / len(v), 4) if v else None for v in rounds]
                  for arm, rounds in acc.items()}
        report['n'] = len(episodes)
        report['read_mass'] = {k: round(sum(v) / len(v), 4) for k, v in sorted(mass.items())}
        report['samples'] = samples
        report['records'] = {k: r.counts() for k, r in self.registries.items()}
        return report


def optimizer_groups(reader, producers: dict[str, list], args) -> list[dict]:
    """The reader at ``--lr``; with ``--backprop-rounds`` > 0 the codecs and the writer's
    span heads at their own (L1b) rates."""
    own = {id(p) for ps in producers.values() for p in ps}
    groups = [{'params': [p for p in reader.parameters() if p.requires_grad and id(p) not in own],
               'lr': args.lr, 'weight_decay': 0.01}]
    if args.backprop_rounds > 0:
        groups += [{'params': producers['codecs'], 'lr': args.l1b_codec_lr, 'weight_decay': 0.01},
                   {'params': producers['writer'], 'lr': args.l1b_writer_lr, 'weight_decay': 0.0}]
    return groups


def round_weights(text: str, rounds: int) -> list[float]:
    if not text:
        return [1.0 / rounds] * rounds
    values = [float(v) for v in text.split(',')]
    if len(values) != rounds:
        raise ValueError('one weight per round')
    return values


# -- stage ---------------------------------------------------------------------------------
def usable(row: dict, kbs: dict, tok, ctx, max_tokens: int, skipped: dict):
    from schnitz.kb.stages.l1 import layout
    reason = None
    if (row.get('verify') or {}).get('type') not in VERIFIABLE:
        reason = 'not_verifiable'
    elif row['kb'] not in kbs or not slots_of(row):
        reason = 'no_kb_or_reads'
    else:
        ep = layout(row, tok)
        if ep.ids.numel() > max_tokens:
            reason = 'too_long'
        elif not ctx.covered(ep):
            reason = 'records_missing'
    if reason:
        skipped[reason] = skipped.get(reason, 0) + 1
        return None
    return ep


def run(args) -> None:
    from schnitz.kb.stages.l1 import Context, Frozen, load_model, open_live
    from schnitz.kb.stages.l2 import load_reader
    if args.backprop_rounds < 0:
        raise ValueError('--backprop-rounds counts earlier writes (0: detached)')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    banks = json.loads((args.banks / 'banks.json').read_text())
    level = args.level or banks['level']
    model = load_model(args)
    lm = model.decoder.base_lm
    if args.decoder_checkpoint:
        from schnitz.bgkit_span import checkpoint_layers
        checkpoint_layers(lm.model.layers)
    frozen = Frozen(lm, args.query_layer, model.core.autocast)
    reader, config = load_reader(model, args.banks, args.l1_reader, args.seed)
    for p in reader.parameters():
        p.requires_grad_(True)
    producers = producer_params(model, reader.stack)
    for p in producers['codecs'] + producers['writer']:   # trained only through the replay
        p.requires_grad_(args.backprop_rounds > 0)
    args.output.mkdir(parents=True, exist_ok=True)
    kbs = open_live(args.banks, args.output, args.live_device, args.sync_every)
    ctx = Context(frozen, reader, kbs)
    exp = Experience(args, model, frozen, reader, kbs, ctx, GenProtocol.from_tokenizer(model.tok),
                     level)
    optimizer = torch.optim.AdamW(optimizer_groups(reader, producers, args), lr=args.lr)
    params = [p for group in optimizer.param_groups for p in group['params']]
    state_path = args.output / 'b9.pt'
    step = 0

    def save():
        tag = f'step{step:08d}-{time.time_ns()}'   # unique: an evaluation writes, too
        for kb in kbs.values():
            kb.checkpoint_live(tag)
        pending = state_path.with_suffix('.pending')
        torch.save({'reader': reader.state_dict(), 'optimizer': optimizer.state_dict(),
                    'writer': model.writer.state_dict(),
                    'step': step, 'rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                    'registries': {k: r.state() for k, r in exp.registries.items()},
                    'live_tag': tag, 'live_updates': {k: kb.live_updates for k, kb in kbs.items()},
                    'config': dataclasses.asdict(config)}, pending)
        pending.replace(state_path)
        for kb in kbs.values():
            for old in kb.live_checkpoints():
                if old != tag:
                    kb.drop_live_checkpoint(old)

    if state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        reader.load_state_dict(state['reader'])
        optimizer.load_state_dict(state['optimizer'])
        if 'writer' in state:
            model.writer.load_state_dict(state['writer'])
        step = state['step']
        rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        exp.registries = {k: Registry.from_state(v) for k, v in state['registries'].items()}
        for kb in kbs.values():       # the KBs exactly as paired with the checkpoint
            kb.restore_live(state['live_tag'], discard_commits=True)
        ctx._rows = {}
    else:
        save()                        # step 0: a restart always has a state to restore
    skipped: dict[str, int] = {}
    tok = model.tok
    train_rows = Transcripts(args.transcripts, 'train', args.limit)
    eval_eps = []
    for row in Transcripts(args.transcripts, 'validation', args.eval_limit):
        if len(eval_eps) >= args.eval_episodes:
            break
        ep = usable(row, kbs, tok, ctx, args.max_tokens, skipped)
        if ep is not None:
            eval_eps.append((row, ep))
    eval_skipped, skipped = dict(skipped), {}
    (args.output / 'config.json').write_text(json.dumps(dict(
        vars(args), level=level, read_config=dataclasses.asdict(config),
        train_transcripts=len(train_rows), eval_episodes=len(eval_eps),
        eval_skipped=eval_skipped, trained=sum(p.numel() for p in params),
        kbs={k: kb.stats() for k, kb in kbs.items()}), indent=2, default=str) + '\n')
    metrics = (args.output / 'metrics.jsonl').open('a', encoding='utf-8')

    def log(record):
        metrics.write(json.dumps(record) + '\n')
        metrics.flush()
        print(json.dumps(record), flush=True)

    if step == 0 and args.eval_every and eval_eps:
        log({'step': 0, 'eval': exp.evaluate(eval_eps, 0)})
        save()
    order: list[int] = []
    window: dict[str, list] = defaultdict(list)
    started = time.time()
    while step < args.steps:
        batch = []
        while len(batch) < args.batch_size:
            if not order:
                order = list(range(len(train_rows)))
                rng.shuffle(order)
            row = train_rows[order.pop()]
            ep = usable(row, kbs, tok, ctx, args.max_tokens, skipped)
            if ep is not None:
                batch.append((row, ep))
        reader.train()
        optimizer.zero_grad(set_to_none=True)
        for row, ep in batch:
            for r in exp.train_episode(row, ep, step):
                t = r['round']
                window[f'acc_r{t}'].append(float(bool(r['correct'])))
                window[f'nll_r{t}'].append(r['nll'])
                window[f'reads_r{t}'].append(r['reads'])
                window['gold_weight'].append(r['gold_weight'])
                for k, v in r['mass'].items():
                    window[f'mass_{k}_r{t}'].append(v)
                for k, v in r['sft_mass'].items():
                    window[f'sft_mass_{k}_r{t}'].append(v)
        for stats in exp.producer_stats:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    window[f'replay_{k}'].append(v)
        exp.producer_stats = []
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        step += 1
        if args.rekey_every and step % args.rekey_every == 0:
            for kb in kbs.values():
                reader.rekey(kb)
        if step % args.log_every == 0:
            log({'step': step, **{k: round(sum(v) / len(v), 4) for k, v in sorted(window.items())},
                 'records': {k: r.counts() for k, r in exp.registries.items()},
                 'skipped': dict(skipped), 'elapsed_s': round(time.time() - started)})
            window = defaultdict(list)
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            for kb in kbs.values():
                reader.rekey(kb)
            reader.eval()
            if eval_eps:
                log({'step': step, 'eval': exp.evaluate(eval_eps, step)})
            save()


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--banks', type=Path, required=True,
                        help='train.py l1 build output over the task transcripts')
    parser.add_argument('--l1-reader', type=Path,
                        help='L1 reader.pt to start from (default: the banks\' initial heads)')
    parser.add_argument('--transcripts', type=Path, nargs='+', required=True,
                        help='memory transcripts v3 of verifiable task corpora')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--reader-state', type=Path, required=True, help='B3 writer.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=2, help='episodes per optimizer step')
    parser.add_argument('--limit', type=int, help='train transcripts per directory')
    parser.add_argument('--eval-limit', type=int, default=256,
                        help='validation transcripts read per directory')
    parser.add_argument('--eval-episodes', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--level', help='ratio level of writes (default: the banks\')')
    parser.add_argument('--max-new', type=int, default=256, help='generated tokens per attempt')
    parser.add_argument('--max-reads', type=int, default=8, help='reads per attempt')
    parser.add_argument('--temperature', type=float, default=0.0, help='0: greedy')
    parser.add_argument('--open-with-search', action='store_true',
                        help='force each attempt to open with a memory_search() call (for a '
                             'decoder that has not learned the call yet)')
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--gold-start', type=float, default=1.0)
    parser.add_argument('--gold-anneal', type=int, default=0,
                        help='steps over which the global gold weight falls to 0 (0: constant)')
    parser.add_argument('--gold-decay', type=float, default=0.5,
                        help="per-task factor per supersede of the task's own record")
    parser.add_argument('--no-gold', action='store_true')
    parser.add_argument('--heldout-lineage', choices=('task', 'any'), default='task')
    parser.add_argument('--backprop-rounds', type=int, default=2,
                        help='earlier rounds\' writes the gradient of a round reaches through '
                             'the recomputed written items (L1b replay; 0: writes detached, '
                             'writer and codecs frozen)')
    parser.add_argument('--l1b-replay', choices=FEEDS, default='free',
                        help='producer replay of the writes (free: the stored forward, exact)')
    parser.add_argument('--l1b-codec-lr', type=float, default=3e-5)
    parser.add_argument('--l1b-writer-lr', type=float, default=3e-6)
    parser.add_argument('--round-weights', default='', help='SFT weight per round, e.g. 0.2,0.3,0.5')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--item-lr', type=float, default=0.0,
                        help='item values in place (L1a) during B9; 0: fixed')
    parser.add_argument('--retrieval-weight', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--query-layer', type=int, default=8)
    parser.add_argument('--rekey-every', type=int, default=25)
    parser.add_argument('--live-device', default='cpu')
    parser.add_argument('--sync-every', type=int, default=500)
    parser.add_argument('--decoder-checkpoint', action='store_true')
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--cuda-fraction', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=0)
