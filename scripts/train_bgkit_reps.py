"""B2 of the BGKit restart: the frozen S2 decoder learns to generate BGKit spans.

The decoder (with its S2 LoRA) is frozen. Trainable: the span marker, its
continuous ratio code, the rep head and the emit/stop head (``sdkb.bgkit_span``).
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
(``sdkb.bgkit_span.attach_write_adapter``: active only at span positions of a
write, so reading and ordinary text stay S2); ``--rollout-passes``/``--sample-*``
train on the writer's own reps (parallel passes, each feeding the previous
pass's reps at a ramping fraction of span positions); ``--gate-open-start`` /
``--gate-open-steps`` open the adapter on non-span positions from 0 to 1, so it
becomes a global LoRA; ``--merge-at`` then folds it and S2's LoRA exactly into the
weights and trains the whole decoder. Whenever the adapter acts outside spans or
the decoder is unfrozen, a replay KL to a frozen S2 copy on plain corpus text and
on reading teacher reps keeps S2's behaviour (``replay`` weight).

Evaluation on held-out bank sources at each space ratio and on BGKit's eval
stores at x4/x16/x64: task NLL with no context, full text, teacher reps, student
teacher-forced reps and student free-running reps (fed back, teacher length), the
captured fraction of the full-text gain, and the free-running stop-length error. Training-only; runs in the
``sdkb-bgkit`` container.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from sdkb.bgkit_span import (MEMORY_PROMPT, SUMMARIZE_PROMPTS, SpanWriter,
                             attach_write_adapter, span_mask, span_targets)

SPACES = ('s0', 's1', 's2', 's3')
ADAPTER_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj', 'w1', 'w2', 'w3')


def _heldout(record_id: str) -> bool:
    return int(hashlib.sha256(('b2-heldout:' + record_id).encode()).hexdigest()[:8], 16) % 200 == 0


class TeacherCache:
    """Random access into the B1 shards (memory-mapped safetensors)."""

    def __init__(self, root: Path, sources: Path):
        from safetensors import safe_open
        self.texts = [json.loads(line)['text'] for line in sources.open(encoding='utf-8')]
        shard_size = json.loads((root / 'manifest.json').read_text())['identity']['shard_size']
        self.handles, self.items = {}, []
        self.offsets: dict[tuple[int, str], torch.Tensor] = {}
        self.counts: dict[tuple[int, str], torch.Tensor] = {}
        self.factors: dict[tuple[int, str], torch.Tensor] = {}
        for meta in sorted(root.glob('shard-*.json')):
            shard = int(meta.stem.split('-')[1])
            if not meta.with_suffix('.safetensors').exists():
                continue
            info = json.loads(meta.read_text())
            handle = safe_open(str(meta.with_suffix('.safetensors')), framework='pt')
            self.handles[shard] = handle
            for tag in SPACES:
                counts = handle.get_tensor(f'{tag}_counts').long()
                self.counts[shard, tag] = counts
                self.offsets[shard, tag] = torch.cumsum(counts, 0) - counts
                self.factors[shard, tag] = 1.0 / handle.get_tensor(f'{tag}_ratio').float()
            base = shard * shard_size
            for row, (record_id, tokens) in enumerate(zip(info['record_ids'], info['tokens'])):
                self.items.append((shard, row, record_id, tokens, base + row))

    def reps(self, shard: int, row: int, tag: str) -> torch.Tensor:
        start = int(self.offsets[shard, tag][row])
        count = int(self.counts[shard, tag][row])
        return self.handles[shard].get_slice(f'{tag}_reps')[start:start + count]

    def factor(self, shard: int, row: int, tag: str) -> float:
        return float(self.factors[shard, tag][row])


class Model:
    def __init__(self, args):
        from bgkit_core.host_memory_guard import cap_cuda
        cap_cuda(args.cuda_fraction)
        from bgkit2.data.autoencode import Collator
        from bgkit2.data.templates import Templates, decoder_sentinel
        from bgkit2.training.standalone import load_models

        core = load_models(args.experiment, str(args.checkpoint))
        self.core, self.decoder, self.tok = core, core.decoder, core.tok
        for param in list(self.decoder.parameters()) + list(core.encoder.parameters()):
            param.requires_grad_(False)
        self.tpl = Templates.from_tokenizer(self.tok, style='chat')
        self.collate = Collator(self.tpl)
        self.encoder_prompts = Templates.compression_prompt_ids(self.tok)
        sent_str, sent_id = decoder_sentinel(self.tok)
        self.prompts = {}
        named = [('memory', MEMORY_PROMPT)] + [(f'summarize-{task}', text)
                                               for task, text in SUMMARIZE_PROMPTS.items()]
        for name, text in named:
            ids = self.tok.apply_chat_template([{'role': 'user', 'content': text + sent_str}],
                                               add_generation_prompt=True, tokenize=True)
            if hasattr(ids, 'input_ids'):
                ids = ids['input_ids']
            cut = ids.index(sent_id)
            self.prompts[name] = (torch.tensor(ids[:cut]), torch.tensor(ids[cut + 1:]))
        embed = self.decoder.embed_tokens.weight
        self.target_norm = float(embed.float().norm(dim=-1).mean())
        self.writer = SpanWriter(embed.shape[1], self.target_norm,
                                 marker_init=embed[sent_id]).to(core.device)
        self.instr = {task: ids.cpu() for task, ids in self.tpl.instr.items()}
        self.eos = torch.tensor([self.tpl.eos_id])
        self.device = core.device
        # frozen S2 copy for the replay KL, taken before any hook is attached
        opening = args.gate_open_start >= 0 or args.merge_at >= 0
        self.reference = copy.deepcopy(self.decoder) if opening else None
        self.gate, self.adapter, self.gate_value, self.merged = None, None, 0.0, False
        if args.adapter_rank:
            self.gate, self.adapter = attach_write_adapter(
                self.decoder.base_lm.model.layers, ADAPTER_TARGETS, args.adapter_rank,
                2.0 * args.adapter_rank)
            self.adapter.to(self.device)

    @property
    def replaying(self) -> bool:
        return self.reference is not None and (self.merged or self.gate_value > 0)

    def merge(self) -> None:
        """Fold the open write adapter and S2's LoRA into the weights; train the decoder."""
        if self.adapter is not None:
            if self.gate_value < 1.0:
                raise ValueError('merge needs the write adapter fully open')
            for adapter in self.adapter.values():
                adapter.merge()
        dec = self.decoder
        if dec._has_adapter:
            dec.lm = dec.lm.merge_and_unload()
            dec._has_adapter = False
        dec.base_lm.float()
        for param in dec.base_lm.parameters():
            param.requires_grad_(True)
        self.gate, self.adapter, self.merged = None, None, True

    def param_groups(self, args) -> list[dict]:
        groups = [{'params': list(self.writer.parameters()), 'lr': args.lr}]
        if self.adapter is not None:
            groups.append({'params': list(self.adapter.parameters()), 'lr': args.adapter_lr})
        if self.merged:
            groups.append({'params': list(self.decoder.base_lm.parameters()),
                           'lr': args.decoder_lr})
        return groups

    def trained_state(self) -> dict:
        state = {'writer': self.writer.state_dict()}
        if self.adapter is not None:
            state['adapter'] = self.adapter.state_dict()
        if self.merged:
            state['merged'] = True
            state['decoder'] = self.decoder.base_lm.state_dict()
        return state

    def load_trained(self, state: dict) -> None:
        self.writer.load_state_dict(state['writer'])
        if state.get('merged'):
            self.gate_value = 1.0
            self.merge()
            self.decoder.base_lm.load_state_dict(state['decoder'])
        elif 'adapter' in state and self.adapter is not None:
            self.adapter.load_state_dict(state['adapter'])

    def hidden(self, dec, inputs_embeds, attention_mask, spans=None) -> torch.Tensor:
        """Final hidden states; the write adapter weighs span positions 1 and all
        others ``gate_value`` (only on the trained decoder, never the reference)."""
        if dec is not self.decoder or self.gate is None or (spans is None
                                                           and self.gate_value == 0):
            return dec._hidden(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        weights = torch.full(attention_mask.shape + (1,), self.gate_value,
                             device=attention_mask.device)
        if spans is not None:
            weights = torch.maximum(weights, spans.float())
        with self.gate.active(weights):
            return dec._hidden(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def text_ids(self, text: str, limit: int = 1024) -> torch.Tensor:
        return torch.tensor(self.tok(text, add_special_tokens=False)['input_ids'][:limit])

    # -- writer ---------------------------------------------------------
    def write(self, examples, feed: list[torch.Tensor]):
        """Writer forward with ``feed`` reps as span inputs (teacher- or self-fed).

        Returns per example the span-position hidden states (marker, feed_1..)."""
        embed = self.decoder.embed_tokens
        seqs, starts = [], []
        for ex, fed in zip(examples, feed):
            pre, post = self.prompts[ex['prompt']]
            ids = torch.cat([pre, ex['ids'], post]).to(self.device)
            marker = self.writer.marker_embedding(
                torch.tensor(ex['factor'], device=self.device)).unsqueeze(0)
            seqs.append(torch.cat([embed(ids).float(), marker, fed.to(self.device).float()]))
            starts.append(ids.shape[0])
        width = max(s.shape[0] for s in seqs)
        dim = seqs[0].shape[1]
        inputs = torch.zeros(len(seqs), width, dim, device=self.device, dtype=embed.weight.dtype)
        mask = torch.zeros(len(seqs), width, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            inputs[i, :seq.shape[0]] = seq.to(inputs.dtype)
            mask[i, :seq.shape[0]] = 1
        spans = span_mask(starts, [1 + fed.shape[0] for fed in feed], width, self.device)
        hidden = self.hidden(self.decoder, inputs, mask, spans)
        return [hidden[i, starts[i]:starts[i] + 1 + feed[i].shape[0]] for i in range(len(seqs))]

    def free_run(self, examples, lengths: list[int]):
        """Self-fed generation of ``lengths[i]`` reps; also the first predicted stop."""
        reps = [torch.zeros(0, self.decoder.embed_tokens.weight.shape[1], device=self.device)
                for _ in examples]
        stops = [None] * len(examples)
        for step in range(max(lengths)):
            hidden = self.write(examples, reps)
            for i, h in enumerate(hidden):
                last = h[-1:]
                if stops[i] is None and self.writer.stop(last.float()).argmax(-1).item() == 1:
                    stops[i] = step
                if step < lengths[i]:
                    reps[i] = torch.cat([reps[i], self.writer.rep(last)])
        for i, h in enumerate(self.write(examples, reps)):
            if stops[i] is None and self.writer.stop(h[-1:].float()).argmax(-1).item() == 1:
                stops[i] = reps[i].shape[0]
        return reps, stops

    # -- reader ---------------------------------------------------------
    @torch.no_grad()
    def teacher(self, samples, factors: list[float]) -> list[torch.Tensor]:
        """Frozen S2 encoder reps (BGKit's own compression prompt) at per-row ratios."""
        batch = self.collate(samples).to(self.device)
        ratio = torch.tensor([1.0 / f for f in factors], device=self.device)
        with self.core.autocast():
            out = self.core.encode(batch, ratio)
        return [out.reps[i][out.rep_mask[i]].to(torch.bfloat16) for i in range(len(samples))]

    def read(self, examples, reps: list[torch.Tensor] | None, full: bool = False,
             decoder=None, index: torch.Tensor | None = None):
        """S2 layout for each example's task; returns (target logits (N, V), target ids (N,))."""
        dec = decoder or self.decoder
        suffix = [torch.cat([self.instr[ex['task']], ex['target'], self.eos]) for ex in examples]
        start = [self.instr[ex['task']].shape[0] for ex in examples]
        prefix = [self.tpl.prefix.cpu()] * len(examples)
        if full:
            reps = [dec.embed(ex['ids'].to(self.device)) for ex in examples]
        if reps is None:
            batch = dec.build_batch(prefix, None, None, suffix, suffix_label_start=start)
        else:
            width = max(r.shape[0] for r in reps)
            padded = torch.zeros(len(reps), max(width, 1), reps[0].shape[1], device=self.device)
            mask = torch.zeros(len(reps), max(width, 1), dtype=torch.bool, device=self.device)
            for i, r in enumerate(reps):
                padded[i, :r.shape[0]] = r.float()
                mask[i, :r.shape[0]] = True
            batch = dec.build_batch(prefix, padded, mask, suffix, suffix_label_start=start,
                                    decoder_space=True)
        hidden = self.hidden(dec, batch.inputs_embeds, batch.attention_mask)
        targets = batch.labels[:, 1:]
        valid = targets != -100
        chosen = hidden[:, :-1][valid]
        targets = targets[valid]
        if index is not None:
            chosen, targets = chosen[index], targets[index]
        return dec.base_lm.lm_head(chosen).float(), targets

    def text_logits(self, examples, decoder=None, index: torch.Tensor | None = None):
        """Plain-text next-token logits over each example's source (no span, no chat)."""
        dec = decoder or self.decoder
        ids = [torch.cat([torch.tensor([self.tpl.bos_id]), ex['ids']]) for ex in examples]
        width = max(x.shape[0] for x in ids)
        batch = torch.full((len(ids), width), self.tpl.pad_id, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for i, x in enumerate(ids):
            batch[i, :x.shape[0]], mask[i, :x.shape[0]] = x, 1
        batch, mask = batch.to(self.device), mask.to(self.device)
        hidden = self.hidden(dec, dec.embed(batch), mask)
        chosen = hidden[:, :-1][mask[:, 1:].bool()]
        return dec.base_lm.lm_head(chosen if index is None else chosen[index]).float()


def _example(cache: TeacherCache, model: Model, item, tag: str) -> dict:
    shard, row, record_id, tokens, source = item
    ids = model.text_ids(cache.texts[source])
    return {'ids': ids, 'target': ids, 'task': 'reconstruct', 'prompt': 'memory',
            'teacher': cache.reps(shard, row, tag), 'factor': cache.factor(shard, row, tag)}


def _classical(model: Model, samples, factors: list[float]) -> list[dict]:
    teacher = model.teacher(samples, factors)
    return [{'ids': s.ctx_ids.long(), 'target': s.target_ids.long(), 'task': s.task,
             'prompt': f'summarize-{s.task}', 'teacher': t, 'factor': f}
            for s, t, f in zip(samples, teacher, factors)]


def _classical_dataset(model: Model, stores, weights, ctx_max: int, seed: int, size=None):
    from bgkit2.data.autoencode import AutoencodeDataset
    from bgkit2.data.token_store import TokenStore
    return AutoencodeDataset([TokenStore(p) for p in stores], weights, ctx_min=64,
                             ctx_max=ctx_max, cont_min=32, cont_max=256, p_continue=0.5,
                             seed=seed, epoch_size=size, task_prompt_ids=model.encoder_prompts)


def _batches(items, rng: random.Random, batch_size: int, budget: int):
    while True:
        pool = rng.sample(items, min(len(items), 64 * batch_size))
        pool.sort(key=lambda item: item[3])
        groups, group = [], []
        for item in pool:
            if group and (len(group) >= batch_size or (len(group) + 1) * item[3] > budget):
                groups.append(group)
                group = []
            group.append(item)
        if group:
            groups.append(group)
        rng.shuffle(groups)
        yield from groups


def _kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, -1), F.log_softmax(target, -1), log_target=True,
                    reduction='batchmean')


def _subset(total: int, limit: int, device) -> torch.Tensor:
    return torch.randperm(total, device=device)[:limit]


def train_step(model: Model, examples, weights: dict, passes: int = 0,
               sample: float = 0.0, replay_tokens: int = 1024) -> dict:
    """One step. ``passes`` > 0 with ``sample`` > 0 trains on the writer's own reps: a
    fixed random fraction ``sample`` of span inputs is replaced by the previous pass's
    predictions (detached); gradients flow through the final pass."""
    writer = model.writer
    teacher_feed = [ex['teacher'].to(model.device).float() for ex in examples]
    feed = teacher_feed
    with model.core.autocast():
        if passes and sample > 0:
            own = [torch.rand(t.shape[0], 1, device=model.device) < sample for t in teacher_feed]
            for _ in range(passes):
                with torch.no_grad():
                    states = model.write(examples, feed)
                    preds = [writer.rep(h[:-1]) for h in states]
                feed = [torch.where(m, p, t) for m, p, t in zip(own, preds, teacher_feed)]
        # k+1 states (marker, R_1..R_k): the first k predict R_1..R_k, all k+1 emit/stop
        full = model.write(examples, feed)
        counts = [t.shape[0] for t in teacher_feed]
        preds = [writer.rep(h[:-1]) for h in full]
        pred = torch.cat(preds)
        cos = 1 - F.cosine_similarity(pred, torch.cat(teacher_feed), dim=-1).mean()
        stop_logits = writer.stop(torch.cat(full).float())
        stop = F.cross_entropy(stop_logits, span_targets(counts, model.device))
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
    loss.backward()
    return {'loss': loss.item(), **result, 'reps': len(pred)}


@torch.no_grad()
def _score(model: Model, groups) -> dict:
    sums: dict[str, float] = {}
    tokens, count, length_err, stop_hits = 0, 0, 0.0, 0
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
            stop_hits += stop == k
            length_err += abs((k if stop is None else stop) - k) / max(k, 1)
    nll = {name: value / tokens for name, value in sums.items()}
    extra = {}
    if model.reference is not None:
        extra = {'replay_kl': round(replay / max(replay_n, 1), 4)}
    gain = max(nll['noctx'] - nll['full'], 1e-9)
    return {'nll': {k: round(v, 4) for k, v in nll.items()},
            'captured': {k: round((nll['noctx'] - nll[k]) / gain, 4)
                         for k in ('teacher', 'student_tf', 'student_free')},
            'stop_exact': round(stop_hits / count, 4),
            'length_rel_err': round(length_err / count, 4), **extra}


def evaluate(model: Model, cache: TeacherCache, items, classical, batch_size: int) -> dict:
    out = {}
    for tag in SPACES:
        out[f'bank/{tag}'] = _score(model, (
            [_example(cache, model, item, tag) for item in items[i:i + batch_size]]
            for i in range(0, len(items), batch_size)))
    for factor in (4.0, 16.0, 64.0):
        out[f'classical/x{int(factor)}'] = _score(model, (
            _classical(model, classical[i:i + batch_size], [factor] * len(classical[i:i + batch_size]))
            for i in range(0, len(classical), batch_size)))
    return out


def _gate(args, step: int) -> float:
    if args.gate_open_start < 0 or step < args.gate_open_start:
        return 0.0
    return min(1.0, (step - args.gate_open_start + 1) / max(args.gate_open_steps, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=64)
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
    args = parser.parse_args()
    weights = {k: float(v) for k, v in (pair.split('=') for pair in args.weights.split(','))}

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
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'writer.pt'
    step = 0
    state = None
    if state_path.exists():
        state = torch.load(state_path, map_location=model.device)
        step = state['step']
        model.gate_value = _gate(args, step)
        model.load_trained(state)
    elif args.init_writer:
        model.writer.load_state_dict(
            torch.load(args.init_writer, map_location=model.device)['writer'])

    def build_optimizer(start: int):
        optimizer = torch.optim.AdamW(model.param_groups(args), weight_decay=0.01)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + start + 1) / args.warmup))
        return optimizer, schedule

    if 0 <= args.merge_at < args.gate_open_start + args.gate_open_steps - 1 and args.adapter_rank:
        raise ValueError('--merge-at must come after the gate is fully open')
    optimizer, schedule = build_optimizer(step - args.merge_at if model.merged else step)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    metrics = (args.output / 'metrics.jsonl').open('a', encoding='utf-8')
    config = dict(vars(args), train_items=len(train), heldout_items=len(heldout),
                  target_norm=model.target_norm, adapter_targets=ADAPTER_TARGETS, memory_prompt=MEMORY_PROMPT,
                  summarize_prompts=SUMMARIZE_PROMPTS)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')

    def log(record):
        metrics.write(json.dumps(record) + '\n')
        metrics.flush()
        print(json.dumps(record), flush=True)

    if step == 0:
        log({'step': 0, 'eval': evaluate(model, cache, heldout, classical_eval,
                                         args.classical_batch)})
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    window: dict[str, float] = {}
    started = time.time()
    while step < args.steps:
        if rng.random() < args.classical_fraction:
            samples, budget = [], args.batch_tokens
            while len(samples) < args.classical_batch:
                sample = classical_train[rng.randrange(len(classical_train))]
                budget -= sample.target_ids.shape[0] + sample.ctx_ids.shape[0]
                if samples and budget < 0:
                    break
                samples.append(sample)
            factors = [2 ** rng.uniform(0, 7) for _ in samples]  # x1 .. x128, log-uniform
            examples = _classical(model, samples, factors)
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
        result = train_step(model, examples, weights, args.rollout_passes, sample,
                            args.replay_tokens)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                       1.0)
        optimizer.step()
        schedule.step()
        step += 1
        for key, value in result.items():
            window[f'{stream}/{key}'] = window.get(f'{stream}/{key}', 0.0) + value
            window[f'{stream}/n'] = window.get(f'{stream}/n', 0.0) + 1 / len(result)
        if step % args.log_every == 0:
            means = {k: v / max(window[k.split('/')[0] + '/n'], 1) for k, v in window.items()
                     if not k.endswith('/n')}
            log({'step': step, **{k: round(v, 4) for k, v in means.items()},
                 'lr': schedule.get_last_lr()[0], 'elapsed_s': round(time.time() - started)})
            window = {}
        if step % args.eval_every == 0 or step == args.steps:
            torch.save({**model.trained_state(), 'optimizer': optimizer.state_dict(),
                        'step': step}, state_path.with_suffix('.pending'))
            state_path.with_suffix('.pending').replace(state_path)
            log({'step': step, 'eval': evaluate(model, cache, heldout, classical_eval,
                                                args.classical_batch)})


if __name__ == '__main__':
    main()
