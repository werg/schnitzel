"""B2 of the BGKit restart: the frozen S2 decoder learns to generate BGKit spans.

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
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from schnitz.bgkit_span import (MEMORY_PROMPT, MERGE_PROMPT, SUMMARIZE_PROMPTS, SpanWriter,
                             attach_write_adapter, checkpoint_layers, span_mask,
                             span_targets)

SPACES = ('s0', 's1', 's2', 's3')
ADAPTER_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj', 'w1', 'w2', 'w3')


def _heldout(record_id: str) -> bool:
    return int(hashlib.sha256(('b2-heldout:' + record_id).encode()).hexdigest()[:8], 16) % 200 == 0


class TeacherCache:
    """Random access into the B1 shards (memory-mapped safetensors)."""

    def __init__(self, root: Path, sources: Path | None, texts: list[str] | None = None):
        from safetensors import safe_open
        self.texts = texts if texts is not None else [
            json.loads(line)['text'] for line in sources.open(encoding='utf-8')]
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
        self.checkpoint_merged = args.merge_checkpoint
        self.stop_pos_weight = getattr(args, 'stop_pos_weight', 1.0)
        self.core, self.decoder, self.tok = core, core.decoder, core.tok
        for param in list(self.decoder.parameters()) + list(core.encoder.parameters()):
            param.requires_grad_(False)
        self.tpl = Templates.from_tokenizer(self.tok, style='chat')
        self.collate = Collator(self.tpl)
        self.encoder_prompts = Templates.compression_prompt_ids(self.tok)
        sent_str, sent_id = decoder_sentinel(self.tok)
        self.prompts = {}
        named = [('memory', MEMORY_PROMPT), ('merge', MERGE_PROMPT)] + [(f'summarize-{task}', text)
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
        if self.checkpoint_merged:
            checkpoint_layers(dec.base_lm.model.layers)
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
            if not self.merged:  # a resumed merged state is loaded into place
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

    def question_instr(self, query: str) -> torch.Tensor:
        """Decoder instruction after a context slot: the question, then the answer turn."""
        from bgkit2.data.templates import decoder_sentinel
        sent_str, sent_id = decoder_sentinel(self.tok)
        ids = self.tok.apply_chat_template(
            [{'role': 'user', 'content': f'Context:\n{sent_str}\n{query}'}],
            add_generation_prompt=True, tokenize=True)
        if hasattr(ids, 'input_ids'):
            ids = ids['input_ids']
        return torch.tensor(ids[ids.index(sent_id) + 1:])

    def text_ids(self, text: str, limit: int = 1024) -> torch.Tensor:
        return torch.tensor(self.tok(text, add_special_tokens=False)['input_ids'][:limit])

    # -- writer ---------------------------------------------------------
    @torch.no_grad()
    def prefix(self, examples) -> dict:
        """Run each write's prompt and source once and keep what a span continuation
        needs: per attention layer the keys and values, per convolution layer the
        last ``L_cache - 1`` convolution inputs (B * x) at each row's own end.

        Computed with the current weights and the write adapter at its current
        weight outside spans, so it is exact for no-gradient passes; a gradient pass
        uses it only while nothing trainable acts on the prefix (``_rollout``)."""
        inner = self.decoder.base_lm.model
        if any('source_embeds' in ex for ex in examples):
            seqs = [self.write_inputs(ex) for ex in examples]
            lengths = torch.tensor([x.shape[0] for x in seqs], device=self.device)
            embeds = torch.zeros(len(seqs), int(lengths.max()), seqs[0].shape[1],
                                 device=self.device)
            mask = torch.zeros(embeds.shape[:2], dtype=torch.long, device=self.device)
            for i, x in enumerate(seqs):
                embeds[i, :x.shape[0]], mask[i, :x.shape[0]] = x, 1
            batch = mask
        else:
            embeds = None
            ids = []
            for ex in examples:
                pre, post = self.prompts[ex['prompt']]
                ids.append(torch.cat([pre, ex['ids'], post]))
            lengths = torch.tensor([x.shape[0] for x in ids], device=self.device)
            batch = torch.full((len(ids), int(lengths.max())), self.tpl.pad_id, dtype=torch.long)
            mask = torch.zeros(batch.shape, dtype=torch.long)
            for i, x in enumerate(ids):
                batch[i, :x.shape[0]], mask[i, :x.shape[0]] = x, 1
            batch, mask = batch.to(self.device), mask.to(self.device)
        conv, hooks = {}, []
        rows = torch.arange(len(examples), device=self.device)
        for index, layer in enumerate(inner.layers):
            if layer.is_attention_layer:
                continue
            keep = layer.conv.L_cache - 1

            def capture(module, inputs, output, index=index, keep=keep):
                b, _, x = output.chunk(3, dim=-1)
                bx = b * x
                at = lengths[:, None] - keep + torch.arange(keep, device=self.device)[None]
                conv[index] = bx[rows[:, None], at]
            hooks.append(layer.conv.in_proj.register_forward_hook(capture))
        scope = contextlib.nullcontext()
        if self.gate is not None and self.gate_value > 0:
            scope = self.gate.active(torch.full(batch.shape + (1,), self.gate_value,
                                                device=self.device))
        try:
            with self.core.autocast(), scope:
                out = inner(inputs_embeds=self.decoder.embed(batch) if embeds is None else embeds,
                            attention_mask=mask, use_cache=True)
        finally:
            for hook in hooks:
                hook.remove()
        cache = out.past_key_values
        kv = {index: (cache.layers[index].keys, cache.layers[index].values)
              for index, layer in enumerate(inner.layers) if layer.is_attention_layer}
        return {'lengths': lengths, 'mask': mask.bool(), 'kv': kv, 'conv': conv}

    def span_hidden(self, prefix: dict, spans: list[torch.Tensor]) -> list[torch.Tensor]:
        """Final hidden states of span inputs continuing each row's cached prefix
        (LFM2 layer maths replicated: RMS norms, attention with RoPE at each row's own
        offset over prefix + causal span keys, short causal convolutions seeded with
        the prefix's last inputs). The write adapter is active on every span position."""
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb
        inner = self.decoder.base_lm.model
        embed = self.decoder.embed_tokens.weight
        size = [x.shape[0] for x in spans]
        width = max(size)
        x = torch.zeros(len(spans), width, embed.shape[1], device=self.device, dtype=embed.dtype)
        for i, span in enumerate(spans):
            x[i, :span.shape[0]] = span.to(x.dtype)
        position = prefix['lengths'][:, None] + torch.arange(width, device=self.device)[None]
        cos, sin = inner.rotary_emb(x, position_ids=position)
        causal = torch.ones(width, width, dtype=torch.bool, device=self.device).tril()
        allowed = torch.cat([prefix['mask'][:, None, None, :].expand(-1, 1, width, -1),
                             causal[None, None].expand(len(spans), 1, -1, -1)], dim=-1)
        scope = (self.gate.active(torch.ones(len(spans), width, 1, device=self.device))
                 if self.gate is not None else contextlib.nullcontext())
        with scope:
            for index, layer in enumerate(inner.layers):
                residual, h = x, layer.operator_norm(x)
                if layer.is_attention_layer:
                    att = layer.self_attn
                    shape = (len(spans), width, -1, att.head_dim)
                    q = att.q_layernorm(att.q_proj(h).view(shape)).transpose(1, 2)
                    k = att.k_layernorm(att.k_proj(h).view(shape)).transpose(1, 2)
                    v = att.v_proj(h).view(shape).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    pk, pv = prefix['kv'][index]
                    k = torch.cat([pk.to(k.dtype), k], dim=2)
                    v = torch.cat([pv.to(v.dtype), v], dim=2)
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                       scale=att.scaling, enable_gqa=True)
                    h = att.out_proj(o.transpose(1, 2).reshape(len(spans), width, -1))
                else:
                    conv = layer.conv
                    b, c, xx = conv.in_proj(h).chunk(3, dim=-1)
                    keep = conv.L_cache - 1
                    z = torch.cat([prefix['conv'][index].to(b.dtype), b * xx], dim=1)
                    out = conv.conv(z.transpose(1, 2))[..., keep:keep + width]
                    h = conv.out_proj((c.transpose(1, 2) * out).transpose(1, 2))
                x = h + residual
                x = x + layer.feed_forward(layer.ffn_norm(x))
            x = inner.embedding_norm(x)
        return [x[i, :n] for i, n in enumerate(size)]

    def write(self, examples, feed: list[torch.Tensor], prefix: dict | None = None):
        """Writer forward with ``feed`` reps as span inputs (teacher- or self-fed).

        Returns per example the span-position hidden states (marker, feed_1..). With a
        ``prefix`` (``Model.prefix``) only the span positions are computed."""
        if prefix is not None:
            spans = [torch.cat([self.writer.marker_embedding(
                torch.tensor(ex['factor'], device=self.device)).unsqueeze(0),
                fed.to(self.device).float()]) for ex, fed in zip(examples, feed)]
            return self.span_hidden(prefix, spans)
        embed = self.decoder.embed_tokens
        seqs, starts = [], []
        for ex, fed in zip(examples, feed):
            source = self.write_inputs(ex)
            marker = self.writer.marker_embedding(
                torch.tensor(ex['factor'], device=self.device)).unsqueeze(0)
            seqs.append(torch.cat([source, marker, fed.to(self.device).float()]))
            starts.append(source.shape[0])
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

    def write_inputs(self, ex: dict) -> torch.Tensor:
        """The writer's prompt and source as input embeddings. The source is text
        (``ids``) or, for a merge, stored spans in the decoder's input space
        (``source_embeds``, records separated by a blank line)."""
        pre, post = self.prompts[ex['prompt']]
        embed = self.decoder.embed_tokens
        if 'source_embeds' not in ex:
            return embed(torch.cat([pre, ex['ids'], post]).to(self.device)).float()
        return torch.cat([embed(pre.to(self.device)).float(),
                          ex['source_embeds'].to(self.device).float(),
                          embed(post.to(self.device)).float()])

    def free_run(self, examples, lengths: list[int]):
        """Self-fed generation of ``lengths[i]`` reps; also the first predicted stop."""
        reps = [torch.zeros(0, self.decoder.embed_tokens.weight.shape[1], device=self.device)
                for _ in examples]
        stops = [None] * len(examples)
        prefix = self.prefix(examples)
        for step in range(max(lengths)):
            hidden = self.write(examples, reps, prefix)
            for i, h in enumerate(hidden):
                last = h[-1:]
                if stops[i] is None and self.writer.stop(last.float()).argmax(-1).item() == 1:
                    stops[i] = step
                if step < lengths[i]:
                    reps[i] = torch.cat([reps[i], self.writer.rep(last)])
        for i, h in enumerate(self.write(examples, reps, prefix)):
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
        # an example may bring its own instruction (e.g. a question) instead of the task's
        instr = [ex['instr'] if 'instr' in ex else self.instr[ex['task']] for ex in examples]
        suffix = [torch.cat([ins, ex['target'], self.eos]) for ins, ex in zip(instr, examples)]
        start = [ins.shape[0] for ins in instr]
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

    def build(self, model: Model, row: dict, tag: str, rng: random.Random, related: int):
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
        return records, view


def qa_step(model: Model, episodes: QAEpisodes, rows, weights: dict, rng: random.Random,
            related: int, passes: int, sample: float, sequential: int = 0) -> dict:
    """Writer writes the records (question-free); the reader answers from them."""
    tag = rng.choice(SPACES)
    built = [episodes.build(model, row, tag, rng, related) for row in rows]
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
    loss.backward()
    return {'loss': loss.item(), 'qa_nll': nll.item(), 'cos': cos.item(), 'stop': stop.item(),
            'records': len(records)}


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
                                                      - nll['student_free'], 4)}}
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
    args = parser.parse_args()
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
        log({'step': 0, 'eval': run_eval()})
    batches = _batches(train, rng, args.batch_size, args.batch_tokens)
    window: dict[str, float] = {}
    started = time.time()
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
        if stream == 'qa':
            result = qa_step(model, qa_train, rng.sample(qa_train.rows, args.qa_batch), weights,
                             rng, rng.randint(0, args.qa_related), args.rollout_passes, sample,
                             args.sequential_reps)
        else:
            result = train_step(model, examples, weights, args.rollout_passes, sample,
                                args.replay_tokens, args.sequential_reps)
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
            log({'step': step, 'eval': run_eval()})


if __name__ == '__main__':
    main()
