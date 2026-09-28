"""The decoder with its span writer, protocol tokens and ports: the model every
training stage shares (BGKit S2 decoder, LFM2.5-350M; restart plan and
docs/knowledge-base-stack.md). Moved from ``scripts/train_bgkit_reps.py``.

``LEVELS`` are the writer's ratio levels of the B1 teacher cache (not KB spaces).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from schnitz.bgkit_span import (MEMORY_PROMPT, MERGE_PROMPT, SUMMARIZE_PROMPTS, PortHeads,
                                SpanWriter, attach_write_adapter, checkpoint_layers, span_mask)
from schnitz.span_protocol import ProtocolTokens, untie

LEVELS = ('s0', 's1', 's2', 's3')
ADAPTER_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj', 'w1', 'w2', 'w3')


def length_factors(tokens: int, spaces: int = 4) -> list[float]:
    """Compression factor per space for a source of ``tokens`` tokens: the B1 cache's
    length schedule (``scripts/cache_bgkit_teacher.py``), c_0 = clamp(sqrt(N)/2, 4, 32),
    c_s = min(128, c_0 * 2^s)."""
    base = min(32.0, max(4.0, math.sqrt(max(tokens, 1)) / 2))
    return [min(128.0, base * 2 ** space) for space in range(spaces)]


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
            for tag in LEVELS:
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
        # B4: prompts that state the compression factor (powers of two, x1 .. x128)
        named += [(f'summarize-{task}@{2 ** e}', text.replace('with BGKit', f'with BGKit at about x{2 ** e}'))
                  for task, text in SUMMARIZE_PROMPTS.items() for e in range(8)]
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
        self.protocol, self.protocol_losses, self.record_protocol = None, [], False
        self.port = None  # B4b soft output port heads (``install_port``)
        self.tail = torch.tensor(self.tok('<|im_end|>', add_special_tokens=False)['input_ids'])
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

    def install_protocol(self) -> None:
        """B4: span protocol tokens with LM-head rows (``schnitz.span_protocol``) on the
        merged decoder; the writer's marker and stop head become the ``<|bg|>`` input
        embedding and the ``<|rep|>``/``<|/bg|>`` rows (initialized from them)."""
        if not self.merged:
            raise ValueError('the span protocol is installed on the merged decoder')
        lm = self.decoder.base_lm
        untie(lm)
        embed = lm.get_input_embeddings()

        def text(words: str) -> torch.Tensor:
            ids = self.tok(words, add_special_tokens=False)['input_ids']
            return embed.weight.detach()[ids].float().mean(0)

        stop = self.writer.stop
        init = {'inputs:bg': self.writer.marker.detach(), 'inputs:bg_end': text('\n'),
                'inputs:mem': text('Memory:'), 'inputs:mem_end': text('\n'),
                'inputs:port': text('Question:'), 'inputs:port_end': text('\n'),
                'outputs:rep': stop.weight[0].detach(), 'outputs:bg_end': stop.weight[1].detach()}
        protocol = ProtocolTokens(embed, lm.lm_head, self.writer.ratio, init).to(self.device)
        protocol.span_bias.data.copy_(stop.bias.detach())
        protocol.install(embed, lm.lm_head)

        class SpanStop(torch.nn.Module):  # the writer's stop decision is now the LM rows
            def forward(self, hidden):
                return protocol.span_logits(hidden)

        self.writer.stop = SpanStop()
        self.writer.marker_embedding = protocol.marker
        self.protocol = protocol

    def install_port(self) -> None:
        """B4b: the soft output port's own rep and stop heads (``PortHeads``). The
        marker is the protocol's ``<|port|>`` input embedding plus the writer's ratio
        code at x1; the heads start as copies of the writer's (rep head, and the
        ``<|rep|>``/``<|/bg|>`` rows with their bias as the stop head), so the first
        port reps are already writer-like, then train on their own."""
        if self.protocol is None:
            raise ValueError('the soft output port needs the span protocol (--protocol)')
        protocol = self.protocol
        port = PortHeads(self.writer.marker.shape[0], self.target_norm,
                         lambda factor: protocol.embedding('port')
                         + protocol.ratio(torch.ones_like(factor)))
        port.rep.load_state_dict(self.writer.rep.state_dict())
        with torch.no_grad():
            port.stop.weight.copy_(protocol.outputs[[protocol.index('rep'),
                                                     protocol.index('bg_end')]].float())
            port.stop.bias.copy_(protocol.span_bias.float())
        self.port = port.to(self.device)

    def port_reps(self, ids: list[torch.Tensor]) -> list[torch.Tensor]:
        """Soft input port: the frozen S2 encoder's x1 reps of each text."""
        from bgkit2.data.autoencode import Sample
        samples = [Sample(ctx_ids=x.long(), target_ids=x.long()[:1], task='reconstruct',
                          store=0, doc=0) for x in ids]
        return self.teacher(samples, [1.0] * len(samples))

    def param_groups(self, args) -> list[dict]:
        groups = [{'params': list(self.writer.parameters()), 'lr': args.lr}]
        if self.protocol is not None:
            groups.append({'params': list(self.protocol.parameters()),
                           'lr': getattr(args, 'protocol_lr', args.lr)})
        if self.adapter is not None:
            groups.append({'params': list(self.adapter.parameters()), 'lr': args.adapter_lr})
        if self.merged:
            groups.append({'params': list(self.decoder.base_lm.parameters()),
                           'lr': args.decoder_lr})
        if self.port is not None:  # last, so a state saved before the port still loads
            groups.append({'params': list(self.port.parameters()),
                           'lr': getattr(args, 'port_lr', args.lr)})
        return groups

    def trained_state(self) -> dict:
        state = {'writer': self.writer.state_dict()}
        if self.adapter is not None:
            state['adapter'] = self.adapter.state_dict()
        if self.merged:
            state['merged'] = True
            state['decoder'] = self.decoder.base_lm.state_dict()
        if self.protocol is not None:
            state['protocol'] = self.protocol.state_dict()
        if self.port is not None:
            state['port'] = self.port.state_dict()
        return state

    def load_trained(self, state: dict) -> None:
        if state.get('merged') and not self.merged:  # a resumed merged state is loaded into place
            self.gate_value = 1.0
            self.merge()
        if 'protocol' in state and self.protocol is None:
            self.install_protocol()  # unties the LM head before the decoder state is loaded
        self.writer.load_state_dict(state['writer'])
        if state.get('merged'):
            self.decoder.base_lm.load_state_dict(state['decoder'])
        elif 'adapter' in state and self.adapter is not None:
            self.adapter.load_state_dict(state['adapter'])
        if 'protocol' in state:
            self.protocol.load_state_dict(state['protocol'])
        if 'port' in state:  # absent in states saved before B4b
            if self.port is None:
                self.install_port()
            self.port.load_state_dict(state['port'])

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
        if any(key in ex for ex in examples for key in ('source_embeds', 'inputs', 'prefix_ids')):
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

    def write(self, examples, feed: list[torch.Tensor], prefix: dict | None = None, heads=None):
        """Writer forward with ``feed`` reps as span inputs (teacher- or self-fed).

        Returns per example the span-position hidden states (marker, feed_1..). With a
        ``prefix`` (``Model.prefix``) only the span positions are computed. ``heads`` is
        the head set whose marker opens the span (default the memory writer; the soft
        output port's ``PortHeads`` open ``<|port|>`` and close ``<|/port|>``)."""
        heads = self.writer if heads is None else heads
        opening, closing = getattr(heads, 'tokens', ('bg', 'bg_end'))
        if prefix is not None:
            spans = [torch.cat([heads.marker_embedding(
                torch.tensor(ex['factor'], device=self.device)).unsqueeze(0),
                fed.to(self.device).float()]) for ex, fed in zip(examples, feed)]
            return self.span_hidden(prefix, spans)
        embed = self.decoder.embed_tokens
        seqs, starts = [], []
        protocol = self.protocol is not None and (torch.is_grad_enabled() or self.record_protocol)
        for ex, fed in zip(examples, feed):
            source = self.write_inputs(ex)
            marker = heads.marker_embedding(
                torch.tensor(ex['factor'], device=self.device)).unsqueeze(0)
            parts = [source, marker, fed.to(self.device).float()]
            if protocol:  # close the span and end the turn (B4): <|/bg|> then <|im_end|>
                parts += [self.protocol.embedding(closing)[None].float(),
                          embed(ex.get('tail', self.tail).to(self.device)).float()]
            seqs.append(torch.cat(parts))
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
        if protocol:
            self.protocol_losses.append(self._protocol_losses(hidden, examples, feed, starts,
                                                              opening))
        return [hidden[i, starts[i]:starts[i] + 1 + feed[i].shape[0]] for i in range(len(seqs))]

    def _protocol_losses(self, hidden, examples, feed, starts,
                         opening: str = 'bg') -> dict[str, torch.Tensor]:
        """Opening ``<|bg|>`` (or ``opening``) from the last prompt position, the turn
        end after the closing token, and (for prompts that state it) the ratio head."""
        head = self.decoder.base_lm.lm_head
        states, targets, ratio_pred, ratio_true = [], [], [], []
        for i, (ex, fed) in enumerate(zip(examples, feed)):
            tail = ex.get('tail', self.tail)
            close = starts[i] + 1 + fed.shape[0]          # the <|/bg|> position
            states.append(hidden[i, [starts[i] - 1] + list(range(close, close + tail.shape[0]))])
            targets.append(torch.cat([torch.tensor([self.protocol.token(opening)]), tail]))
            if ex.get('ratio_stated'):
                ratio_pred.append(self.protocol.ratio_head(hidden[i, starts[i] - 1].float()))
                ratio_true.append(torch.tensor([math.log2(ex['factor']) / 8], device=self.device))
        logits = head(torch.cat(states)).float()
        targets = torch.cat(targets).to(self.device)
        out = {'open_close': F.cross_entropy(logits, targets),
               'open_close_acc': (logits.argmax(-1) == targets).float().mean().detach()}
        if ratio_pred:
            out['ratio'] = F.mse_loss(torch.cat(ratio_pred), torch.cat(ratio_true))
        return out

    def take_protocol_losses(self, weights: dict) -> tuple[torch.Tensor | None, dict]:
        """Weighted sum of the protocol losses recorded since the last call."""
        records, self.protocol_losses = self.protocol_losses, []
        if not records:
            return None, {}
        total, logged = 0.0, {}
        for key, weight in (('open_close', weights.get('protocol', 1.0)),
                            ('ratio', weights.get('ratio', 0.1))):
            values = [r[key] for r in records if key in r]
            if values:
                value = torch.stack(values).mean()
                total = total + weight * value
                logged[key] = value.item()
        logged['open_close_acc'] = torch.stack([r['open_close_acc'] for r in records]).mean().item()
        return total, logged

    def write_inputs(self, ex: dict) -> torch.Tensor:
        """The writer's prompt and source as input embeddings. The source is text
        (``ids``) or, for a merge, stored spans in the decoder's input space
        (``source_embeds``, records separated by a blank line). A write in context
        brings its whole causal prefix instead: rendered token ids (``prefix_ids``,
        B4c write sites) or embeddings (``inputs``, the B4b answer position)."""
        embed = self.decoder.embed_tokens
        if 'inputs' in ex:
            return ex['inputs'].to(self.device).float()
        if 'prefix_ids' in ex:
            return embed(ex['prefix_ids'].to(self.device)).float()
        pre, post = self.prompts[ex['prompt']]
        if 'source_embeds' not in ex:
            return embed(torch.cat([pre, ex['ids'], post]).to(self.device)).float()
        return torch.cat([embed(pre.to(self.device)).float(),
                          ex['source_embeds'].to(self.device).float(),
                          embed(post.to(self.device)).float()])

    def free_run(self, examples, lengths: list[int], heads=None):
        """Self-fed generation of ``lengths[i]`` reps; also the first predicted stop."""
        heads = self.writer if heads is None else heads
        reps = [torch.zeros(0, self.decoder.embed_tokens.weight.shape[1], device=self.device)
                for _ in examples]
        stops = [None] * len(examples)
        prefix = self.prefix(examples)
        for step in range(max(lengths)):
            hidden = self.write(examples, reps, prefix, heads)
            for i, h in enumerate(hidden):
                last = h[-1:]
                if stops[i] is None and heads.stop(last.float()).argmax(-1).item() == 1:
                    stops[i] = step
                if step < lengths[i]:
                    reps[i] = torch.cat([reps[i], heads.rep(last)])
        for i, h in enumerate(self.write(examples, reps, prefix, heads)):
            if stops[i] is None and heads.stop(h[-1:].float()).argmax(-1).item() == 1:
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
        if self.protocol is not None and dec is self.decoder:
            reps = self._delimit(examples, reps, wrap=not full)
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

    def _delimit(self, examples, reps, wrap: bool):
        """B4 read layout: memory spans between ``<|mem|>`` and ``<|/mem|>``, then an
        optional soft input port span (``ex['port']``) between ``<|port|>`` and
        ``<|/port|>``, all in the context slot."""
        p, width = self.protocol, self.writer.marker.shape[0]
        blocks = []
        for i, ex in enumerate(examples):
            parts = []
            if reps is not None:
                rep = reps[i].to(self.device).float()
                parts += ([p.embedding('mem')[None], rep, p.embedding('mem_end')[None]]
                          if wrap else [rep])
            if 'port' in ex:
                parts += [p.embedding('port')[None], ex['port'].to(self.device).float(),
                          p.embedding('port_end')[None]]
            blocks.append(torch.cat(parts) if parts else torch.zeros(0, width, device=self.device))
        return blocks if reps is not None or any('port' in ex for ex in examples) else None

    def answer_inputs(self, view: dict, span: torch.Tensor | None) -> torch.Tensor:
        """Input embeddings of ``read``'s layout up to the answer (template prefix, the
        delimited memory span and input port, the instruction ending in the assistant
        turn): where the B4b soft output port opens instead of the text answer."""
        if self.protocol is None:
            raise ValueError('the answer layout with delimiters needs the span protocol')
        embed = self.decoder.embed_tokens
        block = self._delimit([view], None if span is None else [span], wrap=True)
        parts = [embed(self.tpl.prefix.to(self.device)).float()]
        if block is not None:
            parts.append(block[0])
        instr = view['instr'] if 'instr' in view else self.instr[view['task']]
        parts.append(embed(instr.to(self.device)).float())
        return torch.cat(parts)

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



class Neighbours:
    """The exact-cosine neighbour table of ``scripts/bgkit_neighbors.py`` over the
    teacher cache's records: ``of(item, k)`` gives the k nearest other records."""

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

def frozen_reader(checkpoint, experiment: str, reader_state=None,
                  cuda_fraction: float = 0.2) -> Model:
    """The frozen decoder (and writer) of the stack stages: S2, or the merged decoder
    of a writer-stage state (``reader_state``), with no trainable parameters."""
    import argparse
    model = Model(argparse.Namespace(cuda_fraction=cuda_fraction, experiment=experiment,
                                     checkpoint=checkpoint, adapter_rank=16 if reader_state else 0,
                                     gate_open_start=-1, merge_at=-1, merge_checkpoint=False))
    if reader_state:
        model.load_trained(torch.load(reader_state, map_location=model.device))
    for param in list(model.decoder.parameters()) + list(model.writer.parameters()):
        param.requires_grad_(False)
    if model.protocol is not None:
        for param in model.protocol.parameters():
            param.requires_grad_(False)
    return model

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
