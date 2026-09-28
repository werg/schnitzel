"""L1: live KB items trained in place, end to end (docs/knowledge-base-stack.md, 5.1 step 6).

Two commands:

``build`` (offline bank creation, invariant 1). For every dataset KB named by the
memory transcripts (``scripts/prepare_memory_transcripts.py``; slot ``kb``), every
record a slot names becomes one item per space A-D: the frozen writer encodes the
record text into a span (``--span-source writer``: B3 free-running at the densest
length-scaled ratio s0; ``teacher``: the cached S2 teacher span of the B1 cache,
R6 records only), the K1 forward codecs (``--codecs`` stack.pt, or random init) map
the span into the spaces, and the item's key per space is the initial key head
applied to the frozen decoder's query-layer state averaged over the record text.
The initial key heads are seeded and saved beside the banks, so queries and keys
start in one geometry; the trainer starts from them. Item time is the record's
``created_at``; provenance names the record id. Training reads stored items only.

``train``. The frozen decoder (``--reader-state``: the B3 merged decoder, protocol
installed if the state has none; later B4) reads each transcript rendered with the
LFM2 chat template, ``memory_search()`` calls without arguments (a v1 transcript's
query text is dropped), loss on assistant tokens only. At every call the read path
(``schnitz.kb.read``) takes the query-layer state at the call's own closing
parenthesis, retrieves per space from the episode's KB (authorization: the episode's
dataset only), gates, combines (S_s, R) and splices the span between ``<|mem|>`` and
``<|/mem|>`` of the call's tool message. Queries are exact causal prefixes: the
query at call k is computed from a pass over the prefix up to the call with the
spans of all earlier reads spliced in (a pass per site, truncated at the query
layer, recomputed in backward), so gradients reach earlier reads through later
queries too; then one full pass gives the task loss. Jointly trained: item values
and keys in place (``kb_store`` live mode, sparse Adam per item and key), key
heads, S_s and R; the decoder is frozen. An auxiliary retrieval loss (listwise, over
the scored candidates, the targets the search missed and in-batch negatives: the
target items of the batch's other slots in the same KB, never another KB's) supervises
routing; recall@k per space is logged. ``--init-reader`` starts from a K2 run's key
heads and gate offsets (K2 -> L1a; the search keys are refreshed from them).

Phases (``--phase``, or alternating with ``--phase-schedule a:2000,b:500``):

- *L1a, items in place*: item values updated by one sparse Adam step per touched
  (KB, space) after the reader's optimizer step; the writer is detached.
- *L1b, through the sources* (``Producers``): the items a read keeps are recomputed
  from their stored sources (a bank item: the writer's free run of its record under
  the memory prompt, then the codecs; a written item: the writer's free run at its
  write site from the logged prefix and reads), at the stored forward's serialized
  precision, so at the start of L1b the recomputed items equal the stored payloads
  (checked every step: ``l1b_match_*``). The gradients of all reads of a step
  accumulate on the recomputed items, then the producers are recomputed with
  gradients and the task loss reaches the writer's span heads (ratio code, rep head)
  and the codecs (``--l1b-train`` names the sets; default also keys, S_s and R).
  Live item values are not updated in L1b; an item modified in place by L1a is read
  as its producers' recomputation there (L2 reconciles the two).

Writes (``--writes``, v3 transcripts): each ``memory_write()`` site is rendered with an
empty ``<|bg|><|/bg|>`` pair (not targets; the task pass keeps it empty in every arm).
After a step's reads, the frozen writer generates each site's span in place from the
episode's causal prefix with its reads spliced in (``<|bg|>`` marker, B4c's length
schedule), the codecs and item-key heads make one item per space, and the items enter
the episode's own KB (time = the episode's query time, producer ``write``, source the
write site; a revisited site is superseded). So only later steps read them, never the
same step or the writing episode itself; ``written_*`` counts how often reads retrieve
items that earlier episodes wrote.

Evaluation arms on validation transcripts (invariant 9): ``noctx`` (empty memory
spans), ``text`` (the slot records' text as the tool result: the information-
matched text control), ``retrieved`` (the L1 read), ``shuffled`` (another episode's
reads), ``gold`` (the target items, gate 1, no retrieval) and ``gold_shuffled``.
Reported with ``kb_eval.nll_summary`` (captured fractions of the text arm's gain,
content nats over the shuffled controls), recall, effective items per read and the
share of reads that retrieve written items. Without ``--writes``, ``memory_write``
calls and their acknowledgements are left out of the render.

Entry point: ``scripts/train.py l1 build|train ...``. Training-only; the decoder
parts run in ``sdkb-bgkit``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from schnitz.kb.bank import (SpanCache, Transcripts, build_caches, kb_dir, read_sources,
                             record_sources, slots_of)
from schnitz.kb.decoder import LEVELS, length_factors
from schnitz.kb.read import (DEFAULT_CANDIDATES, DEFAULT_KEEP, ItemCache, L1Reader,
                             ReadConfig, producer_index, source_index, splice)
from schnitz.kb.stack import KeyHeads
from schnitz.kb_eval import distribution, effective_count, nll_summary
from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, NewItem, Provenance
from schnitz.memory_transcripts import render_ids, render_text
from schnitz.span_tokens import MEMORY_TOOLS, SPAN_TOKENS

MEM, MEM_END = SPAN_TOKENS['mem'][0], SPAN_TOKENS['mem_end'][0]
MEM_ID = SPAN_TOKENS['mem'][1]
BG_ID = SPAN_TOKENS['bg'][1]
MEMORY_NAMES = {t['name'] for t in MEMORY_TOOLS}
CALL = re.compile(r'memory_search\(\s*\)')


# -- transcripts ---------------------------------------------------------------------
def query_time(row: dict) -> int:
    prov = row.get('provenance') or {}
    return int(prov.get('source_query_time', prov.get('query_time', 2)))


def _is_write(m: dict) -> bool:
    content = m.get('content')
    if isinstance(content, dict) and 'write_result' in content:
        return True
    calls = m.get('tool_calls') or []
    return bool(calls) and all(tc['function']['name'] == 'memory_write' for tc in calls) \
        and not m.get('content')


def chat(row: dict, texts: dict[str, str] | None = None,
         keep_writes: bool = False) -> tuple[list[dict], list[dict]]:
    """Messages and tools for the chat template: memory_search calls without arguments
    (the current schema), slots as empty ``<|mem|><|/mem|>`` pairs, or with ``texts``
    the slot records' text (the oracle text arm); other dict contents as JSON. Write
    calls and their acknowledgements are dropped unless ``keep_writes``."""
    messages = []
    for m in row['messages']:
        if not keep_writes and _is_write(m):
            continue
        m = dict(m)
        content = m.get('content')
        if isinstance(content, dict) and 'slot' in content:
            ids = content['slot']['record_ids']
            m['content'] = MEM + MEM_END if texts is None else '\n\n'.join(texts[r] for r in ids)
        elif isinstance(content, dict):
            m['content'] = json.dumps(content, ensure_ascii=False)
        if m.get('tool_calls'):
            m['tool_calls'] = [
                {**tc, 'function': {**tc['function'], 'arguments': {}}}
                if tc['function']['name'] == 'memory_search' else tc for tc in m['tool_calls']]
        messages.append(m)
    tools = list(MEMORY_TOOLS) + [t for t in row.get('tools') or []
                                  if t['name'] not in MEMORY_NAMES]
    return messages, tools


@dataclasses.dataclass
class Episode:
    episode_id: str
    kb: str
    ids: torch.Tensor           # (T,)
    targets: torch.Tensor       # positions t whose token is a loss target (predicted at t - 1)
    calls: list[int]            # query position of each memory_search call, in order
    mems: list[int]             # position of each slot's <|mem|>, in the same order
    slots: list[dict]
    query_time: int
    row: dict
    writes: list[int] = dataclasses.field(default_factory=list)  # <|bg|> of each write site
    write_sites: list[dict] = dataclasses.field(default_factory=list)

    @property
    def rendered_writes(self) -> bool:
        return bool(self.row.get('_writes'))


def _render(tok, messages, tools, writes: bool = False):
    if writes:     # the v3 render: an empty <|bg|><|/bg|> pair after each write call
        return render_ids(tok, messages, tools)
    out = tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=True,
                                  return_assistant_tokens_mask=True)
    return list(out['input_ids']), list(out['assistant_masks'])


def layout(row: dict, tok, texts: dict[str, str] | None = None,
           keep_writes: bool = False, writes: bool = False) -> Episode:
    """Token layout of a transcript. The query position of a call is the token holding
    its closing parenthesis (so calls in one block have their own positions).

    ``writes`` (v3 transcripts with write sites): write calls and acknowledgements are
    kept and each write call is followed by an empty ``<|bg|><|/bg|>`` pair in the same
    assistant turn (``schnitz.memory_transcripts``); ``Episode.writes`` holds each
    site's ``<|bg|>`` position. The pair's two tokens are not L1 targets (the decoder
    is frozen; B4c trains the writes) and the task pass keeps the pair empty in every
    arm, as B4c's write-site prefixes keep earlier write spans empty: the generated
    span goes to the KB, and arms differ only in their reads."""
    keep_writes = keep_writes or writes
    messages, tools = chat(row, texts, keep_writes)
    ids, mask = _render(tok, messages, tools, writes)
    slots = slots_of(row)
    calls, mems = [], []
    bg = [t for t, x in enumerate(ids) if x == BG_ID] if writes else []
    sites = list(row.get('write_sites') or []) if writes else []
    if len(bg) != len(sites):
        raise ValueError(f'{row["episode_id"]}: {len(bg)} write spans, {len(sites)} write sites')
    if texts is None:
        text = render_text(messages, tools, tok) if writes else \
            tok.apply_chat_template(messages, tools=tools, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc['input_ids']) != ids:
            raise ValueError('tokenizing the rendered text differs from the chat template')
        owner = {}
        for t, (a, b) in enumerate(enc['offset_mapping']):
            for ch in range(a, b):
                owner.setdefault(ch, t)
        for block in re.finditer(r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>', text, re.S):
            for call in CALL.finditer(block.group(1)):
                calls.append(owner[block.start(1) + call.end() - 1])
        mems = [t for t, x in enumerate(ids) if x == MEM_ID]
        if not (len(calls) == len(mems) == len(slots)):
            raise ValueError(f'{row["episode_id"]}: {len(calls)} calls, {len(mems)} memory '
                             f'slots, {len(slots)} slots')
        if any(m <= c for c, m in zip(calls, mems)):
            raise ValueError('a result precedes its call')
    span_tokens = {p for b in bg for p in (b, b + 1)}
    targets = torch.tensor([t for t in range(1, len(ids)) if mask[t] and t not in span_tokens],
                           dtype=torch.long)
    if writes:
        row = {**row, '_writes': True}
    return Episode(row['episode_id'], row['kb'], torch.tensor(ids), targets, calls, mems,
                   slots, query_time(row), row, bg, sites)


# -- frozen decoder ------------------------------------------------------------------
class _Stop(Exception):
    pass


class Frozen:
    """The frozen reader: embeddings (protocol hooks included), the hidden state after
    ``query_layer`` layers (a truncated pass), final hidden states and LM-head logits."""

    def __init__(self, lm, query_layer: int, autocast=None):
        self.lm, self.inner = lm, lm.model
        self.query_layer = query_layer
        self.autocast = autocast or (lambda: torch.autocast('cpu', enabled=False))
        if not 1 <= query_layer <= len(self.inner.layers):
            raise ValueError('query layer out of range')

    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    @torch.no_grad()
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids.to(self.device)).float()

    def mid(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        box = {}

        def hook(module, args, output):
            box['h'] = output[0] if isinstance(output, tuple) else output
            raise _Stop

        handle = self.inner.layers[self.query_layer - 1].register_forward_hook(hook)
        try:
            with self.autocast():
                self.inner(inputs_embeds=x, attention_mask=attention_mask, use_cache=False)
        except _Stop:
            pass
        finally:
            handle.remove()
        return box['h'].float()

    def final(self, x: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            return self.inner(inputs_embeds=x, use_cache=False).last_hidden_state

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            return self.lm.lm_head(hidden).float()


# -- episodes through the read path ----------------------------------------------------
def _by_source(origin: dict[str, tuple[str, tuple[str, ...]]]) -> dict[str, list[str]]:
    """Item ids per source id (``source_index`` from a producer index)."""
    out: dict[str, list[str]] = {}
    for item_id, (_, sources) in origin.items():
        for source in sources:
            out.setdefault(source, []).append(item_id)
    return out


def write_source(episode_id: str, site: int) -> str:
    """The source id of an in-context write: the episode's write site."""
    return f'write:{episode_id}#{site}'


def write_item_id(episode_id: str, site: int, space: str) -> str:
    """The (opaque, deterministic) item id of a write site's item in ``space``: a
    revisited episode's new write supersedes its previous one."""
    return 'w' + hashlib.sha1(f'{episode_id}#{site}#{space}'.encode()).hexdigest()[:31]


class Context:
    """What a pass needs: the frozen decoder, the reader, the KBs and their item index."""

    def __init__(self, frozen: Frozen, reader: L1Reader, kbs: dict[str, KnowledgeBase],
                 autocast=None):
        self.frozen, self.reader, self.kbs = frozen, reader, kbs
        self.autocast = autocast or frozen.autocast
        # item id -> (producer, sources) of the current items, per KB and space
        self.origin = {name: {s: producer_index(kb, s) for s in kb.spaces}
                       for name, kb in kbs.items()}
        self.index = {name: {s: _by_source(origin) for s, origin in spaces.items()}
                      for name, spaces in self.origin.items()}
        self.written = {name: {s: {i for i, (p, _) in origin.items() if p == 'write'}
                               for s, origin in spaces.items()}
                        for name, spaces in self.origin.items()}
        self._rows: dict[tuple[str, str], dict[str, int]] = {}

    def added(self, dataset: str, space: str, ids: list[str], producer: str,
              sources: list[tuple[str, ...]]) -> None:
        """Items committed during training (writes): origin, written set, row cache."""
        for item_id, src in zip(ids, sources):
            self.origin[dataset][space][item_id] = (producer, src)
            if producer == 'write':
                self.written[dataset][space].add(item_id)
        self._rows.pop((dataset, space), None)

    def own_writes(self, ep: Episode) -> dict[str, set[tuple[str, str]]]:
        """An episode's own write items per space: never retrieved by its own reads (a
        revisited episode would otherwise read what it wrote about its own answer)."""
        return {s: {(ep.kb, write_item_id(ep.episode_id, j, s)) for j in range(len(ep.writes))}
                for s in self.kbs[ep.kb].spaces} if ep.writes else {}

    def rows(self, dataset: str, space: str) -> dict[str, int]:
        """Row index of every item id of a space (for usage statistics)."""
        key = (dataset, space)
        if key not in self._rows:
            self._rows[key] = {i: r for r, i in enumerate(self.kbs[dataset]._row_ids[space])}
        return self._rows[key]

    def covered(self, ep: Episode) -> bool:
        index = self.index.get(ep.kb)
        return index is not None and all(
            all(r in index[s] for s in index) for slot in ep.slots for r in slot['record_ids'])

    def targets(self, ep: Episode, j: int) -> dict[str, list[tuple[str, str]]]:
        slot = ep.slots[j]
        if slot.get('kb', ep.kb) != ep.kb:
            raise PermissionError('a slot names another KB than its episode')
        index = self.index[ep.kb]
        return {s: [(ep.kb, i) for r in slot['record_ids'] for i in index[s].get(r, ())]
                for s in index}


def run_episode(ctx: Context, ep: Episode, cache: ItemCache, mode: str = 'retrieve',
                spans: list[torch.Tensor] | None = None, retrieval_only: bool = False,
                negatives: dict | None = None, producer=None):
    """Task NLL (summed over target tokens) of one transcript and its reads.

    ``mode`` 'retrieve' or 'gold' computes each read at its call from the exact causal
    prefix; 'fixed' splices the given ``spans`` (controls). ``retrieval_only`` (K2):
    the reads' spans enter later prefixes detached and no task pass runs (NLL None),
    so only the retrieval loss trains, through the queries and the item keys.
    ``negatives`` (per space, the episode's own KB only): in-batch negatives of the
    retrieval loss. ``producer`` (L1b): read items' values from their sources. Reads
    never retrieve the episode's own write items (``Context.own_writes``)."""
    exclude = ctx.own_writes(ep)
    embeds = ctx.frozen.embed(ep.ids)
    reads = []
    if mode == 'fixed':
        spans = list(spans)
    else:
        spans = []
        before = [sum(m < c for m in ep.mems) for c in ep.calls]
        k = 0
        while k < len(ep.calls):
            group = [k]
            while group[-1] + 1 < len(ep.calls) and before[group[-1] + 1] == before[k]:
                group.append(group[-1] + 1)
            b = before[k]
            end = ep.calls[group[-1]] + 1
            x, index = splice(embeds[:end], ep.mems[:b], spans[:b])
            if torch.is_grad_enabled() and any(s.requires_grad for s in spans[:b]):
                # recomputed in backward; retrieval stays outside the recomputed function
                h = checkpoint(ctx.frozen.mid, x[None], use_reentrant=False)[0]
            else:
                with torch.no_grad():
                    h = ctx.frozen.mid(x[None])[0]
            for j in group:
                with ctx.autocast():
                    read = ctx.reader.read(h[index[ep.calls[j]]], [ctx.kbs[ep.kb]], [ep.kb],
                                           ep.query_time, cache, targets=ctx.targets(ep, j),
                                           gold=mode == 'gold', negatives=negatives,
                                           exclude=exclude,
                                           producer=None if mode == 'gold' else producer)
                reads.append(read)
                spans.append(read.span.float().detach() if retrieval_only
                             else read.span.float())
            k = group[-1] + 1
    if retrieval_only:
        return None, int(ep.targets.numel()), reads, spans
    x, index = splice(embeds, ep.mems, spans)
    hidden = ctx.frozen.final(x[None])[0]
    positions = index[ep.targets] - 1
    logits = ctx.frozen.logits(hidden[positions])
    nll = F.cross_entropy(logits, ep.ids[ep.targets].to(logits.device), reduction='sum')
    return nll, int(ep.targets.numel()), reads, spans


# -- in-context writes and the producers' forward ------------------------------------------
def ste_round(x: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """``x`` at the serialized precision in the forward (what the store or span cache
    holds), the identity in the backward."""
    return x + (x.to(dtype).to(x.dtype) - x).detach()


@dataclasses.dataclass
class WriteRequest:
    """One write site of an episode run in retrieve mode: its causal prefix (tokens
    before the site's ``<|bg|>``), the ``<|mem|>`` positions in it and the spans read
    there (detached), and the span's ratio and length (B4c's schedule: the site's
    teacher text length at ``level``; the text itself is never rendered)."""
    ep: Episode
    site: int
    prefix_ids: torch.Tensor
    mems: list[int]
    reads: list[torch.Tensor]
    factor: float
    count: int

    @property
    def source(self) -> str:
        return write_source(self.ep.episode_id, self.site)


def write_requests(ep: Episode, spans: list[torch.Tensor], text_ids, level: int
                   ) -> list[WriteRequest]:
    """The write sites of an episode after its reads (``spans`` in call order). Only
    reads before a site enter its prefix (invariant 2)."""
    out = []
    for j, (p, site) in enumerate(zip(ep.writes, ep.write_sites)):
        b = sum(m < p for m in ep.mems)
        tokens = int(text_ids(site['teacher_text']).shape[0])
        factor = length_factors(tokens)[level]
        out.append(WriteRequest(ep, j, ep.ids[:p].clone(), list(ep.mems[:b]),
                                [s.detach().float().cpu() for s in spans[:b]], factor,
                                max(1, math.ceil(tokens / factor))))
    return out


def free_run_grad(model, examples, lengths: list[int]) -> list[torch.Tensor]:
    """``Model.free_run`` with gradients (the same operations, no stop decisions): each
    rep is the writer's rep head on the last state of the span so far, fed back."""
    heads = model.writer
    width = model.decoder.embed_tokens.weight.shape[1]
    reps = [torch.zeros(0, width, device=model.device) for _ in examples]
    prefix = model.prefix(examples)

    def last(*fed):
        return tuple(h[-1:] for h in model.write(examples, list(fed), prefix, heads))
    for step in range(max(lengths)):
        if torch.is_grad_enabled():
            # each step recomputed in backward: activations O(span) instead of O(span^2)
            hidden = checkpoint(last, *reps, use_reentrant=False)
        else:
            hidden = last(*reps)
        for i, h in enumerate(hidden):
            if step < lengths[i]:
                reps[i] = torch.cat([reps[i], heads.rep(h)])
    return reps


class Writer:
    """The writer (``Model``: prefix, write, free run, span heads) with the stack's
    codecs and the reader's item-key heads: in-context writes during L1 episodes and
    the producers' forward that L1b replays."""

    def __init__(self, model, stack, reader: L1Reader, frozen: Frozen, autocast=None,
                 batch: int = 8):
        self.model, self.stack, self.reader, self.frozen = model, stack, reader, frozen
        self.autocast = autocast or frozen.autocast
        self.batch = batch

    def inputs(self, prefix_ids: torch.Tensor, mems: list[int], reads: list[torch.Tensor]
               ) -> torch.Tensor:
        """A write's input embeddings: the prefix with its reads spliced in."""
        x, _ = splice(self.frozen.embed(prefix_ids), mems, reads)
        return x

    @torch.no_grad()
    def generate(self, requests: list[WriteRequest]) -> list[torch.Tensor]:
        """Each site's span, free-running from its causal prefix (``<|bg|>`` marker at
        the site's ratio), as the frozen writer generates it in place."""
        out = []
        for start in range(0, len(requests), self.batch):
            chunk = requests[start:start + self.batch]
            examples = [{'inputs': self.inputs(r.prefix_ids, r.mems, r.reads),
                         'factor': r.factor} for r in chunk]
            with self.autocast():
                reps, _ = self.model.free_run(examples, [r.count for r in chunk])
            out += [r.float() for r in reps]
        return out

    def encode(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        """The codecs' items of a span (as the bank build encodes a record's span)."""
        with self.autocast():
            items = self.stack.encode(span)
        return {s: v.float() for s, v in items.items()}


class WriteLog:
    """The sources of in-context writes, for L1b's producer replay: per write its
    prefix token ids, ``<|mem|>`` positions, the read spans spliced there, the ratio
    and the generated span (float32, the codecs' exact input). One safetensors file
    plus a JSON index per training step under ``root`` (``None``: in memory only); the
    newest entry of a source wins. ``truncate(step)`` drops steps after a checkpoint."""

    def __init__(self, root: Path | None):
        self.root, self.index, self.memory, self.pending = root, {}, {}, {}
        self._handles: dict[str, object] = {}
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            for meta in sorted(root.glob('step-*.json')):
                if meta.with_suffix('.safetensors').exists():
                    for source, entry in json.loads(meta.read_text()).items():
                        self.index[source] = {**entry, 'file': meta.stem}

    def __contains__(self, source: str) -> bool:
        return source in self.index or source in self.pending

    def add(self, req: WriteRequest, span: torch.Tensor, step: int,
            group: list[str] | None = None) -> None:
        """``group``: the sources generated in the same writer batch (replay unit)."""
        self.pending[req.source] = {
            'kb': req.ep.kb, 'mems': req.mems, 'factor': req.factor, 'step': step,
            'group': list(group or [req.source]),
            'tensors': {'prefix_ids': req.prefix_ids.to(torch.int32).cpu(),
                        'span': span.detach().float().cpu(),
                        **{f'read{j}': r.float().cpu() for j, r in enumerate(req.reads)}}}

    def flush(self, step: int) -> None:
        """Publish the pending writes as step ``step`` (the count of completed steps)."""
        if not self.pending:
            return
        if self.root is None:
            for source, entry in self.pending.items():
                self.memory[source] = entry
                self.index[source] = {k: v for k, v in entry.items() if k != 'tensors'}
            self.pending = {}
            return
        from safetensors.torch import save_file
        name = f'step-{step:08d}'
        tensors, meta = {}, {}
        for n, (source, entry) in enumerate(self.pending.items()):
            for key, value in entry['tensors'].items():
                tensors[f'{n}.{key}'] = value.contiguous()
            meta[source] = {'n': n, 'reads': sum(k.startswith('read') for k in entry['tensors']),
                            **{k: v for k, v in entry.items() if k != 'tensors'}}
        save_file(tensors, str(self.root / f'{name}.safetensors'))
        (self.root / f'{name}.json').write_text(json.dumps(meta) + '\n')
        for source, entry in meta.items():
            self.index[source] = {**entry, 'file': name}
        self.pending = {}

    def get(self, source: str) -> dict:
        entry = self.index[source]
        if self.root is None:
            tensors = self.memory[source]['tensors']
        else:
            from safetensors import safe_open
            handle = self._handles.get(entry['file'])
            if handle is None:
                handle = safe_open(str(self.root / f'{entry["file"]}.safetensors'),
                                   framework='pt')
                self._handles[entry['file']] = handle
            tensors = {key: handle.get_tensor(f'{entry["n"]}.{key}') for key in
                       ['prefix_ids', 'span'] + [f'read{j}' for j in range(entry['reads'])]}
        return {'prefix_ids': tensors['prefix_ids'].long(), 'span': tensors['span'],
                'mems': entry['mems'], 'factor': entry['factor'],
                'reads': [tensors[f'read{j}'] for j in range(len(entry['mems']))]}

    def truncate(self, step: int) -> None:
        """Forget the writes of steps after ``step`` (their KB commits were discarded)."""
        self.pending = {}
        if self.root is None:
            return
        self._handles = {}
        for meta in sorted(self.root.glob('step-*.json')):
            if int(meta.stem.split('-')[1]) > step:
                meta.with_suffix('.safetensors').unlink(missing_ok=True)
                meta.unlink()
        self.index = {}
        for meta in sorted(self.root.glob('step-*.json')):
            for source, entry in json.loads(meta.read_text()).items():
                self.index[source] = {**entry, 'file': meta.stem}


def commit_writes(ctx: Context, writer: Writer, requests: list[WriteRequest],
                  spans: list[torch.Tensor], step: int, log: WriteLog | None = None) -> int:
    """Items of each written span (codecs per space, keys from the item-key heads) into
    the episode's own KB: time = the episode's query time, provenance ``write`` with
    the write site as source and ``step``; a site written before is superseded (same
    ids). Returns the number of writes."""
    groups = {}
    for start in range(0, len(requests), writer.batch):   # ``Writer.generate``'s batches
        chunk = [r.source for r in requests[start:start + writer.batch]]
        groups.update({source: chunk for source in chunk})
    latest: dict[tuple[str, int], tuple[WriteRequest, torch.Tensor]] = {}
    for req, span in zip(requests, spans):        # a batch may hold an episode twice
        latest[(req.ep.episode_id, req.site)] = (req, span)
    by_kb: dict[str, list] = {}
    with torch.no_grad():
        for req, span in latest.values():
            by_kb.setdefault(req.ep.kb, []).append((req, writer.encode(span)))
            if log is not None:
                log.add(req, span, step, groups[req.source])
        for name, entries in by_kb.items():
            kb = ctx.kbs[name]
            for s in kb.spaces:
                keys = ctx.reader.item_keys(s, [items[s] for _, items in entries])
                fresh, again = [], []
                for (req, items), key in zip(entries, keys):
                    item_id = write_item_id(req.ep.episode_id, req.site, s)
                    item = NewItem(items[s].cpu(), key.float().cpu(),
                                   Provenance((req.source,), 'write', step), 1.0,
                                   req.ep.query_time, id=item_id)
                    (again if item_id in ctx.origin[name][s] else fresh).append(item)
                if fresh:
                    kb.append(s, fresh)
                if again:
                    kb.supersede(s, again)
                done = fresh + again
                ctx.added(name, s, [i.id for i in done], 'write',
                          [i.provenance.sources for i in done])
    return len(latest)


class Producers:
    """L1b (owner: full gradient through the sources): the items a read retrieves are
    recomputed from their stored sources with gradients into the producers.

    - A bank item (producer ``codec``, one source record): the writer's span of the
      record under the memory prompt at the bank's ratio level, then the codecs.
    - A written item (producer ``write``): the writer's span at the write site from
      the logged prefix and read spans (``WriteLog``), then the codecs.

    ``mode`` 'free' replays the free run itself (the producer's actual forward:
    ``free_run_grad``, each rep fed back); 'teacher' feeds the stored span and takes
    the rep head's predictions (one pass, cheaper, but conditioned on the stored
    span's rounded reps rather than the unrounded ones the free run fed back). The
    span enters the codecs at its serialized precision (bf16 in the span cache for
    bank items, float32 for writes) and the item at the store's (bf16), each by
    ``ste_round``, so at the start of L1b the recomputed item equals the stored
    payload (``stats``: ``match_exact``, ``match_rel``). The forward is deterministic
    (no dropout, no sampling); the backward recomputes it and checks the drift.

    Items that L1a modified in place have no source-recompute equivalent: in L1b the
    value read is the producers' recomputation, not the live value (L2 reconciles the
    two). Items without a recomputable source (another producer, a record without
    text or cached span, a write not in the log) keep their live value.

    Per step: ``begin``; reads call ``values`` (forward without gradient, once per
    source, item values as leaves so the gradients of all reads of the step
    accumulate); ``backward`` recomputes the sources that received gradients with
    gradients and backpropagates the accumulated item gradients into the writer's span
    heads and the codecs (before the optimizer step: invariant 3)."""

    def __init__(self, writer: Writer, ctx: Context, texts: dict[str, str],
                 caches: dict[str, SpanCache], level: int, log: WriteLog | None,
                 text_ids, mode: str = 'free', batch: int = 1):
        if mode not in ('free', 'teacher'):
            raise ValueError('mode is free or teacher')
        self.writer, self.ctx, self.texts, self.caches = writer, ctx, texts, caches
        self.level, self.log, self.text_ids, self.mode, self.batch = level, log, text_ids, mode, batch
        self.leaves: dict[tuple, dict[str, torch.Tensor]] = {}
        self.items: dict[tuple, dict[str, str]] = {}
        self.stats: dict[str, list] = {}
        self.unit_of: dict[tuple, tuple] = {}

    def begin(self) -> None:
        self.leaves, self.items, self.stats, self.unit_of = {}, {}, {}, {}

    def source(self, dataset: str, space: str, item_id: str) -> tuple | None:
        producer, sources = self.ctx.origin[dataset][space].get(item_id, (None, ()))
        if len(sources) != 1:
            return None
        if producer == 'codec' and sources[0] in self.texts and dataset in self.caches \
                and sources[0] in self.caches[dataset]:
            return ('codec', dataset, sources[0])
        if producer == 'write' and self.log is not None and sources[0] in self.log.index:
            return ('write', dataset, sources[0])
        return None

    def _group(self, key: tuple) -> tuple:
        """The replay unit of a written item: the sources generated together with it
        (the writer's batch at the site, as logged), so the replay has the
        generation's batch composition; the item alone when that is unknown."""
        group = self.log.index[key[2]].get('group') or [key[2]]
        unit = tuple(('write', self.log.index[g]['kb'], g) for g in group if g in self.log.index)
        return unit if key in unit else (key,)

    def values(self, space: str, refs) -> list[torch.Tensor | None]:
        keys = [self.source(d, space, i) for d, i in refs]
        todo = [k for k in dict.fromkeys(keys) if k is not None and k not in self.leaves]
        codec = [k for k in todo if k[0] == 'codec']
        units = [tuple(codec[i:i + self.batch]) for i in range(0, len(codec), self.batch)]
        units += list(dict.fromkeys(self._group(k) for k in todo if k[0] == 'write'))
        for unit in units:
            if all(k in self.leaves for k in unit):
                continue
            with torch.no_grad():
                outs = self.forward(list(unit))
            for key, out in zip(unit, outs):
                if key not in self.leaves:
                    self.leaves[key] = {s: v.detach().requires_grad_() for s, v in out.items()}
                    self.unit_of[key] = unit
        for key, (_, item_id) in zip(keys, refs):
            if key is not None and space not in self.items.setdefault(key, {}):
                self.items[key][space] = item_id
                self._verify(key, space, item_id)
        return [None if k is None else self.leaves[k][space] for k in keys]

    def _example(self, key: tuple) -> tuple[dict, torch.Tensor, bool]:
        kind, dataset, source = key
        device = self.writer.model.device
        if kind == 'codec':
            ids = self.text_ids(self.texts[source])
            factor = length_factors(int(ids.shape[0]))[self.level]
            feed = self.caches[dataset].get(source).to(device).float()
            return {'ids': ids, 'prompt': 'memory', 'factor': factor}, feed, True
        entry = self.log.get(source)
        x = self.writer.inputs(entry['prefix_ids'], entry['mems'], entry['reads'])
        return {'inputs': x, 'factor': entry['factor']}, entry['span'].to(device), False

    def forward(self, keys: list[tuple]) -> list[dict[str, torch.Tensor]]:
        """The producers' items of each source (gradients when enabled)."""
        model = self.writer.model
        parts = [self._example(k) for k in keys]
        examples = [e for e, _, _ in parts]
        with self.writer.autocast():
            if self.mode == 'free':
                reps = free_run_grad(model, examples, [f.shape[0] for _, f, _ in parts])
            else:
                hidden = model.write(examples, [f for _, f, _ in parts], model.prefix(examples))
                reps = [model.writer.rep(h[:-1]) for h in hidden]
        out = []
        for rep, (_, _, cached) in zip(reps, parts):
            span = ste_round(rep.float()) if cached else rep.float()
            out.append({s: ste_round(v) for s, v in self.writer.encode(span).items()})
        return out

    @torch.no_grad()
    def _verify(self, key: tuple, space: str, item_id: str) -> None:
        """A recomputed item against the stored payload (bf16) of the same item."""
        got = self.leaves[key][space]
        stored = self.ctx.kbs[key[1]].read(space, [item_id])[0].values.float().to(got.device)
        if got.shape != stored.shape:
            self.stats.setdefault('match_rel', []).append(float('inf'))
            return
        self.stats.setdefault('match_exact', []).append(float((got == stored).float().mean()))
        self.stats.setdefault('match_rel', []).append(
            float((got - stored).norm() / stored.norm().clamp_min(1e-12)))

    def backward(self) -> dict:
        """Backpropagate the step's accumulated item gradients through the producers:
        each replay unit holding an item with a gradient is recomputed with gradients
        in the composition of its forward (``drift``: the largest difference to it)."""
        todo = [k for k, leaves in self.leaves.items()
                if any(v.grad is not None for v in leaves.values())]
        drift = 0.0
        for unit in dict.fromkeys(self.unit_of[k] for k in todo):
            outs = self.forward(list(unit))
            tensors, grads = [], []
            for key, out in zip(unit, outs):
                if self.unit_of.get(key) != unit:
                    continue
                for s, value in out.items():
                    leaf = self.leaves[key][s]
                    drift = max(drift, float((value.detach() - leaf).abs().max()))
                    if leaf.grad is not None:
                        tensors.append(value)
                        grads.append(leaf.grad)
            if tensors:
                torch.autograd.backward(tensors, grads)
        out = {'sources': len(self.leaves), 'backward_sources': len(todo), 'drift': drift}
        for name, values in self.stats.items():
            out[name] = round(sum(values) / len(values), 6) if values else None
            if name == 'match_rel' and values:
                out['match_rel_max'] = max(values)
        return out


# -- model loading (GPU container) ------------------------------------------------------
def load_model(args):
    """The frozen decoder (and writer): ``schnitz.kb.decoder.frozen_reader`` with the
    span protocol installed when the reader state has none (B3); nothing trains."""
    from schnitz.kb.decoder import frozen_reader
    from schnitz.span_tokens import check_tokenizer
    model = frozen_reader(args.checkpoint, args.experiment, args.reader_state,
                          args.cuda_fraction)
    if model.protocol is None:
        if not model.merged:
            raise ValueError('L1 reads through a merged decoder (B3 state or later)')
        model.install_protocol()
        for param in model.protocol.parameters():
            param.requires_grad_(False)
    check_tokenizer(model.tok)
    return model


STACK_DIMS = {'state': 512, 'hidden': 256, 'layers': 3}


def load_stack(path: Path | None, target_norm: float, device, seed: int):
    """A ``schnitz.kb.stack.Stack``: a K1 ``stack.pt`` (dims from its config.json), a
    banks ``stack.pt`` (dims stored with it), or random init (``path`` None)."""
    from schnitz.kb.stack import Stack
    dims, state = dict(STACK_DIMS), None
    if path is not None:
        state = torch.load(path, map_location='cpu')
        if 'dims' in state:
            dims = state['dims']
        elif (path.parent / 'config.json').exists():
            config = json.loads((path.parent / 'config.json').read_text())
            dims = {k: config.get(k, v) for k, v in dims.items()}
    torch.manual_seed(seed)
    stack = Stack(target_norm, dims['state'], dims['hidden'], dims['layers'], False)
    step = 0
    if state is not None:
        stack.load_state_dict(state['stack'])
        step = int(state.get('step', 0))
    return stack.to(device).eval(), step, dims


# -- build --------------------------------------------------------------------------
def initial_heads(path: Path, hidden: int, key_hidden: int, seed: int) -> KeyHeads:
    """The initial query and item-key heads (seeded), shared by every KB of a build
    and the trainer's starting point."""
    torch.manual_seed(seed)
    heads = KeyHeads(hidden, key_hidden)
    if path.exists():
        heads.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        torch.save(heads.state_dict(), path)
    return heads


def _writer_meta(reader_state: Path) -> dict:
    """A span cache's writer identity, as ``train.py bank`` records it (so caches are
    shared between the bank stage and the L1 build)."""
    state = torch.load(reader_state, map_location='cpu', mmap=True)
    return {'writer_state': str(reader_state), 'step': int(state.get('step', -1))}


@torch.no_grad()
def build(args) -> None:
    """One KB per dataset from the records the transcripts' slots name (plus
    ``--distractors``): the writer's span of each record (``schnitz.kb.bank``; cached in
    ``<output>/spans`` or ``--span-cache``), the codecs' items per space and their keys
    from the initial item-key heads. Resumable: records already in a KB are skipped."""
    records = record_sources(args.transcripts, {'train': args.limit,
                                                'validation': args.eval_limit},
                             args.distractors)
    model = load_model(args)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher, caches = None, {}
    if args.span_source == 'teacher':
        from schnitz.kb.decoder import TeacherCache
        teacher = TeacherCache(args.cache, None, texts=[])
        teacher.by_id = {item[2]: item for item in teacher.items}
    else:
        span_root = args.span_cache or args.output / 'spans'   # one SpanCache per KB
        built = build_caches(span_root, model, records, args.level,
                             args.span_batch_size or args.batch_size,
                             meta=_writer_meta(args.reader_state))
        caches = {kb: SpanCache(span_root / kb_dir(kb)) for kb in built}

    def span_of(record_id: str) -> torch.Tensor:
        if teacher is None:
            return caches[records[record_id]['kb']].get(record_id).to(model.device).float()
        if record_id not in teacher.by_id:
            raise ValueError(f'record {record_id} has no cached teacher span')
        shard, row = teacher.by_id[record_id][:2]
        return teacher.reps(shard, row, args.level).to(model.device).float()

    stack_path = args.output / 'stack.pt'
    resumed = stack_path.exists()
    stack, codec_step, dims = load_stack(stack_path if resumed else args.codecs,
                                         model.target_norm, model.device, args.seed)
    if not resumed:
        if args.codecs is None:   # a random-init stack takes its span statistics here
            sample = sorted(records)[:1024]
            stack.set_statistics([span_of(r) for r in sample])
        torch.save({'stack': stack.state_dict(), 'step': codec_step, 'dims': dims,
                    'source': str(args.codecs)}, stack_path)
    hidden = model.decoder.base_lm.config.hidden_size
    heads = initial_heads(args.output / 'key_heads_init.pt', hidden, args.key_hidden,
                          args.seed).to(model.device)
    by_kb: dict[str, list[str]] = {}
    for r, rec in records.items():
        by_kb.setdefault(rec['kb'], []).append(r)
    report = {}
    started = time.time()
    for kb_name, recs in sorted(by_kb.items()):
        root = args.output / kb_dir(kb_name)
        kb = KnowledgeBase(root, writable=True) if (root / 'manifest.json').exists() else \
            KnowledgeBase.create(root, name=kb_dir(kb_name), dataset=kb_name,
                                 origin={'command': 'train.py l1 build',
                                         'span_source': args.span_source, 'level': args.level,
                                         'codecs': str(args.codecs), 'codec_step': codec_step,
                                         'reader_state': str(args.reader_state)})
        done = set(source_index(kb, 'D'))
        todo = sorted(r for r in recs if r not in done)
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            with model.core.autocast():
                encoded = [stack.encode(span_of(r)) for r in batch]
            per_space = {s: [] for s in DEFAULT_SPACES}
            for r, items in zip(batch, encoded):
                for s in DEFAULT_SPACES:
                    values = items[s].float()
                    per_space[s].append(NewItem(
                        values.cpu(), heads.item_key(s, values).float().cpu(),
                        Provenance((r,), 'codec', codec_step), 1.0,
                        records[r]['created_at']))
            for s, items in per_space.items():
                kb.append(s, items)
        print(json.dumps({'kb': kb_name, 'items_per_space': len(recs),
                          'elapsed_s': round(time.time() - started)}), flush=True)
        report[kb_name] = {'dir': root.name, 'records': len(recs), 'stats': kb.stats()}
        kb.close()
    manifest = {'command': 'build', 'transcripts': [str(d) for d in args.transcripts],
                'limit': args.limit, 'eval_limit': args.eval_limit,
                'span_source': args.span_source, 'level': args.level,
                'span_cache': str(args.span_cache or args.output / 'spans'),
                'codecs': str(args.codecs), 'distractors': args.distractors,
                'span_batch_size': args.span_batch_size or args.batch_size,
                'codec_step': codec_step, 'reader_state': str(args.reader_state),
                'seed': args.seed, 'kbs': report}
    (args.output / 'banks.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'built': {k: v['records'] for k, v in report.items()}}), flush=True)


# -- train --------------------------------------------------------------------------
def open_live(banks: Path, output: Path, device=None,
              sync_every: int = 0) -> dict[str, KnowledgeBase]:
    """The live copies of the banks under ``output/kbs`` (copied on first start), with
    the live state resident on ``device`` (``KnowledgeBase.load_live``)."""
    live_root = output / 'kbs'
    manifest = json.loads((banks / 'banks.json').read_text())
    if not live_root.exists():
        pending = output / 'kbs.pending'
        shutil.rmtree(pending, ignore_errors=True)
        pending.mkdir(parents=True)
        for info in manifest['kbs'].values():
            shutil.copytree(banks / info['dir'], pending / info['dir'],
                            ignore=shutil.ignore_patterns('writer.lock'))
        pending.rename(live_root)
    kbs = {}
    for name, info in manifest['kbs'].items():
        kb = KnowledgeBase(live_root / info['dir'], writable=True)
        for s in kb.spaces:
            if not kb.is_live(s):
                kb.enable_live(s)
        kb.load_live(device=device, sync_every=sync_every)
        kbs[name] = kb
    return kbs


def _read_stats(reads, sink: dict, written: dict | None = None) -> None:
    """Per-read statistics; with ``written`` (dataset -> space -> ids of written items)
    also how often a read retrieves items written by earlier episodes (``written_s``:
    share of reads with a written item among those read; ``written_mass_s``: share of
    the read's gate mass on written items). Own writes are excluded from reads."""
    for read in reads:
        sink.setdefault('n', []).append(read.n)
        for key, value in read.recall_at.items():
            sink.setdefault(key, []).append(value)
        for s, info in read.spaces.items():
            if info.recall is not None:
                sink.setdefault(f'recall_{s}', []).append(info.recall)
                sink.setdefault(f'recall_read_{s}', []).append(info.recall_read)
            if len(info.gates):
                sink.setdefault(f'eff_{s}', []).append(effective_count(info.gates.tolist())['entropy'])
                sink.setdefault(f'mass_{s}', []).append(info.mass)
                if info.recomputed:
                    sink.setdefault(f'recomputed_{s}', []).append(info.recomputed / len(info.refs))
            if written is not None and info.refs:
                hit = [i in written.get(d, {}).get(s, ()) for d, i in info.refs]
                gates = info.gates.float()
                sink.setdefault(f'written_{s}', []).append(float(any(hit)))
                sink.setdefault(f'written_mass_{s}', []).append(
                    float(gates[torch.tensor(hit)].sum() / gates.sum().clamp_min(1e-12)))


def _mean(sink: dict) -> dict:
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(sink.items()) if v}


def retrieval_weight(args, step: int) -> float:
    """The retrieval loss weight, annealed linearly to its floor as the task loss takes
    over (stack doc 5.2); constant in K2 (``--retrieval-only``)."""
    if args.retrieval_only or not args.retrieval_anneal:
        return args.retrieval_weight
    done = min(step / args.retrieval_anneal, 1.0)
    return args.retrieval_weight + done * (args.retrieval_floor - args.retrieval_weight)


def _balance(ctx: Context, reads, usage: dict) -> torch.Tensor | None:
    """Spread-out use (stack doc 5.2): ``balance_loss`` per (KB, space) over every
    scored (read, candidate) pair of the episode's reads, averaged; updates the
    usage averages with the read mass. Indexed by store row (superseded rows keep
    their slot)."""
    from schnitz.kb.losses import UsageEMA, balance_loss
    pairs: dict[tuple[str, str], tuple[list[int], list[torch.Tensor]]] = {}
    for read in reads:
        for s, info in read.spaces.items():
            if info.scored_gates is None:
                continue
            for (dataset, item_id), gate in zip(info.scored, info.scored_gates):
                rows = ctx.rows(dataset, s)
                entry = pairs.setdefault((dataset, s), ([], []))
                entry[0].append(rows[item_id])
                entry[1].append(gate)
    losses = []
    for (dataset, s), (ids, gates) in pairs.items():
        key = f'{dataset}/{s}'
        size = len(ctx.kbs[dataset]._row_ids[s])
        if key not in usage:
            usage[key] = UsageEMA(size)
        usage[key].grow(size)
        ids_t, gates_t = torch.tensor(ids), torch.stack(gates)
        losses.append(balance_loss(ids_t, gates_t, usage[key]))
        usage[key].update(ids_t, gates_t.detach().cpu())
    return torch.stack(losses).mean() if losses else None


def batch_negatives(ctx: Context, episodes: list[Episode], limit: int,
                    rng: random.Random) -> dict[str, dict[str, list]]:
    """In-batch negatives per KB and space: the target items of every slot of the
    batch's episodes, grouped by the episode's own KB (never across KBs: a KB is an
    authorization domain), at most ``limit`` per (KB, space), sampled."""
    pool: dict[str, dict[str, list]] = {}
    for ep in episodes:
        for j in range(len(ep.slots)):
            for s, refs in ctx.targets(ep, j).items():
                pool.setdefault(ep.kb, {}).setdefault(s, []).extend(refs)
    out = {}
    for kb, spaces in pool.items():
        out[kb] = {}
        for s, refs in spaces.items():
            refs = list(dict.fromkeys(refs))
            out[kb][s] = rng.sample(refs, limit) if len(refs) > limit else refs
    return out


# which parameter sets train in each phase (``--l1b-train`` chooses L1b's)
PHASE_SETS = ('keys', 'operators', 'recombiner', 'codecs', 'writer')
L1A_SET = ('keys', 'operators', 'recombiner')


def parameter_sets(reader: L1Reader, writer_model=None) -> dict[str, list]:
    """The trainable parameter sets: key heads and gate offsets, S_s, R, the codecs
    and the writer's span heads (marker, ratio code, rep head)."""
    sets = {'keys': list(reader.keys.parameters()) + list(reader.gate_offset.parameters()),
            'operators': list(reader.operators.parameters()),
            'recombiner': list(reader.stack.recombiner.parameters()),
            'codecs': list(reader.stack.codecs.parameters()),
            'writer': [] if writer_model is None else list(writer_model.writer.parameters())}
    return sets


def set_phase(sets: dict[str, list], names) -> list:
    """Enable gradients for the named sets only; returns the enabled parameters."""
    on = []
    for name, params in sets.items():
        for p in params:
            p.requires_grad_(name in names)
        if name in names:
            on += params
    return on


def parse_schedule(text: str | None, phase: str) -> list[tuple[str, int]]:
    """``a:2000,b:500`` -> [('l1a', 2000), ('l1b', 500)] (cycled); without a schedule
    the single ``phase``."""
    if not text:
        return [(phase, 1)]
    out = []
    for part in text.split(','):
        name, _, count = part.partition(':')
        name = {'a': 'l1a', 'b': 'l1b'}.get(name.strip(), name.strip())
        if name not in ('l1a', 'l1b') or not count.strip().isdigit() or int(count) < 1:
            raise ValueError(f'bad phase schedule entry {part!r} (e.g. a:2000,b:500)')
        out.append((name, int(count)))
    return out


def phase_at(schedule: list[tuple[str, int]], step: int) -> str:
    """The phase of 0-based ``step`` under a cycled schedule."""
    at = step % sum(n for _, n in schedule)
    for name, n in schedule:
        if at < n:
            return name
        at -= n
    raise AssertionError


def train_step(ctx: Context, episodes: list[Episode], optimizer, args, step: int = 0,
               usage: dict | None = None, *, phase: str | None = None,
               trainable: list | None = None, writer: Writer | None = None,
               producers: Producers | None = None, log: WriteLog | None = None,
               rng: random.Random | None = None, text_ids=None) -> dict:
    """One step: gradients of all episodes accumulate, then one optimizer step and (L1a)
    one sparse live update per touched (KB, space) for the item values.

    L1a: items in place, writer detached. L1b: read items recomputed from their
    sources (``producers``), their accumulated gradients backpropagated into the
    producers before the optimizer step; live item values are not updated. K2 is
    L1a with ``--retrieval-only``. With a ``writer`` and episodes with write sites,
    the sites' spans are generated at the end of the step (after every read of the
    step: the batch's episodes never see each other's writes) and committed to the
    episodes' KBs, so reads of later steps can retrieve them."""
    phase = phase or args.phase
    if phase == 'l1b' and producers is None:
        raise ValueError('L1b needs the producers (gradients through the sources)')
    usage = {} if usage is None else usage
    started = time.time()
    cache = ItemCache(ctx.frozen.device, train=True)
    optimizer.zero_grad(set_to_none=True)
    tokens = sum(int(ep.targets.numel()) for ep in episodes)
    weight = retrieval_weight(args, step)
    stats: dict[str, list] = {}
    nll_total, aux_total, balance_total = 0.0, 0.0, 0.0
    negatives = batch_negatives(ctx, episodes, args.inbatch_negatives,
                                rng or random.Random(step)) if args.inbatch_negatives else {}
    producer = producers if phase == 'l1b' else None
    if producer is not None:
        producer.begin()
    requests: list[WriteRequest] = []
    for ep in episodes:
        nll, _, reads, spans = run_episode(ctx, ep, cache, 'retrieve',
                                           retrieval_only=args.retrieval_only,
                                           negatives=negatives.get(ep.kb), producer=producer)
        terms = [] if nll is None else [nll / tokens]
        aux = [r.aux for r in reads if r.aux is not None]
        if aux and weight:
            aux_mean = torch.stack(aux).mean()
            terms.append(weight * aux_mean / len(episodes))
            aux_total += aux_mean.item() / len(episodes)
        if args.balance_weight and not args.retrieval_only:
            balance = _balance(ctx, reads, usage)
            if balance is not None:
                terms.append(args.balance_weight * balance / len(episodes))
                balance_total += balance.item() / len(episodes)
        loss = sum(terms) if terms else None
        if loss is not None and loss.requires_grad:
            loss.backward()
        if nll is not None:
            nll_total += nll.item()
        _read_stats(reads, stats, ctx.written)
        if writer is not None and ep.writes:
            requests += write_requests(ep, spans, text_ids, args.write_level_index)
    out: dict = {}
    if producer is not None:
        t0 = time.time()
        out['l1b'] = producer.backward()
        out['l1b_backward_s'] = round(time.time() - t0, 3)
    params = trainable if trainable is not None else ctx.reader.trainable()
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    optimizer.step()
    item_lr = 0.0 if args.retrieval_only or phase == 'l1b' else args.item_lr
    counts = cache.apply(item_lr)
    if requests:
        t0 = time.time()
        spans = writer.generate(requests)
        out['writes'] = commit_writes(ctx, writer, requests, spans, step + 1, log)
        if log is not None:
            log.flush(step + 1)
        out['write_s'] = round(time.time() - t0, 3)
    out.update({'phase': phase, 'aux': aux_total, 'retrieval_weight': weight, 'tokens': tokens,
                **counts, **_mean(stats), 'step_s': round(time.time() - started, 3)})
    if args.balance_weight and not args.retrieval_only:
        out['balance'] = balance_total
    if not args.retrieval_only:
        out['nll'] = nll_total / tokens
    return out


@torch.no_grad()
def evaluate(ctx: Context, episodes: list[Episode], texts: dict[str, str], tok) -> dict:
    cache = ItemCache(ctx.frozen.device, train=False)
    sums = {a: 0.0 for a in ('noctx', 'full', 'retrieved', 'shuffled', 'gold', 'gold_shuffled')}
    tokens = 0
    stats: dict[str, list] = {}
    gold_stats: dict[str, list] = {}
    got = {'retrieved': [], 'gold': []}
    for ep in episodes:
        for mode, name in (('retrieve', 'retrieved'), ('gold', 'gold')):
            nll, n, reads, spans = run_episode(ctx, ep, cache, mode)
            sums[name] += nll.item()
            got[name].append(spans)
            _read_stats(reads, stats if name == 'retrieved' else gold_stats,
                        ctx.written if name == 'retrieved' else None)
        tokens += n
        empty = [torch.zeros(0, ctx.reader.config.span_width) for _ in ep.mems]
        sums['noctx'] += run_episode(ctx, ep, cache, 'fixed', empty)[0].item()
        text_ep = layout(ep.row, tok, texts, writes=ep.rendered_writes)
        if int(text_ep.targets.numel()) != n:
            raise ValueError('the text arm changes the target tokens')
        sums['full'] += run_episode(ctx, text_ep, cache, 'fixed', [])[0].item()
    for i, ep in enumerate(episodes):   # another episode's reads, cyclically per site
        for name, control in (('retrieved', 'shuffled'), ('gold', 'gold_shuffled')):
            other = got[name][(i + 1) % len(episodes)]
            spans = [other[j % len(other)] if other else
                     torch.zeros(0, ctx.reader.config.span_width) for j in range(len(ep.mems))]
            sums[control] += run_episode(ctx, ep, cache, 'fixed', spans)[0].item()
    report = nll_summary(sums, tokens, ('retrieved', 'shuffled', 'gold', 'gold_shuffled'),
                         {'retrieved': 'shuffled', 'gold': 'gold_shuffled'})
    report['episodes'], report['tokens'] = len(episodes), tokens
    report['reads'] = _mean(stats)
    report['gold_reads'] = _mean(gold_stats)
    report['span_reps'] = distribution(stats.get('n', []))
    report['written_items'] = {name: len(spaces.get('D', ())) for name, spaces in ctx.written.items()
                               if spaces.get('D')}
    return report


def _pairs(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split('=') for p in text.split(',') if p)}


def load_init_reader(reader: L1Reader, path: Path) -> list[str]:
    """K2 -> L1a: the key heads (query and item heads, scales) and gate offsets of a
    K2 run's ``reader.pt`` (or a bare state dict). Returns the loaded names."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('reader', state)
    wanted = {k: v for k, v in state.items() if k.startswith(('keys.', 'gate_offset.'))}
    own = reader.state_dict()
    missing = [k for k in own if k.startswith('keys.') and k not in wanted]
    if missing:
        raise ValueError(f'{path} has no key heads ({len(missing)} missing, e.g. {missing[0]})')
    for k, v in wanted.items():
        if k not in own or own[k].shape != v.shape:
            raise ValueError(f'{path}: {k} does not match this reader')
    reader.load_state_dict({**own, **{k: v.to(own[k].device) for k, v in wanted.items()}})
    return sorted(wanted)


def load_producers(args, ctx: Context, writer: Writer, log: WriteLog, model) -> Producers:
    """L1b's producers: the bank records' text and span caches (the bank build's), the
    write log. The writer must be the one that built the bank's spans."""
    manifest = json.loads((args.banks / 'banks.json').read_text())
    if manifest.get('span_source', 'writer') != 'writer':
        raise ValueError('L1b needs a bank built from writer spans (--span-source writer)')
    if str(manifest.get('reader_state')) != str(args.reader_state):
        print(json.dumps({'warning': 'bank writer state differs from --reader-state',
                          'bank': manifest.get('reader_state'),
                          'reader_state': str(args.reader_state)}), flush=True)
    root = Path(manifest['span_cache'])
    caches = {}
    for name in ctx.kbs:
        path = root / kb_dir(name)
        if (path / 'manifest.json').exists():
            caches[name] = SpanCache(path)
    wanted = {src[0] for spaces in ctx.origin.values() for origin in spaces.values()
              for producer, src in origin.values() if producer == 'codec' and len(src) == 1}
    dirs = {str(d) for d in manifest['transcripts']}
    texts = {r: v['text'] for r, v in read_sources(dirs, wanted).items()}
    return Producers(writer, ctx, texts, caches, LEVELS.index(manifest['level']), log,
                     model.text_ids, args.l1b_replay, args.l1b_batch)


def train(args) -> None:
    model = load_model(args)
    lm = model.decoder.base_lm
    if args.decoder_checkpoint:
        from schnitz.bgkit_span import checkpoint_layers
        checkpoint_layers(lm.model.layers)
    frozen = Frozen(lm, args.query_layer, model.core.autocast)
    candidates = dict(DEFAULT_CANDIDATES, **_pairs(args.candidates))
    keep = dict(DEFAULT_KEEP, **_pairs(args.keep))
    schedule = parse_schedule(args.phase_schedule, args.phase)
    phases = {name for name, _ in schedule}
    l1b_set = tuple(x for x in args.l1b_train.split(',') if x)
    unknown = set(l1b_set) - set(PHASE_SETS)
    if unknown:
        raise ValueError(f'unknown L1b parameter sets {sorted(unknown)}; choose from {PHASE_SETS}')
    # R (and the span statistics) from the banks' stack, so reads decode what the
    # bank's codecs encoded; the operators take the stack's dimensions
    stack, _, dims = load_stack(args.banks / 'stack.pt', model.target_norm, 'cpu', args.seed)
    stack.recombiner.checkpoint_layers = not args.no_operator_checkpoint
    config = ReadConfig(candidates=candidates, keep=keep, hidden=lm.config.hidden_size,
                        span_width=lm.get_input_embeddings().weight.shape[1],
                        target_norm=model.target_norm, state=dims['state'],
                        op_hidden=dims['hidden'], layers=dims['layers'],
                        key_hidden=args.key_hidden, gate_offset=args.gate_offset,
                        max_reps=args.max_reps, checkpointing=not args.no_operator_checkpoint)
    torch.manual_seed(args.seed)
    reader = L1Reader(config, stack)
    reader.keys.load_state_dict(torch.load(args.banks / 'key_heads_init.pt', map_location='cpu'))
    reader.to(model.device)
    sets = parameter_sets(reader, model)
    groups = [{'params': [p for name in ('keys', 'operators', 'recombiner', 'codecs')
                          for p in sets[name]], 'lr': args.lr, 'weight_decay': 0.01},
              {'params': sets['writer'], 'lr': args.writer_lr, 'weight_decay': 0.0}]
    for p in sets['writer'] + sets['codecs']:     # AdamW takes them; phases enable them
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    set_phase(sets, L1A_SET)
    usage: dict = {}
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'reader.pt'
    kbs = open_live(args.banks, args.output, args.live_device, args.sync_every)
    if not state_path.exists():
        for kb in kbs.values():         # a crash before the first save restarts from the banks
            written = any(p == 'write' for s in kb.spaces
                          for p, _ in producer_index(kb, s).values())
            if kb.live_updates or written:
                raise ValueError(f'{kb.root} has live updates or writes but no reader '
                                 'checkpoint; remove the output directory to restart')
    step = 0
    rng = random.Random(args.seed)
    log = WriteLog(args.output / 'writes') if args.writes else None
    if state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        reader.load_state_dict(state['reader'])
        optimizer.load_state_dict(state['optimizer'])
        if 'writer' in state:
            model.writer.load_state_dict(state['writer'])
        step = state['step']
        rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        from schnitz.kb.losses import UsageEMA
        for key, (share, touched) in state.get('usage', {}).items():
            usage[key] = UsageEMA(len(share))
            usage[key].share, usage[key].touched = share.cpu(), touched.cpu()
        # items and keys back to exactly the state paired with the reader checkpoint
        # (writes committed after it are discarded with their log entries)
        for name, kb in kbs.items():
            kb.restore_live(state['live_tag'], discard_commits=True)
            if kb.live_updates != state['live_updates'][name]:
                raise ValueError(f'{name}: restored live state does not match the checkpoint')
        if log is not None:
            log.truncate(step)
    ctx = Context(frozen, reader, kbs)
    tok = model.tok
    writer = Writer(model, stack, reader, frozen, batch=args.write_batch) \
        if args.writes or 'l1b' in phases else None
    producers = load_producers(args, ctx, writer, log, model) if 'l1b' in phases else None
    args.write_level_index = LEVELS.index(args.write_level)
    if args.init_reader and step == 0:
        loaded = load_init_reader(reader, args.init_reader)
        for kb in kbs.values():      # the search keys from the loaded item-key heads
            reader.rekey(kb)
        print(json.dumps({'init_reader': str(args.init_reader), 'loaded': len(loaded)}),
              flush=True)

    skipped: dict[str, int] = {}

    def episode(row) -> Episode | None:
        """The layout of a usable transcript, else None (counted by reason)."""
        reason = None
        if not slots_of(row) or row['kb'] not in kbs:
            reason = 'no_reads_or_kb'
        else:
            ep = layout(row, tok, writes=args.writes and bool(row.get('write_sites')))
            if ep.ids.numel() > args.max_tokens:
                reason = 'too_long'
            elif not ctx.covered(ep):
                reason = 'records_missing'
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            return None
        return ep

    train_rows = Transcripts(args.transcripts, 'train', args.limit)
    eval_eps = [ep for ep in map(episode, Transcripts(args.transcripts, 'validation',
                                                      args.eval_limit)) if ep is not None]
    eval_skip, skipped = dict(skipped), {}
    eval_eps = eval_eps[:args.eval_items]
    wanted = {r for ep in eval_eps for slot in ep.slots for r in slot['record_ids']}
    texts = {r: v['text'] for r, v in
             read_sources({ep.row['_dir'] for ep in eval_eps}, wanted).items()}
    (args.output / 'config.json').write_text(json.dumps(dict(
        vars(args), read_config=dataclasses.asdict(config),
        params=sum(p.numel() for p in reader.parameters()), train_transcripts=len(train_rows),
        eval_episodes=len(eval_eps), eval_skipped=eval_skip,
        kbs={k: kb.stats() for k, kb in kbs.items()}), indent=2, default=str) + '\n')
    metrics = (args.output / 'metrics.jsonl').open('a', encoding='utf-8')

    def log_record(record):
        metrics.write(json.dumps(record) + '\n')
        metrics.flush()
        print(json.dumps(record), flush=True)

    def save():
        """The reader checkpoint paired with an exact live checkpoint of every KB
        (``checkpoint_live``); the previous pair is dropped once the new one is in place."""
        tag = f'step{step:08d}'
        for kb in kbs.values():
            if tag not in kb.live_checkpoints():
                kb.checkpoint_live(tag)
        pending = state_path.with_suffix('.pending')
        torch.save({'reader': reader.state_dict(), 'optimizer': optimizer.state_dict(),
                    'writer': model.writer.state_dict(),
                    'step': step, 'rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                    'live_tag': tag,
                    'usage': {k: (u.share, u.touched) for k, u in usage.items()},
                    'live_updates': {k: kb.live_updates for k, kb in kbs.items()},
                    'config': dataclasses.asdict(config)}, pending)
        pending.replace(state_path)
        for kb in kbs.values():
            for old in kb.live_checkpoints():
                if old != tag:
                    kb.drop_live_checkpoint(old)

    if step == 0 and args.eval_every:
        log_record({'step': 0, 'eval': evaluate(ctx, eval_eps, texts, tok)})
    order: list[int] = []
    window: dict[str, list] = {}
    started = time.time()

    def rekey() -> None:     # the search's key cache from the current item-key heads
        for kb in kbs.values():
            reader.rekey(kb)
    while step < args.steps:
        batch = []
        while len(batch) < args.batch_size:
            if not order:
                order = list(range(len(train_rows)))
                rng.shuffle(order)
            ep = episode(train_rows[order.pop()])
            if ep is not None:
                batch.append(ep)
        phase = phase_at(schedule, step)
        trainable = set_phase(sets, L1A_SET if phase == 'l1a' else l1b_set)
        reader.train()
        result = train_step(ctx, batch, optimizer, args, step, usage, phase=phase,
                            trainable=trainable, writer=writer if args.writes else None,
                            producers=producers, log=log, rng=rng, text_ids=model.text_ids)
        step += 1
        for key, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        window.setdefault(f'{key}_{k}', []).append(v)
            elif isinstance(value, (int, float)):
                window.setdefault(key, []).append(value)
        window.setdefault(f'steps_{phase}', []).append(1)
        if args.rekey_every and step % args.rekey_every == 0:
            rekey()
        if step % args.log_every == 0:
            log_record({'step': step, 'phase': phase, **_mean(window), 'skipped': dict(skipped),
                        'usage': {k: u.stats() for k, u in usage.items()} if args.log_usage
                        else None, 'elapsed_s': round(time.time() - started)})
            window = {}
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            rekey()
            save()
            reader.eval()
            log_record({'step': step, 'eval': evaluate(ctx, eval_eps, texts, tok)})


def add_args(parser: argparse.ArgumentParser) -> None:
    """Arguments of both actions (``build`` and ``train``) on one parser."""
    parser.add_argument('action', choices=('build', 'train'))
    parser.add_argument('--transcripts', type=Path, nargs='+', required=True,
                        help='memory transcript directories (v1 or v2)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--reader-state', type=Path, required=True,
                        help='B3 writer.pt (merged decoder); later B4')
    parser.add_argument('--codecs', type=Path, help='K1 stack.pt; random init when omitted')
    parser.add_argument('--limit', type=int, help='train transcripts per directory')
    parser.add_argument('--eval-limit', type=int, default=256,
                        help='validation transcripts per directory')
    parser.add_argument('--query-layer', type=int, default=8)
    parser.add_argument('--key-hidden', type=int, default=512)
    parser.add_argument('--cuda-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, help='build: records (32); train: episodes (8)')
    build_args = parser.add_argument_group('build')
    build_args.add_argument('--span-source', choices=('writer', 'teacher'), default='writer')
    build_args.add_argument('--level', default='s0', help='ratio level of the written spans')
    build_args.add_argument('--span-cache', type=Path,
                            help='SpanCache directory (default <output>/spans)')
    build_args.add_argument('--cache', type=Path, help='B1 teacher cache (span source teacher)')
    build_args.add_argument('--span-batch-size', type=int,
                            help='records per writer free run (default --batch-size). On '
                                 'the GPU the free run depends on the batch composition '
                                 '(padding), so L1b replays a bank exactly only when its spans '
                                 'were written one at a time (1)')
    build_args.add_argument('--distractors', type=int, default=0,
                            help='extra records per KB beyond those the transcripts name')
    t = parser.add_argument_group('train')
    t.add_argument('--banks', type=Path, help='output of build')
    t.add_argument('--steps', type=int, default=20000)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--item-lr', type=float, default=3e-3)
    t.add_argument('--retrieval-weight', type=float, default=0.5)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--candidates', default='', help='scored per space, e.g. A=8,B=16,C=32,D=64')
    t.add_argument('--keep', default='', help='read per space (nonzero gates), e.g. A=2,D=4')
    t.add_argument('--gate-offset', type=float, default=0.5,
                   help='initial gate offset b_s in cosine units')
    t.add_argument('--balance-weight', type=float, default=0.01,
                   help='spread-out use (balance loss over scored candidates)')
    t.add_argument('--log-usage', action='store_true')
    t.add_argument('--rekey-every', type=int, default=25,
                   help='refresh the stored (search) keys from the item-key heads')
    t.add_argument('--retrieval-anneal', type=int, default=0,
                   help='steps over which the retrieval weight decays linearly to '
                        '--retrieval-floor (0: constant)')
    t.add_argument('--retrieval-floor', type=float, default=0.0)
    t.add_argument('--retrieval-only', action='store_true',
                   help='K2: only the retrieval loss; trains key heads and item keys')
    t.add_argument('--inbatch-negatives', type=int, default=64,
                   help='per space: target items of the batch\'s other slots (same KB only) '
                        'scored as negatives in the retrieval loss (0: off)')
    t.add_argument('--init-reader', type=Path,
                   help='K2 reader.pt: start from its key heads and gate offsets (K2 -> L1a)')
    t.add_argument('--phase', choices=('l1a', 'l1b'), default='l1a',
                   help='the phase when there is no --phase-schedule')
    t.add_argument('--phase-schedule',
                   help='alternating phases, cycled, e.g. a:2000,b:500 (L1a then L1b steps)')
    t.add_argument('--l1b-train', default='writer,codecs,keys,operators,recombiner',
                   help=f'parameter sets trained in L1b, from {",".join(PHASE_SETS)}')
    t.add_argument('--l1b-replay', choices=('free', 'teacher'), default='free',
                   help='L1b producer replay: the free run itself (the stored forward) or '
                        'teacher-fed on the stored span (one pass)')
    t.add_argument('--l1b-batch', type=int, default=1,
                   help='bank sources per producer pass (1: the replay is exact when the bank\'s '
                        'spans were written one at a time, build --span-batch-size 1)')
    t.add_argument('--writer-lr', type=float, default=3e-5, help='L1b: writer span heads')
    t.add_argument('--writes', action='store_true',
                   help='v3 write sites: the frozen writer generates each site\'s span in '
                        'place and its items enter the episode\'s KB for later steps')
    t.add_argument('--write-level', default='s0', choices=LEVELS,
                   help='ratio level of in-context write spans (B4c\'s length schedule)')
    t.add_argument('--write-batch', type=int, default=8)
    t.add_argument('--max-reps', type=int, default=16, help='span budget per read')
    t.add_argument('--max-tokens', type=int, default=3072)
    t.add_argument('--live-device', default='cpu',
                   help='where the resident live state lives (cpu or cuda; reads gather once '
                        'per space and call; CPU and GPU share memory on Spark)')
    t.add_argument('--sync-every', type=int, default=500,
                   help='resident live state: sync to disk every N updates (resume is exact '
                        'through the live checkpoints taken with each reader checkpoint)')
    t.add_argument('--eval-every', type=int, default=500)
    t.add_argument('--eval-items', type=int, default=128)
    t.add_argument('--log-every', type=int, default=25)
    t.add_argument('--decoder-checkpoint', action='store_true',
                   help='recompute frozen decoder layers in backward')
    t.add_argument('--no-operator-checkpoint', action='store_true')


def run(args) -> None:
    if args.action == 'build':
        args.batch_size = args.batch_size or 32
        build(args)
    else:
        if args.banks is None:
            raise SystemExit('train needs --banks')
        args.batch_size = args.batch_size or 8
        train(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
