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
A bank that an L1b phase will replay must be built with ``--span-batch-size 1``: the
writer's free run on the GPU depends on its batch composition, and only spans written
one at a time are what L1b's one-at-a-time replay recomputes (the default batches the
spans for throughput, about 3x faster; ``l1 train`` warns on such a bank).

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
- *L1b, through the sources* (``schnitz.kb.producer.Producers``, the replay L2 and
  B9 use too): the items a read keeps are recomputed from their stored sources (a
  bank item: the writer's free run of its record under the memory prompt, then the
  codecs; a written item: the writer's free run at its write site from the logged
  prefix and reads, ``WriteLog``), at the stored forward's serialized precision and
  batch composition, so at the start of L1b the recomputed items equal the stored
  payloads (bank spans written with ``build --span-batch-size 1``; writes replayed
  with their writer batch; checked every step: ``l1b_match_*``). The gradients of all
  reads of a step accumulate on the recomputed items, then the producers are
  recomputed with gradients and the task loss reaches the writer's span heads (ratio
  code, rep head) and the codecs (``--l1b-train`` names the sets; default also keys,
  S_s and R). The producers have their own learning rates (``--l1b-codec-lr``,
  ``--l1b-writer-lr``); after each step a few replay units are recomputed to log how
  far the step moved their items (``l1b_change_rel``). Live item values are not
  updated in L1b; an item modified in place by L1a is read as its producers'
  recomputation there (L2 reconciles the two).

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
from schnitz.kb.producer import Producers, Writer, WriteLog, produce_items, producer_params
from schnitz.kb.read import (DEFAULT_CANDIDATES, DEFAULT_KEEP, ItemCache, KeyOptimizer,
                             L1Reader, ReadConfig, current_ids, producer_index, source_index,
                             splice)
from schnitz.kb.stack import SPACES, KeyHeads
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
                 autocast=None, views: dict | None = None):
        self.frozen, self.reader, self.kbs = frozen, reader, kbs
        self.autocast = autocast or frozen.autocast
        # --rows-from-stack: per dataset a ``superpose.SuperposedKB`` whose rows (combiner
        # outputs over this KB's items, the leaves) are what reads see
        self.views = views or {}
        self.key_optimizer = None      # learned keys (``read.KeyOptimizer``), set by train
        # item id -> (producer, sources) of the current items, per KB and space
        self.origin = {name: {s: producer_index(kb, s) for s in kb.spaces}
                       for name, kb in kbs.items()}
        self.index = {name: {s: _by_source(origin) for s, origin in spaces.items()}
                      for name, spaces in self.origin.items()}
        self.written = {name: {s: {i for i, (p, _) in origin.items() if p == 'write'}
                               for s, origin in spaces.items()}
                        for name, spaces in self.origin.items()}
        self._rows: dict[tuple[str, str], tuple[int, dict[str, int]]] = {}

    def new_cache(self, train: bool, step: int = 0, producer=None):
        """The step's item cache: an ``ItemCache``, or with views a ``SuperposedCache``."""
        if self.views:
            from schnitz.kb.superpose import SuperposedCache
            return SuperposedCache(self.views, self.frozen.device, train=train, step=step,
                                   producer=producer)
        return ItemCache(self.frozen.device, train=train)

    def added(self, dataset: str, space: str, ids: list[str], producer: str,
              sources: list[tuple[str, ...]]) -> None:
        """Items committed during training (writes): origin, written set, row cache; with
        views the new leaves are placed into the fields of their nearest rows."""
        for item_id, src in zip(ids, sources):
            self.origin[dataset][space][item_id] = (producer, src)
            if producer == 'write':
                self.written[dataset][space].add(item_id)
        self._rows.pop((dataset, space), None)
        view = self.views.get(dataset)
        if view is not None and space in view.graphs and ids:
            view.insert(space, ids)

    def own_writes(self, ep: Episode) -> dict[str, set[tuple[str, str]]]:
        """An episode's own write items per space: never retrieved by its own reads (a
        revisited episode would otherwise read what it wrote about its own answer)."""
        return {s: {(ep.kb, write_item_id(ep.episode_id, j, s)) for j in range(len(ep.writes))}
                for s in self.kbs[ep.kb].spaces} if ep.writes else {}

    def rows(self, dataset: str, space: str) -> dict[str, int]:
        """Row index of the current version of every item id of a space (for usage
        statistics); rebuilt when rows were appended (writes, supersedes)."""
        key = (dataset, space)
        names = self.kbs[dataset]._row_ids[space]
        cached = self._rows.get(key)
        if cached is None or cached[0] != len(names):
            cached = self._rows[key] = (len(names), {i: r for r, i in enumerate(names)})
        return cached[1]

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
                negatives: dict | None = None, producer=None,
                weights: dict[str, float] | None = None):
    """Task NLL (summed over target tokens) of one transcript and its reads.

    ``mode`` 'retrieve' or 'gold' computes each read at its call from the exact causal
    prefix; 'fixed' splices the given ``spans`` (controls). ``retrieval_only`` (K2):
    the reads' spans enter later prefixes detached and no task pass runs (NLL None),
    so only the retrieval loss trains, through the queries and the item keys.
    ``negatives`` (per space, the episode's own KB only): in-batch negatives of the
    retrieval loss. ``producer`` (L1b): read items' values from their sources. Reads
    never retrieve the episode's own write items (``Context.own_writes``). ``weights``:
    per-item gate multipliers (item id -> w; B9's gold weight)."""
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
                                           producer=None if mode == 'gold'
                                           or getattr(cache, 'superposed', False) else producer,
                                           weights=weights)
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


# -- in-context writes ------------------------------------------------------------------
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


def generate_writes(writer: Writer, requests: list[WriteRequest]) -> list[torch.Tensor]:
    """Each site's span, free-running from its causal prefix (``<|bg|>`` marker at the
    site's ratio), as the frozen writer generates it in place (``Writer.generate``)."""
    examples = [{'inputs': writer.inputs(r.prefix_ids, r.mems, r.reads), 'factor': r.factor}
                for r in requests]
    return writer.generate(examples, [r.count for r in requests])


def commit_writes(ctx: Context, writer: Writer, requests: list[WriteRequest],
                  spans: list[torch.Tensor], step: int, log: WriteLog | None = None) -> int:
    """Items of each written span (codecs per space, keys from the item-key heads) into
    the episode's own KB: time = the episode's query time, provenance ``write`` with
    the write site as source and ``step``; a site written before is superseded (same
    ids). Returns the number of writes."""
    groups = {}
    for start in range(0, len(requests), writer.batch):   # ``generate_writes``' batches
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
                log.add(req.source, kb=req.ep.kb, prefix_ids=req.prefix_ids, mems=req.mems,
                        reads=req.reads, factor=req.factor, span=span, step=step,
                        group=groups[req.source])
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
            with model.core.autocast():     # the producers' codec step (schnitz.kb.producer)
                encoded = [produce_items(stack, span_of(r)) for r in batch]
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


def _read_stats(reads, sink: dict, written: dict | None = None, cache=None) -> None:
    """Per-read statistics; with ``written`` (dataset -> space -> ids of written items)
    also how often a read retrieves items written by earlier episodes (``written_s``:
    share of reads with a written item among those read; ``written_mass_s``: share of
    the read's gate mass on written items). Own writes are excluded from reads. With a
    superposed ``cache`` a row counts by the share of its mass from written leaves."""
    shared = getattr(cache, 'superposed', False)
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
                if shared:
                    share = [cache.share_of(d, s, i, set(written.get(d, {}).get(s, ())))
                             for d, i in info.refs]
                else:
                    share = [float(i in written.get(d, {}).get(s, ())) for d, i in info.refs]
                gates = info.gates.float()
                sink.setdefault(f'written_{s}', []).append(float(any(x > 0 for x in share)))
                sink.setdefault(f'written_mass_{s}', []).append(
                    float((gates * torch.tensor(share)).sum() / gates.sum().clamp_min(1e-12)))


def _mean(sink: dict) -> dict:
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(sink.items()) if v}


def retrieval_weight(args, step: int) -> float:
    """The retrieval loss weight, annealed linearly to its floor as the task loss takes
    over (stack doc 5.2); constant in K2 (``--retrieval-only``)."""
    if args.retrieval_only or not args.retrieval_anneal:
        return args.retrieval_weight
    done = min(step / args.retrieval_anneal, 1.0)
    return args.retrieval_weight + done * (args.retrieval_floor - args.retrieval_weight)


def _balance(ctx: Context, reads, usage: dict, cache=None) -> torch.Tensor | None:
    """Spread-out use (stack doc 5.2): ``balance_loss`` per (KB, space) over every
    scored (read, candidate) pair of the episode's reads, averaged; updates the
    usage averages with the read mass. Indexed by store row (superseded rows keep
    their slot), or with a superposed cache by the view's row slots."""
    from schnitz.kb.losses import UsageEMA, balance_loss
    shared = getattr(cache, 'superposed', False)
    pairs: dict[tuple[str, str], tuple[list[int], list[torch.Tensor]]] = {}
    for read in reads:
        for s, info in read.spaces.items():
            if info.scored_gates is None:
                continue
            for (dataset, item_id), gate in zip(info.scored, info.scored_gates):
                rows = cache.rows(dataset, s) if shared else ctx.rows(dataset, s)
                entry = pairs.setdefault((dataset, s), ([], []))
                entry[0].append(rows[item_id])
                entry[1].append(gate)
    losses = []
    for (dataset, s), (ids, gates) in pairs.items():
        key = f'{dataset}/{s}'
        size = len(cache.rows(dataset, s)) if shared else len(ctx.kbs[dataset]._row_ids[s])
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


# which parameter sets train in each phase (``--l1b-train`` chooses L1b's). ``write``: the
# aggregators of ``--rows-from-stack`` (empty otherwise)
PHASE_SETS = ('keys', 'operators', 'recombiner', 'codecs', 'writer', 'write')
L1A_SET = ('keys', 'operators', 'recombiner')
READ_SET = ('keys', 'operators', 'recombiner')
WRITE_SET = ('write',)
PHASES = {'a': 'l1a', 'b': 'l1b', 'r': 'r', 'w': 'w'}


def parameter_sets(reader: L1Reader, writer_model=None, write_ops=None) -> dict[str, list]:
    """The trainable parameter sets: key heads and gate offsets, S_s (read-time combine,
    ``--read-combine s_s``), R (the read's), the codecs, the writer's span heads (marker,
    ratio code, rep head) and the aggregators (``--rows-from-stack``)."""
    return {'keys': list(reader.keys.parameters()) + list(reader.gate_offset.parameters()),
            'operators': list(reader.operators.parameters()),
            'recombiner': list(reader.recombiner.parameters()),
            **producer_params(writer_model, reader.stack),
            'write': [] if write_ops is None else list(write_ops.parameters())}


def phase_set(phase: str, args, l1b_set=()) -> tuple[str, ...]:
    """The sets a phase trains: ``l1a`` the read side, with ``--rows-from-stack`` also the
    aggregators (only on consolidation steps when ``--consolidate-every``); ``r`` the read
    side; ``w`` the aggregators (the leaves or rows move by their live update); ``l1b``
    ``--l1b-train``."""
    if phase == 'l1b':
        return tuple(l1b_set)
    if phase == 'r':
        return READ_SET
    if phase == 'w':
        return WRITE_SET
    joint = getattr(args, 'rows_from_stack', None) and not getattr(args, 'consolidate_every', 0)
    return L1A_SET + (WRITE_SET if joint else ())


def optimizer_groups(sets: dict[str, list], args) -> list[dict]:
    """AdamW groups: the reader (keys, S_s, R) at ``--lr``; the producers at their own
    L1b rates (``--l1b-codec-lr``, ``--l1b-writer-lr``), since one step at the reader's
    rate moved the recomputed items by 25-45%; the aggregators (``--rows-from-stack``)
    at ``--write-lr``."""
    groups = [{'params': [p for name in ('keys', 'operators', 'recombiner') for p in sets[name]],
               'lr': args.lr, 'weight_decay': 0.01},
              {'params': sets['codecs'], 'lr': args.l1b_codec_lr, 'weight_decay': 0.01},
              {'params': sets['writer'], 'lr': args.l1b_writer_lr, 'weight_decay': 0.0}]
    if sets.get('write'):
        groups.append({'params': sets['write'], 'lr': getattr(args, 'write_lr', None) or args.lr,
                       'weight_decay': 0.01})
    return groups


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
    """``a:2000,b:500`` -> [('l1a', 2000), ('l1b', 500)] (cycled); ``r`` (read side
    only) and ``w`` (aggregators and leaves) alternate the two sides; without a
    schedule the single ``phase``."""
    if not text:
        return [(phase, 1)]
    out = []
    for part in text.split(','):
        name, _, count = part.partition(':')
        name = PHASES.get(name.strip(), name.strip())
        if name not in PHASES.values() or not count.strip().isdigit() or int(count) < 1:
            raise ValueError(f'bad phase schedule entry {part!r} (e.g. a:2000,b:500, r:100,w:100)')
        out.append((name, int(count)))
    return out


def phase_at(schedule: list[tuple[str, int]], step: int, read_warmup: int = 0) -> str:
    """The phase of 0-based ``step`` under a cycled schedule (the first ``read_warmup``
    steps are read-side only)."""
    if step < read_warmup:
        return 'r'
    at = (step - read_warmup) % sum(n for _, n in schedule)
    for name, n in schedule:
        if at < n:
            return name
        at -= n
    raise AssertionError


def train_step(ctx: Context, episodes: list[Episode], optimizer, args, step: int = 0,
               usage: dict | None = None, *, phase: str | None = None,
               trainable: list | None = None, writer: Writer | None = None,
               producers: Producers | None = None, log: WriteLog | None = None,
               rng: random.Random | None = None, text_ids=None, anchor=None) -> dict:
    """One step: gradients of all episodes accumulate, then one optimizer step and (L1a)
    one sparse live update per touched (KB, space) for the item values.

    L1a: items in place, writer detached. L1b: read items recomputed from their
    sources (``producers``), their accumulated gradients backpropagated into the
    producers before the optimizer step; live item values are not updated. K2 is
    L1a with ``--retrieval-only``. ``r``: the read side only (no live update; with
    ``--rows-from-stack`` the rows' gradients are not propagated into the fields);
    ``w``: the aggregators and the leaves. With ``--rows-from-stack`` the rows are
    recomputed with their graphs before the optimizer step (``SuperposedCache.backward``);
    on a consolidation step (``--consolidate-every``) the aggregators step and the leaves
    are re-fitted so the touched rows return to their values before it, instead of the
    leaves' own update. ``anchor`` (``--read-anchor``) adds the probe-set KL. With a
    ``writer`` and episodes with write sites, the sites' spans are generated at the end
    of the step (after every read of the step: the batch's episodes never see each
    other's writes) and committed to the episodes' KBs, so reads of later steps can
    retrieve them."""
    phase = phase or args.phase
    if phase == 'l1b' and producers is None:
        raise ValueError('L1b needs the producers (gradients through the sources)')
    usage = {} if usage is None else usage
    started = time.time()
    producer = producers if phase == 'l1b' else None
    cache = ctx.new_cache(train=True, step=step, producer=producer)
    optimizer.zero_grad(set_to_none=True)
    tokens = sum(int(ep.targets.numel()) for ep in episodes)
    weight = retrieval_weight(args, step)
    stats: dict[str, list] = {}
    nll_total, aux_total, balance_total = 0.0, 0.0, 0.0
    negatives = batch_negatives(ctx, episodes, args.inbatch_negatives,
                                rng or random.Random(step)) if args.inbatch_negatives else {}
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
            balance = _balance(ctx, reads, usage, cache)
            if balance is not None:
                terms.append(args.balance_weight * balance / len(episodes))
                balance_total += balance.item() / len(episodes)
        loss = sum(terms) if terms else None
        if loss is not None and loss.requires_grad:
            loss.backward()
        if nll is not None:
            nll_total += nll.item()
        _read_stats(reads, stats, ctx.written, cache)
        if writer is not None and ep.writes:
            requests += write_requests(ep, spans, text_ids, args.write_level_index)
    out: dict = {}
    if anchor is not None and getattr(args, 'read_anchor', 0) > 0 and anchor.probes:
        a_loss, a_stats = anchor.loss(rng or random.Random(step))
        (args.read_anchor * a_loss).backward()
        out['anchor'] = a_stats
    consolidating = bool(getattr(args, 'consolidate_every', 0) and ctx.views
                         and phase in ('l1a', 'w')
                         and (step + 1) % args.consolidate_every == 0)
    before = {}
    if getattr(cache, 'superposed', False):
        touched = cache.touched_tops()
        if consolidating:
            before = {ref: cache.tops[ref].detach().clone() for ref in touched}
        if phase != 'r':
            t0 = time.time()
            out['superpose'] = cache.backward(getattr(args, 'write_balance', 0.0), usage)
            out['superpose']['backward_s'] = round(time.time() - t0, 3)
    if producer is not None:
        t0 = time.time()
        out['l1b'] = producer.backward()
        out['l1b_backward_s'] = round(time.time() - t0, 3)
    params = trainable if trainable is not None else ctx.reader.trainable()
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    optimizer.step()
    change_units = getattr(args, 'l1b_change_units', 4)
    if producer is not None and change_units >= 0:
        # how far the step moved the recomputed items (at the producers' own rates)
        t0 = time.time()
        out['l1b'].update(producer.change(change_units))
        out['l1b_change_s'] = round(time.time() - t0, 3)
    item_lr = 0.0 if args.retrieval_only or phase in ('l1b', 'r') else args.item_lr
    if consolidating:
        from schnitz.kb.superpose import consolidate
        t0 = time.time()
        out['consolidate'] = consolidate(ctx.views, list(before), before,
                                         steps=args.consolidate_steps, item_lr=args.item_lr,
                                         device=ctx.frozen.device)
        out['consolidate']['s'] = round(time.time() - t0, 3)
        counts = {'items': 0}
    else:
        keys = getattr(ctx, 'key_optimizer', None) if item_lr > 0 else None
        counts = cache.apply(item_lr, key_optimizer=keys)
    if requests:
        t0 = time.time()
        spans = generate_writes(writer, requests)
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


# -- superposition metrics, drift and the read anchor -------------------------------------
def superposition_metrics(ctx: Context, reads_by: dict) -> dict:
    """Per KB and space: sources per row, rows per source (items per source), effective
    items per read (from the evaluation reads' gates). With views from the fields
    (``SuperposedKB.metrics``); otherwise from the KB's rewrite lineage (a rows KB), or
    one source per item for a plain KB."""
    from schnitz.kb.read import current_ids
    from schnitz.kb_eval import superposition_report
    out = {}
    for name, kb in ctx.kbs.items():
        reads = reads_by.get(name, {})
        if name in ctx.views:
            out[name] = ctx.views[name].metrics(reads)
            continue
        comp = kb.source_composition()
        out[name] = {}
        for s in kb.spaces:
            ids = current_ids(kb, s)
            rep = superposition_report({i: comp[i] for i in ids}, reads.get(s, ()))
            out[name][s] = {'items': rep['items'], 'sources': rep['sources'],
                            'sources_per_item': rep['sources_per_item'],
                            'items_per_source': rep['items_per_source'],
                            'effective_items_per_read': rep['effective_items_per_read']['entropy']}
    return out


def summarize_metrics(metrics: dict) -> dict:
    """Per space, the means over KBs of the main superposition numbers."""
    out: dict[str, dict] = {}
    for spaces in metrics.values():
        for s, m in spaces.items():
            row = out.setdefault(s, {})
            for key in ('sources_per_item', 'items_per_source', 'effective_items_per_read'):
                value = m.get(key, {})
                if isinstance(value, dict) and value.get('n'):
                    row.setdefault(key, []).append(value['mean'])
            if 'leaves_per_row' in m:
                row.setdefault('leaves_per_row', []).append(m['leaves_per_row']['mean'])
    return {s: {k: round(sum(v) / len(v), 3) for k, v in row.items() if v}
            for s, row in out.items()}


@torch.no_grad()
def item_drift(ctx: Context, sample: int = 256) -> dict:
    """Per space: mean relative distance of the (live) items from their stored payloads
    (for a rows KB: the rows' drift from their codec-derived initialization), and
    ``key_<s>`` the keys' drift (1 - cosine of the live to the stored key), over up to
    ``sample`` items per KB. With ``--rows-from-stack`` the leaves' live keys hold the
    key-head corrections, so ``corr_<s>`` reports their mean norm instead."""
    from schnitz.kb.read import current_ids
    corrections = bool(ctx.views)
    rel: dict[str, list[float]] = {}
    for kb in ctx.kbs.values():
        for s in kb.spaces:
            if not (kb.writable and kb.is_live(s)):
                continue
            ids = current_ids(kb, s)[:sample]
            if not ids:
                continue
            for live, stored in zip(kb.read(s, ids, live=True), kb.read(s, ids)):
                a, b = live.values.float(), stored.values.float()
                rel.setdefault(s, []).append(float((a - b).norm() / b.norm().clamp_min(1e-12)))
                if corrections:
                    rel.setdefault(f'corr_{s}', []).append(float(live.key.float().norm()))
                else:
                    rel.setdefault(f'key_{s}', []).append(1 - float(F.cosine_similarity(
                        live.key.float(), stored.key.float(), dim=-1)))
    return {s: round(sum(v) / len(v), 5) for s, v in rel.items()}


class ReadAnchor:
    """``--read-anchor``: KL of the read side's reads on a fixed probe set of held-out
    episodes of every KB (the first read of each) against their outputs at the last
    refresh (every ``--anchor-every`` steps). A probe keeps the query-layer state and the
    items it read (values and scales, detached); its read is recomputed by
    ``L1Reader.reread`` (gradients into the key/query heads and R), then the frozen
    decoder reads ``<|mem|> span <|/mem|>`` followed by the first ``tokens`` tokens of the
    slot's first record text; the loss is KL(anchored || current) on those tokens."""

    def __init__(self, ctx: Context, episodes: list[Episode], texts: dict[str, str], text_ids,
                 tokens: int = 32, batch: int = 4):
        self.ctx, self.episodes, self.batch = ctx, episodes, batch
        self.texts = {}
        for ep in episodes:
            r = ep.slots[0]['record_ids'][0]
            if r in texts:
                self.texts[ep.episode_id] = text_ids(texts[r])[:tokens]
        self.probes: list[dict] = []

    @torch.no_grad()
    def refresh(self, step: int) -> int:
        ctx = self.ctx
        cache = ctx.new_cache(train=False, step=step)
        self.probes = []
        for ep in self.episodes:
            if ep.episode_id not in self.texts or not ep.calls:
                continue
            embeds = ctx.frozen.embed(ep.ids[:ep.calls[0] + 1])
            h = ctx.frozen.mid(embeds[None])[0][ep.calls[0]]
            with ctx.autocast():
                read = ctx.reader.read(h, [ctx.kbs[ep.kb]], [ep.kb], ep.query_time, cache)
            kb = ctx.kbs[ep.kb]
            values = {s: [v.detach().clone() for v, _ in cache.get(kb, s, [i for _, i in info.refs])]
                      for s, info in read.spaces.items() if info.refs}
            keys = {s: [k.detach().clone() for k in cache.keys(kb, s, [i for _, i in
                                                                    read.spaces[s].refs])]
                    for s in values} if ctx.reader.config.learned_keys else None
            scales = {s: list(read.spaces[s].scales) for s in values}
            if not values:
                continue
            probe = {'state': h, 'values': values, 'scales': scales, 'keys': keys,
                     'ids': self.texts[ep.episode_id]}
            probe['anchored'] = self._logits(probe, read.span.detach().float())
            self.probes.append(probe)
        return len(self.probes)

    def _logits(self, probe: dict, span: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([torch.tensor([MEM_ID, SPAN_TOKENS['mem_end'][1]]), probe['ids']])
        x, index = splice(self.ctx.frozen.embed(ids), [0], [span])
        hidden = self.ctx.frozen.final(x[None])[0]
        return self.ctx.frozen.logits(hidden[index[2:] - 1])

    def span(self, probe: dict) -> torch.Tensor:
        with self.ctx.autocast():
            return self.ctx.reader.reread(probe['state'], probe['values'], probe['scales'],
                                          probe['keys']).float()

    def loss(self, rng: random.Random) -> tuple[torch.Tensor, dict]:
        from schnitz.kb.losses import kl
        picked = rng.sample(self.probes, min(self.batch, len(self.probes)))
        terms = [kl(self._logits(p, self.span(p)), p['anchored']) for p in picked]
        loss = torch.stack(terms).mean()
        return loss, {'kl': round(loss.item(), 6), 'probes': len(picked)}

    @torch.no_grad()
    def drift(self) -> dict:
        """The probe set's KL to the anchored reads now (read-output drift)."""
        if not self.probes:
            return {}
        from schnitz.kb.losses import kl
        values = [float(kl(self._logits(p, self.span(p)), p['anchored'])) for p in self.probes]
        return {'kl': round(sum(values) / len(values), 6), 'probes': len(values)}


@torch.no_grad()
def evaluate(ctx: Context, episodes: list[Episode], texts: dict[str, str], tok,
             previous: dict | None = None) -> dict:
    """The evaluation arms, read statistics, superposition metrics per KB and space, the
    items' drift from their stored payloads and, given ``previous`` (the last evaluation's
    per-episode retrieved NLL), retention per KB (``kb_eval.retention``, lower NLL is
    better). ``report['_per_episode']`` carries this evaluation's per-episode NLL."""
    from schnitz.kb_eval import retention
    cache = ctx.new_cache(train=False)
    sums = {a: 0.0 for a in ('noctx', 'full', 'retrieved', 'shuffled', 'gold', 'gold_shuffled')}
    tokens = 0
    stats: dict[str, list] = {}
    gold_stats: dict[str, list] = {}
    got = {'retrieved': [], 'gold': []}
    per_episode: dict[str, tuple[str, float]] = {}
    gates_by: dict[str, dict[str, list]] = {}
    for ep in episodes:
        for mode, name in (('retrieve', 'retrieved'), ('gold', 'gold')):
            nll, n, reads, spans = run_episode(ctx, ep, cache, mode)
            sums[name] += nll.item()
            got[name].append(spans)
            _read_stats(reads, stats if name == 'retrieved' else gold_stats,
                        ctx.written if name == 'retrieved' else None, cache)
            if name == 'retrieved':
                per_episode[ep.episode_id] = (ep.kb, nll.item() / max(n, 1))
                for read in reads:
                    for s, info in read.spaces.items():
                        if len(info.gates):
                            gates_by.setdefault(ep.kb, {}).setdefault(s, []).append(
                                info.gates.tolist())
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
    metrics = superposition_metrics(ctx, gates_by)
    report['superposition'] = summarize_metrics(metrics)
    report['superposition_kbs'] = metrics
    report['item_drift'] = item_drift(ctx)
    if previous:
        by_kb: dict[str, tuple[dict, dict]] = {}
        for eid, (kb, value) in per_episode.items():
            if eid in previous:
                pair = by_kb.setdefault(kb, ({}, {}))
                pair[0][eid], pair[1][eid] = previous[eid][1], value
        report['retention'] = {kb: retention(b, a, higher_is_better=False)
                               for kb, (b, a) in by_kb.items()}
    report['_per_episode'] = per_episode
    return report


def _pairs(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split('=') for p in text.split(',') if p)}


def load_init_reader(reader: L1Reader, path: Path, full: bool = False) -> list[str]:
    """K2 -> L1a: the key heads (query and item heads, scales) and gate offsets of a
    K2 run's ``reader.pt`` (or a bare state dict); with ``full`` every read-side tensor
    of a matching reader (R, S_s too). Returns the loaded names."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('reader', state)
    prefixes = ('keys.', 'gate_offset.') + (('read_r.', 'operators.', 'stack.recombiner.')
                                            if full else ())
    wanted = {k: v for k, v in state.items() if k.startswith(prefixes)}
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
    if args.l1b_replay == 'free' and (manifest.get('span_batch_size') != 1
                                      or args.l1b_batch != 1):
        # the GPU free run depends on its batch composition (padding): exact replay needs
        # the bank's spans written one at a time and replayed one at a time; a batched bank
        # is off by up to ~30% for some records (invariant 3), so it is refused
        message = ('L1b free replay is not exact for this bank (bank span batch '
                   f'{manifest.get("span_batch_size")}, --l1b-batch {args.l1b_batch}); build '
                   'with --span-batch-size 1, or pass --l1b-allow-inexact')
        if not args.l1b_allow_inexact:
            raise ValueError(message)
        print(json.dumps({'warning': message}), flush=True)
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
    return Producers(writer, origin=ctx.origin, texts=texts, caches=caches,
                     level=manifest['level'], log=log, mode=args.l1b_replay,
                     batch=args.l1b_batch,
                     stored=lambda d, s, i: ctx.kbs[d].read(s, [i])[0].values)


def superpose_config(args, base: dict | None = None):
    """The ``SuperposeConfig`` of this run: a stack run's (``base``), with the flags
    given here overriding it."""
    from schnitz.kb.superpose import SuperposeConfig, parse_fields, parse_pairs
    config = SuperposeConfig.from_dict(base or {})
    field = parse_fields(getattr(args, 'field', None))
    budget = parse_pairs(getattr(args, 'budget', None), float)
    if field:
        config.field = {**config.field, **field}
    if budget:
        config.budget = budget
    for name in ('overlap', 'temperature', 'deep_grad', 'cache_every', 'graph_every',
                 'max_positives'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    if getattr(args, 'unbatched', False):
        config.batched = False
    if getattr(args, 'max_pairs', None):
        config.max_pairs = args.max_pairs
    config.seed = getattr(args, 'seed', config.seed)
    return config


@torch.no_grad()
def rows(args) -> None:
    """``l1 rows``: a rows KB per dataset from the banks (``superpose.build_rows``): the
    storage budget of learnable records, placed by farthest-point sampling over the
    leaves' keys, initialized as the share-weighted mean of their fields (values and
    keys); lineage and shares from the leaves by
    ``kb_store.rewrite``. Written as a banks directory (``banks.json`` naming the leaf
    banks under ``rows``, ``stack.pt`` and ``key_heads_init.pt`` copied), so ``l1 train
    --banks <rows>`` trains the rows in place (the read phase)."""
    from schnitz.kb.superpose import build_rows
    config = superpose_config(args)
    config.depth = 1
    manifest = json.loads((args.banks / 'banks.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, info in manifest['kbs'].items():
        dest = args.output / info['dir']
        if (dest / 'manifest.json').exists():
            report[name] = 'exists'
            continue
        leaves = KnowledgeBase(args.banks / info['dir'])
        try:
            kb, stats = build_rows(leaves, dest, config)
            kb.close()
        finally:
            leaves.close()
        report[name] = {'rows': {s: v['rows'] for s, v in stats.items()},
                        'leaves': {s: v['leaves'] for s, v in stats.items()},
                        'field_fill': {s: v['levels'][0]['field_fill'] for s, v in stats.items()}}
        print(json.dumps({'kb': name, **report[name]}), flush=True)
    for file in ('stack.pt', 'key_heads_init.pt'):
        shutil.copy2(args.banks / file, args.output / file)
    out = dict(manifest, rows={'leaves': str(args.banks), 'config': dataclasses.asdict(config),
                               'kbs': report})
    (args.output / 'banks.json').write_text(json.dumps(out, indent=2) + '\n')


def export_snapshot(ctx: Context, reader: L1Reader, dest: Path, step: int) -> dict:
    """The rows as they are now: every KB exported frozen (the free rows' live values and
    keys, or with views the stack's rows through ``SuperposedKB.export``), plus the
    reader's state, as ``<dest>/step<N>``; ``<dest>/latest.json`` names the newest (the
    write fit's ``--follow-snapshots``)."""
    tag = dest / f'step{step:08d}'
    pending = tag.with_name(tag.name + '.pending')
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True)
    for name, kb in ctx.kbs.items():
        if name in ctx.views:
            ctx.views[name].export(pending / kb.root.name, step=step).close()
        else:
            kb.export_live(pending / kb.root.name).close()
    torch.save({'reader': reader.state_dict(), 'config': dataclasses.asdict(reader.config),
                'step': step}, pending / 'reader.pt')
    (pending / 'targets.json').write_text(json.dumps({'step': step, 'kbs': {
        k: kb.root.name for k, kb in ctx.kbs.items()}}) + '\n')
    shutil.rmtree(tag, ignore_errors=True)
    pending.rename(tag)
    (dest / 'latest.json').write_text(json.dumps({'step': step, 'dir': tag.name}) + '\n')
    return {'snapshot': str(tag)}


def load_write_stack(path: Path, dims: dict, args):
    """``--rows-from-stack``: the write fit's aggregators (``producers.pt`` 'stack') and
    its superposition config (``config.json`` 'superpose'), with this run's overrides."""
    from schnitz.kb.superpose import WriteOps
    run_config = json.loads((path / 'config.json').read_text())
    config = superpose_config(args, run_config.get('superpose'))
    ops = WriteOps(config.depth, dims, config.per_level, temperature=config.temperature)
    state = torch.load(path / 'producers.pt', map_location='cpu', weights_only=False)
    ops.load_state_dict(state['stack'])
    return ops, config, run_config, state


ITEM_LR = 3e-3          # live items of a leaf bank (L1a, K2) and leaves under the stack
ROWS_ITEM_LR = 1e-2     # free rows in the read phase (smoke: 3e-2 overfits after 100 steps)


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
                        max_reps=args.max_reps, checkpointing=not args.no_operator_checkpoint,
                        read_combine=args.read_combine)
    banks_manifest = json.loads((args.banks / 'banks.json').read_text())
    if args.item_lr is None:     # the read phase moves free rows faster than leaf items
        args.item_lr = ROWS_ITEM_LR if 'rows' in banks_manifest and not args.rows_from_stack \
            else ITEM_LR
    config.learned_keys = args.keys == 'learned' or (args.keys == 'auto'
                                                     and 'rows' in banks_manifest)
    # learned keys are unit keys; with --rows-from-stack the live keys are the leaves' key
    # corrections (unconstrained vectors)
    key_optimizer = KeyOptimizer(args.key_lr, normalize=not args.rows_from_stack) \
        if config.learned_keys or args.rows_from_stack else None
    torch.manual_seed(args.seed)
    reader = L1Reader(config, stack)
    reader.keys.load_state_dict(torch.load(args.banks / 'key_heads_init.pt', map_location='cpu'))
    reader.to(model.device)
    # --rows-from-stack: reads see the write stack's rows over the leaves (the leaf banks
    # the rows banks were built from), anchored at the fitted rows
    write_ops = superpose = None
    leaf_banks = args.banks
    if args.rows_from_stack:
        if 'rows' not in banks_manifest:
            raise ValueError('--rows-from-stack needs --banks to be a rows banks dir (l1 rows)')
        leaf_banks = Path(banks_manifest['rows']['leaves'])
        write_ops, superpose, stack_run, fit_state = load_write_stack(args.rows_from_stack,
                                                                      dims, args)
        write_ops.to(model.device)
    sets = parameter_sets(reader, model, write_ops)
    groups = optimizer_groups(sets, args)
    for p in sets['writer'] + sets['codecs'] + sets['write']:   # AdamW takes them
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    set_phase(sets, L1A_SET)
    usage: dict = {}
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'reader.pt'
    kbs = open_live(leaf_banks, args.output, args.live_device, args.sync_every)
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
        if len(state['optimizer']['param_groups']) != len(groups):
            raise ValueError(f'{state_path} has {len(state["optimizer"]["param_groups"])} '
                             'optimizer groups (before the L1b codec/writer rates); '
                             'restart from the banks or from --init-reader')
        optimizer.load_state_dict(state['optimizer'])
        if 'writer' in state:
            model.writer.load_state_dict(state['writer'])
        if write_ops is not None:
            write_ops.load_state_dict(state['write_ops'])
        if key_optimizer is not None:
            key_optimizer.load_state_dict(state.get('key_optimizer', {}))
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
    # before the write fit's item-key heads are installed (they take precedence)
    if args.init_reader and step == 0:
        loaded = load_init_reader(reader, args.init_reader, full=args.init_reader_full)
        for kb in kbs.values():      # the search keys from the loaded item-key heads
            reader.rekey(kb)
        print(json.dumps({'init_reader': str(args.init_reader), 'loaded': len(loaded)}),
              flush=True)
    views = {}
    if write_ops is not None:
        from schnitz.kb.superpose import SuperposedKB, head_leaf_key, rows_of
        # leaf keys are the item-key heads on the content plus a per-leaf correction held
        # in the leaf's live key; both start where the write fit left them
        if not state_path.exists():
            heads = {k[len('item.'):]: v for k, v in fit_state.get('heads', {}).items()
                     if k.startswith('item.')}
            if heads:
                reader.keys.item.load_state_dict(heads)
            ids_of = fit_state.get('correction_ids', {})
            for name, kb in kbs.items():
                for space in kb.spaces:
                    ids = ids_of.get(name, {}).get(space)
                    corr = fit_state.get('corrections', {}).get(f'{name}/{space}')
                    current = current_ids(kb, space)
                    if ids is None or corr is None:
                        ids, corr = current, torch.zeros(len(current),
                                                          kb.spaces[space].key_width)
                    if ids:
                        kb.set_live_keys(space, list(ids), corr.float())
        targets_root = Path(stack_run.get('targets_dir') or args.rows_from_stack / 'targets')
        for name, kb in kbs.items():
            anchor_root = targets_root / kb.root.name
            anchor_kb = KnowledgeBase(anchor_root if (anchor_root / 'manifest.json').exists()
                                      else args.banks / kb.root.name)
            views[name] = SuperposedKB(kb, write_ops, superpose, rows_of(anchor_kb),
                                       leaf_key=head_leaf_key(reader.keys),
                                       device=model.device, autocast=model.core.autocast)
            anchor_kb.close()
    ctx = Context(frozen, reader, kbs, views=views)
    ctx.key_optimizer = key_optimizer
    tok = model.tok
    writer = Writer(model, stack, frozen.embed, batch=args.write_batch) \
        if args.writes or 'l1b' in phases else None
    producers = load_producers(args, ctx, writer, log, model) if 'l1b' in phases else None
    args.write_level_index = LEVELS.index(args.write_level)

    # level-1 field sizes: the nominal size, or with a range (``--field A=4:16``) one drawn
    # log-uniformly every ``--refield-every`` steps from (seed, step), so resume is exact
    nominal = {s: superpose.field_size(s) for s in SPACES} if superpose else {}
    ranged = bool(views) and any(lo < hi for lo, hi in
                                 (superpose.field_range(s) for s in SPACES))
    fields_now = dict(nominal)

    def rebuild(fields: dict | None = None) -> dict:
        """Fields from the leaves' and rows' current keys (``--rows-from-stack``)."""
        chosen = fields or fields_now
        return {name: view.rebuild(step, fields={s: f for s, f in chosen.items()
                                                 if s in view.rows})
                for name, view in views.items()}

    def refield() -> None:
        nonlocal fields_now
        draw = random.Random(f'{args.seed}:field:{step}')
        fields_now = {s: superpose.sample_field(s, draw) for s in SPACES}
        for view in views.values():
            view.refield({s: f for s, f in fields_now.items() if s in view.graphs}, step)
            view.rekey()

    budgets = rebuild() if views else None

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
    storage = {name: {s: {'items': st['current'], 'positions': st['positions']}
                      for s, st in kb.stats().items()} for name, kb in kbs.items()}
    (args.output / 'config.json').write_text(json.dumps(dict(
        vars(args), read_config=dataclasses.asdict(config),
        superpose=None if superpose is None else dataclasses.asdict(superpose),
        rows_budget=budgets, storage=storage, rows=banks_manifest.get('rows'),
        read_budget={'candidates': candidates, 'keep': keep, 'max_reps': args.max_reps},
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
                    'config': dataclasses.asdict(config),
                    **({'key_optimizer': key_optimizer.state_dict()}
                       if key_optimizer is not None else {}),
                    **({'write_ops': write_ops.state_dict(),
                        'superpose': dataclasses.asdict(superpose)} if write_ops is not None
                       else {})}, pending)
        pending.replace(state_path)
        for kb in kbs.values():
            for old in kb.live_checkpoints():
                if old != tag:
                    kb.drop_live_checkpoint(old)

    anchor = None
    if args.read_anchor > 0:
        per_kb: dict[str, list[Episode]] = {}
        for ep in eval_eps:
            if len(per_kb.setdefault(ep.kb, [])) < args.anchor_probes:
                per_kb[ep.kb].append(ep)
        anchor = ReadAnchor(ctx, [ep for eps in per_kb.values() for ep in eps], texts,
                            model.text_ids, args.anchor_tokens, args.anchor_batch)
        anchor.refresh(step)
    previous: dict | None = None

    def run_eval() -> dict:
        nonlocal previous
        report = evaluate(ctx, eval_eps, texts, tok, previous)
        previous = report.pop('_per_episode')
        if anchor is not None:
            report['read_anchor'] = anchor.drift()
        return report

    if step == 0 and args.eval_every:
        log_record({'step': 0, 'eval': run_eval()})
    order: list[int] = []
    window: dict[str, list] = {}
    started = time.time()

    def rekey() -> None:     # the search's key cache from the current item-key heads
        for kb in kbs.values():
            reader.rekey(kb)
        for view in views.values():
            view.rekey()
    while step < args.steps:
        batch = []
        while len(batch) < args.batch_size:
            if not order:
                order = list(range(len(train_rows)))
                rng.shuffle(order)
            ep = episode(train_rows[order.pop()])
            if ep is not None:
                batch.append(ep)
        phase = phase_at(schedule, step, args.read_warmup)
        names = phase_set(phase, args, l1b_set)
        if args.consolidate_every and views and phase in ('l1a', 'w') \
                and (step + 1) % args.consolidate_every == 0:
            names = tuple(dict.fromkeys(names + WRITE_SET))
        trainable = set_phase(sets, names)
        reader.train()
        result = train_step(ctx, batch, optimizer, args, step, usage, phase=phase,
                            trainable=trainable, writer=writer if args.writes else None,
                            producers=producers, log=log, rng=rng, text_ids=model.text_ids,
                            anchor=anchor)
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
        if views and superpose.graph_every and step % superpose.graph_every == 0:
            rebuild()
        elif ranged and args.refield_every and step % args.refield_every == 0:
            refield()
        if ranged:
            for s, f in fields_now.items():
                window.setdefault(f'field_{s}', []).append(f)
        if anchor is not None and args.anchor_every and step % args.anchor_every == 0:
            anchor.refresh(step)
        if step % args.log_every == 0:
            log_record({'step': step, 'phase': phase, **_mean(window), 'skipped': dict(skipped),
                        'usage': {k: u.stats() for k, u in usage.items()} if args.log_usage
                        else None, 'elapsed_s': round(time.time() - started)})
            window = {}
        if args.export_rows_every and step % args.export_rows_every == 0:
            log_record({'step': step, **export_snapshot(ctx, reader, args.output / 'snapshots',
                                                        step)})
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            rekey()
            if views:
                rebuild(nominal)       # evaluations at the nominal field size
            save()
            reader.eval()
            log_record({'step': step, 'eval': run_eval()})


def add_args(parser: argparse.ArgumentParser) -> None:
    """Arguments of all actions (``build``, ``rows``, ``train``) on one parser."""
    parser.add_argument('action', choices=('build', 'rows', 'train'))
    parser.add_argument('--transcripts', type=Path, nargs='+',
                        help='memory transcript directories (v1 or v2; build and train)')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--reader-state', type=Path, help='B3 writer.pt (merged decoder); later B4')
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
                            help='records per writer free run (default --batch-size, 32). '
                                 'L1B NEEDS 1: on the GPU the free run depends on its batch '
                                 'composition (padding), and L1b replays each bank item one at '
                                 'a time, so only a bank written with --span-batch-size 1 is '
                                 'replayed exactly (batch-16 spans differ from batch-1 spans by '
                                 '1.1%% on average, up to 29%%). Batch 1 writes about 2.1 '
                                 'records/s against 5-7 at 16 on the shared GPU (28 Sep, 96 '
                                 'records), so the default stays batched for bank-scale '
                                 'builds that no L1b phase replays')
    build_args.add_argument('--distractors', type=int, default=0,
                            help='extra records per KB beyond those the transcripts name')
    sup = parser.add_argument_group('superposed KB (rows; --rows-from-stack)')
    sup.add_argument('--field', help='field size per space for the row count, e.g. '
                                     'A=8,B=16,C=16,D=16 (rows = ceil(overlap / field x leaves))')
    sup.add_argument('--overlap', type=int, help='fields (rows) each input joins (default 3)')
    sup.add_argument('--budget', help='rows as a fraction of the leaves (overrides --field), '
                                      'e.g. 0.25 or A=0.5,D=0.2')
    sup.add_argument('--temperature', type=float, help='share softmax temperature (0.1)')
    sup.add_argument('--deep-grad', type=float,
                     help='--rows-from-stack: share of a row\'s inputs recomputed one level '
                          'down with gradients per step (0.25)')
    sup.add_argument('--cache-every', type=int,
                     help='--rows-from-stack: level-(L-1) cache lifetime in steps (10)')
    sup.add_argument('--graph-every', type=int,
                     help='--rows-from-stack: field reassignment period in steps (100; also at '
                          'every checkpoint)')
    sup.add_argument('--max-positives', type=int,
                     help='--rows-from-stack: covering rows kept as retrieval positives (8)')
    sup.add_argument('--refield-every', type=int, default=10,
                     help='--rows-from-stack with a --field range: draw new level-1 field '
                     'sizes every N steps (10)')
    sup.add_argument('--write-balance', type=float, default=0.01,
                     help='--rows-from-stack: balance loss on the rows\' write loads (0.01)')
    sup.add_argument('--unbatched', action='store_true',
                     help='aggregators row by row (the reference path; default: all rows of a '
                     'level in one pass)')
    sup.add_argument('--max-pairs', type=int,
                     help='(output, input) position pairs per batched aggregator pass (262144)')
    t = parser.add_argument_group('train')
    t.add_argument('--banks', type=Path, help='output of build (or of rows)')
    t.add_argument('--steps', type=int, default=20000)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--item-lr', type=float, default=None,
                   help='per-item Adam rate of the live items (3e-3; on a rows banks dir, the '
                   'read phase, 1e-2: rows drift about 10%% from their init in 300 steps)')
    t.add_argument('--retrieval-weight', type=float, default=0.5)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--candidates', default='', help='scored per space, e.g. A=8,B=16,C=32,D=64')
    t.add_argument('--keep', default='', help='read per space (nonzero gates), e.g. A=2,D=4')
    t.add_argument('--gate-offset', type=float, default=0.5,
                   help='initial gate offset b_s in cosine units')
    t.add_argument('--keys', choices=('auto', 'learned', 'derived'), default='auto',
                   help='learned: every item\'s live key is a free parameter moved by the '
                        'retrieval and gate gradients (scores and search use it); derived: '
                        'keys are the item-key heads of the values (refreshed by --rekey-every); '
                        'auto: learned on rows banks (l1 rows), else derived')
    t.add_argument('--key-lr', type=float, default=1e-3,
                   help='learned keys: per-key Adam learning rate (keys renormalized)')
    t.add_argument('--read-combine', choices=('r', 's_s'), default='r',
                   help='r: R reads every space\'s retrieved items directly, conditioned on the '
                        'query keys (owner, 28 Sep); s_s: per-space S_s then R (the earlier '
                        'smokes\' path, kept as an ablation)')
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
    t.add_argument('--init-reader-full', action='store_true',
                   help='with --init-reader: the whole read side (key heads, gate offsets, R, '
                        'S_s), e.g. the read phase\'s reader for --rows-from-stack')
    t.add_argument('--phase', choices=('l1a', 'l1b', 'r', 'w'), default='l1a',
                   help='the phase when there is no --phase-schedule')
    t.add_argument('--phase-schedule',
                   help='alternating phases, cycled, e.g. a:2000,b:500 (L1a then L1b steps); '
                        'r (read side: R and heads) and w (aggregators and leaves/rows) '
                        'alternate the two sides, e.g. r:100,w:100 (off by default)')
    t.add_argument('--read-warmup', type=int, default=0,
                   help='first N steps read side only (R and heads; rows, leaves and '
                        'aggregators frozen)')
    t.add_argument('--rows-from-stack', type=Path,
                   help='an L2 stack run (--producer stack): reads see its rows, recomputed '
                        'from the leaf banks of the rows banks (--banks) by its aggregators, '
                        'anchored at its targets; leaves, aggregators, R and heads train '
                        'jointly (not the default schedule)')
    t.add_argument('--write-lr', type=float,
                   help='--rows-from-stack: the aggregators\' learning rate (default --lr)')
    t.add_argument('--consolidate-every', type=int, default=0,
                   help='--rows-from-stack: aggregators step only every N steps, after which '
                        'the leaves are re-fitted so the touched rows return to their values '
                        '(item-preserving; 0: off, the aggregators train every step)')
    t.add_argument('--consolidate-steps', type=int, default=3,
                   help='re-fit steps of a consolidation')
    t.add_argument('--read-anchor', type=float, default=0.0,
                   help='weight of the KL of reads of held-out probe episodes of every KB '
                        'against their reads at the last refresh (key/query heads and R; 0: off)')
    t.add_argument('--anchor-every', type=int, default=200, help='probe refresh period')
    t.add_argument('--anchor-probes', type=int, default=2, help='probe episodes per KB')
    t.add_argument('--anchor-batch', type=int, default=4, help='probes per step')
    t.add_argument('--anchor-tokens', type=int, default=32,
                   help='record text tokens the probe KL is measured on')
    t.add_argument('--export-rows-every', type=int, default=0,
                   help='export the rows (frozen KBs and the reader) to <output>/snapshots '
                        'every N steps for a write fit that follows them (0: off)')
    t.add_argument('--l1b-train', default='writer,codecs,keys,operators,recombiner',
                   help=f'parameter sets trained in L1b, from {",".join(PHASE_SETS)}')
    t.add_argument('--l1b-replay', choices=('free', 'teacher', 'self'), default='free',
                   help='L1b producer replay (schnitz.kb.producer): the free run itself (the '
                        'stored forward), teacher-fed on the stored span (one pass), or one '
                        'pass fed with the writer\'s own free run')
    t.add_argument('--l1b-allow-inexact', action='store_true',
                   help='run L1b free replay on a bank whose spans were written in batches '
                        '(not exact; refused otherwise)')
    t.add_argument('--l1b-batch', type=int, default=1,
                   help='bank sources per producer pass (1: the replay is exact when the bank\'s '
                        'spans were written one at a time, build --span-batch-size 1)')
    t.add_argument('--l1b-codec-lr', type=float, default=3e-5,
                   help='L1b: the codecs\' learning rate (their own AdamW group)')
    t.add_argument('--l1b-writer-lr', type=float, default=3e-6,
                   help='L1b: the writer span heads\' learning rate (their own AdamW group)')
    t.add_argument('--l1b-change-units', type=int, default=4,
                   help='L1b: replay units recomputed after each optimizer step to log the '
                        'relative change of their items (l1b_change_rel; 0: all, -1: off)')
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
    if args.action == 'rows':
        if args.banks is None:
            raise SystemExit('rows needs --banks')
        rows(args)
        return
    for name in ('transcripts', 'checkpoint', 'reader_state'):
        if getattr(args, name) is None:
            raise SystemExit(f'{args.action} needs --{name.replace("_", "-")}')
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
