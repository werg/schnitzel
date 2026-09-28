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
heads, router, S_s and R; the decoder is frozen. An auxiliary retrieval loss
(balanced BCE on gate logits: the slot's target items up, other candidates down)
supervises routing; recall@k per space is logged.

Evaluation arms on validation transcripts (invariant 9): ``noctx`` (empty memory
spans), ``text`` (the slot records' text as the tool result: the information-
matched text control), ``retrieved`` (the L1 read), ``shuffled`` (another episode's
reads), ``gold`` (the target items, gate 1, no retrieval) and ``gold_shuffled``.
Reported with ``kb_eval.nll_summary`` (captured fractions of the text arm's gain,
content nats over the shuffled controls), recall and effective items per read.
Writes: ``memory_write`` calls (and their acknowledgements) are left out of the L1
render (``--keep-writes`` keeps them): their text argument is the dropped v0.5 form,
writes are single-pass latent spans trained in B4 (owner, 28 September), and L1 does
not train writes.

Entry point (until ``scripts/train.py <stage>`` exists): ``python -m
schnitz.kb.stages.l1 build|train ...``. Training-only; the decoder parts run in
``sdkb-bgkit``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from schnitz.kb.read import (DEFAULT_CANDIDATES, DEFAULT_KEEP, ItemCache, L1Reader,
                             ReadConfig, source_index, splice)
from schnitz.kb.stack import KeyHeads
from schnitz.kb_eval import distribution, effective_count, nll_summary
from schnitz.kb_store import DEFAULT_SPACES, KnowledgeBase, NewItem, Provenance
from schnitz.span_tokens import MEMORY_TOOLS, SPAN_TOKENS

# the writer's length schedule (s0 factor) still lives in scripts/
SCRIPTS = Path(__file__).resolve().parents[4] / 'scripts'


def _scripts() -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))

MEM, MEM_END = SPAN_TOKENS['mem'][0], SPAN_TOKENS['mem_end'][0]
MEM_ID = SPAN_TOKENS['mem'][1]
MEMORY_NAMES = {t['name'] for t in MEMORY_TOOLS}
CALL = re.compile(r'memory_search\(\s*\)')


# -- transcripts ---------------------------------------------------------------------
class Transcripts:
    """The first ``limit`` transcripts of ``split`` per directory, in file order, read
    lazily by byte offset (the full corpora do not fit in memory as parsed rows)."""

    def __init__(self, dirs: list[Path], split: str, limit: int | None):
        self.entries: list[tuple[Path, int, str]] = []
        for d in dirs:
            path = d / f'transcripts-{split}.jsonl'
            if not path.exists():
                continue
            with path.open('rb') as handle:
                n = 0
                while limit is None or n < limit:
                    offset = handle.tell()
                    if not handle.readline():
                        break
                    self.entries.append((path, offset, str(d)))
                    n += 1

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        path, offset, d = self.entries[i]
        with path.open('rb') as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        row['_dir'] = d
        return row

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_rows(dirs: list[Path], split: str, limit: int | None) -> list[dict]:
    return list(Transcripts(dirs, split, limit))


def slots_of(row: dict) -> list[dict]:
    return [m['content']['slot'] for m in row['messages']
            if isinstance(m.get('content'), dict) and 'slot' in m['content']]


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


def _render(tok, messages, tools):
    out = tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=True,
                                  return_assistant_tokens_mask=True)
    return list(out['input_ids']), list(out['assistant_masks'])


def layout(row: dict, tok, texts: dict[str, str] | None = None,
           keep_writes: bool = False) -> Episode:
    """Token layout of a transcript. The query position of a call is the token holding
    its closing parenthesis (so calls in one block have their own positions)."""
    messages, tools = chat(row, texts, keep_writes)
    ids, mask = _render(tok, messages, tools)
    slots = slots_of(row)
    calls, mems = [], []
    if texts is None:
        text = tok.apply_chat_template(messages, tools=tools, tokenize=False)
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
    targets = torch.tensor([t for t in range(1, len(ids)) if mask[t]], dtype=torch.long)
    return Episode(row['episode_id'], row['kb'], torch.tensor(ids), targets, calls, mems,
                   slots, query_time(row), row)


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
class Context:
    """What a pass needs: the frozen decoder, the reader, the KBs and their item index."""

    def __init__(self, frozen: Frozen, reader: L1Reader, kbs: dict[str, KnowledgeBase],
                 autocast=None):
        self.frozen, self.reader, self.kbs = frozen, reader, kbs
        self.autocast = autocast or frozen.autocast
        self.index = {name: {s: source_index(kb, s) for s in kb.spaces} for name, kb in kbs.items()}
        self._rows: dict[tuple[str, str], dict[str, int]] = {}

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
                spans: list[torch.Tensor] | None = None, retrieval_only: bool = False):
    """Task NLL (summed over target tokens) of one transcript and its reads.

    ``mode`` 'retrieve' or 'gold' computes each read at its call from the exact causal
    prefix; 'fixed' splices the given ``spans`` (controls). ``retrieval_only`` (K2):
    the reads' spans enter later prefixes detached and no task pass runs (NLL None),
    so only the retrieval loss trains, through the queries and the item keys."""
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
                                           gold=mode == 'gold')
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
def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '__', name)


def needed_records(rows: list[dict]) -> dict[str, dict[str, str]]:
    """kb -> record id -> transcript dir (whose manifest names the corpus)."""
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for slot in slots_of(row):
            if slot['kb'] != row['kb']:
                raise PermissionError(f'{row["episode_id"]}: slot of another KB')
            for r in slot['record_ids']:
                out.setdefault(row['kb'], {})[r] = row['_dir']
    return out


def read_sources(dirs: set[str], wanted: set[str],
                 extra: dict[str, int] | None = None) -> dict[str, dict]:
    """Text and time of the ``wanted`` records. ``extra`` asks for up to that many more
    records of a KB (in corpus order; distractors, so a KB holds more than the slots
    name); every record is returned with its ``kb``."""
    records, extra, added = {}, dict(extra or {}), {}
    for d in sorted(dirs):
        manifest = json.loads((Path(d) / 'manifest.json').read_text())
        corpus = Path(manifest['input'])
        with (corpus / 'sources.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                rec = json.loads(line)
                kb = (f'{manifest["kb"]}:{rec.get("domain", "")}'
                      if manifest.get('kb_per_domain') else manifest['kb'])
                take = rec['record_id'] in wanted
                if not take and added.get(kb, 0) < extra.get(kb, 0):
                    take = True
                    added[kb] = added.get(kb, 0) + 1
                if take:
                    records[rec['record_id']] = {'text': rec['text'], 'kb': kb,
                                                 'created_at': int(rec['created_at'])}
    return records


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


@torch.no_grad()
def build(args) -> None:
    _scripts()
    from cache_bgkit_teacher import length_factors
    rows = []
    for split, limit in (('train', args.limit), ('validation', args.eval_limit)):
        rows += load_rows(args.transcripts, split, limit)
    needed = needed_records(rows)
    wanted = {r for recs in needed.values() for r in recs}
    records = read_sources({d for recs in needed.values() for d in recs.values()}, wanted,
                           {kb: args.distractors for kb in needed})
    missing = wanted - set(records)
    if missing:
        raise ValueError(f'{len(missing)} slot records are not in the corpus sources')
    for rid, rec in records.items():      # distractor records join their own KB
        if rid not in wanted and rec['kb'] in needed:
            needed[rec['kb']][rid] = ''
    model = load_model(args)
    args.output.mkdir(parents=True, exist_ok=True)
    stack_path = args.output / 'stack.pt'
    resumed = stack_path.exists()
    stack, codec_step, dims = load_stack(stack_path if resumed else args.codecs,
                                         model.target_norm, model.device, args.seed)
    # a random-init stack has no span statistics yet: taken from the first batch
    needs_statistics = not resumed and args.codecs is None
    hidden = model.decoder.base_lm.config.hidden_size
    heads = initial_heads(args.output / 'key_heads_init.pt', hidden, args.key_hidden,
                          args.seed).to(model.device)
    teacher = None
    if args.span_source == 'teacher':
        from schnitz.kb.decoder import TeacherCache
        teacher = TeacherCache(args.cache, None, texts=[])
        teacher.by_id = {item[2]: item for item in teacher.items}
    report = {}
    started = time.time()
    for kb_name, recs in sorted(needed.items()):
        root = args.output / _safe(kb_name)
        kb = KnowledgeBase(root, writable=True) if (root / 'manifest.json').exists() else \
            KnowledgeBase.create(root, name=_safe(kb_name), dataset=kb_name,
                                 origin={'command': 'train.py l1 build',
                                         'span_source': args.span_source,
                                         'codecs': str(args.codecs), 'codec_step': codec_step,
                                         'reader_state': str(args.reader_state),
                                         'query_layer': args.query_layer})
        done = set(source_index(kb, 'D'))
        todo = sorted((r for r in recs if r not in done),
                      key=lambda r: len(records[r]['text']))
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            ids = [model.text_ids(records[r]['text']) for r in batch]
            if teacher is not None:
                spans = []
                for r in batch:
                    if r not in teacher.by_id:
                        raise ValueError(f'record {r} has no cached teacher span')
                    shard, row = teacher.by_id[r][:2]
                    spans.append(teacher.reps(shard, row, 's0').to(model.device).float())
            else:
                factors = [length_factors(int(x.shape[0]))[0] for x in ids]
                examples = [{'ids': x, 'prompt': 'memory', 'factor': f}
                            for x, f in zip(ids, factors)]
                lengths = [max(1, math.ceil(x.shape[0] / f)) for x, f in zip(ids, factors)]
                with model.core.autocast():
                    spans, _ = model.free_run(examples, lengths)
                spans = [s.float() for s in spans]
            if needs_statistics:
                stack.set_statistics(spans)
                needs_statistics = False
            if not stack_path.exists():
                torch.save({'stack': stack.state_dict(), 'step': codec_step, 'dims': dims,
                            'source': str(args.codecs)}, stack_path)
            per_space = {s: [] for s in DEFAULT_SPACES}
            with model.core.autocast():
                encoded = [stack.encode(span) for span in spans]
            for i, r in enumerate(batch):
                for s in DEFAULT_SPACES:
                    values = encoded[i][s].float()
                    per_space[s].append(NewItem(
                        values.cpu(), heads.item_key(s, values).float().cpu(),
                        Provenance((r,), 'codec', codec_step), 1.0, records[r]['created_at']))
            for s, items in per_space.items():
                kb.append(s, items)
            print(json.dumps({'kb': kb_name, 'done': start + len(batch), 'of': len(todo),
                              'elapsed_s': round(time.time() - started)}), flush=True)
        report[kb_name] = {'dir': root.name, 'records': len(recs), 'stats': kb.stats()}
        kb.close()
    manifest = {'command': 'build', 'transcripts': [str(d) for d in args.transcripts],
                'limit': args.limit, 'eval_limit': args.eval_limit,
                'span_source': args.span_source, 'codecs': str(args.codecs),
                'distractors': args.distractors, 'codec_step': codec_step, 'reader_state': str(args.reader_state),
                'query_layer': args.query_layer, 'seed': args.seed, 'kbs': report}
    (args.output / 'banks.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'built': {k: v['records'] for k, v in report.items()}}), flush=True)


# -- train --------------------------------------------------------------------------
def open_live(banks: Path, output: Path) -> dict[str, KnowledgeBase]:
    """The live copies of the banks under ``output/kbs`` (copied on first start)."""
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
        kbs[name] = kb
    return kbs


def _read_stats(reads, sink: dict) -> None:
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
    usage averages with the read mass."""
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
        size = len(ctx.rows(dataset, s))
        if key not in usage:
            usage[key] = UsageEMA(size)
        usage[key].grow(size)
        ids_t, gates_t = torch.tensor(ids), torch.stack(gates)
        losses.append(balance_loss(ids_t, gates_t, usage[key]))
        usage[key].update(ids_t, gates_t.detach().cpu())
    return torch.stack(losses).mean() if losses else None


def train_step(ctx: Context, episodes: list[Episode], optimizer, args, step: int = 0,
               usage: dict | None = None) -> dict:
    """One L1a step (items in place; K2 with ``--retrieval-only``): gradients of all
    episodes accumulate, then one optimizer step for the reader and one sparse live
    update per touched (KB, space) for the item values."""
    if args.phase != 'l1a':
        # L1b (through the sources): recompute each retrieved item's write from its
        # stored source with gradients (selective producer replay, the serialized
        # forward exactly: invariant 3) so the task loss reaches the writer's span heads
        # and the codecs. Not built; this is where it plugs in.
        raise NotImplementedError('L1b (gradients through the sources) is not built')
    usage = {} if usage is None else usage
    cache = ItemCache(ctx.frozen.device, train=True)
    optimizer.zero_grad(set_to_none=True)
    tokens = sum(int(ep.targets.numel()) for ep in episodes)
    weight = retrieval_weight(args, step)
    stats: dict[str, list] = {}
    nll_total, aux_total, balance_total = 0.0, 0.0, 0.0
    for ep in episodes:
        nll, _, reads, _ = run_episode(ctx, ep, cache, 'retrieve',
                                       retrieval_only=args.retrieval_only)
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
        _read_stats(reads, stats)
    torch.nn.utils.clip_grad_norm_(ctx.reader.trainable(), args.clip)
    optimizer.step()
    counts = cache.apply(0.0 if args.retrieval_only else args.item_lr)
    out = {'aux': aux_total, 'retrieval_weight': weight, 'tokens': tokens, **counts,
           **_mean(stats)}
    if args.balance_weight and not args.retrieval_only:
        out['balance'] = balance_total
    if not args.retrieval_only:
        out['nll'] = nll_total / tokens
    return out


@torch.no_grad()
def evaluate(ctx: Context, episodes: list[Episode], texts: dict[str, str], tok,
             keep_writes: bool = False) -> dict:
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
            _read_stats(reads, stats if name == 'retrieved' else gold_stats)
        tokens += n
        empty = [torch.zeros(0, ctx.reader.config.span_width) for _ in ep.mems]
        sums['noctx'] += run_episode(ctx, ep, cache, 'fixed', empty)[0].item()
        text_ep = layout(ep.row, tok, texts, keep_writes)
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
    return report


def _pairs(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split('=') for p in text.split(',') if p)}


def train(args) -> None:
    model = load_model(args)
    lm = model.decoder.base_lm
    if args.decoder_checkpoint:
        from schnitz.bgkit_span import checkpoint_layers
        checkpoint_layers(lm.model.layers)
    frozen = Frozen(lm, args.query_layer, model.core.autocast)
    candidates = dict(DEFAULT_CANDIDATES, **_pairs(args.candidates))
    keep = dict(DEFAULT_KEEP, **_pairs(args.keep))
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
    optimizer = torch.optim.AdamW(reader.trainable(), lr=args.lr, weight_decay=0.01)
    usage: dict = {}
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'reader.pt'
    kbs = open_live(args.banks, args.output)
    if not state_path.exists():
        for kb in kbs.values():         # a crash before the first save restarts from the banks
            if kb.live_updates:
                raise ValueError(f'{kb.root} has live updates but no reader checkpoint; '
                                 'remove the output directory to restart')
    step = 0
    rng = random.Random(args.seed)
    if state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        reader.load_state_dict(state['reader'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        from schnitz.kb.losses import UsageEMA
        for key, (share, touched) in state.get('usage', {}).items():
            usage[key] = UsageEMA(len(share))
            usage[key].share, usage[key].touched = share.cpu(), touched.cpu()
        # items and keys back to exactly the state paired with the reader checkpoint
        for name, kb in kbs.items():
            kb.restore_live(state['live_tag'])
            if kb.live_updates != state['live_updates'][name]:
                raise ValueError(f'{name}: restored live state does not match the checkpoint')
    ctx = Context(frozen, reader, kbs)
    tok = model.tok

    skipped: dict[str, int] = {}

    def episode(row) -> Episode | None:
        """The layout of a usable transcript, else None (counted by reason)."""
        reason = None
        if not slots_of(row) or row['kb'] not in kbs:
            reason = 'no_reads_or_kb'
        else:
            ep = layout(row, tok, keep_writes=args.keep_writes)
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

    def log(record):
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
        log({'step': 0, 'eval': evaluate(ctx, eval_eps, texts, tok, args.keep_writes)})
    order: list[int] = []
    window: dict[str, list[float]] = {}
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
        reader.train()
        result = train_step(ctx, batch, optimizer, args, step, usage)
        step += 1
        for key, value in result.items():
            window.setdefault(key, []).append(value)
        if args.rekey_every and step % args.rekey_every == 0:
            rekey()
        if step % args.log_every == 0:
            log({'step': step, **_mean(window), 'skipped': dict(skipped),
                 'usage': {k: u.stats() for k, u in usage.items()} if args.log_usage else None,
                 'elapsed_s': round(time.time() - started)})
            window = {}
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            rekey()
            save()
            reader.eval()
            log({'step': step, 'eval': evaluate(ctx, eval_eps, texts, tok, args.keep_writes)})


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
    build_args.add_argument('--cache', type=Path, help='B1 teacher cache (span source teacher)')
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
    t.add_argument('--phase', choices=('l1a', 'l1b'), default='l1a',
                   help='L1a items in place; L1b (through the sources) is not built')
    t.add_argument('--max-reps', type=int, default=16, help='span budget per read')
    t.add_argument('--max-tokens', type=int, default=3072)
    t.add_argument('--keep-writes', action='store_true')
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
