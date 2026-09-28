"""B9 building blocks: learning by experience over rounds (restart plan B9;
docs/knowledge-base-stack.md, 5.1 step 9). The stage is ``schnitz.kb.stages.b9``.

- ``Registry``: the B9 records of one dataset KB and how each was produced: ``model``
  (the model's own write, nothing gold in its lineage), ``hinted`` (its round read a
  record that carries gold for some task, or it supersedes one that did) and ``gold``
  (the teacher trajectory written by the bank writer, read at a receding weight). A
  task has one current own record: each round's write supersedes the previous one
  (same item ids, new version; the old version stays in the store for measurement,
  never searched). ``saw_gold`` lists the tasks whose gold entered a record's
  lineage, so held-out tasks can be restricted to records that never saw gold for
  them.
- ``KBView``: a KB as one round sees it, for the read path (``schnitz.kb.read``):
  hidden items are never searched (masked in the store's exact scan, so the top k are
  over the visible items) or read, and
  substituted items return another item's values under their own key (the swapped
  control). Authorization stays with the KB's dataset (the reader's ``allowed``).
- The gold record's receding weight w is the read path's per-item gate multiplier
  (``L1Reader.read(..., weights=)``, ``l1.run_episode(..., weights=)``). Gates only
  scale mass in the MLP-matrix operator, so w is exactly that item's share of the read
  mass (w = 0 removes it exactly).
- ``GenProtocol`` / ``generate``: an attempt generated autoregressively (KV cache) by
  the frozen decoder, executing a ``memory_search()`` read whenever the model emits
  the call: the query is the query-layer state at the call's closing parenthesis from
  a pass over the exact prefix (earlier read spans spliced in), the read's span goes
  into the tool message between ``<|mem|>`` and ``<|/mem|>``, and generation
  continues in a new assistant turn.
- ``GoldSchedule``: w = global schedule x per-task factor (``decay`` per supersede of
  the task's own record).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass, field
import re

import torch
from torch import Tensor

from schnitz.kb_store import KnowledgeBase, SearchHits

KINDS = ('model', 'hinted', 'gold')


# -- records -----------------------------------------------------------------------------
@dataclass
class B9Record:
    record_id: str
    task: str
    kind: str                       # one of KINDS
    split: str                      # 'train' | 'eval'
    round: int
    step: int
    items: dict[str, str]           # space -> item id
    saw_gold: tuple[str, ...] = ()  # tasks whose gold entered this record's lineage
    generation: int = 0             # evaluation generation (eval records)
    correct: bool | None = None
    current: bool = True


class Registry:
    """B9 records of one dataset KB (see the module docstring)."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.records: dict[str, B9Record] = {}
        self.own: dict[str, str] = {}          # task -> current own record id
        self.gold: dict[str, str] = {}         # task -> gold record id
        self.supersedes: dict[str, int] = {}   # task -> own records superseded so far
        self.generation = 0                    # current evaluation generation
        self._by_item: dict[str, str] = {}     # current item id -> record id

    # -- bookkeeping --------------------------------------------------------------------
    def record_of(self, item_id: str) -> B9Record | None:
        r = self._by_item.get(item_id)
        return None if r is None else self.records[r]

    def lineage(self, item_ids: Iterable[str]) -> set[str]:
        """Tasks whose gold entered the lineage of the records owning ``item_ids``
        (corpus items carry none)."""
        out: set[str] = set()
        for i in item_ids:
            rec = self.record_of(i)
            if rec is not None:
                out |= set(rec.saw_gold)
        return out

    def add_gold(self, task: str, record_id: str, items: Mapping[str, str], step: int) -> B9Record:
        if task in self.gold:
            raise ValueError(f'task {task} already has a gold record')
        rec = B9Record(record_id, task, 'gold', 'train', -1, step, dict(items), (task,))
        self._add(rec)
        self.gold[task] = record_id
        return rec

    def add_own(self, task: str, record_id: str, items: Mapping[str, str], *, round: int,
                step: int, split: str, read_items: Iterable[str] = (),
                correct: bool | None = None) -> B9Record:
        """Register the round's write. ``read_items``: the items the round read with a
        nonzero gate (their records' gold lineage is inherited, as is the lineage of the
        own record this one supersedes). Returns the new record."""
        if split not in ('train', 'eval'):
            raise ValueError('split is train or eval')
        saw = self.lineage(read_items)
        previous = self.own.get(task)
        if previous is not None:
            old = self.records[previous]
            if set(old.items.values()) != set(items.values()):
                raise ValueError('a task record supersedes its previous version (same item ids)')
            saw |= set(old.saw_gold)
            old.current = False
            self.supersedes[task] = self.supersedes.get(task, 0) + 1
        rec = B9Record(record_id, task, 'hinted' if saw else 'model', split, round, step,
                       dict(items), tuple(sorted(saw)),
                       self.generation if split == 'eval' else 0, correct)
        self._add(rec)
        self.own[task] = record_id
        return rec

    def _add(self, rec: B9Record) -> None:
        if rec.record_id in self.records:
            raise ValueError(f'record {rec.record_id} exists')
        self.records[rec.record_id] = rec
        for item_id in rec.items.values():
            self._by_item[item_id] = rec.record_id

    # -- views --------------------------------------------------------------------------
    def visibility(self, task: str, split: str, *, gold_weight: Callable[[str], float],
                   mode: str = 'normal', swap_with: str | None = None,
                   heldout_lineage: str = 'task') -> tuple[set[str], dict[str, str], dict[str, float]]:
        """(hidden item ids, substitutions, gate weights) of a round of ``task``.

        Training rounds see every current training record; gold items at their weight
        (hidden at weight 0). Held-out (``split='eval'``) rounds see only records of
        this evaluation generation's own task and training records whose lineage never
        saw gold for this task (``heldout_lineage='task'``) or never saw gold at all
        (``'any'``; gold records are then hidden too). ``mode``: 'normal', 'removed'
        (the task's own prior record hidden) or 'swapped' (its values replaced by the
        own record of task ``swap_with``)."""
        if mode not in ('normal', 'removed', 'swapped'):
            raise ValueError(f'unknown mode {mode!r}')
        if heldout_lineage not in ('task', 'any'):
            raise ValueError('heldout_lineage is task or any')
        hidden: set[str] = set()
        weights: dict[str, float] = {}
        for rec in self.records.values():
            if not rec.current:
                continue
            ids = set(rec.items.values())
            if rec.split == 'eval' and (split == 'train' or rec.task != task
                                        or rec.generation != self.generation):
                hidden |= ids
                continue
            if rec.kind == 'gold':
                w = float(gold_weight(rec.task))
                if w <= 0 or (split == 'eval' and heldout_lineage == 'any'):
                    hidden |= ids
                    continue
                weights.update(dict.fromkeys(ids, w))
            if split == 'eval':
                if task in rec.saw_gold or (heldout_lineage == 'any' and rec.saw_gold):
                    hidden |= ids
        substitute: dict[str, str] = {}
        own = self.own.get(task)
        if own is not None and self.records[own].current and mode != 'normal':
            mine = self.records[own].items
            if mode == 'removed':
                hidden |= set(mine.values())
            else:
                other = self.own.get(swap_with) if swap_with is not None else None
                if other is None:
                    raise ValueError(f'no own record of {swap_with!r} to swap in')
                theirs = self.records[other].items
                substitute = {mine[s]: theirs[s] for s in mine if s in theirs}
        return hidden, substitute, weights

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(KINDS, 0)
        for rec in self.records.values():
            if rec.current:
                out[rec.kind] += 1
        out['superseded'] = sum(not r.current for r in self.records.values())
        return out

    # -- persistence --------------------------------------------------------------------
    def state(self) -> dict:
        return {'dataset': self.dataset, 'generation': self.generation,
                'records': [dataclasses.asdict(r) for r in self.records.values()],
                'own': dict(self.own), 'gold': dict(self.gold),
                'supersedes': dict(self.supersedes)}

    @classmethod
    def from_state(cls, state: dict) -> Registry:
        reg = cls(state['dataset'])
        reg.generation = state['generation']
        for r in state['records']:
            rec = B9Record(**{**r, 'saw_gold': tuple(r['saw_gold'])})
            reg.records[rec.record_id] = rec
            if rec.current:
                for item_id in rec.items.values():
                    reg._by_item[item_id] = rec.record_id
        reg.own, reg.gold, reg.supersedes = state['own'], state['gold'], state['supersedes']
        return reg


@dataclass
class GoldSchedule:
    """w(task, step) = start x max(0, 1 - step / anneal) (constant if anneal is 0) x
    decay ** (supersedes of the task's own record)."""
    start: float = 1.0
    anneal: int = 0
    decay: float = 0.5

    def global_weight(self, step: int) -> float:
        if self.anneal <= 0:
            return self.start
        return self.start * max(0.0, 1.0 - step / self.anneal)

    def weight(self, registry: Registry, task: str, step: int) -> float:
        return self.global_weight(step) * self.decay ** registry.supersedes.get(task, 0)


# -- a round's view of the KB --------------------------------------------------------------
class KBView:
    """A KB with ``hidden`` items removed from search and reads, and ``substitute``
    items (id -> other id) read with the other item's values (their own key, time and
    id). Everything else is the KB's own (``__getattr__``)."""

    def __init__(self, kb: KnowledgeBase, hidden: Iterable[str] = (),
                 substitute: Mapping[str, str] | None = None):
        self._kb = kb
        self.hidden = frozenset(hidden)
        self.substitute = dict(substitute or {})
        if self.hidden & set(self.substitute):
            raise ValueError('an item is hidden and substituted')

    def __getattr__(self, name):
        return getattr(self._kb, name)

    def search(self, space: str, queries: Tensor, k: int, **options) -> SearchHits:
        exclude = set(options.pop('exclude', ())) | self.hidden
        return self._kb.search(space, queries, k, exclude=exclude, **options)

    def read(self, space: str, ids: Sequence[str], **options):
        blocked = [i for i in ids if i in self.hidden]
        if blocked:
            raise PermissionError(f'items {blocked[:3]} are hidden in this view')
        items = self._kb.read(space, ids, **options)
        if not self.substitute:
            return items
        out = []
        for item in items:
            other = self.substitute.get(item.id)
            if other is not None:
                values = self._kb.read(space, [other], **options)[0].values
                item = dataclasses.replace(item, values=values)
            out.append(item)
        return out


# -- generation with reads ---------------------------------------------------------------
@dataclass
class GenProtocol:
    """Token ids of the chat protocol around memory calls (LFM2 template)."""
    call_start: int
    call_end: int
    im_end: int
    mem: int
    turn_end: list[int]          # after a tool-call block: '<|im_end|>' '\n'
    tool_open: list[int]         # '<|im_start|>tool\n<|mem|>'
    tool_close: list[int]        # '<|/mem|><|im_end|>\n'
    assistant_open: list[int]    # '<|im_start|>assistant\n'
    write_call: list[int]        # '<|im_start|>assistant\n<|tool_call_start|>[memory_write()]<|tool_call_end|>'
    decode: Callable[[list[int]], str] = field(repr=False, default=None)

    @classmethod
    def from_tokenizer(cls, tok) -> GenProtocol:
        from schnitz.span_tokens import SPAN_TOKENS

        def ids(text: str) -> list[int]:
            return list(tok(text, add_special_tokens=False)['input_ids'])
        mem, mem_end = SPAN_TOKENS['mem'][0], SPAN_TOKENS['mem_end'][0]
        return cls(tok.convert_tokens_to_ids('<|tool_call_start|>'),
                   tok.convert_tokens_to_ids('<|tool_call_end|>'),
                   tok.convert_tokens_to_ids('<|im_end|>'), SPAN_TOKENS['mem'][1],
                   ids('<|im_end|>\n'), ids('<|im_start|>tool\n' + mem),
                   ids(mem_end + '<|im_end|>\n'), ids('<|im_start|>assistant\n'),
                   ids('<|im_start|>assistant\n<|tool_call_start|>[memory_write()]'
                       '<|tool_call_end|>'),
                   lambda t: tok.decode(t, skip_special_tokens=False))


SEARCH = re.compile(r'memory_search\(\s*\)')
ONLY_SEARCH = re.compile(r'^\s*\[\s*memory_search\(\s*\)(\s*,\s*memory_search\(\s*\))*\s*\]\s*$')


@dataclass
class Attempt:
    embeds: Tensor                 # (T, width) the whole round: prompt, turns, read spans
    tokens: list[int | None]       # per row: token id, or None for a span row
    answer: str                    # the final assistant turn's text
    generated: int                 # tokens generated by the model
    calls: list[int] = field(default_factory=list)   # row index of each call's query
    mems: list[int] = field(default_factory=list)    # row index of each <|mem|>
    reads: list = field(default_factory=list)
    stop: str = ''                 # 'answer', 'call', 'write', 'length'
    cache: object = None           # the ItemCache the reads fetched their items from


def call_positions(proto: GenProtocol, block: list[int]) -> tuple[list[int], bool]:
    """Offsets (within ``block``) of the token holding each ``memory_search()`` call's
    closing parenthesis, and whether the block holds only such calls."""
    pieces = [proto.decode([t]) for t in block]
    text = ''.join(pieces)
    owner, at = [], 0
    for j, piece in enumerate(pieces):
        owner += [j] * len(piece)
        at += len(piece)
    positions = [owner[m.end() - 1] for m in SEARCH.finditer(text)]
    return positions, bool(ONLY_SEARCH.match(text))


def greedy(logits: Tensor, step: int) -> int:
    return int(logits.argmax(-1))


@torch.no_grad()
def generate(lm, embed: Callable[[list[int]], Tensor], mid: Callable[[Tensor], Tensor],
             read: Callable[[Tensor], object], prompt: list[int], proto: GenProtocol,
             max_new: int, policy: Callable[[Tensor, int], int] = greedy,
             max_reads: int = 8, autocast=None) -> Attempt:
    """Generate one attempt from ``prompt`` with the frozen decoder ``lm`` (HF causal LM
    with ``inputs_embeds`` and a KV cache). ``embed(ids)`` gives input embeddings (span
    protocol rows included), ``mid(x)`` the query-layer states of a full pass over
    ``x`` (T, width), ``read(state)`` a read (``.span`` (n, width)) for one query state.

    When the model closes a tool-call block that holds only ``memory_search()`` calls,
    each call's query state is taken at its closing parenthesis from one pass over the
    exact prefix, the reads are executed, and their spans are placed in tool messages
    (``<|mem|>`` span ``<|/mem|>``); generation continues in a new assistant turn. A
    block with any other call (an external tool call, or ``memory_write``) ends the
    attempt, as does ``<|im_end|>`` after an assistant text turn or ``max_new`` tokens.
    Reads beyond ``max_reads`` get empty spans."""
    scope = autocast or (lambda: torch.autocast('cpu', enabled=False))
    rows: list[Tensor] = [embed(prompt)]
    tokens: list[int | None] = list(prompt)
    attempt = Attempt(torch.empty(0), [], '', 0)
    cache = None

    def feed(x: Tensor) -> Tensor:
        """Run rows ``x`` (n, width) through the cached model one position at a time
        (a multi-token continuation of a convolution cache is not supported); returns
        the last logits."""
        nonlocal cache
        logits = None
        with scope():
            if cache is None:
                out = lm(inputs_embeds=x[None], use_cache=True)
                cache, logits = out.past_key_values, out.logits[0, -1]
            else:
                for j in range(x.shape[0]):
                    out = lm(inputs_embeds=x[None, j:j + 1], past_key_values=cache,
                             use_cache=True)
                    cache, logits = out.past_key_values, out.logits[0, -1]
        return logits.float()

    def push(ids: list[int]) -> Tensor:
        x = embed(ids)
        rows.append(x)
        tokens.extend(ids)
        return feed(x)

    logits = feed(rows[0])
    turn: list[int] = []            # the current assistant turn's generated tokens
    block_start = None
    last_text = ''
    while True:
        if attempt.generated >= max_new:
            attempt.stop = 'length'
            last_text = last_text or proto.decode(turn)
            break
        t = policy(logits, attempt.generated)
        attempt.generated += 1
        turn.append(t)
        if t == proto.call_start:
            block_start = len(tokens)
        logits = push([t])
        if t == proto.call_end and block_start is not None:
            block = tokens[block_start + 1:len(tokens) - 1]
            offsets, only = call_positions(proto, block)
            if not only:
                attempt.stop = 'write' if 'memory_write' in proto.decode(block) else 'call'
                if attempt.stop == 'call':
                    last_text = proto.decode(turn)
                break
            positions = [block_start + 1 + o for o in offsets]
            x = torch.cat(rows)
            states = mid(x[:positions[-1] + 1])
            reads = []
            for p in positions:
                if len(attempt.reads) + len(reads) < max_reads:
                    reads.append(read(states[p]))
                else:
                    reads.append(None)
            logits = push(list(proto.turn_end))
            for p, r in zip(positions, reads):
                attempt.calls.append(p)
                attempt.reads.append(r)
                push(list(proto.tool_open))
                attempt.mems.append(len(tokens) - 1)
                span = r.span.detach().float() if r is not None else rows[0][:0]
                if span.shape[0]:
                    rows.append(span.to(rows[0].dtype))
                    tokens.extend([None] * span.shape[0])
                    feed(span.to(rows[0].dtype))
                logits = push(list(proto.tool_close))
            logits = push(list(proto.assistant_open))
            turn, block_start = [], None
            continue
        if t == proto.im_end:
            attempt.stop = 'answer'
            last_text = proto.decode(turn[:-1])
            break
    attempt.embeds = torch.cat(rows)
    attempt.tokens = tokens
    attempt.answer = last_text.strip()
    return attempt


def write_ids(attempt: Attempt, proto: GenProtocol) -> list[int]:
    """The token ids that open the round's write site after the attempt (none when the
    attempt ended by calling ``memory_write()`` itself)."""
    if attempt.stop == 'write':
        return []
    closed = attempt.tokens and attempt.tokens[-1] == proto.im_end
    return (list(proto.turn_end[1:]) if closed else list(proto.turn_end)) + list(proto.write_call)


def write_source(attempt: Attempt, proto: GenProtocol) -> tuple[Tensor, list[int], list[Tensor]]:
    """The round's write site as a write source (``schnitz.kb.producer.WriteLog``): token
    ids, the ``<|mem|>`` positions among them and the read span after each (in read
    order), so ``Writer.inputs(ids, mems, spans)`` is ``write_prefix`` row for row."""
    ids, mems, spans = [], [], []
    at = set(attempt.mems)
    j, rows = 0, attempt.tokens
    while j < len(rows):
        if rows[j] is None:
            raise ValueError('a span row outside a memory slot')
        ids.append(int(rows[j]))
        if j in at:
            k = j + 1
            while k < len(rows) and rows[k] is None:
                k += 1
            mems.append(len(ids) - 1)
            spans.append(attempt.embeds[j + 1:k].detach().float())
            j = k
            continue
        j += 1
    return torch.tensor(ids + write_ids(attempt, proto)), mems, spans


def write_prefix(attempt: Attempt, proto: GenProtocol,
                 embed: Callable[[list[int]], Tensor]) -> Tensor:
    """The round's write site: the attempt, its turn closed, then the ``memory_write()``
    call up to ``<|tool_call_end|>`` (the position that opens the ``<|bg|>`` span). An
    attempt that ended by calling ``memory_write()`` itself is already there."""
    if attempt.stop == 'write':
        return attempt.embeds
    return torch.cat([attempt.embeds, embed(write_ids(attempt, proto)).to(attempt.embeds.dtype)])


def read_mass(reads: Iterable, registry: Registry, task: str) -> dict[str, float]:
    """Share of read gate mass on the task's own record, gold records, other B9
    records and corpus items (over every read with a nonzero gate)."""
    sums = {'own': 0.0, 'gold': 0.0, 'other_b9': 0.0, 'corpus': 0.0}
    for read in reads:
        if read is None:
            continue
        for info in read.spaces.values():
            for (_, item_id), g in zip(info.refs, info.gates.tolist()):
                rec = registry.record_of(item_id)
                key = 'corpus' if rec is None else 'gold' if rec.kind == 'gold' else \
                    'own' if rec.task == task else 'other_b9'
                sums[key] += float(g)
    total = sum(sums.values())
    out = {k: round(v / total, 4) for k, v in sums.items()} if total > 0 else {}
    scored = [registry.record_of(i) is not None for read in reads if read is not None
              for info in read.spaces.values() for _, i in info.scored]
    if scored:     # B9 records among the scored candidates (retrieved, maybe not read)
        out['b9_scored'] = round(sum(scored) / len(scored), 4)
    return out


def read_items(reads: Iterable) -> set[str]:
    """Items read with a nonzero gate."""
    out = set()
    for read in reads:
        if read is None:
            continue
        for info in read.spaces.values():
            out |= {i for (_, i), g in zip(info.refs, info.gates.tolist()) if g > 0}
    return out
