"""L2: producers reproduce the L1a items (docs/knowledge-base-stack.md, 5.1 step 8; the
two-step route of 5.2 "Gradients into producers").

L1a trains item values and keys in place with the writer detached. L2 trains the
producers so that the producer path of each item's source reproduces the item L1a
left: the writer's span of the source record under the memory prompt at the bank's
ratio level, rounded to the span cache's bf16, then the forward codecs (the shared
producer replay ``schnitz.kb.producer.Producers`` that L1b and B9 use; ``--feed``:
teacher-fed with the bank's cached span, one gradient pass on the writer's own free run,
or the free run replayed with gradients), and for items whose lineage says they came
from a rewrite, S_s over their produced inputs (gates share x mass, conditioned on the
item's key). The decoder is frozen; the writer's span heads train (rep head and ratio
code), optionally the codecs (``--train codecs``) and S_s (``operators``).

Two actions:

``export`` (also run by ``train`` when the targets are missing): the live KBs of an L1
run (``--l1-run``: ``reader.pt`` and ``kbs/``) at the reader checkpoint's live tag are
copied to a scratch directory, restored to that tag (``restore_live``) and exported as
frozen KBs (``export_live``) under ``<output>/targets``; the L1 run is not touched.
``--targets`` names already exported KBs instead.

``train``: targets are the current items of the exported KBs. Codec items are grouped
by their source record (one item per space); derived items (rewrite outputs) are
produced through their lineage (one level: inputs must be codec items, since an
export keeps only metadata of superseded rows; deeper lineages are counted and
skipped). Losses per record (``schnitz.kb.producer``): per space 1 - cosine per
position and MSE over the target's mean square, key cosine (the L1 reader's item-key
heads on the produced values against the target's live key), and the frozen decoder
reading R(produced items) against R(target items) on the source's reconstruction
(KL); R is the L1 reader's (trained in L1a, frozen here). Records whose id hashes to
the held-out bucket are never trained on.

Evaluation (held-out records, and a training sample): reproduction error per space,
key cosine, and the functional gap: the reconstruction NLL of reading R(produced),
R(target) and R(bank item) (the producer's output before L1a: what a producer that
did not follow L1a gives), with no memory as the floor; ``--eval-episodes N`` also
re-runs L1's evaluation arms (``schnitz.kb.stages.l1.evaluate``) on N validation
transcripts with every item replaced by its produced version (a KB written from the
producers) against the targets. Training-only; runs in ``sdkb-bgkit``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from schnitz.kb.bank import SpanCache, Transcripts, kb_dir, read_sources, slots_of
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.producer import (FEEDS, L2Weights, Producers, Writer, functional_loss,
                                 item_losses, key_loss, l2_loss, recombine, rewrite_item)
from schnitz.kb.stack import SPACES
from schnitz.kb_store import KnowledgeBase, NewItem, Provenance


# -- targets -------------------------------------------------------------------------
def export_targets(l1_run: Path, dest: Path, scratch: Path) -> dict:
    """Frozen exports of an L1 run's live KBs at its reader checkpoint's live tag."""
    state = torch.load(l1_run / 'reader.pt', map_location='cpu', weights_only=False)
    tag = state['live_tag']
    dest.mkdir(parents=True, exist_ok=True)
    report = {'l1_run': str(l1_run), 'live_tag': tag, 'step': int(state['step']), 'kbs': {}}
    for src in sorted(p for p in (l1_run / 'kbs').iterdir() if (p / 'manifest.json').exists()):
        if (dest / src.name).exists():
            report['kbs'][src.name] = 'exists'
            continue
        work = scratch / src.name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(src, work, ignore=shutil.ignore_patterns('writer.lock'))
        kb = KnowledgeBase(work, writable=True)
        try:
            if tag in kb.live_checkpoints():
                kb.restore_live(tag, discard_commits=True)
            kb.export_live(dest / src.name).close()
            report['kbs'][src.name] = {'dataset': kb.dataset, 'live_updates': kb.live_updates}
        finally:
            kb.close()
            shutil.rmtree(work, ignore_errors=True)
    (dest / 'targets.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


@dataclasses.dataclass
class Record:
    """One source record's items in one KB: target (and bank) values and keys per space."""
    kb: str                        # dataset
    record_id: str
    items: dict[str, str]          # space -> item id
    target: dict[str, torch.Tensor]
    keys: dict[str, torch.Tensor]
    bank: dict[str, torch.Tensor] | None = None


@dataclasses.dataclass
class Derived:
    """A rewrite output: its lineage inputs' sources and gates (share x mass)."""
    kb: str
    space: str
    item_id: str
    target: torch.Tensor
    key: torch.Tensor
    inputs: list[tuple[str, float]]    # (input's source record id, share x mass)


def heldout(record_id: str, mod: int) -> bool:
    return mod > 0 and int(hashlib.sha256(('l2-heldout:' + record_id).encode())
                           .hexdigest()[:8], 16) % mod == 0


def _input_metadata(kb: KnowledgeBase) -> dict[tuple[str, int], dict]:
    """(id, version) -> {sources, mass, derived, lineage} for superseded rows (history)
    and current rows."""
    out = {}
    for entry in kb._history():
        out[(entry['id'], entry['version'])] = entry
    for space in kb.spaces:
        for item_id in _current(kb, space):
            item = kb.read(space, [item_id])[0]
            out[(item.id, item.version)] = {
                'sources': list(item.provenance.sources), 'mass': item.mass,
                'derived': item.derived, 'lineage': [[a, v, sh] for (a, v), sh
                                                     in zip(item.lineage, item.shares)]}
    return out


def _current(kb: KnowledgeBase, space: str) -> list[str]:
    from schnitz.kb.read import current_ids
    return current_ids(kb, space)


def collect_targets(kb: KnowledgeBase, bank: KnowledgeBase | None = None
                    ) -> tuple[list[Record], list[Derived], dict]:
    """Records (codec items grouped by their single source) and derived items of a KB."""
    records: dict[str, Record] = {}
    derived: list[Derived] = []
    skipped = defaultdict(int)
    meta = None
    for space in kb.spaces:
        ids = _current(kb, space)
        for start in range(0, len(ids), 512):
            for item in kb.read(space, ids[start:start + 512]):
                if item.derived and item.lineage:
                    meta = meta if meta is not None else _input_metadata(kb)
                    inputs, ok = [], True
                    for (input_id, version), share in zip(item.lineage, item.shares):
                        m = meta.get((input_id, version))
                        if m is None or m['derived'] or len(m['sources']) != 1:
                            ok = False
                            break
                        inputs.append((m['sources'][0], float(share) * float(m['mass'])))
                    if not ok:
                        skipped['deep_lineage'] += 1
                        continue
                    derived.append(Derived(kb.dataset, space, item.id, item.values.float(),
                                           item.key.float(), inputs))
                    continue
                if len(item.provenance.sources) != 1:
                    skipped['several_sources'] += 1
                    continue
                r = item.provenance.sources[0]
                rec = records.setdefault(r, Record(kb.dataset, r, {}, {}, {}))
                if space in rec.items:
                    skipped['second_item_in_space'] += 1
                    continue
                rec.items[space] = item.id
                rec.target[space] = item.values.float()
                rec.keys[space] = item.key.float()
    if bank is not None:
        for rec in records.values():
            rec.bank = {}
            for space, item_id in rec.items.items():
                if bank.has(item_id):
                    rec.bank[space] = bank.read(space, [item_id])[0].values.float()
    return list(records.values()), derived, dict(skipped)


# -- the producer path ------------------------------------------------------------------
def producers(model, reader, level: str, feed: str, caches: dict[str, SpanCache],
              texts: dict[str, str]) -> Producers:
    """The shared producer replay (``schnitz.kb.producer.Producers``) over bank records:
    the writer's span of each record under the memory prompt at ``level`` (fed per
    ``feed``), rounded to the span cache's bf16, then the reader's codecs."""
    return Producers(Writer(model, reader.stack), texts=texts, caches=caches, level=level,
                     mode=feed)


def _codec(rec) -> tuple[str, str, str]:
    return ('codec', rec.kb, rec.record_id)


def derive(prod: Producers, reader, target: Derived,
           produced_inputs: list[dict[str, torch.Tensor]],
           neighbour_keys: bool = False) -> torch.Tensor:
    """A rewrite output from its produced inputs (S_s at gate share x mass)."""
    inputs = [(items[target.space], gate, None) for items, (_, gate)
              in zip(produced_inputs, target.inputs)]
    with prod.writer.autocast():
        return rewrite_item(reader.operators[target.space], inputs, target.key,
                            target.target.shape[0], neighbour_keys)


def _inputs(prod: Producers, target: Derived) -> list[dict[str, torch.Tensor]]:
    return prod.forward([('codec', target.kb, r) for r, _ in target.inputs])


def record_losses(prod: Producers, reader, records: list[Record], weights: L2Weights,
                  functional: bool = True) -> tuple[torch.Tensor, dict]:
    """The L2 loss of a batch of records and its parts (means over the batch)."""
    model = prod.writer.model
    device = model.device
    spans = prod.spans([_codec(r) for r in records])
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    produced_spans, target_spans, examples = [], [], []
    for rec, span in zip(records, spans):
        items = prod.encode(span)
        target = {s: v.to(device) for s, v in rec.target.items()}
        for k, v in item_losses({s: items[s] for s in target}, target).items():
            parts[k].append(v)
        parts['key'].append(torch.stack([key_loss(reader.keys, s, items[s], rec.keys[s])
                                         for s in target]).mean())
        if functional:
            n = span.shape[0]
            with prod.writer.autocast():
                produced_spans.append(recombine(reader.stack, {s: items[s] for s in target}, n))
                with torch.no_grad():
                    target_spans.append(recombine(reader.stack, target, n))
            ids = prod.ids(rec.record_id)
            examples.append({'ids': ids, 'target': ids, 'task': 'reconstruct'})
    means = {k: torch.stack(v).mean() for k, v in parts.items()}
    logged = {}
    if functional:
        with prod.writer.autocast():
            kl, info = functional_loss(model, examples, produced_spans, target_spans)
        means['kl'] = kl
        logged.update(info)
    loss = l2_loss(means, weights)
    logged.update({k: round(v.item(), 5) for k, v in means.items()})
    return loss, logged


def derived_losses(prod: Producers, reader, targets: list[Derived], weights: L2Weights,
                   neighbour_keys: bool = False) -> tuple[torch.Tensor, dict]:
    """Per-space and key losses of rewrite outputs produced through their lineage."""
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for t in targets:
        out = derive(prod, reader, t, _inputs(prod, t), neighbour_keys)
        for k, v in item_losses({t.space: out}, {t.space: t.target.to(out.device)}).items():
            parts[k].append(v)
        parts['key'].append(key_loss(reader.keys, t.space, out, t.key))
    means = {k: torch.stack(v).mean() for k, v in parts.items()}
    return l2_loss(means, dataclasses.replace(weights, kl=0.0)), \
        {k: round(v.item(), 5) for k, v in means.items()}


# -- evaluation ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_records(prod: Producers, reader, records: list[Record], batch_size: int) -> dict:
    """Reproduction error per space, key cosine, and reading R(produced / target / bank
    items) against no memory, on the records' reconstruction (nats per token)."""
    model = prod.writer.model
    sums: dict[str, float] = defaultdict(float)
    per: dict[str, list[float]] = defaultdict(list)
    tokens = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        spans = prod.spans([_codec(r) for r in batch])
        examples, arms = [], defaultdict(list)
        for rec, span in zip(batch, spans):
            items = prod.encode(span)
            target = {s: v.to(model.device) for s, v in rec.target.items()}
            for k, v in item_losses({s: items[s] for s in target}, target).items():
                per[k].append(v.item())
            for s in target:
                per[f'keycos_{s}'].append(1 - key_loss(reader.keys, s, items[s],
                                                       rec.keys[s]).item())
            if rec.bank:
                bank = {s: v.to(model.device) for s, v in rec.bank.items()}
                for k, v in item_losses(bank, {s: target[s] for s in bank}).items():
                    per[f'bank_{k}'].append(v.item())
            n = span.shape[0]
            with prod.writer.autocast():
                arms['produced'].append(recombine(reader.stack,
                                                  {s: items[s] for s in target}, n))
                arms['target'].append(recombine(reader.stack, target, n))
                if rec.bank:
                    arms['bank'].append(recombine(reader.stack, {
                        s: v.to(model.device) for s, v in rec.bank.items()}, n))
            ids = prod.ids(rec.record_id)
            examples.append({'ids': ids, 'target': ids, 'task': 'reconstruct'})
        with prod.writer.autocast():
            reads = {'noctx': model.read(examples, None)}
            target_logits = None
            for name, spans_ in arms.items():
                if len(spans_) == len(examples):
                    reads[name] = model.read(examples, spans_)
            target_logits = reads['target'][0]
            sums['kl_produced'] += F.kl_div(F.log_softmax(reads['produced'][0], -1),
                                            F.log_softmax(target_logits, -1), log_target=True,
                                            reduction='sum').item()
            if 'bank' in reads:
                sums['kl_bank'] += F.kl_div(F.log_softmax(reads['bank'][0], -1),
                                            F.log_softmax(target_logits, -1), log_target=True,
                                            reduction='sum').item()
        for name, (logits, labels) in reads.items():
            sums[f'nll_{name}'] += F.cross_entropy(logits, labels, reduction='sum').item()
        tokens += int(reads['noctx'][1].numel())
    out = {k: round(v / max(tokens, 1), 4) for k, v in sums.items()}
    if 'nll_produced' in out and 'nll_target' in out:
        out['functional_gap'] = round(out['nll_produced'] - out['nll_target'], 4)
    if 'nll_bank' in out and 'nll_target' in out:
        out['l1a_drift_gap'] = round(out['nll_bank'] - out['nll_target'], 4)
    out.update({k: round(sum(v) / len(v), 4) for k, v in sorted(per.items())})
    out['records'], out['tokens'] = len(records), tokens
    return out


@torch.no_grad()
def write_produced_kbs(prod: Producers, reader, targets: dict[str, KnowledgeBase], dest: Path,
                       batch_size: int, neighbour_keys: bool = False
                       ) -> dict[str, KnowledgeBase]:
    """A KB per dataset holding every target item replaced by its produced version (same
    ids and times; keys from the reader's item-key heads), for L1's evaluation arms."""
    out = {}
    shutil.rmtree(dest, ignore_errors=True)
    for name, kb in targets.items():
        records, derived, _ = collect_targets(kb)
        new = KnowledgeBase.create(dest / kb.root.name, name=kb.name + '@produced',
                                   dataset=kb.dataset, origin={'command': 'train.py l2',
                                                               'from': str(kb.root)})
        per_space: dict[str, list[NewItem]] = defaultdict(list)
        times = {s: {} for s in kb.spaces}
        for s in kb.spaces:
            for item in kb.read(s, _current(kb, s)):
                times[s][item.id] = (item.time, item.provenance.sources)
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            spans = prod.spans([_codec(r) for r in batch])
            for rec, span in zip(batch, spans):
                items = prod.encode(span)
                for s, item_id in rec.items.items():
                    t, sources = times[s][item_id]
                    per_space[s].append(NewItem(
                        items[s].cpu(), reader.keys.item_key(s, items[s]).cpu(),
                        Provenance(tuple(sources), 'codec'), 1.0, t, item_id))
        for t in derived:
            value = derive(prod, reader, t, _inputs(prod, t), neighbour_keys)
            time_, sources = times[t.space][t.item_id]
            per_space[t.space].append(NewItem(value.cpu(), reader.keys.item_key(
                t.space, value).cpu(), Provenance(tuple(sources), 'rewrite'), 1.0, time_,
                t.item_id))
        for s, items in per_space.items():
            new.append(s, items)
        new.close()
        out[name] = KnowledgeBase(dest / kb.root.name)
    return out


def task_arms(args, model, reader, frozen_layer: int, kbs: dict[str, KnowledgeBase],
              transcripts: list[Path], limit: int) -> dict:
    """L1's evaluation arms (``l1.evaluate``) over ``kbs`` on validation transcripts."""
    from schnitz.kb.stages import l1
    frozen = l1.Frozen(model.decoder.base_lm, frozen_layer, model.core.autocast)
    ctx = l1.Context(frozen, reader, kbs)
    episodes = []
    for row in Transcripts(transcripts, 'validation', None):
        if len(episodes) >= limit:
            break
        if not slots_of(row) or row['kb'] not in kbs:
            continue
        ep = l1.layout(row, model.tok)
        if ep.ids.numel() <= args.max_tokens and ctx.covered(ep):
            episodes.append(ep)
    if not episodes:
        return {'episodes': 0}
    wanted = {r for ep in episodes for slot in ep.slots for r in slot['record_ids']}
    texts = {r: v['text'] for r, v in
             read_sources({ep.row['_dir'] for ep in episodes}, wanted).items()}
    reader.eval()
    report = l1.evaluate(ctx, episodes, texts, model.tok)
    report.pop('_per_episode', None)
    return report


# -- the write fit: the S_s stack reproduces the learned rows (``--producer stack``) ----------
@dataclasses.dataclass
class RowTargets:
    """The learned rows of one KB: per space the row ids, values and keys (a frozen export
    of the read phase's rows KB, or a snapshot of it)."""
    kb: str                                   # dataset
    dir: str                                  # KB directory name
    rows: dict[str, list[str]]
    values: dict[str, list[torch.Tensor]]
    keys: dict[str, torch.Tensor]


def load_row_targets(root: Path, dirs: dict[str, str]) -> dict[str, RowTargets]:
    """Row targets of every dataset KB found under ``root`` (``dirs``: dataset -> dir)."""
    from schnitz.kb.read import current_ids
    out = {}
    for name, sub in dirs.items():
        path = root / sub
        if not (path / 'manifest.json').exists():
            continue
        kb = KnowledgeBase(path)
        try:
            rows, values, keys = {}, {}, {}
            for space in kb.spaces:
                ids = current_ids(kb, space)
                if not ids:
                    continue
                items = kb.read(space, ids)
                rows[space] = ids
                values[space] = [it.values.float() for it in items]
                keys[space] = torch.stack([it.key.float() for it in items])
            out[name] = RowTargets(kb.dataset, sub, rows, values, keys)
        finally:
            kb.close()
    return out


def latest_snapshot(root: Path) -> tuple[int, Path] | None:
    """The newest row snapshot of a read phase (``l1 train --export-rows-every``)."""
    path = root / 'latest.json'
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    return int(info['step']), root / info['dir']


class StackFit:
    """The write fit over every KB: the leaves (the leaf banks' stored items, in memory),
    the rows (targets) as anchors, one ``superpose.SuperposedKB`` per KB with the stack's
    aggregators, and the fit loss (``superpose.fit_losses``) per space plus a balance
    loss on the rows' loads. Leaf keys are ``head_leaf_key``: the reader's item-key heads
    on the content plus a free per-leaf correction (``corrections``, zero at the start),
    both trained. The read side is cut off at the rows' key/value pairs."""

    def __init__(self, ops, heads, config, leaf_banks: Path, manifest: dict, device,
                 heldout_mod: int):
        self.ops, self.heads, self.config = ops, heads, config
        self.device = torch.device(device)
        self.heldout_mod = heldout_mod
        self.leaves, self.stored, self.records, self.datasets = {}, {}, {}, {}
        self.producers = None
        for name, info in manifest['kbs'].items():
            path = leaf_banks / info['dir']
            if not (path / 'manifest.json').exists():
                continue
            kb = KnowledgeBase(path)
            self.leaves[name] = kb
            self.datasets[name] = kb.dataset
            self.stored[name] = {}
            for space in kb.spaces:
                ids = _current(kb, space)
                items = kb.read(space, ids)
                self.stored[name][space] = {it.id: (it.values.float().to(self.device),
                                                    it.key.float().to(self.device))
                                            for it in items}
                # a leaf's record, when a codec produced it from one (its producer source)
                for it in items:
                    if it.provenance.producer == 'codec' and len(it.provenance.sources) == 1:
                        self.records.setdefault(kb.dataset, {}).setdefault(space, {})[
                            it.id] = it.provenance.sources[0]
        self.corrections = torch.nn.ParameterDict({
            f'{name}/{space}': torch.nn.Parameter(torch.zeros(
                len(items), kb_width(self.leaves[name], space), device=self.device))
            for name, spaces in self.stored.items() for space, items in spaces.items()})
        self.index = {name: {space: {i: n for n, i in enumerate(items)}
                             for space, items in spaces.items()}
                      for name, spaces in self.stored.items()}
        self.views: dict = {}
        self.targets: dict[str, RowTargets] = {}
        self.usage: dict = {}

    def leaf_key(self, name: str):
        from schnitz.kb.superpose import head_leaf_key

        def correction(space, item_id, raw):
            return self.corrections[f'{name}/{space}'][self.index[name][space][item_id]]

        def corrections(space, ids, raws):
            from schnitz.kb.superpose import to_device
            rows = to_device(torch.tensor([self.index[name][space][i] for i in ids]),
                             self.device)
            return self.corrections[f'{name}/{space}'][rows]
        return head_leaf_key(self.heads, correction, corrections)

    def resolve(self, dataset: str, space: str, item_id: str):
        """A leaf's producer source for ``Producers`` (a codec item of one record with
        text and a cached span), else None (the leaf keeps its stored value)."""
        record = self.records.get(dataset, {}).get(space, {}).get(item_id)
        prod = self.producers
        if record is None or prod is None or record not in prod.texts \
                or dataset not in prod.caches or record not in prod.caches[dataset]:
            return None
        return ('codec', dataset, record)

    def stored_value(self, dataset: str, space: str, item_id: str) -> torch.Tensor | None:
        """A leaf's stored bank item (the producers' exactness check)."""
        for name, ds in self.datasets.items():
            if ds == dataset:
                got = self.stored[name].get(space, {}).get(item_id)
                return None if got is None else got[0]
        return None

    def _values(self, name: str, space: str, ids) -> list[torch.Tensor]:
        """Leaf values: with producers the writer's and codecs' recomputation (leaf tensors
        whose gradients ``producers.backward`` carries into writer and codecs), else the
        stored bank items."""
        stored = [self.stored[name][space][i][0] for i in ids]
        if self.producers is None:
            return stored
        fresh = self.producers.values(space, [(self.datasets[name], i) for i in ids])
        return [v if f is None else f for v, f in zip(stored, fresh)]

    def values_fn(self, name: str):
        def values(space, ids):
            got = self._values(name, space, ids)
            return [(v, self.stored[name][space][i][1]) for v, i in zip(got, ids)]
        return values

    def begin(self) -> None:
        """A new step: producer recomputations are stale after an optimizer step."""
        if self.producers is not None:
            self.producers.begin()

    def load(self, targets: dict[str, RowTargets], step: int = 0) -> dict:
        """New row targets: views anchored at the rows, candidates from the leaves' keys."""
        from schnitz.kb.superpose import SuperposedKB
        self.targets = {k: t for k, t in targets.items() if k in self.leaves}
        self.views, stats = {}, {}
        for name, t in self.targets.items():
            view = SuperposedKB(self.leaves[name], self.ops, self.config,
                                {s: (t.rows[s], t.keys[s], [v.shape[0] for v in t.values[s]])
                                 for s in t.rows},
                                values_fn=self.values_fn(name), leaf_key=self.leaf_key(name),
                                device=self.device, live=False)
            stats[name] = view.rebuild(step, reanchor=False, rekey=False)
            self.views[name] = view
        return stats

    def refield(self, fields: dict[str, float], step: int) -> None:
        for view in self.views.values():
            view.refield({s: f for s, f in fields.items() if s in view.graphs}, step)

    def split(self, held: bool) -> dict[str, list[tuple[str, int]]]:
        """(KB, row index) per space, the held-out rows (hash bucket 0) or the others."""
        out: dict[str, list[tuple[str, int]]] = {}
        for name, t in self.targets.items():
            for space, ids in t.rows.items():
                for n, row in enumerate(ids):
                    if heldout(row, self.heldout_mod) == held:
                        out.setdefault(space, []).append((name, n))
        return out

    def _leaf(self, name: str):
        view, key = self.views[name], self.leaf_key(name)

        def leaf(space, ids):
            from schnitz.kb.superpose import leaf_keys, to_device
            g = view.graphs[space]
            values = self._values(name, space, ids)
            keys = leaf_keys(key, space, ids, values,
                             [self.stored[name][space][i][1] for i in ids])
            masses = to_device(torch.tensor([float(g.mass[g.index[i]]) for i in ids]),
                               self.device).unbind(0)
            return list(zip(values, keys, masses))
        return leaf

    def loss(self, picks: dict[str, list[tuple[str, int]]], weights,
             balance: float = 0.0) -> tuple[torch.Tensor, dict]:
        from schnitz.kb.losses import UsageEMA, balance_loss
        from schnitz.kb.superpose import fit_losses
        total, logged = None, {}
        count = sum(len(r) for r in picks.values())
        for space, rows in picks.items():
            for name in dict.fromkeys(n for n, _ in rows):
                t, view = self.targets[name], self.views[name]
                idx = [n for k, n in rows if k == name]
                loss, parts, loads = fit_losses(view, space, [t.rows[space][n] for n in idx],
                                                [t.values[space][n] for n in idx],
                                                t.keys[space][idx], self._leaf(name),
                                                (weights.cos, weights.mse, weights.key))
                loss = loss * len(idx) / count
                if balance:
                    key = f'{name}/{space}'
                    usage = self.usage.setdefault(key, UsageEMA(len(t.rows[space])))
                    ids = torch.tensor(idx)
                    b = balance_loss(ids, loads.cpu(), usage)
                    usage.update(ids, loads.detach().cpu())
                    loss = loss + balance * b * len(idx) / count
                    logged.setdefault('balance', []).append(b.item())
                total = loss if total is None else total + loss
                for k, v in parts.items():
                    logged.setdefault(f'{k}_{space}', []).append(v)
        return total, {k: round(sum(v) / len(v), 5) for k, v in logged.items()}

    @torch.no_grad()
    def measure(self, picks: dict[str, list[tuple[str, int]]], views=None) -> dict:
        """Reproduction per space (1 - cosine, relative MSE, key cosine) of the rows,
        computed fresh."""
        views = views or self.views
        out: dict[str, list[float]] = {}
        for view in views.values():
            view.clear()
        for space, rows in picks.items():
            by_kb: dict[str, list[int]] = {}
            for name, n in rows:
                g = views[name].graphs[space]
                if self.targets[name].rows[space][n] in g.top_index:
                    by_kb.setdefault(name, []).append(n)
            for name, ns in by_kb.items():
                t, view = self.targets[name], views[name]
                g = view.graphs[space]
                got_rows = view.items(space, g.depth, [g.top_index[t.rows[space][n]]
                                                       for n in ns])
                for n, (value, key, _) in zip(ns, got_rows):
                    self._measure_one(out, space, t, n, value, key)
        return {k: round(sum(v) / len(v), 5) for k, v in sorted(out.items())} | \
            {'rows': sum(len(r) for r in picks.values())}

    @staticmethod
    def _measure_one(out: dict, space: str, t, n: int, value: torch.Tensor, key: torch.Tensor) -> None:
        from schnitz.kb.producer import item_losses
        got = item_losses({space: value}, {space: t.values[space][n]})
        out.setdefault(f'cos_{space}', []).append(got[f'cos_{space}'].item())
        out.setdefault(f'mse_{space}', []).append(got[f'mse_{space}'].item())
        out.setdefault(f'keycos_{space}', []).append(float(F.cosine_similarity(
            key.cpu(), t.keys[space][n], dim=-1)))

    def write_stats(self) -> dict:
        """Per KB and space: share entropy, row load distribution, churn, tau."""
        return {name: {s: view.write_stats(s) for s in view.graphs}
                for name, view in self.views.items()}

    def cost(self, picks, weights, fields: dict[str, float], step: int) -> dict:
        """Time and peak memory of one fit forward and backward at the given field sizes
        (no optimizer step; gradients are dropped)."""
        self.refield(fields, step)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss, _ = self.loss(picks, weights)
        loss.backward()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        out = {'step_s': round(time.time() - t0, 3),
               'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 3)
               if self.device.type == 'cuda' else None}
        for p in list(self.ops.parameters()) + list(self.heads.parameters()) + \
                list(self.corrections.parameters()):
            p.grad = None
        return out

    def density(self, fraction: float, picks, step: int) -> dict:
        """The fit on KBs with only a ``fraction`` of the rows (every k-th row; the same
        leaves), so every field is larger: robustness to KB density."""
        from schnitz.kb.superpose import SuperposedKB
        keep = max(1, round(1 / fraction))
        views = {}
        for name, t in self.targets.items():
            rows = {s: ([r for n, r in enumerate(t.rows[s]) if n % keep == 0],
                        t.keys[s][::keep], [v.shape[0] for v in t.values[s][::keep]])
                    for s in t.rows}
            view = SuperposedKB(self.leaves[name], self.ops, self.config, rows,
                                values_fn=self.values_fn(name), leaf_key=self.leaf_key(name),
                                device=self.device, live=False)
            view.rebuild(step, reanchor=False, rekey=False)
            views[name] = view
        kept = {s: [(k, n) for k, n in rows if n % keep == 0] for s, rows in picks.items()}
        return self.measure(kept, views)

    def export(self, dest: Path, step: int) -> dict[str, KnowledgeBase]:
        """The stack's rows as frozen KBs (``SuperposedKB.export``), one per dataset."""
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        out = {}
        for name, view in self.views.items():
            view.export(dest / self.targets[name].dir, step=step).close()
            out[name] = KnowledgeBase(dest / self.targets[name].dir)
        return out

    def state(self) -> dict:
        return {'heads': self.heads.state_dict(),
                'corrections': {k: v.detach().cpu() for k, v in self.corrections.items()},
                'correction_ids': {name: {s: list(items) for s, items in spaces.items()}
                                   for name, spaces in self.stored.items()}}

    def close(self) -> None:
        for kb in self.leaves.values():
            kb.close()


def kb_width(kb: KnowledgeBase, space: str) -> int:
    return kb.spaces[space].key_width


def _open_kbs(root: Path, dirs: dict[str, str]) -> dict[str, KnowledgeBase]:
    return {name: KnowledgeBase(root / sub) for name, sub in dirs.items()
            if (root / sub / 'manifest.json').exists()}


def stack_task_arms(args, model, reader, kbs: dict[str, KnowledgeBase], transcripts) -> dict:
    report = task_arms(args, model, reader, args.query_layer, kbs, transcripts,
                       args.eval_episodes)
    report.pop('superposition_kbs', None)
    return report


def train_stack(args) -> None:
    """``l2 train --producer stack``: the write fit (see ``StackFit``). Targets: the rows of
    a read phase (``--l1-run`` exported at its checkpoint, or ``--targets``), or the newest
    snapshot under ``--follow-snapshots`` (reloaded when a newer one appears). Each step
    draws a level-1 field size per space log-uniformly from its ``--field`` range. The fit
    needs no decoder; each evaluation reports the fit (held-out and training rows) at the
    nominal field size, a sweep of field sizes (quality, time and peak memory per step),
    KBs with half and a quarter of the rows, the write side's shares, loads and churn,
    and with ``--eval-episodes`` L1's arms with the read phase's reader on the rows
    (targets) and on the stack's rows; ``--stack-eval-banks`` builds rows for other banks
    through the stack alone."""
    from schnitz.kb.stack import KeyHeads
    from schnitz.kb.stages.l1 import load_stack, superpose_config
    from schnitz.kb.superpose import WriteOps
    l1_config = _l1_config(args)
    for name in ('checkpoint', 'reader_state', 'banks'):
        if getattr(args, name) is None and l1_config.get(name):
            setattr(args, name, Path(l1_config[name]))
    if args.banks is None:
        raise SystemExit('--banks (a rows banks dir, l1 rows) is needed (or --l1-run)')
    if args.experiment is None:
        args.experiment = l1_config.get('experiment', 'bgkit2_s2_showcase')
    if args.query_layer is None:
        args.query_layer = int(l1_config.get('query_layer', 8))
    manifest = json.loads((args.banks / 'banks.json').read_text())
    if 'rows' not in manifest:
        raise SystemExit(f'{args.banks} is not a rows banks dir (train.py l1 rows)')
    trained = set((args.train or 'operators,keys').split(','))
    if trained - {'operators', 'keys', 'writer', 'codecs'}:
        raise SystemExit('the write fit trains operators, keys, writer and codecs')
    out = Run(args.output)
    dirs = {name: info['dir'] for name, info in manifest['kbs'].items()}
    snapshot_step = -1
    reader_pt = args.l1_run / 'reader.pt' if args.l1_run else args.l1_reader
    if args.follow_snapshots:
        found = latest_snapshot(args.follow_snapshots)
        if found is None:
            raise SystemExit(f'no snapshot under {args.follow_snapshots} yet')
        snapshot_step, targets_dir = found
        reader_pt = targets_dir / 'reader.pt'
    else:
        targets_dir = args.targets or args.output / 'targets'
        if not (targets_dir / 'targets.json').exists() and args.targets is None:
            if args.l1_run is None:
                raise SystemExit('train needs --targets, --l1-run or --follow-snapshots')
            print(json.dumps({'export': export_targets(args.l1_run, targets_dir,
                                                       args.output / 'scratch')}), flush=True)
    _, _, dims = load_stack(args.banks / 'stack.pt', 1.0, 'cpu', args.seed)
    config = superpose_config(args, manifest['rows'].get('config'))
    config.depth = args.stack_depth
    device = 'cuda' if torch.cuda.is_available() and args.cuda_fraction > 0 else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
    torch.manual_seed(args.seed)
    ops = WriteOps(config.depth, dims, config.per_level, temperature=config.temperature)
    if args.stack_init:
        state = torch.load(args.stack_init, map_location='cpu', weights_only=False)
        if 'stack' in state:
            ops.load_state_dict(state['stack'])
        else:
            print(json.dumps({'stack_init_tensors': ops.load_init(state.get('state', state))}),
                  flush=True)
    # the leaves' key heads: the read phase's item-key heads (unused by its learned keys,
    # so the heads the leaves' stored keys came from)
    reader_state = torch.load(reader_pt, map_location='cpu', weights_only=False)
    read_config = reader_state.get('config', {})
    heads = KeyHeads(read_config.get('hidden', 1024), read_config.get('key_hidden', 512))
    heads.load_state_dict({k[len('keys.'):]: v for k, v in reader_state['reader'].items()
                           if k.startswith('keys.')})
    ops.to(device)
    heads.to(device)
    leaf_banks = Path(manifest['rows']['leaves'])
    fit = StackFit(ops, heads, config, leaf_banks, manifest, device, args.heldout_mod)
    # writer and codecs through the stack: the leaves recomputed by the shared producer
    # replay (the record's writer span, then the codecs), as in L1b
    model = prod = codec_stack = None
    producer_sets = {'writer': [], 'codecs': []}
    if trained & {'writer', 'codecs'}:
        from schnitz.kb.producer import producer_params
        from schnitz.kb.stages.l1 import load_model
        model = load_model(args)
        model.writer.eval()           # replay forward = recomputation (no dropout)
        codec_reader, _ = load_reader(model, leaf_banks, None, args.seed)
        codec_stack = codec_reader.stack
        leaf_manifest = json.loads((leaf_banks / 'banks.json').read_text())
        span_root = Path(leaf_manifest.get('span_cache') or leaf_banks / 'spans')
        caches = {fit.datasets[name]: SpanCache(span_root / kb_dir(name))
                  for name in fit.leaves if (span_root / kb_dir(name) / 'manifest.json').exists()}
        wanted = {r for spaces in fit.records.values() for ids in spaces.values()
                  for r in ids.values()}
        texts = {r: v['text'] for r, v in read_sources(
            {str(t) for t in leaf_manifest['transcripts']}, wanted).items()}
        prod = Producers(Writer(model, codec_stack), resolve=fit.resolve,
                         stored=fit.stored_value, texts=texts,
                         caches=caches, level=args.level or leaf_manifest.get('level', 's0'),
                         mode=args.feed, batch=args.producer_batch)
        fit.producers = prod
        chosen = producer_params(model, codec_stack)
        for name in ('writer', 'codecs'):
            if name in trained:
                producer_sets[name] = chosen[name]
                for q in chosen[name]:
                    q.requires_grad_(True)
        print(json.dumps({'producers': {'records': len(wanted), 'texts': len(texts),
                                        'caches': sorted(caches), 'feed': args.feed}}),
              flush=True)
    rng = random.Random(args.seed)
    step = 0
    state = out.load('producers.pt', 'cpu')
    if state is not None:
        ops.load_state_dict(state['stack'])
        heads.load_state_dict(state['heads'])
        if prod is not None and 'writer' in state:
            model.writer.load_state_dict(state['writer'])
            codec_stack.codecs.load_state_dict(state['codecs'])
        for k, v in state['corrections'].items():
            fit.corrections[k].data.copy_(v)
        step = state['step']
        rng.setstate(state['rng'])
        snapshot_step = state.get('targets_step', snapshot_step)
    budgets = fit.load(load_row_targets(targets_dir, dirs), step)
    # the key paths (the stack's row-key heads, the leaves' item-key heads and corrections)
    # at their own rate: their outputs are unit keys that a full-rate step overshoots
    key_params = list(ops.key_heads.parameters())
    if 'keys' in trained:
        key_params += list(heads.item.parameters()) + list(fit.corrections.parameters())
    else:
        for p in list(heads.parameters()) + list(fit.corrections.parameters()):
            p.requires_grad_(False)
    ids = {id(p) for p in key_params}
    params = [p for p in ops.parameters() if id(p) not in ids] + key_params
    groups = [{'params': [p for p in ops.parameters() if id(p) not in ids], 'lr': args.lr},
              {'params': key_params, 'lr': args.key_lr}]
    for name, lr in (('writer', args.writer_lr), ('codecs', args.codec_lr)):
        if producer_sets[name]:            # their own low rates, as L1b's producers
            groups.append({'params': producer_sets[name], 'lr': lr})
            params += producer_sets[name]
    optimizer, schedule = warmup_optimizer(groups, args.warmup, step, lr=args.lr,
                                           weight_decay=0.0)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    weights = L2Weights.parse(args.weights)
    out.write_config(dict(vars(args), producer='stack', superpose=dataclasses.asdict(config),
                          targets_dir=str(targets_dir), rows_budget=budgets,
                          trained=sorted(trained),
                          trained_params=sum(p.numel() for p in params)))
    train_split, held_split = fit.split(False), fit.split(True)
    nominal = {s: config.field_size(s) for s in SPACES}
    reader = None

    def evaluate(tag: int) -> None:
        nonlocal model, reader
        sample = {s: random.Random(1).sample(rows, min(len(rows), args.eval_records))
                  for s, rows in train_split.items()}
        held = {s: rows[:args.eval_records] for s, rows in held_split.items()}
        fit.refield(nominal, tag)
        report = {'heldout': fit.measure(held), 'train_sample': fit.measure(sample),
                  'write': fit.write_stats()}
        sweep = {}
        batch = {s: rows[:args.batch_size] for s, rows in train_split.items()}
        for name, pick in (('min', 0), ('mid', 1), ('max', 2), ('beyond', 3)):
            fields = {}
            for s in SPACES:
                lo, hi = config.field_range(s)
                fields[s] = (lo, math.sqrt(lo * hi), hi, 2 * hi)[pick]
            fit.refield(fields, tag)
            sweep[name] = {'field': {s: round(f, 1) for s, f in fields.items()},
                           'heldout': fit.measure(held),
                           **fit.cost(batch, weights, fields, tag)}
        report['field_sweep'] = sweep
        fit.refield(nominal, tag)
        report['density'] = {f'{int(100 * f)}%': fit.density(f, held, tag) for f in (0.5, 0.25)}
        if args.eval_episodes:
            if model is None:
                from schnitz.kb.stages.l1 import load_model
                model = load_model(args)
            if reader is None:
                reader, _ = load_reader(model, args.banks, reader_pt, args.seed)
            transcripts = [Path(p) for p in manifest['transcripts']]
            targets = _open_kbs(targets_dir, dirs)
            produced = fit.export(args.output / 'produced', tag)
            report['task_arms'] = {'rows': stack_task_arms(args, model, reader, targets,
                                                           transcripts),
                                   'stack': stack_task_arms(args, model, reader, produced,
                                                            transcripts)}
            for kbs in (targets, produced):
                for kb in kbs.values():
                    kb.close()
            if args.stack_eval_banks:
                report['new_banks'] = stack_new_banks(args, model, reader, fit, config, tag)
        out.log({'step': tag, 'targets_step': snapshot_step, 'eval': report})

    if step == 0 and args.eval_every:
        evaluate(0)
    window = Window()
    started = time.time()
    while step < args.steps:
        picks = {s: [rows[rng.randrange(len(rows))] for _ in range(args.batch_size)]
                 for s, rows in train_split.items() if rows}
        fields = {s: config.sample_field(s, rng) for s in SPACES}
        fit.begin()
        fit.refield(fields, step)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = fit.loss(picks, weights, args.balance_weight)
        loss.backward()
        if prod is not None:          # the leaves' gradients into the writer and codecs
            got = prod.backward()
            parts.update({f'producer_{k}': v for k, v in got.items()
                          if isinstance(v, (int, float)) and v is not None})
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        schedule.step()
        step += 1
        window.add({'loss': loss.item(), **parts, 'step_s': time.time() - t0,
                    **{f'field_{s}': f for s, f in fields.items()},
                    **({'peak_gb': torch.cuda.max_memory_allocated() / 2**30}
                       if device == 'cuda' else {})})
        if step % args.log_every == 0:
            out.log({'step': step, **window.means(), 'elapsed_s': round(time.time() - started)})
        if args.follow_snapshots and step % max(1, args.follow_every) == 0:
            found = latest_snapshot(args.follow_snapshots)
            if found is not None and found[0] > snapshot_step:
                snapshot_step, targets_dir = found
                reader_pt, reader = targets_dir / 'reader.pt', None
                fit.load(load_row_targets(targets_dir, dirs), step)
                train_split, held_split = fit.split(False), fit.split(True)
                out.log({'step': step, 'targets_step': snapshot_step, 'reloaded': str(targets_dir)})
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            extra = {} if prod is None else {'writer': model.writer.state_dict(),
                                             'codecs': codec_stack.codecs.state_dict()}
            out.save('producers.pt', {'stack': ops.state_dict(), **fit.state(), **extra,
                                      'optimizer': optimizer.state_dict(), 'step': step,
                                      'rng': rng.getstate(), 'targets_step': snapshot_step,
                                      'superpose': dataclasses.asdict(config)})
            evaluate(step)
    fit.close()


def stack_new_banks(args, model, reader, fit: StackFit, config, step: int) -> dict:
    """Rows for other banks through the stack alone (no L1): per KB a rows KB placed and
    initialized by ``build_rows`` (the untrained field mean), then the stack's rows at
    those anchors (leaf keys from the fit's heads, no corrections: the items are new);
    L1's arms on the mean-initialized rows, the stack's rows and the leaves themselves."""
    from schnitz.kb.superpose import SuperposedKB, build_rows, head_leaf_key, rows_of
    banks = args.stack_eval_banks
    manifest = json.loads((banks / 'banks.json').read_text())
    work = args.output / 'new_banks'
    shutil.rmtree(work, ignore_errors=True)
    arms = {'leaves': {}, 'mean_rows': {}, 'stack_rows': {}}
    key = head_leaf_key(fit.heads, lambda space, item_id, raw: torch.zeros_like(raw))
    for name, info in manifest['kbs'].items():
        leaves = KnowledgeBase(banks / info['dir'])
        rows_kb, _ = build_rows(leaves, work / 'mean' / info['dir'], config)
        view = SuperposedKB(leaves, fit.ops, config, rows_of(rows_kb), leaf_key=key,
                            device=fit.device, live=False)
        view.rebuild(step, reanchor=False, rekey=False)
        rows_kb.close()
        view.export(work / 'stack' / info['dir'], step=step).close()
        arms['leaves'][name] = leaves
        arms['mean_rows'][name] = KnowledgeBase(work / 'mean' / info['dir'])
        arms['stack_rows'][name] = KnowledgeBase(work / 'stack' / info['dir'])
    transcripts = [Path(p) for p in manifest['transcripts']]
    report = {arm: stack_task_arms(args, model, reader, kbs, transcripts)
              for arm, kbs in arms.items()}
    for kbs in arms.values():
        for kb in kbs.values():
            kb.close()
    return report


# -- stage -------------------------------------------------------------------------------
def _l1_config(args) -> dict:
    if args.l1_run is None:
        return {}
    path = args.l1_run / 'config.json'
    return json.loads(path.read_text()) if path.exists() else {}


def load_reader(model, banks: Path, reader_state: Path | None, seed: int):
    """The L1 reader (key heads, S_s, the stack with R) from an L1 ``reader.pt``, or its
    initial state from the banks (initial key heads, the banks' stack)."""
    from schnitz.kb.read import L1Reader, ReadConfig
    from schnitz.kb.stages.l1 import load_stack
    stack, _, dims = load_stack(banks / 'stack.pt', model.target_norm, 'cpu', seed)
    lm = model.decoder.base_lm
    state = None
    if reader_state is not None:
        state = torch.load(reader_state, map_location='cpu', weights_only=False)
        if 'decoder_layers' in state or 'decoder_lora' in state:
            # the reader's queries come from its own decoder layers (--decoder-train-below),
            # which this consumer does not load yet: refuse rather than query with the parent's
            raise NotImplementedError(f'{reader_state} was trained with its own decoder layers '
                                      '(--decoder-train-below); loading them here is not built')
        config = ReadConfig(**state['config'])
    else:
        config = ReadConfig(hidden=lm.config.hidden_size,
                            span_width=lm.get_input_embeddings().weight.shape[1],
                            target_norm=model.target_norm, state=dims['state'],
                            op_hidden=dims['hidden'], layers=dims['layers'])
    config.checkpointing = False
    torch.manual_seed(seed)
    reader = L1Reader(config, stack)
    if state is not None:
        reader.load_state_dict(state['reader'])
    else:
        reader.keys.load_state_dict(torch.load(banks / 'key_heads_init.pt', map_location='cpu'))
    for p in reader.parameters():
        p.requires_grad_(False)
    return reader.to(model.device), config


def train(args) -> None:
    if args.producer == 'stack':
        train_stack(args)
        return
    from schnitz.kb.stages.l1 import load_model
    l1_config = _l1_config(args)
    for name in ('checkpoint', 'reader_state', 'banks'):
        if getattr(args, name) is None and l1_config.get(name):
            setattr(args, name, Path(l1_config[name]))
        if getattr(args, name) is None:
            raise SystemExit(f'--{name.replace("_", "-")} is needed (or --l1-run)')
    if args.experiment is None:
        args.experiment = l1_config.get('experiment', 'bgkit2_s2_showcase')
    if args.query_layer is None:
        args.query_layer = int(l1_config.get('query_layer', 8))
    out = Run(args.output)
    targets_dir = args.targets or args.output / 'targets'
    if not (targets_dir / 'targets.json').exists() and args.targets is None:
        if args.l1_run is None:
            raise SystemExit('train needs --targets or --l1-run')
        print(json.dumps({'export': export_targets(args.l1_run, targets_dir,
                                                   args.output / 'scratch')}), flush=True)
    banks = json.loads((args.banks / 'banks.json').read_text())
    level = args.level or banks['level']
    transcripts = [Path(p) for p in banks['transcripts']]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_model(args)
    reader_pt = args.l1_run / 'reader.pt' if args.l1_run else args.l1_reader
    reader, config = load_reader(model, args.banks, reader_pt, args.seed)
    trained = set((args.train or 'writer').split(','))
    params = []
    for name in ('rep', 'ratio'):
        if 'writer' in trained:
            for p in getattr(model.writer, name).parameters():
                p.requires_grad_(True)
                params.append(p)
    if 'codecs' in trained:
        for p in reader.stack.codecs.parameters():
            p.requires_grad_(True)
            params.append(p)
    if 'operators' in trained:
        for p in reader.operators.parameters():
            p.requires_grad_(True)
            params.append(p)
    if not params:
        raise SystemExit('nothing to train (--train writer,codecs,operators)')
    # targets, bank items, texts, span caches
    targets, records, derived, skipped = {}, [], [], {}
    span_root = Path(banks.get('span_cache') or args.banks / 'spans')
    caches = {}
    for name, info in banks['kbs'].items():
        path = targets_dir / info['dir']
        if not path.exists():
            continue
        kb = KnowledgeBase(path)
        bank = KnowledgeBase(args.banks / info['dir'])
        recs, der, skip = collect_targets(kb, bank)
        bank.close()
        targets[name] = kb
        records += recs
        derived += der
        skipped[name] = skip
        if (span_root / kb_dir(name) / 'manifest.json').exists():
            caches[name] = SpanCache(span_root / kb_dir(name))
    wanted = {r.record_id for r in records} | {s for d in derived for s, _ in d.inputs}
    texts = {r: v['text'] for r, v in read_sources({str(p) for p in transcripts}, wanted).items()}
    missing = wanted - set(texts)
    if missing:
        raise ValueError(f'{len(missing)} target sources are not in the corpora')
    records.sort(key=lambda r: (r.kb, r.record_id))
    train_recs = [r for r in records if not heldout(r.record_id, args.heldout_mod)]
    held = [r for r in records if heldout(r.record_id, args.heldout_mod)][:args.eval_records]
    train_sample = random.Random(1).sample(train_recs, min(len(train_recs), args.eval_records))
    train_derived = [d for d in derived if not any(heldout(s, args.heldout_mod)
                                                   for s, _ in d.inputs)]
    prod = producers(model, reader, level, args.feed, caches, texts)
    weights = L2Weights.parse(args.weights)
    step = 0
    state = out.load('producers.pt', model.device)
    if state is not None:
        model.writer.load_state_dict(state['writer'])
        reader.load_state_dict(state['reader'])
        step = state['step']
        rng.setstate(state['rng'])
    optimizer, schedule = warmup_optimizer(params, args.warmup, step, lr=args.lr,
                                           weight_decay=0.0)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
    out.write_config(dict(vars(args), level=level, read_config=dataclasses.asdict(config),
                          records=len(records), train_records=len(train_recs),
                          heldout_records=len(held), derived=len(derived), skipped=skipped,
                          trained_params=sum(p.numel() for p in params)))

    def evaluate(tag_step: int) -> None:
        model.writer.eval()
        report = {'heldout': evaluate_records(prod, reader, held, args.batch_size) if held
                  else None,
                  'train_sample': evaluate_records(prod, reader, train_sample, args.batch_size)}
        if args.eval_episodes:
            dest = args.output / 'produced'
            produced = write_produced_kbs(prod, reader, targets, dest, args.batch_size,
                                          args.neighbour_keys)
            report['task_arms'] = {
                'target': task_arms(args, model, reader, args.query_layer, targets,
                                    transcripts, args.eval_episodes),
                'produced': task_arms(args, model, reader, args.query_layer, produced,
                                      transcripts, args.eval_episodes)}
            for kb in produced.values():
                kb.close()
        out.log({'step': tag_step, 'eval': report})

    if step == 0 and args.eval_every:
        evaluate(0)
    window = Window()
    order: list[int] = []
    started = time.time()
    while step < args.steps:
        if not order:
            order = list(range(len(train_recs)))
            rng.shuffle(order)
        batch = [train_recs[order.pop()] for _ in range(min(args.batch_size, len(order)))]
        model.writer.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = record_losses(prod, reader, batch, weights, functional=weights.kl > 0)
        if train_derived and args.derived_every and step % args.derived_every == 0:
            d_loss, d_parts = derived_losses(prod, reader, rng.sample(train_derived, min(
                len(train_derived), args.batch_size)), weights, args.neighbour_keys)
            loss = loss + d_loss
            parts.update({f'derived_{k}': v for k, v in d_parts.items()})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        schedule.step()
        step += 1
        window.add({'loss': loss.item(), **parts})
        if step % args.log_every == 0:
            out.log({'step': step, **window.means(), 'elapsed_s': round(time.time() - started)})
        if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
            out.save('producers.pt', {'writer': model.writer.state_dict(),
                                      'reader': reader.state_dict(),
                                      'optimizer': optimizer.state_dict(), 'step': step,
                                      'rng': rng.getstate()})
            evaluate(step)
    for kb in targets.values():
        kb.close()


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('action', choices=('export', 'train'))
    parser.add_argument('--l1-run', type=Path, help='L1 train output (reader.pt, kbs/, config.json)')
    parser.add_argument('--targets', type=Path,
                        help='exported (frozen) KBs, one directory per dataset KB as in the banks')
    parser.add_argument('--banks', type=Path, help='L1 build output (default: the L1 run\'s)')
    parser.add_argument('--l1-reader', type=Path,
                        help='L1 reader.pt when --l1-run is not given (default: initial heads)')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--reader-state', type=Path, help='writer-stage state (B3 writer.pt)')
    parser.add_argument('--query-layer', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--level', help='ratio level of the writes (default: the banks\')')
    parser.add_argument('--feed', choices=FEEDS, default='teacher',
                        help="writer feed (schnitz.kb.producer.write_spans): the bank's cached "
                             "span (teacher), one gradient pass on its own free run (self), or "
                             "the free run replayed with gradients through every step (free, "
                             "L1b's replay)")
    parser.add_argument('--train', default=None,
                        help='comma list of writer (span heads), codecs, operators (S_s); '
                             'default writer (--producer record), operators (stack)')
    parser.add_argument('--producer', choices=('record', 'stack'), default='record',
                        help='record: each record\'s producer path reproduces its L1a items; '
                             'stack: the write fit, S_s levels over the leaves regress the read '
                             'phase\'s rows (values and keys; no decoder in the loop)')
    parser.add_argument('--stack-depth', type=int, default=2, help='stack: aggregator levels L')
    parser.add_argument('--stack-init', type=Path,
                        help='stack: initial aggregators (a stack producers.pt, or per-space '
                             'operator weights such as K3\'s)')
    parser.add_argument('--field', help='stack: level-1 field size per space, a number or a '
                                     'range sampled log-uniformly per step, e.g. '
                                     'A=4:16,B=8:32,C=8:32,D=8:32 (default: the rows KB\'s)')
    parser.add_argument('--overlap', type=int, help='stack: fields each input joins (3)')
    parser.add_argument('--temperature', type=float, help='stack: share temperature (0.1)')
    parser.add_argument('--follow-snapshots', type=Path,
                        help='stack: fit the newest row snapshot under this dir (l1 train '
                             '--export-rows-every) and reload when a newer one appears')
    parser.add_argument('--follow-every', type=int, default=50,
                        help='stack: steps between checks for a newer snapshot')
    parser.add_argument('--key-lr', type=float, default=1e-4,
                        help='stack: rate of the key paths (row-key heads, item-key heads, '
                        'leaf corrections)')
    parser.add_argument('--writer-lr', type=float, default=3e-6,
                        help='stack with --train writer: the writer\'s rate (L1b\'s 3e-6)')
    parser.add_argument('--codec-lr', type=float, default=3e-5,
                        help='stack with --train codecs: the codecs\' rate (L1b\'s 3e-5)')
    parser.add_argument('--producer-batch', type=int, default=1,
                        help='stack with --train writer/codecs: records replayed together '
                        '(1: exact for banks written one at a time, as L1b)')
    parser.add_argument('--unbatched', action='store_true',
                        help='stack: aggregators row by row (reference path)')
    parser.add_argument('--max-pairs', type=int,
                        help='stack: position pairs per batched aggregator pass (262144)')
    parser.add_argument('--balance-weight', type=float, default=0.01,
                        help='stack: balance loss on the rows\' loads (against collapse)')
    parser.add_argument('--stack-eval-banks', type=Path,
                        help='stack: other banks (l1 build) to build rows for through the stack '
                             'alone and evaluate (with --eval-episodes)')
    parser.add_argument('--neighbour-keys', action='store_true', help='S_s with input keys (K3b)')
    parser.add_argument('--weights', default='cos=1,mse=1,key=1,kl=1')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8, help='records per step')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--derived-every', type=int, default=1,
                        help='add a batch of rewrite outputs every N steps (if any)')
    parser.add_argument('--heldout-mod', type=int, default=10,
                        help='records whose hash falls in bucket 0 of N are held out')
    parser.add_argument('--eval-records', type=int, default=64)
    parser.add_argument('--eval-episodes', type=int, default=0,
                        help="re-run L1's evaluation arms on N validation transcripts with "
                             'produced items (0: off)')
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--cuda-fraction', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=0)


def run(args) -> None:
    if args.action == 'export':
        if args.l1_run is None:
            raise SystemExit('export needs --l1-run')
        report = export_targets(args.l1_run, args.targets or args.output / 'targets',
                                args.output / 'scratch')
        print(json.dumps(report), flush=True)
        return
    train(args)
