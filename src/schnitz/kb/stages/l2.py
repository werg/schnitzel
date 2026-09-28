"""L2: producers reproduce the L1a items (docs/knowledge-base-stack.md, 5.1 step 8; the
two-step route of 5.2 "Gradients into producers").

L1a trains item values and keys in place with the writer detached. L2 trains the
producers so that the producer path of each item's source reproduces the item L1a
left: the writer's span of the source record under the memory prompt at the bank's
ratio level (``schnitz.kb.producer.write_spans``, teacher-fed with the bank's cached
span or free-running), the forward codecs, and for items whose lineage says they came
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
import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from schnitz.kb.bank import SpanCache, Transcripts, kb_dir, read_sources, slots_of
from schnitz.kb.loop import Run, Window, warmup_optimizer
from schnitz.kb.producer import (L2Weights, functional_loss, item_losses, key_loss, l2_loss,
                                 produce_items, recombine, rewrite_item, span_length,
                                 write_spans)
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
class Producers:
    """Writer span heads, codecs and S_s of the producer path, with the frozen decoder,
    the L1 reader's key heads and R."""

    def __init__(self, model, reader, level: str, feed: str, caches: dict[str, SpanCache],
                 texts: dict[str, str], neighbour_keys: bool = False):
        self.model, self.reader, self.level, self.feed = model, reader, level, feed
        self.caches, self.texts = caches, texts
        self.neighbour_keys = neighbour_keys
        self._ids: dict[str, torch.Tensor] = {}

    def ids(self, record_id: str) -> torch.Tensor:
        if record_id not in self._ids:
            self._ids[record_id] = self.model.text_ids(self.texts[record_id])
        return self._ids[record_id]

    def spans(self, kbs: list[str], record_ids: list[str]) -> list[torch.Tensor]:
        examples, counts, teacher = [], [], []
        for kb, r in zip(kbs, record_ids):
            ids = self.ids(r)
            factor, n = span_length(ids.shape[0], self.level)
            cache = self.caches.get(kb)
            if cache is not None and r in cache:
                n = int(cache.index[r][2])
                if self.feed == 'teacher':
                    teacher.append(cache.get(r).float())
            elif self.feed == 'teacher':
                raise ValueError(f'record {r} has no cached span to feed')
            examples.append({'ids': ids, 'prompt': 'memory', 'factor': factor})
            counts.append(n)
        return write_spans(self.model, examples, counts, self.feed,
                           teacher if self.feed == 'teacher' else None)

    def items(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        with self._autocast():
            return produce_items(self.reader.stack, span)

    def derived(self, target: Derived, produced_inputs: list[dict[str, torch.Tensor]]):
        inputs = [(items[target.space], gate, None) for items, (_, gate)
                  in zip(produced_inputs, target.inputs)]
        with self._autocast():
            return rewrite_item(self.reader.operators[target.space], inputs, target.key,
                                target.target.shape[0], self.neighbour_keys)

    def _autocast(self):
        core = getattr(self.model, 'core', None)
        return core.autocast() if core is not None else contextlib.nullcontext()


def record_losses(prod: Producers, records: list[Record], weights: L2Weights,
                  functional: bool = True) -> tuple[torch.Tensor, dict]:
    """The L2 loss of a batch of records and its parts (means over the batch)."""
    device = prod.model.device
    spans = prod.spans([r.kb for r in records], [r.record_id for r in records])
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    produced_spans, target_spans, examples = [], [], []
    for rec, span in zip(records, spans):
        items = prod.items(span)
        target = {s: v.to(device) for s, v in rec.target.items()}
        for k, v in item_losses({s: items[s] for s in target}, target).items():
            parts[k].append(v)
        parts['key'].append(torch.stack([key_loss(prod.reader.keys, s, items[s], rec.keys[s])
                                         for s in target]).mean())
        if functional:
            n = span.shape[0]
            with prod._autocast():
                produced_spans.append(recombine(prod.reader.stack, {s: items[s] for s in target}, n))
                with torch.no_grad():
                    target_spans.append(recombine(prod.reader.stack, target, n))
            ids = prod.ids(rec.record_id)
            examples.append({'ids': ids, 'target': ids, 'task': 'reconstruct'})
    means = {k: torch.stack(v).mean() for k, v in parts.items()}
    logged = {}
    if functional:
        with prod._autocast():
            kl, info = functional_loss(prod.model, examples, produced_spans, target_spans)
        means['kl'] = kl
        logged.update(info)
    loss = l2_loss(means, weights)
    logged.update({k: round(v.item(), 5) for k, v in means.items()})
    return loss, logged


def derived_losses(prod: Producers, targets: list[Derived], weights: L2Weights
                   ) -> tuple[torch.Tensor, dict]:
    """Per-space and key losses of rewrite outputs produced through their lineage."""
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for t in targets:
        sources = [r for r, _ in t.inputs]
        spans = prod.spans([t.kb] * len(sources), sources)
        out = prod.derived(t, [prod.items(s) for s in spans])
        for k, v in item_losses({t.space: out}, {t.space: t.target.to(out.device)}).items():
            parts[k].append(v)
        parts['key'].append(key_loss(prod.reader.keys, t.space, out, t.key))
    means = {k: torch.stack(v).mean() for k, v in parts.items()}
    return l2_loss(means, dataclasses.replace(weights, kl=0.0)), \
        {k: round(v.item(), 5) for k, v in means.items()}


# -- evaluation ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_records(prod: Producers, records: list[Record], batch_size: int) -> dict:
    """Reproduction error per space, key cosine, and reading R(produced / target / bank
    items) against no memory, on the records' reconstruction (nats per token)."""
    model = prod.model
    sums: dict[str, float] = defaultdict(float)
    per: dict[str, list[float]] = defaultdict(list)
    tokens = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        spans = prod.spans([r.kb for r in batch], [r.record_id for r in batch])
        examples, arms = [], defaultdict(list)
        for rec, span in zip(batch, spans):
            items = prod.items(span)
            target = {s: v.to(model.device) for s, v in rec.target.items()}
            for k, v in item_losses({s: items[s] for s in target}, target).items():
                per[k].append(v.item())
            for s in target:
                per[f'keycos_{s}'].append(1 - key_loss(prod.reader.keys, s, items[s],
                                                       rec.keys[s]).item())
            if rec.bank:
                bank = {s: v.to(model.device) for s, v in rec.bank.items()}
                for k, v in item_losses(bank, {s: target[s] for s in bank}).items():
                    per[f'bank_{k}'].append(v.item())
            n = span.shape[0]
            with prod._autocast():
                arms['produced'].append(recombine(prod.reader.stack,
                                                  {s: items[s] for s in target}, n))
                arms['target'].append(recombine(prod.reader.stack, target, n))
                if rec.bank:
                    arms['bank'].append(recombine(prod.reader.stack, {
                        s: v.to(model.device) for s, v in rec.bank.items()}, n))
            ids = prod.ids(rec.record_id)
            examples.append({'ids': ids, 'target': ids, 'task': 'reconstruct'})
        with prod._autocast():
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
def write_produced_kbs(prod: Producers, targets: dict[str, KnowledgeBase], dest: Path,
                       batch_size: int) -> dict[str, KnowledgeBase]:
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
            spans = prod.spans([r.kb for r in batch], [r.record_id for r in batch])
            for rec, span in zip(batch, spans):
                items = prod.items(span)
                for s, item_id in rec.items.items():
                    t, sources = times[s][item_id]
                    per_space[s].append(NewItem(
                        items[s].cpu(), prod.reader.keys.item_key(s, items[s]).cpu(),
                        Provenance(tuple(sources), 'codec'), 1.0, t, item_id))
        for t in derived:
            spans = prod.spans([t.kb] * len(t.inputs), [r for r, _ in t.inputs])
            value = prod.derived(t, [prod.items(s) for s in spans])
            time_, sources = times[t.space][t.item_id]
            per_space[t.space].append(NewItem(value.cpu(), prod.reader.keys.item_key(
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
    return l1.evaluate(ctx, episodes, texts, model.tok)


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
    trained = set(args.train.split(','))
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
    prod = Producers(model, reader, level, args.feed, caches, texts, args.neighbour_keys)
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
        report = {'heldout': evaluate_records(prod, held, args.batch_size) if held else None,
                  'train_sample': evaluate_records(prod, train_sample, args.batch_size)}
        if args.eval_episodes:
            dest = args.output / 'produced'
            produced = write_produced_kbs(prod, targets, dest, args.batch_size)
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
        loss, parts = record_losses(prod, batch, weights, functional=weights.kl > 0)
        if train_derived and args.derived_every and step % args.derived_every == 0:
            d_loss, d_parts = derived_losses(prod, rng.sample(train_derived, min(
                len(train_derived), args.batch_size)), weights)
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
    parser.add_argument('--feed', choices=('teacher', 'free'), default='teacher',
                        help="writer input: the bank's cached span, or its own free run")
    parser.add_argument('--train', default='writer',
                        help='comma list of writer (span heads), codecs, operators (S_s)')
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
