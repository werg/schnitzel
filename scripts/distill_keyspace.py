"""Benchmark teachers and run the cached head-only keyspace warmup.

``benchmark`` measures teacher and lexical source-disjoint recall on a held-out
portion of the training sites. ``warmup`` fits frozen per-space teacher
projections and temperatures, distills direct per-space key heads over the whole
eligible field, writes a complete coherent journal, and saves a direct-interface
checkpoint for the continued trajectory stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, load_model
import torch
from torch.nn import functional as F

from schnitz.agent import SchnitzelAgent
from schnitz.bank_coherence import verify_refresh_coverage
from schnitz.checkpoints import resolve_checkpoint, save_checkpoint
from schnitz.key_index import PublishedKeyIndex
from schnitz.keyspace_distillation import (LexicalField, StandardizedHead,
                                        convert_to_direct, field_kl, flat_positives,
                                        lexical_field_loss, recall_summary,
                                        source_disjoint_split, support_ranks,
                                        union_field_loss)
from schnitz.operations import atomic_json
from schnitz.optimizers import make_optimizer
from schnitz.store import DiskStore, StoredRecord
from schnitz.training import autocast_context, config_from_run
from schnitz.training_bank import TrainingBank
from schnitz.trajectories import file_sha256

LIMITS = (16, 8, 4, 4)


class Warmup:
    """Cached states, site metadata and the causal eligibility field."""

    def __init__(self, cache: Path, sources: Path, episodes: Path, *,
                 heldout_fraction: float, device: str) -> None:
        self.cache, self.device = cache, device
        self.manifest = json.loads((cache / 'manifest.json').read_text())
        if file_sha256(sources) != self.manifest['source_manifest_sha256']:
            raise ValueError('Sources differ from the state cache')
        self.ids = json.loads((cache / 'writer-ids.json').read_text())
        self.position = {record_id: index for index, record_id in enumerate(self.ids)}
        rows = {row['record_id']: row for row in map(json.loads, sources.open(encoding='utf-8'))}
        self.texts = [rows[record_id]['text'] for record_id in self.ids]
        domains = [rows[record_id].get('domain', 'research') for record_id in self.ids]
        times = [rows[record_id]['created_at'] for record_id in self.ids]
        self.sites = [json.loads(line) for line in (cache / 'query-sites.jsonl').open()]
        queries = {}
        for episode in map(json.loads, episodes.open(encoding='utf-8')):
            queries[episode['episode_id']] = episode['query']
        self.query_text = [queries[site['episode_id']] for site in self.sites]
        self.positives = [tuple(self.position[record_id] for record_id in site['required_ids'])
                          for site in self.sites]
        self.heldout = source_disjoint_split(
            [tuple(site['required_ids']) for site in self.sites],
            fraction=heldout_fraction, seed='keyspace-warmup-v1')
        # Causal eligibility: the site's domain and strictly earlier creation.
        self.domains = np.asarray(domains)
        self.times = np.asarray(times)
        self.site_domains = [site['domain'] for site in self.sites]
        self.site_times = [site['query_time'] for site in self.sites]
        self.lexical = LexicalField(self.texts)
        # One device mask per distinct causal scope; sites index into them.
        scopes = sorted(set(zip(self.site_domains, self.site_times)))
        self.scope_masks = torch.from_numpy(np.stack([
            (self.domains == domain) & (self.times < time) for domain, time in scopes
        ])).to(device)
        scope_index = {scope: index for index, scope in enumerate(scopes)}
        self.site_scope = torch.tensor([scope_index[scope] for scope in
                                        zip(self.site_domains, self.site_times)],
                                       dtype=torch.long, device=device)

    def eligible(self, rows: np.ndarray) -> torch.Tensor:
        return self.scope_masks[self.site_scope[torch.as_tensor(rows, device=self.device)]]

    def teacher(self, directory: Path, source_view: str, query_view: str):
        documents = load_file(str(directory / f'source-{source_view}.safetensors'))['embeddings']
        queries = load_file(str(directory / f'query-{query_view}.safetensors'))['embeddings']
        if documents.shape[0] != len(self.ids) or queries.shape[0] != len(self.sites):
            raise ValueError('Teacher rows differ from the state cache')
        return (documents.to(self.device, torch.float32),
                queries.to(self.device, torch.float32))


def _ranks_for(scores_fn, warmup: Warmup, rows: np.ndarray, batch: int = 256):
    ranks = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        eligible = warmup.eligible(chunk)
        scores = scores_fn(chunk)
        ranks.extend(support_ranks(scores, [warmup.positives[row] for row in chunk], eligible))
    return ranks


def _teacher_summary(ranks: list[tuple[int, ...]]) -> dict:
    count = len(ranks)
    best = sorted(min(row) for row in ranks)
    return {'sites': count,
            'recall_at_16': sum(min(row) <= 16 for row in ranks) / count,
            'recall_at_256': sum(min(row) <= 256 for row in ranks) / count,
            'all_support_recall_at_16': sum(max(row) <= 16 for row in ranks) / count,
            'median_best_rank': best[count // 2]}


def lexical_baseline(warmup: Warmup, rows: np.ndarray, query_texts: list[str]) -> dict:
    def scores(chunk):
        return torch.from_numpy(warmup.lexical.scores(
            [query_texts[row] for row in chunk])).to(warmup.device)
    return _teacher_summary(_ranks_for(scores, warmup, rows))


def teacher_baseline(warmup: Warmup, documents, queries, rows: np.ndarray) -> dict:
    return _teacher_summary(_ranks_for(lambda chunk: queries[chunk] @ documents.T,
                                       warmup, rows))


def fit_projection(warmup: Warmup, documents, queries, *, key_dim: int, steps: int,
                   seed: int) -> torch.nn.Linear:
    """Contrastive teacher-to-key projection on training-split sites only."""
    generator = torch.Generator().manual_seed(seed)
    projection = torch.nn.Linear(documents.shape[1], key_dim, bias=False)
    torch.nn.init.orthogonal_(projection.weight, generator=generator)
    projection = projection.to(warmup.device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
    train_rows = np.flatnonzero(~warmup.heldout)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        chunk = rng.choice(train_rows, size=min(512, len(train_rows)), replace=False)
        keys = F.normalize(projection(documents), dim=-1)
        address = F.normalize(projection(queries[chunk]), dim=-1)
        logits = (address @ keys.T / .05).masked_fill(
            ~warmup.eligible(chunk), torch.finfo(torch.float32).min)
        rows, columns = flat_positives([warmup.positives[site] for site in chunk],
                                       warmup.device)
        loss = -logits.log_softmax(-1)[rows, columns].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    projection.requires_grad_(False)
    return projection


def fit_temperature(warmup: Warmup, documents, queries, rows: np.ndarray) -> dict:
    """Calibrate on held-out training sites; temperature cannot change ranks."""
    scores = []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        scores.append((queries[chunk] @ documents.T).masked_fill(
            ~warmup.eligible(chunk), float('-inf')))
    best = None
    for temperature in np.geomspace(.005, .5, 25):
        losses = []
        for start, block in zip(range(0, len(rows), 256), scores, strict=True):
            chunk = rows[start:start + 256]
            positions = flat_positives([warmup.positives[site] for site in chunk],
                                       warmup.device)
            losses.extend((-(block / temperature).log_softmax(-1)[positions]).tolist())
        value = float(np.mean(losses))
        if best is None or value < best['heldout_support_nll']:
            best = {'temperature': float(temperature), 'heldout_support_nll': value}
    return best


def benchmark(args) -> dict:
    warmup = Warmup(args.cache, args.sources, args.episodes,
                    heldout_fraction=args.heldout_fraction, device=args.device)
    rows = np.flatnonzero(warmup.heldout)
    result = {'heldout_sites': int(len(rows)), 'training_sites': int((~warmup.heldout).sum()),
              'lexical_arguments': lexical_baseline(warmup, rows, warmup.query_text),
              'teachers': {}}
    for pair in args.pair:
        teacher, source_view, query_view = pair.split(':')
        documents, queries = warmup.teacher(args.teachers / teacher, source_view, query_view)
        result['teachers'][pair] = teacher_baseline(warmup, documents, queries, rows)
        print(json.dumps({pair: result['teachers'][pair]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    return result


def warmup_run(args) -> dict:
    if args.output.exists():
        raise ValueError('Warmup output must be fresh')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    began = time.perf_counter()

    def progress(stage, **values):
        print(json.dumps({'stage': stage, 'seconds': round(time.perf_counter() - began, 1)}
                         | values), flush=True)
    warmup = Warmup(args.cache, args.sources, args.episodes,
                    heldout_fraction=args.heldout_fraction, device=args.device)
    run = Path(warmup.manifest['run'])
    checkpoint = resolve_checkpoint(run, verify=True)
    if checkpoint.name != warmup.manifest['checkpoint'] or file_sha256(
            checkpoint / 'model.safetensors') != warmup.manifest['writer_checkpoint_sha256']:
        raise ValueError('State cache and parent checkpoint differ')
    config = config_from_run(run)
    spaces = len(config.memory.payload_dims)
    assignments = [pair.split(':') for pair in args.space]
    if len(assignments) != spaces:
        raise ValueError('Assign one teacher/source/query view per space')
    heldout = np.flatnonzero(warmup.heldout)
    train_rows = np.flatnonzero(~warmup.heldout)
    progress('loaded', heldout=int(len(heldout)), training=int(len(train_rows)))
    # Frozen teacher projections and temperatures (training split only).
    teachers, teacher_report = [], []
    for space, (teacher, source_view, query_view) in enumerate(assignments):
        documents, queries = warmup.teacher(args.teachers / teacher, source_view, query_view)
        raw = teacher_baseline(warmup, documents, queries, heldout)
        progress('teacher_baseline', space=space)
        projection = fit_projection(warmup, documents, queries, key_dim=config.memory.key_dim,
                                    steps=args.projection_steps, seed=args.seed + space)
        with torch.no_grad():
            projected_documents = F.normalize(projection(documents), dim=-1)
            projected_queries = F.normalize(projection(queries), dim=-1)
        progress('projection', space=space)
        projected = teacher_baseline(warmup, projected_documents, projected_queries, heldout)
        temperature = fit_temperature(warmup, projected_documents, projected_queries, heldout)
        teachers.append((projected_documents, projected_queries, temperature['temperature']))
        teacher_report.append({'space': f's{space}', 'teacher': teacher,
                               'source_view': source_view, 'query_view': query_view,
                               'raw_heldout': raw, 'projected_heldout': projected,
                               **temperature})
        print(json.dumps(teacher_report[-1]), flush=True)
        del documents, queries
    lexical = lexical_baseline(warmup, heldout, warmup.query_text)
    progress('lexical_baseline')

    # Direct heads folded from the parent's shared heads, trained in fp32 over
    # standardized cached states (an exact reparametrization at initialization).
    parent_state = load_file(str(checkpoint / 'model.safetensors'))
    direct_state = convert_to_direct(parent_state, spaces)
    cached = load_file(str(args.cache / 'writer.safetensors'))
    states = cached['states'].to(args.device).float()
    features = load_file(str(args.cache / 'queries.safetensors'))['features'].to(
        args.device).float()
    train_index = torch.as_tensor(train_rows, device=args.device)
    writer_mean, writer_scale = states.mean(0), states.std(0).clamp_min(1e-6)
    query_mean = features[train_index].mean(0)
    query_scale = features[train_index].std(0).clamp_min(1e-6)

    def head(kind, space, mean, scale):
        return StandardizedHead(
            direct_state[f'{kind}.{space}.weight'].float().to(args.device),
            direct_state[f'{kind}.{space}.bias'].float().to(args.device), mean, scale)
    writer_heads = [head('writer_key_heads', space, writer_mean, writer_scale)
                    for space in range(spaces)]
    query_heads = [head('query_key_heads', space, query_mean, query_scale)
                   for space in range(spaces)]
    writer_z = writer_heads[0].standardize(states)
    query_z = query_heads[0].standardize(features)
    del states
    # Actual-model equivalence at initialization. The parent's keys came from a
    # BF16 head over nearly collinear states, so compare directions, not ties.
    equivalence = []
    with torch.no_grad():
        for space in range(spaces):
            direct = F.normalize(writer_heads[space](writer_z), dim=-1)
            shared = cached[f'shared_key_s{space}'].to(args.device)
            cosine = (direct * shared).sum(-1)
            equivalence.append({'space': f's{space}',
                                'max_abs_key_difference': float((direct - shared).abs().max()),
                                'min_key_cosine': float(cosine.min())})
    print(json.dumps({'initial_equivalence': equivalence}), flush=True)
    del cached

    def student_logits(space, rows):
        keys = F.normalize(writer_heads[space](writer_z), dim=-1)
        address = F.normalize(query_heads[space](query_z[torch.as_tensor(rows,
                                                                          device=args.device)]),
                              dim=-1)
        return address @ keys.T

    def evaluate(rows):
        with torch.no_grad():
            space_ranks = [_ranks_for(lambda chunk, space=space: student_logits(space, chunk),
                                      warmup, rows) for space in range(spaces)]
        return recall_summary(space_ranks, LIMITS)

    def selection_score(summary):
        spaces_summary = summary['spaces']
        return (summary['union_any_support_recall']
                + sum(space['all_support_recall'] for space in spaces_summary)
                / len(spaces_summary))

    def snapshot():
        return [(item.weight.detach().clone(), item.bias.detach().clone())
                for item in writer_heads + query_heads]

    history = [{'step': 0, 'heldout': evaluate(heldout)}]
    print(json.dumps(history[-1]), flush=True)
    # Keep the best heads on held-out training sites (never the validation gate).
    best = {'step': 0, 'score': selection_score(history[0]['heldout']), 'heads': snapshot()}
    progress('initial_evaluation')
    # A 100k-record softmax needs a larger logit scale than sampled fields;
    # learn one per space (ranking and reader gates use raw cosine).
    log_scales = torch.full((spaces,), float(np.log(args.logit_scale)), device=args.device,
                            requires_grad=args.learn_logit_scale)
    parameters = [tensor for item in writer_heads + query_heads
                  for tensor in (item.weight, item.bias)]
    groups = [{'params': parameters, 'lr': args.learning_rate}]
    if args.learn_logit_scale:
        groups.append({'params': [log_scales],
                       'lr': 1e-2 if args.optimizer == 'adamw' else 1.0})
    optimizer = (torch.optim.AdamW(groups, weight_decay=0.0) if args.optimizer == 'adamw'
                 else torch.optim.SGD(groups, momentum=.9))

    def rate(step):
        warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
        return warm * .5 * (1 + np.cos(np.pi * step / args.steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        chunk = rng.choice(train_rows, size=args.batch_size, replace=False)
        eligible = warmup.eligible(chunk)
        lexical_scores = torch.from_numpy(warmup.lexical.scores(
            [warmup.query_text[row] for row in chunk])).to(args.device)
        scales = log_scales.clamp(max=float(np.log(100))).exp()
        logits = [student_logits(space, chunk) * scales[space] for space in range(spaces)]
        distill = torch.stack([
            field_kl(teachers[space][1][chunk] @ teachers[space][0].T / teachers[space][2],
                     logits[space], eligible) for space in range(spaces)]).mean()
        support = union_field_loss(logits, [warmup.positives[row] for row in chunk], eligible)
        lexical_loss = lexical_field_loss(logits, lexical_scores, eligible)
        loss = args.distill_weight * distill + args.support_weight * support \
            + args.lexical_weight * lexical_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.log_every == 0 or step == args.steps:
            history.append({'step': step, 'loss': float(loss.detach()),
                            'distill': float(distill.detach()),
                            'logit_scales': [round(float(value), 2) for value in
                                             log_scales.detach().exp()],
                            'support': float(support.detach()),
                            'lexical': float(lexical_loss.detach()),
                            'heldout': evaluate(heldout)})
            print(json.dumps(history[-1]), flush=True)
            score = selection_score(history[-1]['heldout'])
            if score > best['score']:
                best = {'step': step, 'score': score, 'heads': snapshot()}
    with torch.no_grad():
        for item, (weight, bias) in zip(writer_heads + query_heads, best['heads'],
                                        strict=True):
            item.weight.copy_(weight)
            item.bias.copy_(bias)
    progress('selected', step=best['step'], score=best['score'])
    if args.no_save:
        result = {'history': history, 'selected_step': best['step'],
                  'selection_score': best['score'], 'teachers': teacher_report,
                  'settings': {key: (str(value) if isinstance(value, Path) else value)
                               for key, value in vars(args).items() if key != 'func'}}
        args.output.mkdir(parents=True)
        atomic_json(args.output / 'sweep.json', result)
        return {'selected_step': best['step'], 'selection_score': best['score']}
    training_seconds = time.perf_counter() - began

    progress('distilled')
    # Save a direct-interface checkpoint and a complete coherent journal.
    args.output.mkdir(parents=True)
    for space in range(spaces):
        for kind, item in (('writer_key_heads', writer_heads[space]),
                           ('query_key_heads', query_heads[space])):
            weight, bias = item.folded()
            direct_state[f'{kind}.{space}.weight'] = weight.cpu()
            direct_state[f'{kind}.{space}.bias'] = bias.cpu()
    config.memory.key_interface = 'direct'
    agent = SchnitzelAgent(config)
    # load_model restores tied tensors that a raw safetensors state omits.
    missing, unexpected = load_model(agent, str(checkpoint / 'model.safetensors'),
                                     strict=False)
    heads = {f'{kind}.{space}.{part}' for kind in ('writer_key_heads', 'query_key_heads')
             for space in range(spaces) for part in ('weight', 'bias')}
    shared = {name for name in parent_state if name.startswith(
        ('key_head.', 'address_maps.', 'query_maps.', 'routing_query_head.'))}
    if set(missing) != heads or set(unexpected) != shared:
        raise ValueError(f'Unexpected direct conversion: {missing=}, {unexpected=}')
    with torch.no_grad():
        for name in heads:
            agent.get_parameter(name).copy_(direct_state[name])
    bank_manifest_path = args.bank / 'manifest.json'
    bank_manifest = json.loads(bank_manifest_path.read_text())
    base = DiskStore(args.bank / 'bank.sqlite')
    index = PublishedKeyIndex(base, namespace=bank_manifest['namespace'],
                              generation=bank_manifest['generation'],
                              spaces=tuple(bank_manifest['spaces']),
                              expected_sources=bank_manifest['sources'])
    parent_state_token = warmup.manifest['journal_state']
    parent_journal = DiskStore(run / 'training_cache.sqlite')
    if parent_journal.mutable_bank_state() != parent_state_token:
        raise ValueError('Parent journal moved beyond the cached checkpoint')
    journal_dir = args.output / 'journal'
    journal_dir.mkdir()
    journal_path = journal_dir / 'training_cache.sqlite'
    pending = journal_dir / 'training_cache.pending.sqlite'
    with parent_journal.connect() as source, sqlite3.connect(pending) as target:
        source.backup(target)
    os.replace(pending, journal_path)
    journal = DiskStore(journal_path)
    bank = TrainingBank(base, journal, index)
    storage = getattr(torch, config.memory.storage_dtype)
    with safe_open(str(args.cache / 'writer.safetensors'), framework='pt') as handle:
        payloads = [handle.get_tensor(f'payload_s{space}') for space in range(spaces)]
    step_token = json.loads((checkpoint / 'manifest.json').read_text())['step']
    raw_states = load_file(str(args.cache / 'writer.safetensors'))['states']
    agent_heads = agent.writer_key_heads.to(args.device)
    with torch.no_grad():
        for start in range(0, len(warmup.ids), args.refresh_chunk):
            stop = min(start + args.refresh_chunk, len(warmup.ids))
            chunk_states = raw_states[start:stop].to(args.device)
            keys = [F.normalize(agent._fp32_head(agent_heads[space], chunk_states, 'writer'),
                                dim=-1).cpu() for space in range(spaces)]
            records = [StoredRecord(warmup.ids[row], keys[space][row - start],
                                    payloads[space][row].to(storage),
                                    namespace=index.namespace, space=f's{space}',
                                    generation=index.generation)
                       for row in range(start, stop) for space in range(spaces)]
            bank.update(records, optimizer_step=step_token)
    agent = agent.cpu()
    progress('journal_written')
    verify_refresh_coverage(journal_path, namespace=index.namespace,
                            spaces=tuple(index.spaces), source_count=len(warmup.ids),
                            parent_cursor=parent_state_token['cursor'])
    optimizer_stub = make_optimizer(agent)
    fingerprint = file_sha256(args.cache / 'manifest.json')
    atomic_json(args.output / 'spatial-inputs.json', json.loads(
        (run / 'spatial-inputs.json').read_text()) | {'keyspace_warmup': True})
    saved = save_checkpoint(agent, optimizer_stub, args.output, 0, random.Random(args.seed),
                            journal, fingerprint, keep=2)
    model_sha = file_sha256(saved / 'model.safetensors')
    # The saved writer must reproduce journal keys without cached shortcuts.
    agent = agent.to(args.device).eval()
    probe_ids = [warmup.ids[index] for index in rng.choice(len(warmup.ids), 64, replace=False)]
    reproduction = _reproduce_keys(agent, config, bank_manifest, args.sources, probe_ids, index)
    manifest = {
        'format': 1, 'complete': True, 'key_interface': 'direct',
        'parent_run': str(args.output.resolve()), 'parent_checkpoint': saved.name,
        'writer_checkpoint_sha256': model_sha,
        'bank_manifest_sha256': file_sha256(bank_manifest_path),
        'source_manifest_sha256': warmup.manifest['source_manifest_sha256'],
        'parent_bank_state': parent_state_token, 'refreshed_sources': len(warmup.ids),
        'journal_state': journal.mutable_bank_state(), 'checkpoint_step': 0,
        'state_cache_manifest_sha256': fingerprint,
    }
    atomic_json(journal_dir / 'manifest.json', manifest)
    result = {
        'protocol': ('Head-only direct key warmup over cached frozen states; full eligible '
                     'field; teacher projections and temperatures fit on training split.'),
        'parent_checkpoint': str(checkpoint), 'state_cache': str(args.cache),
        'settings': {key: (str(value) if isinstance(value, Path) else value)
                     for key, value in vars(args).items() if key != 'func'},
        'heldout_sites': int(len(heldout)), 'training_sites': int(len(train_rows)),
        'teachers': teacher_report, 'lexical_heldout': lexical,
        'initial_equivalence': equivalence, 'history': history,
        'writer_reproduction': reproduction, 'journal_manifest': manifest,
        'selected_step': best['step'], 'selection_score': best['score'],
        'training_seconds': training_seconds, 'config': asdict(config),
    }
    atomic_json(args.output / 'warmup.json', result)
    return {key: value for key, value in result.items() if key not in {'history', 'config'}} | {
        'final_heldout': history[-1]['heldout']}


def _reproduce_keys(agent, config, bank_manifest, sources: Path, ids: list[str], index) -> dict:
    from schnitz.document_ingestion import (grouped_ingestion_prefixes,
                                         source_ingestion_groups, writer_prefix_ids)
    rows = [json.loads(line) for line in sources.open(encoding='utf-8')]
    source_rows = {row['record_id']: row for row in rows}
    groups = source_ingestion_groups(rows)
    tokens = []
    for record_id in ids:
        document_id, parts, mode = groups[record_id]
        messages = grouped_ingestion_prefixes(
            document_id, parts, generation=bank_manifest['generation'],
            scope={'domain': source_rows[record_id].get('domain', 'research')},
            mode=mode)[record_id]
        tokens.append(torch.tensor([writer_prefix_ids(agent.tokenizer, messages)],
                                   dtype=torch.long, device=agent.device))
    with torch.no_grad(), autocast_context(config):
        outputs = agent.produce_batch(tokens)
    worst = 0.0
    for space in range(len(config.memory.payload_dims)):
        stored = index.keys_for_ids(f's{space}', ids, domain='research', query_time=2 ** 62)
        worst = max(worst, float((outputs[2 * space].float().cpu() - stored.float()).abs().max()))
    return {'records': len(ids), 'max_abs_key_difference': worst}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, function in (('benchmark', benchmark), ('warmup', warmup_run)):
        command = commands.add_parser(name)
        command.set_defaults(func=function)
        command.add_argument('--cache', type=Path, required=True)
        command.add_argument('--sources', type=Path, required=True)
        command.add_argument('--episodes', type=Path, required=True)
        command.add_argument('--teachers', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--heldout-fraction', type=float, default=.08)
        command.add_argument('--device', default='cuda')
        command.add_argument('--seed', type=int, default=101)
    commands.choices['benchmark'].add_argument('--pair', action='append', required=True)
    warm = commands.choices['warmup']
    warm.add_argument('--bank', type=Path, required=True)
    warm.add_argument('--space', action='append', required=True,
                      help='teacher:source_view:query_view, once per space in order')
    warm.add_argument('--steps', type=int, default=3000)
    warm.add_argument('--batch-size', type=int, default=256)
    # Folded heads have RMS near 5e-3; keep Adam steps near 1% of that.
    warm.add_argument('--learning-rate', type=float, default=5e-5)
    warm.add_argument('--logit-scale', type=float, default=20.0)
    warm.add_argument('--learn-logit-scale', action=argparse.BooleanOptionalAction,
                      default=True)
    warm.add_argument('--optimizer', choices=('adamw', 'sgd'), default='adamw')
    warm.add_argument('--warmup-steps', type=int, default=0)
    warm.add_argument('--no-save', action='store_true',
                      help='tuning sweep: record history, skip journal and checkpoint')
    warm.add_argument('--distill-weight', type=float, default=1.0)
    warm.add_argument('--support-weight', type=float, default=1.0)
    warm.add_argument('--lexical-weight', type=float, default=.05)
    warm.add_argument('--projection-steps', type=int, default=400)
    warm.add_argument('--refresh-chunk', type=int, default=1024)
    warm.add_argument('--log-every', type=int, default=250)
    arguments = parser.parse_args()
    print(json.dumps(arguments.func(arguments), indent=2, default=str))
