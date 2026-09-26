"""Held-out evaluation of a key-table checkpoint on the full bank.

Each site reads only its own unassisted search results (no supplied supports),
as inference would, over the whole bank (no curriculum restriction). The bank
state is exactly the checkpoint's: a consistent snapshot of the run's journal is
rolled back to the checkpoint cursor, and search uses the checkpoint's key table.

Key modes:

``table``
    Stored keys are the trained key table. Measures whether the key space
    generalizes to unseen queries.
``decoder``
    The evaluated gold records are written anew by the checkpoint's frozen
    decoder (an offline write, not re-encoding at inference): their keys are the
    decoder's predicted keys and their payloads the decoder's outputs. Measures
    whether newly written documents would be found and read. All other records
    keep their table keys and journal payloads.

Reports per-space unassisted recall and support ranks, and the answer NLL on the
trajectories' supervised tokens.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile

import numpy as np
from safetensors.torch import load_model
import torch
from torch.nn import functional as F

from sdkb.agent import SDKBAgent
from sdkb.document_ingestion import (grouped_ingestion_prefixes, source_ingestion_groups,
                                     writer_prefix_ids)
from sdkb.key_index import PublishedKeyIndex
from sdkb.offline_bank import writer_prompt_generation
from sdkb.operations import atomic_json
from sdkb.recurrence import SpatialReadSite
from sdkb.routing import cosine_similarities
from sdkb.spatial_data import SpatialTrajectoryIndex, validate_spatial_row
from sdkb.store import DiskStore, ReadPlan, Selection
from sdkb.training import autocast_context, config_from_run
from sdkb.training_bank import TrainingBank
from sdkb.trajectories import file_sha256


def _checkpoint(run: Path, step: int | None) -> Path:
    candidates = sorted(p for p in (run / 'checkpoints').glob('step-*') if p.is_dir())
    if step is not None:
        candidates = [p for p in candidates
                      if json.loads((p / 'manifest.json').read_text())['step'] == step]
    if not candidates:
        raise FileNotFoundError('No matching checkpoint')
    return candidates[-1]


def evaluate(run: Path, bank_dir: Path, data_path: Path, output: Path, *,
             step: int | None, keys: str, limits: tuple[int, ...], sources: Path | None,
             batch_size: int = 4, max_rows: int | None = None,
             scratch: Path | None = None) -> dict:
    if output.exists():
        raise ValueError('Evaluation output must be fresh')
    if keys not in {'table', 'decoder'} or (keys == 'decoder' and sources is None):
        raise ValueError('Decoder keys need the source manifest')
    checkpoint = _checkpoint(run, step)
    step = json.loads((checkpoint / 'manifest.json').read_text())['step']
    record_state = torch.load(run / f'record-state-{step:09d}.pt', weights_only=False)
    config = config_from_run(run)
    manifest = json.loads((bank_dir / 'manifest.json').read_text())
    base = DiskStore(bank_dir / 'bank.sqlite')
    index = PublishedKeyIndex(base, namespace=manifest['namespace'],
                              generation=manifest['generation'],
                              spaces=tuple(manifest['spaces']),
                              expected_sources=manifest['sources'])
    bank_state = json.loads((checkpoint / 'bank-state.json').read_text())

    # Exact checkpoint bank: snapshot the live journal, then roll it back.
    workdir = Path(tempfile.mkdtemp(prefix='keytable-eval-', dir=scratch))
    snapshot = workdir / 'journal.sqlite'
    with sqlite3.connect(f"file:{run / 'training_cache.sqlite'}?mode=ro", uri=True) as src, \
            sqlite3.connect(snapshot) as dst:
        src.backup(dst)
    journal = DiskStore(snapshot, secure_delete=False)
    bank = TrainingBank(base, journal, index)
    bank.rollback(int(bank_state['cursor']))

    # The checkpoint's key table replaces stored keys for every record.
    table = record_state['table']
    for space, array in enumerate(index.spaces.values()):
        if len(array.ids) != len(table['keys'][space]):
            raise ValueError('Key table and bank differ')
        array.keys[:] = table['keys'][space].float().numpy()
    index.invalidate()

    agent = SDKBAgent(config).to(config.train.device)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    agent.eval()
    data = SpatialTrajectoryIndex(data_path)
    count = len(data) if max_rows is None else min(len(data), max_rows)
    gold = sorted({record_id for i in range(count) for site in data[i]['sites']
                   for record_id in site['required_ids']})

    overlay: dict[tuple[str, str], torch.Tensor] = {}
    agreement = []
    if keys == 'decoder':
        rows = {}
        with sources.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                rows[row['record_id']] = row
        groups = source_ingestion_groups(list(rows.values()))
        generation = writer_prompt_generation(manifest)

        @lru_cache(maxsize=4096)
        def prefixes(document_id, parts, mode, domain):
            return grouped_ingestion_prefixes(document_id, parts, generation=generation,
                                              scope={'domain': domain}, mode=mode)

        def writer_ids(record_id):
            document_id, parts, mode = groups[record_id]
            messages = prefixes(document_id, parts, mode,
                                rows[record_id].get('domain', 'research'))[record_id]
            return torch.tensor([writer_prefix_ids(agent.tokenizer, messages)],
                                dtype=torch.long, device=agent.device)

        dtype = getattr(torch, config.memory.storage_dtype)
        with torch.no_grad(), autocast_context(config):
            for start in range(0, len(gold), 16):
                batch = gold[start:start + 16]
                outputs = agent.produce_batch([writer_ids(r) for r in batch],
                                              with_key_state=True)
                predicted = agent.writer_space_keys(outputs[-1])
                for space, (name, array) in enumerate(index.spaces.items()):
                    positions = [int(np.searchsorted(array.ids, r)) for r in batch]
                    new = F.normalize(predicted[space].float(), dim=-1).cpu()
                    old = torch.from_numpy(array.keys[positions].copy())
                    agreement.append(F.cosine_similarity(new, old, dim=-1))
                    array.keys[positions] = new.numpy()
                    for row, record_id in enumerate(batch):
                        overlay[(name, record_id)] = (
                            outputs[2 * space + 1][row].to(dtype).float().cpu())
        index.invalidate()

    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError('Evaluation must not encode sources while reading')
    agent.produce = agent.produce_batch = forbidden_writer
    site_rows = []
    nll_sum, nll_tokens = 0.0, 0

    for start in range(0, count, batch_size):
        rows_batch = [data[i] for i in range(start, min(start + batch_size, count))]
        for row in rows_batch:
            validate_spatial_row(row)
        maximum = max(len(row['input_ids']) for row in rows_batch)
        input_ids = torch.full((len(rows_batch), maximum), agent.tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=agent.device)
        attention = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row_index, row in enumerate(rows_batch):
            length = len(row['input_ids'])
            input_ids[row_index, :length] = torch.tensor(row['input_ids'], device=agent.device)
            attention[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(row['labels'], device=agent.device)
        levels = tuple(site['level'] for site in rows_batch[0]['sites'])
        sites = tuple(SpatialReadSite(
            torch.tensor([row['sites'][s]['query_position'] for row in rows_batch],
                         device=agent.device),
            torch.tensor([row['sites'][s]['workspace_start'] for row in rows_batch],
                         device=agent.device),
            levels[s]) for s in range(len(levels)))

        def provider(level, active, query, routing_query):
            indices = [s for s, value in enumerate(levels) if value == level]
            metadata = [rows_batch[r]['sites'][s] for s in indices
                        for r in range(len(rows_batch))]
            payloads, weights = [], []
            per_site = [{'call_id': item['call_id'], 'episode_id': item['episode_id'],
                         'level': level, 'required_ids': item['required_ids'],
                         'support_ranks': []} for item in metadata]
            for space, limit in enumerate(limits):
                name = f's{space}'
                array = index.spaces[name]
                address = agent.routing_address(routing_query, space)
                q = F.normalize(address.detach().float(), dim=-1).cpu().numpy()
                scores = q @ array.keys.T
                chosen_plans, candidate_rows = [], []
                for row_index, item in enumerate(metadata):
                    eligible = ((array.domains == item['domain'])
                                & (array.times < item['query_time']) & ~array.deleted)
                    field = np.where(eligible, scores[row_index], -np.inf)
                    order = np.lexsort((array.ids, -field))
                    found = [str(array.ids[i]) for i in order[:limit] if eligible[i]]
                    ranks = []
                    for record_id in item['required_ids']:
                        at = int(np.searchsorted(array.ids, record_id))
                        ranks.append(int((field > field[at]).sum()) + 1)
                    per_site[row_index]['support_ranks'].append(ranks)
                    chosen_plans.append(ReadPlan(index.namespace, name, index.generation,
                                                 item['domain'], item['query_time'],
                                                 tuple(Selection(r, 0.0) for r in found)))
                    candidate_rows.append([str(array.ids[i]) for i in order[:max(limit, 256)]
                                           if eligible[i]])
                values = bank.fetch_many(chosen_plans)
                values = [[overlay.get((name, selection.record_id), value)
                           for selection, value in zip(plan.selections, row, strict=True)]
                          for plan, row in zip(chosen_plans, values, strict=True)]
                width = max(map(len, values))
                device_rows, row_weights = [], []
                for row_index, (value, candidates) in enumerate(zip(values, candidate_rows,
                                                                    strict=True)):
                    stacked = torch.stack([v.to(agent.device, torch.float32) for v in value])
                    device_rows.append(F.pad(stacked, (0, 0, 0, width - stacked.shape[0])))
                    if config.memory.distance_gating:
                        candidate_keys = index.keys_for_ids(
                            name, candidates, domain=metadata[row_index]['domain'],
                            query_time=metadata[row_index]['query_time']).to(agent.device)
                        raw = cosine_similarities(address[row_index:row_index + 1],
                                                  candidate_keys)[0]
                        chosen = raw[:len(value)][None]
                        local, _ = agent.distance_gates[space](
                            address[row_index:row_index + 1], chosen, raw[None],
                            torch.ones_like(chosen, dtype=torch.bool),
                            torch.ones_like(raw[None], dtype=torch.bool))
                        weight = local[0]
                    else:
                        weight = query.new_ones(len(value))
                    row_weights.append(torch.cat((weight, query.new_zeros(width - len(value)))))
                payloads.append(torch.stack(device_rows))
                weights.append(torch.stack(row_weights))
            site_rows.extend(per_site)
            return agent._read_padded_batch(payloads, weights, query)

        with torch.no_grad(), autocast_context(config):
            hidden = agent.spatial_recurrent_hidden(input_ids, attention, sites, provider)
            supervised = labels >= 0
            logits = agent.backbone.logits(hidden[supervised]).float()
            nll_sum += float(F.cross_entropy(logits, labels[supervised], reduction='sum'))
            nll_tokens += int(supervised.sum())

    spaces = []
    for space, limit in enumerate(limits):
        best = [min(row['support_ranks'][space]) for row in site_rows]
        worst = [max(row['support_ranks'][space]) for row in site_rows]
        spaces.append({
            'space': f's{space}', 'limit': limit,
            'any_support_recall': sum(rank <= limit for rank in best) / len(best),
            'all_support_recall': sum(rank <= limit for rank in worst) / len(worst),
            'recall_at_256': sum(rank <= 256 for rank in best) / len(best),
            'median_best_rank': statistics.median(best),
        })
    union = sum(any(min(row['support_ranks'][space]) <= limit
                    for space, limit in enumerate(limits)) for row in site_rows) / len(site_rows)
    result = {
        'protocol': ('Unassisted packed memory.search over the full bank at the checkpoint '
                     'journal cursor; exact eligible ranks; no supplied supports; '
                     f'keys={keys}.'),
        'run': str(run), 'checkpoint': checkpoint.name, 'step': step,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'bank_cursor': bank_state['cursor'], 'data': str(data_path),
        'data_sha256': data.sha256, 'rows': count, 'sites': len(site_rows),
        'gold_records': len(gold), 'limits': list(limits), 'keys': keys,
        'spaces': spaces, 'union_any_support_recall': union,
        'answer_nll': nll_sum / max(nll_tokens, 1), 'answer_tokens': nll_tokens,
    }
    if agreement:
        agreement = torch.cat(agreement)
        result['decoder_table_key_cosine'] = {'mean': float(agreement.mean()),
                                              'min': float(agreement.min())}
    output.mkdir(parents=True)
    atomic_json(output / 'eval.json', result)
    with (output / 'sites.jsonl').open('w', encoding='utf-8') as handle:
        for row in site_rows:
            handle.write(json.dumps(row) + '\n')
    for path in workdir.glob('*'):
        path.unlink()
    workdir.rmdir()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--keys', choices=('table', 'decoder'), default='table')
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--limits', nargs='+', type=int, default=[64, 32, 16, 8])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.bank, args.data, args.output, step=args.step,
                              keys=args.keys, limits=tuple(args.limits),
                              sources=args.sources, batch_size=args.batch_size,
                              max_rows=args.max_rows, scratch=args.scratch), indent=2))
