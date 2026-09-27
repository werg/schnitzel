"""Unassisted packed ``memory.search`` gate against a coherent stored-key journal.

Each site reads only its own search results (no supplied supports), exactly as
inference would; later levels therefore see the checkpoint's own earlier reads.
Ranks are exact over each site's causally eligible stored keys. No source is
re-encoded: the writer is disabled during evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from safetensors.torch import load_model
import torch
from torch.nn import functional as F

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.key_index import PublishedKeyIndex
from schnitz.operations import atomic_json
from schnitz.recurrence import SpatialReadSite
from schnitz.routing import cosine_similarities
from schnitz.spatial_data import SpatialTrajectoryIndex, validate_spatial_row
from schnitz.store import DiskStore, ReadPlan, Selection
from schnitz.training import autocast_context, config_from_run
from schnitz.training_bank import TrainingBank
from schnitz.trajectories import file_sha256


def evaluate(run: Path, bank_dir: Path, data_path: Path, output: Path, *,
             journal: Path | None, limits: tuple[int, ...], batch_size: int = 4) -> dict:
    if output.exists():
        raise ValueError('Gate output must be fresh')
    checkpoint = resolve_checkpoint(run, verify=True)
    config = config_from_run(run)
    manifest = json.loads((bank_dir / 'manifest.json').read_text())
    base = DiskStore(bank_dir / 'bank.sqlite')
    index = PublishedKeyIndex(base, namespace=manifest['namespace'],
                              generation=manifest['generation'],
                              spaces=tuple(manifest['spaces']),
                              expected_sources=manifest['sources'])
    if journal is None:
        journal = run / 'training_cache.sqlite'
        expected_state = json.loads((checkpoint / 'bank-state.json').read_text())
    else:
        staged = json.loads((journal.parent / 'manifest.json').read_text())
        if (not staged.get('complete') or staged['writer_checkpoint_sha256']
                != file_sha256(checkpoint / 'model.safetensors')):
            raise ValueError('Journal was not written by the evaluated checkpoint')
        expected_state = staged['journal_state']
    mutable = DiskStore(journal)
    if mutable.mutable_bank_state() != expected_state:
        raise ValueError('Journal differs from the evaluated checkpoint state')
    bank = TrainingBank(base, mutable, index)
    agent = SchnitzelAgent(config).to(config.train.device)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    agent.eval()

    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError('Gate evaluation must not re-encode sources')
    agent.produce = agent.produce_batch = forbidden_writer
    data = SpatialTrajectoryIndex(data_path)
    site_rows = []

    for start in range(0, len(data), batch_size):
        rows = [data[i] for i in range(start, min(start + batch_size, len(data)))]
        for row in rows:
            validate_spatial_row(row)
        maximum = max(len(row['input_ids']) for row in rows)
        input_ids = torch.full((len(rows), maximum), agent.tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=agent.device)
        attention = torch.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, :len(row['input_ids'])] = torch.tensor(
                row['input_ids'], device=agent.device)
            attention[row_index, :len(row['input_ids'])] = 1
        levels = tuple(site['level'] for site in rows[0]['sites'])
        sites = tuple(SpatialReadSite(
            torch.tensor([row['sites'][s]['query_position'] for row in rows], device=agent.device),
            torch.tensor([row['sites'][s]['workspace_start'] for row in rows], device=agent.device),
            levels[s]) for s in range(len(levels)))

        def provider(level, active, query, routing_query):
            indices = [s for s, value in enumerate(levels) if value == level]
            metadata = [rows[r]['sites'][s] for s in indices for r in range(len(rows))]
            payloads, weights = [], []
            per_site = [{'call_id': item['call_id'], 'episode_id': item['episode_id'],
                         'level': level, 'required_ids': item['required_ids'],
                         'selected': [], 'support_ranks': []} for item in metadata]
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
                    # Ineligible rows sort last, so found is a prefix of candidates.
                    found = [str(array.ids[i]) for i in order[:limit] if eligible[i]]
                    ranks = []
                    for record_id in item['required_ids']:
                        at = int(np.flatnonzero(array.ids == record_id)[0])
                        ranks.append(int((field > field[at]).sum()) + 1)
                    per_site[row_index]['selected'].append(found)
                    per_site[row_index]['support_ranks'].append(ranks)
                    chosen_plans.append(ReadPlan(index.namespace, name, index.generation,
                                                 item['domain'], item['query_time'],
                                                 tuple(Selection(record_id, 0.0)
                                                       for record_id in found)))
                    candidate_ids = [str(array.ids[i]) for i in order[:max(limit, 256)]
                                     if eligible[i]]
                    candidate_rows.append(candidate_ids)
                values = bank.fetch_many(chosen_plans)
                count = max(map(len, values))
                device_rows, row_weights = [], []
                for row_index, (value, candidates) in enumerate(zip(values, candidate_rows,
                                                                    strict=True)):
                    stacked = torch.stack([v.to(agent.device, torch.float32) for v in value])
                    device_rows.append(F.pad(stacked, (0, 0, 0, count - stacked.shape[0])))
                    if config.memory.distance_gating:
                        keys = index.keys_for_ids(
                            name, candidates, domain=metadata[row_index]['domain'],
                            query_time=metadata[row_index]['query_time']).to(agent.device)
                        raw = cosine_similarities(address[row_index:row_index + 1], keys)[0]
                        chosen = raw[:len(value)][None]
                        local, _ = agent.distance_gates[space](
                            address[row_index:row_index + 1], chosen, raw[None],
                            torch.ones_like(chosen, dtype=torch.bool),
                            torch.ones_like(raw[None], dtype=torch.bool))
                        weight = local[0]
                    else:
                        weight = query.new_ones(len(value))
                    row_weights.append(torch.cat((weight, query.new_zeros(count - len(value)))))
                payloads.append(torch.stack(device_rows))
                weights.append(torch.stack(row_weights))
            site_rows.extend(per_site)
            return agent._read_padded_batch(payloads, weights, query)

        with torch.no_grad(), autocast_context(config):
            agent.spatial_recurrent_hidden(input_ids, attention, sites, provider)

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
    by_level = {}
    for level in sorted({row['level'] for row in site_rows}):
        chosen = [row for row in site_rows if row['level'] == level]
        by_level[str(level)] = sum(any(min(row['support_ranks'][space]) <= limit
                                       for space, limit in enumerate(limits))
                                   for row in chosen) / len(chosen)
    result = {
        'protocol': ('Unassisted packed memory.search; exact eligible stored-key ranks; '
                     'no supplied supports; writer disabled.'),
        'run': str(run), 'checkpoint': checkpoint.name,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'key_interface': config.memory.key_interface,
        'journal': str(journal), 'journal_state': expected_state,
        'data_sha256': data.sha256, 'sites': len(site_rows), 'limits': list(limits),
        'spaces': spaces, 'union_any_support_recall': union,
        'union_any_support_recall_by_level': by_level,
        'random_any_support_expectation_s0': limits[0] / manifest['sources'],
    }
    output.mkdir(parents=True)
    atomic_json(output / 'gate.json', result)
    with (output / 'sites.jsonl').open('w', encoding='utf-8') as handle:
        for row in site_rows:
            handle.write(json.dumps(row) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--limits', nargs='+', type=int, default=[16, 8, 4, 4])
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.bank, args.data, args.output,
                              journal=args.journal, limits=tuple(args.limits),
                              batch_size=args.batch_size), indent=2))
