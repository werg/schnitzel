"""Cache every bank record's writer key-slot state for live key recomputation.

Runs the bank's exact writer forward-only over its source manifest, in index
(record-ID) order, and stores one fp32 key-slot state per record. The bank's
stored keys must equal the writer's direct key heads applied to these states;
the check re-derives keys for a sample and compares them with storage.
"""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import random
import time

from safetensors.torch import load_model, save_file
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.document_ingestion import (grouped_ingestion_prefixes, source_ingestion_groups,
                                     writer_prefix_ids)
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import writer_prompt_generation
from schnitz.operations import atomic_json
from schnitz.store import DiskStore
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def build(run: Path, bank: Path, sources: Path, *, batch_size: int = 32,
          check: int = 512) -> dict:
    manifest = json.loads((bank / 'manifest.json').read_text())
    checkpoint = resolve_checkpoint(run, verify=True)
    writer_sha = file_sha256(checkpoint / 'model.safetensors')
    if writer_sha != manifest['identity']['writer_checkpoint_sha256']:
        raise ValueError('Key states must come from the exact writer that built the bank')
    if file_sha256(sources) != manifest['identity']['source_manifest_sha256']:
        raise ValueError('Source manifest differs from bank creation')
    output = bank / f'key-states-{writer_sha[:12]}.safetensors'
    if output.exists():
        raise FileExistsError(output)
    config = config_from_run(run)
    agent = SchnitzelAgent(config).to(config.train.device)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    agent.eval()
    rows = {}
    with sources.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            rows[row['record_id']] = row
    store = DiskStore(bank / 'bank.sqlite')
    index = PublishedKeyIndex(store, namespace=manifest['namespace'],
                              generation=manifest['generation'],
                              spaces=tuple(manifest['spaces']),
                              expected_sources=manifest['sources'])
    ids = [str(record_id) for record_id in next(iter(index.spaces.values())).ids]
    groups = source_ingestion_groups(list(rows.values()))
    prompt_generation = writer_prompt_generation(manifest)

    @lru_cache(maxsize=256)
    def prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(document_id, parts, generation=prompt_generation,
                                          scope={'domain': domain}, mode=mode)

    def writer_ids(record_id):
        row = rows[record_id]
        document_id, parts, mode = groups[record_id]
        messages = prefixes(document_id, parts, mode, row.get('domain', 'research'))[record_id]
        return torch.tensor([writer_prefix_ids(agent.tokenizer, messages)],
                            dtype=torch.long, device=agent.device)

    states = torch.empty(len(ids), agent.width)
    order = sorted(range(len(ids)), key=lambda i: len(rows[ids[i]]['text']))
    began = time.perf_counter()
    with torch.no_grad(), autocast_context(config):
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            outputs = agent.produce_batch([writer_ids(ids[i]) for i in batch],
                                          with_key_state=True)
            states[torch.tensor(batch)] = outputs[-1].float().cpu()
            if start // batch_size % 200 == 0:
                print(json.dumps({'encoded': start + len(batch), 'of': len(ids),
                                  'seconds': round(time.perf_counter() - began)}), flush=True)
    sample = sorted(random.Random(1701).sample(range(len(ids)), min(check, len(ids))))
    positions = torch.tensor(sample)
    with torch.no_grad():
        derived = agent.writer_space_keys(states[positions].to(agent.device))
    cosines = []
    for space, (name, array) in enumerate(index.spaces.items()):
        stored = torch.from_numpy(array.keys[sample])
        cos = torch.nn.functional.cosine_similarity(derived[space].float().cpu(), stored, dim=-1)
        cosines.append({'space': name, 'min': float(cos.min()), 'median': float(cos.median())})
    save_file({'states': states.contiguous(),
               'versions': torch.zeros(len(ids), dtype=torch.long)}, str(output),
              metadata={'ids_sha256': hashlib.sha256(
                  '\n'.join(ids).encode()).hexdigest(),
                  'writer_checkpoint_sha256': writer_sha,
                  'bank_generation': manifest['generation']})
    report = {'output': str(output), 'records': len(ids), 'writer_checkpoint_sha256': writer_sha,
              'bank_generation': manifest['generation'], 'key_check': cosines,
              'seconds': time.perf_counter() - began}
    if min(item['min'] for item in cosines) < 0.999:
        report['warning'] = 'Cached states do not reproduce stored keys'
    atomic_json(bank / f'key-states-{writer_sha[:12]}.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check', type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.bank, args.sources, batch_size=args.batch_size,
                           check=args.check), indent=2))
