"""Fixed dense lexical keys: a stored-byte control for learned global addressing."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import runpy

import numpy as np

from schnitz.archiving import ensure_free
from schnitz.checkpoints import _atomic_text, stop_on_signal
from schnitz.data import load_episodes
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.trajectories import file_sha256


LEXICAL = runpy.run_path(str(Path(__file__).with_name('evaluate_lexical_routing.py')))


def encode(counts, *, width, seed):
    if (not isinstance(width, int) or isinstance(width, bool) or width < 1
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('Positive integer width and nonnegative integer seed required')
    value = np.zeros(width, dtype=np.float32)
    for token, count in sorted(counts.items()):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError('Positive integer token counts required')
        bits = hashlib.shake_256(f'{seed}:{token}'.encode()).digest((width+7)//8)
        signs = 1-2*np.unpackbits(np.frombuffer(bits, dtype=np.uint8))[:width].astype(np.float32)
        value += (1+math.log(count))*signs
    norm = np.linalg.norm(value)
    return value/norm if norm else value


def build(records, output, *, width, seed):
    """Offline construction; only dense keys and visibility metadata are published."""
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        metadata, keys, settings = reopen(output)
        expected = [{k: v for k, v in row.items() if k != 'terms'} for row in records]
        if metadata != expected or settings != {'width': width, 'seed': seed}:
            raise ValueError('Existing dense index identity differs')
        for row, key in zip(records, keys, strict=True):
            if not np.array_equal(encode(row['terms'], width=width, seed=seed), key):
                raise ValueError('Existing dense key differs from its source')
        return
    keys = np.stack([encode(row['terms'], width=width, seed=seed) for row in records])
    metadata = [{k: v for k, v in row.items() if k != 'terms'} for row in records]
    buffer = io.BytesIO()
    np.save(buffer, keys, allow_pickle=False)
    pending = output/'keys.npy.pending'
    # The manifest is published last. A partial directory is never a valid index.
    import os
    with pending.open('wb') as file:
        file.write(buffer.getvalue())
        file.flush()
        os.fsync(file.fileno())
    pending.replace(output/'keys.npy')
    atomic_json(output/'records.json', metadata)
    atomic_json(output/'manifest.json', {'settings': {'width': width, 'seed': seed},
        'sha256': {name: file_sha256(output/name) for name in ('keys.npy', 'records.json')}})


def reopen(output):
    manifest = json.loads((output/'manifest.json').read_text())
    for name in ('keys.npy', 'records.json'):
        if file_sha256(output/name) != manifest['sha256'][name]:
            raise ValueError('Dense index digest differs')
    records = json.loads((output/'records.json').read_text())
    keys = np.load(output/'keys.npy', allow_pickle=False)
    settings = manifest['settings']
    if (keys.shape != (len(records), settings['width']) or keys.dtype != np.float32
            or not np.isfinite(keys).all() or any('terms' in r for r in records)):
        raise ValueError('Invalid persisted dense keys or metadata')
    return records, keys, settings


def eligible(record, query_time, namespace, space, generation, domain):
    return ((record['namespace'], record['space'], record['generation'], record['domain'])
            == (namespace, space, generation, domain) and record['created_at'] < query_time)


def rank_dense(records, keys, query, *, width, seed, query_time, top_k,
               namespace='global', space='s0', generation='frozen-v1', domain='research'):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    vector = encode(LEXICAL['terms'](query), width=width, seed=seed)
    if not np.any(vector):
        return []
    indices = [i for i, r in enumerate(records) if eligible(r, query_time, namespace, space, generation, domain)]
    if not indices:
        return []
    scores = keys[indices] @ vector
    return sorted([{'record_id': records[i]['record_id'], 'score': float(score)}
                   for i, score in zip(indices, scores, strict=True)],
                  key=lambda row: (-row['score'], row['record_id']))[:top_k]


def rank_sparse_tf(records, query, *, query_time, top_k):
    """Unprojected TF baseline isolates projection loss from the earlier IDF change."""
    counts = LEXICAL['terms'](query)
    q = {t: 1+math.log(n) for t, n in counts.items()}
    norm = math.sqrt(sum(x*x for x in q.values()))
    if not norm:
        return []
    scores = []
    for record in records:
        if not eligible(record, query_time, 'global', 's0', 'frozen-v1', 'research'):
            continue
        v = {t: 1+math.log(n) for t, n in record['terms'].items()}
        vnorm = math.sqrt(sum(x*x for x in v.values()))
        score = sum(q.get(t, 0)*x for t, x in v.items())/(norm*vnorm) if vnorm else 0.
        scores.append({'record_id': record['record_id'], 'score': score})
    return sorted(scores, key=lambda row: (-row['score'], row['record_id']))[:top_k]


def run(bank, corpus, reference, output, widths, seeds):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        if stop_requested(output):
            raise RuntimeError('Dense lexical evaluation stop requested')
        prior = json.loads(reference.read_text())
        identity = {'bank_sha256': file_sha256(bank), 'episodes_sha256': file_sha256(corpus),
                    'reference_sha256': file_sha256(reference), 'script_sha256': file_sha256(__file__),
                    'lexical_helper_sha256': file_sha256(Path(__file__).with_name('evaluate_lexical_routing.py')),
                    'widths': widths, 'seeds': seeds}
        for name in ('bank_sha256', 'episodes_sha256'):
            if prior['inputs'][name] != identity[name]:
                raise ValueError('Reference bank/corpus differs')
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Dense lexical protocol changed')
        ensure_free(output, 10*1024**3)
        atomic_json(output/'inputs.json', identity)
        index = LEXICAL['build_index'](bank, corpus)
        learned = {r['episode']: r for r in prior['rows'] if r['condition'] == 'all'}
        episodes = [e for e in load_episodes(corpus) if e.episode_id in learned]
        if {e.episode_id for e in episodes} != learned.keys():
            raise ValueError('Reference query set differs')
        jobs = [(None, None)] + [(width, seed) for width in widths for seed in seeds]
        for width, seed in jobs:
            if stop['signal'] is not None or stop_requested(output):
                raise RuntimeError('Dense lexical evaluation stopped; completed arms remain reusable')
            label = 'sparse-tf' if width is None else f'width-{width}-seed-{seed}'
            destination = output/(label+'.json')
            if destination.exists():
                if json.loads(destination.read_text())['inputs'] != identity:
                    raise ValueError('Completed dense lexical identity differs')
                continue
            resources = {}
            if width is not None:
                path = output/label
                ensure_free(output, 10*1024**3)
                build(index['records'], path, width=width, seed=seed)
                # Ranking consumes only reopened keys/metadata plus causal query text.
                records, keys, settings = reopen(path)
                resources = {'key_blob_bytes': keys.nbytes,
                             'serialized_index_bytes': sum(p.stat().st_size for p in path.iterdir()),
                             'index_manifest_sha256': file_sha256(path/'manifest.json')}
            rows = []
            for episode in episodes:
                if stop['signal'] is not None or stop_requested(output):
                    raise RuntimeError('Dense lexical evaluation stopped; completed arms remain reusable')
                old = learned[episode.episode_id]
                if (old['environment'], old['task_family'], old['answer']) != (
                        episode.environment, episode.task_family, episode.answer):
                    raise ValueError('Reference query identity differs')
                ranked = (rank_sparse_tf(index['records'], episode.query, query_time=episode.query_time, top_k=2)
                          if width is None else rank_dense(records, keys, episode.query, **settings,
                                                           query_time=episode.query_time, top_k=2))
                for count in (1, 2):
                    selected = [r['record_id'] for r in ranked[:count]]
                    rows.append({'episode': episode.episode_id, 'environment': episode.environment,
                        'task_family': episode.task_family, 'top_k': count, 'selected_ids': selected,
                        **LEXICAL['support_metrics'](episode, selected)})
            summary = defaultdict(dict)
            for family in sorted({r['task_family'] for r in rows}):
                for count in (1, 2):
                    group = [r for r in rows if r['task_family'] == family and r['top_k'] == count]
                    summary[family][str(count)] = {'n': len(group),
                        'complete_support': sum(r['complete_support'] for r in group),
                        'all_required': sum(r['all_required'] for r in group)}
            _atomic_text(destination, json.dumps({'inputs': identity, 'width': width, 'seed': seed,
                **resources, 'summary': summary, 'rows': rows,
                'notice': 'Known literal-name fixture; no generation, learned-key improvement or capacity claim. '
                          'Dense keys use fixed source/query token hashing, without IDF or a runtime token index. '
                          'Equal key bytes do not equal learned encoder architecture or compute.'}, indent=2)+'\n')
            print(json.dumps({'completed': label, **resources, 'summary': summary}), flush=True)
        if file_sha256(bank) != identity['bank_sha256'] or file_sha256(corpus) != identity['episodes_sha256']:
            raise ValueError('Frozen bank or corpus changed during evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'corpus', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--widths', type=int, nargs='+', default=[64, 128, 256, 1024])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 47])
    args = parser.parse_args()
    run(args.bank, args.corpus, args.reference, args.output, args.widths, args.seeds)
