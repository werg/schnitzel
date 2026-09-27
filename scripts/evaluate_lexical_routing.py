"""Offline lexical address control; not the learned single-vector SCHNITZELJAGD router."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from schnitz.data import load_episodes
from schnitz.trajectories import file_sha256


def terms(text):
    """Treat every source command as inert text; no field/identifier-specific rules."""
    return dict(Counter(re.findall(r'[a-z0-9_]+', text.lower())))


def build_index(bank, corpus):
    sources = {}
    for episode in load_episodes(corpus):
        for source in episode.supports:
            if source.record_id in sources and sources[source.record_id] != source:
                raise ValueError('Changed immutable source version')
            sources[source.record_id] = source
    records = []
    with sqlite3.connect('file:'+quote(str(bank.resolve()))+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT record_id,source_id,namespace,space,generation,domain,created_at '
                              'FROM records WHERE deleted=0 ORDER BY namespace,space,generation,record_id'):
            value = dict(row)
            source = sources.get(value['source_id'])
            if source is None:
                raise ValueError('Bank source is absent from the sealed corpus')
            if value['created_at'] != source.created_at:
                raise ValueError('Bank source time differs from the corpus')
            value['source_text_sha256'] = hashlib.sha256(source.text.encode()).hexdigest()
            value['terms'] = terms(source.text)
            records.append(value)
    return {'inputs': {'bank_sha256': file_sha256(bank), 'corpus_sha256': file_sha256(corpus),
                       'script_sha256': file_sha256(__file__)}, 'records': records,
            'notice': 'Source-derived sparse address features, created offline. No query targets or required IDs. '
                      'This retains lexical content and has a different information/byte budget from learned keys.'}


def rank(records, query, *, query_time, top_k, namespace='global', space='s0',
         generation='frozen-v1', domain='research'):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    eligible = [r for r in records if (r['namespace'], r['space'], r['generation'], r['domain'])
                == (namespace, space, generation, domain) and r['created_at'] < query_time]
    # Invisible records must not even influence IDF or normalization.
    df = Counter(token for record in eligible for token in record['terms'])
    idf = {token: 1+math.log((1+len(eligible))/(1+count)) for token, count in df.items()}
    q = {token: (1+math.log(count))*idf[token] for token, count in terms(query).items() if token in idf}
    qnorm = math.sqrt(sum(value*value for value in q.values()))
    if not qnorm:
        return []
    scored = []
    for record in eligible:
        vector = {token: (1+math.log(count))*idf[token] for token, count in record['terms'].items()}
        norm = math.sqrt(sum(value*value for value in vector.values()))
        dot = sum(value*vector.get(token, 0) for token, value in q.items())
        if norm and dot:
            scored.append({'record_id': record['record_id'], 'score': dot/(qnorm*norm)})
    return sorted(scored, key=lambda r: (-r['score'], r['record_id']))[:top_k]


def support_metrics(episode, selected):
    ids = set(selected)
    groups = episode.sufficient_groups or (episode.required_ids,)
    return {'complete_support': any(set(group) <= ids for group in groups),
            'all_required': set(episode.required_ids) <= ids}


def run(bank, corpus, reference, output):
    from schnitz.archiving import ensure_free
    from schnitz.checkpoints import _atomic_text
    from schnitz.operations import run_lock, stop_requested
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        if stop_requested(output):
            raise RuntimeError('Lexical diagnostic stop requested')
        prior = json.loads(reference.read_text())
        bank_hash, corpus_hash = file_sha256(bank), file_sha256(corpus)
        if prior['inputs']['bank_sha256'] != bank_hash or prior['inputs']['episodes_sha256'] != corpus_hash:
            raise ValueError('Learned reference bank/corpus differs')
        index = build_index(bank, corpus)
        index_path = output/'index.json'
        serialized = json.dumps(index, sort_keys=True, separators=(',', ':'))+'\n'
        if index_path.exists() and index_path.read_text() != serialized:
            raise ValueError('Existing lexical index differs; use a fresh output')
        ensure_free(output, 10*1024**3)
        if not index_path.exists():
            _atomic_text(index_path, serialized)
        # Cross an explicit persisted boundary. Ranking uses no source text or annotations.
        records = json.loads(index_path.read_text())['records']
        learned = {r['episode']: r for r in prior['rows'] if r['condition'] == 'all'}
        rows = []
        for episode in load_episodes(corpus):
            if episode.episode_id not in learned:
                continue
            if stop_requested(output):
                raise RuntimeError('Lexical diagnostic stopped; index remains reusable')
            old = learned[episode.episode_id]
            if (old['environment'], old['task_family'], old['answer']) != (
                    episode.environment, episode.task_family, episode.answer):
                raise ValueError('Learned reference query identity differs')
            for count in (1, 2):
                selected = rank(records, episode.query, query_time=episode.query_time, top_k=count)
                ids = [r['record_id'] for r in selected]
                rows.append({'episode': episode.episode_id, 'environment': episode.environment,
                             'task_family': episode.task_family, 'top_k': count, 'selected_ids': ids,
                             'scores': [r['score'] for r in selected],
                             **support_metrics(episode, ids),
                             'selected_count': len(ids), 'required_count': len(episode.required_ids),
                             'learned_selected_ids': old['selected_ids'],
                             **{'learned_'+key: value for key, value in
                                support_metrics(episode, old['selected_ids']).items()}})
        if {r['episode'] for r in rows} != learned.keys():
            raise ValueError('Reference query set was not reproduced')
        summary = defaultdict(dict)
        for family in sorted({r['task_family'] for r in rows}):
            for count in (1, 2):
                group = [r for r in rows if r['task_family'] == family and r['top_k'] == count]
                summary[family][str(count)] = {'queries': len(group),
                    'complete_support': sum(r['complete_support'] for r in group),
                    'all_required': sum(r['all_required'] for r in group),
                    'learned_complete_support': sum(r['learned_complete_support'] for r in group),
                    'learned_all_required': sum(r['learned_all_required'] for r in group),
                    'learned_count_histogram': dict(Counter(len(r['learned_selected_ids']) for r in group))}
        if file_sha256(bank) != bank_hash or file_sha256(corpus) != corpus_hash:
            raise ValueError('Frozen bank or corpus changed during diagnostic')
        result = {'inputs': index['inputs'] | {'reference_sha256': file_sha256(reference),
                    'index_sha256': file_sha256(index_path)}, 'candidate_records': len(records),
                  'serialized_index_bytes': index_path.stat().st_size, 'summary': summary, 'rows': rows,
                  'notice': 'Known-corpus retrieval-only control using eligible-source TF-IDF cosine. Fixed top-one '
                            'and top-two budgets; learned policy may use a different count. Source lexical features '
                            'are an additional index, not a learned 64-dimensional key or equal-byte comparison. '
                            'No latent inference, generation, capability or ANN/latency claim.'}
        ensure_free(output, 10*1024**3)
        _atomic_text(output/'results.json', json.dumps(result, indent=2)+'\n')
        print(json.dumps({'records': len(records), 'summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'corpus', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.bank, args.corpus, args.reference, args.output)
