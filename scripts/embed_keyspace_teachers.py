"""Embed declared source and causal query views with frozen local teachers.

Rows follow the state cache: sources in ``writer-ids.json`` order and queries in
``query-sites.jsonl`` order. Outputs are training-only teacher targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from schnitz.document_ingestion import source_ingestion_groups
from schnitz.keyspace_views import QUERY_VIEWS, SOURCE_VIEWS, query_view, source_view
from schnitz.operations import atomic_json
from schnitz.spatial_data import SpatialTrajectoryIndex
from schnitz.teacher_encoders import TEACHERS, TeacherEncoder
from schnitz.trajectories import file_sha256


def view_texts(cache: Path, sources: Path, episodes: Path, data: Path, *,
               source_views: set[str], query_views: set[str]) -> tuple[dict, dict]:
    rows = {row['record_id']: row for row in map(json.loads, sources.open(encoding='utf-8'))}
    groups = source_ingestion_groups(list(rows.values()))
    # The writer order is published before the query pass finishes.
    ids = json.loads((cache / 'writer-ids.json').read_text())
    if set(ids) != rows.keys():
        raise ValueError('Sources differ from the state cache')
    source_texts = {view: [source_view(view, record_id, rows, groups) for record_id in ids]
                    for view in source_views}
    if not query_views:
        return source_texts, {}
    manifest = json.loads((cache / 'manifest.json').read_text())
    if file_sha256(sources) != manifest['source_manifest_sha256']:
        raise ValueError('Sources differ from the state cache')
    trajectories = SpatialTrajectoryIndex(data)
    if trajectories.sha256 != manifest['data_sha256']:
        raise ValueError('Packed trajectories differ from the state cache')
    queries = {}
    for episode in map(json.loads, episodes.open(encoding='utf-8')):
        queries[episode['episode_id']] = episode['query']
    sites = [json.loads(line) for line in (cache / 'query-sites.jsonl').open()]
    tokenizer = None
    if query_views & {'prefix_window', 'short_window'}:
        config = json.loads((Path(manifest['run']) / 'config.json').read_text())['model']
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'])
    query_texts = {view: [] for view in query_views}
    loaded = {}
    for site in sites:
        index = site['trajectory_index']
        if index not in loaded:
            loaded = {index: trajectories[index]}
        row = loaded[index]
        for view in query_views:
            query_texts[view].append(query_view(
                view, queries[site['episode_id']], tokenizer=tokenizer,
                input_ids=row['input_ids'], query_position=site['query_position']))
    return source_texts, query_texts


def embed(cache: Path, sources: Path, episodes: Path, data: Path, output: Path,
          pairs: list[tuple[str, str, str]], *, batch_size: int = 128,
          sources_only: bool = False) -> dict:
    for teacher, source, query in pairs:
        if teacher not in TEACHERS or source not in SOURCE_VIEWS or query not in QUERY_VIEWS:
            raise ValueError(f'Unknown teacher/view pair: {teacher}:{source}:{query}')
    output.mkdir(parents=True, exist_ok=True)
    source_texts, query_texts = view_texts(
        cache, sources, episodes, data,
        source_views={source for _, source, _ in pairs},
        query_views=set() if sources_only else {query for _, _, query in pairs})
    summary = {}
    for teacher in dict.fromkeys(name for name, _, _ in pairs):
        wanted = [(source, query) for name, source, query in pairs if name == teacher]
        directory = output / teacher
        directory.mkdir(exist_ok=True)
        jobs = ([('source', view) for view in dict.fromkeys(s for s, _ in wanted)]
                + ([] if sources_only else
                   [('query', view) for view in dict.fromkeys(q for _, q in wanted)]))
        jobs = [(role, view) for role, view in jobs
                if not (directory / f'{role}-{view}.safetensors').exists()]
        if not jobs:
            continue
        encoder = TeacherEncoder(teacher)
        for role, view in jobs:
            began = time.perf_counter()
            texts = source_texts[view] if role == 'source' else query_texts[view]
            vectors = encoder.encode(texts, role='document' if role == 'source' else 'query',
                                     batch_size=batch_size)
            path = directory / f'{role}-{view}.safetensors'
            save_file({'embeddings': vectors.to(torch.float16).contiguous()}, str(path))
            summary[f'{teacher}/{role}-{view}'] = {
                'rows': len(texts), 'dimension': vectors.shape[1],
                'seconds': time.perf_counter() - began}
            print(json.dumps({f'{teacher}/{role}-{view}': summary[f'{teacher}/{role}-{view}']}),
                  flush=True)
        atomic_json(directory / 'teacher.json', {
            'teacher': teacher, 'model_id': encoder.spec.model_id,
            'revision': encoder.revision, 'pooling': encoder.spec.pooling,
            'query_prefix': encoder.spec.query_prefix,
            'document_prefix': encoder.spec.document_prefix,
            'writer_ids_sha256': file_sha256(cache / 'writer-ids.json')})
        del encoder
        torch.cuda.empty_cache()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair', action='append', required=True,
                        help='teacher:source_view:query_view')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(embed(args.cache, args.sources, args.episodes, args.data, args.output,
                           [tuple(pair.split(':')) for pair in args.pair],
                           batch_size=args.batch_size,
                           sources_only=args.sources_only), indent=2))
