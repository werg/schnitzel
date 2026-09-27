"""Embed keyed teacher views of sources and queries for keyspace pretraining.

Unlike ``embed_keyspace_teachers.py``, rows are keyed by record or episode ID,
so any trajectory packing or source subset can look them up. Queries use their
visible ``memory.search`` arguments only, never answers or later observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import save_file
import torch

from schnitz.document_ingestion import source_ingestion_groups
from schnitz.keyspace_views import query_view, source_view
from schnitz.operations import atomic_json
from schnitz.teacher_encoders import TEACHERS, TeacherEncoder
from schnitz.trajectories import file_sha256


def _load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            rows.extend(map(json.loads, handle))
    return rows


def embed(sources: list[Path], queries: list[Path], output: Path,
          spaces: list[tuple[str, str, str]], *, batch_size: int = 128) -> dict:
    source_rows = {}
    for row in _load(sources):
        if source_rows.setdefault(row['record_id'], row) is not row:
            raise ValueError('Duplicate source identity across source files')
    groups = source_ingestion_groups(list(source_rows.values()))
    source_ids = sorted(source_rows)
    query_rows = {}
    for row in _load(queries):
        query_rows.setdefault(row['episode_id'], row['query'])
    query_ids = sorted(query_rows)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'ids.json', {'sources': source_ids, 'queries': query_ids})
    summary = {}
    for teacher in dict.fromkeys(name for name, _, _ in spaces):
        if teacher not in TEACHERS:
            raise ValueError(f'Unknown teacher: {teacher}')
        views = [(source, query) for name, source, query in spaces if name == teacher]
        directory = output / teacher
        directory.mkdir(exist_ok=True)
        jobs = ([('source', view) for view in dict.fromkeys(s for s, _ in views)]
                + [('query', view) for view in dict.fromkeys(q for _, q in views)])
        jobs = [job for job in jobs
                if not (directory / f'{job[0]}-{job[1]}.safetensors').exists()]
        if not jobs:
            continue
        encoder = TeacherEncoder(teacher)
        for role, view in jobs:
            began = time.perf_counter()
            if role == 'source':
                texts = [source_view(view, record_id, source_rows, groups)
                         for record_id in source_ids]
            else:
                if view not in {'arguments', 'content'}:
                    raise ValueError('Keyed query targets use argument views only')
                texts = [query_view(view, query_rows[episode_id]) for episode_id in query_ids]
            vectors = encoder.encode(texts, role='document' if role == 'source' else 'query',
                                     batch_size=batch_size)
            save_file({'embeddings': vectors.to(torch.float16).contiguous()},
                      str(directory / f'{role}-{view}.safetensors'))
            summary[f'{teacher}/{role}-{view}'] = {'rows': len(texts),
                                                   'seconds': time.perf_counter() - began}
            print(json.dumps({f'{teacher}/{role}-{view}': summary[f'{teacher}/{role}-{view}']}),
                  flush=True)
        atomic_json(directory / 'teacher.json', {
            'teacher': teacher, 'model_id': encoder.spec.model_id,
            'revision': encoder.revision, 'ids_sha256': file_sha256(output / 'ids.json')})
        del encoder
        torch.cuda.empty_cache()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--queries', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--space', action='append', required=True,
                        help='teacher:source_view:query_view, once per space')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(embed(args.sources, args.queries, args.output,
                           [tuple(item.split(':')) for item in args.space],
                           batch_size=args.batch_size), indent=2))
