"""Add prior, distinct-article distractors to verified located-source episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from schnitz.data import Source, episode_from_dict
from schnitz.trajectories import file_sha256


def prepare(episodes_file: Path, sources_file: Path, output: Path, *,
            distractors: int = 3, require_external: bool = True,
            candidate_source_limit: int | None = None) -> dict:
    if (distractors < 1 or (candidate_source_limit is not None and candidate_source_limit < 1)
            or output.exists() or not output.parent.is_dir()):
        raise ValueError('Positive distractor count and fresh output parent required')
    if (require_external and
            output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev):
        raise ValueError('Routing data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Routing preparation requires a 10 GiB disk reserve')
    rows = [json.loads(line) for line in sources_file.open()]
    by_id = {row['record_id']: row for row in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError('Distinct, nonempty source manifest required')
    if candidate_source_limit is not None and candidate_source_limit > len(rows):
        raise ValueError('Candidate source limit exceeds the manifest')
    candidate_ids = tuple(sorted(row['record_id'] for row in
                                 (rows if candidate_source_limit is None
                                  else rows[:candidate_source_limit])))
    prepared = []
    for line in episodes_file.open():
        episode = episode_from_dict(json.loads(line))
        located = {('passage_qa', 'source_article_title'),
                   ('located_passage_span', 'article-and-passage-prefix-v1')}
        if ((episode.task_family, episode.provenance.get('query_locator')) not in located
                or episode.support_annotation != 'verified'
                or len(episode.required_ids) != 1
                or len(episode.supports) != 1):
            raise ValueError('Expected one verified title- or content-located support')
        required = episode.required_ids[0]
        if required not in by_id or episode.supports[0].record_id != required:
            raise ValueError('Required source differs from the source manifest')
        title = by_id[required]['provenance']['article_title']
        start = int(hashlib.sha256(f'local-routing:{episode.episode_id}'.encode()).hexdigest()[:16], 16)
        selected = []
        selected_titles = {title}
        for index in range(len(candidate_ids)):
            candidate_id = candidate_ids[(start + index) % len(candidate_ids)]
            candidate = by_id[candidate_id]
            candidate_title = candidate['provenance']['article_title']
            if candidate_id != required and candidate_title not in selected_titles:
                selected.append(candidate_id)
                selected_titles.add(candidate_title)
                if len(selected) == distractors:
                    break
        if len(selected) != distractors:
            raise ValueError('Not enough distinct-article distractors')
        extras = tuple(Source(**{key: by_id[source_id][key] for key in
                                 ('record_id', 'text', 'created_at', 'kind')})
                       for source_id in selected)
        if any(source.created_at >= episode.query_time for source in extras):
            raise ValueError('Distractor source violates the causal query boundary')
        prepared.append(asdict(replace(episode, supports=episode.supports + extras,
            provenance=episode.provenance | {'distractor_ids': selected,
                                              'distractor_policy': 'distinct_article_sha256_v1'})))
    if not prepared:
        raise ValueError('At least one routing episode required')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        path = pending / 'episodes.jsonl'
        with path.open('w') as handle:
            for episode in prepared:
                handle.write(json.dumps(episode, sort_keys=True, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = {'format': 1, 'episodes': len(prepared), 'distractors': distractors,
                  'candidate_source_limit': candidate_source_limit,
                  'candidate_ids_sha256': hashlib.sha256(json.dumps(candidate_ids).encode()).hexdigest(),
                  'input_episodes_sha256': file_sha256(episodes_file),
                  'source_manifest_sha256': file_sha256(sources_file),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'episodes_sha256': file_sha256(path),
                  'notice': 'Verified source remains earlier than its question; distinct-article distractors are candidates, not verified anti-evidence.'}
        (pending / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distractors', type=int, default=3)
    parser.add_argument('--candidate-source-limit', type=int,
                        help='Use only the first N source rows as distractor candidates')
    parser.add_argument('--allow-same-device', action='store_true',
                        help='For platforms without a separate artifact disk')
    args = parser.parse_args()
    print(json.dumps(prepare(args.episodes, args.sources, args.output,
                             distractors=args.distractors,
                             require_external=not args.allow_same_device,
                             candidate_source_limit=args.candidate_source_limit), indent=2))
