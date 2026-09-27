"""Remove neural query identity from oracle-supplied identifier examples."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from schnitz.data import evidence_ids, load_episodes, save_episodes
from schnitz.episode_index import EpisodeIndex
from schnitz.trajectories import file_sha256


QUERY = 'What endpoint is specified by the selected earlier record? Return only the endpoint.'


def fixed_query_episode(episode):
    if episode.task_family != 'multiuse/identifier':
        return episode
    required = evidence_ids(episode, 'required')
    sources = [s for s in episode.supports if s.record_id in required]
    if (len(required) != 1 or len(sources) != 1 or sources[0].kind != 'permission'
            or re.fullmatch(r'api_[0-9a-f]{6}', episode.answer) is None
            or f'Its endpoint is {episode.answer}.' not in sources[0].text):
        raise ValueError('Identifier target must belong to the single prior selected source')
    digest = hashlib.sha256(QUERY.encode()).hexdigest()
    episode_id = hashlib.sha256(f'{episode.episode_id}:fixed-query:{digest}'.encode()).hexdigest()[:16]
    provenance = episode.provenance | {
        'fixed_query_parent_episode_id': episode.episode_id,
        'original_query_sha256': hashlib.sha256(episode.query.encode()).hexdigest(),
        'fixed_query_sha256': digest,
        'fixed_query_scope': 'Oracle-supplied record; constant query cannot identify a global retrieval target.',
    }
    return replace(episode, episode_id=episode_id, query=QUERY, provenance=provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError('Use new output and manifest paths')
    episodes = [fixed_query_episode(e) for e in load_episodes(args.input)]
    identifiers = [e for e in episodes if e.task_family == 'multiuse/identifier']
    if not identifiers:
        raise ValueError('No identifier examples transformed')
    save_episodes(args.output, episodes)
    EpisodeIndex(args.output)  # Validate source identity consistency and episode IDs.
    args.manifest.write_text(json.dumps({
        'input_sha256': file_sha256(args.input), 'output_sha256': file_sha256(args.output),
        'script_sha256': file_sha256(__file__), 'episodes': len(episodes),
        'identifier_queries': len(identifiers), 'distinct_identifier_prompts': len({e.query for e in identifiers}),
        'distinct_identifier_targets': len({e.answer for e in identifiers}),
        'query': QUERY, 'notice': 'Source identities/contents/times and all targets are unchanged. '
                                'Only identifier query text, episode identity and provenance change. '
                                'Oracle communication diagnostic, not learned retrieval.',
    }, indent=2) + '\n')
