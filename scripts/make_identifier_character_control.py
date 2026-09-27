"""Pair repeated whole-endpoint supervision with source-matched character questions."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from schnitz.data import evidence_ids, load_episodes, save_episodes
from schnitz.episode_index import EpisodeIndex
from schnitz.trajectories import file_sha256


FULL_QUERY = 'What endpoint is specified by the selected earlier record? Return only the endpoint.'
ORDINALS = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth')


def validate_identifier(episode):
    if episode.task_family != 'multiuse/identifier' or re.fullmatch(r'api_[0-9a-f]{6}', episode.answer) is None:
        raise ValueError('Expected a generated six-hex identifier episode')
    required = evidence_ids(episode, 'required')
    selected = [s for s in episode.supports if s.record_id in required]
    if (len(required) != 1 or len(selected) != 1 or selected[0].kind != 'permission' or
            re.findall(r'(?<=Its endpoint is )api_[0-9a-f]{6}(?=\.)', selected[0].text) != [episode.answer]):
        raise ValueError('Identifier target must be the actual endpoint in the selected source')


def version(episode, query, role, *, position=None):
    identity = f'{episode.episode_id}:endpoint-supervision:{role}:{position}:{query}'
    return {'episode_id': hashlib.sha256(identity.encode()).hexdigest()[:16], 'query': query,
            'provenance': episode.provenance | {'endpoint_supervision_parent': episode.episode_id,
                'endpoint_supervision_role': role, 'endpoint_character_position': position,
                'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
                'scope': 'Oracle-selected prior source; query has no global retrieval identity.'}}


def full_episode(episode, replica=None):
    validate_identifier(episode)
    return replace(episode, **version(episode, FULL_QUERY, 'whole', position=replica))


def character_episode(episode, position):
    if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= 6:
        raise ValueError('Character position must be an integer from 1 to 6')
    validate_identifier(episode)
    query = (f'What is the {ORDINALS[position-1]} character of the six-character code after api_ '
             'in the endpoint specified by the selected earlier record? Count from the left. '
             'Return only that one hexadecimal character.')
    return replace(episode, **version(episode, query, 'character', position=position),
                   answer=episode.answer[4+position-1], choices=tuple('0123456789abcdef'),
                   task_family='identifier_character')


def paired_episodes(episodes):
    full, characters = [], []
    for episode in episodes:
        if episode.task_family != 'multiuse/identifier':
            full.append(episode)
            characters.append(episode)
            continue
        base = full_episode(episode)
        full.append(base)
        characters.append(base)
        for position in range(1, 7):
            full.append(full_episode(episode, replica=position))
            characters.append(character_episode(episode, position))
    return full, characters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--full-output', type=Path, required=True)
    parser.add_argument('--character-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    original = load_episodes(args.input)
    full, characters = paired_episodes(original)
    for path, episodes in [(args.full_output, full), (args.character_output, characters)]:
        if path.exists():
            raise FileExistsError(path)
        save_episodes(path, episodes)
        EpisodeIndex(path)
    args.manifest.write_text(json.dumps({'input_sha256': file_sha256(args.input),
        'full_sha256': file_sha256(args.full_output), 'character_sha256': file_sha256(args.character_output),
        'script_sha256': file_sha256(__file__), 'episodes_per_arm': len(full),
        'character_queries': sum(e.task_family == 'identifier_character' for e in characters),
        'notice': 'Matched source selection/order and original nonidentifier tasks. Six whole-answer repeats '
                  'versus six single-character queries per endpoint; target/token budgets differ. '
                  'New opaque query IDs; immutable sources, prior times and original field identities retained.'},
        indent=2)+'\n')
