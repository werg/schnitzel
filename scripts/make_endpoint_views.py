"""Alternate synthetic prior histories: identical queries, versioned endpoint memories."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from schnitz.data import load_episodes, save_episodes
from schnitz.operations import atomic_json, run_lock
from schnitz.trajectories import file_sha256


def opaque(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def endpoint(value, version):
    return 'api_'+hashlib.sha256(f'endpoint-view:{version}:{value}'.encode()).hexdigest()[:6]


def endpoint_view(episode, version):
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError('Nonnegative integer view required')
    if not episode.task_family.startswith('multiuse/'):
        raise ValueError('Generated multiuse episodes required')
    if any(s.created_at >= episode.query_time for s in episode.supports):
        raise ValueError('Future source in endpoint view')
    original_sources = {s.record_id: s for s in episode.supports}
    if len(original_sources) != len(episode.supports) or not set(episode.required_ids) <= original_sources.keys():
        raise ValueError('Ambiguous or missing required sources')
    if episode.task_family == 'multiuse/identifier':
        if (len(episode.required_ids) != 1 or
                f'Its endpoint is {episode.answer}.' not in original_sources[episode.required_ids[0]].text):
            raise ValueError('Endpoint target does not belong to its required source')
    if version == 0:
        return episode
    mapping = {s.record_id: opaque(f'endpoint-view:{version}:source:{s.record_id}') for s in episode.supports}
    sources, names = [], {}
    for source in episode.supports:
        text = source.text
        if source.kind == 'permission':
            matches = re.findall(r'(?<=Its endpoint is )api_[0-9a-f]{6}(?=\.)', text)
            if len(matches) != 1:
                raise ValueError('Malformed generated endpoint source')
            old = matches[0]
            new = endpoint(source.record_id, version)
            if new == old:
                raise ValueError('Endpoint hash collision; choose another declared view scheme')
            names[old] = new
            text = text.replace(f'Its endpoint is {old}.', f'Its endpoint is {new}.')
        sources.append(replace(source, record_id=mapping[source.record_id], text=text))
    identifier = episode.task_family == 'multiuse/identifier'
    if identifier and episode.answer not in names:
        raise ValueError('Endpoint target does not occur in its source world')
    return replace(episode, episode_id=opaque(f'endpoint-view:{version}:query:{episode.episode_id}'),
        supports=tuple(sources), required_ids=tuple(mapping[rid] for rid in episode.required_ids),
        sufficient_groups=tuple(tuple(mapping[rid] for rid in group) for group in episode.sufficient_groups),
        answer=names[episode.answer] if identifier else episode.answer,
        choices=tuple(names.get(c, endpoint(f'{episode.environment}:{c}', version)) for c in episode.choices)
                if identifier else episode.choices)


def make_views(episodes, versions):
    if versions < 2:
        raise ValueError('At least two histories per query required')
    worlds = list(dict.fromkeys(e.environment for e in episodes))
    result, provenance, source_values = [], {}, {}
    for world in worlds:
        group = [e for e in episodes if e.environment == world]
        for version in range(versions):
            for e in group:
                changed = endpoint_view(e, version)
                result.append(changed)
                for original, source in zip(e.supports, changed.supports, strict=True):
                    if source_values.setdefault(source.record_id, source) != source:
                        raise ValueError('Conflicting immutable source versions')
                    lineage = {'base_source_id': original.record_id, 'view': version,
                               'text_sha256': hashlib.sha256(source.text.encode()).hexdigest()}
                    if provenance.setdefault(source.record_id, lineage) != lineage:
                        raise ValueError('Ambiguous source-version provenance')
    if len({e.episode_id for e in result}) != len(result):
        raise ValueError('Duplicate episode versions')
    for e in episodes:
        if e.task_family == 'multiuse/identifier':
            answers = {endpoint_view(e, version).answer for version in range(versions)}
            if len(answers) != versions:
                raise ValueError('Endpoint views must have distinct targets per unchanged query')
    return result, provenance


def _run(source, output, versions):
    episodes, provenance = make_views(load_episodes(source), versions)
    identity = {'source_sha256': file_sha256(source), 'versions': versions, 'script_sha256': file_sha256(__file__),
                'queries': len(episodes), 'query_worlds': len({e.environment for e in episodes}),
                'source_versions': len(provenance),
                'notice': 'Alternate prior histories for oracle latent training. Identical queries have different supplied memories; '
                          'this corpus is not a learned-global-routing benchmark or one chronological live namespace.'}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'inputs.json').exists():
        old = json.loads((output/'inputs.json').read_text())
        if (any(old[k] != v for k, v in identity.items()) or
                old['episodes_sha256'] != file_sha256(output/'train.jsonl') or
                old['provenance_sha256'] != file_sha256(output/'source-versions.json')):
            raise ValueError('Existing endpoint-view corpus differs')
        return
    if any(output.iterdir()):
        raise ValueError('Incomplete output exists; inspect it and use an explicit new destination')
    save_episodes(output/'train.jsonl', episodes)
    atomic_json(output/'source-versions.json', provenance)
    identity.update(episodes_sha256=file_sha256(output/'train.jsonl'),
                    provenance_sha256=file_sha256(output/'source-versions.json'))
    atomic_json(output/'inputs.json', identity)


def run(source, output, versions=8):
    with run_lock(output, clear_stop=False):
        _run(source, output, versions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--versions', type=int, default=8)
    args = parser.parse_args()
    run(args.source, args.output, args.versions)
