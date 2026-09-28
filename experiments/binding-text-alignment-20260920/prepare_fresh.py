"""Seal a fresh confirmation corpus before inspecting either alignment endpoint."""
import argparse
import hashlib
from pathlib import Path
import re
import runpy
import shutil

from sdkb.data import make_multiuse_world, save_episodes
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def identities(episodes):
    ids, texts, endpoints = set(), set(), set()
    for episode in episodes:
        for source in episode.supports:
            if source.record_id in ids:
                continue
            ids.add(source.record_id)
            texts.add(hashlib.sha256(source.text.encode()).hexdigest())
            endpoints.update(re.findall(r'api_[0-9a-f]{6}', source.text))
    inverted = {'api_'+''.join(format(int(c, 16)^15, 'x') for c in s[4:]) for s in endpoints}
    return ids, texts, endpoints | inverted


def prepare(output, exclusions, *, worlds=32, split='alignment-fresh-confirmation-20260920'):
    if worlds < 1:
        raise ValueError('Positive world count required')
    if shutil.disk_usage(output.parent).free < 10*1024**3:
        raise OSError('Fresh-corpus preparation requires 10 GiB disk reserve')
    if output.exists():
        raise FileExistsError(output)
    full_episode = runpy.run_path(str(Path(__file__).parents[2]/'scripts/make_identifier_character_control.py'))['full_episode']
    blocked = (set(), set(), set())
    excluded = []
    for path in exclusions:
        episodes = EpisodeIndex(path)
        found = identities(episodes)
        for target, values in zip(blocked, found, strict=True):
            target.update(values)
        excluded.append(dict(path=str(path), sha256=episodes.sha256, episodes=len(episodes)))
    accepted, rejected, rows = [], [], []
    seed = 0
    while len(accepted) < worlds:
        episodes = make_multiuse_world(seed, split=split, bindings=2)
        found = identities(episodes)
        if any(a & b for a, b in zip(blocked, found, strict=True)):
            rejected.append(seed)
        else:
            accepted.append(seed)
            rows.extend(full_episode(e) if e.task_family == 'multiuse/identifier' else e for e in episodes)
            for target, values in zip(blocked, found, strict=True):
                target.update(values)
        seed += 1
    output.mkdir()
    path = output/'episodes.jsonl'
    save_episodes(path, rows)
    atomic_json(output/'declaration.json', dict(
        split=split, worlds=worlds, episodes=len(rows), accepted_seeds=accepted, rejected_seeds=rejected,
        episodes_path=str(path), episodes_sha256=file_sha256(path), exclusions=excluded,
        generator_sha256=file_sha256(__file__),
        # the package was renamed sdkb -> schnitz (27 Sep 2026); same data generator
        data_generator_sha256=file_sha256(next(p for p in (Path(__file__).parents[2]/'src/sdkb/data.py',
                                                        Path(__file__).parents[2]/'src/schnitz/data.py')
                                               if p.exists())),
        question_generator_sha256=file_sha256(Path(__file__).parents[2]/'scripts/make_identifier_character_control.py'),
        policy='No model results used in corpus construction. Both original and inverted endpoints, source IDs and exact source-text hashes are disjoint from the explicitly listed corpora and other accepted worlds. This is not pretrained-model decontamination.',
        evaluation_gate=dict(min_original_identifiers=8, min_correct_identifier_pairs=4,
                             min_original_actions=126, inspected_split_identifier_count=64,
                             inspected_split_action_count=128),
        gate_policy='Run both arms on this fresh corpus only if alignment meets every declared gate on the inspected primary split. Otherwise retain the sealed corpus unused. Gates are an exploratory compute decision, not significance tests or acceptance criteria.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    prepare(args.output, args.exclude)
