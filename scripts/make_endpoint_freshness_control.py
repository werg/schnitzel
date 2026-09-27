"""Create slot-major endpoint-vocabulary controls with matched sampler schedules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import runpy
import shutil

from schnitz.data import make_multiuse_world, save_episodes
from schnitz.episode_index import EpisodeIndex
from schnitz.trajectories import file_sha256


def build(worlds, split):
    if not isinstance(worlds, int) or isinstance(worlds, bool) or worlds < 1 or worlds & (worlds-1):
        raise ValueError('World count must be a positive power of two')
    pair = runpy.run_path(str(Path(__file__).with_name('make_identifier_character_control.py')))['paired_episodes']
    by_world = [pair(make_multiuse_world(i, split=split, bindings=2))[0] for i in range(worlds)]
    if any(len(rows) != 22 for rows in by_world):
        raise ValueError('Expected 22 matched task slots per world')
    # Integer multiples of power-of-two world counts preserve slot selection for
    # the same random.getrandbits stream. Verify the actual schedule, not only this layout.
    return [by_world[world][slot] for slot in range(22) for world in range(worlds)]


def sampling_plan(episodes, *, seed, steps, accumulation):
    rng = random.Random(seed)
    families, targets = Counter(), Counter()
    schedule = hashlib.sha256()
    for _step in range(steps):
        depth = rng.choice([2, 3])
        for _micro in range(accumulation):
            episode = rng.choice(episodes)
            families[episode.task_family] += 1
            schedule.update(json.dumps([depth, episode.task_family, len(episode.supports),
                                        len(episode.required_ids)], separators=(',', ':')).encode()+b'\n')
            if episode.task_family == 'multiuse/identifier':
                targets[episode.answer] += 1
            for _source in episode.supports:
                rng.random()  # Existing trainer's live/cached draw at live_fraction=1.
    return {'schedule_sha256': schedule.hexdigest(), 'final_rng_state': rng.getstate(),
            'families': dict(families), 'distinct_identifier_targets_sampled': len(targets),
            'identifier_target_exposure_histogram': dict(Counter(targets.values()))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--worlds', type=int, nargs='+', default=[256, 8192])
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=127)
    args = parser.parse_args()
    if args.steps < 1 or len(args.worlds) != len(set(args.worlds)):
        parser.error('Positive steps and distinct world counts are required')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('Create the intended artifact root before preparing data')
    if shutil.disk_usage(args.output.parent).free < 10*1024**3:
        raise OSError('Dataset preparation would violate the 10 GiB free-space reserve')
    args.output.mkdir(exist_ok=False)
    reports, reference = {}, None
    for count in args.worlds:
        episodes = build(count, args.split)
        path = args.output/f'worlds-{count}.jsonl'
        save_episodes(path, episodes)
        EpisodeIndex(path)
        plan = sampling_plan(episodes, seed=args.seed, steps=args.steps, accumulation=4)
        identity = plan['schedule_sha256'], plan.pop('final_rng_state')
        if reference is not None and identity != reference:
            raise ValueError('Requested corpora do not preserve the actual sampled schedule')
        reference = identity
        reports[str(count)] = {'episodes': len(episodes), 'sha256': file_sha256(path),
            'distinct_identifier_targets': len({e.answer for e in episodes if e.task_family == 'multiuse/identifier'}),
            **plan}
    (args.output/'manifest.json').write_text(json.dumps({'split': args.split, 'steps': args.steps,
        'seed': args.seed, 'accumulation': 4, 'script_sha256': file_sha256(__file__),
        'question_generator_sha256': file_sha256(Path(__file__).with_name('make_identifier_character_control.py')),
        'corpora': reports, 'notice': 'Same initial source/task/loop schedule is possible with these slot-major '
        'corpora. World content and target freshness differ. Equal updates are not equal token compute.'},
        indent=2)+'\n')
