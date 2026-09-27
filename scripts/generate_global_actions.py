"""Extend frozen full-bank action generation to every evaluated world."""
import argparse
import json
from pathlib import Path

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.frozen_scoring import FrozenScorer
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore, ReadPlan, Selection
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, reference_path, bank, episodes_file, router, count_policy, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(output.parent, clear_stop=False):
        reference = json.loads(reference_path.read_text())
        if not reference['global_bank']:
            raise ValueError('Expected a full-bank evaluation reference')
        prior = reference['inputs']
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=router, independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        agent.requires_grad_(False)
        if (prior['source_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json') or
                prior['episodes_sha256'] != file_sha256(episodes_file) or prior['bank_sha256'] != file_sha256(bank) or
                prior['routing_probe'] != adapter or prior['read_count_policy'] != count):
            raise ValueError('Generation reference source/bank/policy identity differs')
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Generation re-encoded a source or compact code')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer, store = FrozenScorer(agent), DiskStore(bank)
        plans = {r['episode']: r for r in reference['rows'] if r['condition'] == 'all'}
        previous = {(r['episode'], r['condition']): r for r in reference['generation_rows']}
        rows, checked = [], 0
        episodes = [e for e in load_episodes(episodes_file) if e.task_family == 'multiuse/action']
        with autocast_context(config):
            for index, e in enumerate(episodes, 1):
                if stop_requested(output.parent):
                    raise RuntimeError('Generation stopped; reference remains intact')
                prompt = agent.prompt_ids(e.query)
                plan = ReadPlan('global', 's0', 'frozen-v1', 'research', e.query_time,
                                tuple(Selection(r, 0.) for r in plans[e.episode_id]['selected_ids']))
                for condition in ('all', 'selected_pair', 'none', 'zero_values'):
                    memory = None
                    if condition != 'none':
                        session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                            query_time=e.query_time, oracle_ids=e.required_ids if condition == 'selected_pair' else None,
                            fixed_plans=None if condition == 'selected_pair' else [[plan]],
                            ablate_values=condition == 'zero_values')
                        memory = session.memory
                    prediction = scorer.generate(prompt, memory, max_new_tokens=prior['max_new_tokens'])
                    row = {'episode': e.episode_id, 'environment': e.environment, 'task_family': e.task_family,
                           'condition': condition, 'answer': e.answer, 'prediction': prediction,
                           'exact_match': prediction == e.answer}
                    old = previous.get((e.episode_id, condition))
                    if old is not None:
                        if old != row:
                            raise AssertionError('Expanded generation changed a previously generated row')
                        checked += 1
                    rows.append(row)
                if index % 16 == 0:
                    print(json.dumps({'completed_actions': index}), flush=True)
        if file_sha256(bank) != prior['bank_sha256']:
            raise ValueError('Frozen bank changed during generation')
        atomic_json(output, {'source_inputs': prior, 'source_reference_sha256': file_sha256(reference_path),
            'script_sha256': file_sha256(__file__), 'worlds': len({e.environment for e in episodes}),
            'prior_generation_rows_reproduced': checked, 'generation_rows': rows,
            'summary': {c: {'n': len(g := [r for r in rows if r['condition'] == c]),
                            'correct': sum(r['exact_match'] for r in g)} for c in ('all', 'none', 'zero_values', 'selected_pair')},
            'notice': 'Post-result expansion to all action queries, fixed endpoints and captured learned plans; '
                      'no candidates, training, writer or model selection.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'bank', 'episodes', 'router', 'count-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.reference, args.bank, args.episodes, args.router, args.count_policy, args.output)
