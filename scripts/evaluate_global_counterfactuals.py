"""Counterfactual action sensitivity under captured full-bank learned read plans."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes, counterfactual_multiuse, save_episodes
from schnitz.evaluation import build_shared_bank
from schnitz.evaluation_interventions import TargetedPayloadStore
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.frozen_scoring import FrozenScorer
from schnitz.offline_bank import canonical_json
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore, ReadPlan, Selection
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, reference_path, episodes_file, router, count_policy, output, generations_file=None, targeted_bank=None, alternative_banks=None):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        reference = json.loads(reference_path.read_text())
        if not reference['global_bank']:
            raise ValueError('Supply the completed full-bank reference')
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=router, independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        agent.requires_grad_(False)
        prior = reference['inputs']
        if (prior['source_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json') or
                prior['episodes_sha256'] != file_sha256(episodes_file) or prior['routing_probe'] != adapter or
                prior['read_count_policy'] != count):
            raise ValueError('Counterfactual reference identity differs')
        identity = {'reference_sha256': file_sha256(reference_path), 'source_inputs': prior,
                    'script_sha256': file_sha256(__file__),
                    'selection': 'Captured original full-bank learned selections; no counterfactual reranking'}
        generation_reference = reference
        if generations_file is not None:
            generation_reference = json.loads(generations_file.read_text())
            if (generation_reference['source_inputs'] != prior or
                    generation_reference['source_reference_sha256'] != file_sha256(reference_path)):
                raise ValueError('Expanded generation reference differs')
            identity['generation_reference_sha256'] = file_sha256(generations_file)
            identity['generation_worlds'] = generation_reference['worlds']
        base_store = None
        if targeted_bank is not None:
            if file_sha256(targeted_bank) != prior['bank_sha256']:
                raise ValueError('Targeted base bank differs from the original evaluation')
            base_store = DiskStore(targeted_bank)
            identity['selection'] = 'Captured original full-bank selections; change only the query-required source record'
            identity['targeted_base_bank_sha256'] = prior['bank_sha256']
        if alternative_banks is not None:
            identity['alternative_banks'] = str(alternative_banks)
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Counterfactual inputs changed')
        atomic_json(output / 'inputs.json', identity)
        original = load_episodes(episodes_file)
        baseline = {r['episode']: r for r in reference['rows'] if r['condition'] == 'all'}
        generated = {r['episode']: r for r in generation_reference['generation_rows'] if r['condition'] == 'all'}
        if set(baseline) != {e.episode_id for e in original}:
            raise ValueError('Counterfactual episode grid differs')
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Counterfactual inference invoked a writer or compactor')
        if alternative_banks is not None:
            agent.produce = forbidden
        groups, stores = {}, {}
        for kind in ('permission', 'restoration'):
            groups[kind] = [counterfactual_multiuse(e, kind) for e in original]
            bank_root = alternative_banks or output
            data_path = bank_root / f'{kind}.jsonl'
            if alternative_banks is not None:
                if load_episodes(data_path) != groups[kind] or not (bank_root / f'{kind}.sqlite').is_file():
                    raise ValueError('Alternative counterfactual bank corpus differs')
            else:
                save_episodes(data_path, groups[kind])
            stores[kind] = DiskStore(bank_root / f'{kind}.sqlite')
            writer_identity = hashlib.sha256(canonical_json({'source': prior['source_manifest_sha256'],
                'adapter': adapter['probe_sha256'], 'episodes': file_sha256(data_path)}).encode()).hexdigest()
            build_shared_bank(agent, stores[kind], groups[kind], writer_identity=writer_identity)
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer = FrozenScorer(agent)
        for kind, group in groups.items():
            destination = output / f'{kind}.json'
            if destination.exists():
                if json.loads(destination.read_text())['inputs'] != identity:
                    raise ValueError('Completed counterfactual identity differs')
                continue
            rows, generations = [], []
            with autocast_context(config):
                for e, old in zip(group, original, strict=True):
                    if e.task_family != 'multiuse/action':
                        continue
                    if stop_requested(output):
                        raise RuntimeError('Counterfactual evaluation stopped; completed variants remain reusable')
                    if e.query != old.query or e.required_ids != old.required_ids or e.query_time != old.query_time:
                        raise ValueError('Counterfactual changed the causal query or evidence identities')
                    previous = baseline[e.episode_id]
                    if previous['answer'] != old.answer:
                        raise ValueError('Reference answer differs')
                    plan = ReadPlan('global', 's0', 'frozen-v1', 'research', e.query_time,
                                    tuple(Selection(rid, 0.) for rid in previous['selected_ids']))
                    prompt = agent.prompt_ids(e.query)
                    targets = frozenset(s.record_id for s in old.supports
                                        if s.record_id in old.required_ids and s.kind == kind)
                    if len(targets) != 1:
                        raise ValueError('Action intervention requires exactly one target rule record')
                    read_store = (TargetedPayloadStore(base_store, stores[kind], targets)
                                  if base_store is not None else stores[kind])
                    session = read_session(agent, read_store, prompt, namespace='global', generation='frozen-v1',
                                           query_time=e.query_time, fixed_plans=[[plan]])
                    selected = list(dict.fromkeys(r for ids in session.selected_ids for r in ids))
                    if selected != previous['selected_ids']:
                        raise ValueError('Captured counterfactual selection changed')
                    score = scorer.score(prompt, session.memory, e.answer, e.choices)
                    if base_store is not None and not targets.intersection(selected):
                        if score['choice_sequence_nll'] != previous['choice_sequence_nll']:
                            raise AssertionError('An unselected targeted record changed decoder scores')
                    rows.append({'episode': e.episode_id, 'environment': e.environment, 'variant': kind,
                        'answer': e.answer, 'selected_ids': selected, **score,
                        'intervention_target_ids': sorted(targets) if base_store is not None else None,
                        'should_change': old.answer != e.answer,
                        'both_correct': previous['choice_correct'] and score['choice_correct'],
                        'prediction_changed': previous['predicted_action'] != score['predicted_action']})
                    if e.episode_id in generated:
                        prediction = scorer.generate(prompt, session.memory, max_new_tokens=prior['max_new_tokens'])
                        old_generation = generated[e.episode_id]
                        if base_store is not None and not targets.intersection(selected):
                            if prediction != old_generation['prediction']:
                                raise AssertionError('An unselected targeted record changed generation')
                        generations.append({'episode': e.episode_id, 'environment': e.environment, 'variant': kind,
                            'answer': e.answer, 'prediction': prediction, 'exact_match': prediction == e.answer,
                            'should_change': old.answer != e.answer,
                            'both_correct': old_generation['exact_match'] and prediction == e.answer,
                            'prediction_changed': old_generation['prediction'] != prediction})
            def summary(group, metric):
                changed = [r for r in group if r['should_change']]
                unchanged = [r for r in group if not r['should_change']]
                return {'n': len(group), 'correct': sum(r[metric] for r in group),
                    'changed_pairs': len(changed), 'both_correct_on_change': sum(r['both_correct'] for r in changed),
                    'unchanged_pairs': len(unchanged), 'false_changes': sum(r['prediction_changed'] for r in unchanged)}
            atomic_json(destination, {'inputs': identity, 'bank_sha256': file_sha256(stores[kind].path),
                'choices': summary(rows, 'choice_correct'), 'generation': summary(generations, 'exact_match'),
                'rows': rows, 'generation_rows': generations, 'decoder_reuse': scorer.report(),
                'notice': 'Sensitivity to stored rule values under fixed learned retrieval, not end-to-end rerouting or agent success.'})
            print(canonical_json({'completed_variant': kind, 'choices': summary(rows, 'choice_correct'),
                                  'generation': summary(generations, 'exact_match')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'episodes', 'router', 'count-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--generations', type=Path)
    parser.add_argument('--targeted-bank', type=Path, help='Base bank for changing only the required rule record')
    parser.add_argument('--alternative-banks', type=Path, help='Reuse existing verified offline counterfactual banks')
    args = parser.parse_args()
    run(args.source, args.reference, args.episodes, args.router, args.count_policy, args.output, args.generations, args.targeted_bank, args.alternative_banks)
