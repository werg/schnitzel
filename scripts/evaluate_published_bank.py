"""Evaluate learned stored-only retrieval against a published corpus generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation import stored_transfer_evaluation
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import (assert_bank_reader_compatible, canonical_json,
                               publish_offline_generation)
from schnitz.operations import atomic_json
from schnitz.sessions import read_session
from schnitz.store import DiskStore, ReadPlan, Selection
from schnitz.training import autocast_context, config_from_run
from schnitz.trajectories import file_sha256


def supplied_mixed_plans(agent, searcher, episodes, *, namespace: str,
                         generation: str, limits: tuple[int, ...]) -> tuple[dict, dict]:
    """Match corpus training's correct and source-swapped causal read plans."""
    plans, swapped_plans = {}, {}
    with torch.no_grad(), autocast_context(agent.config):
        for episode in episodes:
            if episode.support_annotation != 'verified' or len(episode.required_ids) != 1:
                raise ValueError('Matched bank plan requires one verified positive')
            domain = episode.provenance.get('domain', 'research')
            read_plans, swapped_read_plans = [], []

            def provider(completed, _query, routing_query):
                if completed != 1:
                    return None
                for space, limit in enumerate(limits):
                    found = searcher.search(agent.routing_address(routing_query, space)[0],
                        top_k=max(limit + 1, 8), namespace=namespace, space=f's{space}',
                        generation=generation, domain=domain, query_time=episode.query_time)
                    ids = list(episode.required_ids)
                    ids.extend(item.record_id for item in found.selections
                               if item.record_id not in ids)
                    ids = ids[:limit]
                    wrong_id = next((item.record_id for item in found.selections
                                     if item.record_id not in ids and
                                     item.record_id not in episode.required_ids), None)
                    if wrong_id is None:
                        raise ValueError('Matched bank plan needs an eligible unselected source')
                    read_plans.append(ReadPlan(namespace, f's{space}', generation, domain,
                        episode.query_time, tuple(Selection(rid, 0.0) for rid in ids)))
                    swapped_read_plans.append(ReadPlan(namespace, f's{space}', generation,
                        domain, episode.query_time, tuple(Selection(rid, 0.0)
                        for rid in (wrong_id, *ids[1:]))))
                return None

            agent.plan_loop_memory(agent.prompt_ids(episode.query), provider,
                                   include_routing_query=True)
            if len(read_plans) != len(limits):
                raise ValueError('Causal native read was not reached')
            plans[episode.episode_id] = [read_plans]
            swapped_plans[episode.episode_id] = [swapped_read_plans]
    return plans, swapped_plans


def generate_from_published_bank(agent, store, episodes, *, namespace: str,
                                 generation: str, selection: str,
                                 fixed_plans: dict | None, swapped_plans: dict | None,
                                 max_new_tokens: int) -> dict:
    """Generate without target tokens or source re-encoding from captured plans."""
    rows = []
    with torch.no_grad(), autocast_context(agent.config):
        for episode in episodes:
            prompt = agent.prompt_ids(episode.query)
            common = dict(namespace=namespace, generation=generation,
                          query_time=episode.query_time,
                          domain=episode.provenance.get('domain', 'research'))
            plan = None if fixed_plans is None else fixed_plans[episode.episode_id]
            oracle = episode.required_ids if selection == 'oracle' else None
            correct = read_session(agent, store, prompt, oracle_ids=oracle,
                                   fixed_plans=plan, **common)
            zero = read_session(agent, store, prompt, fixed_plans=correct.plans,
                                fixed_gate_weights=correct.gate_weights,
                                ablate_values=True, **common)
            arms = {'all': correct.memory, 'zero_values': zero.memory}
            if swapped_plans is not None:
                wrong = read_session(agent, store, prompt,
                    fixed_plans=swapped_plans[episode.episode_id], **common)
                arms['source_swap'] = wrong.memory
            predictions = {arm: agent.generate_from_memory(prompt, memory,
                max_new_tokens=max_new_tokens) for arm, memory in arms.items()}
            rows.append({'episode': episode.episode_id, 'answer': episode.answer,
                         'predictions': predictions,
                         'exact_match': {arm: pred == episode.answer for arm, pred
                                         in predictions.items()}})
    conditions = tuple(rows[0]['predictions']) if rows else ()
    return {'protocol': 'Greedy generation from published stored payloads; '
                        'source writer disabled and targets withheld.',
            'max_new_tokens': max_new_tokens, 'episodes': len(rows),
            'exact_match': {condition: sum(row['exact_match'][condition] for row in rows)
                            for condition in conditions}, 'rows': rows}


def evaluate(run: Path, bank_dir: Path, episodes_file: Path, output: Path, *,
             max_episodes: int = 16, limits: tuple[int, ...] = (8, 4, 2, 1),
             selection: str = 'learned', generate_episodes: int = 0,
             generate_max_tokens: int = 32) -> dict:
    if max_episodes < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive evaluation count and a fresh output parent required')
    if selection not in {'learned', 'oracle', 'supplied_mixed'}:
        raise ValueError('Evaluation selection must be learned, oracle or supplied_mixed')
    if generate_episodes < 0 or generate_episodes > max_episodes or generate_max_tokens < 1:
        raise ValueError('Generation needs a bounded episode count and positive token budget')
    config = config_from_run(run)
    if len(limits) != len(config.memory.payload_dims):
        raise ValueError('One retrieval limit per active space required')
    if any(k < 1 or k > cap for k, cap in zip(limits, config.memory.neighbors, strict=True)):
        raise ValueError('Evaluation limits must fit the model neighborhood caps')
    bank_manifest_path = bank_dir / 'manifest.json'
    bank_manifest = json.loads(bank_manifest_path.read_text())
    store = DiskStore(bank_dir / 'bank.sqlite')
    verified = publish_offline_generation(store, identity=bank_manifest['identity'],
        namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
        spaces=tuple(bank_manifest['spaces']), shard_ids=tuple(bank_manifest['shards']),
        source_count=bank_manifest['sources'], verify_only=True)
    if canonical_json(verified) != canonical_json({key: bank_manifest[key] for key in verified}):
        raise ValueError('Published bank failed byte verification')
    checkpoint = resolve_checkpoint(run, verify=True)
    compatibility = assert_bank_reader_compatible(
        run, checkpoint, bank_dir, bank_manifest, config)
    config.train.retrieval = 'oracle' if selection == 'supplied_mixed' else selection
    config.train.selected_producers_only = False  # training-only producer policy
    config.train.bank_dir = None
    config.train.bank_read_limits = []
    config.train.payload_contrast_weight = 0.0
    config.train.bank_payload_contrast_weight = 0.0
    config.memory.neighbors = list(limits)
    config.validate()
    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError('Published-bank evaluation must not re-encode a source')
    agent.produce = forbidden_writer
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes:
        raise ValueError('Evaluation needs at least one episode')
    fixed_plans, swapped_plans = None, None
    if selection == 'supplied_mixed':
        index = PublishedKeyIndex(store, namespace=bank_manifest['namespace'],
            generation=bank_manifest['generation'], spaces=tuple(bank_manifest['spaces']),
            expected_sources=bank_manifest['sources'])
        fixed_plans, swapped_plans = supplied_mixed_plans(agent, index, episodes,
            namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
            limits=limits)
    with torch.no_grad():
        report = stored_transfer_evaluation(agent, store, episodes,
            namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
            drop_supports=True, fixed_plans_by_episode=fixed_plans)
        if swapped_plans is not None:
            swapped = stored_transfer_evaluation(agent, store, episodes,
                namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
                fixed_plans_by_episode=swapped_plans, full_evidence_only=True)
            report['source_swap_summary'] = swapped['summary']['all']
            report['source_swap_mean_nll_gap'] = (
                report['source_swap_summary']['mean_target_nll'] -
                report['summary']['all']['mean_target_nll'])
        if generate_episodes:
            report['generation'] = generate_from_published_bank(agent, store,
                episodes[:generate_episodes], namespace=bank_manifest['namespace'],
                generation=bank_manifest['generation'], selection=selection,
                fixed_plans=fixed_plans, swapped_plans=swapped_plans,
                max_new_tokens=generate_max_tokens)
    report['notice'] = ('Published-corpus stored-only diagnostic. '
                        + ('Oracle selection uses verified support labels. ' if selection == 'oracle'
                           else ('Supplied-mixed selection includes one verified support and causal-query neighbors. '
                                 if selection == 'supplied_mixed' else
                                 'Learned selection uses only the causal query. '))
                        + 'Teacher NLL and recall do not establish free-generation accuracy or agent success.')
    identity = {'run_model_sha256': file_sha256(checkpoint / 'model.safetensors'),
                'bank_manifest_sha256': file_sha256(bank_manifest_path),
                'episodes_sha256': file_sha256(episodes_file),
                'max_episodes': max_episodes, 'read_limits': limits,
                'bank_compatibility': compatibility,
                'generate_episodes': generate_episodes,
                'generate_max_tokens': generate_max_tokens if generate_episodes else None,
                'selection': ('learned exact scan; supplied support is never used for retrieval'
                              if selection == 'learned' else
                              'supplied positive plus causal exact neighbors; stored payloads only'
                              if selection == 'supplied_mixed' else
                              'oracle verified support ID; stored payloads only')}
    output.mkdir()
    atomic_json(output / 'inputs.json', identity)
    atomic_json(output / 'results.json', report)
    return {key: value for key, value in report.items() if key != 'rows'} | {
        'output': str(output), 'episodes': len(episodes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=16)
    parser.add_argument('--limits', nargs='+', type=int, default=[8, 4, 2, 1])
    parser.add_argument('--selection', choices=['learned', 'oracle', 'supplied_mixed'],
                        default='learned')
    parser.add_argument('--generate-episodes', type=int, default=0)
    parser.add_argument('--generate-max-tokens', type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.bank, args.episodes, args.output,
                              max_episodes=args.max_episodes,
                              limits=tuple(args.limits), selection=args.selection,
                              generate_episodes=args.generate_episodes,
                              generate_max_tokens=args.generate_max_tokens), indent=2))
