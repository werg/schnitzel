#!/usr/bin/env python3
"""Expose every entity in a world to a frozen text/latent reader.

The oracle supplies world membership, not the relevant entity's source pair.
True sufficient groups remain unchanged and govern support-removal controls.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import time

import torch
from safetensors.torch import load_model

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes, counterfactual_multiuse
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.evaluation import build_shared_bank, score_answers
from schnitz.metrics import summarize_rows, counterfactual_metrics, paired_world_bootstrap
from schnitz.operations import atomic_json
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context, resource_report, reset_resource_peaks
from schnitz.trajectories import file_sha256


def progress(event, **fields):
    """Console-only telemetry; a disconnected log must not lose an evaluation."""
    try:
        print(json.dumps({'event': event, **fields}), flush=True)
    except OSError:
        pass


@torch.no_grad()
def evaluate(run, episodes_file, output, *, learned_world=False, read_budget=2,
             routing_probe=None, independent_routing_query=False, read_count_policy=None):
    config = config_from_run(run)
    if config.train.arm not in {'memory', 'oracle_text', 'direct_latent'}:
        raise ValueError('Use a text or latent checkpoint')
    if config.memory.read_steps != 1:
        raise ValueError('This world-context diagnostic requires one complete read')
    if learned_world:
        if config.train.arm != 'memory' or len(config.memory.payload_dims) != 1 or read_budget < 1:
            raise ValueError('Learned world routing requires a memory arm, one space and positive read budget')
        config.memory.neighbors = [read_budget]
    episodes = load_episodes(episodes_file)
    universe = frozenset(s.record_id for e in episodes for s in e.supports)
    variants = {kind: [counterfactual_multiuse(e, kind) for e in episodes]
                for kind in ('restoration', 'permission')}
    progress('evaluation_checkpoint_verify', run=str(run))
    checkpoint = resolve_checkpoint(run, verify=True)
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    reset_resource_peaks(config.train.device)
    adapter = None
    if routing_probe is not None:
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=routing_probe,
                                          independent_routing_query=independent_routing_query)
    else:
        if independent_routing_query:
            raise ValueError('Supply a routing probe for the independent query override')
        agent = SchnitzelAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    count_policy = (attach_read_count_policy(agent, checkpoint, read_count_policy)
                    if read_count_policy is not None else None)
    output.mkdir(parents=True, exist_ok=False)
    store = DiskStore(output / 'bank.sqlite')
    progress('evaluation_offline_write', variant='all', queries=len(episodes))
    writes = {'all': build_shared_bank(agent, store, episodes)}
    for kind, changed in variants.items():
        progress('evaluation_offline_write', variant=kind, queries=len(changed))
        writes[kind] = build_shared_bank(agent, store, changed, namespace=kind)
    def forbidden(*args, **kwargs):
        raise AssertionError('Writer invoked after offline bank creation')
    agent.produce = forbidden
    store = DiskStore(store.path)
    rows, originals, plans = [], {e.episode_id: e for e in episodes}, {}
    start = time.perf_counter()
    with autocast_context(config):
        for kind, group in [('all', episodes), *variants.items()]:
            progress('evaluation_stored_reads', variant=kind, queries=len(group))
            for index, e in enumerate(group, 1):
                conditions = (['all', 'selected_pair', 'none', 'zero_values'] +
                              [f'drop_{i}' for i in range(len(e.required_ids))]) if kind == 'all' else ['cf_' + kind]
                for condition in conditions:
                    selected = tuple(s.record_id for s in e.supports)
                    if condition == 'selected_pair':
                        selected = e.required_ids
                    elif condition == 'none':
                        selected = ()
                    elif condition.startswith('drop_'):
                        selected = tuple(rid for rid in selected if rid != e.required_ids[int(condition[5:])])
                    text = '\n'.join(s.text for s in e.supports if s.record_id in selected)
                    prompt = agent.prompt_ids(e.query, text if config.train.arm == 'oracle_text' else '')
                    memory = None
                    if config.train.arm != 'oracle_text' and selected:
                        namespace = 'global' if kind == 'all' else kind
                        fixed = ([[replace(p, namespace=namespace) for p in step] for step in plans[e.episode_id]]
                                 if condition == 'zero_values' or kind != 'all' else None)
                        session = read_session(agent, store, prompt, namespace=namespace, generation='frozen-v1',
                            query_time=e.query_time,
                            oracle_ids=None if learned_world and condition != 'selected_pair' else selected,
                            exclude_ids=universe - set(selected) if learned_world else frozenset(),
                            ablate_values=condition == 'zero_values', fixed_plans=fixed)
                        memory = session.memory
                        selected = tuple(dict.fromkeys(rid for ids in session.selected_ids for rid in ids))
                        if condition == 'all':
                            plans[e.episode_id] = session.plans
                    groups = e.sufficient_groups or (e.required_ids,)
                    row = dict(episode=e.episode_id, environment=e.environment, task_family=e.task_family,
                        condition=condition, answer=e.answer, selected_ids=list(selected),
                        selected_record_count=len(selected), complete_support=any(set(g) <= set(selected) for g in groups),
                        **score_answers(agent, prompt, memory, e.answer, e.choices))
                    if kind != 'all':
                        row['counterfactual_should_change'] = e.answer != originals[e.episode_id].answer
                    rows.append(row)
                if index % 32 == 0 or index == len(group):
                    progress('evaluation_progress', variant=kind, completed_queries=index,
                             total_queries=len(group), scored_rows=len(rows),
                             elapsed_read_seconds=time.perf_counter() - start)
    names = ('cf_restoration', 'cf_permission')
    families = sorted({e.task_family for e in episodes})
    result = dict(protocol='World-scoped evidence, competing entities, frozen stored-only read.',
        checkpoint=str(checkpoint), checkpoint_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
        episodes_sha256=file_sha256(episodes_file), writes=writes, summary=summarize_rows(rows),
        by_family={f: summarize_rows([r for r in rows if r['task_family'] == f]) for f in families},
        counterfactuals_by_family={f: counterfactual_metrics([r for r in rows if r['task_family'] == f], names)
                                  for f in families},
        paired_gains={f: {c: paired_world_bootstrap([r for r in rows if r['task_family'] == f], b=c)
                         for c in ('selected_pair', 'zero_values', 'none')} for f in families},
        read_evaluation_seconds=time.perf_counter() - start, resources=resource_report(), rows=rows,
        notice='All-world versus selected-pair reads differ in information/compute budget. '
               'World membership is supplied; routing across worlds is not tested. '
               'Zero-payload interventions apply only to latent arms; text stays unchanged.')
    if learned_world:
        result['protocol'] = 'World-scoped exact learned ranking of stored keys; frozen stored-only reads.'
        result['routing'] = {'read_budget': read_budget, 'candidate_scope': 'supplied world membership',
                             'search': 'exact scan, not ANN',
                             'interventions': 'Zero values and counterfactuals preserve original read plans; '
                                              'support removal excludes the record before reranking.'}
        result['notice'] = ('Selected-pair is an oracle control. Learned ranking receives all eligible world '
                            'records, never required-support IDs. World membership is supplied; '
                            'this is not cross-world retrieval or learned authorization.')
    result['routing_probe'] = adapter
    result['read_count_policy'] = count_policy
    atomic_json(output / 'results.json', result)
    atomic_json(output / 'summary.json', {k: v for k, v in result.items() if k != 'rows'})
    progress('evaluation_committed', output=str(output), scored_rows=len(rows))
    return output / 'summary.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--learned-world', action='store_true', help='Rank eligible world records by stored keys')
    parser.add_argument('--read-count-policy', type=Path)
    parser.add_argument('--routing-probe', type=Path)
    parser.add_argument('--independent-routing-query', action='store_true')
    parser.add_argument('--read-budget', type=int, default=2, help='Record budget for --learned-world')
    args = parser.parse_args()
    print(json.dumps({'summary': str(evaluate(args.run, args.episodes, args.output,
                     learned_world=args.learned_world, read_budget=args.read_budget,
                     routing_probe=args.routing_probe, independent_routing_query=args.independent_routing_query,
                     read_count_policy=args.read_count_policy))}, indent=2))
