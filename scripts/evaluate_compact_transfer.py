"""Counterfactual and candidate-free confirmation using only persisted compact codes."""
import argparse
import json
from pathlib import Path
import re

import torch

from probe_stored_compaction import persist_codes
from schnitz.checkpoints import resolve_checkpoint
from schnitz.cluster_store import ClusterBank, state_fingerprint
from schnitz.compaction import SyntheticCompactor
from schnitz.data import load_episodes, counterfactual_multiuse, save_episodes
from schnitz.evaluation import build_shared_bank, score_answers
from schnitz.evaluation_adapter import load_frozen_agent
from schnitz.frozen_scoring import FrozenScorer
from schnitz.metrics import summarize_rows, counterfactual_metrics
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def load_compactor(agent, path, source_hash, reader_hash):
    state = torch.load(path, weights_only=True, map_location='cpu')
    identity = state['identity']
    if (state['step'] != identity['steps'] or identity['checkpoint_manifest_sha256'] != source_hash
            or identity['reader_hash'] != reader_hash or identity['compactor']['records'] != 1):
        raise ValueError('Compactor endpoint incomplete or source/reader/code budget changed')
    compactor = SyntheticCompactor(agent.config.memory.payload_dims[0], identity['compactor']['width'], 1).to(agent.device).eval().requires_grad_(False)
    compactor.load_state_dict(state['compactor'])
    return compactor, {'sha256': file_sha256(path), 'steps': state['step'],
                       'source_manifest_sha256': source_hash, 'reader_hash': reader_hash}


@torch.no_grad()
def run(source, fit, output, *, reuse_decoder=False, worlds=32, episodes_file=None, extra_compactors=None):
    if worlds < 1:
        raise ValueError('Positive world count required')
    extra_compactors = extra_compactors or {}
    if any(not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in {'raw', 'mean', 'trained'} for name in extra_compactors):
        raise ValueError('Extra compactor names must be distinct safe names, excluding raw/mean/trained')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        if not all((fit / f'{method}-stored.json').exists() for method in ('mean', 'trained')):
            raise ValueError('Wait for the completed frozen-compactor fit and initial stored evaluation')
        checkpoint = resolve_checkpoint(source, verify=True)
        source_hash = file_sha256(checkpoint / 'manifest.json')
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        reader_hash = state_fingerprint(agent.reader)
        compactors, endpoints = {}, {}
        for name, path in {'trained': fit / 'resume.pt', **extra_compactors}.items():
            compactors[name], endpoints[name] = load_compactor(agent, path, source_hash, reader_hash)
        rebuild = episodes_file is not None or bool(extra_compactors)
        episodes_file = episodes_file or fit / 'heldout/episodes.jsonl'
        episodes = load_episodes(episodes_file)
        selected_worlds = set(list(dict.fromkeys(e.environment for e in episodes))[:worlds])
        episodes = [e for e in episodes if e.environment in selected_worlds]
        groups = {'all': episodes, **{kind: [counterfactual_multiuse(e, kind) for e in episodes]
                                    for kind in ('permission', 'restoration')}}
        inputs = {'source_manifest_sha256': source_hash,
                  'compactor_state_sha256': file_sha256(fit / 'resume.pt'),
                  'episodes_sha256': file_sha256(episodes_file), 'endpoints': endpoints,
                  'script_sha256': file_sha256(__file__), 'max_new_tokens': 24, 'generation_worlds': min(8, len(selected_worlds))}
        inputs.update(reuse_decoder=reuse_decoder, selected_worlds=sorted(selected_worlds))
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != inputs:
            raise ValueError('Confirmation inputs changed')
        atomic_json(output / 'inputs.json', inputs)
        stores, banks = {}, {}
        for kind, group in groups.items():
            path = fit / 'heldout/bank.sqlite' if kind == 'all' and not rebuild else output / f'{kind}.sqlite'
            stores[kind] = DiskStore(path)
            if kind != 'all' or rebuild:
                save_episodes(output / f'{kind}.jsonl', group)
                build_shared_bank(agent, stores[kind], group,
                                  writer_identity=inputs['source_manifest_sha256'])
            banks[kind] = {}
            for method in ('mean', *compactors):
                banks[kind][method] = (ClusterBank(stores[kind], view=method, reader_hash=reader_hash)
                                      if kind == 'all' and not rebuild else
                                      persist_codes(agent, compactors.get(method), stores[kind], group, method))
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Writer or compactor invoked during stored-only confirmation')
        agent.produce = forbidden
        for compactor in compactors.values():
            compactor.forward = forbidden
        scorer = FrozenScorer(agent) if reuse_decoder else None
        worlds = list(dict.fromkeys(e.environment for e in episodes))[:8]
        originals = {e.episode_id: e for e in episodes}
        for method in ('raw', 'mean', *compactors):
            destination = output / f'{method}.json'
            if destination.exists():
                continue
            rows, generations = [], []
            with autocast_context(config):
                for kind, group in groups.items():
                    for index, episode in enumerate(group):
                        if stop_requested(output):
                            raise RuntimeError('Stored confirmation stopped; completed arms remain reusable')
                        prompt = agent.prompt_ids(episode.query)
                        conditions = (['all', 'none', 'zero_values'] +
                                      [f'drop_{i}' for i in range(len(episode.required_ids))]) if kind == 'all' else ['cf_' + kind]
                        for condition in conditions:
                            ids = episode.required_ids
                            if condition == 'none':
                                ids = ()
                            elif condition.startswith('drop_'):
                                ids = tuple(rid for i, rid in enumerate(ids) if i != int(condition[5:]))
                            memory, accounting = None, []
                            if ids:
                                session = read_session(agent, stores[kind], prompt, namespace='global',
                                    generation='frozen-v1', query_time=episode.query_time, oracle_ids=ids,
                                    cluster_bank=banks[kind].get(method), ablate_values=condition == 'zero_values')
                                memory, accounting = session.memory, session.payload_accounting
                            row = {'episode': episode.episode_id, 'environment': episode.environment,
                                   'task_family': episode.task_family, 'condition': condition, 'answer': episode.answer,
                                   'selected_ids': list(ids), 'payload_accounting': accounting,
                                   'complete_support': any(set(g) <= set(ids) for g in
                                                           (episode.sufficient_groups or (episode.required_ids,))),
                                   **(scorer.score(prompt, memory, episode.answer, episode.choices) if scorer else
                                      score_answers(agent, prompt, memory, episode.answer, episode.choices))}
                            if kind != 'all':
                                row['counterfactual_should_change'] = episode.answer != originals[episode.episode_id].answer
                            rows.append(row)
                            if kind == 'all' and episode.environment in worlds and condition in {'all', 'none', 'zero_values'}:
                                prediction = (scorer.generate(prompt, memory, max_new_tokens=24) if scorer else
                                              agent.generate_from_memory(prompt, memory, max_new_tokens=24))
                                generations.append({k: row[k] for k in ('episode', 'environment', 'task_family', 'condition', 'answer')} |
                                                   {'prediction': prediction, 'exact_match': prediction == episode.answer})
                        if (index + 1) % 64 == 0:
                            print(json.dumps({'method': method, 'variant': kind, 'queries': index + 1}), flush=True)
            families = sorted({e.task_family for e in episodes})
            atomic_json(destination, {'inputs': inputs, 'method': method, 'rows': rows, 'generation_rows': generations,
                'by_family': {f: summarize_rows([r for r in rows if r['task_family'] == f]) for f in families},
                'counterfactuals_by_family': {f: counterfactual_metrics([r for r in rows if r['task_family'] == f],
                                                                        ('cf_permission', 'cf_restoration')) for f in families},
                'generation_by_family': {f: {c: {'n': len(selected), 'exact_match': sum(r['exact_match'] for r in selected) / len(selected)}
                     for c in ('all', 'none', 'zero_values')
                     if (selected := [r for r in generations if r['task_family'] == f and r['condition'] == c])} for f in families},
                'code_storage': {kind: banks[kind][method].sizes() for kind in groups} if method != 'raw' else None,
                'decoder_reuse': scorer.report() if scorer else None,
                'notice': 'Oracle selected clusters; exact raw subset fallback. No learned routing, net disk savings or agent claim.'})
            print(json.dumps({'completed': method}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-decoder', action='store_true')
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--episodes', type=Path)
    parser.add_argument('--extra-compactor', action='append', default=[], metavar='NAME=RESUME_PATH')
    args = parser.parse_args()
    extra = {}
    for value in args.extra_compactor:
        name, path = value.split('=', 1)
        if name in extra:
            raise ValueError('Duplicate extra compactor name')
        extra[name] = Path(path)
    run(args.source, args.fit, args.output, reuse_decoder=args.reuse_decoder, worlds=args.worlds,
        episodes_file=args.episodes, extra_compactors=extra)
