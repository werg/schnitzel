"""Resumable offline banks followed by frozen oracle transfer and free generation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re

import torch

from schnitz.archiving import ensure_free
from schnitz.checkpoints import _atomic_text, resolve_checkpoint, stop_on_signal
from schnitz.data import load_episodes, counterfactual_multiuse, evidence_ids
from schnitz.evaluation import build_shared_bank, build_persistent_codes, stored_transfer_evaluation
from schnitz.evaluation_adapter import load_frozen_agent
from schnitz.frozen_scoring import FrozenScorer
from schnitz.metrics import counterfactual_metrics
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.runtime import available_host_memory, compute_watchdog
from schnitz.sessions import read_session
from schnitz.store import DiskStore, ReadPlan, Selection
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def identifier_counterfactual(episode):
    """Synthetic endpoint-only intervention, preserving IDs, rules and causal time."""
    if not episode.task_family.startswith('multiuse/'):
        raise ValueError('Generated multiuse episode required')
    def invert(value):
        if re.fullmatch(r'api_[0-9a-f]{6}', value) is None:
            raise ValueError('Expected generated six-hex endpoint')
        return 'api_'+''.join(format(int(c, 16) ^ 15, 'x') for c in value[4:])
    sources = []
    for source in episode.supports:
        text = source.text
        if source.kind == 'permission':
            text, count = re.subn(r'(?<=Its endpoint is )api_[0-9a-f]{6}(?=\.)',
                                 lambda match: invert(match[0]), text)
            if count != 1:
                raise ValueError('Malformed generated endpoint source')
        sources.append(replace(source, text=text))
    identifier = episode.task_family == 'multiuse/identifier'
    return replace(episode, supports=tuple(sources),
                   answer=invert(episode.answer) if identifier else episode.answer,
                   choices=tuple(map(invert, episode.choices)) if identifier else episode.choices)


@torch.no_grad()
def run(source, episodes_path, output, *, compact_method='raw'):
    if compact_method not in {'raw', 'mean', 'trained'}:
        raise ValueError('Choose raw, mean or trained code reads')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        checkpoint = resolve_checkpoint(source, verify=True)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_path), 'script_sha256': file_sha256(__file__),
                    'max_new_tokens': 24, 'routing': 'oracle required sources', 'compact_method': compact_method}
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Oracle evaluation identity changed')
        atomic_json(output/'inputs.json', identity)
        if (output/'results.json').exists():
            completed = json.loads((output/'results.json').read_text())
            if completed['inputs'] != identity:
                raise ValueError('Completed oracle evaluation identity changed')
            if not (output/'bank.sqlite').is_file() or file_sha256(output/'bank.sqlite') != completed['bank_sha256']:
                raise ValueError('Completed oracle evaluation bank changed')
            return
        saved = {}
        paths = [*output.glob('scoring-*.json'), output/'generation-progress.json']
        for path in paths:
            if path.exists():
                value = json.loads(path.read_text())
                if value.get('format') != 1 or value['inputs'] != identity:
                    raise ValueError('Partial oracle evaluation identity changed')
                saved[path.name] = value
        if saved:
            if not (output/'bank.sqlite').is_file():
                raise ValueError('Partial oracle evaluation bank changed')
            bank_hash = file_sha256(output/'bank.sqlite')
            if any(value['bank_sha256'] != bank_hash for value in saved.values()):
                raise ValueError('Partial oracle evaluation bank changed')
        config = config_from_run(checkpoint)
        resource_stop = None
        def want_stop():
            nonlocal resource_stop
            available = available_host_memory()
            if available is not None and available < config.train.min_system_available_bytes:
                resource_stop = f'host memory reserve reached: {available} < {config.train.min_system_available_bytes} bytes'
            return resource_stop is not None or stop['signal'] is not None or stop_requested(output)
        if want_stop():
            raise RuntimeError(f"Stopped before evaluation model allocation: {resource_stop or 'control/signal'}")
        if config.train.arm != 'memory' or config.train.retrieval != 'oracle' or config.train.evidence_scope != 'required':
            raise ValueError('This diagnostic requires oracle required-source memory')
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        produce = agent.produce
        def guarded_produce(ids):
            if want_stop():
                raise RuntimeError(f"Stopped during atomic offline bank creation: {resource_stop or 'control/signal'}")
            ensure_free(output, 0, config.train.min_free_disk_bytes)
            with compute_watchdog(config.train.stall_timeout_seconds, device=agent.device):
                return produce(ids)
        agent.produce = guarded_produce
        if compact_method != 'raw':
            if len(config.memory.payload_dims) != 1:
                raise ValueError('Compact confirmation requires one space')
            if compact_method == 'trained' and agent.compactor is None:
                raise ValueError('Trained-code confirmation requires a checkpointed compactor')
            config.memory.compaction = 'mean' if compact_method == 'mean' else 'synthetic'
        episodes = load_episodes(episodes_path)
        variants = {k: [counterfactual_multiuse(e, k) for e in episodes] for k in ('permission', 'restoration')}
        variants['identifier'] = [identifier_counterfactual(e) for e in episodes]
        store = DiskStore(output/'bank.sqlite')
        writes, banks, code_storage = {}, {}, {}
        for name, group in [('global', episodes), *variants.items()]:
            writes[name] = build_shared_bank(agent, store, group, namespace=name,
                writer_identity=identity['checkpoint_manifest_sha256'])
            if compact_method != 'raw':
                banks[name], code_storage[name] = build_persistent_codes(agent, store, group, namespace=name,
                                                                             view_name='oracle-'+name+'-'+compact_method)
            if want_stop():
                raise RuntimeError(f"Stopped after atomic bank publication: {resource_stop or 'control/signal'}")
        bank_hash = file_sha256(store.path)
        if any(value['bank_sha256'] != bank_hash for value in saved.values()):
            raise ValueError('Partial oracle evaluation bank changed during setup')
        def publish(name, **value):
            encoded = json.dumps(dict(format=1, inputs=identity, bank_sha256=bank_hash, **value)) + '\n'
            ensure_free(output, len(encoded.encode()) * 2, config.train.min_free_disk_bytes)
            _atomic_text(output/name, encoded)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Oracle inference called writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        store = DiskStore(store.path)
        plans = {}
        def progress(event, *, phase='scoring', variant='global'):
            if want_stop():
                raise RuntimeError(f"Stopped during reproducible stored evaluation: {resource_stop or 'control/signal'}")
            try:
                print(json.dumps({'phase': phase, 'variant': variant, **event}), flush=True)
            except OSError:
                pass
        if 'scoring-global.json' in saved:
            entry = saved['scoring-global.json']
            scores = entry['scores']
            plans = {episode: [[ReadPlan(**(p | {'selections': tuple(Selection(**s) for s in p['selections'])}))
                                for p in step] for step in steps] for episode, steps in entry['plans'].items()}
        else:
            scores = stored_transfer_evaluation(agent, store, episodes, drop_supports=True,
                capture_plans=plans, progress=progress, cluster_bank=banks.get('global'),
                use_codes_for_all_conditions=compact_method != 'raw')
            publish('scoring-global.json', scores=scores,
                    plans={e: [[asdict(p) for p in step] for step in steps] for e, steps in plans.items()})
        originals = {e.episode_id: e for e in episodes}
        for name, group in variants.items():
            path = 'scoring-'+name+'.json'
            if path in saved:
                changed = saved[path]['scores']
            else:
                changed = stored_transfer_evaluation(agent, store, group, namespace=name,
                    full_evidence_only=True, fixed_plans_by_episode=plans,
                    progress=lambda event, variant=name: progress(event, variant=variant), cluster_bank=banks.get(name),
                    use_codes_for_all_conditions=compact_method != 'raw')
                for row in changed['rows']:
                    row['condition'] = 'cf_'+name
                    row['counterfactual_should_change'] = row['answer'] != originals[row['episode']].answer
                publish(path, scores=changed)
            scores['rows'].extend(changed['rows'])
        scorer = FrozenScorer(agent)
        rows = saved.get('generation-progress.json', {}).get('rows', [])
        cursor = 0
        with autocast_context(config):
            for name, group in [('global', episodes), *variants.items()]:
                for index, e in enumerate(group, 1):
                    prompt = agent.prompt_ids(e.query)
                    for condition in (('all', 'zero_values', 'none') if name == 'global' else ('cf_'+name,)):
                        metadata = dict(episode=e.episode_id, environment=e.environment, task_family=e.task_family,
                                        condition=condition, answer=e.answer)
                        if cursor < len(rows):
                            row = rows[cursor]
                            if (any(row[k] != v for k, v in metadata.items())
                                    or row['exact_match'] != (row['prediction'] == e.answer)):
                                raise ValueError('Partial generation prefix changed')
                            cursor += 1
                            continue
                        if want_stop():
                            publish('generation-progress.json', rows=rows)
                            raise RuntimeError(f"Stopped during reproducible free generation: {resource_stop or 'control/signal'}")
                        memory, accounting = None, []
                        if condition != 'none':
                            fixed = [[replace(p, namespace=name) for p in step] for step in plans[e.episode_id]]
                            session = read_session(agent, store, prompt, namespace=name, generation='frozen-v1',
                                query_time=e.query_time, oracle_ids=tuple(evidence_ids(e, 'required')),
                                ablate_values=condition == 'zero_values', fixed_plans=fixed, cluster_bank=banks.get(name))
                            memory, accounting = session.memory, session.payload_accounting
                        prediction = scorer.generate(prompt, memory, max_new_tokens=24)
                        row = {'episode': e.episode_id, 'environment': e.environment, 'task_family': e.task_family,
                               'condition': condition, 'answer': e.answer, 'prediction': prediction,
                               'exact_match': prediction == e.answer, 'payload_accounting': accounting}
                        if name != 'global':
                            row['counterfactual_should_change'] = e.answer != originals[e.episode_id].answer
                        rows.append(row)
                        cursor += 1
                        if cursor % 32 == 0 or want_stop():
                            publish('generation-progress.json', rows=rows)
                    if index % 32 == 0 or index == len(group):
                        progress({'completed_queries': index, 'total_queries': len(group),
                                  'completed_generations': len(rows), 'total_generations': 6*len(episodes)},
                                 phase='generation', variant=name)
        if cursor != len(rows):
            raise ValueError('Partial generation contains extra rows')
        publish('generation-progress.json', rows=rows)
        families = sorted({e.task_family for e in episodes})
        summary = {f: {c: {'n': len(group), 'correct': sum(r['exact_match'] for r in group)}
                      for c in sorted({r['condition'] for r in rows})
                      if (group := [r for r in rows if r['task_family'] == f and r['condition'] == c])}
                   for f in families}
        cf_names = tuple('cf_'+name for name in variants)
        scores['counterfactuals_by_family'] = {f: counterfactual_metrics(
            [r for r in scores['rows'] if r['task_family'] == f], cf_names) for f in families}
        generation_cf = {f: counterfactual_metrics(
            [dict(r, choice_correct=r['exact_match'], predicted_action=r['prediction'])
             for r in rows if r['task_family'] == f], cf_names) for f in families}
        atomic_json(output/'results.json', {'inputs': identity, 'writes': writes, 'scores': scores,
            'generation_rows': rows, 'generation_summary': summary, 'generation_counterfactuals': generation_cf,
            'bank_sha256': file_sha256(store.path), 'code_storage': code_storage,
            'notice': 'Oracle-selected causal sources, frozen stored-only reads, all-world free generation. '
                      'Rule-type changes use fixed original plans. No learned-routing or agent-success claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--compact-method', choices=['raw', 'mean', 'trained'], default='raw')
    args = parser.parse_args()
    run(args.source, args.episodes, args.output, compact_method=args.compact_method)
