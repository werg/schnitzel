"""Write-once, serialize/reload, stored-only teacher-continuation evaluation.

Likelihood is imitation/conditioning, not task execution success. There is no live
teacher or tool execution, and the reader never calls the source writer.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import random
import shutil

import torch
from safetensors.torch import load_model
from .agent import SchnitzelAgent
from .checkpoints import resolve_checkpoint, stop_on_signal
from .episode_index import EpisodeIndex
from .data import evidence_ids
from .sessions import read_session
from .store import DiskStore, ReadPlan, Selection, StoredRecord, lookup_record
from .operations import atomic_json, run_lock, stop_requested
from .trajectories import file_sha256
from . import runtime
from .training import autocast_context, config_from_run, output_records, stored_channel, resource_report, reset_resource_peaks


@torch.no_grad()
def build_teacher_bank(agent, store, episodes, *, writer_identity=None, want_stop=None):
    if agent.training:
        raise ValueError('Freeze writer in eval mode before materializing memory')
    sources = {}
    for e in episodes:
        for s in e.supports:
            if s.record_id in sources and sources[s.record_id] != s:
                raise ValueError('Conflicting source content')
            sources[s.record_id] = s
    outputs, complete = OrderedDict(), True
    writer_calls, peak_cached_sources = 0, 0
    if agent.config.train.arm in {'memory', 'direct_latent'}:
        def encoded(rid):
            nonlocal writer_calls, peak_cached_sources
            if rid in outputs:
                outputs.move_to_end(rid)
                return outputs[rid]
            with autocast_context(agent.config):
                source_ids = agent.text_ids(sources[rid].text, source=True)
                with runtime.compute_watchdog(agent.config.train.stall_timeout_seconds,
                                              device=agent.config.train.device):
                    outputs[rid] = tuple(t.detach().cpu() for t in stored_channel(
                        agent, agent.produce(source_ids)))
            writer_calls += 1
            if len(outputs) > 64:
                outputs.popitem(last=False)
            peak_cached_sources = max(peak_cached_sources, len(outputs))
            return outputs[rid]
        ids, seen = sorted(sources), set()
        peers = {rid: ids[(i + 1) % len(ids)] for i, rid in enumerate(ids)}
        def records():
            for e in episodes:
                for s in e.supports:
                    key = (e.environment, s.record_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    original, peer = encoded(s.record_id), encoded(peers[s.record_id])
                    wrong = tuple(original[i] if i % 2 == 0 else peer[i] for i in range(len(original)))
                    for variant, tensors in [('all', original), ('wrong_values', wrong)]:
                        yield from output_records(agent, s, tensors, variant + '/' + e.environment, 'teacher-eval-v1')
        if writer_identity is None:
            store.put_many(records())
        else:
            from . import offline_bank
            if not writer_identity:
                raise ValueError('Verified writer identity cannot be empty')
            scoped = defaultdict(dict)
            for e in episodes:
                for s in e.supports:
                    scoped[e.environment][s.record_id] = s
            identity = {'version': 1, 'writer': writer_identity,
                        'sources_sha256': hashlib.sha256(offline_bank.canonical_json(
                            [asdict(sources[rid]) for rid in ids]).encode()).hexdigest(),
                        'model': asdict(agent.config.model), 'memory': asdict(agent.config.memory),
                        'max_source_tokens': agent.config.train.max_source_tokens,
                        'compute_precision': agent.config.train.precision}
            spaces = tuple(f's{i}' for i in range(len(agent.config.memory.payload_dims)))
            source_environment = {rid: environment for environment, group in
                                  sorted(scoped.items()) for rid in sorted(group)}
            for environment, group in sorted(scoped.items()):
                if want_stop is not None and want_stop():
                    complete = False
                    break
                namespace = 'all/' + environment
                def original_records():
                    for rid, source in sorted(group.items()):
                        yield from output_records(agent, source, encoded(rid), namespace, 'teacher-eval-v1')
                offline_bank.ensure_offline_records(store, original_records, identity=identity,
                    namespace=namespace, generation='teacher-eval-v1', spaces=spaces,
                    expected_count=len(group) * len(spaces))
            if complete:
                for environment, group in sorted(scoped.items()):
                    if want_stop is not None and want_stop():
                        complete = False
                        break
                    namespace = 'wrong_values/' + environment
                    def wrong_records():
                        for rid in sorted(group):
                            for space in spaces:
                                original = lookup_record(store, rid, namespace='all/' + environment,
                                    space=space, generation='teacher-eval-v1')
                                peer = lookup_record(store, peers[rid],
                                    namespace='all/' + source_environment[peers[rid]],
                                    space=space, generation='teacher-eval-v1')
                                yield StoredRecord(rid, original.key, peer.payload, namespace=namespace,
                                    space=space, generation='teacher-eval-v1', domain=original.domain,
                                    created_at=original.created_at, source_id=original.source_id)
                    offline_bank.ensure_offline_records(store, wrong_records, identity=identity,
                        namespace=namespace, generation='teacher-eval-v1', spaces=spaces,
                        expected_count=len(group) * len(spaces))
    return dict(unique_sources=len(sources), writer_calls=writer_calls,
                writer_peak_cached_sources=peak_cached_sources,
                complete=complete,
                wrong_values_distinct_record_available=len(sources) > 1,
                wrong_values_policy='Cyclic distinct-record payload permutation; semantic disagreement not guaranteed.')


def paired_nll_benefit(rows, draws=500):
    pairs, groups = defaultdict(dict), defaultdict(list)
    for r in rows:
        pairs[r['episode']][r['condition']] = r
    for p in pairs.values():
        if 'all' in p and 'none' in p:
            groups[p['all']['trajectory']].append(p['none']['mean_nll'] - p['all']['mean_nll'])
    if not groups:
        return {'gain': None, 'ci95': None}
    values, rng, samples = list(groups.values()), random.Random(0), []
    for _ in range(draws):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return dict(gain=sum(map(sum, values)) / sum(map(len, values)),
                ci95=[samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]],
                trajectories=len(groups), sign='positive = reduced mean-token NLL with memory',
                scope='Bootstrap trajectories, not training seeds.')


@torch.no_grad()
def stored_teacher_evaluation(agent, store, episodes, *, generate_tokens=0, cluster_bank=None,
                              rows=None, saved_plans=None, progress=None, want_stop=None):
    if agent.training or not episodes:
        raise ValueError('Nonempty evaluation set and eval-mode agent required')
    rows = list(rows or [])
    plans_by_episode = dict(saved_plans or {})
    conditions = ('all', 'none', 'zero_values', 'wrong_values') + (('compact',) if cluster_bank else ())
    jobs = [(e, condition) for e in episodes for condition in conditions]
    if len(rows) > len(jobs) or any(
            (row.get('episode'), row.get('condition')) != (e.episode_id, condition)
            or row.get('token_count', 0) < 1 or not math.isfinite(row.get('sequence_nll', math.nan))
            for row, (e, condition) in zip(rows, jobs)):
        raise ValueError('Invalid completed teacher-scoring prefix')
    for e, condition in jobs[:len(rows)]:
        if (condition == 'all' and agent.config.train.arm in {'memory', 'direct_latent'}
                and e.episode_id not in plans_by_episode):
            raise ValueError('Missing original read plan for completed teacher score')
    def restored_plans(episode_id):
        return [[ReadPlan(**(p | {'selections': tuple(Selection(**s) for s in p['selections'])}))
                 for p in step] for step in plans_by_episode[episode_id]]
    with autocast_context(agent.config):
        for index, (e, condition) in enumerate(jobs):
            if index < len(rows):
                continue
            if want_stop is not None and want_stop():
                return None
            original_plans = (restored_plans(e.episode_id) if e.episode_id in plans_by_episode else None)
            arm = agent.config.train.arm
            visible = evidence_ids(e, agent.config.train.evidence_scope)
            evidence = '\n'.join(s.text for s in e.supports if s.record_id in visible)
            prompt = agent.prompt_ids(e.query, evidence if arm == 'oracle_text' and condition != 'none' else '')
            memory, selected = None, []
            if arm in {'memory', 'direct_latent'} and condition != 'none':
                namespace = ('wrong_values' if condition == 'wrong_values' else 'all') + '/' + e.environment
                with runtime.compute_watchdog(agent.config.train.stall_timeout_seconds,
                                              device=agent.config.train.device):
                    session = read_session(agent, store, prompt, namespace=namespace, generation='teacher-eval-v1',
                                           query_time=e.query_time,
                                           oracle_ids=visible if agent.config.train.retrieval == 'oracle' else None,
                                           ablate_values=condition == 'zero_values',
                                           cluster_bank=cluster_bank if condition == 'compact' else None,
                                           fixed_plans=None if condition == 'all' else
                                               [[replace(p, namespace=namespace) for p in step] for step in original_plans])
                memory, selected = session.memory, session.selected_ids
                if condition == 'all':
                    plans_by_episode[e.episode_id] = [[asdict(p) for p in step] for step in session.plans]
            elif arm == 'shared_compute':
                with runtime.compute_watchdog(agent.config.train.stall_timeout_seconds,
                                              device=agent.config.train.device):
                    memory = agent.shared_compute_tokens(prompt)
            target = agent.target_ids(e.answer)
            with runtime.compute_watchdog(agent.config.train.stall_timeout_seconds,
                                          device=agent.config.train.device):
                nll = float(agent.conditioned_nll(prompt, target, memory, reduction='sum'))
            r = dict(episode=e.episode_id, trajectory=e.environment, dataset=e.provenance.get('dataset'),
                     condition=condition, token_count=target.numel(), sequence_nll=nll, mean_nll=nll / target.numel(),
                     selected_ids=selected, support_annotation=e.support_annotation)
            if condition == 'compact':
                r['payload_accounting'] = session.payload_accounting
            if generate_tokens:
                with runtime.compute_watchdog(agent.config.train.stall_timeout_seconds,
                                              device=agent.config.train.device):
                    prediction = agent.generate_from_memory(prompt, memory, max_new_tokens=generate_tokens)
                r.update(prediction=prediction, reference_exact_match=prediction.strip() == e.answer.strip())
            rows.append(r)
            if progress is not None:
                progress(rows, plans_by_episode)
    summary = {}
    for c in conditions:
        subset = [r for r in rows if r['condition'] == c]
        nll = sum(r['sequence_nll'] for r in subset) / sum(r['token_count'] for r in subset)
        summary[c] = dict(episodes=len(subset), token_weighted_nll=nll,
                          perplexity=math.exp(nll) if nll < 50 else None,
                          macro_mean_nll=sum(r['mean_nll'] for r in subset) / len(subset))
    return dict(protocol='Write-once/serialize/reload, time-filtered per-trajectory stored-only reads.',
                payload_intervention_routing='Captured original IDs and scores at every read; no intervention rerouting.',
                summary=summary, paired_memory_nll_benefit=paired_nll_benefit(rows), rows=rows,
                notice='Teacher likelihood is not task success. Payload interventions affect latent arms only; '
                       'supplied context is not annotated sufficient. No environment commands are executed.')


def evaluate_teacher_run(run, episodes_file, *, max_episodes=64, generate_tokens=0, output=None):
    if max_episodes < 1 or generate_tokens < 0:
        raise ValueError('Invalid evaluation limits')
    config = config_from_run(Path(run))
    from .runtime import available_host_memory, configure_memory
    configure_memory(config.train)
    checkpoint = resolve_checkpoint(run, verify=True)
    index = EpisodeIndex(episodes_file)
    ids = random.Random(config.train.seed + 101).sample(range(len(index)), min(max_episodes, len(index)))
    episodes = [index[i] for i in ids]
    if not episodes:
        raise ValueError('Nonempty teacher evaluation set required')
    directory = Path(output) if output else Path(run).parent / (Path(run).name + '-teacher-eval')
    identity = dict(version=1, checkpoint_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    checkpoint_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file), evaluator_sha256=file_sha256(__file__),
                    max_episodes=max_episodes, generate_tokens=generate_tokens,
                    sampled_indices=ids)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory), stop_on_signal() as signals:
        identity_file = directory / 'inputs.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text()) != identity:
                raise ValueError('Teacher evaluation identity changed')
        elif any(directory.iterdir()):
            raise ValueError('Unidentified teacher evaluation directory; use a new output path')
        else:
            atomic_json(identity_file, identity)
        result_file = directory / 'results.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            return {k: v for k, v in result.items() if k != 'rows'} | {
                'status': 'complete', 'directory': str(directory)}
        progress_file = directory / 'progress.json'
        prior = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if prior and prior.get('inputs') != identity:
            raise ValueError('Teacher evaluation progress identity changed')
        rows, plans = prior.get('rows', []), prior.get('plans', {})
        def want_stop():
            available = available_host_memory()
            return bool(signals['signal'] or stop_requested(directory) or
                        (available is not None and available < config.train.min_system_available_bytes) or
                        shutil.disk_usage(directory).free < config.train.min_free_disk_bytes)
        def save_progress(current_rows, current_plans):
            nonlocal rows, plans
            rows, plans = list(current_rows), dict(current_plans)
            atomic_json(progress_file, dict(inputs=identity, rows=current_rows, plans=current_plans))
        if want_stop():
            save_progress(rows, plans)
            return {'status': 'checkpointed', 'directory': str(directory)}
        reset_resource_peaks(config.train.device)
        torch.set_num_threads(config.train.threads)
        agent = SchnitzelAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        store = DiskStore(directory / 'bank.sqlite')
        writes = build_teacher_bank(agent, store, episodes,
                                    writer_identity=identity['checkpoint_model_sha256'], want_stop=want_stop)
        if not writes['complete'] or want_stop():
            save_progress(rows, plans)
            return {'status': 'checkpointed', 'directory': str(directory)}
        bank, code_manifest = None, None
        if config.memory.compaction != 'none' and config.train.arm == 'memory':
            from .evaluation import build_persistent_codes
            grouped = defaultdict(list)
            for e in episodes:
                grouped[e.environment].append(e)
            skipped = 0
            for environment, group in grouped.items():
                if want_stop():
                    save_progress(rows, plans)
                    return {'status': 'checkpointed', 'directory': str(directory)}
                bank, manifest = build_persistent_codes(agent, store, group,
                    namespace='all/' + environment, generation='teacher-eval-v1')
                skipped += manifest['skipped_overlapping_or_unprofitable_groups']
            code_manifest = bank.sizes() | {'skipped_overlapping_or_unprofitable_groups': skipped}
        result = stored_teacher_evaluation(agent, DiskStore(store.path), episodes,
                                           generate_tokens=generate_tokens, cluster_bank=bank,
                                           rows=rows, saved_plans=plans,
                                           progress=save_progress, want_stop=want_stop)
        if result is None:
            save_progress(rows, plans)
            return {'status': 'checkpointed', 'directory': str(directory)}
        if code_manifest is not None:
            result['persistent_codes'] = code_manifest
        result.update(write_phase=writes, resources=resource_report(), store=store.sizes())
        atomic_json(result_file, result)
        return {k: v for k, v in result.items() if k != 'rows'} | {
            'status': 'complete', 'directory': str(directory)}
