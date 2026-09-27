"""Write-once/many-use, globally heterogeneous, serialized-memory evaluation."""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import json
import hashlib
import time

import torch
from safetensors.torch import load_model

from .agent import SchnitzelAgent
from .checkpoints import resolve_checkpoint
from .data import Episode, load_episodes, counterfactual_boolean, counterfactual_multiuse, evidence_ids
from .metrics import summarize_rows, paired_world_bootstrap
from .routing import complete_support_recall
from .sessions import read_session
from .store import DiskStore
from .training import autocast_context, output_records, stored_channel, config_from_run, resource_report, reset_resource_peaks


@torch.no_grad()
def score_answers(agent, prompt, memory, answer: str, choices: tuple[str, ...]) -> dict:
    """Primary choice score is full sequence log probability, including EOS.

    Retain mean-token NLL and mean-score choices as separate diagnostics rather
    than silently treating them as the same evaluation (v0.1 used mean scores).
    """
    candidates = choices or (answer,)
    total, means = [], []
    for candidate in candidates:
        target = agent.target_ids(candidate)
        loss = agent.conditioned_nll(prompt, target, memory, reduction='sum')
        total.append(float(loss))
        means.append(float(loss) / target.numel())
    truth = candidates.index(answer)
    best = min(range(len(candidates)), key=lambda i: total[i])
    mean_best = min(range(len(candidates)), key=lambda i: means[i])
    return {'target_mean_nll': means[truth], 'target_sequence_nll': total[truth],
            'predicted_action': candidates[best] if choices else None,
            'choice_correct': candidates[best] == answer if choices else None,
            'mean_score_prediction': candidates[mean_best] if choices else None,
            'choice_sequence_nll': dict(zip(candidates, total, strict=True))}


@torch.no_grad()
def build_shared_bank(agent, store: DiskStore, episodes: list[Episode], *, namespace: str = 'global',
                      generation: str = 'frozen-v1', writer_identity: str | None = None) -> dict:
    """Canonical source IDs are written exactly once across all future queries."""
    if agent.training:
        raise ValueError('Writer must be frozen in eval mode')
    unique = {}
    for episode in episodes:
        for source in episode.supports:
            if source.record_id in unique and unique[source.record_id] != source:
                raise ValueError('Source ID maps to incompatible experiences')
            unique[source.record_id] = source
    if agent.config.train.arm in {'memory', 'direct_latent'}:
        def records():
            with autocast_context(agent.config):
                for source in unique.values():
                    result = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
                    yield from output_records(agent, source, result, namespace, generation)
        if writer_identity is None:
            store.put_many(records())
            created = True
        else:
            from .offline_bank import canonical_json, ensure_offline_records
            if not writer_identity:
                raise ValueError('Verified writer identity cannot be empty')
            identity = {'version': 1, 'writer': writer_identity,
                        'sources_sha256': hashlib.sha256(canonical_json(
                            [asdict(unique[rid]) for rid in sorted(unique)]).encode()).hexdigest(),
                        'model': asdict(agent.config.model), 'memory': asdict(agent.config.memory),
                        'max_source_tokens': agent.config.train.max_source_tokens,
                        'compute_precision': agent.config.train.precision}
            created = ensure_offline_records(store, records, identity=identity,
                namespace=namespace, generation=generation,
                spaces=tuple(f's{i}' for i in range(len(agent.config.memory.payload_dims))),
                expected_count=len(unique) * len(agent.config.memory.payload_dims))
        writes = len(unique) if created else 0
    else:
        writes = 0
    return {'unique_sources': len(unique), 'writer_calls': writes, 'queries': len(episodes),
            'worlds': len({e.environment for e in episodes})}


@torch.no_grad()
def stored_transfer_evaluation(agent, store: DiskStore, episodes: list[Episode], *,
                               namespace: str = 'global', generation: str = 'frozen-v1',
                               compact: bool = False, drop_supports: bool = False, cluster_bank=None,
                               fixed_plans_by_episode=None, capture_plans=None,
                               full_evidence_only: bool = False, progress=None,
                               use_codes_for_all_conditions: bool = False) -> dict:
    """Never calls the writer. Retrieval competes across worlds in a shared bank."""
    if agent.training:
        raise ValueError('Evaluation requires frozen weights')
    if use_codes_for_all_conditions and cluster_bank is None:
        raise ValueError('A stored cluster bank is required for code conditions')
    rows = []
    with autocast_context(agent.config):
        for index, episode in enumerate(episodes, 1):
            original_plans = (None if fixed_plans_by_episode is None else
                              [[replace(p, namespace=namespace) for p in step]
                               for step in fixed_plans_by_episode[episode.episode_id]])
            original_gate_weights = None
            conditions = ['all']
            if not full_evidence_only:
                conditions.extend(('none', 'zero_values'))
                if compact:
                    conditions.append('compact')
                if cluster_bank is not None and not use_codes_for_all_conditions:
                    conditions.append('persistent')
                if drop_supports:
                    conditions.extend(f'drop_{i}' for i in range(len(episode.required_ids)))
            for condition in conditions:
                arm = agent.config.train.arm
                excluded = frozenset([episode.required_ids[int(condition.split('_')[1])]]) if condition.startswith('drop_') else frozenset()
                selected = tuple(rid for rid in evidence_ids(episode, agent.config.train.evidence_scope) if rid not in excluded)
                if condition == 'none':
                    selected = ()
                text = '\n'.join(s.text for s in episode.supports if s.record_id in selected)
                prompt = agent.prompt_ids(episode.query, text if arm == 'oracle_text' else '')
                memory, selected_spaces, plans = None, [[] for _ in agent.config.memory.payload_dims], []
                if arm in {'memory', 'direct_latent'} and condition != 'none':
                    session = read_session(agent, store, prompt, namespace=namespace, generation=generation,
                                           query_time=episode.query_time,
                                           oracle_ids=selected if agent.config.train.retrieval == 'oracle' else None,
                                           exclude_ids=excluded, ablate_values=condition == 'zero_values',
                                           compact=condition == 'compact',
                                           cluster_bank=cluster_bank if condition == 'persistent' or use_codes_for_all_conditions else None,
                                           fixed_plans=original_plans if condition in {'all', 'zero_values', 'compact', 'persistent'} else None,
                                           fixed_gate_weights=original_gate_weights
                                           if condition in {'zero_values', 'compact', 'persistent'} else None)
                    memory, selected_spaces, plans = session.memory, session.selected_ids, session.plans
                    if condition == 'all':
                        original_plans = plans
                        original_gate_weights = session.gate_weights
                        if capture_plans is not None:
                            capture_plans[episode.episode_id] = plans
                elif arm == 'shared_compute':
                    memory = agent.shared_compute_tokens(prompt)
                elif arm == 'oracle_text':
                    selected_spaces = [list(selected)]
                score = score_answers(agent, prompt, memory, episode.answer, episode.choices)
                selected_set = set().union(*(set(ids) for ids in selected_spaces))
                groups = [set(g) for g in episode.sufficient_groups or (episode.required_ids,)]
                rows.append({'episode': episode.episode_id, 'environment': episode.environment,
                             'task_family': episode.task_family, 'condition': condition,
                             'answer': episode.answer, **score, 'selected_ids': selected_spaces,
                             'read_count': len(plans),
                             'payload_accounting': session.payload_accounting if arm in {'memory', 'direct_latent'} and condition != 'none' else [],
                             'complete_support': complete_support_recall(selected_set, groups)})
            if progress is not None and (index % 32 == 0 or index == len(episodes)):
                progress({'completed_queries': index, 'total_queries': len(episodes), 'scored_rows': len(rows)})
    families = {family: summarize_rows([r for r in rows if r['task_family'] == family])
                for family in sorted({r['task_family'] for r in rows})}
    return {'schema_version': 2, 'protocol': 'write-once, serialize/reload, global stored-only bank',
            'choice_scoring': 'sum of target token NLL including EOS',
            'notice': 'Synthetic transfer diagnostics, not capacity-substitution evidence.',
            'payload_intervention_routing': 'Captured original plans; payload changes never reroute later reads.',
            'rows': rows, 'summary': summarize_rows(rows), 'by_family': families,
            'paired_memory_benefit': paired_world_bootstrap(rows), 'resources': resource_report()}


def evaluate_transfer_run(run: str | Path, episodes_path: str | Path, *,
                          compact: bool = False, drop_supports: bool = False,
                          boolean_counterfactuals: bool = False, persistent_compact: bool = False,
                          binding_counterfactuals: bool = False) -> dict:
    if boolean_counterfactuals and binding_counterfactuals:
        raise ValueError('Choose one counterfactual family per evaluation')
    run = Path(run)
    config = config_from_run(run)
    reset_resource_peaks(config.train.device)
    if (compact or persistent_compact) and config.memory.compaction == 'none':
        raise ValueError('Run was not configured with a compactor')
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    from .runtime import configure_memory
    configure_memory(config.train)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    load_model(agent, str(resolve_checkpoint(run) / 'model.safetensors'), device=config.train.device)
    episodes = load_episodes(episodes_path)
    output = run / ('transfer-v2-' + str(time.time_ns()))
    output.mkdir()
    store = DiskStore(output / 'bank.sqlite')
    writes = build_shared_bank(agent, store, episodes)
    codes, code_manifest = None, None
    if persistent_compact:
        codes, code_manifest = build_persistent_codes(agent, store, episodes)
    original_plans = {}
    result = stored_transfer_evaluation(agent, DiskStore(store.path), episodes,
                                        compact=compact, drop_supports=drop_supports, cluster_bank=codes,
                                        capture_plans=original_plans)
    if code_manifest is not None:
        result["persistent_codes"] = code_manifest
    if boolean_counterfactuals or binding_counterfactuals:
        from .metrics import counterfactual_metrics
        original_answers = {e.episode_id: e.answer for e in episodes}
        names = ('a', 'b') if boolean_counterfactuals else ('restoration', 'permission')
        transform = counterfactual_boolean if boolean_counterfactuals else counterfactual_multiuse
        for bit in names:
            variants = [transform(e, bit) for e in episodes]
            build_shared_bank(agent, store, variants, namespace=f"flip-{bit}")
            variant_results = stored_transfer_evaluation(agent, DiskStore(store.path), variants,
                                                         namespace=f"flip-{bit}",
                                                         full_evidence_only=True,
                                                         fixed_plans_by_episode=original_plans if config.train.arm in {'memory', 'direct_latent'} else None)
            for row in variant_results["rows"]:
                if row["condition"] == "all":
                    row["condition"] = f"cf_{bit}"
                    row["counterfactual_should_change"] = row["answer"] != original_answers[row["episode"]]
                    result["rows"].append(row)
        conditions = tuple('cf_' + name for name in names)
        result["counterfactuals"] = counterfactual_metrics(result["rows"], names=conditions)
        if boolean_counterfactuals:
            result["xor_counterfactuals"] = counterfactual_metrics(
                [r for r in result["rows"] if r["task_family"] == "boolean/xor"], names=conditions)
        else:
            result['counterfactuals_by_family'] = {
                family: counterfactual_metrics([r for r in result['rows'] if r['task_family'] == family], names=conditions)
                for family in sorted({e.task_family for e in episodes})}
        result["counterfactual_write_note"] = "Separate offline variant banks; IDs/query/time fixed. Writer disabled during reads."
    result['write_phase'] = writes
    result['store'] = store.sizes()
    result['directory'] = str(output)
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


@torch.no_grad()
def build_persistent_codes(agent, store: DiskStore, episodes: list[Episode], *,
                           namespace='global', generation='frozen-v1', view_name='frozen-codes-v1'):
    from .cluster_store import ClusterBank, state_fingerprint
    from .store import ReadPlan, Selection
    if agent.config.train.arm != 'memory' or len(agent.config.memory.payload_dims) != 1:
        raise ValueError('Persistent prototype requires the single-space memory arm')
    reader_hash = state_fingerprint(agent.reader)
    bank = ClusterBank(store, view=view_name, reader_hash=reader_hash)
    seen, covered, skipped = set(), set(), 0
    with autocast_context(agent.config):
        for episode in episodes:
            ids = tuple(sorted(episode.required_ids))
            if ids in seen:
                continue
            seen.add(ids)
            size = 1 if agent.config.memory.compaction == 'mean' else agent.config.memory.compact_records
            if len(ids) <= size:
                skipped += 1
                continue
            if covered.intersection(ids):
                skipped += 1
                continue
            plan = ReadPlan(namespace, 's0', generation, 'research', episode.query_time,
                            tuple(Selection(rid, 0.) for rid in ids))
            raw = torch.stack(store.fetch(plan)).float().to(agent.device)[None]
            view = agent._compact_values(raw, raw.new_ones(raw.shape[:2]))
            if view.output_records >= view.input_records:
                skipped += 1
                continue
            values = view.values[0].to(getattr(torch, agent.config.memory.storage_dtype))
            existing = bank.fetch(plan)
            if existing.used_clusters:
                if existing.raw_fallback_ids:
                    raise ValueError('Existing compact view has a different cluster partition')
                torch.testing.assert_close(existing.values[0], values.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(existing.weights[0], view.weights[0].float().cpu(), rtol=0, atol=0)
            else:
                bank.put(plan, values, view.weights[0])
            covered.update(ids)
    return bank, bank.sizes() | {'skipped_overlapping_or_unprofitable_groups': skipped,
                                 'reader_hash': reader_hash,
                                 'scope': 'Full-cluster replacement; raw fallback for partial selections.'}
