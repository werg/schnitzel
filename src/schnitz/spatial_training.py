"""Whole-trajectory recurrent training from pinned stored latent revisions."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from .agent import SchnitzelAgent
from .bank_replay import BankWriterReplay
from .key_geometry import BankLoad, koleo_loss, variance_covariance_loss
from .record_gradients import GradientSink
from .recurrence import SpatialReadSite
from .routing import cosine_scores, cosine_similarities, union_support_loss
from .routing_curriculum import (RoutingCandidateIndex, lexical_alignment_loss,
                                 routing_mix)
from .spatial_data import validate_spatial_row
from .store import ReadPlan, Selection
from .storage_contract import BatchKeyIndexBackend, StoredReadBackend


@dataclass
class SpatialForwardResult:
    loss: Tensor
    nll: Tensor
    routing: Tensor
    metrics: dict[str, Any]
    write_outputs: tuple[Tensor, ...] | None = None
    geometry: Tensor | None = None


def _trajectory_write_outputs(agent: SchnitzelAgent, hidden: Tensor,
                              rows: list[dict[str, Any]]) -> tuple[Tensor, ...] | None:
    states = []
    slots = agent.config.memory.write_slots + 1
    for row_index, row in enumerate(rows):
        for write in row.get('write_sites', ()):
            if 'workspace_start' not in write or row.get('write_slots') != slots - 1:
                raise ValueError('Authored trajectories require configured write workspaces')
            start = write['workspace_start']
            states.append(hidden[row_index, start:start + slots])
    if not states:
        return None
    return agent.produce_from_write_states(torch.stack(states))


@dataclass
class _SpatialBatchState:
    rows: list[dict[str, Any]]
    labels: Tensor
    execution: Any
    sites: tuple[SpatialReadSite, ...]
    site_indices: dict[int, tuple[int, ...]]
    routing_terms: list[Tensor]
    live_routing_terms: list[Tensor]
    easy_terms: list[Tensor]
    hard_terms: list[Tensor]
    lexical_terms: list[Tensor]
    key_stability_terms: list[Tensor]
    selected_counts: list[int]
    learned_hits: list[int]
    positive_labels: list[int]
    positive_hits: list[int]
    candidate_positive_hits: list[int]
    candidate_reciprocal_rank: list[float]
    gate_mass: list[float]
    gate_effective_records: list[float]
    gate_temperature: list[float]
    gate_radius: list[float]
    gate_support_share: list[float]
    gate_uniform_share: list[float]
    gate_gold_sites: list[int]
    variance_terms: list[Tensor]
    covariance_terms: list[Tensor]
    koleo_terms: list[Tensor]
    load_terms: list[Tensor]
    gate_records: list[dict[str, list[float]]] | None = None
    forced_sites: int = 0
    read_sites: int = 0


@dataclass(frozen=True)
class _ReadWave:
    active: tuple[SpatialReadSite, ...]
    query: Tensor
    addresses: tuple[Tensor, ...]
    search_addresses: tuple[Tensor, ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _LoadedSpace:
    candidate_ids: tuple[tuple[str, ...], ...]
    easy_ids: tuple[tuple[str, ...], ...]
    required_ids: tuple[tuple[str, ...], ...]
    found_ids: tuple[tuple[str, ...], ...]
    selected_ids: tuple[tuple[str, ...], ...]
    values: tuple[tuple[Tensor, ...], ...]
    forced: tuple[bool, ...] = ()
    supplied_ids: tuple[tuple[str, ...], ...] = ()


def _validate_layout(agent: SchnitzelAgent, rows: list[dict[str, Any]],
                     limits: tuple[int, ...], routing_candidates: int) -> tuple[int, tuple[int, ...]]:
    if not rows or routing_candidates < 1:
        raise ValueError("Spatial bank training needs rows and routing candidates")
    for row in rows:
        validate_spatial_row(row)
    memory = agent.config.memory
    if (len(limits) != len(memory.payload_dims)
            or any(limit < 1 or limit > cap
                   for limit, cap in zip(limits, memory.neighbors, strict=True))):
        raise ValueError("Spatial limits must fit every configured memory space")
    if any(row["read_slots"] != memory.read_slots for row in rows):
        raise ValueError("Packed and configured spatial slot counts differ")
    site_count = len(rows[0]["sites"])
    levels = tuple(site["level"] for site in rows[0]["sites"])
    if any(len(row["sites"]) != site_count
           or tuple(site["level"] for site in row["sites"]) != levels for row in rows):
        raise ValueError("A spatial batch needs one shared site and level layout")
    if any(len(site["required_ids"]) > min(limits)
           for row in rows for site in row["sites"]):
        raise ValueError("Every space must have room for all verified positives")
    return site_count, levels


def _begin_batch(agent: SchnitzelAgent, rows: list[dict[str, Any]], *,
                 limits: tuple[int, ...], routing_candidates: int,
                 pad_token_id: int) -> _SpatialBatchState:
    site_count, levels = _validate_layout(agent, rows, limits, routing_candidates)
    batch = len(rows)
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((batch, maximum), pad_token_id, dtype=torch.long,
                           device=agent.device)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[row_index, :length] = torch.tensor(row["input_ids"], device=agent.device)
        labels[row_index, :length] = torch.tensor(row["labels"], device=agent.device)
        attention[row_index, :length] = 1
    sites = tuple(SpatialReadSite(
        torch.tensor([row["sites"][site]["query_position"] for row in rows],
                     dtype=torch.long, device=agent.device),
        torch.tensor([row["sites"][site]["workspace_start"] for row in rows],
                     dtype=torch.long, device=agent.device),
        levels[site],
    ) for site in range(site_count))
    site_indices = {level: tuple(index for index, value in enumerate(levels)
                                 if value == level) for level in set(levels)}
    execution = agent.begin_spatial_recurrent(input_ids, attention, sites)
    return _SpatialBatchState(
        rows, labels, execution, sites, site_indices, [], [], [], [], [], [],
        [0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0.0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
        [], [], [], [],
        [_empty_gate_records() for _ in agent.config.memory.payload_dims],
    )


GATE_RECORD_GROUPS = ('forced_gold', 'retrieved_gold', 'other')


def _empty_gate_records() -> dict[str, list[float]]:
    """Per record group: [weight sum, dw/ds sum, count]."""
    return {group: [0.0, 0.0, 0.0] for group in GATE_RECORD_GROUPS}


def _gate_record_summary(records: list[dict[str, list[float]]]) -> dict[str, list[float | None]]:
    summary = {}
    for group in GATE_RECORD_GROUPS:
        summary[f'gate_weight_{group}'] = [
            space[group][0] / space[group][2] if space[group][2] else None for space in records]
        summary[f'gate_slope_{group}'] = [
            space[group][1] / space[group][2] if space[group][2] else None for space in records]
    return summary


def _issue_wave(agent: SchnitzelAgent, state: _SpatialBatchState) -> _ReadWave | None:
    while state.execution.recurrent.completed < state.execution.recurrent.loops:
        pending = agent.spatial_recurrent_query(state.execution)
        if pending is not None:
            active, query, routing_query = pending
            level = state.execution.recurrent.completed
            metadata = tuple(
                (state.rows[row]["sites"][site] | {
                    'routing_step': state.rows[row].get('routing_step', 0),
                    'training_step': state.rows[row].get('training_step', 0),
                    'gold_force_probability': state.rows[row].get(
                        'gold_force_probability',
                        agent.config.train.gold_force_probability)})
                for site in state.site_indices[level]
                for row in range(len(state.rows))
            )
            addresses = agent.routing_addresses(routing_query)
            # Synchronize only the small address matrices before handing work to
            # CPU threads. The differentiable device expressions remain retained.
            search_addresses = tuple(address.detach().float().cpu()
                                     for address in addresses)
            return _ReadWave(active, query, addresses, search_addresses, metadata)
        state.execution = agent.advance_empty_spatial_level(state.execution)
    return None


def _site_uniform(item: Mapping[str, Any], tag: str) -> float:
    """Reproducible uniform draw per read site, step and purpose."""
    text = '\0'.join((str(item.get('episode_id', '')), str(item.get('call_id', '')),
                      str(item.get('query_position', '')), str(item.get('training_step', 0)),
                      tag))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') / 2 ** 64


def _read_selection(found: tuple[str, ...], required: tuple[str, ...],
                    explored: tuple[str, ...], limit: int,
                    forced: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Top-ranked records, exploration in the tail, and optionally forced gold.

    Forced supports that were not retrieved displace the lowest-ranked
    non-support records, so every read keeps its size. Supports come first,
    as in the always-supplied reference.
    """
    chosen = list(found[:limit - len(explored)]) + [
        record_id for record_id in explored if record_id not in found[:limit]]
    chosen = chosen[:limit]
    missing = tuple(record_id for record_id in required if record_id not in chosen) if forced else ()
    while len(chosen) + len(missing) > limit:
        for position in range(len(chosen) - 1, -1, -1):
            if chosen[position] not in required:
                del chosen[position]
                break
        else:
            raise ValueError('A read cannot hold every forced support')
    chosen.extend(missing)
    support = [record_id for record_id in required if record_id in chosen]
    return tuple(support + [record_id for record_id in chosen if record_id not in support]), missing


def _load_wave(store: StoredReadBackend, index: BatchKeyIndexBackend, wave: _ReadWave, *,
               limits: tuple[int, ...], routing_candidates: int,
               routing_teacher: RoutingCandidateIndex | None = None,
               load: BankLoad | None = None, exploration_fraction: float = 0.0,
               ) -> tuple[_LoadedSpace, ...]:
    """Perform selection and serialized payload reads without touching the GPU graph."""
    loaded = []
    forced = tuple(_site_uniform(item, 'gold') < item['gold_force_probability']
                   for item in wave.metadata)
    for space, limit in enumerate(limits):
        name = f"s{space}"
        plans = index.search_batch(
            wave.search_addresses[space],
            top_k=max(limit, routing_candidates), namespace=index.namespace,
            space=name, generation=index.generation,
            domains=tuple(item["domain"] for item in wave.metadata),
            query_times=tuple(item["query_time"] for item in wave.metadata),
        )
        candidates, easy_rows, required_rows, found_rows, chosen_plans = [], [], [], [], []
        supplied_rows = []
        explore_count = (round(exploration_fraction * limit)
                         if load is not None and exploration_fraction else 0)
        for plan, item, force in zip(plans, wave.metadata, forced, strict=True):
            found = tuple(selection.record_id for selection in plan.selections)
            required = tuple(item["required_ids"])
            explored = ()
            if explore_count:
                seed = int(_site_uniform(item, f'explore-{space}') * 2 ** 62)
                eligible = (index.eligible_mask(name, domain=item['domain'],
                                                query_time=item['query_time'])
                            if hasattr(index, 'eligible_mask') else None)
                proposals = load.explore(space, 4 * explore_count, seed,
                                         exclude=set(found[:limit]) | set(required),
                                         eligible=eligible)
                explored = index.eligible_ids(
                    name, proposals, domain=item['domain'],
                    query_time=item['query_time'])[:explore_count]
            easy = ()
            if routing_teacher is not None:
                sampled = routing_teacher.sample(
                    item['episode_id'], item['routing_step'],
                    domain=item['domain'], query_time=item['query_time'], limit=32)
                in_batch = tuple(record_id for other in wave.metadata
                                 for record_id in other['required_ids']
                                 if other['episode_id'] != item['episode_id'])
                proposed = tuple(dict.fromkeys(sampled + in_batch))
                easy = index.eligible_ids(
                    name, proposed, domain=item['domain'],
                    query_time=item['query_time'])
            candidate_ids = tuple(dict.fromkeys(found + required + easy + explored))
            chosen, supplied = _read_selection(found, required, explored, limit, force)
            supplied_rows.append(supplied)
            candidates.append(candidate_ids)
            easy_rows.append(tuple(dict.fromkeys(required + easy)))
            required_rows.append(required)
            found_rows.append(found)
            chosen_plans.append(ReadPlan(
                index.namespace, name, index.generation, item["domain"],
                item["query_time"], tuple(Selection(record_id, 0.0)
                                          for record_id in chosen),
            ))
        values = store.fetch_many(chosen_plans)
        loaded.append(_LoadedSpace(
            tuple(candidates), tuple(easy_rows), tuple(required_rows), tuple(found_rows),
            tuple(tuple(selection.record_id for selection in plan.selections)
                  for plan in chosen_plans),
            tuple(tuple(row) for row in values),
            forced,
            tuple(supplied_rows),
        ))
    return tuple(loaded)


def _consume_wave(agent: SchnitzelAgent, index: BatchKeyIndexBackend,
                  state: _SpatialBatchState,
                  wave: _ReadWave, loaded: tuple[_LoadedSpace, ...], *,
                  limits: tuple[int, ...],
                  writer_replay: BankWriterReplay | None = None,
                  live: Mapping[str, tuple[Tensor, ...]] | None = None,
                  routing_teacher: RoutingCandidateIndex | None = None,
                  load: BankLoad | None = None,
                  sink: GradientSink | None = None) -> None:
    memory = agent.config.memory
    train = agent.config.train
    replay_budget = agent.config.train.writer_replay_records_per_site
    if live is None:
        live = {}
        if writer_replay is not None and replay_budget:
            live = writer_replay.capture(_wave_replay_ids(
                wave, loaded, replay_budget=replay_budget))
    payloads, weights = [], []
    support_scores = [[] for _ in wave.metadata]
    live_support_scores = [[] for _ in wave.metadata]
    support_indices = [[] for _ in wave.metadata]
    easy_scores = [[] for _ in wave.metadata]
    live_easy_scores = [[] for _ in wave.metadata]
    easy_indices = [[] for _ in wave.metadata]
    lexical_terms = [[] for _ in wave.metadata]
    for space, (width, limit, result) in enumerate(
            zip(memory.payload_dims, limits, loaded, strict=True)):
        address = wave.addresses[space]
        device_rows, weight_rows = [], []
        if (train.spread_variance_weight or train.spread_covariance_weight) and len(address) > 1:
            variance, covariance = variance_covariance_loss(address)
            state.variance_terms.append(variance)
            state.covariance_terms.append(covariance)
        for row_index, (candidate_ids, required, found, selected_ids, values, item) in enumerate(zip(
                result.candidate_ids, result.required_ids, result.found_ids,
                result.selected_ids, result.values, wave.metadata, strict=True)):
            stored_keys = index.keys_for_ids(
                f"s{space}", candidate_ids, domain=item["domain"],
                query_time=item["query_time"],
            ).to(agent.device)
            if sink is not None:
                # Same values as the synchronized index, now differentiable in the
                # key heads and in each record's cached writer key-slot state.
                stored_keys = sink.candidate_keys(space, candidate_ids)
                sink.near(found[:max(1, round(train.record_neighborhood_fraction * limit))]
                          + required)
            keys = torch.stack([
                live[record_id][2 * space][0] if record_id in live else stored_keys[position]
                for position, record_id in enumerate(candidate_ids)
            ])
            # Retrieval sorts by cosine; a sharper training softmax supplies a
            # useful gradient even when the candidate field is large. Reader
            # distance gates below continue to use unscaled cosine geometry.
            scores = (cosine_scores(address[row_index:row_index + 1], stored_keys)[0]
                      * agent.config.train.routing_logit_scale)
            live_scores = (cosine_scores(address[row_index:row_index + 1], keys)[0]
                           * agent.config.train.routing_logit_scale
                           if agent.config.train.routing_live_weight else None)
            if agent.config.train.key_stability_weight:
                state.key_stability_terms.extend(
                    1 - F.cosine_similarity(
                        live[record_id][2 * space][0].float()[None],
                        stored_keys[position].float()[None])[0]
                    for position, record_id in enumerate(candidate_ids)
                    if record_id in live)
            if train.koleo_weight and (sink is not None or any(
                    record_id in live for record_id in candidate_ids)):
                state.koleo_terms.append(koleo_loss(keys))
            if load is not None:
                if train.load_penalty_weight:
                    state.load_terms.append(load.penalty(
                        space, address[row_index:row_index + 1], keys, candidate_ids))
                load.record(space, found[:limit])
            hard_ids = tuple(dict.fromkeys(found + required))
            hard_positions = [candidate_ids.index(record_id) for record_id in hard_ids]
            support_scores[row_index].append(scores[hard_positions])
            if live_scores is not None:
                live_support_scores[row_index].append(live_scores[hard_positions])
            support_indices[row_index].append(
                tuple(hard_ids.index(record_id) for record_id in required))
            if routing_teacher is not None:
                easy_ids = result.easy_ids[row_index]
                easy_positions = [candidate_ids.index(record_id) for record_id in easy_ids]
                easy_scores[row_index].append(scores[easy_positions])
                if live_scores is not None:
                    live_easy_scores[row_index].append(live_scores[easy_positions])
                easy_indices[row_index].append(
                    tuple(easy_ids.index(record_id) for record_id in required))
                query_text = item.get('routing_query_text')
                if query_text is None:
                    raise ValueError('Routing teacher requires visible query text')
                lexical = routing_teacher.lexical_similarities(
                    query_text, candidate_ids, domain=item['domain'],
                    query_time=item['query_time'])
                lexical_terms[row_index].append(lexical_alignment_loss(
                    [scores], torch.tensor(lexical, dtype=scores.dtype,
                                           device=scores.device)))
            state.learned_hits[space] += int(set(required) <= set(found[:limit]))
            state.positive_labels[space] += len(required)
            state.positive_hits[space] += sum(record_id in found[:limit]
                                              for record_id in required)
            state.candidate_positive_hits[space] += sum(record_id in found
                                                        for record_id in required)
            state.candidate_reciprocal_rank[space] += sum(
                1 / (found.index(record_id) + 1) if record_id in found else 0
                for record_id in required)
            state.selected_counts[space] += len(values)
            row_values = torch.stack([
                (live[record_id][2 * space + 1][0] if record_id in live
                 else sink.payload(space, record_id, value) if sink is not None
                 else value.to(device=agent.device, dtype=torch.float32))
                for record_id, value in zip(selected_ids, values, strict=True)
            ])
            device_rows.append(row_values)
            if memory.distance_gating:
                selected_positions = [candidate_ids.index(record_id)
                                      for record_id in selected_ids]
                raw_scores = cosine_similarities(
                    address[row_index:row_index + 1], keys)[0]
                chosen_scores = raw_scores[selected_positions][None]
                candidate_raw = raw_scores[hard_positions][None]
                selected_mask = torch.ones_like(chosen_scores, dtype=torch.bool)
                candidate_mask = torch.ones_like(candidate_raw, dtype=torch.bool)
                required_set = set(required)
                required_mask = torch.tensor(
                    [[record_id in required_set for record_id in selected_ids]],
                    dtype=torch.bool, device=agent.device)
                row_weights, diagnostics = agent.distance_gates[space](
                    address[row_index:row_index + 1], chosen_scores, candidate_raw,
                    selected_mask, candidate_mask, required_mask,
                    agent.config.train.support_gate_floor)
                weight_rows.append(row_weights[0])
                supplied = set(result.supplied_ids[row_index]) if result.supplied_ids else set()
                record_weights = row_weights[0].detach().tolist()
                record_slopes = diagnostics['slope'][0].tolist()
                for record_id, weight, slope in zip(selected_ids, record_weights,
                                                    record_slopes, strict=True):
                    group = ('forced_gold' if record_id in supplied else
                             'retrieved_gold' if record_id in required_set else 'other')
                    totals = state.gate_records[space][group]
                    totals[0] += weight
                    totals[1] += slope
                    totals[2] += 1
                if bool(required_mask.any()):
                    detached = row_weights[0].detach()
                    state.gate_support_share[space] += float(
                        (detached * required_mask[0]).sum() / detached.sum().clamp_min(1e-12))
                    state.gate_uniform_share[space] += float(required_mask.float().mean())
                    state.gate_gold_sites[space] += 1
                state.gate_mass[space] += float(diagnostics['mass'].detach())
                state.gate_effective_records[space] += float(
                    diagnostics['effective_records'].detach())
                state.gate_temperature[space] += float(diagnostics['temperature'].detach())
                state.gate_radius[space] += float(diagnostics['radius'].detach())
            else:
                weight_rows.append(wave.query.new_ones(len(values)))
        count = max(map(len, result.values))
        if any(row.shape[1] != width for row in device_rows):
            raise ValueError("Stored spatial payload width differs from configuration")
        payloads.append(torch.stack([
            F.pad(row, (0, 0, 0, count - row.shape[0])) for row in device_rows
        ]))
        weights.append(torch.stack([
            torch.cat((row, wave.query.new_zeros(count - row.shape[0])))
            for row in weight_rows]))
    for row_index, (scores, indices) in enumerate(zip(
            support_scores, support_indices, strict=True)):
        hard = union_support_loss(scores, indices)
        state.hard_terms.append(hard)
        if routing_teacher is None:
            state.routing_terms.append(hard)
            if agent.config.train.routing_live_weight:
                state.live_routing_terms.append(union_support_loss(
                    live_support_scores[row_index], indices))
        else:
            easy = union_support_loss(easy_scores[row_index], easy_indices[row_index])
            state.easy_terms.append(easy)
            easy_weight, hard_weight = routing_mix(
                wave.metadata[row_index]['routing_step'],
                routing_teacher.ramp_steps)
            state.routing_terms.append(easy_weight * easy + hard_weight * hard)
            if agent.config.train.routing_live_weight:
                live_hard = union_support_loss(live_support_scores[row_index], indices)
                live_easy = union_support_loss(
                    live_easy_scores[row_index], easy_indices[row_index])
                state.live_routing_terms.append(
                    easy_weight * live_easy + hard_weight * live_hard)
            state.lexical_terms.append(torch.stack(lexical_terms[row_index]).mean())
    state.read_sites += len(wave.metadata)
    state.forced_sites += sum(loaded[0].forced) if loaded[0].forced else len(wave.metadata)
    results = agent._read_padded_batch(payloads, weights, wave.query)
    state.execution = agent.advance_spatial_recurrent(
        state.execution, wave.active, results)


def _wave_replay_ids(wave: _ReadWave, loaded: tuple[_LoadedSpace, ...], *,
                     replay_budget: int) -> tuple[str, ...]:
    required_priority, other_priority = [], []
    for row in range(len(wave.metadata)):
        required = loaded[0].required_ids[row]
        if len(required) > replay_budget:
            raise ValueError('Writer replay budget must cover every supplied support')
        required_priority.extend(required)
        for result in loaded:
            other_priority.extend(result.selected_ids[row])
    priority = required_priority + other_priority
    return tuple(dict.fromkeys(priority))[:replay_budget * len(wave.metadata)]


def _finish_batch(agent: SchnitzelAgent, state: _SpatialBatchState) -> SpatialForwardResult:
    hidden = agent.finish_spatial_recurrent(state.execution)
    write_outputs = _trajectory_write_outputs(agent, hidden, state.rows)
    supervised = state.labels >= 0
    logits = agent.backbone.logits(hidden[supervised]).float()
    nll = F.cross_entropy(logits, state.labels[supervised])
    expected_reads = len(state.rows) * len(state.sites)
    if not state.routing_terms or state.read_sites != expected_reads:
        raise ValueError("Every spatial site must execute exactly one bank read")
    stored_routing = torch.stack(state.routing_terms).mean()
    live_routing = (torch.stack(state.live_routing_terms).mean()
                    if state.live_routing_terms else stored_routing.new_zeros(()))
    lexical = (torch.stack(state.lexical_terms).mean()
               if state.lexical_terms else stored_routing.new_zeros(()))
    key_stability = (torch.stack(state.key_stability_terms).mean()
                     if state.key_stability_terms else stored_routing.new_zeros(()))
    hard_loss = torch.stack(state.hard_terms).mean()
    easy_loss = (torch.stack(state.easy_terms).mean()
                 if state.easy_terms else stored_routing.new_zeros(()))
    easy_weight, hard_weight = (routing_mix(
        state.rows[0].get('routing_step', 0),
        state.rows[0].get('routing_hard_ramp_steps', 3000))
                                if state.easy_terms else (0.0, 1.0))
    routing = (stored_routing + agent.config.train.routing_live_weight * live_routing
               + .05 * lexical
               + agent.config.train.key_stability_weight * key_stability)
    train = agent.config.train

    def mean(terms: list[Tensor]) -> Tensor:
        return torch.stack(terms).mean() if terms else stored_routing.new_zeros(())

    variance, covariance = mean(state.variance_terms), mean(state.covariance_terms)
    koleo, load_penalty = mean(state.koleo_terms), mean(state.load_terms)
    geometry = (train.spread_variance_weight * variance
                + train.spread_covariance_weight * covariance
                + train.koleo_weight * koleo + train.load_penalty_weight * load_penalty)
    loss = nll + train.routing_weight * routing + geometry
    denominator = state.read_sites
    metrics = {
        "read_sites": state.read_sites,
        "write_sites": sum(len(row.get("write_sites", ())) for row in state.rows),
        "selected_counts": [count / denominator for count in state.selected_counts],
        "learned_positive_recall": [hits / denominator for hits in state.learned_hits],
        "learned_positive_item_recall": [
            hits / labels for hits, labels in
            zip(state.positive_hits, state.positive_labels, strict=True)
        ],
        "candidate_positive_item_recall": [
            hits / labels for hits, labels in
            zip(state.candidate_positive_hits, state.positive_labels, strict=True)
        ],
        "candidate_positive_mrr": [
            total / labels for total, labels in
            zip(state.candidate_reciprocal_rank, state.positive_labels, strict=True)
        ],
        "positive_labels": state.positive_labels,
        "lexical_alignment": float(lexical.detach()),
        "routing_stored_loss": float(stored_routing.detach()),
        "routing_live_loss": float(live_routing.detach()),
        "key_stability_loss": float(key_stability.detach()),
        "routing_easy_loss": float(easy_loss.detach()),
        "routing_hard_loss": float(hard_loss.detach()),
        "routing_easy_weight": easy_weight,
        "routing_hard_weight": hard_weight,
        "gate_mass": [value / denominator for value in state.gate_mass],
        "gate_effective_records": [value / denominator
                                   for value in state.gate_effective_records],
        "gate_temperature": [value / denominator for value in state.gate_temperature],
        "gate_radius": [value / denominator for value in state.gate_radius],
        # Gate selectivity: share of gate mass on gold records present in the read,
        # against the share a uniform gate would give them.
        "gate_support_share": [value / max(count, 1) for value, count in
                               zip(state.gate_support_share, state.gate_gold_sites,
                                   strict=True)],
        "gate_uniform_share": [value / max(count, 1) for value, count in
                               zip(state.gate_uniform_share, state.gate_gold_sites,
                                   strict=True)],
        "gold_forced_fraction": state.forced_sites / denominator,
        "gate_record_totals": state.gate_records,
        **_gate_record_summary(state.gate_records),
        "spread_variance": float(variance.detach()),
        "spread_covariance": float(covariance.detach()),
        "koleo": float(koleo.detach()),
        "load_penalty": float(load_penalty.detach()),
        "selected_payload_bytes": sum(
            count * width * 2 for count, width in
            zip(state.selected_counts, agent.config.memory.payload_dims, strict=True)
        ) / len(state.rows),
        "supervised_tokens": int(supervised.sum()),
    }
    return SpatialForwardResult(loss, nll, routing, metrics, write_outputs, geometry)


def spatial_bank_forward(agent: SchnitzelAgent, store: StoredReadBackend,
                         index: BatchKeyIndexBackend,
                         rows: list[dict[str, Any]], *, limits: tuple[int, ...],
                         routing_candidates: int, pad_token_id: int = 0,
                         plan_observer: Callable[[str, str, ReadPlan], None] | None = None,
                         ) -> SpatialForwardResult:
    """Train many causal sites; same-level bank searches execute as one matrix scan.

    Supplied verified positives keep the reader useful while the address projections
    learn against global candidates. Only pinned serialized payload revisions enter
    the read path; a TrainingBank may publish refreshed revisions after the step.
    """
    if not rows or routing_candidates < 1:
        raise ValueError("Spatial bank training needs rows and routing candidates")
    for row in rows:
        validate_spatial_row(row)
    memory = agent.config.memory
    if (len(limits) != len(memory.payload_dims)
            or any(limit < 1 or limit > cap
                   for limit, cap in zip(limits, memory.neighbors, strict=True))):
        raise ValueError("Spatial limits must fit every configured memory space")
    if any(row["read_slots"] != memory.read_slots for row in rows):
        raise ValueError("Packed and configured spatial slot counts differ")
    site_count = len(rows[0]["sites"])
    levels = tuple(site["level"] for site in rows[0]["sites"])
    if any(len(row["sites"]) != site_count
           or tuple(site["level"] for site in row["sites"]) != levels for row in rows):
        raise ValueError("A spatial batch needs one shared site and level layout")
    if any(len(site["required_ids"]) > min(limits)
           for row in rows for site in row["sites"]):
        raise ValueError("Every space must have room for all verified positives")

    batch = len(rows)
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((batch, maximum), pad_token_id, dtype=torch.long,
                           device=agent.device)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[row_index, :length] = torch.tensor(row["input_ids"], device=agent.device)
        labels[row_index, :length] = torch.tensor(row["labels"], device=agent.device)
        attention[row_index, :length] = 1
    sites = tuple(SpatialReadSite(
        torch.tensor([row["sites"][site]["query_position"] for row in rows],
                     dtype=torch.long, device=agent.device),
        torch.tensor([row["sites"][site]["workspace_start"] for row in rows],
                     dtype=torch.long, device=agent.device),
        levels[site],
    ) for site in range(site_count))
    site_indices = {level: tuple(index for index, value in enumerate(levels) if value == level)
                    for level in set(levels)}
    routing_terms: list[Tensor] = []
    selected_counts = [0] * len(memory.payload_dims)
    learned_hits = [0] * len(memory.payload_dims)
    positive_labels = [0] * len(memory.payload_dims)
    positive_hits = [0] * len(memory.payload_dims)
    candidate_positive_hits = [0] * len(memory.payload_dims)
    candidate_reciprocal_rank = [0.0] * len(memory.payload_dims)
    gate_mass = [0.0] * len(memory.payload_dims)
    gate_effective_records = [0.0] * len(memory.payload_dims)
    gate_temperature = [0.0] * len(memory.payload_dims)
    gate_radius = [0.0] * len(memory.payload_dims)
    read_sites = 0

    def provider(level, active, query, routing_query):
        nonlocal read_sites
        indices = site_indices[level]
        if len(active) != len(indices):
            raise ValueError("Spatial provider site order changed")
        metadata = [rows[row]["sites"][site]
                    for site in indices for row in range(batch)]
        payloads, weights = [], []
        support_scores = [[] for _ in metadata]
        support_indices = [[] for _ in metadata]
        for space, (width, limit) in enumerate(
                zip(memory.payload_dims, limits, strict=True)):
            name = f"s{space}"
            address = agent.routing_address(routing_query, space)
            plans = index.search_batch(
                address, top_k=max(limit, routing_candidates),
                namespace=index.namespace, space=name, generation=index.generation,
                domains=tuple(item["domain"] for item in metadata),
                query_times=tuple(item["query_time"] for item in metadata),
            )
            chosen_plans, candidate_rows, selected_rows = [], [], []
            for row_index, (plan, item) in enumerate(zip(plans, metadata, strict=True)):
                found = [selection.record_id for selection in plan.selections]
                required = list(item["required_ids"])
                candidate_ids = list(dict.fromkeys(found + required))
                keys = index.keys_for_ids(name, candidate_ids, domain=item["domain"],
                                          query_time=item["query_time"]).to(agent.device)
                scores = cosine_scores(address[row_index:row_index + 1], keys)[0]
                positive_indices = tuple(candidate_ids.index(record_id)
                                         for record_id in required)
                support_scores[row_index].append(scores)
                support_indices[row_index].append(positive_indices)
                chosen = required + [record_id for record_id in found
                                     if record_id not in required]
                chosen = chosen[:limit]
                candidate_rows.append((keys, candidate_ids, required))
                selected_rows.append(chosen)
                learned_hits[space] += int(set(required) <= set(found[:limit]))
                positive_labels[space] += len(required)
                positive_hits[space] += sum(record_id in found[:limit]
                                            for record_id in required)
                candidate_positive_hits[space] += sum(record_id in found
                                                      for record_id in required)
                candidate_reciprocal_rank[space] += sum(
                    1 / (found.index(record_id) + 1) if record_id in found else 0
                    for record_id in required)
                selected_counts[space] += len(chosen)
                chosen_plans.append(ReadPlan(
                    index.namespace, name, index.generation, item["domain"],
                    item["query_time"], tuple(Selection(record_id, 0.0)
                                              for record_id in chosen),
                ))
                if plan_observer is not None:
                    plan_observer(item["call_id"], name, chosen_plans[-1])
            values = store.fetch_many(chosen_plans)
            count = max(map(len, values))
            device_rows = [torch.stack([
                value.to(device=agent.device, dtype=torch.float32) for value in row
            ])
                           for row in values]
            if any(row.shape[1] != width for row in device_rows):
                raise ValueError("Stored spatial payload width differs from configuration")
            payloads.append(torch.stack([
                F.pad(row, (0, 0, 0, count - row.shape[0])) for row in device_rows]))
            weight_rows = []
            for row_index, (value, chosen, candidate) in enumerate(zip(
                    device_rows, selected_rows, candidate_rows, strict=True)):
                if memory.distance_gating:
                    keys, candidate_ids, required = candidate
                    raw = cosine_similarities(
                        address[row_index:row_index + 1], keys)[0]
                    positions = [candidate_ids.index(record_id) for record_id in chosen]
                    support = set(required)
                    local, diagnostics = agent.distance_gates[space](
                        address[row_index:row_index + 1], raw[positions][None], raw[None],
                        torch.ones((1, len(chosen)), dtype=torch.bool, device=agent.device),
                        torch.ones((1, len(candidate_ids)), dtype=torch.bool,
                                   device=agent.device),
                        torch.tensor([[record_id in support for record_id in chosen]],
                                     dtype=torch.bool, device=agent.device),
                        agent.config.train.support_gate_floor)
                    row_weight = local[0]
                    gate_mass[space] += float(diagnostics['mass'].detach())
                    gate_effective_records[space] += float(
                        diagnostics['effective_records'].detach())
                    gate_temperature[space] += float(diagnostics['temperature'].detach())
                    gate_radius[space] += float(diagnostics['radius'].detach())
                else:
                    row_weight = query.new_ones(value.shape[0])
                weight_rows.append(torch.cat((row_weight,
                    query.new_zeros(count - value.shape[0]))))
            weights.append(torch.stack(weight_rows))
        routing_terms.extend(
            union_support_loss(scores, positive)
            for scores, positive in zip(support_scores, support_indices, strict=True))
        read_sites += len(metadata)
        return agent._read_padded_batch(payloads, weights, query)

    hidden = agent.spatial_recurrent_hidden(input_ids, attention, sites, provider)
    write_outputs = _trajectory_write_outputs(agent, hidden, rows)
    supervised = labels >= 0
    # The tied vocabulary projection is large. Blank workspaces, prompts, tool
    # results and padding have no target, so projecting them wastes both memory and
    # compute without changing the objective.
    logits = agent.backbone.logits(hidden[supervised]).float()
    nll = F.cross_entropy(logits, labels[supervised])
    if not routing_terms or read_sites != batch * site_count:
        raise ValueError("Every spatial site must execute exactly one bank read")
    routing = torch.stack(routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    denominator = read_sites
    metrics = {
        "read_sites": read_sites,
        "write_sites": sum(len(row.get("write_sites", ())) for row in rows),
        "selected_counts": [count / denominator for count in selected_counts],
        "learned_positive_recall": [hits / denominator for hits in learned_hits],
        "learned_positive_item_recall": [
            hits / labels for hits, labels in
            zip(positive_hits, positive_labels, strict=True)
        ],
        "candidate_positive_item_recall": [
            hits / labels for hits, labels in
            zip(candidate_positive_hits, positive_labels, strict=True)
        ],
        "candidate_positive_mrr": [
            total / labels for total, labels in
            zip(candidate_reciprocal_rank, positive_labels, strict=True)
        ],
        "positive_labels": positive_labels,
        "gate_mass": [value / denominator for value in gate_mass],
        "gate_effective_records": [value / denominator
                                   for value in gate_effective_records],
        "gate_temperature": [value / denominator for value in gate_temperature],
        "gate_radius": [value / denominator for value in gate_radius],
        "selected_payload_bytes": sum(count * width * 2 for count, width in
                                      zip(selected_counts, memory.payload_dims, strict=True))
                                  / batch,
        "supervised_tokens": int(supervised.sum()),
    }
    return SpatialForwardResult(loss, nll, routing, metrics, write_outputs)


def spatial_bank_pipeline_forward(
        agent: SchnitzelAgent, store: StoredReadBackend, index: BatchKeyIndexBackend,
        rows: list[dict[str, Any]], *, limits: tuple[int, ...],
        routing_candidates: int, microbatch_size: int, inflight: int,
        pad_token_id: int = 0,
        writer_replay: BankWriterReplay | None = None,
        routing_teacher: RoutingCandidateIndex | None = None,
        load: BankLoad | None = None,
        sink: GradientSink | None = None) -> SpatialForwardResult:
    """Overlap retained microbatch graphs with CPU search and serialized payload I/O.

    GPU continuations execute in deterministic round-robin order. Retrieval workers
    may finish out of order, but each captured plan remains attached to the query and
    causal metadata that produced it. The number of in-flight graphs is explicitly
    bounded, and retrieval never runs inside decoder checkpoint recomputation.
    """
    if microbatch_size < 1 or inflight < 1:
        raise ValueError("Pipeline microbatch size and in-flight count must be positive")
    _validate_layout(agent, rows, limits, routing_candidates)
    chunks = [rows[start:start + microbatch_size]
              for start in range(0, len(rows), microbatch_size)]
    if len(chunks) > 1 and inflight < 2:
        raise ValueError("Multiple pipeline microbatches require at least two in flight")
    results: list[tuple[int, SpatialForwardResult]] = []
    pending: deque[tuple[int, _SpatialBatchState, _ReadWave, float,
                         Future[tuple[tuple[_LoadedSpace, ...], float, float]]]] = deque()
    next_chunk = 0
    request_seconds, retrieval_seconds, ready_queue_seconds = [], [], []
    blocked_seconds = 0.0
    max_pending = 0

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        share = position - lower
        return ordered[lower] * (1 - share) + ordered[upper] * share

    def timed_load(wave: _ReadWave) -> tuple[tuple[_LoadedSpace, ...], float, float]:
        started = time.perf_counter()
        loaded = _load_wave(store, index, wave, limits=limits,
                            routing_candidates=routing_candidates,
                            routing_teacher=routing_teacher, load=load,
                            exploration_fraction=agent.config.train.exploration_fraction)
        return loaded, started, time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(inflight, len(chunks)),
                            thread_name_prefix="schnitz-bank-read") as pool:
        def launch(index_in_step: int) -> None:
            state = _begin_batch(
                agent, chunks[index_in_step], limits=limits,
                routing_candidates=routing_candidates, pad_token_id=pad_token_id)
            wave = _issue_wave(agent, state)
            if wave is None:
                results.append((index_in_step, _finish_batch(agent, state)))
                return
            issued = time.perf_counter()
            future = pool.submit(timed_load, wave)
            pending.append((index_in_step, state, wave, issued, future))

        while next_chunk < len(chunks) and len(pending) < inflight:
            launch(next_chunk)
            next_chunk += 1
            max_pending = max(max_pending, len(pending))
        while pending:
            level = pending[0][1].execution.recurrent.completed
            wavefront = []
            while pending and pending[0][1].execution.recurrent.completed == level:
                index_in_step, state, wave, issued, future = pending.popleft()
                wait_started = time.perf_counter()
                loaded, started, completed = future.result()
                consumed = time.perf_counter()
                blocked_seconds += consumed - wait_started
                request_seconds.append(completed - issued)
                retrieval_seconds.append(completed - started)
                ready_queue_seconds.append(max(0.0, consumed - completed))
                wavefront.append((index_in_step, state, wave, loaded))
            live = None
            if writer_replay is not None:
                replay_budget = agent.config.train.writer_replay_records_per_site
                record_ids = tuple(dict.fromkeys(
                    record_id
                    for _, _, wave, loaded in wavefront
                    for record_id in _wave_replay_ids(
                        wave, loaded, replay_budget=replay_budget)
                ))
                live = writer_replay.capture(record_ids)
            for index_in_step, state, wave, loaded in wavefront:
                _consume_wave(agent, index, state, wave, loaded, limits=limits,
                              live=live, routing_teacher=routing_teacher, load=load,
                              sink=sink)
                following = _issue_wave(agent, state)
                if following is None:
                    results.append((index_in_step, _finish_batch(agent, state)))
                    if next_chunk < len(chunks):
                        launch(next_chunk)
                        next_chunk += 1
                else:
                    issued = time.perf_counter()
                    future = pool.submit(timed_load, following)
                    pending.append((index_in_step, state, following, issued, future))
                max_pending = max(max_pending, len(pending))

    ordered = [result for _, result in sorted(results)]
    supervised = sum(result.metrics["supervised_tokens"] for result in ordered)
    read_sites = sum(result.metrics["read_sites"] for result in ordered)
    batch = len(rows)
    nll = sum(result.nll * result.metrics["supervised_tokens"]
              for result in ordered) / supervised
    routing = sum(result.routing * result.metrics["read_sites"]
                  for result in ordered) / read_sites
    geometry = sum(result.geometry * result.metrics["read_sites"]
                   for result in ordered) / read_sites
    loss = nll + agent.config.train.routing_weight * routing + geometry
    spaces = len(agent.config.memory.payload_dims)
    metrics = {
        "read_sites": read_sites,
        "write_sites": sum(result.metrics["write_sites"] for result in ordered),
        "selected_counts": [
            sum(result.metrics["selected_counts"][space]
                * result.metrics["read_sites"] for result in ordered) / read_sites
            for space in range(spaces)
        ],
        "learned_positive_recall": [
            sum(result.metrics["learned_positive_recall"][space]
                * result.metrics["read_sites"] for result in ordered) / read_sites
            for space in range(spaces)
        ],
        "learned_positive_item_recall": [
            sum(result.metrics["learned_positive_item_recall"][space]
                * result.metrics["positive_labels"][space] for result in ordered)
            / sum(result.metrics["positive_labels"][space] for result in ordered)
            for space in range(spaces)
        ],
        "candidate_positive_item_recall": [
            sum(result.metrics["candidate_positive_item_recall"][space]
                * result.metrics["positive_labels"][space] for result in ordered)
            / sum(result.metrics["positive_labels"][space] for result in ordered)
            for space in range(spaces)
        ],
        "candidate_positive_mrr": [
            sum(result.metrics["candidate_positive_mrr"][space]
                * result.metrics["positive_labels"][space] for result in ordered)
            / sum(result.metrics["positive_labels"][space] for result in ordered)
            for space in range(spaces)
        ],
        "positive_labels": [
            sum(result.metrics["positive_labels"][space] for result in ordered)
            for space in range(spaces)
        ],
        "lexical_alignment": sum(
            result.metrics["lexical_alignment"] * result.metrics["read_sites"]
            for result in ordered) / read_sites,
        **{
            name: sum(result.metrics[name] * result.metrics['read_sites']
                      for result in ordered) / read_sites
            for name in ('routing_stored_loss', 'routing_live_loss',
                         'key_stability_loss',
                         'routing_easy_loss', 'routing_hard_loss',
                         'routing_easy_weight', 'routing_hard_weight',
                         'gold_forced_fraction', 'spread_variance', 'spread_covariance',
                         'koleo', 'load_penalty')
        },
        **{
            name: [
                sum(result.metrics[name][space] * result.metrics["read_sites"]
                    for result in ordered) / read_sites
                for space in range(spaces)
            ] for name in ("gate_mass", "gate_effective_records",
                           "gate_temperature", "gate_radius",
                           "gate_support_share", "gate_uniform_share")
        },
        **_gate_record_summary([
            {group: [sum(result.metrics['gate_record_totals'][space][group][i]
                         for result in ordered) for i in range(3)]
             for group in GATE_RECORD_GROUPS}
            for space in range(spaces)]),
        "selected_payload_bytes": sum(
            result.metrics["selected_payload_bytes"] * len(chunk)
            for result, chunk in zip(ordered, chunks, strict=True)
        ) / batch,
        "supervised_tokens": supervised,
        "pipeline_microbatches": len(chunks),
        "pipeline_inflight": min(inflight, len(chunks)),
        "pipeline_max_pending": max_pending,
        "pipeline_retrieval_seconds_p50": percentile(retrieval_seconds, .50),
        "pipeline_retrieval_seconds_p95": percentile(retrieval_seconds, .95),
        "pipeline_retrieval_seconds_p99": percentile(retrieval_seconds, .99),
        "pipeline_request_seconds_p95": percentile(request_seconds, .95),
        "pipeline_ready_queue_seconds_p95": percentile(ready_queue_seconds, .95),
        "pipeline_blocked_seconds": blocked_seconds,
    }
    write_outputs = None
    if any(result.write_outputs is not None for result in ordered):
        if any(result.write_outputs is None for result in ordered):
            raise ValueError('Pipeline chunks disagree about authored write workspaces')
        write_outputs = tuple(torch.cat([result.write_outputs[index]
                                         for result in ordered], 0)
                              for index in range(len(ordered[0].write_outputs)))
    return SpatialForwardResult(loss, nll, routing, metrics, write_outputs, geometry)
