"""Stored-only causal read sessions. No source trajectory or producer is accepted."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import Tensor

from .store import ReadPlan, Selection
from .storage_contract import KeySearchBackend
from .recurrence import LoopMemory
from .routing import cosine_similarities
from .store import lookup_record


@dataclass
class ReadSession:
    memory: Tensor | LoopMemory | None
    plans: list[list[ReadPlan]]
    selected_ids: list[list[str]]
    query_keys: list[Tensor]
    payload_accounting: list[dict] = field(default_factory=list)
    gate_weights: list[list[Tensor]] = field(default_factory=list)


def _gate_weights(agent, store: KeySearchBackend, *, query: Tensor, routing_query: Tensor,
                  selected_ids: list[list[str]], candidate_plans: list[ReadPlan]) -> list[Tensor] | None:
    """Evaluate learned continuous gates over a captured materialized field."""
    if not agent.config.memory.distance_gating:
        return None
    result = []
    for space, (ids, candidates) in enumerate(zip(selected_ids, candidate_plans, strict=True)):
        if not ids:
            result.append(query.new_empty(0))
            continue
        scope = dict(namespace=candidates.namespace, space=f's{space}',
                     generation=candidates.generation, domain=candidates.domain,
                     query_time=candidates.query_time)
        candidate_ids = list(dict.fromkeys(item.record_id for item in candidates.selections))
        candidate_ids.extend(record_id for record_id in ids if record_id not in candidate_ids)
        keys = torch.stack([lookup_record(store, record_id, **scope).key
                            for record_id in candidate_ids]).to(agent.device)
        address = agent.routing_address(routing_query, space)
        scores = cosine_similarities(address, keys)[0]
        positions = [candidate_ids.index(record_id) for record_id in ids]
        weights, _ = agent.distance_gates[space](
            address, scores[positions][None], scores[None],
            torch.ones((1, len(ids)), dtype=torch.bool, device=agent.device),
            torch.ones((1, len(candidate_ids)), dtype=torch.bool, device=agent.device))
        result.append(weights[0])
    return result


@torch.no_grad()
def read_session(agent, store: KeySearchBackend, prompt: Tensor, *, namespace: str,
                 generation: str, query_time: int, domain: str = "research",
                 oracle_ids: tuple[str, ...] | None = None,
                 ablate_values: bool = False, compact: bool = False,
                 exclude_ids: frozenset[str] = frozenset(), cluster_bank=None,
                 fixed_plans: list[list[ReadPlan]] | None = None,
                 fixed_gate_weights: list[list[Tensor]] | None = None) -> ReadSession:
    """Gather a captured read plan at each safe boundary, then update working slots.

    Learned retrieval searches the whole selected namespace and excludes previous
    IDs. Oracle mode supplies a fixed evidence plan, not target-dependent subsets.
    The cumulative payloads are reread at every boundary to revalidate visibility.
    This correctness path bounds *working slots*, not host payloads or I/O cost.
    """
    if agent.training:
        raise ValueError("Stored sessions require eval mode")
    r = agent.config.memory
    if fixed_gate_weights == []:
        fixed_gate_weights = None
    if fixed_plans is not None:
        validate_fixed_plans(fixed_plans, r, namespace, generation, domain, query_time, exclude_ids)
    if fixed_gate_weights is not None and (
            fixed_plans is None or len(fixed_gate_weights) != len(fixed_plans)):
        raise ValueError('Captured gate weights require one entry per fixed read plan')
    if r.distance_gating and (r.stream_reads or cluster_bank is not None
                              or agent.config.train.arm == 'direct_latent'):
        raise ValueError('Adaptive distance gates currently require materialized MLP/attention payloads')
    if cluster_bank is not None and (r.stream_reads or len(r.payload_dims) != 1 or compact):
        raise ValueError("Persistent codes require the single-space materialized reader, without re-compaction")
    if r.read_timing == "loop_boundary":
        if compact or r.stream_reads:
            raise ValueError("In-loop sessions require materialized payloads without runtime compaction")
        return loop_read_session(agent, store, prompt, namespace=namespace, generation=generation,
                                 query_time=query_time, domain=domain, oracle_ids=oracle_ids,
                                 ablate_values=ablate_values, exclude_ids=exclude_ids, fixed_plans=fixed_plans,
                                 fixed_gate_weights=fixed_gate_weights, cluster_bank=cluster_bank)
    if r.stream_reads and compact:
        raise ValueError("Streamed compaction is not implemented; use raw streaming or materialized compaction")
    selected = [[] for _ in r.payload_dims]
    memory, plans, keys, accounting, captured_weights = None, [], [], [], []
    for read in range(r.read_steps):
        if fixed_plans is not None and read >= len(fixed_plans):
            break
        q, routing_query = agent.query_pair(prompt, memory)
        keys.append(routing_query.detach().cpu())
        step_plans, candidate_plans, progressed = [], [], False
        payloads = []
        for space, dim in enumerate(r.payload_dims):
            if fixed_plans is not None:
                plan = fixed_plans[read][space]
            elif oracle_ids is None:
                requested = agent.requested_records(
                    q, r.neighbors[space] if r.read_steps == 1 else r.read_top_k)
                candidates = store.search(agent.routing_address(routing_query, space)[0], namespace=namespace,
                                    space=f"s{space}", generation=generation, domain=domain,
                                    query_time=query_time,
                                    top_k=max(requested, r.gate_density_k if r.distance_gating else 0),
                                    exclude_ids=exclude_ids | frozenset(selected[space]))
                plan = ReadPlan(namespace, f's{space}', generation, domain, query_time,
                                candidates.selections[:requested])
            else:
                available = [rid for rid in oracle_ids if rid not in selected[space] and rid not in exclude_ids]
                count = len(available) if r.read_steps == 1 else r.read_top_k
                plan = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                tuple(Selection(rid, 0.) for rid in available[:count]))
                candidates = plan
            if fixed_plans is not None:
                candidates = plan
            progressed = progressed or bool(plan.selections)
            selected[space].extend(s.record_id for s in plan.selections)
            step_plans.append(plan)
            candidate_plans.append(candidates)
            cumulative = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                  tuple(Selection(rid, 0.) for rid in selected[space]))
            if not r.stream_reads and cluster_bank is None:
                values = store.fetch(cumulative)
                payloads.append(torch.stack(values).float().to(agent.device) if values else q.new_empty(0, dim))
        if not progressed:
            break
        plans.append(step_plans)
        if cluster_bank is not None:
            packed = cluster_bank.fetch(cumulative)
            value = packed.values.float().to(agent.device)
            if ablate_values:
                value = torch.zeros_like(value)
            memory = agent.reader(value, q, packed.weights.to(agent.device)).tokens
            accounting.append({"clusters": packed.used_clusters, "raw_fallback_ids": packed.raw_fallback_ids,
                               "serialized_value_bytes": packed.serialized_value_bytes,
                               "logical_tensor_bytes": packed.logical_tensor_bytes})
        elif r.stream_reads:
            from .streaming import read_stream
            def chunks(plan=cumulative):
                for values, weights in store.iter_fetch(plan, r.chunk_size):
                    yield torch.zeros_like(values) if ablate_values else values, weights
            memory = read_stream(agent.reader, q, chunks).tokens
        elif agent.config.train.arm == "direct_latent":
            memory = payloads[0].reshape(1, -1, agent.width)
            if ablate_values:
                memory = torch.zeros_like(memory)
        else:
            weights = (fixed_gate_weights[read] if fixed_gate_weights is not None else
                       _gate_weights(agent, store, query=q, routing_query=routing_query,
                                     selected_ids=selected, candidate_plans=candidate_plans))
            memory, _ = agent.read_tokens(payloads, q, compact=compact,
                                          ablate_values=ablate_values, weights=weights)
            if weights is not None:
                captured_weights.append([weight.detach().cpu() for weight in weights])
    return ReadSession(memory, plans, selected, keys[:len(plans)], accounting,
                       captured_weights)


@torch.no_grad()
def loop_read_session(agent, store: KeySearchBackend, prompt: Tensor, *, namespace: str,
                      generation: str, query_time: int, domain: str,
                      oracle_ids: tuple[str, ...] | None, ablate_values: bool,
                      exclude_ids: frozenset[str], fixed_plans=None,
                      fixed_gate_weights=None, cluster_bank=None) -> ReadSession:
    """Prefix-only read-plan construction at actual recurrent core boundaries."""
    if agent.training:
        raise ValueError("Stored sessions require eval mode")
    r = agent.config.memory
    selected = [[] for _ in r.payload_dims]
    plans, keys, accounting, captured_weights, memory = [], [], [], [], None
    if agent.backbone.loops == 1:
        return ReadSession(None, [], selected, [])
    def provider(completed, query, routing_query):
        nonlocal memory
        if completed > r.read_steps or (fixed_plans is not None and completed > len(fixed_plans)):
            return memory
        step_plans, candidate_plans, payloads, progressed = [], [], [], False
        for space, dim in enumerate(r.payload_dims):
            if fixed_plans is not None:
                plan = fixed_plans[completed - 1][space]
            elif oracle_ids is None:
                requested = agent.requested_records(
                    query, r.neighbors[space] if r.read_steps == 1 else r.read_top_k)
                candidates = store.search(agent.routing_address(routing_query, space)[0], namespace=namespace,
                                    space=f"s{space}", generation=generation, domain=domain,
                                    query_time=query_time,
                                    top_k=max(requested, r.gate_density_k if r.distance_gating else 0),
                                    exclude_ids=exclude_ids | frozenset(selected[space]))
                plan = ReadPlan(namespace, f's{space}', generation, domain, query_time,
                                candidates.selections[:requested])
            else:
                available = [rid for rid in oracle_ids if rid not in selected[space] and rid not in exclude_ids]
                limit = len(available) if r.read_steps == 1 else r.read_top_k
                plan = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                tuple(Selection(rid, 0.) for rid in available[:limit]))
                candidates = plan
            if fixed_plans is not None:
                candidates = plan
            progressed = progressed or bool(plan.selections)
            selected[space].extend(s.record_id for s in plan.selections)
            step_plans.append(plan)
            candidate_plans.append(candidates)
            cumulative = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                  tuple(Selection(rid, 0.) for rid in selected[space]))
            if cluster_bank is None:
                values = store.fetch(cumulative)
                payloads.append(torch.stack(values).float().to(agent.device) if values else query.new_empty(0, dim))
        if progressed:
            plans.append(step_plans)
            keys.append(routing_query.detach().cpu())
            if cluster_bank is None:
                weights = (fixed_gate_weights[completed - 1]
                           if fixed_gate_weights is not None else
                           _gate_weights(agent, store, query=query,
                                         routing_query=routing_query,
                                         selected_ids=selected,
                                         candidate_plans=candidate_plans))
                memory, _ = agent.read_tokens(payloads, query,
                                              ablate_values=ablate_values, weights=weights)
                if weights is not None:
                    captured_weights.append([weight.detach().cpu() for weight in weights])
            else:
                packed = cluster_bank.fetch(cumulative)
                values = packed.values.float().to(agent.device)
                if ablate_values:
                    values = torch.zeros_like(values)
                memory = agent.reader(values, query, packed.weights.to(agent.device)).tokens
                accounting.append({"clusters": packed.used_clusters, "raw_fallback_ids": packed.raw_fallback_ids,
                                   "serialized_value_bytes": packed.serialized_value_bytes,
                                   "logical_tensor_bytes": packed.logical_tensor_bytes})
        return memory
    schedule = agent.plan_loop_memory(prompt, provider, include_routing_query=True)
    return ReadSession(schedule, plans, selected, keys, accounting, captured_weights)


def validate_fixed_plans(plans, memory_config, namespace, generation, domain, query_time, excluded):
    """Captured selections never grant visibility or permit changing read boundaries."""
    if len(plans) > memory_config.read_steps:
        raise ValueError('Captured plan exceeds configured read boundaries')
    seen = [set() for _ in memory_config.payload_dims]
    for step in plans:
        if len(step) != len(seen):
            raise ValueError('Captured plan has incompatible space boundaries')
        for space, plan in enumerate(step):
            if (plan.namespace, plan.space, plan.generation, plan.domain, plan.query_time) != (
                    namespace, f's{space}', generation, domain, query_time):
                raise ValueError('Captured plan crosses a namespace/version/time/authorization boundary')
            ids = [s.record_id for s in plan.selections]
            if len(ids) != len(set(ids)) or seen[space].intersection(ids) or set(ids).intersection(excluded):
                raise ValueError('Captured plan duplicates or includes excluded evidence')
            seen[space].update(ids)
