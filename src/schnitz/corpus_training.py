"""Fixed-plan, stored-payload training reads from a published frozen bank."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .agent import ForwardResult, SchnitzelAgent
from .data import Episode
from .recurrence import LoopMemory, LoopWrite
from .routing import cosine_scores, cosine_similarities, group_plan_loss
from .store import ReadPlan, Selection, lookup_record
from .storage_contract import KeySearchBackend


def stored_corpus_forward(agent: SchnitzelAgent, store: KeySearchBackend, episode: Episode, *,
                          generation: str, limits: tuple[int, ...],
                          namespace: str = 'corpus', searcher=None) -> tuple[ForwardResult, dict]:
    """Plan each space from a causal native prefix, then train on fetched values.

    Known sufficient supports are delivered during this initial mixed-selection
    stage; retrieval is trained against the global candidates in the same read.
    No source text reaches the writer or the consumer prompt on this path.
    """
    if (agent.config.memory.read_timing != 'loop_boundary' or agent.backbone.loops != 2 or
            len(limits) != len(agent.config.memory.payload_dims) or
            any(k < len(episode.required_ids) or k > cap for k, cap in
                zip(limits, agent.config.memory.neighbors, strict=True))):
        raise ValueError('Stored corpus training requires one native read and valid per-space limits')
    if episode.support_annotation != 'verified' or not 1 <= len(episode.required_ids) <= 4:
        raise ValueError('Corpus routing needs 1..4 verified sufficient supports')
    if any(source.created_at >= episode.query_time for source in episode.supports):
        raise ValueError('Corpus sources must precede their query')
    domain = episode.provenance.get('domain', 'research')
    contrast = agent.config.train.bank_payload_contrast_weight > 0
    if contrast and len(episode.required_ids) != 1:
        raise ValueError('Stored-bank source swap currently requires one verified positive')
    searcher = searcher or store
    prompt, target = agent.prompt_ids(episode.query), agent.target_ids(episode.answer)
    routing_terms = []
    selected_ids, learned_recalls, swapped_ids = [], [], []
    swapped_tokens = None

    def provider(completed, query, routing_query):
        nonlocal swapped_tokens
        if completed != 1:
            return None
        payloads, swapped_payloads, relevance_weights = [], [], []
        for space, (dim, limit) in enumerate(zip(agent.config.memory.payload_dims, limits, strict=True)):
            space_name = f's{space}'
            address = agent.routing_address(routing_query, space)
            candidates = searcher.search(address[0], top_k=max(
                                      limit + int(contrast),
                                      agent.config.train.bank_routing_candidates), namespace=namespace,
                                      space=space_name, generation=generation, domain=domain,
                                      query_time=episode.query_time)
            found = [selection.record_id for selection in candidates.selections]
            # Every candidate, including supplied positives, is fetched under
            # the same pinned physical snapshot, authorization and time scope.
            candidate_ids = list(dict.fromkeys(found + list(episode.required_ids)))
            keys = torch.stack([lookup_record(store, record_id, namespace=namespace,
                                space=space_name, generation=generation, domain=domain,
                                query_time=episode.query_time).key for record_id in candidate_ids]).to(agent.device)
            scores = cosine_scores(address, keys)[0]
            positive_indices = tuple(candidate_ids.index(record_id) for record_id in episode.required_ids)
            routing_terms.append(group_plan_loss(scores, [positive_indices]))
            chosen_ids = list(episode.required_ids)
            chosen_ids.extend(record_id for record_id in found if record_id not in chosen_ids)
            chosen_ids = chosen_ids[:limit]
            selection = ReadPlan(namespace, space_name, generation, domain, episode.query_time,
                                 tuple(Selection(record_id, 0.0) for record_id in chosen_ids))
            values = store.fetch(selection)
            payloads.append(torch.stack([value.to(agent.device).float() for value in values]))
            if agent.config.memory.distance_gating:
                raw = cosine_similarities(address, keys)[0]
                positions = [candidate_ids.index(record_id) for record_id in chosen_ids]
                support = set(episode.required_ids)
                local, _ = agent.distance_gates[space](
                    address, raw[positions][None], raw[None],
                    torch.ones((1, len(chosen_ids)), dtype=torch.bool,
                               device=agent.device),
                    torch.ones((1, len(candidate_ids)), dtype=torch.bool,
                               device=agent.device),
                    torch.tensor([[record_id in support for record_id in chosen_ids]],
                                 dtype=torch.bool, device=agent.device),
                    agent.config.train.support_gate_floor)
                relevance_weights.append(local[0])
            else:
                relevance_weights.append(query.new_ones(len(chosen_ids)))
            if contrast:
                wrong_id = next((record_id for record_id in found
                                 if record_id not in chosen_ids
                                 and record_id not in episode.required_ids), None)
                if wrong_id is None:
                    raise ValueError('Stored-bank contrast needs an eligible unselected source')
                wrong = store.fetch(ReadPlan(namespace, space_name, generation, domain,
                    episode.query_time, (Selection(wrong_id, 0.0),)))[0]
                swapped = payloads[-1].clone()
                swapped[0] = wrong.to(agent.device).float()
                swapped_payloads.append(swapped)
                swapped_ids.append(wrong_id)
            selected_ids.append(chosen_ids)
            learned_recalls.append(float(set(episode.required_ids) <= set(found[:limit])))
            if payloads[-1].shape != (len(chosen_ids), dim):
                raise ValueError('Stored payload width or selected count changed')
        tokens, _ = agent.read_tokens(payloads, query, weights=relevance_weights)
        if contrast:
            swapped_tokens, _ = agent.read_tokens(swapped_payloads, query,
                                                  weights=relevance_weights)
        return tokens

    memory = agent.plan_loop_memory(prompt, provider, include_routing_query=True)
    nll = agent.conditioned_nll(prompt, target, memory)
    contrast_info = {}
    if contrast:
        wrong_memory = LoopMemory(memory.prefix_length, memory.slots,
                                  memory.loops, (swapped_tokens,))
        wrong_nll = agent.conditioned_nll(prompt, target, wrong_memory)
        contrast_info = {'contrast_loss': F.softplus(
            agent.config.train.bank_payload_contrast_margin + nll - wrong_nll),
            'swapped_source_nll': wrong_nll, 'swapped_ids': swapped_ids,
            'swapped_payload_bytes': sum(dim * 2 for dim in agent.config.memory.payload_dims)}
    routing = torch.stack(routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    result = ForwardResult(loss, nll, routing, nll * 0,
                           [list(range(len(ids))) for ids in selected_ids], raw_nll=nll,
                           read_count=1)
    return result, {'selected_ids': selected_ids,
                    'selected_counts': [len(ids) for ids in selected_ids],
                    'learned_positive_recall': learned_recalls,
                    'selected_payload_bytes': sum(len(ids) * dim * 2 for ids, dim in
                                                  zip(selected_ids, agent.config.memory.payload_dims, strict=True)),
                    'supplied_positive': True} | contrast_info


def stored_corpus_forward_batch(agent: SchnitzelAgent, store: KeySearchBackend,
                                episodes: list[Episode], *, generation: str,
                                limits: tuple[int, ...], namespace: str = 'corpus',
                                searcher=None, token_rows: list[dict] | None = None):
    """Batched consumer compute with per-example exact bank search and fetching."""
    if (not episodes or agent.backbone.loops != 2 or agent.config.memory.read_steps != 1
            or agent.config.memory.compaction != 'none'
            or agent.config.train.bank_payload_contrast_weight):
        raise ValueError('Batched bank training requires R=2, one raw read and no contrast')
    searcher = searcher or store
    batch, r = len(episodes), agent.config.memory
    if token_rows is not None and len(token_rows) != batch:
        raise ValueError('Token row batch differs from episode batch')
    prompts, targets = [], []
    for index, episode in enumerate(episodes):
        row = token_rows[index] if token_rows is not None else None
        if row is not None and row['episode_id'] != episode.episode_id:
            raise ValueError('Pretokenized bank row differs from episode data')
        prompts.append(torch.tensor([row['prompt_ids']], dtype=torch.long, device=agent.device)
                       if row is not None else agent.prompt_ids(episode.query))
        targets.append(torch.tensor([row['target_ids']], dtype=torch.long, device=agent.device)
                       if row is not None else agent.target_ids(episode.answer))
    prompt_lengths = torch.tensor([x.shape[1] for x in prompts], device=agent.device)
    target_lengths = torch.tensor([x.shape[1] for x in targets], device=agent.device)
    sequences = [torch.cat((agent.backbone.embed(prompt), agent.loop_workspace[None],
                            agent.backbone.embed(target[:, :-1])), 1)
                 for prompt, target in zip(prompts, targets, strict=True)]
    totals = prompt_lengths + r.read_slots + target_lengths - 1
    maximum = int(totals.max())
    embeddings = torch.cat([F.pad(row, (0, 0, 0, maximum - row.shape[1]))[None]
                            if row.ndim == 2 else F.pad(row, (0, 0, 0, maximum-row.shape[1]))
                            for row in sequences], 0)
    mask = torch.arange(maximum, device=agent.device)[None] < totals[:, None]
    routing_terms, selected_ids = [], [[[] for _ in r.payload_dims] for _ in episodes]
    recalls = [[0. for _ in r.payload_dims] for _ in episodes]

    def boundary(completed, state, _anchor):
        positions = prompt_lengths + r.read_slots - 1
        features = agent.loop_query_norm(state[torch.arange(batch, device=agent.device), positions])
        query = F.normalize(agent.query_head(features), dim=-1)
        routing_query = agent.routing_input(features, query)
        payloads, weights = [], []
        for space, (dim, limit) in enumerate(zip(r.payload_dims, limits, strict=True)):
            rows, row_weights = [], []
            for row_index, episode in enumerate(episodes):
                domain = episode.provenance.get('domain', 'research')
                address = agent.routing_address(routing_query[row_index:row_index + 1], space)
                found = [item.record_id for item in searcher.search(
                    address[0], top_k=max(limit, agent.config.train.bank_routing_candidates),
                    namespace=namespace, space=f's{space}', generation=generation,
                    domain=domain, query_time=episode.query_time).selections]
                candidate_ids = list(dict.fromkeys(found + list(episode.required_ids)))
                keys = torch.stack([lookup_record(store, record_id, namespace=namespace,
                    space=f's{space}', generation=generation, domain=domain,
                    query_time=episode.query_time).key for record_id in candidate_ids]).to(agent.device)
                scores = cosine_scores(address, keys)[0]
                positives = tuple(candidate_ids.index(record_id) for record_id in episode.required_ids)
                routing_terms.append(group_plan_loss(scores, [positives]))
                chosen = list(episode.required_ids)
                chosen.extend(record_id for record_id in found if record_id not in chosen)
                chosen = chosen[:limit]
                plan = ReadPlan(namespace, f's{space}', generation, domain, episode.query_time,
                                tuple(Selection(record_id, 0.0) for record_id in chosen))
                rows.append(torch.stack([value.to(agent.device).float()
                                         for value in store.fetch(plan)]))
                if r.distance_gating:
                    raw = cosine_similarities(address, keys)[0]
                    positions = [candidate_ids.index(record_id) for record_id in chosen]
                    support = set(episode.required_ids)
                    local, _ = agent.distance_gates[space](
                        address, raw[positions][None], raw[None],
                        torch.ones((1, len(chosen)), dtype=torch.bool,
                                   device=agent.device),
                        torch.ones((1, len(candidate_ids)), dtype=torch.bool,
                                   device=agent.device),
                        torch.tensor([[record_id in support for record_id in chosen]],
                                     dtype=torch.bool, device=agent.device),
                        agent.config.train.support_gate_floor)
                    row_weights.append(local[0])
                else:
                    row_weights.append(query.new_ones(len(chosen)))
                selected_ids[row_index][space] = chosen
                recalls[row_index][space] = float(set(episode.required_ids) <= set(found[:limit]))
            count = max(value.shape[0] for value in rows)
            payloads.append(torch.cat([F.pad(value, (0, 0, 0, count-value.shape[0]))[None]
                                       for value in rows], 0))
            weights.append(torch.cat([
                torch.cat((weight, query.new_zeros(count - value.shape[0])))[None]
                for value, weight in zip(rows, row_weights, strict=True)], 0))
        memory = agent._read_padded_batch(payloads, weights, query)
        return LoopWrite(prompt_lengths, memory)

    hidden = agent.backbone.hidden(embeddings, mask.long(), boundary=boundary)
    losses = []
    for row, target in enumerate(targets):
        start = int(prompt_lengths[row]) + r.read_slots - 1
        logits = agent.backbone.logits(hidden[row:row + 1, start:start+target.shape[1]]).float()
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
    nll = torch.stack(losses).mean()
    routing = torch.stack(routing_terms).mean()
    result = ForwardResult(nll + agent.config.train.routing_weight * routing,
                           nll, routing, nll * 0, selected_ids[0], raw_nll=nll, read_count=1)
    counts = [sum(len(selected_ids[row][space]) for row in range(batch)) / batch
              for space in range(len(r.payload_dims))]
    mean_recalls = [sum(recalls[row][space] for row in range(batch)) / batch
                    for space in range(len(r.payload_dims))]
    payload_bytes = sum(len(selected_ids[row][space]) * dim * 2
                        for row in range(batch) for space, dim in enumerate(r.payload_dims)) / batch
    return result, {'selected_ids': selected_ids, 'selected_counts': counts,
                    'learned_positive_recall': mean_recalls,
                    'selected_payload_bytes': payload_bytes,
                    'supplied_positive': True}
