"""Backbone preflight: execute public embeddings, causality and writer/read gradients."""
from __future__ import annotations

import math
import torch

from .agent import SchnitzelAgent, answer_distribution_kl
from .routing import cosine_scores, group_plan_loss
from .training import autocast_context, environment_report, resource_report


def model_probe(config) -> dict:
    config.validate()
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    from .runtime import configure_memory
    configure_memory(config.train)
    agent = SchnitzelAgent(config).to(config.train.device)
    if config.train.warmstart_memory_gate is not None:
        value = config.train.warmstart_memory_gate
        with torch.no_grad():
            agent.backbone.bridge.memory_logit.fill_(math.log(value / (1 - value)))
    agent.eval()
    ids = agent.text_ids('An earlier experience has a retry rule.', source=True)
    with torch.no_grad(), autocast_context(config):
        embeddings = agent.backbone.embed(ids)
        mask = torch.ones_like(ids)
        base = agent.backbone.base.core(embeddings, mask)
        once = agent.backbone.hidden(embeddings, mask, loops=1)
        twice = agent.backbone.hidden(embeddings, mask, loops=2)
        error_one = float((base - once).abs().max())
        error_two = float((once - twice).abs().max())
        cut = embeddings.shape[1] // 2
        changed = embeddings.clone()
        changed[:, cut:] += torch.randn_like(changed[:, cut:]) * .1
        modified = agent.backbone.hidden(changed, mask, loops=2)
        causal_error = float((twice[:, :cut] - modified[:, :cut]).abs().max())
    tolerance = .03 if config.train.precision == 'bf16' else 1e-4
    native = config.model.recurrence_mode == 'middle_block'
    identity_error = max(error_one, causal_error) if native else max(error_one, error_two, causal_error)
    if not all(torch.isfinite(x).all() for x in (base, once, twice, modified)):
        raise AssertionError('Nonfinite recurrent preflight states')
    if identity_error > tolerance:
        raise AssertionError(f'Backbone identity/causality failed: {error_one}, {error_two}, {causal_error}')
    agent.train()
    if native and config.memory.read_timing == 'loop_boundary':
        agent.backbone.loops = max(2, config.memory.read_steps + 1, config.model.loops)
    with autocast_context(config):
        records = [agent.produce(agent.text_ids(text, source=True)) for text in
                   ['Prior experience: restore state before retry.', 'Prior experience: retry is permitted.']]
        if config.memory.independent_routing_query and config.train.retrieval == 'learned':
            records.append(agent.produce(agent.text_ids('Unrelated prior experience: a different task.', source=True)))
        target = agent.target_ids('Restore and retry.')
        prompt = agent.prompt_ids('What should happen next?')
        result = agent(prompt, target,
                       records, [0, 1], arm='memory')
        objective = result.loss
        routing_loss = None
        if len(config.memory.payload_dims) > 1:
            # Oracle delivery does not train addressing. Probe each space with a
            # separate causal-prefix routing objective and a competing record.
            distractor = agent.produce(agent.text_ids('Unrelated prior experience: a different task.', source=True))
            _, routing_query = agent.query_pair(prompt)
            routing_terms = []
            for space, query_map in enumerate(agent.query_maps):
                keys = torch.cat([record[2 * space] for record in (*records[:2], distractor)], dim=0)
                scores = cosine_scores(query_map(routing_query), keys)[0]
                routing_terms.append(group_plan_loss(scores, [(0, 1)]))
            routing_loss = torch.stack(routing_terms).mean()
            objective = objective + .1 * routing_loss
        if config.train.oracle_distillation_weight:
            with torch.no_grad():
                teacher_logits = agent.conditioned_logits(
                    agent.prompt_ids('What should happen next?',
                                     'Prior experience: restore state before retry.\n'
                                     'Prior experience: retry is permitted.'),
                    target, None, loops=1)
            objective = objective + config.train.oracle_distillation_weight * answer_distribution_kl(
                agent.backbone.logits(result.answer_states), teacher_logits)
    objective.backward()
    gradients = {}
    gradient_parameters = [('write_slots', agent.write_slots), ('value_head', agent.value_head[-1].weight),
                            ('reader_output', (agent.reader.local[0] if hasattr(agent.reader, 'local') else agent.reader).output[-1].weight), ('memory_gate', agent.memory_gate)]
    if len(config.memory.payload_dims) > 1:
        for space, (codec, reader) in enumerate(zip(agent.codecs, agent.reader.local, strict=True)):
            codec_weight = (codec.weight if isinstance(codec, torch.nn.Linear)
                            else getattr(getattr(codec, 'projection', None), 'weight', None))
            if codec_weight is not None:
                gradient_parameters.append((f'codec_{space}', codec_weight))
            gradient_parameters.extend((
                (f'reader_local_{space}', reader.blocks[0].input.weight),
                (f'address_map_{space}', agent.address_maps[space].weight),
                (f'query_map_{space}', agent.query_maps[space].weight),
            ))
            if hasattr(reader.blocks[0], 'source_position'):
                gradient_parameters.extend((
                    (f'reader_source_positions_{space}', reader.blocks[0].source_position),
                    (f'reader_target_positions_{space}', reader.blocks[0].target_position),
                ))
        gradient_parameters.append(('reader_fusion_0', agent.reader.fusion[0].weight))
    if native and config.memory.read_timing == 'loop_boundary':
        gradient_parameters = [(name, parameter) for name, parameter in gradient_parameters
                               if name != 'memory_gate'] + [
            ('bridge_reentry', agent.backbone.bridge.reentry.weight),
            ('bridge_input_gate', agent.backbone.bridge.input_logit),
            ('bridge_update_gate', agent.backbone.bridge.update_logit),
            ('bridge_memory_projection', agent.backbone.bridge.memory_projection.weight),
            ('bridge_memory_gate', agent.backbone.bridge.memory_logit)]
    if config.memory.independent_routing_query and config.train.retrieval == 'learned':
        gradient_parameters.append(('routing_query_head', agent.routing_query_head.weight))
    for name, parameter in gradient_parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f'Missing/nonfinite soft-memory gradient: {name}')
        gradients[name] = float(parameter.grad.norm())
        if gradients[name] <= 0:
            raise AssertionError(f'Zero soft-memory gradient: {name}')
    return {'environment': environment_report(), 'backend': config.model.backend, 'model_id': config.model.model_id,
            'resolved_revision': agent.resolved_revision, 'width': agent.width,
            'parameters': sum(p.numel() for p in agent.parameters()),
            'one_loop_identity_max_error': error_one,
            'two_loop_delta_max': error_two,
            'two_loop_identity_required': not native,
            'zero_gate_two_loop_max_error': None if native else error_two,
            'recurrence': agent.backbone.manifest() if native else {'mode': 'full_stack'},
            'causal_prefix_max_error': causal_error, 'nll': float(result.nll.detach()),
            'routing_probe_loss': None if routing_loss is None else float(routing_loss.detach()),
            'gradient_norms': gradients, 'resources': resource_report(),
            'notice': 'Numerical preflight only; no capability training or cache-performance claim.'}
