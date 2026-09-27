"""Source-bound, address-only overlays for frozen routing-probe evaluation.

These are inference adapters, not standard resumable training checkpoints. The
probe's optimizer state remains in its own external artifact.
"""
from pathlib import Path

from safetensors.torch import load_model
import torch

from .agent import SchnitzelAgent
from .runtime import configure_memory
from .trajectories import file_sha256


def load_frozen_agent(config, checkpoint, *, routing_probe=None, independent_routing_query=False):
    configure_memory(config.train)
    checkpoint = Path(checkpoint)
    state = None
    if routing_probe is not None:
        state = torch.load(routing_probe, weights_only=True, map_location='cpu')
        identity = state['identity']
        if identity['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json'):
            raise ValueError('Routing probe source checkpoint differs')
        if state['step'] != identity['steps']:
            raise ValueError('Routing probe endpoint is incomplete')
        if config.memory.independent_routing_query or len(config.memory.payload_dims) != 1:
            raise ValueError('Probe overlay requires a shared-query, single-space source')
        if set(state['model']) != {'key.weight', 'address.weight', 'query_map.weight', 'query_head.weight'}:
            raise ValueError('Unexpected probe parameter ownership')
        config.memory.independent_routing_query = independent_routing_query
    elif independent_routing_query:
        raise ValueError('Supply a routing probe for this evaluation override')
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    missing, unexpected = load_model(agent, str(checkpoint / 'model.safetensors'), strict=False,
                                     device=config.train.device)
    expected = {'routing_query_head.weight'} if state is not None and independent_routing_query else set()
    if set(missing) != expected or unexpected:
        raise ValueError('Frozen source model topology differs')
    if state is None:
        return agent, None
    weights = state['model']
    if not independent_routing_query and not torch.equal(agent.query_head.weight.detach().cpu(), weights['query_head.weight']):
        raise ValueError('Shared adapter would change the frozen reader query')
    for module, name in [(agent.key_head, 'key'), (agent.address_maps[0], 'address'),
                         (agent.query_maps[0], 'query_map')]:
        module.load_state_dict({'weight': weights[name + '.weight']})
    if independent_routing_query:
        agent.routing_query_head.load_state_dict({'weight': weights['query_head.weight']})
    return agent, {'probe_sha256': file_sha256(routing_probe), 'identity': identity,
                   'independent_routing_query': independent_routing_query,
                   'notice': 'Address-only overlay; base reader, payload writer and decoder remain frozen.'}



def attach_read_count_policy(agent, checkpoint, path):
    """Attach a completed reader-query classifier, bound to the frozen base model."""
    state = torch.load(path, weights_only=True, map_location='cpu')
    identity = state['identity']
    if identity['source_identity']['checkpoint_manifest_sha256'] != file_sha256(Path(checkpoint) / 'manifest.json'):
        raise ValueError('Read-count policy source checkpoint differs')
    if state['step'] != identity['steps'] or identity.get('representation') != 'reader_query':
        raise ValueError('Read-count policy must be a completed reader-query classifier')
    choices = identity['choices']
    if choices != [1, 2]:
        raise ValueError('Only the validated one/two-record policy is supported')
    config = agent.config
    if config.train.arm != 'memory' or len(config.memory.payload_dims) != 1 or config.memory.read_steps != 1:
        raise ValueError('Read-count policy requires single-space, single-read memory inference')
    head = torch.nn.Linear(config.memory.key_dim, len(choices)).to(agent.device).eval()
    head.load_state_dict(state['head'])
    head.requires_grad_(False)
    agent.read_count_head, agent.read_count_choices = head, tuple(choices)
    return {'policy_sha256': file_sha256(path), 'choices': choices, 'representation': 'reader_query',
            'parameters': sum(p.numel() for p in head.parameters()),
            'source_manifest_sha256': identity['source_identity']['checkpoint_manifest_sha256'],
            'notice': 'Frozen count classifier; no support-group labels are used during inference.'}
