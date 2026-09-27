"""Native PyTorch optimizers with explicit, resumable parameter ownership."""
from __future__ import annotations

import torch
from torch import nn


class MuonAdamW:
    """Muon for matrix transforms, AdamW for embeddings/heads and other tensors.

    Uses the installed PyTorch implementation, with no replacement CUDA stack
    or bgkit dependency. Component groups retain their independent learning rates.
    """
    def __init__(self, muon_groups, adam_groups, config):
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('Muon requires a PyTorch runtime providing torch.optim.Muon; '
                               'select adamw on older runtimes. Do not replace vendor Torch on Spark.')
        self.optimizers = {}
        if muon_groups:
            self.optimizers['muon'] = torch.optim.Muon(muon_groups,
                momentum=config.muon_momentum, ns_steps=config.muon_ns_steps,
                weight_decay=config.weight_decay, adjust_lr_fn='match_rms_adamw')
        if adam_groups:
            self.optimizers['adamw'] = torch.optim.AdamW(adam_groups,
                betas=tuple(config.adam_betas), eps=config.adam_eps, weight_decay=config.weight_decay)

    @property
    def param_groups(self):
        return [g for opt in self.optimizers.values() for g in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {'format': 1, 'kind': 'muon_adamw',
                'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()}}

    def load_state_dict(self, state):
        if state.get('kind') != 'muon_adamw' or state.get('format') != 1:
            raise ValueError('Checkpoint does not contain Muon/AdamW optimizer state')
        if set(state['optimizers']) != set(self.optimizers):
            raise ValueError('Muon/AdamW parameter ownership changed')
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])


def make_optimizer(agent):
    config = agent.config.train
    named = dict(agent.named_parameters())
    base = {id(p) for p in agent.backbone.base.parameters()}
    key_prefixes = ('key_head.', 'address_maps.', 'writer_key_heads.')
    if config.writer_key_learning_rate is None:
        groups = [dict(params=[p for p in named.values() if p.requires_grad
                               and (id(p) in base) == backbone],
                       lr=config.backbone_learning_rate if backbone
                       else config.learning_rate)
                  for backbone in (True, False)]
    else:
        groups = [
            dict(params=[p for p in named.values() if p.requires_grad and id(p) in base],
                 lr=config.backbone_learning_rate),
            dict(params=[p for name, p in named.items() if p.requires_grad
                         and id(p) not in base and not name.startswith(key_prefixes)],
                 lr=config.learning_rate),
            dict(params=[p for name, p in named.items() if p.requires_grad
                         and id(p) not in base and name.startswith(key_prefixes)],
                 lr=config.writer_key_learning_rate),
        ]
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(groups, betas=tuple(config.adam_betas),
                                      eps=config.adam_eps, weight_decay=config.weight_decay)
    else:
        excluded = {id(p) for module in agent.modules() if isinstance(module, nn.Embedding)
                    for p in module.parameters()}
        lm = getattr(agent.backbone.base, 'lm', None)
        if lm is not None:
            head = lm.get_output_embeddings()
            if head is not None:
                excluded.update(id(p) for p in head.parameters())
        # Learned slot/fallback-token tables are embeddings, not linear transforms.
        excluded.update(id(p) for name, p in named.items()
                        if name.endswith(('.slot', '_slots', 'workspace', '.null_tokens'))
                        or 'lora_' in name.lower())
        muon, adam = [], []
        for group in groups:
            matrices = [p for p in group['params'] if p.ndim == 2 and id(p) not in excluded]
            matrix_ids = {id(p) for p in matrices}
            others = [p for p in group['params'] if id(p) not in matrix_ids]
            if matrices:
                muon.append(dict(params=matrices, lr=group['lr']))
            if others:
                adam.append(dict(params=others, lr=group['lr']))
        optimizer = MuonAdamW(muon, adam, config)
    names_by_id = {id(p): name for name, p in named.items()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in named.values() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Optimizer must own each trainable parameter exactly once')
    optimizer._sdkb_parameter_names = [[names_by_id[id(p)] for p in g['params']]
                                       for g in optimizer.param_groups]
    return optimizer


def optimizer_report(optimizer):
    return [{'index': i, 'parameters': sum(p.numel() for p in group['params']),
             'tensors': len(group['params']),
             **{k: v for k, v in group.items() if k != 'params'}}
            for i, group in enumerate(optimizer.param_groups)]
