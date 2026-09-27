"""Reuse deterministic serial decoder results within one immutable model evaluation.

Callers must still perform their authorized stored read before supplying memory.
The cache owns no store and cannot authorize or substitute a read plan.
"""
from copy import deepcopy
import hashlib

import torch

from .evaluation import score_answers
from .recurrence import LoopMemory


class FrozenScorer:
    """Exact input-byte keys; model parameters/config must remain immutable."""
    def __init__(self, agent):
        self.agent = agent
        self.scores, self.generations = {}, {}
        self.hits = {'score': 0, 'generation': 0}
        self._check()
        self._parameter_state = self._state()

    def _state(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.agent.parameters())

    def _check(self):
        if self.agent.training or any(p.requires_grad for p in self.agent.parameters()):
            raise ValueError('Decoder reuse requires a frozen eval-mode model')
        if hasattr(self, '_parameter_state') and self._state() != self._parameter_state:
            raise ValueError('Frozen decoder parameters changed; create a new scorer')

    def _key(self, prompt, memory):
        self._check()
        digest = hashlib.sha256()
        def tensor(value):
            if value is None:
                digest.update(b'none;')
                return
            digest.update(f'{value.dtype}:{tuple(value.shape)};'.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        tensor(prompt)
        if isinstance(memory, LoopMemory):
            digest.update(f'loop:{memory.prefix_length}:{memory.slots}:{memory.loops};'.encode())
            for event in memory.events:
                tensor(event)
        elif memory is None or isinstance(memory, torch.Tensor):
            digest.update(b'prefix;')
            tensor(memory)
        else:
            raise TypeError('Unsupported captured memory type')
        device = self.agent.device.type
        digest.update(str((str(self.agent.device), self.agent.backbone.loops,
                           torch.is_autocast_enabled(device), torch.get_autocast_dtype(device),
                           torch.get_float32_matmul_precision())).encode())
        return digest.digest()

    def score(self, prompt, memory, answer, choices):
        key = (self._key(prompt, memory), answer, tuple(choices))
        if key not in self.scores:
            self.scores[key] = score_answers(self.agent, prompt, memory, answer, choices)
        else:
            self.hits['score'] += 1
        return deepcopy(self.scores[key])

    def generate(self, prompt, memory, *, max_new_tokens):
        key = (self._key(prompt, memory), max_new_tokens)
        if key not in self.generations:
            self.generations[key] = self.agent.generate_from_memory(prompt, memory, max_new_tokens=max_new_tokens)
        else:
            self.hits['generation'] += 1
        return self.generations[key]

    def report(self):
        return {'hits': self.hits.copy(), 'unique_score_inputs': len(self.scores),
                'unique_generation_inputs': len(self.generations),
                'notice': 'Serial decoder results reused only after matching full prompt/memory bytes and scoring controls; '
                          'stored reads and visibility checks still execute.'}
