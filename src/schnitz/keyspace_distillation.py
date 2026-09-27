"""Head-only keyspace warmup over cached frozen writer and query states.

Every producer upstream of the key heads is frozen, so ``normalize(W_s h)`` over a
cached writer key-slot state ``h`` is the exact stored key, and every step can
score the whole causally eligible field with fresh keys.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

import numpy as np
import torch
from torch import Tensor


def direct_head_weights(state: dict[str, Tensor], spaces: int) -> dict[str, Tensor]:
    """Fold shared heads and per-space maps into equivalent direct heads.

    ``normalize(A normalize(W h)) == normalize(A W h)`` because the inner
    normalization is a positive rescaling and ``A`` has no bias. The folded
    heads start with zero bias.
    """
    writer, query = state['key_head.weight'].float(), state['query_head.weight'].float()
    if 'routing_query_head.weight' in state:
        query = state['routing_query_head.weight'].float()
    result = {}
    for space in range(spaces):
        writer_weight = state[f'address_maps.{space}.weight'].float() @ writer
        query_weight = state[f'query_maps.{space}.weight'].float() @ query
        result[f'writer_key_heads.{space}.weight'] = writer_weight
        result[f'writer_key_heads.{space}.bias'] = writer_weight.new_zeros(writer_weight.shape[0])
        result[f'query_key_heads.{space}.weight'] = query_weight
        result[f'query_key_heads.{space}.bias'] = query_weight.new_zeros(query_weight.shape[0])
    return result


def convert_to_direct(state: dict[str, Tensor], spaces: int,
                      key_dims: tuple[int, ...] = (), *, seed: int = 0) -> dict[str, Tensor]:
    """Replace shared key parameters; the reader keeps its own ``query_head``.

    Direct heads are fp32 parameters regardless of the parent's storage dtype.
    Wider keys keep the folded rows first; added rows start small and distinct,
    and distance-gate inputs for added coordinates start at zero.
    """
    heads = direct_head_weights(state, spaces)
    if key_dims:
        if len(key_dims) != spaces:
            raise ValueError('One key width is required per space')
        generator = torch.Generator().manual_seed(seed)
        for space, width in enumerate(key_dims):
            for kind in ('writer_key_heads', 'query_key_heads'):
                weight = heads[f'{kind}.{space}.weight']
                if width < weight.shape[0]:
                    raise ValueError('Key width conversion only widens keys')
                extra = torch.randn(width - weight.shape[0], weight.shape[1],
                                    generator=generator) * float(weight.std()) * .1
                heads[f'{kind}.{space}.weight'] = torch.cat((weight, extra), 0)
                heads[f'{kind}.{space}.bias'] = weight.new_zeros(width)
            gate = f'distance_gates.{space}.adjust.weight'
            if gate in state:
                old = state[gate].float()
                heads[gate] = torch.cat((old, old.new_zeros(old.shape[0],
                                                            width - old.shape[1])), 1)
    shared = ({'key_head.weight', 'routing_query_head.weight'}
              | {f'address_maps.{space}.weight' for space in range(spaces)}
              | {f'query_maps.{space}.weight' for space in range(spaces)})
    converted = {name: value for name, value in state.items() if name not in shared}
    converted.update(heads)  # includes any widened distance-gate inputs
    reference = heads['writer_key_heads.0.weight']
    for side in ('writer', 'query'):
        converted[f'{side}_key_state_mean'] = reference.new_zeros(reference.shape[1])
        converted[f'{side}_key_state_scale'] = reference.new_ones(reference.shape[1])
    return converted


class StandardizedHead:
    """Train ``W h + b`` as ``W' z + b'`` with fixed ``z = (h - mean) / scale``.

    The reparametrization is exact at initialization and folds back into a
    plain affine head. It conditions heads over nearly collinear states.
    """

    def __init__(self, weight: Tensor, bias: Tensor, mean: Tensor, scale: Tensor) -> None:
        self.mean, self.scale = mean, scale
        self.weight = (weight * scale).detach().clone().requires_grad_(True)
        self.bias = (bias + weight @ mean).detach().clone().requires_grad_(True)

    def standardize(self, states: Tensor) -> Tensor:
        return (states.float() - self.mean) / self.scale

    def __call__(self, standardized: Tensor) -> Tensor:
        return standardized @ self.weight.T + self.bias

    def folded(self) -> tuple[Tensor, Tensor]:
        weight = self.weight.detach() / self.scale
        return weight, self.bias.detach() - weight @ self.mean


def source_disjoint_split(site_supports: list[tuple[str, ...]], *, fraction: float,
                          seed: str) -> np.ndarray:
    """Hold out whole connected groups of sites that share any verified source."""
    if not 0 < fraction < 1:
        raise ValueError('Held-out fraction must be between zero and one')
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for supports in site_supports:
        if not supports:
            raise ValueError('Every warmup site needs verified supports')
        for other in supports[1:]:
            left, right = find(supports[0]), find(other)
            if left != right:
                parent[max(left, right)] = min(left, right)
    held = []
    for supports in site_supports:
        root = find(supports[0])
        value = int.from_bytes(hashlib.sha256(f'{seed}:{root}'.encode()).digest()[:8], 'little')
        held.append(value / 2 ** 64 < fraction)
    return np.asarray(held)


def field_kl(teacher_logits: Tensor, student_logits: Tensor, eligible: Tensor) -> Tensor:
    """Mean KL(teacher || student) over each row's eligible field."""
    if teacher_logits.shape != student_logits.shape or eligible.shape != student_logits.shape:
        raise ValueError('Teacher, student and eligibility fields must align')
    minimum = torch.finfo(torch.float32).min
    teacher = teacher_logits.float().masked_fill(~eligible, minimum).log_softmax(-1)
    student = student_logits.float().masked_fill(~eligible, minimum).log_softmax(-1)
    terms = teacher.exp() * (teacher - student)
    return terms.masked_fill(~eligible, 0).sum(-1).mean()


def flat_positives(positives: list[tuple[int, ...]], device) -> tuple[Tensor, Tensor]:
    """Row and column indices of every verified support in a batch."""
    rows = [row for row, support in enumerate(positives) for _ in support]
    columns = [position for support in positives for position in support]
    if not rows or any(not support for support in positives):
        raise ValueError('Every site needs verified supports')
    return (torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(columns, dtype=torch.long, device=device))


def union_field_loss(space_logits: list[Tensor], positives: list[tuple[int, ...]],
                     eligible: Tensor) -> Tensor:
    """Each verified support should be likely in at least one space's full field."""
    rows, columns = flat_positives(positives, eligible.device)
    if not bool(eligible[rows, columns].all()):
        raise ValueError('A verified support is outside the causal field')
    minimum = torch.finfo(torch.float32).min
    missing = torch.stack([
        1 - logits.float().masked_fill(~eligible, minimum).softmax(-1)[rows, columns]
        for logits in space_logits]).prod(0)
    return -(1 - missing).clamp_min(1e-12).log().mean()


def lexical_field_loss(space_logits: list[Tensor], lexical: Tensor,
                       eligible: Tensor) -> Tensor:
    """Weak mean-geometry alignment to source text overlap, as in the curriculum."""
    minimum = torch.finfo(torch.float32).min
    learned = torch.stack([logits.float() for logits in space_logits]).mean(0)
    learned = learned.masked_fill(~eligible, minimum).log_softmax(-1)
    usable = (lexical.masked_fill(~eligible, 0).amax(-1) > 0)
    if not bool(usable.any()):
        return learned.sum() * 0
    target = (lexical / .1).masked_fill(~eligible, minimum).softmax(-1)
    terms = (target * (target.clamp_min(1e-30).log() - learned)).masked_fill(~eligible, 0)
    return terms.sum(-1)[usable].mean()


def _terms(text: str) -> Counter[str]:
    return Counter(re.findall(r'[a-z0-9_]+', text.lower()))


class LexicalField:
    """Vectorized form of the curriculum's bounded TF-IDF source-text teacher."""

    def __init__(self, texts: list[str]) -> None:
        from scipy import sparse

        features = [_terms(text[:384]) for text in texts]
        frequencies = Counter()
        for row in features:
            frequencies.update(row.keys())
        self.vocabulary = {term: index for index, term in enumerate(sorted(frequencies))}
        self.idf = np.array([math.log1p(len(texts) / frequencies[term])
                             for term in sorted(frequencies)])
        rows, columns, values = [], [], []
        for row, terms in enumerate(features):
            for term, count in terms.items():
                rows.append(row)
                columns.append(self.vocabulary[term])
                values.append((1 + math.log(count)) * self.idf[self.vocabulary[term]])
        matrix = sparse.csr_matrix((values, (rows, columns)),
                                   shape=(len(texts), len(self.vocabulary)))
        norms = np.sqrt(matrix.multiply(matrix).sum(1)).A1
        self.documents = sparse.diags(1 / np.maximum(norms, 1e-12)) @ matrix

    def scores(self, queries: list[str]) -> np.ndarray:
        from scipy import sparse

        rows, columns, values = [], [], []
        for row, query in enumerate(queries):
            for term, count in _terms(query).items():
                column = self.vocabulary.get(term)
                if column is not None:
                    rows.append(row)
                    columns.append(column)
                    values.append((1 + math.log(count)) * self.idf[column])
        matrix = sparse.csr_matrix((values, (rows, columns)),
                                   shape=(len(queries), len(self.vocabulary)))
        norms = np.sqrt(matrix.multiply(matrix).sum(1)).A1
        matrix = sparse.diags(1 / np.maximum(norms, 1e-12)) @ matrix
        return (matrix @ self.documents.T).toarray().astype(np.float32)


def support_ranks(scores: Tensor, positives: list[tuple[int, ...]],
                  eligible: Tensor) -> list[tuple[int, ...]]:
    """Pessimistic 1-based rank of each support in each row's eligible field.

    Tied scores rank the support behind every tied record, so coarse score
    precision can never inflate recall.
    """
    masked = scores.float().masked_fill(~eligible, float('-inf'))
    rows, columns = flat_positives(positives, scores.device)
    values = masked[rows, columns]
    flat = (masked[rows] >= values[:, None]).sum(-1).tolist()
    ranks, offset = [], 0
    for support in positives:
        ranks.append(tuple(flat[offset:offset + len(support)]))
        offset += len(support)
    return ranks


def recall_summary(space_ranks: list[list[tuple[int, ...]]],
                   limits: tuple[int, ...]) -> dict:
    """Any-support and all-support recall at each space's selected budget."""
    sites = len(space_ranks[0])
    summary = {'sites': sites, 'spaces': []}
    union_any = [False] * sites
    for space, (ranks, limit) in enumerate(zip(space_ranks, limits, strict=True)):
        any_hit = [min(row) <= limit for row in ranks]
        union_any = [left or right for left, right in zip(union_any, any_hit, strict=True)]
        best = sorted(min(row) for row in ranks)
        summary['spaces'].append({
            'space': f's{space}', 'limit': limit,
            'any_support_recall': sum(any_hit) / sites,
            'all_support_recall': sum(max(row) <= limit for row in ranks) / sites,
            'recall_at_256': sum(min(row) <= 256 for row in ranks) / sites,
            'median_best_rank': best[len(best) // 2],
        })
    summary['union_any_support_recall'] = sum(union_any) / sites
    return summary
