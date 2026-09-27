"""Training-only address supervision against causally eligible stored sources.

Lexical similarity is a soft auxiliary target. It never selects a stored record
or changes the unassisted retrieval metric.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

import torch
from torch import Tensor
from torch.nn import functional as F


def routing_mix(step: int, ramp_steps: int = 3000) -> tuple[float, float]:
    """Move from sampled eligible negatives toward the whole-bank hard field."""
    if step < 0 or ramp_steps < 1:
        raise ValueError('Routing step and ramp must be valid')
    hard = .75 * min(step / ramp_steps, 1.0)
    return 1.0 - hard, hard


def _terms(text: str) -> Counter[str]:
    return Counter(re.findall(r'[a-z0-9_]+', text.lower()))


class RoutingCandidateIndex:
    """Small source-text teacher used only while optimizing address vectors."""

    def __init__(self, sources: Path, *, ramp_steps: int = 3000):
        if ramp_steps < 1:
            raise ValueError('Routing ramp must be positive')
        self.ramp_steps = ramp_steps
        self.rows = {}
        frequencies = Counter()
        with Path(sources).open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                record_id = row['record_id']
                if record_id in self.rows:
                    raise ValueError('Duplicate source identity in routing teacher')
                # Bounded metadata is enough for a cheap similarity hint. The
                # trained reader still sees only the bank's latent payload.
                features = _terms(row['text'][:384])
                self.rows[record_id] = (
                    row.get('domain', 'research'), row['created_at'], features)
                frequencies.update(features.keys())
        if not self.rows:
            raise ValueError('Routing teacher needs source records')
        self.ids = tuple(sorted(self.rows))
        self.idf = {term: math.log1p(len(self.rows) / count)
                    for term, count in frequencies.items()}

    def eligible(self, record_id: str, *, domain: str, query_time: int) -> bool:
        row = self.rows.get(record_id)
        return row is not None and row[0] == domain and row[1] < query_time

    def sample(self, episode_id: str, step: int, *, domain: str,
               query_time: int, limit: int) -> tuple[str, ...]:
        if limit < 0:
            raise ValueError('Negative sample count must be nonnegative')
        seed = int.from_bytes(hashlib.sha256(
            f'{episode_id}:{step}'.encode()).digest()[:8], 'little')
        rng = random.Random(seed)
        selected = set()
        for _ in range(max(64, limit * 16)):
            record_id = self.ids[rng.randrange(len(self.ids))]
            if self.eligible(record_id, domain=domain, query_time=query_time):
                selected.add(record_id)
                if len(selected) >= limit:
                    break
        if len(selected) < limit:
            for record_id in self.ids:
                if self.eligible(record_id, domain=domain, query_time=query_time):
                    selected.add(record_id)
                    if len(selected) >= limit:
                        break
        return tuple(sorted(selected))

    def lexical_similarities(self, query: str, ids: tuple[str, ...], *,
                             domain: str | None = None,
                             query_time: int | None = None) -> tuple[float, ...]:
        if domain is not None and query_time is not None and any(
                not self.eligible(record_id, domain=domain, query_time=query_time)
                for record_id in ids):
            return ()
        query_terms = _terms(query)
        q = {term: (1 + math.log(count)) * self.idf.get(term, 0.0)
             for term, count in query_terms.items()}
        qnorm = math.sqrt(sum(value * value for value in q.values()))
        scores = []
        for record_id in ids:
            terms = self.rows[record_id][2]
            norm = math.sqrt(sum(((1 + math.log(count)) * self.idf[term]) ** 2
                                 for term, count in terms.items()))
            dot = sum(q.get(term, 0.0) * (1 + math.log(count)) * self.idf[term]
                      for term, count in terms.items())
            scores.append(dot / (qnorm * norm) if qnorm and norm else 0.0)
        return tuple(scores)


def lexical_alignment_loss(space_scores: list[Tensor],
                           lexical_scores: Tensor) -> Tensor:
    """Weakly align the mean address geometry to source-derived text overlap."""
    if not space_scores or any(score.shape != lexical_scores.shape
                               for score in space_scores):
        raise ValueError('Lexical teacher and key scores must share one candidate field')
    if lexical_scores.max().item() <= 0:
        return space_scores[0].sum() * 0
    learned = torch.stack(space_scores).mean(0)
    target = (lexical_scores / .1).softmax(-1)
    return F.kl_div(learned.log_softmax(-1), target, reduction='sum')
