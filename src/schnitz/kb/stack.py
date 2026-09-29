"""The knowledge-base stack's modules (docs/knowledge-base-stack.md, sections 3-4):
spaces, forward codecs, recombiner, superposition operators and key heads, all
built from the MLP-matrix operator (``schnitz.mlp_matrix``). Every stage uses
these definitions.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from schnitz.kb_store import DEFAULT_SPACES
from schnitz.mlp_matrix import MLPMatrix

# space: (positions per span rep, width); width x ratio is 256 in every space (equal
# information per space), 1024 in all.
# One definition for the store and every stage: ``kb_store.DEFAULT_SPACES``.
SPACES = {name: (spec.ratio, spec.width) for name, spec in DEFAULT_SPACES.items()}
KEY_WIDTH = {name: spec.key_width for name, spec in DEFAULT_SPACES.items()}


class Stack(torch.nn.Module):
    """Forward codecs (span -> items per space) and the recombiner R (items -> span).

    Spans are standardized per dimension with corpus statistics (``mean``, ``std``,
    set by ``set_statistics``): decoder-space spans share a dominant common
    direction (the corpus-mean span alone has cosine 0.82 to a typical S2 span), so
    unstandardized objectives barely see the content that tells spans apart. The
    codecs read (x - mean) / std, R produces y and the span is mean + std * y;
    ``standardize`` gives the space in which reconstruction losses are measured."""

    def __init__(self, target_norm: float, state: int, hidden: int, layers: int,
                 checkpointing: bool, width: int = 1024):
        super().__init__()
        common = dict(state=state, hidden=hidden, layers=layers,
                      checkpoint_layers=checkpointing)
        self.codecs = torch.nn.ModuleDict({
            name: MLPMatrix({'span': width}, w, out_norm=math.sqrt(w), **common)
            for name, (_, w) in SPACES.items()})
        self.recombiner = MLPMatrix({name: w for name, (_, w) in SPACES.items()}, width,
                                    out_norm=math.sqrt(width), **common)
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('std', torch.ones(width))

    @torch.no_grad()
    def set_statistics(self, spans: list[torch.Tensor]) -> None:
        x = torch.cat([s.float() for s in spans]).to(self.mean.device)
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))

    def standardize(self, span: torch.Tensor) -> torch.Tensor:
        return (span.float() - self.mean) / self.std

    def encode(self, span: torch.Tensor) -> dict[str, torch.Tensor]:
        n, x = span.shape[0], self.standardize(span)
        return {name: self.codecs[name]([('span', x, 1.0)], max(1, math.ceil(ratio * n)))[0]
                for name, (ratio, _) in SPACES.items()}

    def decode_standardized(self, items: dict[str, torch.Tensor], keep: dict[str, float],
                            count: int) -> torch.Tensor:
        return self.recombiner([(name, items[name], keep[name]) for name in SPACES], count)[0]

    def decode(self, items: dict[str, torch.Tensor], keep: dict[str, float], count: int):
        return self.mean + self.std * self.decode_standardized(items, keep, count)


def read_count(positions: list[int], spaces: list[str], gates: torch.Tensor,
               budget: int | None = None, target: int | None = None) -> int:
    """How many decoder reps the recombiner produces (owner rule, section 5.2).

    With ``target`` (pretraining: K1, K3) the target's own count. At read time the
    gate-mass-weighted mean of the retrieved items' lengths in decoder reps (an item
    of m positions in a space of ratio r stands for m / r reps), capped by the
    per-read ``budget``."""
    if target is not None:
        return int(target)
    reps = torch.tensor([m / SPACES[s][0] for m, s in zip(positions, spaces)],
                        dtype=torch.float, device=gates.device)
    mass = gates.detach().float().clamp_min(0)
    if float(mass.sum()) <= 0:
        return 1
    count = max(1, round(float((mass * reps).sum() / mass.sum())))
    return min(count, budget) if budget else count


class SuperpositionOperator(nn.Module):
    """S_s: rewrites a neighbourhood of items in one space into ``count`` positions of
    an item for a target key (section 4). K3a conditions on the target key only;
    K3b (``neighbour_keys``) also gives each neighbour item's key to the operator at
    each of its positions (zero-initialized, so K3b continues K3a exactly)."""

    def __init__(self, space: str, state: int = 512, hidden: int = 256, layers: int = 3,
                 checkpoint_layers: bool = False):
        super().__init__()
        width, key = SPACES[space][1], KEY_WIDTH[space]
        self.space = space
        self.op = MLPMatrix({'item': width}, width, state=state, hidden=hidden, layers=layers,
                            cond=key, out_norm=math.sqrt(width), extra=key,
                            checkpoint_layers=checkpoint_layers)

    def forward(self, neighbours: list[tuple[torch.Tensor, torch.Tensor | float, torch.Tensor]],
                target_key: torch.Tensor, count: int, neighbour_keys: bool = False):
        """``neighbours``: (values (m, width), gate, key (key_width,)). Returns the
        rewritten item (count, width) and the neighbourhood's total mass."""
        items = [('item', values, gate, key if neighbour_keys else None)
                 for values, gate, key in neighbours]
        return self.op(items, count, cond=target_key[None])

    def forward_many(self, groups, neighbour_keys: bool = False, max_pairs: int | None = None):
        """``forward`` of many (neighbours, target_key, count) groups in one pass
        (``MLPMatrix.forward_many``). Returns [(item, mass)]."""
        return self.op.forward_many(
            [([('item', v, g, k if neighbour_keys else None) for v, g, k in neighbours],
              count, target_key[None]) for neighbours, target_key, count in groups],
            max_pairs=max_pairs)


class QueryPool(nn.Module):
    """Attention pooling of the query state over the call's causal prefix
    (``KeyHeads(pool=True)``; 29 September retrieval diagnosis).

    The query-layer state at the closing parenthesis of ``memory_search()`` carries
    almost nothing of the request: the call token sits at a fixed template position and
    the frozen decoder was never trained to put a query there (recall-text r8, layer 8:
    sites sharing a target document are no closer than unrelated ones, and heads on it
    cannot even fit the training sites over the full KB). The request is in the earlier
    tokens. The pool has ``heads`` learned attention queries over the normalized
    query-layer states of every position up to and including the call (causal: nothing
    after it); the pooled heads are projected and added to the call state,
    ``h_call + out(pooled)``. ``out`` starts at zero, so an untrained pool returns the
    call state exactly (the query of earlier readers), and its attention starts uniform
    (the prefix mean). A call-conditioned attention (query from the call state) fit the
    training sites as well but generalized worse (held-out R@8 0.23 vs 0.48)."""

    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, heads)
        self.out = nn.Linear(heads * width, width)
        for layer in (self.score, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        """``prefix`` (T, width): the query-layer states of positions 0..call; the last
        row is the call's own state. Returns the pooled query state (width,)."""
        x = self.norm(prefix.float())
        att = torch.softmax(self.score(x), 0)                 # (T, heads)
        return prefix[-1].float() + self.out((att.t() @ x).reshape(-1))


class KeyHeads(nn.Module):
    """Per-space item keys (from an item's values) and query keys (from the decoder's
    middle-layer state at a ``memory_search()`` call), unit-normalized; scores are
    cosine times a learned scale per space. ``pool``: the query state is the call's
    state plus a ``QueryPool`` of its causal prefix (``query_state``)."""

    def __init__(self, query_width: int, hidden: int = 512, pool: bool = False):
        super().__init__()
        self.pool = QueryPool(query_width) if pool else None
        self.item = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU(),
                                nn.Linear(hidden, KEY_WIDTH[name]))
            for name, (_, width) in SPACES.items()})
        self.query = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(query_width), nn.Linear(query_width, hidden),
                                nn.SiLU(), nn.Linear(hidden, KEY_WIDTH[name]))
            for name in SPACES})
        self.log_scale = nn.ParameterDict({name: nn.Parameter(torch.tensor(math.log(10.0)))
                                           for name in SPACES})

    def query_state(self, state: torch.Tensor) -> torch.Tensor:
        """The query state of a call: ``state`` is the call's own query-layer state
        (width,) or the states of its causal prefix (T, width), the call last. Without
        a pool the call's state; with one the ``QueryPool`` output (a single state is a
        prefix of one position)."""
        if self.pool is None:
            return state[-1] if state.ndim == 2 else state
        return self.pool(state if state.ndim == 2 else state[None])

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        """Heads saved without a pool (earlier readers, ``key_heads_init.pt``) load into
        pooled heads with the pool at its initialization (the call state exactly)."""
        if self.pool is not None and not any(k.startswith('pool.') for k in state):
            state = {**state, **{f'pool.{k}': v for k, v in self.pool.state_dict().items()}}
        return super().load_state_dict(state, strict=strict, assign=assign)

    def item_key(self, space: str, values: torch.Tensor) -> torch.Tensor:
        """``values`` (m, width) or (B, m, width) -> unit key(s); mean over positions."""
        return nn.functional.normalize(self.item[space](values.float()).mean(-2), dim=-1)

    def query_key(self, space: str, state: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.query[space](state.float()), dim=-1)

    def scores(self, space: str, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return queries @ keys.t() * self.log_scale[space].exp()
