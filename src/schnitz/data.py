"""Causally separated synthetic support/query episodes with counterfactual worlds.

No teacher API is required. These fixtures test information transfer and composition,
not real-world coding ability or capacity substitution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace, field
from pathlib import Path
import hashlib
import json
import random


@dataclass(frozen=True)
class Source:
    record_id: str
    text: str
    created_at: int
    kind: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    environment: str
    supports: tuple[Source, ...]
    query: str
    answer: str
    required_ids: tuple[str, ...]
    restore: bool
    allowed_capability: int
    capability: int
    query_time: int = 10
    task_family: str = "retry"
    choices: tuple[str, ...] = ()
    sufficient_groups: tuple[tuple[str, ...], ...] = ()
    support_annotation: str = "verified"
    provenance: dict = field(default_factory=dict)


def _opaque(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def evidence_ids(episode: Episode, scope: str = 'required') -> tuple[str, ...]:
    """Choose visible inputs without changing the task's sufficient-support labels."""
    if scope not in {'required', 'available'}:
        raise ValueError('Invalid evidence scope')
    if any(s.created_at >= episode.query_time for s in episode.supports):
        raise ValueError('Evidence must be causally prior to the query')
    return episode.required_ids if scope == 'required' else tuple(s.record_id for s in episode.supports)


def make_episode(seed: int, *, split: str = "train", distractors: int = 2,
                 restore: bool | None = None, allowed_capability: int | None = None,
                 capability: int | None = None) -> Episode:
    rng = random.Random(f"{split}:{seed}")
    environment = "env_" + _opaque(f"{split}:{seed}:env")[:8]
    restore = bool(rng.randrange(2)) if restore is None else restore
    allowed = rng.randrange(2) if allowed_capability is None else allowed_capability
    cap = rng.randrange(2) if capability is None else capability
    if allowed not in {0, 1} or cap not in {0, 1}:
        raise ValueError("Capabilities are binary")
    rid_a, rid_b = (_opaque(f"{split}:{seed}:{kind}") for kind in ("A", "B"))
    a = Source(rid_a, f"Experience in {environment}: after a failed attempt, "
               + ("restoring the snapshot is required before retry." if restore
                  else "restoring the snapshot is forbidden before retry."), 1, "restoration")
    b = Source(rid_b, f"Experience in {environment}: retry is permitted exactly when "
               f"capability={allowed}; otherwise stop.", 2, "permission")
    supports = [a, b]
    for i in range(distractors):
        decoy = "env_" + _opaque(f"{split}:{seed}:decoy:{i}")[:8]
        supports.append(Source(_opaque(f"{split}:{seed}:d:{i}"),
            f"Experience in {decoy}: retry requires capability={rng.randrange(2)}; "
            f"restoration is {'required' if rng.randrange(2) else 'forbidden'}.", 3 + i, "irrelevant"))
    rng.shuffle(supports)
    answer = "STOP" if cap != allowed else ("RESTORE_RETRY" if restore else "RETRY")
    query = (f"In {environment}, an attempt failed and capability={cap}. "
             "Using the environment's rules, choose exactly one action: STOP, RETRY, RESTORE_RETRY.")
    return Episode(_opaque(f"{split}:{seed}:query"), environment, tuple(supports), query, answer,
                   (rid_a, rid_b), restore, allowed, cap, 4 + distractors,
                   choices=("STOP", "RETRY", "RESTORE_RETRY"),
                   sufficient_groups=((rid_b,),) if cap != allowed else ((rid_a, rid_b),))


def counterfactual(episode: Episode, which: str) -> Episode:
    """Same IDs and metadata; flip one rule and recompute the correct later action."""
    if which not in {"restoration", "permission"}:
        raise ValueError("Choose restoration or permission")
    restore = not episode.restore if which == "restoration" else episode.restore
    allowed = 1 - episode.allowed_capability if which == "permission" else episode.allowed_capability
    sources = []
    for s in episode.supports:
        text = s.text
        if which == "restoration" and s.kind == which:
            text = (f"Experience in {episode.environment}: after a failed attempt, "
                    + ("restoring the snapshot is required before retry." if restore
                       else "restoring the snapshot is forbidden before retry."))
        if which == "permission" and s.kind == which:
            text = (f"Experience in {episode.environment}: retry is permitted exactly when "
                    f"capability={allowed}; otherwise stop.")
        sources.append(replace(s, text=text))
    answer = "STOP" if episode.capability != allowed else ("RESTORE_RETRY" if restore else "RETRY")
    groups = ((episode.required_ids[1],),) if episode.capability != allowed else (episode.required_ids,)
    return replace(episode, supports=tuple(sources), answer=answer, restore=restore,
                   allowed_capability=allowed, sufficient_groups=groups)


def save_episodes(path: str | Path, episodes: list[Episode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")


def episode_from_dict(row: dict) -> Episode:
    row = dict(row)
    row.setdefault("environment", row["episode_id"])
    row.setdefault("task_family", "custom")
    row.setdefault("restore", False)
    row.setdefault("allowed_capability", 0)
    row.setdefault("capability", 0)
    row["supports"] = tuple(Source(**({"kind": "experience"} | source)) for source in row["supports"])
    row["required_ids"] = tuple(row["required_ids"])
    row["choices"] = tuple(row.get("choices", ()))
    row["sufficient_groups"] = tuple(tuple(g) for g in row.get("sufficient_groups", ()))
    episode = Episode(**row)
    if any(s.created_at >= episode.query_time for s in episode.supports):
        raise ValueError("Source is not causally prior to the query")
    if episode.episode_id in {s.record_id for s in episode.supports}:
        raise ValueError("Query cannot be its own source")
    ids = [s.record_id for s in episode.supports]
    if len(ids) != len(set(ids)) or not set(episode.required_ids) <= set(ids):
        raise ValueError("Support IDs must be unique and include required_ids")
    if episode.choices and (episode.answer not in episode.choices or len(set(episode.choices)) != len(episode.choices)):
        raise ValueError("Distinct choices must contain the correct answer")
    if any(not g or not set(g) <= set(ids) for g in episode.sufficient_groups):
        raise ValueError("Invalid sufficient support group")
    if not episode.answer or not episode.query:
        raise ValueError("Query and answer cannot be empty")
    return episode


def load_episodes(path: str | Path) -> list[Episode]:
    with open(path, encoding="utf-8") as handle:
        episodes = [episode_from_dict(json.loads(line)) for line in handle if line.strip()]
    seen = {}
    episode_ids = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ValueError("Episode IDs must be unique")
        episode_ids.add(episode.episode_id)
        for source in episode.supports:
            if source.record_id in seen and seen[source.record_id] != source:
                raise ValueError("A shared source ID must identify exactly the same experience")
            seen[source.record_id] = source
    if not episodes:
        raise ValueError("Episode file is empty")
    return episodes


def make_multiuse_world(seed: int, *, split: str = "multiuse", bindings: int = 3) -> list[Episode]:
    """Write a world's sources once, then test five future uses per entity.

    Every entity has independently randomized permission/restoration rules. Exact
    endpoint names are random and cannot be inferred from entity IDs. The two
    action queries require binding facts to an entity and using the right branch.
    Oracle selection always delivers both rule records on action queries, even on
    STOP branches: labels never leak through answer-dependent retrieval cardinality.
    """
    if bindings < 1:
        raise ValueError("bindings must be positive")
    rng = random.Random(f"{split}:{seed}:multiuse")
    environment = "world_" + _opaque(f"{split}:{seed}")[:8]
    entities = ["svc_" + _opaque(f"{split}:{seed}:entity:{i}")[:6] for i in range(bindings)]
    endpoints = ["api_" + _opaque(f"{split}:{seed}:endpoint:{i}")[:6] for i in range(bindings)]
    facts, sources = [], []
    for i, entity in enumerate(entities):
        restore, allowed = bool(rng.randrange(2)), rng.randrange(2)
        a = Source(_opaque(f"{split}:{seed}:{i}:restore"),
                   f"In {environment}, {entity}: restoring the snapshot before retry is "
                   + ("required." if restore else "forbidden."), 2 * i + 1, "restoration")
        b = Source(_opaque(f"{split}:{seed}:{i}:permission"),
                   f"In {environment}, {entity}: retry is permitted exactly when capability={allowed}; "
                   f"otherwise stop. Its endpoint is {endpoints[i]}.", 2 * i + 2, "permission")
        sources.extend((a, b))
        facts.append((restore, allowed, a.record_id, b.record_id))
    rng.shuffle(sources)
    support = tuple(sources)
    episodes = []
    for i, entity in enumerate(entities):
        restore, allowed, a, b = facts[i]
        prefix = f"In {environment}, for {entity}, "
        tasks = [
            ("permission", prefix + "which capability permits retry? Answer 0 or 1.",
             str(allowed), (b,), ("0", "1"), 0, ((b,),)),
            ("restoration", prefix + "is restoring the snapshot required or forbidden before retry?",
             "required" if restore else "forbidden", (a,), ("required", "forbidden"), 0, ((a,),)),
            ("identifier", prefix + "give the exact endpoint identifier.", endpoints[i], (b,),
             tuple(endpoints + ["api_" + _opaque(f"{split}:{seed}:unused")[:6]]), 0, ((b,),)),
        ]
        for cap in (0, 1):
            answer = "STOP" if cap != allowed else ("RESTORE_RETRY" if restore else "RETRY")
            tasks.append((f"action-{cap}", prefix + f"an attempt failed and capability={cap}. "
                          "Choose STOP, RETRY, or RESTORE_RETRY.", answer, (a, b),
                          ("STOP", "RETRY", "RESTORE_RETRY"), cap,
                          ((b,),) if cap != allowed else ((a, b),)))
        for kind, query, answer, required, choices, cap, groups in tasks:
            episodes.append(Episode(_opaque(f"{split}:{seed}:{i}:query:{kind}"), environment,
                                    support, query, answer, required, restore, allowed, cap,
                                    2 * bindings + 10, f"multiuse/{kind.split('-')[0]}", choices, groups))
    return episodes


def counterfactual_multiuse(episode: Episode, which: str) -> Episode:
    """Flip a rule type across a generated world, keeping shared source IDs consistent.

    Every future query sees the same alternate world. Flipping only its queried
    entity would assign conflicting payloads to shared IDs in a global bank.
    """
    import re
    if not episode.task_family.startswith('multiuse/') or which not in {'restoration', 'permission'}:
        raise ValueError('Generated multiuse episode and restoration/permission intervention required')
    sources = []
    for source in episode.supports:
        text = source.text
        if source.kind == which:
            if which == 'restoration':
                text, count = re.subn(r'\b(required|forbidden)\.$',
                    lambda m: ('forbidden' if m[1] == 'required' else 'required') + '.', text)
            else:
                text, count = re.subn(r'(?<=capability=)[01](?=;)', lambda m: str(1 - int(m[0])), text)
            if count != 1:
                raise ValueError('Malformed generated binding rule')
        sources.append(replace(source, text=text))
    restore = not episode.restore if which == 'restoration' else episode.restore
    allowed = 1 - episode.allowed_capability if which == 'permission' else episode.allowed_capability
    family = episode.task_family.split('/')[1]
    groups = episode.sufficient_groups
    if family == 'action':
        answer = 'STOP' if episode.capability != allowed else ('RESTORE_RETRY' if restore else 'RETRY')
        groups = ((episode.required_ids[1],),) if episode.capability != allowed else (episode.required_ids,)
    elif family == 'restoration':
        answer = 'required' if restore else 'forbidden'
    elif family == 'permission':
        answer = str(allowed)
    elif family == 'identifier':
        answer = episode.answer
    else:
        raise ValueError('Unknown binding task family')
    return replace(episode, supports=tuple(sources), answer=answer, restore=restore,
                   allowed_capability=allowed, sufficient_groups=groups)


def make_boolean_world(seed: int, *, split: str = "boolean", operations: tuple[str, ...] = ("xor",)) -> list[Episode]:
    """Balanced independent bit facts for a small end-to-end composition diagnostic."""
    if not operations or any(op not in {'xor', 'and', 'or', 'a', 'b'} for op in operations):
        raise ValueError('Unknown boolean operation')
    world = 'w' + _opaque(f'{split}:{seed}:world')[:6]
    a, b = (seed // 2) % 2, seed % 2
    sources = [Source(_opaque(f'{split}:{seed}:source:{name}'), f'{world} {name}={value}.', i + 1, name)
               for i, (name, value) in enumerate((('a', a), ('b', b)))]
    random.Random(f'{split}:{seed}:order').shuffle(sources)
    by_kind = {s.kind: s.record_id for s in sources}
    results = {'xor': a ^ b, 'and': a & b, 'or': a | b, 'a': a, 'b': b}
    episodes = []
    for op in operations:
        required = (by_kind[op],) if op in {'a', 'b'} else (by_kind['a'], by_kind['b'])
        episodes.append(Episode(_opaque(f'{split}:{seed}:query:{op}'), world, tuple(sources),
                        f'{world}: {op}? Answer 0 or 1.', str(results[op]), required,
                        False, 0, 0, 10, f'boolean/{op}', ('0', '1'), (required,)))
    return episodes


def counterfactual_boolean(episode: Episode, which: str) -> Episode:
    """Flip one stored bit, preserving query, source IDs, timestamps and record order."""
    import re
    if not episode.task_family.startswith('boolean/') or which not in {'a', 'b'}:
        raise ValueError('Boolean episode and a/b intervention required')
    bits, sources = {}, []
    for source in episode.supports:
        match = re.search(r'\b([ab])=([01])\.', source.text)
        if match is None:
            raise ValueError('Malformed boolean source')
        name, value = match.group(1), int(match.group(2))
        if name == which:
            value = 1 - value
            source = replace(source, text=source.text[:match.start(2)] + str(value) + source.text[match.end(2):])
        bits[name] = value
        sources.append(source)
    a, b = bits['a'], bits['b']
    result = {'a': a, 'b': b, 'xor': a ^ b, 'and': a & b, 'or': a | b}[episode.task_family.split('/')[1]]
    return replace(episode, supports=tuple(sources), answer=str(result))
