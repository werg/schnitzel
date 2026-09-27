"""Immutable packed transcripts for whole-sequence spatial memory training."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .data import Episode
from .trajectory_memory import validate_memory_transcript, write_call, write_result


SYSTEM_PROMPT = (
    "Use memory.search whenever stored knowledge can help. Each call has a latent "
    "result attached to its following tool result. Answer each user request exactly."
)


def _call(call_id: str, episode: Episode) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "memory.search",
                "arguments": {
                    "query": episode.query,
                    "scope": {"environment": episode.environment},
                    "purpose": "answer the current request",
                    "budget": "standard",
                },
            },
        }],
    }


def _result(call_id: str, generation: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": "memory.search",
        "tool_call_id": call_id,
        "content": json.dumps({
            "status": "ok",
            "record_ids": [],
            "spaces": {},
            "generation": generation,
            "latent_attachment_id": call_id,
        }, sort_keys=True, separators=(",", ":")),
    }


def _tokens(tokenizer, messages: list[dict[str, Any]]) -> dict[str, list[int]]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if mask is None or len(mask) != len(encoded["input_ids"]):
        raise ValueError("Tokenizer chat template must return an assistant token mask")
    return {"input_ids": list(encoded["input_ids"]), "assistant_mask": list(mask)}


def pack_spatial_trajectory(tokenizer, episodes: Sequence[Episode], *, read_slots: int,
                            generation: str, levels: Sequence[int] | None = None,
                            include_writes: bool = False,
                            write_generation: str = "pending-authored",
                            write_slots: int = 0) -> dict[str, Any]:
    """Pack several visible calls into one sequence with blank result spans."""
    if (not episodes or read_slots < 1 or write_slots < 0
            or include_writes and write_slots < 1):
        raise ValueError("Spatial packing needs episodes and positive read slots")
    levels = tuple(levels or (1,) * len(episodes))
    if len(levels) != len(episodes) or any(level < 1 for level in levels):
        raise ValueError("Supply one positive recurrence level per site")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    boundaries, write_boundaries = [], []
    for index, (episode, level) in enumerate(zip(episodes, levels, strict=True)):
        call_id = f"memory-search-{index}-{episode.episode_id[:12]}"
        messages.append({"role": "user", "content": episode.query})
        messages.append(_call(call_id, episode))
        call_prefix = _tokens(tokenizer, messages)["input_ids"]
        messages.append(_result(call_id, generation))
        result_prefix = _tokens(tokenizer, messages)["input_ids"]
        messages.append({"role": "assistant", "content": episode.answer})
        boundaries.append({
            "call_id": call_id,
            "base_query_position": len(call_prefix) - 1,
            "base_workspace_start": len(result_prefix),
            "_call_prefix": call_prefix,
            "_result_prefix": result_prefix,
            "level": level,
            "required_ids": list(episode.required_ids),
            "query_time": episode.query_time,
            "domain": episode.provenance.get("domain", "research"),
            "episode_id": episode.episode_id,
        })
        if include_writes:
            messages.append({
                "role": "user",
                "content": ("Store one reusable memory from the completed task. Preserve its "
                            "scope, evidence, and applicability conditions."),
            })
            write_id = f"memory-write-{index}-{episode.episode_id[:12]}"
            content = f"For the task '{episode.query}', the verified outcome was '{episode.answer}'."
            messages.append(write_call(
                call_id=write_id, content=content, kind="verified_task_outcome",
                scope={"environment": episode.environment},
                applies_when=f"A later task concerns {episode.environment} and the same facts.",
                evidence_refs=episode.required_ids, confidence="verified_training_record",
                parent_read_call_ids=(call_id,),
            ))
            write_prefix = _tokens(tokenizer, messages)["input_ids"]
            record_id = hashlib.sha256(
                f"{write_generation}:{episode.episode_id}".encode()).hexdigest()[:32]
            messages.append(write_result(call_id=write_id, status="ok",
                                         generation=write_generation, record_id=record_id))
            write_boundaries.append({
                "call_id": write_id, "base_call_position": len(write_prefix) - 1,
                "base_workspace_start": len(write_prefix),
                "episode_id": episode.episode_id, "record_id": record_id,
                "parent_read_call_ids": [call_id], "content": content,
                "_call_prefix": write_prefix,
            })
    validate_memory_transcript(messages)
    encoded = _tokens(tokenizer, messages)
    base_ids, base_mask = encoded["input_ids"], encoded["assistant_mask"]
    for boundary in boundaries:
        if (base_ids[:len(boundary["_call_prefix"])] != boundary["_call_prefix"]
                or base_ids[:len(boundary["_result_prefix"])] != boundary["_result_prefix"]):
            raise ValueError("Chat template changed an earlier site when later messages were appended")
    for boundary in write_boundaries:
        if base_ids[:len(boundary["_call_prefix"])] != boundary["_call_prefix"]:
            raise ValueError("Chat template changed an earlier write when later messages were appended")

    read_insertions = {boundary["base_workspace_start"]: boundary for boundary in boundaries}
    write_insertions = {boundary["base_workspace_start"]: boundary
                        for boundary in write_boundaries}
    if (len(read_insertions) != len(boundaries)
            or len(write_insertions) != len(write_boundaries)
            or set(read_insertions).intersection(write_insertions)):
        raise ValueError("Spatial result positions must be distinct")
    expanded_ids, expanded_assistant, base_to_expanded = [], [], {}
    for base_position in range(len(base_ids) + 1):
        if base_position in read_insertions:
            read_insertions[base_position]["workspace_start"] = len(expanded_ids)
            expanded_ids.extend([-1] * read_slots)
            expanded_assistant.extend([0] * read_slots)
        if base_position in write_insertions:
            write_insertions[base_position]["workspace_start"] = len(expanded_ids)
            expanded_ids.extend([-2] * (write_slots + 1))
            expanded_assistant.extend([0] * (write_slots + 1))
        if base_position < len(base_ids):
            base_to_expanded[base_position] = len(expanded_ids)
            expanded_ids.append(base_ids[base_position])
            expanded_assistant.append(base_mask[base_position])
    labels = [-100] * len(expanded_ids)
    for position, (token_id, supervised) in enumerate(zip(expanded_ids, expanded_assistant, strict=True)):
        if supervised and token_id >= 0 and position:
            labels[position - 1] = token_id
    sites = []
    for boundary in boundaries:
        sites.append({key: boundary[key] for key in (
            "call_id", "workspace_start", "level", "required_ids", "query_time",
            "domain", "episode_id")} | {
            "query_position": base_to_expanded[boundary["base_query_position"]],
            })
    write_sites = [{key: boundary[key] for key in (
        "call_id", "episode_id", "record_id", "parent_read_call_ids", "content")} | {
        "call_position": base_to_expanded[boundary["base_call_position"]],
        "workspace_start": boundary["workspace_start"],
    } for boundary in write_boundaries]
    if not any(label >= 0 for label in labels):
        raise ValueError("Packed trajectory has no supervised assistant tokens")
    identity = [episode.episode_id for episode in episodes]
    return {
        "trajectory_id": hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32],
        "episode_ids": identity,
        "input_ids": expanded_ids,
        "labels": labels,
        "sites": sites,
        "write_sites": write_sites,
        "read_slots": read_slots,
        "write_slots": write_slots,
        "tokens": len(expanded_ids),
        "supervised_tokens": sum(label >= 0 for label in labels),
    }


class SpatialTrajectoryIndex(Sequence):
    def __init__(self, path: str | Path):
        self.path, self.offsets = Path(path), []
        digest, identities = hashlib.sha256(), set()
        with self.path.open("rb") as handle:
            while True:
                offset, line = handle.tell(), handle.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_spatial_row(row)
                if row["trajectory_id"] in identities:
                    raise ValueError("Spatial trajectory IDs must be unique")
                identities.add(row["trajectory_id"])
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("Spatial trajectory file is empty")
        self.sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def validate_spatial_row(row: dict[str, Any]) -> None:
    required = {"trajectory_id", "episode_ids", "input_ids", "labels", "sites", "read_slots"}
    if not required <= set(row) or not row["trajectory_id"] or not row["sites"]:
        raise ValueError("Incomplete spatial trajectory")
    if len(row["input_ids"]) != len(row["labels"]):
        raise ValueError("Spatial IDs and labels differ in length")
    if len(row["episode_ids"]) != len(row["sites"]):
        raise ValueError("Spatial episodes and sites differ in length")
    write_sites = row.get("write_sites", [])
    if write_sites and len(write_sites) != len(row["episode_ids"]):
        raise ValueError("Spatial episodes and writes differ in length")
    blanks = {index for index, token in enumerate(row["input_ids"]) if token == -1}
    write_blanks = {index for index, token in enumerate(row["input_ids"]) if token == -2}
    read_slots = row["read_slots"]
    if not isinstance(read_slots, int) or isinstance(read_slots, bool) or read_slots < 1:
        raise ValueError("Spatial read_slots must be positive")
    covered, calls = set(), set()
    for site in row["sites"]:
        if not {"call_id", "query_position", "workspace_start", "level", "required_ids",
                "query_time", "domain", "episode_id"} <= set(site):
            raise ValueError("Incomplete spatial site")
        if (not 0 <= site["query_position"] < site["workspace_start"]
                or site["workspace_start"] + read_slots > len(row["input_ids"])
                or site["level"] < 1):
            raise ValueError("Invalid spatial causal position or level")
        if row["input_ids"][site["query_position"]] < 0:
            raise ValueError("Spatial query positions must contain visible tokens")
        start = site["workspace_start"]
        span = set(range(start, start + read_slots))
        if span - blanks or covered.intersection(span):
            raise ValueError("Spatial blank workspaces are incomplete or overlapping")
        covered.update(span)
        if site["call_id"] in calls:
            raise ValueError("Spatial call IDs must be unique")
        calls.add(site["call_id"])
        if not site["required_ids"]:
            raise ValueError("Spatial site needs verified positive IDs")
    if any(token < -2 for token in row["input_ids"]) or not blanks or covered != blanks:
        raise ValueError("Spatial rows require only -1 blank sentinels")
    read_calls = calls
    write_slots = row.get("write_slots", 0)
    write_covered = set()
    for write in write_sites:
        if (not {"call_id", "call_position", "episode_id", "record_id",
                 "parent_read_call_ids", "content", "workspace_start"} <= set(write)
                or not 0 <= write["call_position"] < len(row["input_ids"])
                or row["input_ids"][write["call_position"]] < 0
                or not write["call_position"] < write["workspace_start"]
                or not write["record_id"] or not write["content"]
                or not set(write["parent_read_call_ids"]) <= read_calls):
            raise ValueError("Invalid spatial write site or read lineage")
        span = set(range(write["workspace_start"],
                         write["workspace_start"] + write_slots + 1))
        if write_slots < 1 or span - write_blanks or write_covered.intersection(span):
            raise ValueError("Spatial write workspaces are incomplete or overlapping")
        write_covered.update(span)
    if write_covered != write_blanks:
        raise ValueError("Spatial write sentinels must exactly match write workspaces")
