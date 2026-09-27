"""Canonical visible memory-tool events for trajectory-native SCHNITZELJAGD training.

The transcript carries opaque metadata and tool syntax. Latent tensors remain in a
separate attachment store and are referenced by ID from a search result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable


SEARCH_TOOL = "memory.search"
WRITE_TOOL = "memory.write"
MEMORY_TOOLS = frozenset({SEARCH_TOOL, WRITE_TOOL})
RESULT_STATUSES = frozenset({"ok", "empty", "denied", "partial", "timed_out", "failed"})


@dataclass(frozen=True)
class MemoryTranscriptSummary:
    search_calls: int
    write_calls: int
    search_positions: tuple[int, ...]
    write_positions: tuple[int, ...]
    read_dependent_writes: int
    record_ids_written: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _json_content(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def search_call(*, call_id: str, query: str, scope: dict[str, str], purpose: str,
                budget: str = "standard", created_at: int | float | None = None) -> dict[str, Any]:
    """Create one assistant event containing exactly one visible search call."""
    arguments = {"query": query, "scope": scope, "purpose": purpose, "budget": budget}
    event: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": SEARCH_TOOL, "arguments": arguments},
        }],
    }
    if created_at is not None:
        event["created_at"] = created_at
    return event


def search_result(*, call_id: str, status: str, record_ids: Iterable[str],
                  spaces: dict[str, int], generation: str,
                  latent_attachment_id: str | None = None,
                  created_at: int | float | None = None) -> dict[str, Any]:
    """Create the visible envelope for a stored-only latent search result."""
    content: dict[str, Any] = {
        "status": status,
        "record_ids": list(record_ids),
        "spaces": spaces,
        "generation": generation,
    }
    if latent_attachment_id is not None:
        content["latent_attachment_id"] = latent_attachment_id
    event: dict[str, Any] = {
        "role": "tool",
        "name": SEARCH_TOOL,
        "tool_call_id": call_id,
        "content": _json_content(content),
    }
    if created_at is not None:
        event["created_at"] = created_at
    return event


def write_call(*, call_id: str, content: str, kind: str, scope: dict[str, str],
               applies_when: str, evidence_refs: Iterable[str], confidence: str,
               parent_read_call_ids: Iterable[str] = (),
               created_at: int | float | None = None) -> dict[str, Any]:
    """Create one assistant event that proposes exactly one logical record."""
    arguments = {
        "content": content,
        "kind": kind,
        "scope": scope,
        "applies_when": applies_when,
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "parent_read_call_ids": list(parent_read_call_ids),
    }
    event: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": WRITE_TOOL, "arguments": arguments},
        }],
    }
    if created_at is not None:
        event["created_at"] = created_at
    return event


def write_result(*, call_id: str, status: str, generation: str,
                 record_id: str | None = None,
                 created_at: int | float | None = None) -> dict[str, Any]:
    """Create the commit result for one logical memory record."""
    content: dict[str, Any] = {"status": status, "generation": generation}
    if record_id is not None:
        content["record_id"] = record_id
    event: dict[str, Any] = {
        "role": "tool",
        "name": WRITE_TOOL,
        "tool_call_id": call_id,
        "content": _json_content(content),
    }
    if created_at is not None:
        event["created_at"] = created_at
    return event


def _call_from_event(event: dict[str, Any], position: int) -> tuple[str, str, dict[str, Any]] | None:
    calls = event.get("tool_calls", [])
    if not calls:
        return None
    if not isinstance(calls, list):
        raise ValueError(f"tool_calls at position {position} must be a list")
    memory_calls = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError(f"invalid tool call at position {position}")
        function = call["function"]
        if function.get("name") in MEMORY_TOOLS:
            memory_calls.append(call)
    if not memory_calls:
        return None
    if event.get("role") != "assistant" or len(calls) != 1:
        raise ValueError("a memory site must be the sole tool call in its assistant event")
    call = memory_calls[0]
    if call.get("type") != "function":
        raise ValueError("memory tool calls must use function type")
    call_id = _nonempty(call.get("id"), "call_id")
    function = call["function"]
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("memory tool arguments must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("memory tool arguments must be an object")
    return call_id, function["name"], arguments


def _validate_scope(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("memory scope must be a nonempty object")
    if any(not isinstance(key, str) or not key or not isinstance(item, str) or not item
           for key, item in value.items()):
        raise ValueError("memory scope keys and values must be nonempty strings")


def _validate_call(name: str, arguments: dict[str, Any], completed_reads: set[str]) -> bool:
    if name == SEARCH_TOOL:
        _nonempty(arguments.get("query"), "search query")
        _nonempty(arguments.get("purpose"), "search purpose")
        _nonempty(arguments.get("budget"), "search budget")
        _validate_scope(arguments.get("scope"))
        return False
    if "items" in arguments:
        raise ValueError("memory.write accepts one record; use distinct calls at distinct positions")
    for field in ("content", "kind", "applies_when", "confidence"):
        _nonempty(arguments.get(field), f"write {field}")
    _validate_scope(arguments.get("scope"))
    _string_list(arguments.get("evidence_refs"), "write evidence_refs")
    parents = _string_list(arguments.get("parent_read_call_ids"), "write parent_read_call_ids")
    if not set(parents) <= completed_reads:
        raise ValueError("a write can depend only on completed earlier memory.search calls")
    return bool(parents)


def _parse_result(event: dict[str, Any], position: int) -> tuple[str, str, dict[str, Any]] | None:
    if event.get("role") != "tool" or event.get("name") not in MEMORY_TOOLS:
        return None
    call_id = _nonempty(event.get("tool_call_id"), "tool result call_id")
    content = event.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"memory result at position {position} must contain JSON") from exc
    if not isinstance(content, dict):
        raise ValueError(f"memory result at position {position} must contain an object")
    return call_id, event["name"], content


def _validate_result(name: str, result: dict[str, Any]) -> str | None:
    status = result.get("status")
    if status not in RESULT_STATUSES:
        raise ValueError("invalid memory tool result status")
    _nonempty(result.get("generation"), "result generation")
    if name == SEARCH_TOOL:
        allowed = {"status", "record_ids", "spaces", "generation", "latent_attachment_id"}
        if set(result) - allowed:
            raise ValueError("search result envelope contains unsupported or source-bearing fields")
        ids = _string_list(result.get("record_ids"), "search result record_ids")
        spaces = result.get("spaces")
        if not isinstance(spaces, dict) or any(
                not isinstance(key, str) or not key or not isinstance(count, int)
                or isinstance(count, bool) or count < 0 for key, count in spaces.items()):
            raise ValueError("search result spaces must map names to nonnegative counts")
        attachment = result.get("latent_attachment_id")
        if status in {"ok", "partial"} and ids and not attachment:
            raise ValueError("a nonempty latent search result requires an attachment ID")
        if status not in {"ok", "partial"} and ids:
            raise ValueError("an unsuccessful or empty search result cannot select records")
        if attachment is not None:
            _nonempty(attachment, "latent_attachment_id")
        return None
    if set(result) - {"status", "generation", "record_id"}:
        raise ValueError("write result envelope contains unsupported fields")
    record_id = result.get("record_id")
    if status == "ok":
        return _nonempty(record_id, "write result record_id")
    if record_id is not None:
        raise ValueError("an unsuccessful write result cannot publish a record ID")
    return None


def validate_memory_transcript(messages: Iterable[dict[str, Any]]) -> MemoryTranscriptSummary:
    """Validate causal pairing, per-position sites, and read-before-write lineage."""
    pending: dict[str, tuple[str, int]] = {}
    completed_reads: set[str] = set()
    completed_calls: set[str] = set()
    search_positions: list[int] = []
    write_positions: list[int] = []
    written: list[str] = []
    dependent_writes = 0
    last_created_at: int | float | None = None

    for position, event in enumerate(messages):
        if not isinstance(event, dict):
            raise ValueError(f"transcript event {position} must be an object")
        created_at = event.get("created_at")
        if created_at is not None:
            if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
                raise ValueError("created_at must be numeric")
            if last_created_at is not None and created_at < last_created_at:
                raise ValueError("transcript timestamps must be nondecreasing")
            last_created_at = created_at

        call = _call_from_event(event, position)
        if call is not None:
            call_id, name, arguments = call
            if call_id in pending or call_id in completed_calls:
                raise ValueError("memory call IDs must be unique")
            depends_on_read = _validate_call(name, arguments, completed_reads)
            pending[call_id] = (name, position)
            if name == SEARCH_TOOL:
                search_positions.append(position)
            else:
                write_positions.append(position)
                dependent_writes += int(depends_on_read)

        result = _parse_result(event, position)
        if result is not None:
            call_id, name, content = result
            if call_id not in pending:
                raise ValueError("memory tool result must follow its unmatched call")
            expected_name, call_position = pending.pop(call_id)
            if name != expected_name or position <= call_position:
                raise ValueError("memory tool result does not match its earlier call")
            record_id = _validate_result(name, content)
            completed_calls.add(call_id)
            if name == SEARCH_TOOL:
                completed_reads.add(call_id)
            elif record_id is not None:
                written.append(record_id)

    if pending:
        raise ValueError("every memory tool call must have a result in the transcript")
    if len(written) != len(set(written)):
        raise ValueError("committed write record IDs must be unique")
    return MemoryTranscriptSummary(
        search_calls=len(search_positions),
        write_calls=len(write_positions),
        search_positions=tuple(search_positions),
        write_positions=tuple(write_positions),
        read_dependent_writes=dependent_writes,
        record_ids_written=tuple(written),
    )
