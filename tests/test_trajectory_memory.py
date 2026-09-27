import copy

import pytest

from schnitz.trajectory_memory import (
    search_call,
    search_result,
    validate_memory_transcript,
    write_call,
    write_result,
)


def _valid_transcript():
    scope = {"project": "opaque-project"}
    return [
        {"role": "user", "content": "Diagnose the retry failure.", "created_at": 1},
        search_call(call_id="read-1", query="retry failure", scope=scope,
                    purpose="choose diagnostic", created_at=2),
        search_result(call_id="read-1", status="ok", record_ids=["old-1"],
                      spaces={"s0": 1, "s1": 1}, generation="g0",
                      latent_attachment_id="attachment-1", created_at=3),
        {"role": "assistant", "content": "I will inspect the state.", "created_at": 4},
        {"role": "tool", "name": "inspect", "tool_call_id": "inspect-1",
         "content": "snapshot S is valid", "created_at": 5},
        write_call(call_id="write-1", content="Restore snapshot S before retrying.",
                   kind="procedural_lesson", scope=scope, applies_when="failure E17",
                   evidence_refs=["inspect-1"], confidence="observed",
                   parent_read_call_ids=["read-1"], created_at=6),
        write_result(call_id="write-1", status="ok", generation="g1",
                     record_id="new-1", created_at=7),
        search_call(call_id="read-2", query="post-restore validation", scope=scope,
                    purpose="verify final action", created_at=8),
        search_result(call_id="read-2", status="empty", record_ids=[], spaces={"s0": 0},
                      generation="g1", created_at=9),
        {"role": "assistant", "content": "Done.", "created_at": 10},
    ]


def test_distinct_memory_sites_and_read_before_write_lineage():
    summary = validate_memory_transcript(_valid_transcript())
    assert summary.search_calls == 2
    assert summary.write_calls == 1
    assert summary.search_positions == (1, 7)
    assert summary.write_positions == (5,)
    assert summary.read_dependent_writes == 1
    assert summary.record_ids_written == ("new-1",)


def test_rejects_packed_calls_and_multi_item_write():
    packed = _valid_transcript()
    packed[1]["tool_calls"].append(copy.deepcopy(packed[7]["tool_calls"][0]))
    with pytest.raises(ValueError, match="sole tool call"):
        validate_memory_transcript(packed)

    batched = _valid_transcript()
    batched[5]["tool_calls"][0]["function"]["arguments"]["items"] = [{"content": "x"}]
    with pytest.raises(ValueError, match="distinct calls"):
        validate_memory_transcript(batched)


def test_rejects_future_read_lineage_and_source_text_result():
    future = _valid_transcript()
    future[5]["tool_calls"][0]["function"]["arguments"]["parent_read_call_ids"] = ["read-2"]
    with pytest.raises(ValueError, match="completed earlier"):
        validate_memory_transcript(future)

    leaked = _valid_transcript()
    leaked[2]["content"] = leaked[2]["content"][:-1] + ',"source_text":"secret"}'
    with pytest.raises(ValueError, match="source-bearing"):
        validate_memory_transcript(leaked)


def test_nonempty_search_requires_separate_latent_attachment():
    transcript = _valid_transcript()
    transcript[2]["content"] = transcript[2]["content"].replace(
        ',"latent_attachment_id":"attachment-1"', ""
    )
    with pytest.raises(ValueError, match="attachment ID"):
        validate_memory_transcript(transcript)
