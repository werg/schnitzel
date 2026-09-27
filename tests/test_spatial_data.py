import json

import pytest

from schnitz.data import make_episode
from schnitz.spatial_data import (
    SpatialTrajectoryIndex,
    pack_spatial_trajectory,
    validate_spatial_row,
)


class StableChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text, mask = "B", [0]
        for message in messages:
            rendered = f"<{message['role']}>" + str(message.get("content", ""))
            if message.get("tool_calls"):
                rendered += json.dumps(message["tool_calls"], sort_keys=True)
            rendered += "</>"
            text += rendered
            mask.extend([int(message["role"] == "assistant")] * len(rendered))
        return {"input_ids": [ord(char) for char in text], "assistant_masks": mask}


def test_pack_spatial_trajectory_inserts_distinct_blank_sites():
    episodes = [make_episode(index, distractors=0) for index in range(3)]
    row = pack_spatial_trajectory(StableChatTokenizer(), episodes, read_slots=2,
                                  generation="g0", levels=(1, 1, 2))
    validate_spatial_row(row)
    assert len(row["sites"]) == 3
    assert [site["level"] for site in row["sites"]] == [1, 1, 2]
    assert sum(token == -1 for token in row["input_ids"]) == 6
    assert all(site["query_position"] < site["workspace_start"] for site in row["sites"])
    assert row["supervised_tokens"] == sum(label >= 0 for label in row["labels"])
    # Labels predict the next token. The final blank in a workspace may therefore
    # predict the first supervised token after the injected memory.
    assert all(label == -100 or label >= 0 for label in row["labels"])


def test_spatial_index_validates_identity_and_complete_workspaces(tmp_path):
    row = pack_spatial_trajectory(StableChatTokenizer(), [make_episode(0, distractors=0)],
                                  read_slots=2, generation="g0")
    path = tmp_path / "spatial.jsonl"
    path.write_text(json.dumps(row) + "\n")
    index = SpatialTrajectoryIndex(path)
    assert len(index) == 1 and index[0]["trajectory_id"] == row["trajectory_id"]

    broken = dict(row)
    broken["input_ids"] = list(row["input_ids"])
    broken["input_ids"][row["sites"][0]["workspace_start"]] = 3
    with pytest.raises(ValueError, match="incomplete"):
        validate_spatial_row(broken)


def test_packed_writes_are_distinct_visible_sites_with_read_lineage():
    episodes = [make_episode(index, distractors=0) for index in range(3)]
    row = pack_spatial_trajectory(
        StableChatTokenizer(), episodes, read_slots=2, generation="g0",
        levels=(1, 1, 2), include_writes=True, write_generation="g1",
        write_slots=2,
    )
    validate_spatial_row(row)
    assert len(row["write_sites"]) == 3
    assert len({site["call_position"] for site in row["write_sites"]}) == 3
    assert all(write["parent_read_call_ids"] == [read["call_id"]]
               for write, read in zip(row["write_sites"], row["sites"], strict=True))
    assert all(write["call_position"] > read["workspace_start"]
               for write, read in zip(row["write_sites"], row["sites"], strict=True))
    assert sum(token == -2 for token in row['input_ids']) == 3 * (2 + 1)
    assert all(write['workspace_start'] > write['call_position']
               for write in row['write_sites'])
