"""Validated random-access JSONL: offsets/source hashes, not resident episode text."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
from .data import episode_from_dict


class EpisodeIndex(Sequence):
    def __init__(self, path: str | Path):
        self.path, self.offsets = Path(path), []
        h = hashlib.sha256()
        episode_ids, source_hashes = set(), {}
        with self.path.open('rb') as f:
            while True:
                offset, line = f.tell(), f.readline()
                if not line:
                    break
                h.update(line)
                if not line.strip():
                    continue
                episode = episode_from_dict(json.loads(line))
                if episode.episode_id in episode_ids:
                    raise ValueError('Episode IDs must be unique')
                episode_ids.add(episode.episode_id)
                for source in episode.supports:
                    source_hash = hashlib.sha256(json.dumps(asdict(source), sort_keys=True).encode()).digest()
                    if source.record_id in source_hashes and source_hashes[source.record_id] != source_hash:
                        raise ValueError('A shared source ID must identify exactly the same experience')
                    source_hashes[source.record_id] = source_hash
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError('Episode file is empty')
        self.sha256 = h.hexdigest()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        offset = self.offsets[index]
        with self.path.open('rb') as f:
            f.seek(offset)
            return episode_from_dict(json.loads(f.readline()))
