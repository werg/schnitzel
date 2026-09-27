"""Pack support/query episodes into immutable multi-site trajectory token streams."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from schnitz.episode_index import EpisodeIndex
from schnitz.spatial_data import pack_spatial_trajectory
from schnitz.trajectories import file_sha256


def build(episodes_path: Path, output: Path, *, model_id: str, revision: str,
          sites_per_trajectory: int, read_slots: int, mode: str,
          generation: str, include_writes: bool = False,
          write_generation: str = "pending-authored", write_slots: int = 8) -> dict:
    if sites_per_trajectory < 1 or read_slots < 1 or mode not in {"same_level", "two_level"}:
        raise ValueError("Invalid spatial packing settings")
    episodes = EpisodeIndex(episodes_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                               trust_remote_code=False)
    pending = output.with_suffix(output.suffix + ".pending")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest, rows, sites, tokens, supervised = hashlib.sha256(), 0, 0, 0, 0
    with pending.open("wb") as handle:
        for start in range(0, len(episodes) - sites_per_trajectory + 1, sites_per_trajectory):
            group = episodes[start:start + sites_per_trajectory]
            levels = ([1] * sites_per_trajectory if mode == "same_level" else
                      [1 if index % 2 == 0 else 2 for index in range(sites_per_trajectory)])
            row = pack_spatial_trajectory(tokenizer, group, read_slots=read_slots,
                                          generation=generation, levels=levels,
                                          include_writes=include_writes,
                                          write_generation=write_generation,
                                          write_slots=write_slots if include_writes else 0)
            line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            digest.update(line)
            rows += 1
            sites += len(row["sites"])
            tokens += row["tokens"]
            supervised += row["supervised_tokens"]
        handle.flush()
        os.fsync(handle.fileno())
    if not rows:
        raise ValueError("Not enough episodes for one spatial trajectory")
    os.replace(pending, output)
    manifest = {
        "format": 1,
        "source_sha256": file_sha256(episodes_path),
        "output_sha256": digest.hexdigest(),
        "rows": rows,
        "sites": sites,
        "tokens": tokens,
        "supervised_tokens": supervised,
        "sites_per_trajectory": sites_per_trajectory,
        "read_slots": read_slots,
        "mode": mode,
        "model_id": model_id,
        "revision": revision,
        "generation": generation,
        "include_writes": include_writes,
        "write_generation": write_generation if include_writes else None,
        "write_slots": write_slots if include_writes else 0,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    pending_manifest = manifest_path.with_suffix(manifest_path.suffix + ".pending")
    pending_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(pending_manifest, manifest_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sites-per-trajectory", type=int, default=4)
    parser.add_argument("--read-slots", type=int, default=8)
    parser.add_argument("--mode", choices=("same_level", "two_level"), default="same_level")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--include-writes", action="store_true")
    parser.add_argument("--write-generation", default="pending-authored")
    parser.add_argument("--write-slots", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args.episodes, args.output, model_id=args.model_id,
                           revision=args.revision, sites_per_trajectory=args.sites_per_trajectory,
                           read_slots=args.read_slots, mode=args.mode,
                           generation=args.generation, include_writes=args.include_writes,
                           write_generation=args.write_generation,
                           write_slots=args.write_slots), indent=2))
