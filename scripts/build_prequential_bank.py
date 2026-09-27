"""Grow one revisioned mutable bank by replaying causal memory trajectories.

Every event reads the current admissible logical bank. Its all-space writes and
event frontier become visible in one journal transaction after the trajectory ends.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

from safetensors.torch import load_model
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.checkpoints import resolve_checkpoint, stop_on_signal
from schnitz.key_index import PublishedKeyIndex
from schnitz.offline_bank import (canonical_json, publish_offline_generation,
                               stored_memory_identity)
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.spatial_data import SpatialTrajectoryIndex
from schnitz.spatial_training import spatial_bank_forward
from schnitz.store import DiskStore, StoredRecord
from schnitz.training_bank import TrainingBank
from schnitz.training import autocast_context, config_from_run, environment_report, resource_report
from schnitz.trajectories import file_sha256


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _inventory(data: SpatialTrajectoryIndex, parent: PublishedKeyIndex,
               max_events: int | None) -> tuple[int, list[int], set[str]]:
    count = len(data) if max_events is None else min(len(data), max_events)
    if count < 1:
        raise ValueError("The prequential stream needs at least one event")
    cumulative, record_ids, total = [0], set(), 0
    parent_ids = set(map(str, next(iter(parent.spaces.values())).ids))
    for position in range(count):
        writes = data[position].get("write_sites", ())
        if not writes:
            raise ValueError("Every prequential event must contain write tool calls")
        for write in writes:
            record_id = write["record_id"]
            if record_id in record_ids or record_id in parent_ids:
                raise ValueError("Prequential record IDs must be globally unique")
            record_ids.add(record_id)
        total += len(writes)
        cumulative.append(total)
    return count, cumulative, record_ids


def _model_bank_compatible(config, manifest: dict) -> None:
    memory = stored_memory_identity(asdict(config.memory))
    bank_memory = stored_memory_identity(dict(manifest["identity"]["memory"]))
    model = manifest["identity"]["model"]
    if memory != bank_memory or any(
        model[name] != getattr(config.model, name)
        for name in ("model_id", "revision", "writer_loops")
    ):
        raise ValueError("Executor model and parent stored interface differ")


def _validate_write_texts(agent: SchnitzelAgent, texts: list[str], maximum: int) -> None:
    for text in texts:
        ids = agent.tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            ids = [agent.tokenizer.bos_token_id or 1]
        if len(ids) > maximum:
            raise ValueError(
                f"Authored memory uses {len(ids)} tokens, exceeding --max-write-tokens={maximum}")


def build(run: Path, parent_bank: Path, trajectories: Path, output: Path, *,
          external_root: Path, stream: str, limits: tuple[int, ...],
          routing_candidates: int, max_write_tokens: int,
          min_free_bytes: int, max_events: int | None = None) -> dict:
    if (not _under(output, external_root) or output == external_root.resolve()
            or routing_candidates < 1 or max_write_tokens < 1 or min_free_bytes < 0):
        raise ValueError("Use a child of the declared external root and positive budgets")
    checkpoint = resolve_checkpoint(run, verify=True)
    config = deepcopy(config_from_run(run))
    config.train.bank_dir = None
    config.train.episodes_file = None
    config.train.tokenized_episodes_file = None
    config.train.batch_size = 1
    config.validate()
    if config.model.writer_loops != config.model.loops:
        raise ValueError('Integrated authored writes require writer_loops to match this trajectory depth')
    if len(limits) != len(config.memory.payload_dims):
        raise ValueError("Supply one prequential read limit per memory space")

    parent_manifest_path = parent_bank / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_store = DiskStore(parent_bank / "bank.sqlite")
    verified = publish_offline_generation(
        parent_store, identity=parent_manifest["identity"],
        namespace=parent_manifest["namespace"], generation=parent_manifest["generation"],
        spaces=tuple(parent_manifest["spaces"]), shard_ids=tuple(parent_manifest["shards"]),
        source_count=parent_manifest["sources"], verify_only=True,
    )
    if canonical_json(verified) != canonical_json(
            {key: parent_manifest[key] for key in verified}):
        raise ValueError("Parent bank failed byte verification")
    _model_bank_compatible(config, parent_manifest)
    parent_index = PublishedKeyIndex(
        parent_store, namespace=parent_manifest["namespace"],
        generation=parent_manifest["generation"], spaces=tuple(parent_manifest["spaces"]),
        expected_sources=parent_manifest["sources"],
    )
    data = SpatialTrajectoryIndex(trajectories)
    event_count, cumulative_writes, record_ids = _inventory(data, parent_index, max_events)
    parent_max_time = max(int(space.times.max()) for space in parent_index.spaces.values())
    base_time = parent_max_time + 1
    core_identity = {
        "format": 2, "kind": "prequential-mutable-bank",
        "writer_checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "writer_checkpoint": str(checkpoint), "parent_bank": str(parent_bank.resolve()),
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "parent_generation": parent_manifest["generation"],
        "trajectory_file": str(trajectories.resolve()), "trajectory_sha256": data.sha256,
        "events": event_count, "authored_records": len(record_ids), "stream": stream,
        "base_time": base_time, "spaces": list(parent_manifest["spaces"]),
        "memory": asdict(config.memory), "limits": list(limits),
        "routing_candidates": routing_candidates, "max_write_tokens": max_write_tokens,
        "causal_policy": "event N sees logical records admitted before position N",
    }
    identity = core_identity | {
        "namespace": parent_manifest["namespace"],
        "catalog_generation": parent_manifest["generation"],
    }
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Prequential output identity changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Existing prequential output has no identity")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(identity_path, identity)
    journal = DiskStore(output / "bank.sqlite")
    store = TrainingBank(parent_store, journal, parent_index)
    index = parent_index
    frontier = journal.event_frontier(
        namespace=index.namespace, stream=stream, generation=index.generation)
    start = frontier["next_position"]
    authored_count = len(index.spaces[next(iter(index.spaces))].ids) - parent_manifest['sources']
    if start > event_count or authored_count != cumulative_writes[start]:
        raise ValueError("Authored records and committed event frontier disagree")
    expected_time = -1 if start == 0 else base_time + start - 1
    if frontier["visibility_time"] != expected_time:
        raise ValueError("Committed visibility time differs from the causal frontier")
    if shutil.disk_usage(output).free < min_free_bytes:
        raise OSError("External disk reserve is already below --min-free-bytes")

    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    load_model(agent, str(checkpoint / "model.safetensors"), device=config.train.device)
    began, completed = time.perf_counter(), start
    event_log = output / "events.jsonl"
    with run_lock(output), stop_on_signal() as signal_state, event_log.open("a") as log:
        atomic_json(output / "environment.json", environment_report())
        for position in range(start, event_count):
            if signal_state["signal"] is not None or stop_requested(output):
                break
            if shutil.disk_usage(output).free < min_free_bytes:
                raise OSError("External disk reserve reached before the next atomic event")
            row = deepcopy(data[position])
            query_time = base_time + position
            for site in row["sites"]:
                site["query_time"] = query_time
            observed: dict[str, dict[str, tuple[str, ...]]] = {}

            def observe(call_id, space, plan):
                observed.setdefault(call_id, {})[space] = tuple(
                    selection.record_id for selection in plan.selections)

            tick = time.perf_counter()
            with torch.no_grad(), autocast_context(config):
                result = spatial_bank_forward(
                    agent, store, index, [row], limits=limits,
                    routing_candidates=routing_candidates,
                    pad_token_id=agent.tokenizer.pad_token_id or 0,
                    plan_observer=observe,
                )
                writes = row["write_sites"]
                _validate_write_texts(
                    agent, [write["content"] for write in writes], max_write_tokens)
                outputs = result.write_outputs
                if outputs is None or outputs[0].shape[0] != len(writes):
                    raise ValueError('Trajectory did not produce one integrated state per write call')
            records, lineage, dependencies, metadata = [], {}, {}, {}
            storage_dtype = getattr(torch, config.memory.storage_dtype)
            by_call = {site["call_id"]: site for site in row["sites"]}
            for write_index, write in enumerate(writes):
                parent_calls = write["parent_read_call_ids"]
                domains = {by_call[call_id]["domain"] for call_id in parent_calls}
                if len(domains) != 1 or any(
                        set(observed.get(call_id, ())) != set(parent_manifest["spaces"])
                        for call_id in parent_calls):
                    raise ValueError("Write lineage lacks a complete authorization-homogeneous read")
                domain = next(iter(domains))
                record_id = write["record_id"]
                parents, child_ids = set(), set()
                for call_id in parent_calls:
                    for ids in observed.get(call_id, {}).values():
                        for selected_id in ids:
                            parents.add(f"record:{selected_id}")
                            child_ids.add(selected_id)
                lineage[record_id] = tuple(sorted(parents))
                dependencies[record_id] = tuple(sorted(child_ids))
                metadata[record_id] = {
                    "trajectory_id": row["trajectory_id"], "write_call_id": write["call_id"],
                    "episode_id": write["episode_id"],
                    "parent_read_call_ids": write["parent_read_call_ids"],
                    "content_sha256": hashlib.sha256(write["content"].encode()).hexdigest(),
                    "query_time": query_time,
                }
                for space in range(len(config.memory.payload_dims)):
                    records.append(StoredRecord(
                        record_id, outputs[2 * space][write_index].detach(),
                        outputs[2 * space + 1][write_index].to(storage_dtype).detach(),
                        namespace=index.namespace, space=f"s{space}",
                        generation=index.generation, domain=domain,
                        created_at=query_time, source_id=row["trajectory_id"],
                    ))
            store.update(records, children=dependencies, event={
                'stream': stream, 'position': position,
                'event_id': row['trajectory_id'], 'visibility_time': query_time,
                'lineage': lineage, 'record_metadata': metadata,
            })
            completed = position + 1
            authored_count = len(index.spaces[next(iter(index.spaces))].ids) \
                - parent_manifest['sources']
            event_row = {
                "position": position, "event_id": row["trajectory_id"],
                "writes": len(writes), "authored_records": authored_count,
                "query_time": query_time, "loss": float(result.loss),
                "nll": float(result.nll), "routing": float(result.routing),
                "seconds": time.perf_counter() - tick,
                "selected_counts": result.metrics["selected_counts"],
            }
            log.write(json.dumps(event_row) + "\n")
            log.flush()
            atomic_json(output / "progress.json", event_row | {
                "completed_events": completed, "total_events": event_count})
            print(json.dumps(event_row), flush=True)
    bank_sizes = store.sizes()
    summary = {"logical_records": parent_manifest['sources'] + cumulative_writes[completed],
               "journal_cursor": bank_sizes['cursor'],
               "revision_views": bank_sizes['revision_views'],
               "journal_tensor_bytes": bank_sizes['tensor_bytes']}
    manifest = identity | summary | {
        "completed_events": completed, "complete": completed == event_count,
        "elapsed_seconds": time.perf_counter() - began,
        "resources": resource_report(),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-bank", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--stream", default="trajectory-v0.5")
    parser.add_argument("--limits", nargs="+", type=int, default=[16, 8, 4, 4])
    parser.add_argument("--routing-candidates", type=int, default=256)
    parser.add_argument("--max-write-tokens", type=int, default=256)
    parser.add_argument("--min-free-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    print(json.dumps(build(
        args.run, args.parent_bank, args.trajectories, args.output,
        external_root=args.external_root, stream=args.stream,
        limits=tuple(args.limits), routing_candidates=args.routing_candidates,
        max_write_tokens=args.max_write_tokens, min_free_bytes=args.min_free_bytes,
        max_events=args.max_events,
    ), indent=2))
