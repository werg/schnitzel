"""Train spatial reads against a base snapshot plus checkpointed mutable revisions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from safetensors.torch import load_model
import torch

from sdkb.agent import SDKBAgent
from sdkb.bank_curriculum import BankCurriculum
from sdkb.bank_replay import BankWriterReplay
from sdkb.checkpoints import (resolve_checkpoint, restore_checkpoint, save_checkpoint,
                              stop_on_signal)
from sdkb.config import load_config
from sdkb.key_geometry import BankLoad
from sdkb.record_gradients import (GradientSink, KeyStateCache, RecordGradients,
                                   KeyTable, refresh_records, writer_backward, writer_pass)
from sdkb.key_index import PublishedKeyIndex
from sdkb.document_ingestion import (grouped_ingestion_prefixes,
                                     source_ingestion_groups, writer_prefix_ids)
from sdkb.offline_bank import (canonical_json, publish_offline_generation,
                               stored_memory_identity, writer_prompt_generation)
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.optimizers import make_optimizer, optimizer_report
from sdkb.runtime import available_host_memory, configure_memory, reclaim_cuda_cache
from sdkb.spatial_data import SpatialTrajectoryIndex
from sdkb.spatial_training import spatial_bank_forward, spatial_bank_pipeline_forward
from sdkb.routing_curriculum import RoutingCandidateIndex
from sdkb.store import DiskStore
from sdkb.training_bank import TrainingBank
from sdkb.tracking import Tracking
from sdkb.training import EpisodeSampler, autocast_context, environment_report, resource_report
from sdkb.trajectories import file_sha256


# Compute-budget settings that may change on resume (see ``train``).
ADJUSTABLE = ('record_refresh_per_step', 'record_refresh_backward', 'record_flush_budget',
              'record_flush_extra')
# Behavior of settings added after a run launched without them.
LEGACY_DEFAULTS = {'record_refresh_backward': True}


def _fingerprint(data: SpatialTrajectoryIndex, manifest_path: Path, settings: dict) -> str:
    payload = ":".join((data.sha256, file_sha256(manifest_path), canonical_json(settings)))
    return hashlib.sha256(payload.encode()).hexdigest()


def train(config_path: Path, data_path: Path, bank_dir: Path, output: Path,
          init_from: Path | None, *, resume: bool, steps: int, batch_size: int,
          loops: int, limits: tuple[int, ...], routing_candidates: int,
          checkpoint_every: int, train_recurrent_core: bool = False,
          microbatch_size: int | None = None, inflight: int = 1,
          sources_path: Path | None = None,
          max_unused_cuda_gib: float = 4.0,
          gradient_checkpointing: bool = False,
          profile_steps: int = 0,
          retain_writer_replay_activations: bool = False,
          cache_reclaim_host_reserve_gib: float = 16.0,
          maintenance_records_per_step: int = 0,
          routing_episodes_path: Path | None = None,
          routing_weight: float | None = None,
          inherit_bank: bool = False,
          bank_journal_path: Path | None = None,
          writer_key_learning_rate: float | None = None,
          key_stability_weight: float | None = None,
          routing_logit_scale: float | None = None,
          routing_live_weight: float | None = None,
          routing_hard_ramp_steps: int = 3000,
          min_host_available_gib: float = 0.0,
          host_pressure_wait_seconds: float = 600.0,
          gold_force: tuple[float, float, int] | None = None,
          spreading: dict | None = None,
          key_states_path: Path | None = None,
          device_search: bool = False,
          curriculum_sizes: tuple[int, ...] = (),
          curriculum_min_steps: int = 300,
          curriculum_max_steps: int = 3000,
          payload_cache_gib: float = 0.0,
          archive_dir: Path | None = None,
          record_budget: dict | None = None,
          writer_batch_size: int = 16,
          writer_checkpointing: bool = True,
          curriculum_value_cosine: float = 0.99) -> dict:
    if (steps < 1 or batch_size < 1 or loops < 2 or checkpoint_every < 1
            or max_unused_cuda_gib < 0 or profile_steps < 0
            or cache_reclaim_host_reserve_gib < 0
            or maintenance_records_per_step < 0 or routing_hard_ramp_steps < 1
            or min_host_available_gib < 0 or host_pressure_wait_seconds < 0):
        raise ValueError("Invalid spatial training schedule")
    max_unused_cuda_bytes = int(max_unused_cuda_gib * 1024 ** 3)
    min_host_available_bytes = int(min_host_available_gib * 1024 ** 3)
    cache_reclaim_host_reserve_bytes = int(cache_reclaim_host_reserve_gib * 1024 ** 3)
    pipeline = microbatch_size is not None or inflight != 1
    if pipeline:
        microbatch_size = microbatch_size or 1
        if microbatch_size < 1 or microbatch_size >= batch_size or inflight < 2:
            raise ValueError(
                "Pipelining needs a microbatch smaller than the optimizer batch "
                "and at least two batches in flight")
    config = deepcopy(load_config(config_path))
    if gradient_checkpointing:
        config.model.gradient_checkpointing = True
    config.model.loops = loops
    if train_recurrent_core:
        config.model.freeze_backbone = False
        config.model.backbone_train_scope = "recurrent_core"
    config.memory.read_steps = loops - 1
    config.train.steps = steps
    config.train.batch_size = batch_size
    config.train.gradient_accumulation = 1
    config.train.bank_dir = None
    config.train.episodes_file = None
    config.train.tokenized_episodes_file = None
    config.train.live_fraction = 0
    config.train.checkpoint_every = checkpoint_every
    if routing_weight is not None:
        if routing_weight <= 0:
            raise ValueError('Routing weight must be positive')
        config.train.routing_weight = routing_weight
    if writer_key_learning_rate is not None:
        config.train.writer_key_learning_rate = writer_key_learning_rate
    if key_stability_weight is not None:
        config.train.key_stability_weight = key_stability_weight
    if routing_logit_scale is not None:
        config.train.routing_logit_scale = routing_logit_scale
    if routing_live_weight is not None:
        config.train.routing_live_weight = routing_live_weight
    for name, value in (record_budget or {}).items():
        if value is not None:
            setattr(config.train, name, value)
    for name, value in (spreading or {}).items():
        if value is not None:
            setattr(config.train, name, value)
    # Forced gold: probability anneals linearly from start to a floor, never zero
    # unless asked, so gates keep seeing gold records and keys keep their gradient.
    if gold_force is not None:
        force_start, force_floor, force_steps = gold_force
        if not 0 <= force_floor <= force_start <= 1 or force_steps < 0:
            raise ValueError('Invalid forced-gold schedule')
        config.train.gold_force_probability = force_start
    if routing_episodes_path is not None and (sources_path is None or not pipeline):
        raise ValueError('Routing curriculum requires source metadata and pipeline')
    if inherit_bank and (init_from is None or resume):
        raise ValueError('Bank journal inheritance needs a warm-start parent')
    if bank_journal_path is not None and (init_from is None or resume or inherit_bank):
        raise ValueError('A staged bank refresh needs a fresh warm-start output')
    # Retain enough recovery points for a long unattended trajectory stage.
    config.train.keep_checkpoints = max(config.train.keep_checkpoints, 16)
    if config.train.record_gradients:
        # Only the newest record states are kept (each is several GB), so older
        # checkpoints cannot resume; retaining few also lets checkpoint GC release
        # the mutable-bank journal (about 70 MB of payload revisions per step).
        config.train.keep_checkpoints = 3
    config.train.archive_dir = None
    config.train.wandb_group = f"trajectory-spatial-r{loops}"
    config.validate()
    data = SpatialTrajectoryIndex(data_path)
    if routing_episodes_path is not None:
        trajectory_manifest = data_path.with_suffix(data_path.suffix + '.manifest.json')
        packed = json.loads(trajectory_manifest.read_text())
        if (packed['source_sha256'] != file_sha256(routing_episodes_path)
                or packed['output_sha256'] != data.sha256):
            raise ValueError('Routing query text differs from packed causal trajectories')
    bank_manifest_path = bank_dir / "manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text())
    store = DiskStore(bank_dir / "bank.sqlite")
    # Full verification reads the whole bank. Reuse a previous result while the
    # manifest and the database file (size, mtime, inode) are unchanged.
    bank_stat = (bank_dir / "bank.sqlite").stat()
    verification_key = {"manifest_sha256": file_sha256(bank_manifest_path),
                        "size": bank_stat.st_size, "mtime_ns": bank_stat.st_mtime_ns,
                        "inode": bank_stat.st_ino}
    verification_cache = bank_dir / "verification-cache.json"
    cached = (json.loads(verification_cache.read_text())
              if verification_cache.is_file() else {})
    if cached.get("key") == verification_key:
        verified = cached["verified"]
    else:
        verified = publish_offline_generation(
            store, identity=bank_manifest["identity"], namespace=bank_manifest["namespace"],
            generation=bank_manifest["generation"], spaces=tuple(bank_manifest["spaces"]),
            shard_ids=tuple(bank_manifest["shards"]), source_count=bank_manifest["sources"],
            verify_only=True,
        )
        try:
            atomic_json(verification_cache, {"key": verification_key, "verified": verified})
        except OSError:
            pass
    if canonical_json(verified) != canonical_json({key: bank_manifest[key] for key in verified}):
        raise ValueError("Published bank differs from its verified manifest")
    bank_memory = stored_memory_identity(dict(bank_manifest["identity"]["memory"]))
    current_memory = stored_memory_identity(asdict(config.memory))
    if bank_memory.get('key_interface') != current_memory['key_interface']:
        # Base-snapshot keys came from another key interface. Only a complete
        # refreshed journal written by this interface may replace every view.
        interface_journal = None
        if bank_journal_path is not None:
            interface_journal = json.loads(
                (bank_journal_path.parent / 'manifest.json').read_text()).get('key_interface')
        elif resume and (output / 'spatial-inputs.json').exists():
            interface_journal = json.loads(
                (output / 'spatial-inputs.json').read_text()).get('journal_key_interface')
        if interface_journal != current_memory['key_interface']:
            raise ValueError('Key interface differs from the bank without a complete refresh')
        bank_memory['key_interface'] = current_memory['key_interface']
    if (tuple(bank_manifest["spaces"]) != tuple(f"s{i}" for i in range(len(limits)))
            or bank_memory != current_memory):
        raise ValueError("Spatial model and published bank interfaces differ")
    if config.train.writer_replay_records_per_site and (not pipeline or sources_path is None):
        raise ValueError('Writer replay requires --sources and the overlapped pipeline')
    if maintenance_records_per_step and not config.train.writer_replay_records_per_site:
        raise ValueError('Bank maintenance requires writer replay and source inputs')
    source_sha = None
    if sources_path is not None:
        source_sha = file_sha256(sources_path)
        if source_sha != bank_manifest['identity']['source_manifest_sha256']:
            raise ValueError('Writer replay source manifest differs from bank creation')
    settings = {"format": 2, "loops": loops, "limits": limits,
                "routing_candidates": routing_candidates, "batch_size": batch_size,
                "steps": steps, "sampler": "deterministic_shuffled_passes",
                "train_recurrent_core": train_recurrent_core,
                "source_manifest_sha256": source_sha}
    if pipeline:
        settings["pipeline"] = {
            "microbatch_size": microbatch_size, "inflight": inflight,
            "ordering": "deterministic_round_robin",
        }
    if maintenance_records_per_step:
        settings['maintenance_records_per_step'] = maintenance_records_per_step
    if routing_episodes_path is not None:
        settings['routing_curriculum'] = {
            'episodes_sha256': file_sha256(routing_episodes_path),
            'routing_weight': config.train.routing_weight,
            'sampled_negatives': 32,
            'hard_ramp_steps': routing_hard_ramp_steps,
            'lexical_auxiliary_fraction': .05,
        }
    if config.train.key_stability_weight:
        settings['key_stability_weight'] = config.train.key_stability_weight
    if config.train.routing_logit_scale != 1.0:
        settings['routing_logit_scale'] = config.train.routing_logit_scale
    if config.train.routing_live_weight:
        settings['routing_live_weight'] = config.train.routing_live_weight
    if config.train.writer_key_learning_rate is not None:
        settings['writer_key_learning_rate'] = config.train.writer_key_learning_rate
    if gold_force is not None:
        settings['gold_force'] = {'start': gold_force[0], 'floor': gold_force[1],
                                  'anneal_steps': gold_force[2]}
    spread = {name: getattr(config.train, name) for name in (
        'exploration_fraction', 'spread_variance_weight', 'spread_covariance_weight',
        'koleo_weight', 'load_penalty_weight', 'load_decay', 'load_threshold')}
    if config.train.record_gradients:
        if not pipeline or sources_path is None:
            raise ValueError('Record gradients require --sources and the overlapped pipeline')
        if key_states_path is None:
            raise ValueError('Record gradients need the bank key-state cache')
        settings['record_gradients'] = {name: getattr(config.train, name) for name in (
            'record_gradient_decay', 'record_neighborhood_fraction', 'record_flush_budget',
            'record_flush_extra', 'record_refresh_per_step', 'record_gradient_capacity',
            'record_refresh_backward')}
        settings['record_gradients']['key_states_sha256'] = file_sha256(key_states_path)
    if curriculum_sizes:
        if not config.train.key_table:
            raise ValueError('A bank curriculum needs the key table')
        settings['curriculum'] = {'sizes': list(curriculum_sizes),
                                  'min_steps': curriculum_min_steps,
                                  'max_steps': curriculum_max_steps}
        if config.train.key_table:
            settings['record_gradients']['key_table'] = {name: getattr(config.train, name)
                                                         for name in (
                'key_table_learning_rate', 'key_table_optimizer', 'key_table_max_step',
                'key_prediction_weight',
                'key_commitment_weight')}
    if any((spread['exploration_fraction'], spread['spread_variance_weight'],
            spread['spread_covariance_weight'], spread['koleo_weight'],
            spread['load_penalty_weight'])):
        settings['spreading'] = spread
    refresh_manifest = None
    if bank_journal_path is not None:
        refresh_manifest_path = bank_journal_path.parent / 'manifest.json'
        refresh_manifest = json.loads(refresh_manifest_path.read_text())
        if not refresh_manifest.get('complete'):
            raise ValueError('Staged bank refresh is incomplete')
        settings['bank_refresh_manifest_sha256'] = file_sha256(refresh_manifest_path)
        if 'key_interface' in refresh_manifest:
            settings['journal_key_interface'] = refresh_manifest['key_interface']
    if resume and (output / 'spatial-inputs.json').exists():
        prior = json.loads((output / 'spatial-inputs.json').read_text())
        inherited = prior.get('inherit_bank_from')
        if inherited is not None:
            settings['inherit_bank_from'] = inherited
        refreshed = prior.get('bank_refresh_manifest_sha256')
        if refreshed is not None:
            settings['bank_refresh_manifest_sha256'] = refreshed
        if 'journal_key_interface' in prior:
            settings['journal_key_interface'] = prior['journal_key_interface']
    # Compute-budget settings may change on resume without changing the run: the
    # fingerprint keeps their launch-time values (or their absence, for settings
    # added later) and each change is logged in spatial-inputs.json.
    fingerprint_settings = deepcopy(settings)
    schedule_changes = []
    if 'record_gradients' in settings:
        record = settings['record_gradients']
        if resume and (output / 'spatial-inputs.json').exists():
            prior = json.loads((output / 'spatial-inputs.json').read_text())
            schedule_changes = list(prior.get('schedule_changes', []))
            prior_record = prior.get('record_gradients', {})
            origin = prior.get('adjustable_origin') or {
                name: ({'present': True, 'value': prior_record[name]}
                       if name in prior_record else {'present': False})
                for name in ADJUSTABLE}
            for name in ADJUSTABLE:
                previous = prior_record.get(name, LEGACY_DEFAULTS.get(name))
                if previous != record.get(name):
                    schedule_changes.append({'setting': name, 'from': previous,
                                             'to': record.get(name)})
        else:
            origin = {name: {'present': True, 'value': record[name]} for name in ADJUSTABLE}
        for name in ADJUSTABLE:
            if origin[name]['present']:
                fingerprint_settings['record_gradients'][name] = origin[name]['value']
            else:
                fingerprint_settings['record_gradients'].pop(name, None)
        settings['adjustable_origin'] = origin
        settings['schedule_changes'] = schedule_changes
    if curriculum_sizes:
        # A stage-advance threshold, not part of the run's identity; changes are logged.
        prior_threshold = 0.99
        if resume and (output / 'spatial-inputs.json').exists():
            prior_threshold = json.loads((output / 'spatial-inputs.json').read_text()).get(
                'curriculum_value_cosine', 0.99)
        if prior_threshold != curriculum_value_cosine:
            settings.setdefault('schedule_changes', []).append(
                {'setting': 'curriculum_value_cosine', 'from': prior_threshold,
                 'to': curriculum_value_cosine})
        settings['curriculum_value_cosine'] = curriculum_value_cosine
    fingerprint = _fingerprint(data, bank_manifest_path, {
        key: value for key, value in fingerprint_settings.items()
        if key not in ('adjustable_origin', 'schedule_changes', 'curriculum_value_cosine')})

    if resume:
        if init_from is not None:
            raise ValueError("Choose resume or warm start")
        old = json.loads((output / "spatial-inputs.json").read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Spatial data, bank, or scientific schedule changed")
    else:
        if init_from is None:
            raise ValueError("A spatial stage needs an explicit compatible warm start")
        output.mkdir(parents=True, exist_ok=False)

    # Unified memory: cap the caching allocator before any CUDA allocation.
    memory_report = configure_memory(config.train)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(config.train.threads)
    agent = SDKBAgent(config).to(config.train.device)
    for name, parameter in agent.named_parameters():
        if name.startswith(("write_slots", "key_head.", "value_head.",
                            "address_maps.", "writer_key_heads.", "codecs.")):
            parameter.requires_grad_(bool(config.train.writer_replay_records_per_site
                                          or config.train.record_gradients))
    agent.train()
    initialization = None
    if not resume:
        checkpoint = resolve_checkpoint(init_from, verify=True)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if manifest["resolved_model_revision"] != agent.resolved_revision:
            raise ValueError("Warm-start model revision differs")
        missing, unexpected = load_model(agent, str(checkpoint / "model.safetensors"),
                                         strict=False, device=config.train.device)
        allowed_missing = {name for name in missing if name.startswith('distance_gates.')}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(f"Spatial warm start is incompatible: {missing=}, {unexpected=}")
        bank_writer_exact = (file_sha256(checkpoint / "model.safetensors")
                             == bank_manifest["identity"]["writer_checkpoint_sha256"])
        if loops == 2 and not bank_writer_exact:
            raise ValueError("Initial spatial stage must start from the bank writer snapshot")
        # A deeper stage may also start from the exact writer that built this bank
        # (for example a keyspace-pretrained writer that already ran deeper loops).
        if loops > 2 and not bank_writer_exact:
            parent_inputs = init_from / "spatial-inputs.json"
            if (not parent_inputs.is_file()
                    or json.loads(parent_inputs.read_text())["bank_manifest_sha256"]
                    != file_sha256(bank_manifest_path)):
                raise ValueError("Deeper spatial stages require a parent trained on this bank")
        initialization = {"checkpoint": str(checkpoint),
                          "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
                          "optimizer_reset": True, "bank_writer_exact": bank_writer_exact}
        if inherit_bank:
            parent_state = json.loads((checkpoint / 'bank-state.json').read_text())
            parent_cache = DiskStore(init_from / 'training_cache.sqlite')
            if parent_cache.mutable_bank_state() != parent_state:
                raise ValueError('Parent bank journal has moved beyond its checkpoint')
            with parent_cache.connect() as source, sqlite3.connect(
                    output / 'training_cache.sqlite') as target:
                source.backup(target)
            initialization['inherited_bank_state'] = parent_state
        if bank_journal_path is not None:
            if (refresh_manifest['writer_checkpoint_sha256']
                    != initialization['checkpoint_sha256']
                    or refresh_manifest['bank_manifest_sha256']
                    != file_sha256(bank_manifest_path)
                    or refresh_manifest['source_manifest_sha256'] != source_sha):
                raise ValueError('Refreshed bank and warm-start model differ')
            staged = DiskStore(bank_journal_path)
            if staged.mutable_bank_state() != refresh_manifest['journal_state']:
                raise ValueError('Refreshed bank journal differs from its manifest')
            with staged.connect() as source, sqlite3.connect(
                    output / 'training_cache.sqlite') as target:
                source.backup(target)
            initialization['bank_refresh_manifest_sha256'] = settings[
                'bank_refresh_manifest_sha256']
            initialization['inherited_bank_state'] = refresh_manifest['journal_state']
        atomic_json(output / "initialization.json", initialization)
    atomic_json(output / "spatial-inputs.json", settings | {
        "fingerprint": fingerprint, "data": str(data_path), "data_sha256": data.sha256,
        "bank": str(bank_dir), "bank_manifest_sha256": file_sha256(bank_manifest_path),
        "generation": bank_manifest["generation"], "rows": len(data),
    })
    atomic_json(output / "config.json", asdict(config))
    optimizer = make_optimizer(agent)
    archiver = None
    if archive_dir is not None:
        # Live run on fast local storage; completed checkpoints are copied to the
        # archive by a background thread, off the training step.
        from sdkb.archiving import CheckpointArchiver
        from sdkb.tracking import run_identity
        archive_run = Path(archive_dir) / run_identity(output)
        archive_run.mkdir(parents=True, exist_ok=True)
        archiver = CheckpointArchiver(archive_run, keep=config.train.archive_keep_checkpoints,
                                      reserve_bytes=config.train.min_free_disk_bytes)
        atomic_json(archive_run / 'run.json', {'id': run_identity(output),
                                               'source_run': str(output.resolve())})
    rng = random.Random(config.train.seed)
    cache = DiskStore(output / "training_cache.sqlite")
    start = restore_checkpoint(agent, optimizer, output, rng, fingerprint) if resume else 0
    if start > steps:
        raise ValueError("Saved spatial step exceeds requested budget")
    atomic_json(output / "optimizer.json", {
        "kind": config.train.optimizer, "groups": optimizer_report(optimizer),
        "resumed": resume, "completed_steps": start,
    })
    index = PublishedKeyIndex(
        store, namespace=bank_manifest["namespace"], generation=bank_manifest["generation"],
        spaces=tuple(bank_manifest["spaces"]), expected_sources=bank_manifest["sources"],
    )
    training_bank = TrainingBank(store, cache, index)
    if payload_cache_gib > 0:
        # Hot payloads stay in host memory; bank reads otherwise hit disk randomly.
        training_bank.enable_payload_cache(int(payload_cache_gib * 1024 ** 3))
    bank_load = None
    if config.train.exploration_fraction or config.train.load_penalty_weight:
        bank_load = BankLoad(next(iter(index.spaces.values())).ids, len(limits),
                             decay=config.train.load_decay,
                             threshold=config.train.load_threshold)
        if resume and start:
            saved = output / f'bank-load-{start:09d}.pt'
            if not saved.is_file():
                raise ValueError('Retrieval-load state is missing for the resumed step')
            bank_load.load_state_dict(torch.load(saved, weights_only=True))

    key_cache = record_grads = key_table = None
    if str(config.train.device).startswith('cuda') or device_search:
        # Exact scan on the training device: same scores, eligibility and tie
        # order as the NumPy scan, without per-query CPU work.
        index.use_device(config.train.device)
    if config.train.record_gradients:
        from safetensors.torch import load_file as load_tensors
        record_ids = [str(record_id) for record_id in next(iter(index.spaces.values())).ids]
        record_grads = RecordGradients(config.train.record_gradient_decay,
                                       capacity=config.train.record_gradient_capacity)
        if resume and start:
            saved = torch.load(output / f'record-state-{start:09d}.pt', weights_only=False)
            key_cache = KeyStateCache.from_state_dict(record_ids, saved['cache'],
                                                      config.train.device)
            record_grads.load_state_dict(saved['grads'])
        else:
            tensors = load_tensors(str(key_states_path))
            key_cache = KeyStateCache(record_ids, tensors['states'].to(config.train.device),
                                      tensors['versions'])
            # The cache must reproduce the published keys under the starting heads.
            sample = torch.randperm(len(record_ids), generator=torch.Generator().manual_seed(
                config.train.seed))[:256]
            with torch.no_grad():
                derived = key_cache.keys(agent, sample)
            for space, array in enumerate(index.spaces.values()):
                stored = torch.from_numpy(array.keys[sample.numpy()])
                if float(torch.nn.functional.cosine_similarity(
                        derived[space].float().cpu(), stored, dim=-1).min()) < 0.999:
                    raise ValueError('Key-state cache does not reproduce the bank keys')
        if config.train.key_table:
            # Terminal keys start as the published keys; decoder drift never moves them.
            key_table = KeyTable(record_ids, [torch.from_numpy(array.keys.copy()).to(
                config.train.device) for array in index.spaces.values()],
                learning_rate=config.train.key_table_learning_rate,
                optimizer=config.train.key_table_optimizer,
                max_step=config.train.key_table_max_step)
            if resume and start:
                key_table.load_state_dict(saved['table'])
            key_table.sync_index(index)
        else:
            key_cache.sync_index(agent, index)
    curriculum = None
    if curriculum_sizes:
        curriculum = BankCurriculum(
            curriculum_sizes, [[record_id for site in data[row]['sites']
                                for record_id in site['required_ids']]
                               for row in range(len(data))],
            record_ids, seed=config.train.seed, min_steps=curriculum_min_steps,
            max_steps=curriculum_max_steps, value_cosine=curriculum_value_cosine)
        if resume and start:
            curriculum.load_state_dict(saved['curriculum'])
        index.restrict(curriculum.active)
        if bank_load is not None:
            bank_load.restrict(curriculum.active)

    def save_all(step_count):
        save_checkpoint(agent, optimizer, output, step_count, rng, cache,
                        fingerprint, keep=config.train.keep_checkpoints, archiver=archiver)
        if key_cache is not None:
            torch.save({'cache': key_cache.state_dict(), 'grads': record_grads.state_dict(),
                        **({'table': key_table.state_dict()} if key_table is not None else {}),
                        **({'curriculum': curriculum.state_dict()}
                           if curriculum is not None else {})},
                       output / f'record-state-{step_count:09d}.pt')
            for old in sorted(output.glob('record-state-*.pt'))[:-config.train.keep_checkpoints]:
                old.unlink()
        if bank_load is not None:
            torch.save(bank_load.state_dict(), output / f'bank-load-{step_count:09d}.pt')
            for old in sorted(output.glob('bank-load-*.pt'))[:-config.train.keep_checkpoints]:
                old.unlink()

    def force_probability(step_index):
        if gold_force is None:
            return config.train.gold_force_probability
        force_start, force_floor, force_steps = gold_force
        if not force_steps:
            return force_start
        return force_floor + (force_start - force_floor) * max(0.0, 1 - step_index / force_steps)
    source_rows = {}
    if sources_path is not None:
        with sources_path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                source_rows[row['record_id']] = row
        if set(map(str, next(iter(index.spaces.values())).ids)) - source_rows.keys():
            raise ValueError('Writer replay manifest does not cover the published bank')
    source_groups = source_ingestion_groups(list(source_rows.values()))
    routing_teacher = (RoutingCandidateIndex(
        sources_path, ramp_steps=routing_hard_ramp_steps)
                       if routing_episodes_path is not None else None)
    queries = {}
    if routing_episodes_path is not None:
        with routing_episodes_path.open(encoding='utf-8') as handle:
            for line in handle:
                episode = json.loads(line)
                episode_id = episode['episode_id']
                if episode_id in queries and queries[episode_id] != episode['query']:
                    raise ValueError('Episode query identity changed')
                queries[episode_id] = episode['query']

    @lru_cache(maxsize=16384)
    def ingestion_prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(
            document_id, parts, generation=writer_prompt_generation(bank_manifest),
            scope={'domain': domain}, mode=mode)

    @lru_cache(maxsize=65536)
    def writer_tokens(record_id):
        row = source_rows[record_id]
        document_id, parts, mode = source_groups[record_id]
        messages = ingestion_prefixes(
            document_id, parts, mode, row.get('domain', 'research'))[record_id]
        return tuple(writer_prefix_ids(agent.tokenizer, messages))

    class WriterInputs(dict):
        def __getitem__(self, record_id):
            return torch.tensor([writer_tokens(record_id)], dtype=torch.long,
                                device=agent.device)

        def keys(self):
            return source_rows.keys()

    writer_inputs = WriterInputs()
    sampler = EpisodeSampler(len(data), seed=config.train.seed)
    if not resume:
        save_all(0)
    completed, last_saved = start, 0 if not resume else None
    began = time.perf_counter()

    # The launcher acknowledges a stale stop before spawning. Never clear a new
    # request that arrives during model, index, or checkpoint initialization.
    with run_lock(output, clear_stop=False), stop_on_signal() as signal_state, \
            ExitStack() as lifecycle:
        tracker = lifecycle.enter_context(Tracking(config, output))
        environment = environment_report() | {
            "resume": resume, "attempt": tracker.attempt,
            "bank_key_index": {"kind": "resident_exact_cpu", "bytes": index.key_bytes},
            "writer_replay_ingestion_mix": {
                "holistic_complete_blob": "sha256(article-group) parity 1",
                "sequential_source_parts": "sha256(article-group) parity 0",
                "maximum_parts_per_group": 4,
            },
            "spatial": settings,
            "max_unused_cuda_gib": max_unused_cuda_gib,
            "gradient_checkpointing": config.model.gradient_checkpointing,
            "retain_writer_replay_activations": retain_writer_replay_activations,
            "cache_reclaim_host_reserve_gib": cache_reclaim_host_reserve_gib,
            "min_host_available_gib": min_host_available_gib,
            "memory": memory_report,
        }
        atomic_json(output / "environment.json", environment)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            for step in range(start, steps):
                # Shared unified memory: yield to co-tenants before they hit their
                # own rails. Release cached blocks, wait, then checkpoint and stop.
                pressure = False
                if min_host_available_bytes:
                    available = available_host_memory()
                    if available is not None and available < min_host_available_bytes:
                        reclaim_cuda_cache(config.train.device, 0, force=True)
                        waited = 0.0
                        while (waited < host_pressure_wait_seconds
                               and (available_host_memory() or 0) < min_host_available_bytes
                               and signal_state["signal"] is None
                               and not stop_requested(output)):
                            time.sleep(15)
                            waited += 15
                        pressure = (available_host_memory() or 0) < min_host_available_bytes
                        if waited:
                            print(json.dumps({"host_pressure_wait_seconds": waited,
                                              "step": step, "stopping": pressure}), flush=True)
                if pressure or signal_state["signal"] is not None or stop_requested(output):
                    if last_saved != completed:
                        optimizer.zero_grad(set_to_none=True)
                        reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                           force=True)
                        save_all(completed)
                    break
                rows = [data[curriculum.row(step, offset, batch_size) if curriculum is not None
                             else sampler.index(step * batch_size + offset)]
                        for offset in range(batch_size)]
                for row in rows:
                    row['training_step'] = step
                    row['gold_force_probability'] = force_probability(step)
                if bank_load is not None:
                    bank_load.begin_step()
                if routing_teacher is not None:
                    for row in rows:
                        row['routing_step'] = step
                        row['routing_hard_ramp_steps'] = routing_hard_ramp_steps
                        for site in row['sites']:
                            site['routing_query_text'] = queries[site['episode_id']]
                optimizer.zero_grad(set_to_none=True)
                profiling = step < start + profile_steps
                if profiling and torch.cuda.is_available():
                    torch.cuda.synchronize(config.train.device)
                tick = time.perf_counter()
                phase_tick = tick
                phase_seconds = {}

                def finish_phase(name):
                    nonlocal phase_tick
                    if profiling and torch.cuda.is_available():
                        torch.cuda.synchronize(config.train.device)
                    now = time.perf_counter()
                    if profiling:
                        phase_seconds[f'profile_{name}_seconds'] = now - phase_tick
                    phase_tick = now

                if key_table is not None:
                    key_table.begin_step()
                sink = (GradientSink(agent, key_cache, key_table)
                        if key_cache is not None else None)
                replay = (BankWriterReplay(
                              agent, writer_inputs,
                              checkpoint_backward=not retain_writer_replay_activations)
                          if config.train.writer_replay_records_per_site else None)
                with autocast_context(config):
                    if pipeline:
                        result = spatial_bank_pipeline_forward(
                            agent, training_bank, index, rows, limits=limits,
                            routing_candidates=routing_candidates,
                            microbatch_size=microbatch_size, inflight=inflight,
                            pad_token_id=agent.tokenizer.pad_token_id or 0,
                            writer_replay=replay,
                            routing_teacher=routing_teacher,
                            load=bank_load,
                            sink=sink,
                        )
                    else:
                        result = spatial_bank_forward(
                            agent, training_bank, index, rows, limits=limits,
                            routing_candidates=routing_candidates,
                            pad_token_id=agent.tokenizer.pad_token_id or 0,
                        )
                finish_phase('forward')
                result.loss.backward()
                finish_phase('consumer_backward')
                if replay is not None:
                    replay.backward()
                record_metrics = {}
                if sink is not None:
                    # Accumulate this step's per-record cotangents, then push the
                    # accumulated gradients of a close neighbourhood into the writer.
                    timer = time.perf_counter()
                    harvested = sink.harvest(record_grads, step, len(limits))
                    flush = record_grads.select(
                        step, sink.neighborhood, budget=config.train.record_flush_budget,
                        extra=config.train.record_flush_extra)
                    cotangents = [record_grads.pop(record_id, step) for record_id in flush]
                    harvest_seconds = time.perf_counter() - timer
                    with autocast_context(config):
                        if key_table is None:
                            writer_backward(agent, lambda record_id: writer_inputs[record_id],
                                            flush, cotangents)
                        else:
                            # Terminal keys: one encode per record trains key prediction,
                            # pushes payload cotangents and republishes the payloads.
                            # Values are published from the pre-update writer (age 1).
                            passes = {
                                'flushed': writer_pass(
                                    agent, lambda record_id: writer_inputs[record_id],
                                    flush, cotangents, key_table, training_bank, key_cache,
                                    step + 1, prediction_weight=config.train.key_prediction_weight,
                                    commitment_weight=config.train.key_commitment_weight,
                                    drift_sample=16, batch_size=writer_batch_size,
                                    checkpointing=writer_checkpointing),
                                'stale': writer_pass(
                                    agent, lambda record_id: writer_inputs[record_id],
                                    key_cache.stalest(config.train.record_refresh_per_step,
                                                      exclude=flush,
                                                      active=None if curriculum is None
                                                      else curriculum.active),
                                    None, key_table, training_bank, key_cache, step + 1,
                                    prediction_weight=config.train.key_prediction_weight,
                                    commitment_weight=config.train.key_commitment_weight,
                                    drift_sample=16, batch_size=writer_batch_size,
                                    checkpointing=writer_checkpointing,
                                    backward=config.train.record_refresh_backward)}
                    record_metrics = {'record_harvested': harvested,
                                      'record_neighborhood': len(set(sink.neighborhood)),
                                      'record_flushed': len(flush),
                                      'record_harvest_seconds': harvest_seconds,
                                      'record_writer_backward_seconds':
                                          time.perf_counter() - timer - harvest_seconds}
                if key_table is not None:
                    record_metrics.update({f'{name}_{key}': value for name, report
                                           in passes.items() for key, value in report.items()})
                finish_phase('writer_backward')
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in agent.parameters() if parameter.requires_grad],
                    config.train.clip_grad_norm,
                )
                optimizer.step()
                if bank_load is not None:
                    bank_load.commit()
                if key_table is not None:
                    record_metrics.update(key_table.step(step + 1))
                    timer = time.perf_counter()
                    synced = key_table.sync_index(index)
                    record_grads.evict(step + 1)
                    record_metrics.update(record_synced_rows=synced,
                                          record_sync_seconds=time.perf_counter() - timer,
                                          record_buffer=len(record_grads))
                    del sink
                    sink = None
                finish_phase('optimizer')
                maintenance_ids = (() if replay is None else training_bank.maintenance_ids(
                    maintenance_records_per_step, exclude=replay.record_ids,
                    eligible=writer_inputs.keys()))
                refreshed_views = (replay.refresh(
                    training_bank, additional_ids=maintenance_ids,
                    optimizer_step=step + 1) if replay is not None else 0)
                if sink is not None:
                    timer = time.perf_counter()
                    with autocast_context(config):
                        flushed_refresh = refresh_records(
                            agent, lambda record_id: writer_inputs[record_id], flush,
                            training_bank, key_cache, step + 1)
                        stale_refresh = refresh_records(
                            agent, lambda record_id: writer_inputs[record_id],
                            key_cache.stalest(config.train.record_refresh_per_step,
                                              exclude=flush),
                            training_bank, key_cache, step + 1)
                    refresh_seconds = time.perf_counter() - timer
                    key_cache.sync_index(agent, index)
                    record_grads.evict(step + 1)
                    record_metrics.update(
                        record_refresh_seconds=refresh_seconds,
                        record_sync_seconds=time.perf_counter() - timer - refresh_seconds,
                        record_buffer=len(record_grads),
                        **{f'flushed_{k}': v for k, v in flushed_refresh.items()},
                        **{f'stale_{k}': v for k, v in stale_refresh.items()})
                    del sink
                finish_phase('refresh')
                loss_value = float(result.loss.detach())
                nll_value = float(result.nll.detach())
                routing_value = float(result.routing.detach())
                gradient_norm_value = float(gradient_norm)
                result_metrics = result.metrics
                replayed_records = len(set(replay.record_ids)) if replay is not None else 0
                del result, replay, gradient_norm
                optimizer.zero_grad(set_to_none=True)
                cache_metrics = reclaim_cuda_cache(
                    config.train.device, max_unused_cuda_bytes,
                    min_host_available_bytes=cache_reclaim_host_reserve_bytes)
                finish_phase('cache_reclaim')
                completed = step + 1
                elapsed = time.perf_counter() - tick
                row = {
                    "step": completed, "loss": loss_value,
                    "nll": nll_value, "routing": routing_value,
                    "gradient_norm": gradient_norm_value, "step_seconds": elapsed,
                    "tokens": sum(item["tokens"] for item in rows),
                    "tokens_per_second": sum(item["tokens"] for item in rows) / elapsed,
                    "trajectory_rows": batch_size, **result_metrics,
                    "replayed_records": replayed_records,
                    "maintenance_records": len(maintenance_ids),
                    "refreshed_record_views": refreshed_views,
                    **record_metrics,
                    "gold_force_probability": force_probability(step),
                    **({"retrieval_load": bank_load.statistics()}
                       if bank_load is not None else {}),
                    "bank_active_views": training_bank.sizes()['views'],
                    "bank_journal_cursor": training_bank.cursor,
                    "payload_cache_hits": training_bank.payload_cache_hits,
                    "payload_cache_misses": training_bank.payload_cache_misses,
                    **cache_metrics, **phase_seconds,
                }
                if curriculum is not None:
                    def _mean(*names):
                        values = [row[name] for name in names if name in row]
                        return sum(values) / len(values) if values else None
                    recall = result_metrics.get('learned_positive_item_recall')
                    advanced = curriculum.observe(
                        completed,
                        prediction_cosine=_mean('flushed_key_prediction_cosine_mean',
                                                'stale_key_prediction_cosine_mean'),
                        value_cosine=_mean('flushed_value_drift_cosine_mean',
                                           'stale_value_drift_cosine_mean'),
                        recall=sum(recall) / len(recall) if recall else None)
                    if advanced:
                        index.restrict(curriculum.active)
                        if bank_load is not None:
                            bank_load.restrict(curriculum.active)
                        print(json.dumps({'event': 'curriculum_advance', 'step': completed,
                                          **curriculum.report()}), flush=True)
                    row.update(curriculum.report())
                if torch.cuda.is_available() and str(config.train.device).startswith('cuda'):
                    row.update(
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(
                            config.train.device),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(
                            config.train.device),
                    )
                if profiling or completed == 1 or completed % config.train.log_every == 0:
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    tracker.log(row)
                    print(json.dumps(row), flush=True)
                if completed % checkpoint_every == 0:
                    reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                       force=True)
                    save_all(completed)
                    last_saved = completed
            if completed == steps and last_saved != completed:
                reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                   force=True)
                save_all(completed)
                last_saved = completed
    if archiver is not None:
        archiver.close(wait=True)
    summary = {
        "completed_steps": completed, "requested_steps": steps,
        "complete": completed == steps, "elapsed_seconds": time.perf_counter() - began,
        "examples_seen": completed * batch_size, "sites_seen": completed * batch_size
                      * len(data[0]["sites"]), "resources": resource_report(),
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--limits", nargs="+", type=int, default=[16, 8, 4, 4])
    parser.add_argument("--routing-candidates", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--train-recurrent-core", action="store_true")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--inflight", type=int, default=1)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--max-unused-cuda-gib", type=float, default=4.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--retain-writer-replay-activations", action="store_true")
    parser.add_argument("--cache-reclaim-host-reserve-gib", type=float, default=16.0)
    parser.add_argument("--maintenance-records-per-step", type=int, default=0)
    parser.add_argument("--routing-episodes", type=Path)
    parser.add_argument("--routing-weight", type=float)
    parser.add_argument("--inherit-bank", action="store_true")
    parser.add_argument("--bank-journal", type=Path)
    parser.add_argument("--writer-key-learning-rate", type=float)
    parser.add_argument("--key-stability-weight", type=float)
    parser.add_argument("--routing-logit-scale", type=float)
    parser.add_argument("--routing-live-weight", type=float)
    parser.add_argument("--routing-hard-ramp-steps", type=int, default=3000)
    parser.add_argument("--min-host-available-gib", type=float, default=0.0)
    parser.add_argument("--host-pressure-wait-seconds", type=float, default=600.0)
    parser.add_argument("--gold-force", nargs=3, metavar=("START", "FLOOR", "ANNEAL_STEPS"),
                        help="forced-gold probability schedule, e.g. 1.0 0.1 3000")
    parser.add_argument("--exploration-fraction", type=float)
    parser.add_argument("--spread-variance-weight", type=float)
    parser.add_argument("--spread-covariance-weight", type=float)
    parser.add_argument("--koleo-weight", type=float)
    parser.add_argument("--load-penalty-weight", type=float)
    parser.add_argument("--load-decay", type=float)
    parser.add_argument("--load-threshold", type=float)
    parser.add_argument("--key-states", type=Path,
                        help="bank key-state cache (cache_bank_key_states.py) for record gradients")
    parser.add_argument("--device-search", action="store_true",
                        help="exact key search on the device even when training on CPU")
    parser.add_argument("--curriculum-sizes", nargs="+", type=int,
                        help="nested active bank sizes ending with 0 (full bank), "
                             "e.g. 8000 16000 32000 64000 128000 0")
    parser.add_argument("--curriculum-min-steps", type=int, default=300)
    parser.add_argument("--curriculum-max-steps", type=int, default=3000)
    parser.add_argument("--curriculum-value-cosine", type=float, default=0.99,
                        help="value-drift EMA a stage needs before it advances")
    parser.add_argument("--record-refresh-per-step", type=int)
    parser.add_argument("--record-refresh-backward", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--record-flush-budget", type=int)
    parser.add_argument("--record-flush-extra", type=int)
    parser.add_argument("--writer-batch-size", type=int, default=16)
    parser.add_argument("--writer-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="recompute writer activations in the record-gradient backward")
    parser.add_argument("--archive-dir", type=Path,
                        help="copy completed checkpoints here in the background")
    parser.add_argument("--payload-cache-gib", type=float, default=0.0,
                        help="host-memory cache for bank payloads (write-through LRU)")
    args = parser.parse_args()
    print(json.dumps(train(
        args.config, args.data, args.bank, args.output, args.init_from,
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        loops=args.loops, limits=tuple(args.limits),
        routing_candidates=args.routing_candidates,
        checkpoint_every=args.checkpoint_every,
        train_recurrent_core=args.train_recurrent_core,
        microbatch_size=args.microbatch_size, inflight=args.inflight,
        sources_path=args.sources,
        max_unused_cuda_gib=args.max_unused_cuda_gib,
        gradient_checkpointing=args.gradient_checkpointing,
        profile_steps=args.profile_steps,
        retain_writer_replay_activations=args.retain_writer_replay_activations,
        cache_reclaim_host_reserve_gib=args.cache_reclaim_host_reserve_gib,
        maintenance_records_per_step=args.maintenance_records_per_step,
        routing_episodes_path=args.routing_episodes,
        routing_weight=args.routing_weight,
        inherit_bank=args.inherit_bank,
        bank_journal_path=args.bank_journal,
        writer_key_learning_rate=args.writer_key_learning_rate,
        key_stability_weight=args.key_stability_weight,
        routing_logit_scale=args.routing_logit_scale,
        routing_live_weight=args.routing_live_weight,
        routing_hard_ramp_steps=args.routing_hard_ramp_steps,
        min_host_available_gib=args.min_host_available_gib,
        host_pressure_wait_seconds=args.host_pressure_wait_seconds,
        gold_force=(None if args.gold_force is None else
                    (float(args.gold_force[0]), float(args.gold_force[1]),
                     int(args.gold_force[2]))),
        spreading={name: getattr(args, name) for name in (
            'exploration_fraction', 'spread_variance_weight', 'spread_covariance_weight',
            'koleo_weight', 'load_penalty_weight', 'load_decay', 'load_threshold')},
        key_states_path=args.key_states,
        device_search=args.device_search,
        curriculum_sizes=tuple(args.curriculum_sizes or ()),
        curriculum_min_steps=args.curriculum_min_steps,
        curriculum_max_steps=args.curriculum_max_steps,
        curriculum_value_cosine=args.curriculum_value_cosine,
        payload_cache_gib=args.payload_cache_gib,
        archive_dir=args.archive_dir,
        writer_batch_size=args.writer_batch_size,
        writer_checkpointing=args.writer_checkpointing,
        record_budget={name: getattr(args, name) for name in (
            'record_refresh_per_step', 'record_refresh_backward', 'record_flush_budget',
            'record_flush_extra')},
    ), indent=2))
