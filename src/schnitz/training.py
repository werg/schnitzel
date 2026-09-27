"""Executable support/query training and stored-only evaluation reference."""
from __future__ import annotations

from contextlib import nullcontext, ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import hashlib
import math
import os
import platform
import random
import subprocess
import time

import torch
from safetensors.torch import load_model

from .agent import SchnitzelAgent, answer_distribution_kl, answer_state_alignment
from .checkpoints import save_checkpoint, restore_checkpoint, resolve_checkpoint, stop_on_signal
from .config import Config
from .data import Episode, Source, counterfactual, make_episode, save_episodes, load_episodes, evidence_ids
from .replay import ReplayTape
from .store import DiskStore, ReadPlan, Selection, StoredRecord, lookup_record


class EpisodeSampler:
    """Deterministic shuffled passes indexed by completed update and microbatch.

    The index position is derived from the checkpointed step/partial-microbatch
    cursor. It uses a separate RNG so live/cache, depth and noise draws retain
    their own exact-resume state. Only one epoch permutation is resident.
    """

    def __init__(self, count: int, *, seed: int):
        if count < 1:
            raise ValueError('Episode sampler needs a nonempty dataset')
        self.count, self.seed = count, seed
        self._epoch, self._order = None, None

    def index(self, position: int) -> int:
        if position < 0:
            raise ValueError('Episode position must be nonnegative')
        epoch, offset = divmod(position, self.count)
        if epoch != self._epoch:
            self._order = list(range(self.count))
            random.Random(f'sdkb-pass:{self.seed}:{epoch}').shuffle(self._order)
            self._epoch = epoch
        return self._order[offset]


def autocast_context(config: Config):
    if config.train.precision == "bf16":
        return torch.autocast(config.train.device, dtype=torch.bfloat16)
    return nullcontext()


def environment_report() -> dict:
    report = {"python": platform.python_version(), "machine": platform.machine(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "pytorch_allocator_conf": (os.environ.get("PYTORCH_ALLOC_CONF")
                                           or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(0),
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      bf16_supported=torch.cuda.is_bf16_supported(),
                      cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_allocator_backend=torch.cuda.memory.get_allocator_backend())
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"] = None
    try:
        report["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report["git_dirty"] = None
    try:
        import transformers
        report["transformers"] = transformers.__version__
    except ImportError:
        report["transformers"] = None
    return report


def resource_report() -> dict:
    # ru_maxrss is KiB on Linux (the supported Spark platform).
    try:
        import resource
        scale = 1 if platform.system() == 'Darwin' else 1024
        report = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale}
    except ImportError:
        report = {"process_peak_rss_bytes": None}
    report['rss_scope'] = 'process lifetime; not phase-isolated'
    if torch.cuda.is_available():
        report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if Path("/proc/meminfo").exists():
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        report["system_mem_available_bytes"] = int(info["MemAvailable"].split()[0]) * 1024
    return report


def reset_resource_peaks(device: str | None = None):
    """Phase-local CUDA peaks; host ru_maxrss cannot be reset portably."""
    if torch.cuda.is_available() and (device is None or device == "cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def stored_channel(agent: SchnitzelAgent, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    # Cast is part of the captured forward. Cached and live payloads pass through
    # the same representational precision, then return to the resident compute dtype.
    return tuple(x.float() if i % 2 == 0 else x.to(dtype).float() for i, x in enumerate(outputs))


def payload_contrast_loss(base_loss: torch.Tensor, correct_nll: torch.Tensor,
                          swapped_nll: torch.Tensor, *,
                          weight: float, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor correct likelihood and require a different source to score worse."""
    if weight < 0 or margin < 0:
        raise ValueError('Source-swap weight and margin must be nonnegative')
    contrast = torch.nn.functional.softplus(margin + correct_nll - swapped_nll)
    return base_loss + weight * contrast, contrast


def persist_outputs(store: DiskStore, agent: SchnitzelAgent, source: Source,
                    outputs: tuple[torch.Tensor, ...], namespace: str, generation: str) -> None:
    for record in output_records(agent, source, outputs, namespace, generation):
        store.put(record)


def output_records(agent: SchnitzelAgent, source: Source, outputs: tuple[torch.Tensor, ...],
                   namespace: str, generation: str):
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    for space in range(len(agent.config.memory.payload_dims)):
        yield StoredRecord(source.record_id, outputs[2 * space][0],
            outputs[2 * space + 1][0].to(dtype), namespace=namespace, space=f"s{space}",
            generation=generation, created_at=source.created_at, source_id=source.record_id)


def read_cached(store: DiskStore, agent: SchnitzelAgent, source: Source,
                namespace: str, generation: str) -> tuple[torch.Tensor, ...]:
    outputs = []
    for space in range(len(agent.config.memory.payload_dims)):
        record = lookup_record(store, source.record_id, namespace=namespace,
                               space=f"s{space}", generation=generation)
        outputs.extend((record.key[None].to(agent.device), record.payload[None].float().to(agent.device)))
    return tuple(outputs)


def train(config: Config, output: str | Path, *, resume: bool = False,
          stop_after: int | None = None, init_from: str | Path | None = None,
          stop_output: str | Path | None = None) -> dict:
    from .operations import run_lock
    if (Path(output) / 'manifest.json').is_file():
        raise ValueError('Immutable checkpoint cannot be a training output; resume its run directory '
                         'or restore the archive into a new run directory')
    with run_lock(output), stop_on_signal() as stop, ExitStack() as lifecycle:
        return _train(config, output, resume=resume, stop_after=stop_after, init_from=init_from,
                      stop_output=stop_output, stop=stop, lifecycle=lifecycle)


def _train(config, output, *, resume, stop_after, init_from, stop_output, stop, lifecycle):
    from .operations import apply_checkpoint_policy, CHECKPOINT_POLICY_FIELDS
    config = deepcopy(config)
    apply_checkpoint_policy(config, output, stop_output)
    config.validate()
    if resume and init_from is not None:
        raise ValueError("Choose resume or warm-start, not both")
    if config.train.reinitialize_reader and not resume and init_from is None:
        raise ValueError('Reader reinitialization requires a warm-start source')
    if config.train.warmstart_memory_gate is not None and not resume and init_from is None:
        raise ValueError('Memory gate override requires an explicit warm-start source')
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    from .episode_index import EpisodeIndex
    from .routing import validate_routing_dataset
    episodes = (EpisodeIndex(config.train.episodes_file) if config.train.episodes_file else
                [make_episode(i, split=f"train-{config.train.seed}", distractors=config.train.distractors)
                 for i in range(config.train.train_worlds)])
    token_index = None
    if config.train.tokenized_episodes_file:
        if not isinstance(episodes, EpisodeIndex):
            raise ValueError('Pretokenized training requires an immutable episode file')
        from .token_index import TokenIndex
        token_index = TokenIndex(config.train.tokenized_episodes_file,
            episode_sha256=episodes.sha256, model_id=config.model.model_id,
            revision=config.model.revision, arm=config.train.arm)
        if len(token_index) != len(episodes):
            raise ValueError('Tokenized episode row count differs from episode data')
    validate_routing_dataset(config, episodes)
    if config.train.bank_dir is not None and any(
            len(episode.required_ids) > min(config.train.bank_read_limits) for episode in episodes):
        raise ValueError('Bank read limits must fit every verified sufficient support set')
    if config.train.payload_contrast_weight and any(
            episode.support_annotation != 'verified' or len(episode.required_ids) != 1
            or len(episode.supports) < 2 or
            not any(source.record_id not in episode.required_ids for source in episode.supports)
            for episode in episodes):
        raise ValueError('Source-swap contrast requires one verified positive and an eligible distractor')
    if config.train.optimization_scope == 'compactor' and config.train.retrieval == 'oracle' and not any(
            len(evidence_ids(e, config.train.evidence_scope)) > config.memory.compact_records for e in episodes):
        raise ValueError('Compactor-only training cannot reduce any selected group; lower compact_records')
    output = Path(output)
    if config.train.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use configs/tiny_cpu.yaml for offline tests.")
    execution_upgrade = False
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        resolve_checkpoint(output)
        old_config = asdict(config_from_run(output))
        current = asdict(config)
        old_effective_batch = (old_config['train']['gradient_accumulation'] *
                               old_config['train'].get('batch_size', 1))
        current_effective_batch = (current['train']['gradient_accumulation'] *
                                   current['train']['batch_size'])
        if old_effective_batch != current_effective_batch:
            raise ValueError('Resume execution upgrade must preserve examples per optimizer update')
        execution_upgrade = any((
            old_config['train']['gradient_accumulation'] != current['train']['gradient_accumulation'],
            old_config['train'].get('batch_size', 1) != current['train']['batch_size'],
            old_config['model']['gradient_checkpointing'] != current['model']['gradient_checkpointing'],
            old_config['memory']['checkpoint_chunks'] != current['memory']['checkpoint_chunks'],
        ))
        old_config['train']['steps'] = current['train']['steps']
        for name in CHECKPOINT_POLICY_FIELDS:
            old_config['train'][name] = current['train'][name]
        # These alter execution/storage of derived IDs, not learned parameters,
        # optimizer ownership, corpus identity, or examples per update.
        old_config['model']['gradient_checkpointing'] = current['model']['gradient_checkpointing']
        old_config['memory']['checkpoint_chunks'] = current['memory']['checkpoint_chunks']
        for name in ('gradient_accumulation', 'batch_size', 'tokenized_episodes_file'):
            old_config['train'][name] = current['train'][name]
        if old_config != current:
            raise ValueError('Resume training hyperparameters differ; only steps/checkpoint policy may change')
    from .archiving import ensure_free
    from .operations import stop_requested
    from .tracking import Tracking, run_identity
    ensure_free(output, reserve_bytes=config.train.min_free_disk_bytes)
    if config.train.archive_dir and not Path(config.train.archive_dir).is_dir():
        raise FileNotFoundError('Archive root must be an existing directory/mounted external disk')
    if resume and not (output / 'run-identity.json').exists():
        identity_file = resolve_checkpoint(output, verify=True) / 'run-identity.json'
        if identity_file.exists():
            (output / 'run-identity.json').write_bytes(identity_file.read_bytes())
    tracker = lifecycle.enter_context(Tracking(config, output))
    resource_stop = None
    def want_stop():
        return resource_stop is not None or stop['signal'] is not None or stop_requested(output) or (stop_output is not None and stop_requested(stop_output))
    torch.set_num_threads(config.train.threads)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    from .runtime import configure_memory, compute_watchdog, available_host_memory, memory_metrics
    runtime_limits = configure_memory(config.train)
    reset_resource_peaks(config.train.device)
    rng = random.Random(config.train.seed)
    agent = SchnitzelAgent(config).to(config.train.device)
    agent.train()
    if init_from is not None:
        source_checkpoint = resolve_checkpoint(init_from, verify=True)
        old = config_from_run(Path(init_from))
        structural = ("key_dim", "write_slots", "read_slots", "payload_dims")
        if any(getattr(old.memory, name) != getattr(config.memory, name) for name in structural):
            raise ValueError("Warm-start changes the stored interface")
        reader_shape = ('reader_width', 'reader_rounds', 'reader')
        if not config.train.reinitialize_reader and any(
                getattr(old.memory, name) != getattr(config.memory, name) for name in reader_shape):
            raise ValueError('Warm-start changes the reader architecture without explicit reinitialization')
        if config.train.reinitialize_reader and (old.memory.compaction != 'none' or config.memory.compaction != 'none'):
            raise ValueError('Reader reinitialization requires raw records without a coupled compactor')
        if old.model.backend != config.model.backend or old.model.model_id != config.model.model_id:
            raise ValueError("Warm-start changes the student backbone")
        old_mode = old.model.recurrence_mode
        converting = old_mode != config.model.recurrence_mode
        if converting and not (config.train.allow_recurrence_conversion and old_mode == "full_stack"
                               and config.model.recurrence_mode == "middle_block"):
            raise ValueError("Changing recurrence requires explicit full-stack-to-middle conversion")
        if not converting and old_mode == "middle_block" and (
                old.model.recurrent_start, old.model.recurrent_end) != (
                config.model.recurrent_start, config.model.recurrent_end):
            raise ValueError("Warm-start changes the recurrent layer partition")
        manifest_path = source_checkpoint / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["resolved_model_revision"] != agent.resolved_revision:
            raise ValueError("Warm-start base revision differs")
        reinitialized = []
        reader = agent.reader
        if config.train.reinitialize_reader:
            reinitialized = ['reader.' + name for name in reader.state_dict()]
            # Preserve the module insertion order for exact optimizer ownership
            # on resume. The source reader is deliberately excluded from loading.
            agent.reader = torch.nn.Identity()
        try:
            missing, unexpected = load_model(agent, str(source_checkpoint / "model.safetensors"),
                                            strict=False, device=config.train.device)
        finally:
            agent.reader = reader
        allowed_missing = ("compactor.",)
        allowed_unexpected = ("compactor.",)
        if config.train.reinitialize_reader:
            allowed_unexpected += ('reader.',)
        routing_conversion = config.memory.independent_routing_query and not old.memory.independent_routing_query
        if routing_conversion:
            allowed_missing += ('routing_query_head.',)
        if converting:
            allowed_missing += ("backbone.bridge.", "loop_workspace", "loop_query_norm.")
            allowed_unexpected += ("backbone.loop_gate", "backbone.feedback_norm.")
        if (any(not name.startswith(allowed_missing) for name in missing)
                or any(not name.startswith(allowed_unexpected) for name in unexpected)):
            raise ValueError(f"Incompatible warm-start state: missing={missing}, unexpected={unexpected}")
        if routing_conversion:
            agent.routing_query_head.load_state_dict(agent.query_head.state_dict())
        prior_memory_gate = None
        if config.train.warmstart_memory_gate is not None:
            prior_memory_gate = float(agent.backbone.bridge.memory_logit.detach().sigmoid())
            value = config.train.warmstart_memory_gate
            with torch.no_grad():
                agent.backbone.bridge.memory_logit.fill_(math.log(value / (1 - value)))
        provenance = {"routing_query_conversion": routing_conversion, "checkpoint": str(source_checkpoint), "optimizer_reset": True,
                      "reader_reinitialized": config.train.reinitialize_reader,
                      "reinitialized_parameters": sorted(reinitialized),
                      "missing_initialized": sorted(missing), "unused": sorted(unexpected),
                      "recurrence_conversion": converting,
                      "warmstart_memory_gate": config.train.warmstart_memory_gate,
                      "prior_memory_gate": prior_memory_gate}
        (output / "initialization.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if config.train.optimization_scope == "compactor":
        for name, parameter in agent.named_parameters():
            parameter.requires_grad_(name.startswith("compactor."))
        agent.eval()
        agent.compactor.train()
    if config.train.optimization_scope == 'routing':
        for name, parameter in agent.named_parameters():
            parameter.requires_grad_(name.startswith(('key_head.', 'address_maps.', 'query_maps.', 'routing_query_head.',
                                                              'writer_key_heads.', 'query_key_heads.')))
        # Keep frozen feature/payload/reader paths deterministic. The top-level
        # training flag still honors the configured routing warmup.
        for module in agent.children():
            module.eval()
    bank = None
    bank_index = None
    bank_manifest = None
    bank_manifest_sha = None
    if config.train.bank_dir is not None:
        from .offline_bank import (canonical_json, publish_offline_generation,
                                   stored_memory_identity)
        from .trajectories import file_sha256
        directory = Path(config.train.bank_dir)
        bank_manifest_path = directory / 'manifest.json'
        if not bank_manifest_path.is_file():
            raise FileNotFoundError('A fully published bank manifest is required')
        bank_manifest = json.loads(bank_manifest_path.read_text())
        bank_manifest_sha = file_sha256(bank_manifest_path)
        bank = DiskStore(directory / 'bank.sqlite')
        verified = publish_offline_generation(bank, identity=bank_manifest['identity'],
                    namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
                    spaces=tuple(bank_manifest['spaces']), shard_ids=tuple(bank_manifest['shards']),
                    source_count=bank_manifest['sources'], verify_only=True)
        if canonical_json(verified) != canonical_json({key: bank_manifest[key] for key in verified}):
            raise ValueError('Published bank manifest differs from verified stored records')
        if tuple(bank_manifest['spaces']) != tuple(f's{i}' for i in range(len(config.memory.payload_dims))):
            raise ValueError('Bank spaces are incompatible with this model')
        if (bank_manifest['identity']['model'] != asdict(config.model)
                or stored_memory_identity(bank_manifest['identity']['memory'])
                != stored_memory_identity(asdict(config.memory))):
            raise ValueError('Bank writer architecture or storage transform changed')
        if (bank_manifest['identity'].get('compute_precision') != config.train.precision or
                bank_manifest['identity'].get('max_source_tokens') != config.train.max_source_tokens):
            raise ValueError('Bank writer precision or source token contract changed')
        if not resume:
            if init_from is None or file_sha256(source_checkpoint / 'model.safetensors') != bank_manifest['identity']['writer_checkpoint_sha256']:
                raise ValueError('Bank training must initialize from its exact frozen writer snapshot')
        from .key_index import PublishedKeyIndex
        bank_index = PublishedKeyIndex(bank, namespace=bank_manifest['namespace'],
                                      generation=bank_manifest['generation'],
                                      spaces=tuple(bank_manifest['spaces']),
                                      expected_sources=bank_manifest['sources'])
        for name, parameter in agent.named_parameters():
            if name.startswith(('write_slots', 'key_head.', 'value_head.', 'address_maps.',
                                 'writer_key_heads.', 'codecs.')):
                parameter.requires_grad_(False)
    from .optimizers import make_optimizer, optimizer_report
    optimizer = make_optimizer(agent)
    start = 0
    progress = {}
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    manifest = environment_report() | {"resolved_model_revision": agent.resolved_revision,
        'runtime_limits': runtime_limits,
        "total_parameters": sum(p.numel() for p in agent.parameters()),
        "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
        "notice": "Prototype measurements; not evidence of capacity substitution."}
    if bank_index is not None:
        manifest['bank_key_index'] = {'kind': 'resident_exact_cpu',
                                      'key_bytes': bank_index.key_bytes,
                                      'query_scopes': 'namespace,space,generation,domain,created_at,deleted'}
    if hasattr(agent.backbone, "manifest"):
        manifest["recurrence"] = agent.backbone.manifest()
        manifest["recurrence"]["training_depths"] = config.train.loop_counts or [config.model.loops]
        manifest["recurrence"]["writer_loops"] = config.model.writer_loops
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    previous_environment = output / 'environment.json'
    if previous_environment.exists():
        previous = previous_environment.read_bytes()
        (attempts / ('previous-' + hashlib.sha256(previous).hexdigest()[:16] + '.json')).write_bytes(previous)
    manifest['resume'] = resume
    manifest['attempt'] = tracker.attempt
    (attempts / (tracker.attempt + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    previous_environment.write_text(json.dumps(manifest, indent=2) + "\n")
    fingerprint = (episodes.sha256 if isinstance(episodes, EpisodeIndex) else
                   hashlib.sha256(json.dumps([asdict(e) for e in episodes], sort_keys=True).encode()).hexdigest())
    if bank_manifest_sha is not None:
        fingerprint = hashlib.sha256((fingerprint + ':' + bank_manifest_sha).encode()).hexdigest()
    data_manifest = output / "data_manifest.json"
    if resume and data_manifest.exists() and json.loads(data_manifest.read_text())["sha256"] != fingerprint:
        raise ValueError("Episode contents changed since checkpoint; refusing stale-cache reuse")
    data_manifest.write_text(json.dumps({"sha256": fingerprint, "episodes": len(episodes)}, indent=2) + "\n")
    save_episodes(output / "train.jsonl", episodes)
    if resume:
        start = restore_checkpoint(agent, optimizer, output, rng, fingerprint, progress=progress,
                                   discard_accumulation=execution_upgrade)
        if config.train.steps < start:
            raise ValueError("Requested total steps precede the saved checkpoint")
    resolved_optimizer = dict(kind=config.train.optimizer, groups=optimizer_report(optimizer),
                              resumed=resume, completed_steps=start)
    (output / 'optimizer.json').write_text(json.dumps(resolved_optimizer, indent=2) + '\n')
    print(json.dumps({'optimizer': resolved_optimizer}), flush=True)
    cache = DiskStore(output / "training_cache.sqlite")
    archiver = None
    if config.train.archive_dir:
        from .archiving import CheckpointArchiver
        archive_run = Path(config.train.archive_dir) / run_identity(output)
        archive_run.mkdir(exist_ok=True)
        from .operations import atomic_json
        archiver = CheckpointArchiver(archive_run, keep=config.train.archive_keep_checkpoints,
                                     reserve_bytes=config.train.min_free_disk_bytes)
        lifecycle.callback(archiver.close)
        atomic_json(archive_run / 'run.json', {'id': run_identity(output), 'source_run': str(output.resolve())})
        if resume:
            archiver.submit(resolve_checkpoint(output, verify=True))
    last_saved_step = None
    if not resume:
        save_checkpoint(agent, optimizer, output, 0, rng, cache, fingerprint,
                        keep=config.train.keep_checkpoints, archiver=archiver)
        last_saved_step = 0
    completed = start
    generation = "mixed-training-v0"  # intentional stale/live training distribution, never evaluation
    sampler = (EpisodeSampler(len(episodes), seed=config.train.seed)
               if config.train.sampling_policy == 'shuffled_passes' else None)
    history = []
    start_time = time.perf_counter()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, config.train.steps):
            if config.train.min_system_available_bytes:
                available = available_host_memory()
                if available is not None and available < config.train.min_system_available_bytes:
                    resource_stop = {'reason': 'host_memory_reserve', 'available_bytes': available}
            if want_stop():
                # A stop arriving during a save needs no second identical write.
                # Only reuse saves from this invocation: a stopped resume may
                # need to commit an extended budget or other allowed config change.
                if last_saved_step != completed or progress:
                    save_checkpoint(agent, optimizer, output, completed, rng, cache, fingerprint,
                                    keep=config.train.keep_checkpoints, archiver=archiver,
                                    accumulation=progress or None)
                break
            # One depth per optimizer update, not per microbatch or replay callback.
            # This RNG is part of the atomic checkpoint. Producer depth stays fixed.
            with ExitStack() as compute:
                compute.enter_context(compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device))
                if progress:
                    agent.backbone.loops = progress['loops']
                    totals, anchor_total = progress['totals'], progress['anchor_total']
                    bank_totals = progress.get('bank_totals')
                    alignment_total = (progress['alignment_total'] if config.train.oracle_alignment_weight else 0.)
                    distillation_total = (progress['distillation_total']
                                          if config.train.oracle_distillation_weight else 0.)
                    contrast_total = progress.get('contrast_total', 0.)
                    wrong_nll_total = progress.get('wrong_nll_total', 0.)
                    micro_start = progress['microbatches']
                    if not 0 < micro_start < config.train.gradient_accumulation:
                        raise ValueError('Invalid saved accumulation position')
                    progress = {}
                else:
                    agent.backbone.loops = (rng.choice(config.train.loop_counts) if config.train.loop_counts
                                            else config.model.loops)
                    optimizer.zero_grad(set_to_none=True)
                    totals = {"loss": 0.0, "nll": 0.0, "routing_loss": 0.0, "compaction_loss": 0.0,
                              "raw_nll": 0.0, "compact_nll": 0.0, "behavior_kl": 0.0, "read_count": 0.0, "parent_kl": 0.0}
                    bank_totals = ({'selected_counts': [0.] * len(config.memory.payload_dims),
                                    'learned_positive_recall': [0.] * len(config.memory.payload_dims),
                                    'selected_payload_bytes': 0.,
                                    'swapped_payload_bytes': 0.} if bank is not None else None)
                    anchor_total, micro_start = 0.0, 0
                    alignment_total = 0.0
                    distillation_total = 0.0
                    contrast_total = 0.0
                    wrong_nll_total = 0.0
                for micro in range(micro_start, config.train.gradient_accumulation):
                    position = ((step * config.train.gradient_accumulation + micro) *
                                config.train.batch_size)
                    indices = ([sampler.index(position + row) for row in range(config.train.batch_size)]
                               if sampler is not None else
                               [rng.randrange(len(episodes)) for _ in range(config.train.batch_size)])
                    episode_batch = [episodes[index] for index in indices]
                    episode = episode_batch[0]
                    tape = ReplayTape(verify_outputs=config.train.verify_replay)
                    visible = evidence_ids(episode, config.train.evidence_scope)
                    read_indices = [i for i, source in enumerate(episode.supports) if source.record_id in visible]
                    with autocast_context(config):
                        if config.train.batch_size > 1 and bank is None:
                            if (config.train.arm != 'memory' or config.train.live_fraction != 1.0
                                    or config.memory.compaction != 'none'
                                    or config.train.payload_contrast_weight
                                    or config.train.oracle_anchor_weight
                                    or config.train.oracle_alignment_weight
                                    or config.train.oracle_distillation_weight):
                                raise ValueError('Batched live training currently requires the raw all-live memory objective')
                            visible_batch = [evidence_ids(item, config.train.evidence_scope)
                                             for item in episode_batch]
                            required_batch = [[i for i, source in enumerate(item.supports)
                                               if source.record_id in visible_ids]
                                              for item, visible_ids in zip(episode_batch, visible_batch, strict=True)]
                            token_rows = ([token_index[index] for index in indices]
                                          if token_index is not None else [None] * len(indices))
                            source_ids, payload_flags, counts = [], [], []
                            for item, ids_row, needed in zip(episode_batch, token_rows,
                                                             required_batch, strict=True):
                                if ids_row is not None and (ids_row['episode_id'] != item.episode_id or
                                        ids_row['source_record_ids'] != [s.record_id for s in item.supports]):
                                    raise ValueError('Pretokenized row identity/order differs from episode data')
                                counts.append(len(item.supports))
                                oracle = (config.train.retrieval == 'oracle' or
                                          step < config.train.routing_warmup)
                                for source_index, source in enumerate(item.supports):
                                    rng.random()  # preserve one live/cache draw per declared source
                                    ids = (torch.tensor([ids_row['source_ids'][source_index]],
                                                        dtype=torch.long, device=agent.device)
                                           if ids_row is not None else agent.text_ids(source.text, source=True))
                                    source_ids.append(ids)
                                    payload_flags.append(not oracle or source_index in needed)
                            flags = torch.tensor(payload_flags, dtype=torch.bool, device=agent.device)
                            def producer():
                                return stored_channel(agent, agent.produce_batch(source_ids,
                                                                                payload_rows=flags))
                            packed = tape.capture(agent, producer) if config.train.replay else producer()
                            records_batch, offset = [], 0
                            for count in counts:
                                records_batch.append([tuple(value[row:row + 1] for value in packed)
                                                      for row in range(offset, offset + count)])
                                offset += count
                            prompts = [(torch.tensor([row['prompt_ids']], dtype=torch.long,
                                                     device=agent.device) if row is not None else
                                        agent.prompt_ids(item.query))
                                       for item, row in zip(episode_batch, token_rows, strict=True)]
                            targets = [(torch.tensor([row['target_ids']], dtype=torch.long,
                                                     device=agent.device) if row is not None else
                                        agent.target_ids(item.answer))
                                       for item, row in zip(episode_batch, token_rows, strict=True)]
                            result = agent.forward_loop_memory_batch(prompts, targets, records_batch,
                                                                     required_batch, step=step)
                            support_text = ''
                        elif config.train.batch_size > 1:
                            from .corpus_training import stored_corpus_forward_batch
                            result, bank_info = stored_corpus_forward_batch(
                                agent, bank, episode_batch, generation=bank_manifest['generation'],
                                limits=tuple(config.train.bank_read_limits),
                                namespace=bank_manifest['namespace'], searcher=bank_index,
                                token_rows=([token_index[index] for index in indices]
                                            if token_index is not None else None))
                            support_text = ''
                        elif bank is not None:
                            from .corpus_training import stored_corpus_forward
                            result, bank_info = stored_corpus_forward(
                                agent, bank, episode, generation=bank_manifest['generation'],
                                limits=tuple(config.train.bank_read_limits),
                                namespace=bank_manifest['namespace'], searcher=bank_index)
                            support_text = ''
                        else:
                            records = []
                            if config.train.arm in {"memory", "direct_latent"}:
                                for source in episode.supports:
                                    if config.train.selected_producers_only and source.record_id not in visible:
                                        # Preserve the Python sampler schedule, including
                                        # the live/cache draw for every declared source.
                                        # The opt-in policy has no cached-value history.
                                        rng.random()
                                        continue
                                    ids = agent.text_ids(source.text, source=True)
                                    # Repeated source IDs use genuinely stored old payloads; first encounter populates the cache.
                                    if config.train.live_fraction < 1:
                                        try:
                                            cached = read_cached(cache, agent, source, "train", generation)
                                        except KeyError:
                                            with torch.no_grad():
                                                cached = stored_channel(agent, agent.produce(ids))
                                            persist_outputs(cache, agent, source, cached, "train", generation)
                                    if rng.random() < config.train.live_fraction:
                                        def producer(ids=ids):
                                            return stored_channel(agent, agent.produce(ids))
                                        value = tape.capture(agent, producer) if config.train.replay else producer()
                                    else:
                                        value = tuple(x.detach() for x in cached)
                                    records.append(value)
                                if config.train.selected_producers_only:
                                    read_indices = list(range(len(records)))
                            support_text = "\n".join(s.text for s in episode.supports if s.record_id in visible)
                            prompt = agent.prompt_ids(episode.query, support_text if config.train.arm == "oracle_text" else "")
                            compact = (config.train.arm == "memory" and config.memory.compaction != "none"
                                       and step >= config.memory.compaction_warmup
                                       and rng.random() < config.memory.compaction_probability)
                            result = agent(prompt, agent.target_ids(episode.answer), records, read_indices,
                                           step=step, compact=compact)
                        loss = result.loss
                        if bank is not None and config.train.bank_payload_contrast_weight:
                            contrast = bank_info['contrast_loss']
                            loss = loss + config.train.bank_payload_contrast_weight * contrast
                            contrast_total += float(contrast.detach()) / config.train.gradient_accumulation
                            wrong_nll_total += (float(bank_info['swapped_source_nll'].detach()) /
                                                config.train.gradient_accumulation)
                        if config.train.payload_contrast_weight:
                            wrong_index = next(i for i, source in enumerate(episode.supports)
                                               if source.record_id not in episode.required_ids)
                            swapped = agent(prompt, agent.target_ids(episode.answer), records,
                                            [wrong_index], step=step)
                            loss, contrast = payload_contrast_loss(
                                loss, result.nll, swapped.nll,
                                weight=config.train.payload_contrast_weight,
                                margin=config.train.payload_contrast_margin)
                            contrast_total += float(contrast.detach()) / config.train.gradient_accumulation
                            wrong_nll_total += float(swapped.nll.detach()) / config.train.gradient_accumulation
                        if config.train.oracle_anchor_weight and config.train.arm == "memory":
                            oracle_prompt = agent.prompt_ids(episode.query, support_text)
                            if config.train.oracle_alignment_weight:
                                if any(set(indices) != set(read_indices) for indices in result.selected):
                                    raise ValueError('Oracle alignment requires the same evidence in completed latent reads and text')
                                target = agent.target_ids(episode.answer)
                                teacher = agent.conditioned_states(oracle_prompt, target, None,
                                                                   loops=config.train.oracle_anchor_loops)
                                anchor_logits = agent.backbone.logits(teacher).float()
                                anchor = torch.nn.functional.cross_entropy(
                                    anchor_logits.reshape(-1, anchor_logits.shape[-1]), target.reshape(-1))
                                alignment = answer_state_alignment(result.answer_states, teacher)
                                loss = loss + config.train.oracle_alignment_weight * alignment
                                alignment_total += float(alignment.detach()) / config.train.gradient_accumulation
                            else:
                                anchor = agent.conditioned_nll(oracle_prompt, agent.target_ids(episode.answer), None,
                                                               loops=config.train.oracle_anchor_loops)
                            loss = loss + config.train.oracle_anchor_weight * anchor
                            anchor_total += float(anchor.detach()) / config.train.gradient_accumulation
                        if config.train.oracle_distillation_weight:
                            if any(set(indices) != set(read_indices) for indices in result.selected):
                                raise ValueError('Oracle distillation requires the same evidence in completed latent reads and text')
                            target = agent.target_ids(episode.answer)
                            oracle_prompt = agent.prompt_ids(episode.query, support_text)
                            with torch.no_grad():
                                teacher_logits = agent.conditioned_logits(oracle_prompt, target, None, loops=1)
                            student_logits = agent.backbone.logits(result.answer_states).float()
                            distillation = answer_distribution_kl(student_logits, teacher_logits)
                            loss = loss + config.train.oracle_distillation_weight * distillation
                            distillation_total += float(distillation.detach()) / config.train.gradient_accumulation
                        loss = loss / config.train.gradient_accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    if loss.requires_grad:
                        loss.backward()
                    # Full source gradient accumulation happens BEFORE any optimizer update.
                    if config.train.replay:
                        tape.backward()
                    for name in totals:
                        value = getattr(result, name)
                        totals[name] += (float(value.detach()) if isinstance(value, torch.Tensor) else
                                         float(value or 0)) / config.train.gradient_accumulation
                    if bank_totals is not None:
                        for name in ('selected_counts', 'learned_positive_recall'):
                            for space, value in enumerate(bank_info[name]):
                                bank_totals[name][space] += value / config.train.gradient_accumulation
                        bank_totals['selected_payload_bytes'] += (
                            bank_info['selected_payload_bytes'] / config.train.gradient_accumulation)
                        bank_totals['swapped_payload_bytes'] = (
                            bank_totals.get('swapped_payload_bytes', 0.) +
                            bank_info.get('swapped_payload_bytes', 0) /
                            config.train.gradient_accumulation)
                    if want_stop() and micro + 1 < config.train.gradient_accumulation:
                        # Replay has finished for this microbatch. Preserve its complete
                        # cotangents, RNG, sampled depth and cache instead of discarding
                        # partial work or stepping an under-accumulated optimizer.
                        progress = dict(microbatches=micro + 1, loops=agent.backbone.loops,
                                        totals=totals, anchor_total=anchor_total,
                                        alignment_total=alignment_total, distillation_total=distillation_total,
                                        bank_totals=bank_totals,
                                        contrast_total=contrast_total, wrong_nll_total=wrong_nll_total)
                        compute.close()
                        save_checkpoint(agent, optimizer, output, completed, rng, cache, fingerprint,
                                        keep=config.train.keep_checkpoints, archiver=archiver,
                                        accumulation=progress)
                        break
                if progress:
                    break
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), config.train.clip_grad_norm,
                                                           error_if_nonfinite=True)
                optimizer.step()
            row = {"step": step + 1, **totals, "oracle_anchor_nll": anchor_total,
                   "oracle_alignment_loss": alignment_total,
                   "oracle_distillation_kl": distillation_total,
                   "payload_contrast_loss": contrast_total,
                   "swapped_source_nll": wrong_nll_total,
                   "optimization_loss": (totals["loss"] + config.train.oracle_anchor_weight * anchor_total
                                         + config.train.oracle_alignment_weight * alignment_total
                                         + config.train.oracle_distillation_weight * distillation_total
                                         + (config.train.payload_contrast_weight +
                                            config.train.bank_payload_contrast_weight) * contrast_total),
                   "grad_norm": float(grad_norm), "loops": agent.backbone.loops,
                   "elapsed_seconds": time.perf_counter() - start_time}
            if bank_totals is not None:
                row['bank'] = bank_totals | {'generation': bank_manifest['generation'],
                                             'supplied_positive': True}
            if sampler is not None:
                consumed = ((step + 1) * config.train.gradient_accumulation *
                            config.train.batch_size)
                row['sampling'] = {'policy': 'shuffled_passes',
                                   'completed_passes': consumed // len(episodes),
                                   'position_in_pass': consumed % len(episodes),
                                   'unique_episodes_seen': min(consumed, len(episodes))}
            if hasattr(agent.backbone, "manifest"):
                row["recurrence"] = agent.backbone.manifest()
            if (step + 1) % config.train.log_every == 0 or step == start:
                row['memory'] = memory_metrics(config.train.device)
            log.write(json.dumps(row) + "\n")
            log.flush()
            tracker.log(row)
            history.append(row)
            if (step + 1) % config.train.log_every == 0 or step == start:
                print(json.dumps(row), flush=True)
            completed = step + 1
            stopping = want_stop() or (stop_after is not None and completed - start >= stop_after)
            if completed % config.train.checkpoint_every == 0 or stopping or completed == config.train.steps:
                save_checkpoint(agent, optimizer, output, completed, rng, cache, fingerprint,
                                keep=config.train.keep_checkpoints, archiver=archiver)
                last_saved_step = completed
            if stopping:
                break
    if archiver is not None:
        archiver.close(wait=not want_stop())
    summary = {"steps": completed, "requested_steps": config.train.steps, 'stop_requested': bool(want_stop()),
               'saved_microbatches': progress.get('microbatches', 0), 'resource_stop': resource_stop,
               "stopped_early": completed < config.train.steps, "last": history[-1] if history else None,
               "environment": manifest, "resources": resource_report(), "store": cache.sizes()}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def config_from_run(path: Path) -> Config:
    from .config import ModelConfig, MemoryConfig, TrainConfig
    checkpoint = resolve_checkpoint(path)
    raw = json.loads((checkpoint / "config.json").read_text())
    return Config(ModelConfig(**raw["model"]), MemoryConfig(**raw["memory"]), TrainConfig(**raw["train"]))


@torch.no_grad()
def build_evaluation_store(agent: SchnitzelAgent, store: DiskStore,
                           episodes: list[Episode], generation: str) -> None:
    if agent.training:
        raise ValueError("Freeze the writer in eval mode before building an evaluation bank")
    if agent.config.train.arm not in {"memory", "direct_latent"}:
        return
    with autocast_context(agent.config):
        for episode in episodes:
            for name, variant in (("original", episode),
                                  ("cf_restoration", counterfactual(episode, "restoration")),
                                  ("cf_permission", counterfactual(episode, "permission"))):
                for source in variant.supports:
                    outputs = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
                    persist_outputs(store, agent, source, outputs, f"{name}/{episode.episode_id}", generation)


@torch.no_grad()
def stored_evaluation(agent: SchnitzelAgent, store: DiskStore, episodes: list[Episode],
                      generation: str, *, generate: bool = False) -> dict:
    """Stored-only causal interventions. Full-sequence scoring; no writer calls."""
    from .evaluation import score_answers
    from .sessions import read_session
    from .metrics import summarize_rows, paired_world_bootstrap, counterfactual_metrics
    from .routing import complete_support_recall
    agent.eval()
    rows = []
    conditions = ["all", "none", "A", "B", "irrelevant", "zero_values", "cf_restoration", "cf_permission"]
    if generate and (agent.config.memory.read_steps > 1 or agent.config.train.arm not in {"memory", "no_memory"}):
        raise ValueError("Use likelihood evaluation for multi-read/control arms")
    for episode in episodes:
        original_plans = None
        original_gate_weights = None
        for condition in conditions:
            cf = condition.startswith("cf_")
            variant = counterfactual(episode, "restoration" if condition == "cf_restoration" else "permission") if cf else episode
            namespace = f"{condition if cf else 'original'}/{episode.episode_id}"
            if condition == "none":
                selected_ids = []
            elif condition == "A":
                selected_ids = [episode.required_ids[0]]
            elif condition == "B":
                selected_ids = [episode.required_ids[1]]
            elif condition == "irrelevant":
                selected_ids = [s.record_id for s in variant.supports if s.kind == "irrelevant"]
            else:
                selected_ids = list(evidence_ids(variant, agent.config.train.evidence_scope))
            with autocast_context(agent.config):
                arm = agent.config.train.arm
                text = "\n".join(s.text for s in variant.supports if s.record_id in selected_ids)
                prompt = agent.prompt_ids(episode.query, text if arm == "oracle_text" else "")
                memory, selected_spaces, read_count = None, [[] for _ in agent.config.memory.payload_dims], 0
                if arm in {"memory", "direct_latent"} and condition != "none":
                    learned = agent.config.train.retrieval == "learned" and condition in {"all", "zero_values"}
                    session = read_session(agent, store, prompt, namespace=namespace, generation=generation,
                                           query_time=episode.query_time,
                                           oracle_ids=None if learned else tuple(selected_ids),
                                           ablate_values=condition == "zero_values",
                                           fixed_plans=[[replace(p, namespace=namespace) for p in step] for step in original_plans]
                                               if condition == 'zero_values' or cf else None,
                                           fixed_gate_weights=original_gate_weights
                                               if condition == 'zero_values' else None)
                    memory, selected_spaces = session.memory, session.selected_ids
                    if condition == 'all':
                        original_plans = session.plans
                        original_gate_weights = session.gate_weights
                    read_count = len(session.plans)
                elif arm == "shared_compute":
                    memory = agent.shared_compute_tokens(prompt)
                elif arm == "oracle_text":
                    selected_spaces = [selected_ids]
                score = score_answers(agent, prompt, memory, variant.answer,
                                      episode.choices or ("STOP", "RETRY", "RESTORE_RETRY"))
                prediction = agent.generate_from_memory(prompt, memory) if generate else None
            selected_set = set().union(*(set(ids) for ids in selected_spaces))
            groups = [set(g) for g in variant.sufficient_groups or (variant.required_ids,)]
            rows.append({"episode": episode.episode_id, "environment": episode.environment,
                         "condition": condition, "answer": variant.answer, **score,
                         "selected_ids": selected_spaces, "read_count": read_count,
                         "complete_support": complete_support_recall(selected_set, groups),
                         "counterfactual_should_change": variant.answer != episode.answer,
                         "generated_text": prediction,
                         "generation_exact_match": prediction == variant.answer if generate else None})
    return {"schema_version": 2, "protocol": "stored-only frozen weights; full-sequence choice NLL with EOS",
            "notice": "Synthetic diagnostics; not coding or capacity-substitution evidence.",
            "summary": summarize_rows(rows), "rows": rows,
            "counterfactuals": counterfactual_metrics(rows),
            "paired_memory_benefit": paired_world_bootstrap(rows), "resources": resource_report()}


def evaluate_run(run: str | Path, *, count: int | None = None, generate: bool = False) -> dict:
    run = Path(run)
    config = config_from_run(run)
    from .runtime import configure_memory
    configure_memory(config.train)
    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device)
    load_model(agent, str(resolve_checkpoint(run) / "model.safetensors"), device=config.train.device)
    agent.eval()
    episodes = [make_episode(i, split=f"heldout-{config.train.seed}", distractors=config.train.distractors)
                for i in range(count or config.train.eval_worlds)]
    evaluation_dir = run / ("evaluation-" + str(time.time_ns()))
    evaluation_dir.mkdir()
    save_episodes(evaluation_dir / "episodes.jsonl", episodes)
    store = DiskStore(evaluation_dir / "stored_payloads.sqlite")
    build_evaluation_store(agent, store, episodes, generation="frozen-eval-v0")
    # Close/reopen the persistent boundary before reading anything.
    store = DiskStore(evaluation_dir / "stored_payloads.sqlite")
    result = stored_evaluation(agent, store, episodes, "frozen-eval-v0", generate=generate)
    result["store"] = store.sizes()
    (evaluation_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"directory": str(evaluation_dir), "summary": result["summary"]}, indent=2))
    return result


@torch.no_grad()
def evaluate_episode_file(run: str | Path, path: str | Path) -> dict:
    """General support/query transfer likelihood; no synthetic action assumptions.

    Writer encodes each new support once into a bank. The comparison then consumes
    only reloaded payloads, with complete, missing, or zeroed support. Full coding
    verifiers and tool execution are deliberately not substituted by this metric.
    """
    run = Path(run)
    config = config_from_run(run)
    from .runtime import configure_memory
    configure_memory(config.train)
    if config.memory.read_timing == "loop_boundary":
        raise ValueError("Use evaluate-transfer, evaluate-teachers or evaluate-depths for native in-loop reads")
    episodes = load_episodes(path)
    torch.set_num_threads(config.train.threads)
    agent = SchnitzelAgent(config).to(config.train.device)
    load_model(agent, str(resolve_checkpoint(run) / "model.safetensors"), device=config.train.device)
    agent.eval()
    output = run / ("transfer-" + str(time.time_ns()))
    output.mkdir()
    store = DiskStore(output / "payloads.sqlite")
    with autocast_context(config):
        for episode in episodes:
            for source in episode.supports:
                encoded = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
                persist_outputs(store, agent, source, encoded, episode.episode_id, "frozen-v0")
    store = DiskStore(output / "payloads.sqlite")
    rows = []
    with autocast_context(config):
        for episode in episodes:
            for condition in ["all", "none", "zero_values"]:
                arm = config.train.arm
                selected = list(evidence_ids(episode, config.train.evidence_scope)) if condition != "none" else []
                text = "\n".join(s.text for s in episode.supports if s.record_id in selected)
                prompt = agent.prompt_ids(episode.query, text if arm == "oracle_text" else "")
                q, routing_query = agent.query_pair(prompt)
                payloads, ids = [], []
                for space, dim in enumerate(config.memory.payload_dims):
                    if config.train.retrieval == "learned" and condition != "none":
                        plan = store.search(agent.routing_address(routing_query, space)[0], namespace=episode.episode_id,
                            space=f"s{space}", generation="frozen-v0", query_time=episode.query_time,
                            top_k=agent.requested_records(q, config.memory.neighbors[space]))
                    else:
                        plan = ReadPlan(episode.episode_id, f"s{space}", "frozen-v0", "research",
                                        episode.query_time, tuple(Selection(i, 0.) for i in selected))
                    values = store.fetch(plan)
                    payloads.append(torch.stack(values).float().to(agent.device) if values else q.new_empty(0, dim))
                    ids.extend(s.record_id for s in plan.selections)
                memory = None
                if any(x.shape[0] for x in payloads):
                    if arm == "memory":
                        memory, _ = agent.read_tokens(payloads, q, ablate_values=condition == "zero_values")
                    elif arm == "direct_latent":
                        memory = payloads[0].reshape(1, -1, agent.width)
                        if condition == "zero_values":
                            memory = torch.zeros_like(memory)
                loss = float(agent.conditioned_nll(prompt, agent.target_ids(episode.answer), memory))
                rows.append({"episode": episode.episode_id, "condition": condition,
                             "mean_target_nll": loss, "selected_ids": ids,
                             "complete_support": set(episode.required_ids) <= set(ids)})
    result = {"protocol": "frozen writer, serialize/reload, stored-only answer likelihood",
              "notice": "Token NLL is a transfer diagnostic, not coding correctness. zero_values only changes latent payloads.",
              "rows": rows, "store": store.sizes(), "resources": resource_report()}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"directory": str(output), "rows": len(rows)}))
    return result
