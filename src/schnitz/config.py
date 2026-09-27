from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    backend: str = "hf"
    model_id: str = "LiquidAI/LFM2.5-230M"
    revision: str = "main"  # resolved commit is logged; pin before scientific runs
    loops: int = 1
    recurrence_mode: str = "full_stack"  # legacy reference or native middle_block
    recurrent_start: int = 4
    recurrent_end: int = 10
    recurrent_input_mix: float = 0.1
    recurrent_update_mix: float = 0.1
    writer_loops: int | None = None  # None preserves old checkpoints; new recipes use 1
    backbone_train_scope: str = "all"  # all or recurrent_core
    tiny_width: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 4
    gradient_checkpointing: bool = False
    freeze_backbone: bool = False


@dataclass
class MemoryConfig:
    key_dim: int = 64
    # shared_maps: one writer/query key head plus per-space linear maps.
    # direct: one writer-state and one query-state key head per space.
    key_interface: str = "shared_maps"
    # Direct heads only: one stored key width per space (empty means key_dim).
    key_dims: list[int] = field(default_factory=list)
    # first: key slot precedes value slots (legacy). last: values cannot see it.
    key_slot_position: str = "first"
    write_slots: int = 8
    read_slots: int = 8
    payload_dims: list[int] = field(default_factory=lambda: [256])
    payload_layout: str = "flat"  # flat, positional operator, or joint full-width tokens
    # joint_tokens: full decoder-width stored tokens per space, jointly mixed
    # from every writer slot (payload_dims[s] == space_tokens[s] * width).
    space_tokens: list[int] = field(default_factory=list)
    neighbors: list[int] = field(default_factory=lambda: [16])
    reader_width: int = 128
    reader_rounds: int = 3
    reader: str = "mlp"
    chunk_size: int = 32
    checkpoint_chunks: bool = False
    storage_dtype: str = "bfloat16"
    compaction: str = "none"
    compact_records: int = 2
    compaction_probability: float = 0.25
    compaction_loss_weight: float = 0.1
    compaction_warmup: int = 100
    compaction_objective: str = "interleaved"  # paired adds raw-task anchoring
    compaction_grouping: str = "whole"
    compaction_group_size: int = 4
    field_memberships: int = 2
    compact_task_weight: float = 1.0
    behavior_kl_weight: float = 0.0
    merge_loss_weight: float = 0.0
    read_timing: str = "prefix"  # prefix or loop_boundary
    read_steps: int = 1  # scheduled causal reads, not a replay memory limit
    stream_reads: bool = False  # single-space stored-only bounded staging
    read_top_k: int = 1
    noise_std: float = 0.0
    quantization_step: float = 0.0
    independent_routing_query: bool = False
    distance_gating: bool = False
    gate_density_k: int = 8
    gate_density_fraction: float = 0.0  # radius at max(k, fraction × records read)
    gate_floor_mode: str = 'max'  # 'additive' keeps gate gradient on floored supports
    gate_min_temperature: float = 0.02
    gate_max_temperature: float = 0.5
    gate_initial_temperature: float = 0.1
    gate_max_radius_adjustment: float = 0.25


@dataclass
class TrainConfig:
    seed: int = 17
    steps: int = 200
    learning_rate: float = 0.0001
    backbone_learning_rate: float = 0.00001
    optimizer: str = 'adamw'  # historical default; new Muon recipes opt in explicitly
    weight_decay: float = .01
    adam_betas: list[float] = field(default_factory=lambda: [.9, .999])
    adam_eps: float = 1e-8
    muon_momentum: float = .95
    muon_ns_steps: int = 5
    gradient_accumulation: int = 4
    batch_size: int = 1  # examples executed together per accumulation microbatch
    tokenized_episodes_file: str | None = None
    sampling_policy: str = 'random_with_replacement'  # or deterministic shuffled_passes
    bank_dir: str | None = None  # verified base snapshot; spatial training adds a mutable overlay
    bank_read_limits: list[int] = field(default_factory=list)  # per-space selected records, <= neighbors
    bank_routing_candidates: int = 8  # exact hard-negative pool per space, before supplied positives
    payload_contrast_weight: float = 0.0  # source-swap ranking on verified one-source episodes
    payload_contrast_margin: float = 0.5
    bank_payload_contrast_weight: float = 0.0  # stored-value source swap under a fixed bank plan
    bank_payload_contrast_margin: float = 0.5
    clip_grad_norm: float = 1.0
    precision: str = "bf16"
    device: str = "cuda"
    replay: bool = True
    verify_replay: bool = False
    live_fraction: float = 0.5
    selected_producers_only: bool = False  # explicit all-live oracle compute policy
    train_worlds: int = 128
    distractors: int = 2
    eval_worlds: int = 16
    log_every: int = 10
    max_source_tokens: int = 512
    max_prompt_tokens: int = 1024
    arm: str = "memory"  # memory/no_memory/oracle_text/direct_latent
    retrieval: str = "oracle"  # oracle or learned (candidate-local top-k)
    routing_weight: float = 0.1
    routing_logit_scale: float = 1.0
    routing_live_weight: float = 0.0
    writer_key_learning_rate: float | None = None
    key_stability_weight: float = 0.0
    routing_warmup: int = 100
    support_gate_floor: float = 0.0
    # Spatial bank reads (R5). Forced gold: with this probability per read site,
    # verified supports that were not retrieved replace the lowest-ranked records.
    gold_force_probability: float = 1.0
    exploration_fraction: float = 0.0  # share of each read drawn by low retrieval load
    # Small spreading regularizers (key_geometry): query-address variance and
    # covariance, KoLeo over each read's candidate keys, and a hub-load penalty.
    spread_variance_weight: float = 0.0
    spread_covariance_weight: float = 0.0
    koleo_weight: float = 0.0
    load_penalty_weight: float = 0.0
    load_decay: float = 0.999
    load_threshold: float = 4.0
    # Error-driven writer learning against a large bank (record_gradients.py):
    # live keys from cached key-slot states, decayed per-record cotangents, writer
    # backward for a close neighbourhood of each step's reads, rolling refresh.
    record_gradients: bool = False
    record_gradient_decay: float = 0.9
    record_neighborhood_fraction: float = 0.25  # of each space's read limit, plus gold
    record_flush_budget: int = 256  # writer backward records per step
    record_flush_extra: int = 32  # largest accumulated gradients outside the neighbourhood
    record_refresh_per_step: int = 256  # re-encodes per step, stalest first
    record_refresh_backward: bool = False  # key-table mode: train key prediction on refreshes
    record_gradient_capacity: int = 16384
    # Terminal per-record key table (record_gradients.KeyTable): search and gates use
    # trainable table keys, updated only where touched; the decoder regresses its
    # predicted keys onto the table whenever it encodes a record, and a small
    # commitment term keeps table keys predictable from content.
    key_table: bool = False
    key_table_learning_rate: float = 1.0e-3
    key_table_optimizer: str = 'sphere_adam'  # or 'adam' (per-coordinate)
    key_table_max_step: float = 0.05  # per-row trust region, about radians per step
    key_prediction_weight: float = 1.0
    key_commitment_weight: float = 0.05  # fraction a row moves toward each prediction
    writer_replay_records_per_site: int = 0
    threads: int = 4
    cuda_memory_fraction: float | None = None
    min_system_available_bytes: int = 0
    stall_timeout_seconds: float = 0
    optimization_scope: str = "all"  # all, compactor, or routing address projections only
    checkpoint_every: int = 1000
    keep_checkpoints: int = 2
    max_target_tokens: int = 512
    loop_counts: list[int] = field(default_factory=list)  # sampled once per optimizer step
    oracle_anchor_loops: int | None = None
    allow_recurrence_conversion: bool = False
    reinitialize_reader: bool = False  # explicit warm-start fork; never applied on resume
    parent_kl_weight: float = 0.0  # fixed one-pass parent, only while native base is frozen
    oracle_anchor_weight: float = 0.0
    oracle_alignment_weight: float = 0.0  # detached selected-text answer-state cosine loss
    oracle_distillation_weight: float = 0.0  # fixed one-pass selected-text output KL
    warmstart_memory_gate: float | None = None  # fresh native warm-start only; exact resume loads checkpoint
    episodes_file: str | None = None  # optional general support/query JSONL
    evidence_scope: str = 'required'  # oracle-selected supports or all causally available candidates
    archive_dir: str | None = None  # existing external storage root; never auto-mount
    archive_keep_checkpoints: int = 3
    min_free_disk_bytes: int = 1024 ** 3
    wandb_mode: str = 'disabled'  # opt-in: offline or online
    wandb_project: str = 'schnitz'
    wandb_entity: str | None = None
    wandb_group: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        m, r, t = self.model, self.memory, self.train
        if (t.cuda_memory_fraction is not None and not 0 < t.cuda_memory_fraction <= 1
                or t.min_system_available_bytes < 0 or t.stall_timeout_seconds < 0):
            raise ValueError('Invalid runtime resource limits')
        if t.optimizer not in {'adamw', 'muon'}:
            raise ValueError('Optimizer must be adamw or muon')
        if (t.writer_key_learning_rate is not None
                and (not math.isfinite(t.writer_key_learning_rate)
                     or t.writer_key_learning_rate <= 0)):
            raise ValueError('Writer key learning rate must be positive and finite')
        if not math.isfinite(t.key_stability_weight) or t.key_stability_weight < 0:
            raise ValueError('Key stability weight must be nonnegative and finite')
        if not math.isfinite(t.routing_logit_scale) or t.routing_logit_scale <= 0:
            raise ValueError('Routing logit scale must be positive and finite')
        if not math.isfinite(t.routing_live_weight) or t.routing_live_weight < 0:
            raise ValueError('Live-key routing weight must be nonnegative and finite')
        if t.routing_live_weight and not t.writer_replay_records_per_site:
            raise ValueError('Live-key routing requires writer replay')
        if t.sampling_policy not in {'random_with_replacement', 'shuffled_passes'}:
            raise ValueError('Unknown episode sampling policy')
        if t.bank_dir is not None:
            if (not t.bank_dir or t.arm != 'memory' or t.retrieval != 'learned'
                    or not m.freeze_backbone or m.recurrence_mode != 'middle_block'
                    or m.loops != 2 or r.read_timing != 'loop_boundary' or r.read_steps != 1
                    or r.compaction != 'none' or t.live_fraction != 0
                    or t.selected_producers_only or t.routing_weight <= 0
                    or t.oracle_anchor_weight or t.oracle_alignment_weight
                    or t.oracle_distillation_weight or t.loop_counts):
                raise ValueError('Bank training requires frozen-writer R=2 learned routing and stored-only reads')
            if (len(t.bank_read_limits) != len(r.neighbors)
                    or not isinstance(t.bank_routing_candidates, int)
                    or isinstance(t.bank_routing_candidates, bool)
                    or t.bank_routing_candidates < 1 or
                    any(not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > cap
                        for k, cap in zip(t.bank_read_limits, r.neighbors, strict=True))):
                raise ValueError('Bank read limits must be positive per-space counts within neighbor caps')
        if (not math.isfinite(t.payload_contrast_weight) or t.payload_contrast_weight < 0 or
                not math.isfinite(t.payload_contrast_margin) or t.payload_contrast_margin < 0):
            raise ValueError('Invalid source-swap contrast weight or margin')
        if t.payload_contrast_weight and (
                t.bank_dir is not None or t.arm != 'memory' or t.retrieval != 'oracle'
                or t.live_fraction != 1.0 or t.selected_producers_only
                or t.evidence_scope != 'required'
                or r.compaction != 'none' or r.read_steps != 1
                or m.recurrence_mode != 'middle_block' or m.loops != 2
                or r.read_timing != 'loop_boundary' or t.oracle_anchor_weight
                or t.oracle_alignment_weight or t.oracle_distillation_weight):
            raise ValueError('Source-swap contrast requires fully live R=2 oracle memory and two selected sources')
        if (not math.isfinite(t.bank_payload_contrast_weight) or t.bank_payload_contrast_weight < 0
                or not math.isfinite(t.bank_payload_contrast_margin)
                or t.bank_payload_contrast_margin < 0):
            raise ValueError('Invalid stored-bank source-swap weight or margin')
        if t.bank_payload_contrast_weight and t.bank_dir is None:
            raise ValueError('Stored-bank source-swap requires a published bank')
        if not math.isfinite(t.oracle_alignment_weight) or t.oracle_alignment_weight < 0:
            raise ValueError('Invalid oracle alignment weight')
        if t.oracle_alignment_weight and (t.arm != 'memory' or t.retrieval != 'oracle'
                or r.read_timing != 'loop_boundary' or r.compaction != 'none'
                or not math.isfinite(t.oracle_anchor_weight) or t.oracle_anchor_weight <= 0):
            raise ValueError('Oracle alignment requires anchored oracle memory with native reads and no compaction')
        if not math.isfinite(t.oracle_distillation_weight) or t.oracle_distillation_weight < 0:
            raise ValueError('Invalid oracle distillation weight')
        if t.oracle_distillation_weight and (
                t.arm != 'memory' or t.retrieval != 'oracle' or t.live_fraction != 1.
                or r.read_timing != 'loop_boundary' or r.read_steps != 1 or r.compaction != 'none'
                or m.recurrence_mode != 'middle_block' or m.loops != 2 or m.writer_loops != 1
                or not m.freeze_backbone or t.oracle_anchor_weight or t.oracle_alignment_weight
                or t.loop_counts):
            raise ValueError('Oracle distillation requires fully live, fixed-parent R=2 oracle memory without another text objective')
        if t.warmstart_memory_gate is not None and (
                not math.isfinite(t.warmstart_memory_gate) or not 0 < t.warmstart_memory_gate < 1
                or m.recurrence_mode != 'middle_block'):
            raise ValueError('Warm-start memory gate must be a probability for native recurrence')
        if t.selected_producers_only and (t.arm != 'memory' or t.retrieval != 'oracle' or t.live_fraction != 1.):
            raise ValueError('Selected-only producers require fully live oracle memory training')
        if (t.batch_size < 1 or not isinstance(t.batch_size, int) or isinstance(t.batch_size, bool)):
            raise ValueError('Training batch size must be a positive integer')
        if (t.weight_decay < 0 or t.adam_eps <= 0 or len(t.adam_betas) != 2
                or any(not 0 <= b < 1 for b in t.adam_betas)
                or not 0 <= t.muon_momentum < 1 or t.muon_ns_steps < 1):
            raise ValueError('Invalid optimizer hyperparameters')
        if m.backend not in {"tiny", "hf"} or m.loops < 1:
            raise ValueError("Invalid backbone configuration")
        if m.recurrence_mode not in {"full_stack", "middle_block"}:
            raise ValueError("Invalid recurrence mode")
        if m.writer_loops is not None and m.writer_loops < 1:
            raise ValueError("Writer depth must be positive")
        if m.backbone_train_scope not in {"all", "recurrent_core"}:
            raise ValueError("Invalid backbone train scope")
        if m.recurrence_mode == "middle_block":
            if not 0 <= m.recurrent_start < m.recurrent_end:
                raise ValueError("Invalid recurrent partition")
            if m.backend == "tiny" and m.recurrent_end > m.tiny_layers:
                raise ValueError("Recurrent partition exceeds tiny decoder depth")
            if not 0 < m.recurrent_input_mix < 1 or not 0 < m.recurrent_update_mix < 1:
                raise ValueError("Recurrent gates must start live, strictly between zero and one")
        elif m.backbone_train_scope == "recurrent_core":
            raise ValueError("Core-only training requires middle-block recurrence")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in t.loop_counts):
            raise ValueError("Sampled loop counts must be positive integers")
        if t.oracle_anchor_loops is not None and t.oracle_anchor_loops < 1:
            raise ValueError("Anchor depth must be positive")
        if r.read_timing not in {"prefix", "loop_boundary"}:
            raise ValueError("Invalid read timing")
        if r.read_timing == "loop_boundary":
            if m.recurrence_mode != "middle_block":
                raise ValueError("In-loop reads require the native recurrent core")
            if r.stream_reads or t.arm == "direct_latent":
                raise ValueError("In-loop training requires materialized set reads")
            counts = t.loop_counts or [m.loops]
            if t.arm in {"memory", "shared_compute"} and min(counts) < 2:
                raise ValueError("Memory training requires a read followed by another core pass")
            if r.read_steps > 1 and min(counts) <= r.read_steps:
                raise ValueError("A scheduled read must leave at least one core pass to use it")
        if len(r.payload_dims) != len(r.neighbors) or not r.payload_dims:
            raise ValueError("payload_dims and neighbors must have the same nonzero length")
        if min(*r.payload_dims, *r.neighbors, r.key_dim, r.write_slots, r.read_slots,
               r.reader_width, r.reader_rounds, r.chunk_size) < 1:
            raise ValueError("All memory dimensions and counts must be positive")
        if r.reader not in {"mlp", "attention"} or r.compaction not in {"none", "mean", "synthetic"}:
            raise ValueError("Invalid reader/compactor")
        if r.payload_layout not in {"flat", "positional", "joint_tokens"}:
            raise ValueError("Invalid payload layout")
        if r.payload_layout == "joint_tokens":
            if (len(r.space_tokens) != len(r.payload_dims) or min(r.space_tokens) < 1
                    or any(dim % tokens for dim, tokens in
                           zip(r.payload_dims, r.space_tokens, strict=True))):
                raise ValueError("Joint token payloads need one token count per space")
            if r.reader != "mlp" or r.compaction != "none":
                raise ValueError("Joint token payloads use the MLP operator without compaction")
        elif r.space_tokens:
            raise ValueError("space_tokens applies only to the joint token layout")
        if r.key_interface not in {"shared_maps", "direct"}:
            raise ValueError("Invalid key interface")
        if r.key_interface == "direct" and r.independent_routing_query:
            raise ValueError("Direct query key heads already separate routing from reading")
        if r.key_dims and (r.key_interface != "direct" or len(r.key_dims) != len(r.payload_dims)
                           or min(r.key_dims) < 1):
            raise ValueError("Per-space key widths need direct heads and one width per space")
        if r.key_slot_position not in {"first", "last"}:
            raise ValueError("Key slot position must be first or last")
        if r.payload_layout == "positional":
            if r.reader != "mlp":
                raise ValueError("The positional interface currently requires the MLP operator")
            if any(dim % r.write_slots for dim in r.payload_dims):
                raise ValueError("Every positional payload must divide into writer slots")
        if (not isinstance(r.gate_density_k, int) or isinstance(r.gate_density_k, bool)
                or r.gate_density_k < 1 or not 0 < r.gate_min_temperature
                < r.gate_initial_temperature < r.gate_max_temperature
                or r.gate_max_radius_adjustment < 0
                or not 0 <= r.gate_density_fraction <= 1
                or r.gate_floor_mode not in {'max', 'additive'}):
            raise ValueError('Invalid adaptive distance-gate configuration')
        if (not 0 <= t.support_gate_floor <= 1
                or not isinstance(t.writer_replay_records_per_site, int)
                or isinstance(t.writer_replay_records_per_site, bool)
                or t.writer_replay_records_per_site < 0):
            raise ValueError('Invalid support floor or key replay budget')
        if (not 0 <= t.gold_force_probability <= 1 or not 0 <= t.exploration_fraction < 1
                or any(not math.isfinite(w) or w < 0 for w in (
                    t.spread_variance_weight, t.spread_covariance_weight,
                    t.koleo_weight, t.load_penalty_weight))
                or not 0 < t.load_decay < 1 or t.load_threshold <= 1):
            raise ValueError('Invalid gold forcing, exploration or spreading settings')
        if t.record_gradients and (
                t.writer_replay_records_per_site or t.routing_live_weight
                or not r.distance_gating or r.key_interface != 'direct'
                or not 0 <= t.record_gradient_decay < 1
                or not 0 < t.record_neighborhood_fraction <= 1
                or min(t.record_flush_budget, t.record_gradient_capacity) < 1
                or min(t.record_flush_extra, t.record_refresh_per_step) < 0):
            raise ValueError('Record gradients replace live writer replay and need direct '
                             'keys, distance gating and valid budgets')
        if t.key_table and (not t.record_gradients or t.key_table_learning_rate <= 0
                            or t.key_table_optimizer not in {'sphere_adam', 'adam'}
                            or t.key_table_max_step <= 0
                            or t.key_prediction_weight < 0
                            or not 0 <= t.key_commitment_weight <= 1):
            raise ValueError('A key table needs record gradients and valid weights')
        if t.writer_replay_records_per_site and not r.distance_gating:
            raise ValueError('Continuous record replay requires distance gating')
        if len(r.payload_dims) > 1 and r.compaction != "none":
            raise ValueError("Integrated compaction runner is single-space; multi-space APIs are separate")
        if r.storage_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Invalid storage precision")
        if t.precision not in {"fp32", "bf16"} or t.device not in {"cpu", "cuda"}:
            raise ValueError("Use fp32/bf16 and cpu/cuda")
        if not 0 <= t.live_fraction <= 1 or not 0 <= r.compaction_probability <= 1:
            raise ValueError("Probabilities must be between zero and one")
        if min(t.steps, t.train_worlds, t.eval_worlds, t.gradient_accumulation, t.log_every,
               t.max_source_tokens, t.max_prompt_tokens, t.max_target_tokens, t.threads, t.checkpoint_every, t.keep_checkpoints) < 1:
            raise ValueError("Training counts must be positive")
        if t.arm not in {"memory", "no_memory", "oracle_text", "direct_latent", "shared_compute"}:
            raise ValueError("Unknown comparison arm")
        if t.retrieval not in {"oracle", "learned"}:
            raise ValueError("Unknown retrieval mode")
        if t.evidence_scope not in {'required', 'available'}:
            raise ValueError('Invalid evidence scope')
        if t.evidence_scope == 'available' and t.retrieval != 'oracle':
            raise ValueError('Available evidence scope requires oracle candidate access; it is not learned routing')
        if r.noise_std < 0 or r.quantization_step < 0 or r.compact_records < 1:
            raise ValueError("Invalid noise or compaction settings")
        if r.compaction_objective not in {"interleaved", "paired"}:
            raise ValueError("Invalid compaction objective")
        if r.compaction_grouping not in {"whole", "random", "local", "overlap"}:
            raise ValueError("Invalid compaction grouping")
        if min(r.compaction_group_size, r.field_memberships, r.read_steps, r.read_top_k) < 1:
            raise ValueError("Positive grouping/read counts required")
        if min(r.compact_task_weight, r.behavior_kl_weight, r.merge_loss_weight) < 0:
            raise ValueError("Loss weights cannot be negative")
        if len(r.payload_dims) > 1 and r.merge_loss_weight:
            raise ValueError("Merge objective currently supports one memory space")
        if r.stream_reads and (len(r.payload_dims) != 1 or t.arm != "memory"):
            raise ValueError("Streaming currently supports one-space latent-memory inference")
        if r.read_steps > 1 and (r.compaction != "none" or t.arm != "memory"):
            raise ValueError("Scheduled multi-read is initially a raw-memory comparison")
        if t.optimization_scope not in {"all", "compactor", "routing"}:
            raise ValueError("Invalid optimization scope")
        if t.optimization_scope == 'routing' and (t.arm != 'memory' or t.retrieval != 'learned'
                                                  or t.routing_weight <= 0 or r.compaction != 'none'):
            raise ValueError('Routing-only runs require learned memory routing, positive routing weight and raw records')
        if t.optimization_scope == "compactor" and (r.compaction != "synthetic" or r.compaction_probability != 1.0 or r.compaction_warmup != 0):
            raise ValueError("Compactor-only runs require synthetic compaction on every step without warmup")
        if t.oracle_anchor_weight < 0 or t.parent_kl_weight < 0:
            raise ValueError("Negative retention loss weight")
        if t.parent_kl_weight and (not m.freeze_backbone or m.recurrence_mode != "middle_block"
                                    or t.arm not in {"oracle_text", "no_memory"}):
            raise ValueError("Parent KL needs a frozen native backbone and identical text inputs")
        if t.loop_counts and m.recurrence_mode == "middle_block" and m.writer_loops is None:
            raise ValueError("Specify writer_loops independently of sampled consumer depth")
        if t.distractors < 0 or min(t.learning_rate, t.backbone_learning_rate) <= 0:
            raise ValueError("Invalid distractor count or learning rate")
        if t.archive_keep_checkpoints < 1 or t.min_free_disk_bytes < 0:
            raise ValueError('Invalid archive retention/free disk reserve')
        if t.wandb_mode not in {'disabled', 'offline', 'online'}:
            raise ValueError('wandb_mode must be disabled, offline or online')


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) - {"model", "memory", "train"}:
        raise ValueError("Unknown configuration sections")
    config = Config(ModelConfig(**raw.get("model", {})), MemoryConfig(**raw.get("memory", {})),
                    TrainConfig(**raw.get("train", {})))
    config.validate()
    return config
