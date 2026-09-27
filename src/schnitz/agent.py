"""Shared-backbone writing, querying, composition and soft-token-conditioned loss."""
from __future__ import annotations

import copy

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import ByteTokenizer, HFBackbone, RecurrentBackbone, TinyBackbone
from .compaction import SyntheticCompactor, contribution_loss, storage_noise, compact_view
from .config import Config
from .recurrence import (
    MiddleBlockBackbone,
    MiddleBlockState,
    LoopMemory,
    LoopWrite,
    LoopWrites,
    SpatialReadSite,
)
from .readers import MultiSpaceReader, SetReader
from .positional import (JointTokenCodec, MultiSpacePositionalReader, PositionalCodec,
                         PositionalCompactor, PositionalSetReader)
from .routing import AdaptiveDistanceGate, cosine_scores, group_plan_loss


def answer_state_alignment(student: Tensor, teacher: Tensor) -> Tensor:
    """Match corresponding next-token states; teacher states never receive this gradient."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError('Alignment requires the same answer positions and hidden width')
    return (1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)).mean()


def answer_distribution_kl(student: Tensor, teacher: Tensor) -> Tensor:
    """Match next-token distributions; the selected-text teacher is fixed and detached."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError('Distillation requires the same answer positions and vocabulary')
    return F.kl_div(student.float().log_softmax(-1), teacher.detach().float().softmax(-1),
                    reduction='none').sum(-1).mean()


@dataclass
class ForwardResult:
    loss: Tensor
    nll: Tensor
    routing_loss: Tensor
    compaction_loss: Tensor
    selected: list[list[int]]
    raw_nll: Tensor | None = None
    compact_nll: Tensor | None = None
    behavior_kl: Tensor | None = None
    read_count: int = 1
    parent_kl: Tensor | None = None
    answer_states: Tensor | None = None


@dataclass(frozen=True)
class SpatialExecution:
    """One whole-sequence graph paused at a recurrent retrieval boundary."""
    recurrent: MiddleBlockState
    sites: tuple[SpatialReadSite, ...]
    batch: int


class SchnitzelAgent(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        m, r = config.model, config.memory
        if m.backend == "tiny":
            base = TinyBackbone(m.tiny_width, m.tiny_layers, m.tiny_heads)
            base.gradient_checkpointing = m.gradient_checkpointing
            self.tokenizer = ByteTokenizer()
            self.resolved_revision = "local-tiny"
        else:
            base = HFBackbone(m.model_id, revision=m.revision,
                              gradient_checkpointing=m.gradient_checkpointing)
            self.tokenizer = base.tokenizer
            self.resolved_revision = base.resolved_revision
        self.backbone = (MiddleBlockBackbone(base, m.loops, start=m.recurrent_start,
                             end=m.recurrent_end, input_mix=m.recurrent_input_mix,
                             update_mix=m.recurrent_update_mix)
                         if m.recurrence_mode == "middle_block" else RecurrentBackbone(base, m.loops))
        if (m.backend == "hf" and m.gradient_checkpointing
                and m.recurrence_mode == "middle_block"):
            # The repeated block dominates retained activations. Retain the
            # one-pass prelude/coda instead of recomputing every decoder layer.
            base.checkpoint_layer_range(m.recurrent_start, m.recurrent_end)
        if m.recurrence_mode == "middle_block":
            self.loop_workspace = nn.Parameter(torch.randn(r.read_slots, base.width) * 0.02)
            self.loop_query_norm = nn.RMSNorm(base.width, eps=1e-5)
        if m.backbone_train_scope == "recurrent_core":
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            layers = base.layers if m.backend == "tiny" else base.lm.base_model.layers
            for layer in layers[m.recurrent_start:m.recurrent_end]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        self.width = base.width
        if m.freeze_backbone:
            for p in self.backbone.base.parameters():
                p.requires_grad_(False)
        # Row 0 is always the key-slot embedding; key_slot_position orders it.
        self.write_slots = nn.Parameter(torch.randn(r.write_slots + 1, self.width) * 0.02)
        self.space_key_dims = tuple(r.key_dims) or (r.key_dim,) * len(r.payload_dims)
        self.value_head = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, self.width))
        self.query_head = nn.Linear(self.width, r.key_dim, bias=False)
        if r.key_interface == "direct":
            # Each space addresses its own projection of writer and query states.
            # Writer key-slot states are nearly collinear, so the heads carry a
            # bias and always run in fp32 (see ``_fp32_head``).
            self.key_head = self.address_maps = self.query_maps = None
            self.writer_key_heads = nn.ModuleList([
                nn.Linear(self.width, dim) for dim in self.space_key_dims])
            self.query_key_heads = nn.ModuleList([
                nn.Linear(self.width, dim) for dim in self.space_key_dims])
            # Fixed input standardization for well-conditioned head training.
            # Identity by default; set with ``set_key_state_statistics``.
            for side in ('writer', 'query'):
                self.register_buffer(f'{side}_key_state_mean', torch.zeros(self.width))
                self.register_buffer(f'{side}_key_state_scale', torch.ones(self.width))
        else:
            self.key_head = nn.Linear(self.width, r.key_dim, bias=False)
            self.address_maps = nn.ModuleList([nn.Linear(r.key_dim, r.key_dim, bias=False) for _ in r.payload_dims])
            self.query_maps = nn.ModuleList([nn.Linear(r.key_dim, r.key_dim, bias=False) for _ in r.payload_dims])
            self.writer_key_heads = self.query_key_heads = None
        self.distance_gates = nn.ModuleList([
            AdaptiveDistanceGate(
                dim, density_k=r.gate_density_k,
                min_temperature=r.gate_min_temperature,
                max_temperature=r.gate_max_temperature,
                initial_temperature=r.gate_initial_temperature,
                max_radius_adjustment=r.gate_max_radius_adjustment,
                density_fraction=r.gate_density_fraction,
                floor_mode=r.gate_floor_mode,
            ) for dim in self.space_key_dims
        ]) if r.distance_gating else None
        canonical_dim = r.write_slots * self.width
        if r.payload_layout == "joint_tokens":
            if any(d != tokens * self.width
                   for d, tokens in zip(r.payload_dims, r.space_tokens, strict=True)):
                raise ValueError("Joint token payloads must store full decoder-width tokens")
            self.codecs = nn.ModuleList([JointTokenCodec(self.width, tokens)
                                         for tokens in r.space_tokens])
        else:
            self.codecs = nn.ModuleList([
                PositionalCodec(self.width, d // r.write_slots)
                for d in r.payload_dims
            ] if r.payload_layout == "positional" else [
                nn.Identity() if d == canonical_dim else nn.Linear(canonical_dim, d)
                for d in r.payload_dims
            ])
        options = dict(width=r.reader_width, slots=r.read_slots, rounds=r.reader_rounds,
                       kind=r.reader, chunk_size=r.chunk_size, checkpoint_chunks=r.checkpoint_chunks)
        if r.payload_layout in {"positional", "joint_tokens"}:
            stored_positions = (list(r.space_tokens) if r.payload_layout == "joint_tokens"
                                else [r.write_slots] * len(r.payload_dims))
            self.reader = (PositionalSetReader(r.payload_dims[0], stored_positions[0],
                                               r.key_dim, self.width, **options)
                           if len(r.payload_dims) == 1 else
                           MultiSpacePositionalReader(r.payload_dims, stored_positions,
                                                      r.key_dim, self.width, **options))
        else:
            self.reader = (SetReader(r.payload_dims[0], r.key_dim, self.width, **options)
                           if len(r.payload_dims) == 1 else
                           MultiSpaceReader(r.payload_dims, r.key_dim, self.width, **options))
        if config.train.arm == "direct_latent" and r.payload_dims != [canonical_dim]:
            raise ValueError("direct_latent requires a single identity-width stored payload")
        self.compactor = ((PositionalCompactor(r.payload_dims[0], r.write_slots,
                                               r.reader_width, r.compact_records,
                                               r.reader_rounds, r.chunk_size)
                           if r.payload_layout == "positional" else
                           SyntheticCompactor(r.payload_dims[0], r.reader_width, r.compact_records))
                          if r.compaction == "synthetic" else None)
        self.memory_norm = nn.LayerNorm(self.width)
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))
        self.routing_query_head = copy.deepcopy(self.query_head) if r.independent_routing_query else None
        self.read_count_head = None  # Optional separately trained, frozen inference policy.
        self.read_count_choices = ()

    def writer_slot_embeddings(self) -> Tensor:
        """Writer slot embeddings in sequence order (key slot first or last)."""
        if self.config.memory.key_slot_position == 'last':
            return torch.cat((self.write_slots[1:], self.write_slots[:1]), 0)
        return self.write_slots

    def split_writer_states(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return (key state, value states) from sequence-ordered writer slots."""
        if self.config.memory.key_slot_position == 'last':
            return states[:, -1], states[:, :-1]
        return states[:, 0], states[:, 1:]

    @property
    def device(self) -> torch.device:
        return self.write_slots.device

    def requested_records(self, query: Tensor, maximum: int) -> int:
        """A frozen causal policy may request fewer records than the configured cap."""
        if self.read_count_head is None:
            return maximum
        if len(self.config.memory.payload_dims) != 1 or self.config.memory.read_steps != 1:
            raise ValueError('Read-count policy requires one space and one complete read')
        if not self.read_count_choices:
            raise ValueError('Read-count policy has no declared choices')
        logits = self.read_count_head(F.normalize(query.float(), dim=-1))
        if logits.numel() != len(self.read_count_choices):
            raise ValueError('Read-count policy expects one query')
        return min(maximum, self.read_count_choices[int(logits.argmax(-1).item())])

    def text_ids(self, text: str, *, source: bool = False) -> Tensor:
        limit = self.config.train.max_source_tokens if source else self.config.train.max_prompt_tokens
        ids = self.tokenizer.encode(text, add_special_tokens=source)
        if not ids:
            ids = [self.tokenizer.bos_token_id or 1]
        # Fail explicitly rather than silently deleting a decisive rule/identifier.
        if len(ids) > limit:
            raise ValueError(f"Text uses {len(ids)} tokens, exceeding configured limit {limit}")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def prompt_ids(self, query: str, support_text: str = "") -> Tensor:
        from .text import render_prompt
        text = render_prompt(self.tokenizer, query, support_text)
        return self.text_ids(text)

    def _fp32_head(self, head: nn.Linear, states: Tensor, side: str) -> Tensor:
        """Direct key heads resolve small differences between collinear states.

        BF16 autocast rounds such scores to a handful of values, so these heads
        always compute in fp32 over fixed-standardized inputs; stored keys are
        serialized as fp32 anyway.
        """
        mean = getattr(self, f'{side}_key_state_mean')
        scale = getattr(self, f'{side}_key_state_scale')
        with torch.autocast(states.device.type, enabled=False):
            return head((states.float() - mean) / scale)

    @torch.no_grad()
    def set_key_state_statistics(self, side: str, mean: Tensor, scale: Tensor) -> None:
        """Change standardization without changing any key or address.

        ``W((h - m)/s) + b`` is re-expressed for new statistics ``(m', s')`` so
        every head computes exactly the same affine function of ``h``.
        """
        if self.writer_key_heads is None or side not in {'writer', 'query'}:
            raise ValueError('Key-state statistics apply to direct writer or query heads')
        old_mean = getattr(self, f'{side}_key_state_mean')
        old_scale = getattr(self, f'{side}_key_state_scale')
        mean, scale = mean.to(old_mean), scale.to(old_scale).clamp_min(1e-6)
        heads = self.writer_key_heads if side == 'writer' else self.query_key_heads
        for head in heads:
            raw_weight = head.weight / old_scale
            raw_bias = head.bias - raw_weight @ old_mean
            head.weight.copy_(raw_weight * scale)
            head.bias.copy_(raw_bias + raw_weight @ mean)
        old_mean.copy_(mean)
        old_scale.copy_(scale)

    def writer_space_keys(self, key_states: Tensor) -> tuple[Tensor, ...]:
        """Normalized per-space stored keys from writer key-slot states."""
        if self.writer_key_heads is not None:
            return tuple(F.normalize(self._fp32_head(head, key_states, 'writer'), dim=-1)
                         for head in self.writer_key_heads)
        key = F.normalize(self.key_head(key_states), dim=-1)
        return tuple(F.normalize(address(key), dim=-1) for address in self.address_maps)

    def routing_input(self, features: Tensor, query: Tensor) -> Tensor:
        """What per-space query addressing consumes for these causal features.

        Shared maps consume the normalized routing query. Direct heads consume
        the normalized query-state features themselves.
        """
        if self.query_key_heads is not None:
            return features
        if self.routing_query_head is None:
            return query
        return F.normalize(self.routing_query_head(features), dim=-1)

    def routing_address(self, routing_input: Tensor, space: int) -> Tensor:
        """Unnormalized search address; cosine scoring normalizes it."""
        if self.query_key_heads is not None:
            if routing_input.shape[-1] != self.width:
                raise ValueError('Direct query key heads need routing features')
            return self._fp32_head(self.query_key_heads[space], routing_input, 'query')
        if routing_input.shape[-1] != self.config.memory.key_dim:
            raise ValueError('Shared query maps need a normalized routing query')
        return self.query_maps[space](routing_input)

    def routing_addresses(self, routing_input: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.routing_address(routing_input, space)
                     for space in range(len(self.config.memory.payload_dims)))

    def produce(self, source_ids: Tensor) -> tuple[Tensor, ...]:
        """One key and fixed canonical slots, then per-space keys/stored payloads.

        Returns (key_s0, payload_s0, key_s1, payload_s1, ...). Capturing this
        boundary replays the writer AND storage transforms, not just raw values.
        """
        tokens = self.backbone.embed(source_ids)
        embeddings = torch.cat((tokens, self.writer_slot_embeddings()[None].expand(
            tokens.shape[0], -1, -1)), 1)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                                      loops=self.config.model.writer_loops)
        tail = hidden[:, -(self.config.memory.write_slots + 1):]
        key_state, value_states = self.split_writer_states(tail)
        keys = self.writer_space_keys(key_state)
        canonical = self.value_head(value_states)
        output = []
        for key, codec in zip(keys, self.codecs, strict=True):
            encoded = codec(canonical if self.config.memory.payload_layout in ('positional', 'joint_tokens')
                            else canonical.flatten(1))
            output.extend((key, encoded))
        return tuple(output)

    def produce_batch(self, source_ids: list[Tensor], *,
                      payload_rows: Tensor | None = None,
                      with_key_state: bool = False) -> tuple[Tensor, ...]:
        """Write variable-length sources in one padded native-backbone call.

        Padding follows every source's write slots, so causal source and slot
        positions match the corresponding unpadded writer execution.  Keys are
        always produced.  ``payload_rows`` may omit payload projection for rows
        that cannot be selected by the fixed oracle plan.
        """
        if not source_ids or any(ids.ndim != 2 or ids.shape[0] != 1 for ids in source_ids):
            raise ValueError('produce_batch expects nonempty single-row token tensors')
        width, slots = self.width, self.config.memory.write_slots + 1
        lengths = torch.tensor([ids.shape[1] for ids in source_ids], device=self.device)
        rows = []
        for ids in source_ids:
            tokens = self.backbone.embed(ids)
            row = torch.cat((tokens, self.writer_slot_embeddings()[None]), 1)
            rows.append(F.pad(row, (0, 0, 0, int(lengths.max()) + slots - row.shape[1])))
        embeddings = torch.cat(rows, 0)
        positions = torch.arange(embeddings.shape[1], device=self.device)[None]
        mask = positions < (lengths + slots)[:, None]
        hidden = self.backbone.hidden(embeddings, mask.long(),
                                      loops=self.config.model.writer_loops)
        indices = lengths[:, None] + torch.arange(slots, device=self.device)[None]
        tail = hidden.gather(1, indices[..., None].expand(-1, -1, width))
        key_state, value_states = self.split_writer_states(tail)
        keys = self.writer_space_keys(key_state)
        if payload_rows is None:
            payload_rows = torch.ones(len(source_ids), dtype=torch.bool, device=self.device)
        if payload_rows.shape != (len(source_ids),) or payload_rows.dtype != torch.bool:
            raise ValueError('payload_rows must be one boolean per source')
        canonical = self.value_head(value_states[payload_rows])
        output = []
        for key, codec, dim in zip(keys, self.codecs,
                                   self.config.memory.payload_dims, strict=True):
            if canonical.shape[0]:
                encoded = codec(canonical if self.config.memory.payload_layout in ('positional', 'joint_tokens')
                                else canonical.flatten(1))
                payload = encoded.new_zeros(len(source_ids), dim).index_copy(
                    0, payload_rows.nonzero().flatten(), encoded)
            else:
                payload = canonical.new_zeros(len(source_ids), dim)
            output.extend((key, payload))
        if with_key_state:
            # The writer's key-slot hidden state, which direct key heads consume.
            output.append(key_state)
        return tuple(output)

    def produce_from_write_states(self, states: Tensor) -> tuple[Tensor, ...]:
        """Project causal key/value slot states embedded in an agent trajectory."""
        slots = self.config.memory.write_slots + 1
        if states.ndim != 3 or states.shape[1:] != (slots, self.width):
            raise ValueError('Trajectory write states have incompatible slots or width')
        key_state, value_states = self.split_writer_states(states)
        keys = self.writer_space_keys(key_state)
        canonical = self.value_head(value_states)
        output = []
        for key, codec in zip(keys, self.codecs, strict=True):
            encoded = codec(canonical if self.config.memory.payload_layout in ('positional', 'joint_tokens')
                            else canonical.flatten(1))
            output.extend((key, encoded))
        return tuple(output)

    def _query_features(self, prompt_ids: Tensor, memory: Tensor | None = None) -> Tensor:
        embeddings = self._context(prompt_ids, memory)
        # Query uses ONLY the prompt, never teacher-forced target tokens.
        h = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long), loops=1)
        return h[:, -1]

    def query(self, prompt_ids: Tensor, memory: Tensor | None = None) -> Tensor:
        """Reader conditioning; use query_pair()[1] for configurable addressing."""
        return F.normalize(self.query_head(self._query_features(prompt_ids, memory)), dim=-1)

    def query_pair(self, prompt_ids: Tensor, memory: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Reader conditioning and one retrieval key, both from the causal prefix."""
        if self.query_key_heads is None and self.routing_query_head is None:
            query = self.query(prompt_ids, memory)
            return query, query
        features = self._query_features(prompt_ids, memory)
        query = F.normalize(self.query_head(features), dim=-1)
        return query, self.routing_input(features, query)

    def _prepare_values(self, payloads: list[Tensor], ablate_values: bool = False,
                        weights: list[Tensor] | None = None):
        r = self.config.memory
        values = [v[None] for v in payloads]
        if ablate_values:
            values = [torch.zeros_like(v) for v in values]
        if self.training:
            values = [storage_noise(v, noise_std=r.noise_std, quantization_step=r.quantization_step)
                      for v in values]
        if weights is None:
            prepared_weights = [v.new_ones(v.shape[:2]) for v in values]
        else:
            if len(weights) != len(values):
                raise ValueError('One relevance-weight vector is required per memory space')
            prepared_weights = []
            for value, weight in zip(values, weights, strict=True):
                if weight.ndim != 1 or weight.shape[0] != value.shape[1]:
                    raise ValueError('Relevance weights differ from selected payloads')
                prepared_weights.append(weight[None].to(device=value.device, dtype=value.dtype))
        return values, prepared_weights

    def _compact_values(self, value: Tensor, weights: Tensor):
        r = self.config.memory
        view = compact_view(value, weights, method=r.compaction, compactor=self.compactor,
                            grouping=r.compaction_grouping, group_size=r.compaction_group_size,
                            memberships=r.field_memberships)
        if r.read_timing == "loop_boundary":
            # Match offline full-cluster code values and FP32 multiplicities.
            # This differentiable cast is part of the consumer/replay graph.
            view.values = view.values.to(getattr(torch, r.storage_dtype)).float()
            view.weights = view.weights.float()
        return view

    def read_tokens(self, payloads: list[Tensor], query: Tensor, *, compact: bool = False,
                    ablate_values: bool = False,
                    weights: list[Tensor] | None = None) -> tuple[Tensor, Tensor]:
        values, weights = self._prepare_values(payloads, ablate_values, weights)
        auxiliary = query.sum() * 0
        if isinstance(self.reader, (MultiSpaceReader, MultiSpacePositionalReader)):
            return self.reader(values, query, weights), auxiliary
        if compact and values[0].shape[1] > 0:
            view = self._compact_values(values[0], weights[0])
            auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
            return self.reader(view.values, query, view.weights).tokens, auxiliary
        if self.training and self.config.memory.merge_loss_weight and values[0].shape[1] > 1:
            r = self.config.memory
            view = compact_view(values[0], weights[0], grouping="local", group_size=r.compaction_group_size)
            auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
        return self.reader(values[0], query, weights[0]).tokens, auxiliary

    def _read_padded_batch(self, payloads: list[Tensor], weights: list[Tensor],
                           query: Tensor) -> Tensor:
        if isinstance(self.reader, (MultiSpaceReader, MultiSpacePositionalReader)):
            return self.reader(payloads, query, weights)
        return self.reader(payloads[0], query, weights[0]).tokens

    def forward_loop_memory_batch(self, prompts: list[Tensor], targets: list[Tensor],
                                  records: list[list[tuple[Tensor, ...]]],
                                  required: list[list[int]], *, step: int = 0) -> ForwardResult:
        """Padded multi-example form of the native R=2, one-read training path."""
        r, t = self.config.memory, self.config.train
        batch = len(prompts)
        if (batch < 1 or len(targets) != batch or len(records) != batch or len(required) != batch
                or self.backbone.loops != 2 or r.read_steps != 1 or r.compaction != 'none'):
            raise ValueError('Batched native training requires aligned inputs, R=2, one raw read')
        prompt_lengths = torch.tensor([x.shape[1] for x in prompts], device=self.device)
        target_lengths = torch.tensor([x.shape[1] for x in targets], device=self.device)
        slot_count = r.read_slots
        sequences = []
        for prompt, target in zip(prompts, targets, strict=True):
            sequences.append(torch.cat((self.backbone.embed(prompt),
                                        self.loop_workspace[None],
                                        self.backbone.embed(target[:, :-1])), 1))
        total_lengths = prompt_lengths + slot_count + target_lengths - 1
        maximum = int(total_lengths.max())
        embeddings = torch.cat([F.pad(row, (0, 0, 0, maximum - row.shape[1]))
                                for row in sequences], 0)
        mask = torch.arange(maximum, device=self.device)[None] < total_lengths[:, None]
        selected = [[[] for _ in r.payload_dims] for _ in range(batch)]
        routing = self.memory_gate * 0
        oracle = t.retrieval == 'oracle' or (self.training and step < t.routing_warmup)
        read_count = 0

        def boundary(completed, state, _anchor):
            nonlocal routing, read_count
            query_positions = prompt_lengths + slot_count - 1
            query_state = state[torch.arange(batch, device=self.device), query_positions]
            features = self.loop_query_norm(query_state)
            query = F.normalize(self.query_head(features), dim=-1)
            routing_query = self.routing_input(features, query)
            payloads, weights = [], []
            for space, dim in enumerate(r.payload_dims):
                chosen_rows = []
                local_routing = routing * 0
                for row in range(batch):
                    keys = (torch.cat([item[2 * space] for item in records[row]], 0)
                            if records[row] else query.new_empty(0, r.key_dim))
                    scores = cosine_scores(self.routing_address(routing_query[row:row + 1], space), keys)[0]
                    if t.retrieval == 'learned' and required[row]:
                        local_routing = local_routing + group_plan_loss(scores, [tuple(required[row])])
                    choice = (required[row] if oracle else
                              scores.argsort(descending=True, stable=True)[:self.requested_records(
                                  query[row:row + 1], r.neighbors[space])].tolist())
                    selected[row][space].extend(choice)
                    chosen_rows.append(torch.cat([records[row][i][2 * space + 1] for i in choice], 0)
                                       if choice else query.new_empty(0, dim))
                routing = routing + local_routing / (batch * len(r.payload_dims))
                count = max((x.shape[0] for x in chosen_rows), default=0)
                payloads.append(torch.cat([F.pad(x, (0, 0, 0, count - x.shape[0]))[None]
                                           for x in chosen_rows], 0))
                weights.append(torch.cat([torch.cat((query.new_ones(x.shape[0]),
                                                     query.new_zeros(count - x.shape[0])))[None]
                                          for x in chosen_rows], 0))
            read_count += 1
            memory = self._read_padded_batch(payloads, weights, query)
            return LoopWrite(prompt_lengths, memory)

        hidden = self.backbone.hidden(embeddings, mask.long(), boundary=boundary)
        max_target = int(target_lengths.max())
        logits_rows, label_rows, state_rows = [], [], []
        for row, target in enumerate(targets):
            start = int(prompt_lengths[row]) + slot_count - 1
            answer_state = hidden[row:row + 1, start:start + target.shape[1]]
            logits = self.backbone.logits(answer_state).float()
            logits_rows.append(F.pad(logits, (0, 0, 0, max_target - target.shape[1])))
            label_rows.append(F.pad(target, (0, max_target - target.shape[1]), value=-100))
            state_rows.append(F.pad(answer_state, (0, 0, 0, max_target - target.shape[1])))
        logits = torch.cat(logits_rows, 0)
        labels = torch.cat(label_rows, 0)
        token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        nll = (token_loss.sum(1) / target_lengths).mean()
        routing = routing / max(1, read_count)
        return ForwardResult(nll + t.routing_weight * routing, nll, routing,
                             self.memory_gate * 0, selected[0], raw_nll=nll,
                             read_count=read_count, answer_states=torch.cat(state_rows, 0))

    def read_pair(self, payloads: list[Tensor], query: Tensor, *, ablate_values: bool = False):
        """Raw and compact paths see exactly the same noisy values and selection."""
        if isinstance(self.reader, (MultiSpaceReader, MultiSpacePositionalReader)):
            raise ValueError("Paired compaction currently requires one space")
        values, weights = self._prepare_values(payloads, ablate_values)
        raw = self.reader(values[0], query, weights[0]).tokens
        view = self._compact_values(values[0], weights[0])
        compact = self.reader(view.values, query, view.weights).tokens
        auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
        return raw, compact, auxiliary

    def shared_compute_tokens(self, prompt: Tensor) -> Tensor | LoopMemory:
        """Same query/reader/soft-slot path, no external information or payload reads."""
        if self.config.memory.read_timing == "loop_boundary":
            last = None
            def provider(completed, query):
                nonlocal last
                if completed <= self.config.memory.read_steps:
                    payloads = [query.new_zeros(n, d) for n, d in
                                zip(self.config.memory.neighbors, self.config.memory.payload_dims, strict=True)]
                    last, _ = self.read_tokens(payloads, query)
                return last
            return self.plan_loop_memory(prompt, provider)
        query = self.query(prompt)
        # Several records prevent a null-read shortcut while carrying no source data.
        payloads = [query.new_zeros(max(1, n), d) for n, d in
                    zip(self.config.memory.neighbors, self.config.memory.payload_dims, strict=True)]
        return self.read_tokens(payloads, query)[0]

    def _context(self, prompt: Tensor, memory: Tensor | LoopMemory | None) -> Tensor:
        context = self.backbone.embed(prompt)
        if isinstance(memory, LoopMemory):
            if memory.prefix_length != prompt.shape[1] or memory.slots != self.config.memory.read_slots:
                raise ValueError("Captured loop-memory layout does not match the consumer prefix")
            return torch.cat((context, self.loop_workspace[None].expand(prompt.shape[0], -1, -1)), 1)
        if memory is not None and memory.shape[1] > 0:
            scale = context.detach().float().square().mean().sqrt().clamp_min(1e-3)
            tokens = self.memory_norm(memory) * scale.to(memory.dtype) * self.memory_gate.sigmoid()
            context = torch.cat((context, tokens.to(context.dtype)), 1)
        return context

    def conditioned_states(self, prompt: Tensor, target: Tensor, memory: Tensor | LoopMemory | None,
                           *, loops: int | None = None) -> Tensor:
        if (isinstance(memory, Tensor) and self.config.memory.read_timing == "loop_boundary"):
            # Explicit fixed-result comparison; normal training/read_session uses
            # state-conditioned LoopMemory rather than this convenience path.
            count = self.backbone.loops if loops is None else loops
            memory = LoopMemory(prompt.shape[1], self.config.memory.read_slots, count,
                                tuple(memory for _ in range(count - 1)))
        context = self._context(prompt, memory)
        embeddings = torch.cat((context, self.backbone.embed(target[:, :-1])), 1)
        kwargs = dict(loops=loops)
        if isinstance(memory, LoopMemory):
            if loops is not None and loops != memory.loops:
                raise ValueError("Changing depth requires regenerating the prefix-only read plan")
            kwargs = dict(loops=memory.loops, boundary=memory.callback)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long), **kwargs)
        return hidden[:, context.shape[1] - 1:]

    def conditioned_logits(self, prompt: Tensor, target: Tensor, memory: Tensor | LoopMemory | None,
                           *, loops: int | None = None) -> Tensor:
        return self.backbone.logits(self.conditioned_states(prompt, target, memory, loops=loops)).float()

    def conditioned_nll(self, prompt: Tensor, target: Tensor, memory: Tensor | LoopMemory | None,
                        *, reduction: str = "mean", loops: int | None = None) -> Tensor:
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Invalid NLL reduction")
        logits = self.conditioned_logits(prompt, target, memory, loops=loops)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction=reduction)

    def loop_query(self, state: Tensor, prefix_length: int) -> Tensor:
        # The last reserved slot is strictly before every teacher-forced target.
        index = prefix_length + self.config.memory.read_slots - 1
        return F.normalize(self.query_head(self.loop_query_norm(state[:, index])), dim=-1)

    def loop_query_pair(self, state: Tensor, prefix_length: int) -> tuple[Tensor, Tensor]:
        if self.query_key_heads is None and self.routing_query_head is None:
            query = self.loop_query(state, prefix_length)
            return query, query
        index = prefix_length + self.config.memory.read_slots - 1
        features = self.loop_query_norm(state[:, index])
        query = F.normalize(self.query_head(features), dim=-1)
        return query, self.routing_input(features, query)

    def begin_spatial_recurrent(self, input_ids: Tensor, attention_mask: Tensor,
                                sites: tuple[SpatialReadSite, ...]) -> SpatialExecution:
        """Run through the first core pass and retain its differentiable state."""
        if not isinstance(self.backbone, MiddleBlockBackbone) or not sites:
            raise ValueError("Spatial reads require a middle-block backbone and sites")
        if (input_ids.ndim != 2 or attention_mask.shape != input_ids.shape
                or input_ids.shape[0] < 1):
            raise ValueError("Spatial token IDs and attention mask must be aligned batches")
        batch, length = input_ids.shape
        occupied = torch.zeros_like(input_ids, dtype=torch.bool)
        for site in sites:
            if (site.query_positions.shape != (batch,) or site.workspace_starts.shape != (batch,)
                    or site.query_positions.dtype not in (torch.int32, torch.int64)
                    or site.workspace_starts.dtype not in (torch.int32, torch.int64)
                    or not 1 <= site.level < self.backbone.loops):
                raise ValueError("Invalid spatial site batch or recurrence level")
            query_positions = site.query_positions.to(input_ids.device)
            starts = site.workspace_starts.to(input_ids.device)
            indices = starts[:, None] + torch.arange(
                self.config.memory.read_slots, device=input_ids.device)[None]
            if (bool((query_positions < 0).any()) or bool((query_positions >= length).any())
                    or bool((indices < 0).any()) or bool((indices >= length).any())
                    or bool((query_positions >= starts).any())):
                raise ValueError("Every spatial workspace must follow its in-range query")
            if bool((attention_mask[torch.arange(batch, device=input_ids.device), query_positions] == 0).any()):
                raise ValueError("Spatial query positions must be visible")
            if bool(occupied.gather(1, indices).any()):
                raise ValueError("Spatial workspaces must not overlap")
            occupied.scatter_(1, indices, True)
        if bool(((input_ids == -1) != occupied).any()) or bool((input_ids < -2).any()):
            raise ValueError("Blank token positions must exactly match spatial workspaces")
        if bool((attention_mask[occupied] == 0).any()):
            raise ValueError("Spatial workspaces must be visible sequence positions")

        embeddings = self.backbone.embed(input_ids.clamp_min(0)).clone()
        rows = torch.arange(batch, device=input_ids.device)[:, None]
        slots = torch.arange(self.config.memory.read_slots, device=input_ids.device)[None]
        for site in sites:
            indices = site.workspace_starts.to(input_ids.device)[:, None] + slots
            embeddings[rows, indices] = self.loop_workspace[None]
        writer_slots = self.config.memory.write_slots + 1
        for row in range(batch):
            positions = (input_ids[row] == -2).nonzero().flatten()
            if positions.numel() % writer_slots:
                raise ValueError('Trajectory write workspaces have incomplete slot groups')
            for start in range(0, positions.numel(), writer_slots):
                group = positions[start:start + writer_slots]
                if (group[-1] - group[0] + 1).item() != writer_slots:
                    raise ValueError('Trajectory write slots must be contiguous')
                embeddings[row, group] = self.writer_slot_embeddings()

        recurrent = self.backbone.begin(embeddings, attention_mask)
        return SpatialExecution(recurrent, sites, batch)

    def spatial_recurrent_query(self, execution: SpatialExecution):
        """Return all queries at the current level, or ``None`` for an empty level."""
        recurrent, batch = execution.recurrent, execution.batch
        if recurrent.completed >= recurrent.loops:
            raise ValueError("A completed spatial execution has no pending query")
        active = tuple(site for site in execution.sites
                       if site.level == recurrent.completed)
        if not active:
            return None
        state = recurrent.state
        features = torch.cat([
            state[torch.arange(batch, device=state.device),
                  site.query_positions.to(state.device)] for site in active
        ], 0)
        features = self.loop_query_norm(features)
        query = F.normalize(self.query_head(features), dim=-1)
        return active, query, self.routing_input(features, query)

    def advance_spatial_recurrent(self, execution: SpatialExecution,
                                  active: tuple[SpatialReadSite, ...],
                                  results: Tensor) -> SpatialExecution:
        """Consume one complete same-level result wave and run the next core pass."""
        expected_active = tuple(site for site in execution.sites
                                if site.level == execution.recurrent.completed)
        if (not active or len(active) != len(expected_active)
                or any(got is not expected for got, expected in
                       zip(active, expected_active, strict=True))):
            raise ValueError("Spatial results differ from the pending recurrence level")
        expected = (len(active) * execution.batch,
                    self.config.memory.read_slots, self.width)
        if not isinstance(results, Tensor) or results.shape != expected:
            raise ValueError(
                f"Spatial provider returned {getattr(results, 'shape', None)}; "
                f"expected {expected}")
        writes = tuple(LoopWrite(
            site.workspace_starts.to(execution.recurrent.state.device),
            results[index * execution.batch:(index + 1) * execution.batch],
        ) for index, site in enumerate(active))
        recurrent = self.backbone.advance(execution.recurrent, LoopWrites(writes))
        return SpatialExecution(recurrent, execution.sites, execution.batch)

    def advance_empty_spatial_level(self, execution: SpatialExecution) -> SpatialExecution:
        """Advance a recurrence level that has no read sites."""
        if any(site.level == execution.recurrent.completed for site in execution.sites):
            raise ValueError("A spatial level with active reads needs their results")
        recurrent = self.backbone.advance(execution.recurrent)
        return SpatialExecution(recurrent, execution.sites, execution.batch)

    def finish_spatial_recurrent(self, execution: SpatialExecution) -> Tensor:
        return self.backbone.finish(execution.recurrent)

    def spatial_recurrent_hidden(self, input_ids: Tensor, attention_mask: Tensor,
                                 sites: tuple[SpatialReadSite, ...], provider) -> Tensor:
        """Run a whole sequence with synchronous batched reads as the reference path."""
        execution = self.begin_spatial_recurrent(input_ids, attention_mask, sites)
        while execution.recurrent.completed < execution.recurrent.loops:
            pending = self.spatial_recurrent_query(execution)
            if pending is None:
                execution = self.advance_empty_spatial_level(execution)
                continue
            active, query, routing_query = pending
            results = provider(execution.recurrent.completed, active, query, routing_query)
            execution = self.advance_spatial_recurrent(execution, active, results)
        return self.finish_spatial_recurrent(execution)

    def plan_loop_memory(self, prompt: Tensor, provider, *, include_routing_query: bool = False) -> LoopMemory:
        """Run ONLY the available prefix, collecting state-conditioned read results.

        provider(completed_core_passes, reader_query) may read stored payloads but must
        never access a target. Captured results can be replayed for different answer
        candidates, preserving a common read plan without calling the writer.
        include_routing_query supplies a third provider argument for address search.
        """
        if not isinstance(self.backbone, MiddleBlockBackbone):
            raise ValueError("A middle-block backbone is required")
        count, length = self.backbone.loops, prompt.shape[1]
        blank = LoopMemory(length, self.config.memory.read_slots, count, (None,) * (count - 1))
        context, events = self._context(prompt, blank), []
        def boundary(completed, state, _anchor):
            query, routing_query = self.loop_query_pair(state, length)
            tokens = (provider(completed, query, routing_query) if include_routing_query
                      else provider(completed, query))
            events.append(tokens)
            return None if tokens is None else LoopWrite(length, tokens)
        if count > 1:
            self.backbone.hidden(context, torch.ones(context.shape[:2], dtype=torch.long, device=self.device),
                                 boundary=boundary, plan_only=True)
        return LoopMemory(length, blank.slots, count, tuple(events))

    def forward_loop_memory(self, prompt: Tensor, target: Tensor,
                            records: list[tuple[Tensor, ...]], required: list[int], *,
                            step: int = 0, ablate_values: bool = False,
                            shared_compute: bool = False, compact: bool = False) -> ForwardResult:
        """Single integrated consumer graph: compute -> retrieve -> inject -> compute.

        Only prefix states determine queries, even though causal teacher-forced
        continuation positions execute in parallel. Inference captures the same
        read events with a prefix-only execution and replays them while scoring.
        """
        r, t = self.config.memory, self.config.train
        count, length = self.backbone.loops, prompt.shape[1]
        if count < 2:
            raise ValueError("In-loop memory training needs at least two core passes")
        selected = [[] for _ in r.payload_dims]
        paired = compact and r.compaction_objective == 'paired'
        compact_memory, compact_events = None, []
        routing, memory, read_count = self.memory_gate * 0, None, 0
        auxiliary = self.memory_gate * 0
        oracle = t.retrieval == "oracle" or (self.training and step < t.routing_warmup)
        blank = LoopMemory(length, r.read_slots, count, (None,) * (count - 1))
        context = self._context(prompt, blank)
        embeddings = torch.cat((context, self.backbone.embed(target[:, :-1])), 1)
        def boundary(completed, state, _anchor):
            nonlocal memory, compact_memory, routing, read_count, auxiliary
            if completed <= r.read_steps:
                query, routing_query = self.loop_query_pair(state, length)
                payloads, progressed = [], False
                for space, dim in enumerate(r.payload_dims):
                    if shared_compute:
                        payloads.append(query.new_zeros(r.neighbors[space], dim))
                        progressed = True
                        continue
                    candidates = [i for i in range(len(records)) if i not in selected[space]]
                    if candidates:
                        keys = torch.cat([records[i][2 * space] for i in candidates], 0)
                        scores = cosine_scores(self.routing_address(routing_query, space), keys)[0]
                        remaining = [candidates.index(i) for i in required if i in candidates]
                        if t.retrieval == "learned" and remaining:
                            routing = routing + group_plan_loss(scores, [tuple(remaining)]) / len(r.payload_dims)
                        k = self.requested_records(query, r.neighbors[space] if r.read_steps == 1 else r.read_top_k)
                        chosen = (remaining if r.read_steps == 1 else remaining[:k]) if oracle else scores.argsort(descending=True, stable=True)[:k].tolist()
                        selected[space].extend(candidates[i] for i in chosen)
                        progressed = progressed or bool(chosen)
                    values = [records[i][2 * space + 1] for i in selected[space]]
                    payloads.append(torch.cat(values, 0) if values else query.new_empty(0, dim))
                if progressed:
                    if paired:
                        memory, compact_memory, local_auxiliary = self.read_pair(payloads, query,
                                                                               ablate_values=ablate_values)
                    else:
                        memory, local_auxiliary = self.read_tokens(payloads, query, compact=compact,
                                                                  ablate_values=ablate_values)
                    auxiliary = auxiliary + local_auxiliary
                    read_count += 1
            if paired:
                compact_events.append(compact_memory)
            return None if memory is None else LoopWrite(length, memory)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                                      boundary=boundary)
        logits = self.backbone.logits(hidden[:, context.shape[1] - 1:]).float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        routing = routing / max(1, read_count)
        auxiliary = auxiliary / max(1, read_count)
        if paired and read_count:
            # A single first-boundary query/selection and noisy value set feed
            # both paths. Sharing this causal prefix expression accumulates both
            # cotangents through the common query/backbone graph before replay.
            compact_plan = LoopMemory(length, r.read_slots, count, tuple(compact_events))
            compact_logits = self.conditioned_logits(prompt, target, compact_plan)
            compact_nll = F.cross_entropy(compact_logits.reshape(-1, compact_logits.shape[-1]), target.reshape(-1))
            behavior = F.kl_div(compact_logits.log_softmax(-1), logits.detach().softmax(-1),
                                reduction='none').sum(-1).mean()
            loss = (nll + r.compact_task_weight * compact_nll + r.compaction_loss_weight * auxiliary
                    + r.behavior_kl_weight * behavior + t.routing_weight * routing)
            return ForwardResult(loss, nll, routing, auxiliary, selected, raw_nll=nll,
                                 compact_nll=compact_nll, behavior_kl=behavior, read_count=read_count)
        weight = r.compaction_loss_weight if compact else r.merge_loss_weight
        return ForwardResult(nll + t.routing_weight * routing + weight * auxiliary,
                             nll, routing, auxiliary, selected, raw_nll=None if compact else nll,
                             compact_nll=nll if compact else None, read_count=read_count,
                             answer_states=(hidden[:, context.shape[1] - 1:]
                                            if t.oracle_alignment_weight or t.oracle_distillation_weight else None))

    def target_ids(self, answer: str) -> Tensor:
        ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        if len(ids) > self.config.train.max_target_tokens:
            raise ValueError(f"Target uses {len(ids)} tokens; limit is {self.config.train.max_target_tokens}. Do not silently truncate complete targets.")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def forward(self, prompt: Tensor, target: Tensor, records: list[tuple[Tensor, ...]],
                required: list[int], *, step: int = 0, arm: str | None = None,
                compact: bool = False, ablate_values: bool = False) -> ForwardResult:
        t, r = self.config.train, self.config.memory
        arm = arm or t.arm
        zero = self.memory_gate * 0
        if arm in {"no_memory", "oracle_text"}:
            if self.training and t.parent_kl_weight:
                logits = self.conditioned_logits(prompt, target, None)
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
                # Exactly the frozen text parent: R=1 bypasses all bridge parameters.
                # No second resident model and no gradient through the teacher path.
                with torch.no_grad():
                    teacher = self.conditioned_logits(prompt, target, None, loops=1)
                kl = F.kl_div(logits.log_softmax(-1), teacher.softmax(-1),
                              reduction="none").sum(-1).mean()
                return ForwardResult(nll + t.parent_kl_weight * kl, nll, zero, zero, [],
                                     read_count=0, parent_kl=kl)
            nll = self.conditioned_nll(prompt, target, None)
            return ForwardResult(nll, nll, zero, zero, [], read_count=0)
        if r.read_timing == "loop_boundary":
            return self.forward_loop_memory(prompt, target, records, required, step=step,
                                            ablate_values=ablate_values, shared_compute=arm == "shared_compute",
                                            compact=compact)
        if arm == "shared_compute":
            nll = self.conditioned_nll(prompt, target, self.shared_compute_tokens(prompt))
            return ForwardResult(nll, nll, zero, zero, [], raw_nll=nll)
        if r.read_steps > 1:
            return self.forward_multiread(prompt, target, records, required, step=step)
        q, routing_query = self.query_pair(prompt)
        selected, payloads = [], []
        routing = zero
        for space, dim in enumerate(r.payload_dims):
            keys = torch.cat([record[2 * space] for record in records], 0) if records else q.new_empty(0, r.key_dim)
            values = torch.cat([record[2 * space + 1] for record in records], 0) if records else q.new_empty(0, dim)
            scores = cosine_scores(F.normalize(self.routing_address(routing_query, space), dim=-1), keys)[0]
            if t.retrieval == "learned" and required and records:
                routing = routing + group_plan_loss(scores, [tuple(required)]) / len(r.payload_dims)
            oracle = t.retrieval == "oracle" or (self.training and step < t.routing_warmup)
            indices = required if oracle else torch.argsort(scores, descending=True, stable=True)[:self.requested_records(q, r.neighbors[space])].tolist()
            selected.append(list(indices))
            payloads.append(values[indices])
        present = any(p.shape[0] for p in payloads)
        if compact and present and r.compaction_objective == "paired":
            raw, compact_memory, auxiliary = self.read_pair(payloads, q, ablate_values=ablate_values)
            raw_logits = self.conditioned_logits(prompt, target, raw)
            compact_logits = self.conditioned_logits(prompt, target, compact_memory)
            raw_nll = F.cross_entropy(raw_logits.reshape(-1, raw_logits.shape[-1]), target.reshape(-1))
            compact_nll = F.cross_entropy(compact_logits.reshape(-1, compact_logits.shape[-1]), target.reshape(-1))
            # Forward KL of raw teacher to compact student, averaged per target position.
            behavior = F.kl_div(compact_logits.log_softmax(-1), raw_logits.detach().softmax(-1),
                                reduction="none").sum(-1).mean()
            loss = (raw_nll + r.compact_task_weight * compact_nll +
                    r.compaction_loss_weight * auxiliary + r.behavior_kl_weight * behavior +
                    t.routing_weight * routing)
            return ForwardResult(loss, raw_nll, routing, auxiliary, selected,
                                 raw_nll, compact_nll, behavior)
        if arm == "direct_latent":
            memory = payloads[0].reshape(1, -1, self.width) if present else None
            auxiliary = zero
        elif present:
            memory, auxiliary = self.read_tokens(payloads, q, compact=compact, ablate_values=ablate_values)
        else:
            memory, auxiliary = None, zero
        nll = self.conditioned_nll(prompt, target, memory)
        weight = r.compaction_loss_weight if compact else r.merge_loss_weight
        loss = nll + t.routing_weight * routing + weight * auxiliary
        return ForwardResult(loss, nll, routing, auxiliary, selected)

    def forward_multiread(self, prompt: Tensor, target: Tensor,
                          records: list[tuple[Tensor, ...]], required: list[int], *, step: int = 0):
        """Scheduled read rounds; follow-up keys depend on already received evidence.

        Recompose the cumulative selected set into fixed working slots. No duplicate
        evidence and no producer-graph truncation. Discrete choices are captured by
        selected IDs; query/reader paths through previous rounds remain differentiable.
        """
        r, t = self.config.memory, self.config.train
        zero = self.memory_gate * 0
        selected = [[] for _ in r.payload_dims]
        memory, routing, reads = None, zero, 0
        oracle = t.retrieval == "oracle" or (self.training and step < t.routing_warmup)
        for _ in range(r.read_steps):
            q, routing_query = self.query_pair(prompt, memory)
            payloads, progressed = [], False
            for space, dim in enumerate(r.payload_dims):
                candidates = [i for i in range(len(records)) if i not in selected[space]]
                if candidates:
                    keys = torch.cat([records[i][2 * space] for i in candidates], 0)
                    scores = cosine_scores(self.routing_address(routing_query, space), keys)[0]
                    remaining = [candidates.index(i) for i in required if i in candidates]
                    if self.training and t.retrieval == "learned" and remaining:
                        routing = routing + group_plan_loss(scores, [tuple(remaining)]) / len(r.payload_dims)
                    chosen = (remaining[:r.read_top_k] if oracle else
                              scores.argsort(descending=True, stable=True)[:self.requested_records(q, r.read_top_k)].tolist())
                    selected[space].extend(candidates[i] for i in chosen)
                    progressed = progressed or bool(chosen)
                values = ([records[i][2 * space + 1] for i in selected[space]])
                payloads.append(torch.cat(values, 0) if values else q.new_empty(0, dim))
            if not progressed:
                break
            reads += 1
            memory, _ = self.read_tokens(payloads, q)
        nll = self.conditioned_nll(prompt, target, memory)
        routing = routing / max(1, reads)
        return ForwardResult(nll + t.routing_weight * routing, nll, routing, zero,
                             selected, raw_nll=nll, read_count=reads)

    @torch.no_grad()
    def generate_with_payloads(self, prompt: Tensor, payloads: list[Tensor] | None,
                               max_new_tokens: int = 24) -> str:
        """Reference greedy decoding: recomputes prefixes; no hybrid cache mutation."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        memory = None
        if payloads is not None and any(p.shape[0] for p in payloads):
            if self.config.memory.read_timing == "loop_boundary":
                # Fixed candidate set, but interpreted at actual recurrent states.
                last = None
                def provider(completed, query):
                    nonlocal last
                    if completed <= self.config.memory.read_steps:
                        last, _ = self.read_tokens(payloads, query)
                    return last
                memory = self.plan_loop_memory(prompt, provider)
            else:
                q = self.query(prompt)
                memory, _ = self.read_tokens(payloads, q)
        return self.generate_from_memory(prompt, memory, max_new_tokens)

    @torch.no_grad()
    def generate_from_memory(self, prompt: Tensor, memory: Tensor | LoopMemory | None, max_new_tokens: int = 24) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(memory, Tensor) and self.config.memory.read_timing == "loop_boundary":
            memory = LoopMemory(prompt.shape[1], self.config.memory.read_slots, self.backbone.loops,
                                tuple(memory for _ in range(self.backbone.loops - 1)))
        context = self._context(prompt, memory)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            embeddings = context
            if generated:
                ids = torch.tensor([generated], device=self.device)
                embeddings = torch.cat((context, self.backbone.embed(ids)), 1)
            kwargs = (dict(loops=memory.loops, boundary=memory.callback)
                      if isinstance(memory, LoopMemory) else {})
            hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long), **kwargs)
            next_id = int(self.backbone.logits(hidden[:, -1]).argmax(-1).item())
            if next_id == self.tokenizer.eos_token_id:
                break
            generated.append(next_id)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

# Keep historical checkpoints and external training scripts source compatible.
SDKBAgent = SchnitzelAgent
