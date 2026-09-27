"""Cache-free causal backbone adapters for SCHNITZELJAGD recurrence.

Native conversion exposes prelude, shared middle layers, and coda without changing
the parent's normalization, mask, or positional behavior. The earlier full-stack
adapter remains a comparison. No path shares KV/convolution caches across loops.
This is an experimental conversion, not a claim that LFM was pretrained as looped.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ByteTokenizer:
    """Offline tests only. Production LFM uses its own tokenizer/chat template."""
    pad_token_id, bos_token_id, eos_token_id, vocab_size = 0, 1, 2, 259

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool = False,
                            add_generation_prompt: bool = True, **_kwargs):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant: "
        return self.encode(text, add_special_tokens=True) if tokenize else text


class TinyBackbone(nn.Module):
    def __init__(self, width: int = 64, layers: int = 2, heads: int = 4,
                 max_length: int = 2048) -> None:
        super().__init__()
        self.width, self.max_length = width, max_length
        self.embedding = nn.Embedding(259, width, padding_idx=0)
        self.position = nn.Embedding(max_length, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 3,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.gradient_checkpointing = False
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def embed(self, ids: Tensor) -> Tensor:
        return self.embedding(ids)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def layer_types(self) -> list[str]:
        return ["full_attention"] * self.layer_count

    def prepare_layers(self, embeddings: Tensor, mask: Tensor):
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Tiny backbone context limit exceeded")
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        x = embeddings + self.position(positions)[None]
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        return x, (causal, mask)

    def run_layers(self, hidden: Tensor, context, start: int, end: int) -> Tensor:
        from torch.utils.checkpoint import checkpoint
        causal, mask = context
        for index in range(start, end):
            layer = self.layers[index]
            def run(value, module=layer):
                return module(value, src_mask=causal, src_key_padding_mask=~mask.bool())
            hidden = (checkpoint(run, hidden, use_reentrant=False)
                      if self.gradient_checkpointing and self.training else run(hidden))
        return hidden

    def finish_layers(self, hidden: Tensor) -> Tensor:
        return self.norm(hidden)

    def core(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        hidden, context = self.prepare_layers(embeddings, mask)
        return self.finish_layers(self.run_layers(hidden, context, 0, self.layer_count))

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.embedding.weight)


class HFBackbone(nn.Module):
    def __init__(self, model_id: str = "LiquidAI/LFM2.5-230M", *, revision: str = "main",
                 dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = True,
                 attention_implementation: str = "sdpa") -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'hf' extra; core tests do not require Transformers") from exc
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, trust_remote_code=False,
            attn_implementation=attention_implementation, use_safetensors=True,
        )
        if getattr(self.lm.config, "model_type", None) not in {"lfm2", "llama"}:
            raise ValueError("This adapter is validated structurally only for lfm2 and llama")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.width = self.lm.config.hidden_size
        self.max_length = min(getattr(self.lm.config, "max_position_embeddings", 32768), 32768)
        self.lm.config.use_cache = False
        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.resolved_revision = getattr(self.lm.config, "_commit_hash", None) or revision

    def checkpoint_layer_range(self, start: int, end: int) -> None:
        """Keep activation recomputation on the repeated native layer range only."""
        layers = self.lm.base_model.layers
        if not 0 <= start < end <= len(layers):
            raise ValueError("Invalid gradient-checkpointing layer range")
        for index, layer in enumerate(layers):
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = start <= index < end

    def embed(self, ids: Tensor) -> Tensor:
        return self.lm.get_input_embeddings()(ids)

    def core(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Configured context limit exceeded")
        return self.lm.base_model(inputs_embeds=embeddings, attention_mask=mask,
                                  use_cache=False, return_dict=True).last_hidden_state

    @property
    def layer_count(self) -> int:
        return len(self.lm.base_model.layers)

    @property
    def layer_types(self) -> list[str]:
        return list(getattr(self.lm.config, "layer_types", ["full_attention"] * self.layer_count))

    def prepare_layers(self, embeddings: Tensor, mask: Tensor):
        """Native pinned-Transformers mask/RoPE construction, never a cache mutation.

        The supported 5.5 and 5.17 LFM2 APIs name their convolution-mask path
        differently. Unsupported API changes should fail the one-pass preflight,
        not silently use a generic attention mask.
        """
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Configured context limit exceeded")
        from transformers.masking_utils import create_causal_mask
        model = self.lm.base_model
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)[None]
        kwargs = dict(config=self.lm.config, inputs_embeds=embeddings,
                      attention_mask=mask, past_key_values=None, position_ids=positions)
        masks = {"full_attention": create_causal_mask(**kwargs)}
        if self.lm.config.model_type == "lfm2":
            try:
                # Transformers 5.17 names the native recurrent/linear mask
                # constructor explicitly.
                from transformers.masking_utils import create_recurrent_attention_mask
            except ImportError:
                # Transformers 5.5 LFM2 passes the ordinary padding mask to its
                # convolution layers (and None for a one-token decoding step).
                # This mirrors Lfm2Model.forward in that pinned API.
                masks["conv"] = mask if embeddings.shape[1] != 1 else None
            else:
                masks["conv"] = create_recurrent_attention_mask(**kwargs)
        rotary = model.rotary_emb(embeddings, position_ids=positions)
        return embeddings, (masks, positions, rotary)

    def run_layers(self, hidden: Tensor, context, start: int, end: int) -> Tensor:
        masks, positions, rotary = context
        for index in range(start, end):
            # Native GradientCheckpointingLayer.__call__ remains responsible for
            # layer recomputation. No callback/retrieval lives inside its closure.
            hidden = self.lm.base_model.layers[index](
                hidden, attention_mask=masks[self.layer_types[index]],
                position_embeddings=rotary, position_ids=positions, past_key_values=None)
            if not isinstance(hidden, Tensor):
                raise TypeError("Unexpected decoder API; run the pinned-version model preflight")
        return hidden

    def finish_layers(self, hidden: Tensor) -> Tensor:
        model = self.lm.base_model
        return (model.embedding_norm if self.lm.config.model_type == "lfm2" else model.norm)(hidden)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm.get_output_embeddings()(hidden)


class RecurrentBackbone(nn.Module):
    """Reuse the entire pretrained stack with a gated, causal refinement update.

    loops=1 is exactly the underlying backbone path. Zero loop gate also preserves
    its output for any loop count. Each proposal uses original embeddings plus a
    normalized same-position previous state. No future-token pooling or cache reuse.
    Weight sharing saves parameters, NOT FLOPs, activations, or decoding latency.
    """
    def __init__(self, core: nn.Module, loops: int = 1) -> None:
        super().__init__()
        if loops < 1:
            raise ValueError("At least one loop is required")
        self.base, self.loops, self.width = core, loops, core.width
        self.loop_gate = nn.Parameter(torch.zeros(()))
        self.feedback_norm = nn.LayerNorm(self.width)

    def embed(self, ids: Tensor) -> Tensor:
        return self.base.embed(ids)

    def hidden(self, embeddings: Tensor, mask: Tensor, loops: int | None = None) -> Tensor:
        count = self.loops if loops is None else loops
        if count < 1:
            raise ValueError("At least one loop is required")
        hidden = self.base.core(embeddings, mask)
        scale = embeddings.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        for _ in range(1, count):
            feedback = self.feedback_norm(hidden) * scale.to(hidden.dtype)
            proposal = self.base.core(embeddings + feedback.to(embeddings.dtype), mask)
            hidden = hidden + self.loop_gate.tanh() * (proposal - hidden)
        return hidden

    def logits(self, hidden: Tensor) -> Tensor:
        return self.base.logits(hidden)
