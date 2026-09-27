"""Frozen local text-embedding teachers for training-only keyspace distillation.

Teachers read source *content* and causal query text only. Their vectors never
enter a stored payload or the inference path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

SEARCH_PREFIX = 'Represent this sentence for searching relevant passages: '


@dataclass(frozen=True)
class TeacherSpec:
    model_id: str
    pooling: str  # cls, mean or last
    query_prefix: str = ''
    document_prefix: str = ''
    dense_modules: tuple[str, ...] = ()
    trust_remote_code: bool = False
    max_length: int = 512


TEACHERS = {
    'qwen3-embedding-0.6b': TeacherSpec(
        'Qwen/Qwen3-Embedding-0.6B', 'last',
        query_prefix=('Instruct: Given a search request, retrieve the stored '
                      'Wikipedia passage that answers it\nQuery:')),
    'embeddinggemma-300m': TeacherSpec(
        'google/embeddinggemma-300m', 'mean',
        query_prefix='task: search result | query: ',
        document_prefix='title: none | text: ', dense_modules=('2_Dense', '3_Dense')),
    'lfm2.5-embedding-350m': TeacherSpec(
        'LiquidAI/LFM2.5-Embedding-350M', 'cls', query_prefix='query: ',
        document_prefix='document: ', trust_remote_code=True),
    'gte-modernbert-base': TeacherSpec('Alibaba-NLP/gte-modernbert-base', 'cls'),
    'arctic-embed-m-v1.5': TeacherSpec(
        'Snowflake/snowflake-arctic-embed-m-v1.5', 'cls', query_prefix=SEARCH_PREFIX),
    'bge-base-en-v1.5': TeacherSpec(
        'BAAI/bge-base-en-v1.5', 'cls', query_prefix=SEARCH_PREFIX),
    'granite-embedding-english-r2': TeacherSpec(
        'ibm-granite/granite-embedding-english-r2', 'cls'),
    'mxbai-embed-large-v1': TeacherSpec(
        'mixedbread-ai/mxbai-embed-large-v1', 'cls', query_prefix=SEARCH_PREFIX),
    'e5-base-v2': TeacherSpec(
        'intfloat/e5-base-v2', 'mean', query_prefix='query: ', document_prefix='passage: '),
}


def _accept_unpacked_shortconv_calls() -> None:
    """Let the embedding model's bidirectional conv patch ignore ``seq_idx``.

    Its remote code replaces ``Lfm2ShortConv`` process-wide, so this teacher must
    never share a process with the causal SCHNITZELJAGD LFM2 student. Teacher batches are
    padded, not packed, so the omitted packed-sequence index is always ``None``.
    """
    from transformers.models.lfm2 import modeling_lfm2

    convolution = modeling_lfm2.Lfm2ShortConv

    def forward(self, *args, seq_idx=None, **kwargs):
        if seq_idx is not None:
            raise ValueError('Packed sequences are unsupported by the teacher patch')
        return self.slow_forward(*args, **kwargs)
    convolution.forward = forward


def pool(hidden: Tensor, mask: Tensor, pooling: str) -> Tensor:
    """Pool padded encoder states; ``last`` accepts left or right padding."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError('Pooling needs aligned hidden states and attention mask')
    if pooling == 'cls':
        return hidden[:, 0]
    if pooling == 'mean':
        weights = mask.to(hidden.dtype)[..., None]
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
    if pooling == 'last':
        length = mask.shape[1]
        # Index of the final attended token in each row.
        positions = torch.arange(length, device=mask.device)[None].expand_as(mask)
        last = torch.where(mask.bool(), positions, -1).max(1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError(f'Unknown pooling: {pooling}')


class TeacherEncoder:
    def __init__(self, name: str, *, device: str = 'cuda',
                 dtype: torch.dtype = torch.bfloat16) -> None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.name, self.spec = name, TEACHERS[name]
        path = Path(snapshot_download(self.spec.model_id, local_files_only=True))
        self.revision = path.name
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=self.spec.trust_remote_code)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=self.spec.trust_remote_code,
            dtype=dtype).to(device).eval()
        if self.spec.model_id.startswith('LiquidAI/LFM2.5-Embedding'):
            _accept_unpacked_shortconv_calls()
        self.dense = []
        for module in self.spec.dense_modules:
            config = json.loads((path / module / 'config.json').read_text())
            if config.get('bias') or not config['activation_function'].endswith('Identity'):
                raise ValueError('Only bias-free identity dense heads are supported')
            self.dense.append(load_file(str(path / module / 'model.safetensors'))[
                'linear.weight'].to(device=device, dtype=torch.float32))
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], *, role: str, batch_size: int = 128) -> Tensor:
        if role not in {'query', 'document'}:
            raise ValueError('Teacher role must be query or document')
        prefix = self.spec.query_prefix if role == 'query' else self.spec.document_prefix
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        output = torch.empty(len(texts), 0)
        chunks = []
        for start in range(0, len(order), batch_size):
            batch = [prefix + texts[index] for index in order[start:start + batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.spec.max_length,
                                     return_tensors='pt').to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            vectors = pool(hidden.float(), encoded['attention_mask'], self.spec.pooling)
            for weight in self.dense:
                vectors = vectors @ weight.T
            chunks.append(F.normalize(vectors, dim=-1).cpu())
        ordered = torch.cat(chunks) if chunks else output
        result = torch.empty_like(ordered)
        result[torch.tensor(order, dtype=torch.long)] = ordered
        return result
