"""Agentic document-ingestion trajectories with explicit memory.write sites."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

from .trajectory_memory import validate_memory_transcript, write_call, write_result


INGESTION_SYSTEM_PROMPT = (
    'Use memory.search for relevant prior knowledge. Preserve useful document '
    'information with explicit memory.write calls. Each call stores one reusable record.'
)


@dataclass(frozen=True)
class DocumentPart:
    text: str
    start: int
    end: int
    ordinal: int
    policy: str


def split_document(text: str, tokenizer, *, max_tokens: int,
                   overlap_words: int = 24) -> tuple[DocumentPart, ...]:
    """Prefer paragraph boundaries, then deterministically split oversized prose."""
    if not text.strip() or max_tokens < 8 or overlap_words < 0:
        raise ValueError('Document splitting needs text and positive token capacity')
    paragraphs = [(match.start(), match.end(), match.group()) for match in
                  re.finditer(r'\S(?:.*?\S)?(?=\n\s*\n|\Z)', text, re.S)]
    spans: list[tuple[int, int, str]] = []
    for start, end, paragraph in paragraphs:
        if len(tokenizer.encode(paragraph, add_special_tokens=False)) <= max_tokens:
            spans.append((start, end, 'paragraph'))
            continue
        words = list(re.finditer(r'\S+', paragraph))
        cursor = 0
        while cursor < len(words):
            lo, hi, best = cursor + 1, len(words), cursor + 1
            while lo <= hi:
                middle = (lo + hi) // 2
                piece = paragraph[words[cursor].start():words[middle - 1].end()]
                if len(tokenizer.encode(piece, add_special_tokens=False)) <= max_tokens:
                    best, lo = middle, middle + 1
                else:
                    hi = middle - 1
            piece_start = start + words[cursor].start()
            piece_end = start + words[best - 1].end()
            spans.append((piece_start, piece_end, 'token-bounded-overlap'))
            if best == len(words):
                break
            cursor = max(cursor + 1, best - overlap_words)
    return tuple(DocumentPart(text[start:end], start, end, ordinal, policy)
                 for ordinal, (start, end, policy) in enumerate(spans))


def holistic_write_count(token_count: int, *, base_tokens: int = 128,
                         maximum: int = 32) -> int:
    """A sublinear baseline; content-aware teachers may emit additional writes."""
    if token_count < 1 or base_tokens < 1 or maximum < 1:
        raise ValueError('Invalid holistic write-count budget')
    return min(maximum, max(1, math.ceil(math.log2(1 + token_count / base_tokens))))


def holistic_ingestion_messages(document_id: str, text: str, *, generation: str,
                                scope: dict[str, str],
                                write_contents: tuple[str, ...]) -> tuple[list[dict], tuple[dict, ...]]:
    """Present one document blob, then supervise a model-chosen write decomposition.

    ``write_contents`` is produced by a teacher, curator, or earlier model pass. It
    is deliberately not derived by the chunker: the input presentation and the
    target decomposition remain independent in holistic mode.
    """
    if (not document_id or not text.strip() or not write_contents
            or any(not content.strip() for content in write_contents)):
        raise ValueError('Holistic ingestion needs one document and nonempty write targets')
    messages = [{'role': 'system', 'content': INGESTION_SYSTEM_PROMPT},
                {'role': 'user', 'content': (
                    f'Document {document_id}:\n{text}\n\nInspect the complete document and '
                    'store its reusable information. Choose the number and contents of '
                    'the memory.write calls based on its structure and information density.')}]
    records = []
    for ordinal, content in enumerate(write_contents):
        record_id = hashlib.sha256(
            f'{generation}:{document_id}:holistic:{ordinal}:{content}'.encode()).hexdigest()[:32]
        call_id = f'ingest-holistic-{ordinal}-{record_id[:12]}'
        messages.append(write_call(
            call_id=call_id, content=content, kind='document_knowledge', scope=scope,
            applies_when=f'Information from document {document_id} is relevant.',
            evidence_refs=(f'{document_id}:0:{len(text)}',),
            confidence='source_document'))
        messages.append(write_result(call_id=call_id, status='ok',
                                     generation=generation, record_id=record_id))
        records.append({'record_id': record_id, 'document_id': document_id,
                        'span_start': 0, 'span_end': len(text), 'ordinal': ordinal,
                        'chunking_policy': 'holistic-model-decomposition',
                        'write_call_id': call_id})
    validate_memory_transcript(messages)
    return messages, tuple(records)


def grouped_ingestion_prefixes(document_id: str,
                               parts: tuple[tuple[str, str], ...], *,
                               generation: str, scope: dict[str, str],
                               mode: str) -> dict[str, list[dict]]:
    """Build per-record causal prefixes for one holistic or streaming document."""
    if (not document_id or not parts or len(parts) > 8
            or len({record_id for record_id, _ in parts}) != len(parts)
            or any(not record_id or not text.strip() for record_id, text in parts)
            or mode not in {'holistic', 'streaming'}):
        raise ValueError('Grouped ingestion needs 1..8 distinct nonempty parts and a mode')
    messages = [{'role': 'system', 'content': INGESTION_SYSTEM_PROMPT}]
    if mode == 'holistic':
        blob = '\n\n'.join(f'Part {index + 1} [{record_id}]:\n{text}'
                            for index, (record_id, text) in enumerate(parts))
        messages.append({'role': 'user', 'content': (
            f'Document {document_id}:\n{blob}\n\nInspect the complete document and '
            'choose a reusable memory decomposition. Emit distinct memory.write calls.')})
    prefixes = {}
    for ordinal, (record_id, text) in enumerate(parts):
        if mode == 'streaming':
            messages.append({'role': 'user', 'content': (
                f'Document {document_id}, part {ordinal + 1} of {len(parts)}:\n{text}\n\n'
                'Store the reusable information from this part before continuing.')})
        call_id = 'ingest-group-' + hashlib.sha256(
            f'{mode}:{document_id}:{record_id}'.encode()).hexdigest()[:20]
        messages.append(write_call(
            call_id=call_id, content=text, kind='document_part', scope=scope,
            applies_when=f'Information from document {document_id} is relevant.',
            evidence_refs=(f'{document_id}:part:{ordinal}',),
            confidence='source_document'))
        prefixes[record_id] = list(messages)
        messages.append(write_result(call_id=call_id, status='ok',
                                     generation=generation, record_id=record_id))
    validate_memory_transcript(messages)
    return prefixes


def source_ingestion_groups(rows: list[dict], *, maximum_parts: int = 4) -> dict[str, tuple]:
    """Assign source rows to stable bounded article groups and presentation modes."""
    if maximum_parts < 1 or maximum_parts > 8:
        raise ValueError('Source ingestion groups support 1..8 parts')
    # Only passages created at the same time share a group: holistic mode shows
    # every part while encoding each one, which would leak later passages into
    # the payloads of earlier ones. Time-1 groups keep their original identities.
    articles: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        title = row.get('provenance', {}).get('article_title') or row['record_id']
        articles.setdefault((str(title), int(row.get('created_at', 1))), []).append(row)
    result = {}
    for (title, created_at), article_rows in articles.items():
        label = title if created_at == 1 else f'{title}@{created_at}'
        for start in range(0, len(article_rows), maximum_parts):
            group = article_rows[start:start + maximum_parts]
            document_id = f'{label}#{start // maximum_parts}'
            mode = ('holistic' if int(hashlib.sha256(document_id.encode()).hexdigest(), 16) % 2
                    else 'streaming')
            parts = tuple((row['record_id'], row['text']) for row in group)
            shared = (document_id, parts, mode)
            for row in group:
                result[row['record_id']] = shared
    return result


def prompted_write_messages(*, document_id: str, text: str, record_id: str,
                            generation: str, scope: dict[str, str], kind: str,
                            span: tuple[int, int] | None = None,
                            prior_messages: tuple[dict, ...] = ()) -> list[dict]:
    """One causal producer prefix through a visible memory.write tool call."""
    if not document_id or not text or not record_id:
        raise ValueError('Prompted writes need document, content and record identities')
    start, end = span or (0, len(text))
    call_id = 'ingest-write-' + hashlib.sha256(
        f'{document_id}:{record_id}:{start}:{end}'.encode()).hexdigest()[:20]
    messages = ([{'role': 'system', 'content': INGESTION_SYSTEM_PROMPT}]
                if not prior_messages else list(prior_messages))
    messages.append({'role': 'user', 'content': (
        f'Document {document_id}, source span [{start}, {end}):\n{text}\n\n'
        'Store the reusable information from this part now. Preserve exact names, '
        'numbers, conditions, and provenance in the write content.')})
    messages.append(write_call(
        call_id=call_id, content=text, kind=kind, scope=scope,
        applies_when=f'Information from document {document_id} is relevant.',
        evidence_refs=(f'{document_id}:{start}:{end}',), confidence='source_document',
    ))
    return messages


def streaming_ingestion_messages(document_id: str, text: str, tokenizer, *,
                                 generation: str, scope: dict[str, str],
                                 max_part_tokens: int,
                                 overlap_words: int = 24) -> tuple[list[dict], tuple[dict, ...]]:
    """Feed parts causally and commit one visible record after every part."""
    messages: list[dict] = [{'role': 'system', 'content': INGESTION_SYSTEM_PROMPT}]
    records = []
    for part in split_document(text, tokenizer, max_tokens=max_part_tokens,
                               overlap_words=overlap_words):
        record_id = hashlib.sha256(
            f'{generation}:{document_id}:{part.start}:{part.end}'.encode()).hexdigest()[:32]
        call_id = f'ingest-write-{part.ordinal}-{record_id[:12]}'
        messages.append({'role': 'user', 'content': (
            f'Document {document_id}, part {part.ordinal + 1}, source span '
            f'[{part.start}, {part.end}):\n{part.text}\n\nStore this part now.')})
        messages.append(write_call(
            call_id=call_id, content=part.text, kind='document_part', scope=scope,
            applies_when=f'Information from document {document_id} is relevant.',
            evidence_refs=(f'{document_id}:{part.start}:{part.end}',),
            confidence='source_document'))
        messages.append(write_result(call_id=call_id, status='ok',
                                     generation=generation, record_id=record_id))
        records.append({'record_id': record_id, 'document_id': document_id,
                        'span_start': part.start, 'span_end': part.end,
                        'ordinal': part.ordinal, 'chunking_policy': part.policy,
                        'write_call_id': call_id})
    validate_memory_transcript(messages)
    return messages, tuple(records)


def writer_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Tokenize the causal transcript through the final memory.write call."""
    if not messages or not messages[-1].get('tool_calls'):
        raise ValueError('Writer prefix must end at a visible tool call')
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True)
    return list(encoded['input_ids'])
