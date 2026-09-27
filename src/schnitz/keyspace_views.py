"""Declared teacher views of stored sources and causal memory.search sites.

Views are fixed rules. They never use record IDs, answers, targets or anything
after the call position. Query windows strip opaque 32-hex identifiers that
earlier causal writes may mention, so a teacher sees content rather than IDs.
"""
from __future__ import annotations

import re

SOURCE_VIEWS = ('chunk', 'passage', 'chunk_neighbor', 'title_lead')
QUERY_VIEWS = ('arguments', 'prefix_window', 'content', 'short_window')
_HEX_ID = re.compile(r'\b[0-9a-f]{32}\b')
_INSTRUCTION = re.compile(
    r'^(Use the previously stored passages\.|Return exactly words|Give only)')


def _title_passage(text: str) -> tuple[str, str]:
    match = re.match(r'Title: (.*?)\nPassage: (.*)', text, re.S)
    return (match.group(1), match.group(2)) if match else ('', text)


def _first_sentence(text: str) -> str:
    match = re.match(r'(.+?[.!?])(\s|$)', text, re.S)
    return match.group(1) if match else text


def source_view(view: str, record_id: str, rows: dict[str, dict],
                groups: dict[str, tuple]) -> str:
    """One stored source under a declared view; neighbors share its article group."""
    text = rows[record_id]['text']
    title, passage = _title_passage(text)
    if view == 'chunk':
        return text
    if view == 'passage':
        return passage
    if view == 'title_lead':
        return f'Title: {title}\n{_first_sentence(passage)}'
    if view == 'chunk_neighbor':
        parts = [part_id for part_id, _ in groups[record_id][1]]
        position = parts.index(record_id)
        neighbor = parts[position - 1] if position else (
            parts[1] if len(parts) > 1 else None)
        if neighbor is None:
            return text
        return text + '\n' + _title_passage(rows[neighbor]['text'])[1]
    raise ValueError(f'Unknown source view: {view}')


def causal_prefix_text(tokenizer, input_ids: list[int], query_position: int,
                       tokens: int) -> str:
    """Decode the last visible prefix tokens through the call position."""
    if query_position < 0 or tokens < 1:
        raise ValueError('Causal window needs a valid call position and size')
    visible = [token for token in input_ids[:query_position + 1] if token >= 0]
    return _HEX_ID.sub('', tokenizer.decode(visible[-tokens:]))


def query_view(view: str, arguments: str, *, tokenizer=None,
               input_ids: list[int] | None = None,
               query_position: int | None = None) -> str:
    if view == 'arguments':
        return arguments
    if view == 'content':
        lines = [line for line in arguments.splitlines()
                 if line.strip() and not _INSTRUCTION.match(line.strip())]
        return '\n'.join(lines) or arguments
    if view in {'prefix_window', 'short_window'}:
        if tokenizer is None or input_ids is None or query_position is None:
            raise ValueError('Prefix windows need the packed causal trajectory')
        return causal_prefix_text(tokenizer, input_ids, query_position,
                                  512 if view == 'prefix_window' else 128)
    raise ValueError(f'Unknown query view: {view}')
