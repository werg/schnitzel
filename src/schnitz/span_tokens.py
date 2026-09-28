"""Span and memory-protocol constants shared by the decoder, data builders and the KB
stack (restart plan 3.2; docs/knowledge-base-stack.md, WP1-WP3).

Span markers are LFM2 reserved tokens given new meaning; ``<|reserved_6|>`` stays
BGKit's splice sentinel. The ids below are those of the LFM2/LFM2.5 tokenizer
family that our decoder uses (checked against LFM2.5-350M).

KB access uses LFM2's native tool calling: a read is a ``memory_search()`` call
without arguments (the query is a vector, the hidden state at the call), a
write a ``memory_write`` call, and a read result is an ordinary ``tool`` message
whose content is a latent span between ``<|mem|>`` and ``<|/mem|>``.
"""
from __future__ import annotations

# name -> (reserved token string, id in the LFM2.5 vocabulary)
SPAN_TOKENS: dict[str, tuple[str, int]] = {
    'bg': ('<|reserved_20|>', 30),        # opens a compression / write span (marker + ratio code)
    'rep': ('<|reserved_21|>', 31),       # LM-head row: emit another rep inside a span
    'bg_end': ('<|reserved_22|>', 32),    # closes a span; its embedding is appended after the reps
    'mem': ('<|reserved_23|>', 33),       # opens the latent payload of a memory_search result
    'mem_end': ('<|reserved_24|>', 34),   # closes it
    'port': ('<|reserved_25|>', 35),      # opens a soft I/O port span
    'port_end': ('<|reserved_26|>', 36),  # closes it
}
SENTINEL = ('<|reserved_6|>', 16)         # BGKit's splice sentinel (unchanged)

# Tool schemas in the form LFM2's chat template accepts (``tools=`` argument).
MEMORY_TOOLS = [
    # a read has no text argument: the query is the decoder's hidden state at the call,
    # projected by one key head per KB space (docs/knowledge-base-stack.md)
    {'name': 'memory_search',
     'description': 'Search the knowledge base for what the task needs next. The result is a '
                    'memory span the model reads directly.',
     'parameters': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'memory_write',
     'description': 'Store reusable information in the knowledge base for later tasks.',
     'parameters': {'type': 'object',
                    'properties': {'content': {'type': 'string',
                                               'description': 'the information to store'}},
                    'required': ['content']}},
]


def token_id(name: str) -> int:
    return SPAN_TOKENS[name][1]


def check_tokenizer(tok) -> None:
    """Raise if ``tok`` does not map the span tokens to the expected ids."""
    vocab = tok.get_vocab()
    for name, (text, ident) in {**SPAN_TOKENS, 'sentinel': SENTINEL}.items():
        if vocab.get(text) != ident:
            raise ValueError(f'{name}: {text} is {vocab.get(text)} in this tokenizer, '
                             f'expected {ident}')
