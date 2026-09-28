"""Rendering of memory-protocol transcripts (version 3), shared by the data builder
(``scripts/prepare_memory_transcripts.py``) and the trainers (B4c in-context writes,
later L1): one implementation of how a transcript row becomes token ids.

A search slot renders as an empty ``<|mem|><|/mem|>`` pair (the trainer puts the
latent span between them); a ``memory_write()`` call is followed, inside the same
assistant turn, by the model's own ``<|bg|>`` ... ``<|/bg|>`` span (rendered as the
empty pair ``<|bg|><|/bg|>``). ``write_site_prefix`` gives the causal prefix of a
write site: everything up to and including the call's ``<|tool_call_end|>``, from a
render of the messages up to the site only (invariant 2), so the span a model
generates there depends on nothing after it.
"""
from __future__ import annotations

from schnitz.span_tokens import SPAN_TOKENS

WRITE_CALL = '[memory_write()]<|tool_call_end|>'
WRITE_FILL = SPAN_TOKENS['bg'][0] + SPAN_TOKENS['bg_end'][0]
SLOT_FILL = SPAN_TOKENS['mem'][0] + SPAN_TOKENS['mem_end'][0]


def _shown(messages: list[dict], fill: str) -> list[dict]:
    return [{**m, 'content': fill} if isinstance(m.get('content'), dict) and 'slot' in m['content']
            else m for m in messages]


def render_text(messages: list[dict], tools: list[dict], tok,
                slot_text: str | None = None) -> str:
    """Render through the tokenizer's chat template with every search slot replaced by
    an empty ``<|mem|><|/mem|>`` pair and an empty ``<|bg|><|/bg|>`` write span right
    after every ``memory_write()`` call, in the same assistant turn (the trainer puts
    the latent spans between them)."""
    fill = slot_text or SLOT_FILL
    text = tok.apply_chat_template(_shown(messages, fill), tools=tools, tokenize=False)
    return text.replace(WRITE_CALL + '<|im_end|>', WRITE_CALL + WRITE_FILL + '<|im_end|>')


def render_ids(tok, messages: list[dict], tools: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids and assistant (loss) mask of ``render_text``: the template's own mask,
    with the ``<|bg|>``/``<|/bg|>`` pair of each write span inserted after the call's
    ``<|tool_call_end|>`` inside the assistant turn (both in the loss)."""
    out = tok.apply_chat_template(_shown(messages, SLOT_FILL), tools=tools, tokenize=True,
                                  return_dict=True, return_assistant_tokens_mask=True)
    ids, mask = list(out['input_ids']), list(out['assistant_masks'])
    call_start = tok.convert_tokens_to_ids('<|tool_call_start|>')
    call_end = tok.convert_tokens_to_ids('<|tool_call_end|>')
    new_ids, new_mask, opened = [], [], None
    for i, (t, m) in enumerate(zip(ids, mask)):
        new_ids.append(t)
        new_mask.append(m)
        if t == call_start:
            opened = i
        elif t == call_end and opened is not None:
            if tok.decode(ids[opened + 1:i]) == '[memory_write()]':
                new_ids += [SPAN_TOKENS['bg'][1], SPAN_TOKENS['bg_end'][1]]
                new_mask += [m, m]
            opened = None
    return new_ids, new_mask


def site_messages(row: dict, site: int) -> list[dict]:
    """The messages a write site may see: all before the site's message and that
    message's tool calls up to and including the write call."""
    entry = row['write_sites'][site]
    index, call = entry['message'], entry['call']
    message = row['messages'][index]
    calls = message.get('tool_calls') or []
    if call >= len(calls) or calls[call]['function']['name'] != 'memory_write':
        raise ValueError(f'{row.get("episode_id")}: write site {site} is not a memory_write call')
    return row['messages'][:index] + [{**message, 'tool_calls': calls[:call + 1]}]


def write_site_prefix(tok, row: dict, site: int) -> list[int]:
    """Token ids of write site ``site`` up to and including its call's
    ``<|tool_call_end|>``: the position that emits ``<|bg|>``, after which the model
    generates the span. Rendered from ``site_messages`` only, so no token after the
    site can enter (earlier write spans stay empty ``<|bg|><|/bg|>`` pairs, search
    slots empty ``<|mem|><|/mem|>`` pairs)."""
    ids, _ = render_ids(tok, site_messages(row, site), row['tools'])
    opens = [i for i, t in enumerate(ids) if t == SPAN_TOKENS['bg'][1]]
    if not opens:
        raise ValueError(f'{row.get("episode_id")}: write site {site} rendered no span')
    return ids[:opens[-1]]
