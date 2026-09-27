"""One prompt boundary shared by preparation, training, and evaluation."""
from __future__ import annotations


def render_prompt(tokenizer, query: str, support_text: str = "") -> str:
    content = ("Prior experiences:\n" + support_text + "\n\n" if support_text else "") + query
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
