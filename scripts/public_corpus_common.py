"""Shared record/episode construction for the September 2026 public corpus additions.

Every added dataset becomes its own domain namespace, with the same source and
episode schema as ``prepare_public_answer_episodes.py``. Answers are supervised
targets only; they are never query text. Provenance article titles are prefixed
by the domain, so the writer's article grouping never merges new passages into
groups of records already in a bank.

Output layout per dataset: ``sources.jsonl``, ``episodes-train.jsonl``,
``episodes-validation.jsonl`` (and optionally ``episodes-test.jsonl`` for
evaluation-only sets) and ``manifest.json``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re

PROMPT = 'Use the previously stored passages. Give only the short response.\nQuestion: '
MODEL_REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'
MAX_CHARS = 700
MAX_QUERY_TOKENS = 120
MAX_ANSWER_TOKENS = 64
MAX_ANSWER_WORDS = 24


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def record_id(domain: str, body: str) -> str:
    return hashlib.sha256(f'{domain}\0{body}'.encode()).hexdigest()[:32]


def load_tokenizer(cache: Path = Path('/cache')):
    from transformers import AutoTokenizer
    snapshot = cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / MODEL_REVISION
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def source(domain: str, text: str, *, title: str = '', created_at: int = 1,
           provenance: dict | None = None) -> dict:
    """One stored passage. ``text`` must already fit ``MAX_CHARS`` (see ``chunk``)."""
    body = f'Title: {clean(title)}\nPassage: {clean(text)}' if title else clean(text)
    if not clean(text):
        raise ValueError('Empty source text')
    meta = {'dataset': domain, **(provenance or {})}
    if title:
        meta['article_title'] = f'{domain}/{clean(title)}'
    return {'record_id': record_id(domain, body), 'text': body, 'domain': domain,
            'created_at': int(created_at), 'kind': 'passage', 'provenance': meta}


def chunk(sentences: list[str], gold: set[int] | None = None, *,
          limit: int = MAX_CHARS) -> list[tuple[str, bool, list[int]]]:
    """Pack whole sentences into passages of at most ``limit`` characters.

    Returns ``(text, contains_gold_sentence, sentence_indices)``. A sentence longer
    than the limit is split at word boundaries; each piece keeps its index.
    """
    gold = gold or set()
    pieces: list[tuple[str, int]] = []
    for index, sentence in enumerate(sentences):
        sentence = clean(sentence)
        while len(sentence) > limit:
            cut = sentence.rfind(' ', 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append((sentence[:cut], index))
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append((sentence, index))
    result, text, indices = [], '', []
    for piece, index in pieces:
        trial = f'{text} {piece}'.strip()
        if text and len(trial) > limit:
            result.append((text, bool(gold & set(indices)), sorted(set(indices))))
            text, indices = piece, [index]
        else:
            text, indices = trial, indices + [index]
    if text:
        result.append((text, bool(gold & set(indices)), sorted(set(indices))))
    return result


def split_sentences(text: str) -> list[str]:
    return [part for part in re.split(r'(?<=[.!?])\s+', clean(text)) if part]


@dataclass
class Filters:
    counts: Counter = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.counts[reason] += 1


def answer_ok(answer: str, question: str, filters: Filters, tokenizer=None, *,
              max_words: int = MAX_ANSWER_WORDS, allow_in_query: bool = False) -> bool:
    answer = clean(answer)
    if not answer:
        filters.reject('no_answer')
        return False
    if len(answer.split()) > max_words:
        filters.reject('answer_too_long')
        return False
    if not allow_in_query and answer.lower() in clean(question).lower():
        filters.reject('answer_in_query')
        return False
    if tokenizer is not None and len(tokenizer.encode(answer, add_special_tokens=False)) > MAX_ANSWER_TOKENS:
        filters.reject('answer_tokens')
        return False
    return True


def episode(*, domain: str, split: str, identifier: str, question: str, answer: str,
            gold: list[dict], supports: list[dict], filters: Filters, tokenizer=None,
            query_time: int = 2, all_required: bool = True, prompt: str = PROMPT,
            annotation: str = 'verified', task_family: str = 'public_qa',
            provenance: dict | None = None, allow_answer_in_query: bool = False) -> dict | None:
    """Build one answer-bearing episode, or return None with a counted reason.

    ``gold`` are the evidence sources. With ``all_required`` the only sufficient
    group is all of them (multi-hop); otherwise any one of them suffices.
    ``supports`` should contain ``gold`` plus any causally available distractors.
    """
    question = clean(question)
    if not gold:
        filters.reject('no_gold')
        return None
    if not answer_ok(answer, question, filters, tokenizer, allow_in_query=allow_answer_in_query):
        return None
    query = prompt + question
    if tokenizer is not None and len(tokenizer.encode(query, add_special_tokens=False)) > MAX_QUERY_TOKENS:
        filters.reject('query_tokens')
        return None
    everything = {item['record_id']: item for item in (*gold, *supports)}
    if any(item['created_at'] >= query_time or item['domain'] != domain
           for item in everything.values()):
        raise ValueError(f'{identifier}: evidence must be causally prior and in-domain')
    gold_ids = list(dict.fromkeys(item['record_id'] for item in gold))
    groups = [gold_ids] if all_required else [[item] for item in gold_ids]
    return {
        'episode_id': f'{domain}-{identifier}', 'environment': f'{domain}-{split}',
        'query': query, 'answer': clean(answer), 'query_time': int(query_time),
        'required_ids': gold_ids, 'sufficient_groups': groups,
        'support_annotation': annotation, 'task_family': task_family,
        'supports': [{'record_id': key, 'text': item['text'], 'created_at': item['created_at'],
                      'kind': 'passage'} for key, item in everything.items()],
        'provenance': {'dataset': domain, 'domain': domain, 'split': split, **(provenance or {})},
    }


class Writer:
    """Collect sources and episodes for one dataset and write them atomically."""

    def __init__(self, output: Path, domain: str):
        if output.exists():
            raise FileExistsError(output)
        self.output, self.domain = output, domain
        self.sources: dict[str, dict] = {}
        self.episodes: dict[str, list[dict]] = {}
        self.filters: dict[str, Filters] = {}
        self._ids: set[str] = set()

    def filters_for(self, split: str) -> Filters:
        return self.filters.setdefault(split, Filters())

    def add(self, split: str, item: dict | None, sources: list[dict]) -> bool:
        if item is None:
            return False
        if item['episode_id'] in self._ids:
            self.filters_for(split).reject('duplicate_episode')
            return False
        for row in sources:
            existing = self.sources.setdefault(row['record_id'], row)
            if existing['created_at'] != row['created_at']:
                raise ValueError('One passage identity with two creation times')
        self._ids.add(item['episode_id'])
        self.episodes.setdefault(split, []).append(item)
        return True

    def close(self, manifest: dict) -> dict:
        used = {record for rows in self.episodes.values() for row in rows
                for record in (s['record_id'] for s in row['supports'])}
        missing = used - self.sources.keys()
        if missing:
            raise ValueError(f'Episodes reference unknown sources: {sorted(missing)[:3]}')
        tmp = self.output.with_name(self.output.name + '.pending')
        tmp.mkdir(parents=True)
        digests = {}
        for split, rows in self.episodes.items():
            path = tmp / f'episodes-{split}.jsonl'
            with path.open('w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (tmp / 'sources.jsonl').open('w', encoding='utf-8') as handle:
            for key in sorted(self.sources):
                handle.write(json.dumps(self.sources[key], ensure_ascii=False) + '\n')
        for path in sorted(tmp.glob('*.jsonl')):
            digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        gold = Counter(len(row['required_ids']) for rows in self.episodes.values() for row in rows)
        result = {'format': 1, 'domain': self.domain, **manifest,
                  'episodes': {split: len(rows) for split, rows in self.episodes.items()},
                  'sources': len(self.sources),
                  'unreferenced_sources': len(self.sources.keys() - used),
                  'gold_per_episode': dict(sorted(gold.items())),
                  'filtered': {split: dict(f.counts) for split, f in self.filters.items()},
                  'sha256': digests}
        (tmp / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, self.output)
        return result
