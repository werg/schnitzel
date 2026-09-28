"""Memory-protocol chat transcripts for LFM2 (knowledge-base stack WP3, restart plan 3.2).

Every episode of an R6 or task corpus becomes one LFM2 chat transcript in the
structured form ``tok.apply_chat_template(messages, tools=tools)`` renders:

- ``system``: a memory-use policy (and the task family's role);
- ``user``: the task, with the corpus prompt's "Use the stored ..." preamble removed
  (the knowledge base is reached by tool calls now, not announced in the prompt);
- search sites: an ``assistant`` message whose ``tool_calls`` are one or more
  ``memory_search(query=...)`` calls, each followed by a ``tool`` message whose
  content is a latent SLOT ``{"slot": {"kb", "record_ids", ...}}``; the trainer
  renders it as ``<|mem|>`` + latent span + ``<|/mem|>`` (``schnitz.span_tokens``);
- the answer: an ``assistant`` message (text, or native tool calls for function
  calling); multi-turn corpora keep their ``turns`` (customers and observations as
  ``user``, API results as ``tool``, agent API calls as native tool calls, whose
  documentation is searched just before the first call of each tool);
- write sites (trajectory tasks by default): a final ``memory_write(content=...)``
  call whose content summarizes the episode's own trajectory, and a ``tool`` ack.

Which records a search returns comes from the source episode (``required_ids``,
the first valid ``sufficient_groups`` entry for R6, related ``supports`` such as
column values, worked examples and background). Searches are grouped into
stages (rules/protocol, schema, values, evidence, tool docs, know-how, examples,
background); multi-hop passages are searched one hop per site, ordered so that
a hop's title is grounded in the question or in passages read before it.

Query text is built from the causal prefix (the request, the trajectory so far,
records already read) and the target record's header (title, table, tool name,
kind) with varied templates, then audited: no n-gram of ``--ngram`` tokens (5)
from anything after the call (answer, later turns, gold SQL/calls) or from the
target records' bodies unless it also occurs in the prefix or the records'
headers, and no short-answer string unless the prefix already contains it. A
query that fails falls back to the next candidate; a generic query is the last
resort and counted.

Enforced checks (counts in ``manifest.json``): every slot record exists in the
episode's KB (``sources.jsonl`` of the corpus; for R6 each dataset domain is its
own KB) with ``created_at`` before the episode's ``query_time``; slots never
contain a record that copies the episode's own long answer (its gold
trajectory) unless it is an explicit ``--gold-slots`` record (train only, flagged
``gold`` with a ``receding_weight``, restart plan B9); validation and test
transcripts never reference gold records. Output is deterministic (per-episode
RNG from ``--seed`` and the episode id) and hashed in the manifest.

Loss policy: every assistant message (content and tool calls, including
``memory_search`` and ``memory_write`` calls) gets loss; system, user and tool
content never does. The LFM2 template marks assistant spans with
``{% generation %}``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schnitz.span_tokens import MEMORY_TOOLS, SPAN_TOKENS  # noqa: E402

NGRAM = 5
SHORT_ANSWER = 100          # answers up to this many characters are also string-checked
OWN_GOLD_TOKENS = 12        # answers shorter than this are not checked for copies
OWN_GOLD_OVERLAP = 0.5      # share of a record's 8-grams inside the answer = a copy
COPY_KINDS = {'worked_example', 'know_how', 'protocol', 'policy', 'background', 'rules'}
WRITE_CHARS = 700
LOSS_POLICY = {'loss_on': 'assistant', 'assistant_parts': ['content', 'tool_calls'],
               'no_loss': ['system', 'user', 'tool'], 'template_generation_tags': True}

SYSTEM = (
    'You can consult a knowledge base. Call memory_search with a short description of what '
    'you need; each result is a memory span you read directly. Look things up before you '
    'rely on them.',
    'A knowledge base holds the reference material for this task. Use memory_search with '
    'short queries to read documentation, schemas, rules, examples or facts, as often as '
    'needed.',
    'Before answering, retrieve what you need with memory_search; results arrive as memory '
    'spans.',
)
ROLE = {
    'function_call': 'Fulfil the request by calling the listed tools; their documentation '
                     'is in the knowledge base.',
    'policy_tool_agent': 'You are a customer service agent ({area}). Follow the agent policy '
                         'stored in the knowledge base; the tool documentation is there too.',
    'agent': 'You act in a text environment, one action per turn. Its protocol, know-how '
             'and worked examples are in the knowledge base.',
    'text_to_sql': 'Database schemas, column values and notes are in the knowledge base.',
    'stored_table_qa': 'Database contents are in the knowledge base.',
}
WRITE_POLICY = 'When the task is done, store reusable know-how with memory_write.'

# search stages, in order; kinds of one stage share a site
STAGES = (('protocol', 'policy', 'rules'), ('schema',), ('column_values', 'table_rows'),
          ('evidence',), ('tool_doc',), ('know_how',), ('worked_example',), ('background',))
SPACE_HINT = {'column_values': 'fine', 'table_rows': 'fine', 'schema': 'fine',
              'evidence': 'fine', 'tool_doc': 'fine', 'background': 'coarse',
              'worked_example': 'coarse', 'protocol': 'coarse', 'policy': 'coarse',
              'know_how': 'coarse', 'rules': 'coarse'}
HEADERS = {
    'schema': r'^Database (?P<db>\S+), table (?P<table>.+?) \(schema\):',
    'column_values': r'^Database (?P<db>\S+), values of (?P<table>[^.\n]+)\.(?P<column>[^:\n]+):',
    'table_rows': r'^Database (?P<db>\S+), table (?P<table>.+?) \(',
    'evidence': r'^Database (?P<db>\S+), note:',
    'tool_doc': r'^(?:Tool: (?P<tool>\S+)|(?P<area>\w+) tool (?P<tool2>[^:\s]+):)',
    'policy': r'^(?P<area>\w+) agent policy:',
    'protocol': r'^(?P<env>\S+) protocol:',
    'worked_example': r'^Worked example(?: \((?P<family>[^)]*)\))?',
    'background': r'^Background: (?P<title>[^\n]+)',
    'passage': r'^Title: (?P<title>[^\n]+)',
    'know_how': r'^Floorplan (?P<plan>\d+)',
}
TEMPLATES = {
    'passage': ('{need}', '{keys}', 'passages about {keys}', 'facts on {keys}',
                'what is stored about {keys}'),
    'passage_title': ('{title}', 'about {title}', '{title}: {keys}', 'facts about {title}',
                      'what the knowledge base says about {title}'),
    'passage_hop': ('more on {keys}', 'follow-up facts for: {keys}', 'the other passage for {keys}'),
    'schema': ('schema of table {table} in {db}', 'columns of {db}.{table}',
               '{table} table definition ({db})', 'which columns does {table} have',
               '{db} tables for: {keys}'),
    'column_values': ('values of {table}.{column}', 'possible {column} values in {table}',
                      'distinct entries of {table} columns {column}', 'stored values for {table}'),
    'table_rows': ('rows of table {table}', 'contents of {db}.{table}',
                   'what is stored in the {table} table', 'data in {table}'),
    'evidence': ('notes on {keys}', 'hints for {db}: {keys}', 'definitions needed for: {keys}',
                 'how {db} encodes {keys}', 'notes for: {need}'),
    'tool_doc': ('documentation for {tool}', 'how to call {tool}', 'parameters of {tool}',
                 '{tool} API docs', 'arguments {tool} expects'),
    'policy': ('{area} agent policy', 'rules for handling {area} customer requests',
               'what the {area} policy says about {keys}'),
    'protocol': ('how to act in {env}', 'action format for {env} tasks',
                 '{env} interaction protocol', 'rules of the {env} environment'),
    'rules': ('rules for this kind of puzzle', 'how to solve puzzles like: {keys}',
              'rules: {keys}'),
    'know_how': ('where objects are usually found in this house', 'known object locations',
                 'where to find things for: {keys}'),
    'worked_example': ('worked examples for: {keys}', 'solved examples similar to {keys}',
                       'how similar {family} tasks were solved', 'examples of {family} solutions',
                       'examples like: {need}'),
    'background': ('background on {keys}', 'reference material about {keys}',
                   'documentation relevant to {keys}', '{title}', 'background for: {need}'),
    'gold_trajectory': ('my earlier attempt at this task', 'previous attempt for: {keys}'),
    'distractor': ('anything else about {keys}', 'related notes: {keys}', '{title}'),
}
GENERIC = {'passage': 'relevant passage', 'worked_example': 'worked examples',
           'background': 'background', 'gold_trajectory': 'earlier attempt'}
STOP = set('''a an the of in on at to for from by with and or but is are was were be been being
do does did has have had what which who whom whose when where why how that this these those it
its as into than then there their they he she his her them i you your we our me my not no yes
can could would should will shall may might must if so such any all some each about after
before over under between during also only just very more most other one two use using stored
give short response question answer please tell find write following return exactly words
through hello hi thanks thank'''.split())
MAX_SLOT = {'worked_example': 4, 'background': 4}  # coarse slots keep a sample of their records


# -- text helpers -----------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', (text or '').lower())


def ngrams(toks: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contains(haystack: list[str], needle: list[str]) -> bool:
    """``needle`` occurs as a contiguous token run in ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    return any(haystack[i] == first and haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def keyphrase(text: str, limit: int = 8) -> str:
    """Content words of ``text`` in order (stopwords and duplicates dropped)."""
    seen, out = set(), []
    for word in re.findall(r"[\w][\w'&+./-]*", text or ''):
        word = word.strip("'.-/")
        low = word.lower()
        if not word or low in STOP or low in seen or (len(low) < 2 and not low.isdigit()):
            continue
        seen.add(low)
        out.append(word)
        if len(out) >= limit:
            break
    return ' '.join(out)


def header(text: str) -> str:
    return (text or '').split('\n', 1)[0]


def fields_of(kind: str, text: str) -> dict:
    found = re.match(HEADERS.get(kind, r'(?!)'), text or '')
    if not found:
        return {}
    got = {k: v for k, v in found.groupdict().items() if v}
    if 'tool2' in got:
        got['tool'] = got.pop('tool2')
    if 'family' in got:
        got['family'] = got['family'].replace('_', ' ')
    return got


def grounded(title: str, prefix_tokens: set[str]) -> bool:
    """At least half of the title's content words already occur in the prefix."""
    words = [w for w in tokens(title) if w not in STOP]
    return bool(words) and sum(w in prefix_tokens for w in words) * 2 >= len(words)


def strip_preamble(query: str, dataset: str) -> str:
    """The corpus prompt without its "Use the stored ..." sentence (the database name
    of text-to-SQL prompts is kept)."""
    query = re.sub(r'^Use the stored (?:notes on|contents of) database (\S+?)\.\s*',
                   r'Database \1. ', query)
    if dataset == 'xlam':
        found = re.search(r'Request: (.*)$', query, flags=re.DOTALL)
        return found.group(1).strip() if found else query
    query = re.sub(r'^Use the (?:previously )?stored [^.\n]*\.\s*', '', query)
    query = re.sub(r'^You are the agent; reply to the customer or call tools\.\s*', '', query)
    return re.sub(r'^Customer: ', '', query).strip()


MARKERS = ('Current question:', 'Claim:', 'Question:', 'Request:', 'Task:', 'Customer:')


def need_text(user: str, dataset: str) -> str:
    """The part of the request that says what is needed (question, claim, goal)."""
    begins = re.search(r'Article: (.+)\nStored passage begins: (.+)', user)
    if begins:
        return f"passage of {begins.group(1)} that begins '{begins.group(2).strip()}'"
    if dataset == 'knights':
        return clean(user.split('\n', 1)[-1])[:400]
    if dataset == 'kodcode':
        return clean(user.rsplit('\n\n', 1)[0])[:400]
    at = max(((user.rfind(m), m) for m in MARKERS), default=(-1, ''))
    if at[0] >= 0:
        rest = user[at[0] + len(at[1]):].strip()
        if at[1] == 'Current question:':
            topic = re.search(r'Earlier in this conversation: Q: (.+?) A:', user)
            return clean(rest + (f' ({topic.group(1)})' if topic else ''))[:400]
        return clean(rest.split('\n')[0])[:400]
    return clean(next((p for p in user.split('\n\n') if len(p.split()) >= 4), user))[:400]


def words(text: str, limit: int) -> str:
    parts = text.split()
    return ' '.join(parts[:limit])


# -- audit -------------------------------------------------------------------------
def leak_reason(query: str, *, prefix: list[str], future: list[str], allowed: list[str],
                answers: list[str], n: int = NGRAM) -> str | None:
    """Why ``query`` is not derived from the causal prefix, or None.

    ``future`` are texts the query must not copy from (answer, later turns, target
    record bodies); their n-grams are allowed only where they also occur in ``prefix``
    or in ``allowed`` (target record headers). A short answer string may appear only if
    the prefix contains it too."""
    q = tokens(query)
    if not q:
        return 'empty'
    ok = ngrams(prefix, n) | {g for text in allowed for g in ngrams(tokens(text), n)}
    mine = ngrams(q, n)
    for text in future:
        if mine & (ngrams(tokens(text), n) - ok):
            return 'ngram'
    for answer in answers:
        a = tokens(answer)
        if a and len(answer) <= SHORT_ANSWER and contains(q, a) and not contains(prefix, a):
            return 'answer'
    return None


def own_gold(answer_grams: set, text: str) -> bool:
    """``text`` is (mostly) a copy of the episode's own answer."""
    grams = ngrams(tokens(text), 8)
    return bool(answer_grams) and len(grams) >= 3 and \
        len(grams & answer_grams) >= OWN_GOLD_OVERLAP * len(grams)


# -- messages ------------------------------------------------------------------------
def call(name: str, arguments: dict) -> dict:
    return {'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def slot(kb: str, record_ids: list[str], **flags) -> dict:
    body = {'kb': kb, 'record_ids': list(record_ids)}
    body.update({k: v for k, v in flags.items() if v is not None})
    return {'role': 'tool', 'name': 'memory_search', 'content': {'slot': body}}


def tool_stub(name: str) -> dict:
    return {'name': name, 'description': 'Documented in the knowledge base; search for it '
            'before calling.', 'parameters': {'type': 'object', 'properties': {}}}


def message_text(message: dict, exclude_memory: bool = True) -> str:
    """Plain text of a message for the audit (slots and memory-call arguments excluded)."""
    parts = []
    content = message.get('content')
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, dict) and 'slot' not in content and 'write_result' not in content:
        parts.append(json.dumps(content, ensure_ascii=False))
    for tc in message.get('tool_calls') or []:
        fn = tc['function']
        if exclude_memory and fn['name'] in ('memory_search', 'memory_write'):
            continue
        parts.append(fn['name'] + ' ' + json.dumps(fn['arguments'], ensure_ascii=False))
    return '\n'.join(parts)


def render_text(messages: list[dict], tools: list[dict], tok,
                slot_text: str | None = None) -> str:
    """Render through the tokenizer's chat template with every slot replaced by an
    empty ``<|mem|><|/mem|>`` pair (the trainer puts the latent span between them)."""
    fill = slot_text or SPAN_TOKENS['mem'][0] + SPAN_TOKENS['mem_end'][0]
    shown = [{**m, 'content': fill} if isinstance(m.get('content'), dict) and 'slot' in m['content']
             else m for m in messages]
    return tok.apply_chat_template(shown, tools=tools, tokenize=False)


def render_check(tok, row: dict) -> Counter:
    """Render ``row`` through the real chat template and check the protocol: one
    tool-call block per calling assistant message, an empty ``<|mem|><|/mem|>`` pair
    per slot outside the loss mask, every memory call inside it and no system, user or
    tool token in it."""
    bad: Counter = Counter()
    messages, tools = row['messages'], row['tools']
    fill = SPAN_TOKENS['mem'][0] + SPAN_TOKENS['mem_end'][0]
    shown = [{**m, 'content': fill} if isinstance(m.get('content'), dict) and 'slot' in m['content']
             else m for m in messages]
    out = tok.apply_chat_template(shown, tools=tools, tokenize=True, return_dict=True,
                                  return_assistant_tokens_mask=True)
    ids, mask = out['input_ids'], out['assistant_masks']
    slots = sum(isinstance(m.get('content'), dict) and 'slot' in m['content'] for m in messages)
    mem = [i for i, t in enumerate(ids) if t in (SPAN_TOKENS['mem'][1], SPAN_TOKENS['mem_end'][1])]
    if len(mem) != 2 * slots:
        bad['mem_token_count'] += 1
    if any(mask[i] for i in mem):
        bad['mem_in_loss'] += 1
    text = tok.decode(ids)
    calling = sum(bool(m.get('tool_calls')) for m in messages if m['role'] == 'assistant')
    if text.count('<|tool_call_start|>') != calling:
        bad['tool_call_blocks'] += 1
    trained = tok.decode([t for t, m in zip(ids, mask) if m])
    for name in ('memory_search', 'memory_write'):
        want = sum(tc['function']['name'] == name for m in messages
                   for tc in m.get('tool_calls') or [])
        if trained.count(name + '(') != want:
            bad[f'{name}_not_in_loss'] += 1
    start, end = tok.convert_tokens_to_ids('<|im_start|>'), tok.convert_tokens_to_ids('<|im_end|>')
    role = None
    for i, t in enumerate(ids):
        if t == start and i + 1 < len(ids):
            role = tok.decode([ids[i + 1]]).strip()
        elif t == end:
            role = None
        elif role in ('system', 'user', 'tool') and mask[i]:
            bad['context_in_loss'] += 1
            break
    return bad


def slot_ids(message: dict) -> list[str]:
    content = message.get('content')
    return content['slot']['record_ids'] if isinstance(content, dict) and 'slot' in content else []


def call_context(messages: list[dict], i: int, j: int, tools_text: str,
                 texts: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """For search call ``j`` of message ``i``: prefix tokens (tool list, earlier messages
    and the records their slots returned, earlier calls of the same message), future
    texts (later messages without memory calls, this call's record bodies) and the
    allowed record headers."""
    prefix = [tools_text]
    for m in messages[:i]:
        prefix.append(message_text(m, exclude_memory=False))
        prefix += [texts.get(r, '') for r in slot_ids(m)]
    prefix += [json.dumps(tc['function']['arguments'], ensure_ascii=False)
               for tc in messages[i]['tool_calls'][:j]]
    records = [texts.get(r, '') for r in slot_ids(messages[i + 1 + j])]
    future = [message_text(m) for m in messages[i + 1:]]
    future += [t.split('\n', 1)[1] if '\n' in t else '' for t in records]
    return tokens('\n'.join(prefix)), future, [header(t) for t in records]


# -- lookups --------------------------------------------------------------------------
class Lookup:
    """One memory_search call to be placed: its records and candidate query templates."""

    def __init__(self, kind: str, records: list[dict], *, fields: dict | None = None,
                 templates: tuple = (), flags: dict | None = None):
        self.kind, self.records = kind, records
        self.fields = fields or {}
        self.templates = templates or TEMPLATES.get(kind, ())
        self.flags = flags or {}


def _lookups_for_stage(kinds: tuple, records: list[dict], fields: dict,
                       rng: random.Random, counts: Counter) -> list[Lookup]:
    """Group a stage's records into calls: one per table / tool, one per kind otherwise."""
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec['kind'] not in kinds:
            continue
        info = fields_of(rec['kind'], rec['text'])
        if rec['kind'] in ('schema', 'column_values', 'table_rows'):
            key = (rec['kind'], info.get('db'), info.get('table'))
        elif rec['kind'] in ('tool_doc', 'evidence'):
            key = (rec['kind'], rec['record_id'])
        else:
            key = (rec['kind'],)
        groups.setdefault(key, []).append(rec)
    out = []
    for key, recs in groups.items():
        cap = MAX_SLOT.get(key[0])
        if cap and len(recs) > cap:
            counts['slot_records_capped'] += len(recs) - cap
            keep = set(rng.sample(range(len(recs)), cap))
            recs = [r for i, r in enumerate(recs) if i in keep]
        info = {}
        for rec in recs:
            for k, v in fields_of(rec['kind'], rec['text']).items():
                info.setdefault(k, v)
        if key[0] == 'column_values':
            cols = [fields_of('column_values', r['text']).get('column') for r in recs]
            info['column'] = ', '.join(dict.fromkeys(c for c in cols if c))
        for k, v in fields.items():
            info.setdefault(k, v)
        info.setdefault('family', 'related')
        out.append(Lookup(key[0], recs, fields=info))
    return out


def hop_order(need: str, records: list[dict]) -> list[dict]:
    """Passages in an order where each title is grounded in the question or in passages
    read before it where possible (multi-hop); ungrounded ones keep their order."""
    known = set(tokens(need))
    left, order = list(records), []
    while left:
        pick = next((r for r in left if grounded(fields_of('passage', r['text']).get('title', ''),
                                                  known)), left[0])
        left.remove(pick)
        order.append(pick)
        known |= set(tokens(pick['text']))
    return order


def fill(template: str, fields: dict) -> str | None:
    try:
        text = template.format(**fields)
    except KeyError:
        return None
    return clean(text) or None


# -- transcript builder ----------------------------------------------------------------
class Options:
    def __init__(self, seed=0, distractor_rate=0.0, gold_slots=False, writes='trajectory',
                 sequential_rate=0.3, query_hook=None, ngram=NGRAM):
        self.seed, self.distractor_rate, self.gold_slots = seed, distractor_rate, gold_slots
        self.writes, self.sequential_rate, self.query_hook = writes, sequential_rate, query_hook
        self.ngram = ngram


def episode_rng(seed: int, episode_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f'{seed}\0{episode_id}'.encode()).hexdigest()[:16], 16))


def gold_record(kb: str, episode: dict) -> dict:
    text = episode['answer']
    rid = hashlib.sha256(f'{kb}\0gold\0{episode["episode_id"]}\0{text}'.encode()).hexdigest()[:32]
    return {'record_id': rid, 'text': text, 'kind': 'gold_trajectory', 'domain': kb,
            'created_at': max(1, int(episode.get('query_time', 2)) - 1),
            'provenance': {'kb': kb, 'episode_id': episode['episode_id'], 'lineage': 'gold'}}


def family_of(episode: dict) -> str:
    fam = episode.get('task_family', '')
    if fam == 'policy_tool_agent':
        return 'policy_tool_agent'
    if episode.get('turns'):
        return 'agent'
    return fam


def _parse_call(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or 'name' not in data:
        return None
    args = data.get('arguments') or {}
    return call(data['name'], args if isinstance(args, dict) else {'value': args})


def write_content(episode: dict, family: str, user: str, dataset: str) -> str:
    """Reusable know-how from the episode's own final trajectory."""
    need = need_text(user, dataset)
    turns = episode.get('turns') or []
    if family == 'agent':
        actions = [re.sub(r'^.*?Action:\s*', '', t['text'], flags=re.DOTALL).strip()
                   for t in turns if t['role'] == 'assistant']
        actions = [clean(a)[:120] for a in actions if a]
        if len(actions) > 16:
            actions = actions[:8] + ['...'] + actions[-7:]
        text = f'Task: {words(need, 40)}\nSolved in {sum(t["role"] == "assistant" for t in turns)} ' \
               f'steps: ' + '; '.join(actions)
    elif family == 'policy_tool_agent':
        names = []
        for t in turns:
            if t['role'] == 'assistant' and t['text'].startswith('Call: '):
                parsed = _parse_call(t['text'][6:])
                if parsed and (not names or names[-1] != parsed['function']['name']):
                    names.append(parsed['function']['name'])
        last = next((t['text'] for t in reversed(turns) if t['role'] == 'assistant'
                     and not t['text'].startswith('Call: ')), '')
        text = (f'Customer need: {words(need, 40)}\nTool sequence: {" -> ".join(names) or "none"}\n'
                f'Outcome: {words(clean(last), 50)}')
    else:
        text = f'Task: {words(need, 60)}\nSolution: {episode["answer"][:400]}'
    return text[:WRITE_CHARS]


class Builder:
    """Turns source episodes of one corpus into transcripts, with checks and counts."""

    def __init__(self, kb: str, index: dict, options: Options, *, per_domain: bool,
                 tool_names: dict[str, list[str]] | None = None):
        self.kb, self.index, self.opt, self.per_domain = kb, index, options, per_domain
        self.tool_names = tool_names or {}
        self.gold_ids: set[str] = set()
        self.pending_gold: dict | None = None
        self.hook = None
        if options.query_hook:
            module, name = options.query_hook.split(':')
            self.hook = getattr(importlib.import_module(module), name)

    # records ------------------------------------------------------------------------
    def _valid(self, rec: dict, query_time: int, counts: Counter) -> bool:
        meta = self.index.get(rec['record_id'])
        if meta is None:
            counts['record_missing'] += 1
            return False
        if meta[0] >= query_time:
            counts['record_after_query'] += 1
            return False
        return True

    def kb_of(self, record_id: str) -> str:
        return f'{self.kb}:{self.index[record_id][2]}' if self.per_domain else self.kb

    # main ---------------------------------------------------------------------------
    def build(self, episode: dict, split: str) -> tuple[dict | None, str | None, Counter]:
        counts: Counter = Counter()
        rng = episode_rng(self.opt.seed, episode['episode_id'])
        qt = int(episode.get('query_time', 2))
        prov = episode.get('provenance', {})
        dataset = prov.get('dataset', '')
        family = family_of(episode)
        supports = {s['record_id']: s for s in episode.get('supports', [])}
        required = episode.get('required_ids', [])
        answer = episode.get('answer', '')
        answer_grams = ngrams(tokens(answer), 8) if len(tokens(answer)) >= OWN_GOLD_TOKENS else set()

        # which records the episode reads
        if family.startswith('public') or family.startswith('synthetic') or family in (
                'passage_span', 'hotpot_multihop'):
            group = next((g for g in episode.get('sufficient_groups') or [required]
                          if g and all(r in supports and self._valid(supports[r], qt, Counter())
                                       for r in g)), None)
            if group is None:
                return None, 'no_valid_sufficient_group', counts
            wanted = [supports[r] for r in group]
            spare = [s for k, s in supports.items() if k not in group]
        else:
            for r in required:
                if r not in supports:
                    return None, 'required_not_in_supports', counts
                if not self._valid(supports[r], qt, counts):
                    return None, 'required_record_invalid', counts
            wanted, spare = [], []
            for k, s in supports.items():
                if k not in required and s['kind'] in ('tool_doc', 'passage'):
                    spare.append(s)
                elif k in required or self._valid(s, qt, counts):
                    wanted.append(s)
        spare = [s for s in spare if self._valid(s, qt, Counter())]
        kept = []
        for rec in wanted:
            if answer_grams and rec['kind'] in COPY_KINDS and own_gold(answer_grams, rec['text']):
                counts['own_gold_dropped'] += 1
                if rec['record_id'] in required:
                    return None, 'required_record_copies_answer', counts
                continue
            kept.append(rec)
        wanted = kept
        if not wanted:
            return None, 'no_records', counts
        kbs = {self.kb_of(r['record_id']) for r in wanted}
        if len(kbs) != 1:
            return None, 'records_span_several_kbs', counts
        kb = kbs.pop()
        spare = [s for s in spare if self.kb_of(s['record_id']) == kb]

        user = strip_preamble(episode['query'], dataset)
        need = need_text(user, dataset)
        keys = keyphrase(need) or words(need, 8)

        # tools and system prompt
        tools = list(MEMORY_TOOLS)
        area = prov.get('area') or episode.get('verify', {}).get('area') or ''
        if family == 'function_call':
            names = [fields_of('tool_doc', s['text']).get('tool') for s in supports.values()
                     if s['kind'] == 'tool_doc']
            tools += [tool_stub(n) for n in dict.fromkeys(n for n in names if n)]
        elif family == 'policy_tool_agent':
            tools += [tool_stub(n) for n in self.tool_names.get(area, [])]
        writes = (self.opt.writes == 'all' or
                  (self.opt.writes == 'trajectory' and family in ('agent', 'policy_tool_agent')))
        system = SYSTEM[rng.randrange(len(SYSTEM))]
        role = ROLE.get(family, '').format(area=area)
        system = ' '.join(p for p in (system, role, WRITE_POLICY if writes else '') if p)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        plans: dict[tuple[int, int], Lookup] = {}

        def site(lookups: list[Lookup]):
            sequential = len(lookups) > 1 and rng.random() < self.opt.sequential_rate
            batches = [[lk] for lk in lookups] if sequential else [lookups]
            for batch in batches:
                messages.append({'role': 'assistant', 'content': '',
                                 'tool_calls': [call('memory_search', {'query': None})
                                                for _ in batch]})
                at = len(messages) - 1
                for j, lk in enumerate(batch):
                    plans[at, j] = lk
                    extra = dict(lk.flags)
                    messages.append(slot(kb, [r['record_id'] for r in lk.records],
                                         space_hint=SPACE_HINT.get(lk.kind), **extra))

        # search sites
        stages: list[list[Lookup]] = []
        if any(r['kind'] == 'passage' for r in wanted):
            for i, rec in enumerate(hop_order(need, [r for r in wanted if r['kind'] == 'passage'])):
                info = {'need': words(need, 30), 'keys': keys,
                        **fields_of('passage', rec['text'])}
                title = ('passage_title',) if 'title' in info else ()
                base = 'passage' if i == 0 else 'passage_hop'
                templates = [t for name in (*title, base) for t in TEMPLATES[name]]
                stages.append([Lookup('passage', [rec], fields=info, templates=tuple(templates))])
        first_tool_search = family == 'policy_tool_agent'
        for kinds in STAGES:
            if first_tool_search and kinds == ('tool_doc',):
                continue
            lookups = _lookups_for_stage(kinds, [r for r in wanted if r['kind'] != 'passage'],
                                         {'keys': keys, 'need': words(need, 30)}, rng, counts)
            if lookups:
                stages.append(lookups)
        if self.opt.gold_slots and split == 'train' and len(tokens(answer)) >= OWN_GOLD_TOKENS:
            gold = gold_record(kb, episode)
            self.gold_ids.add(gold['record_id'])
            counts['gold_slots'] += 1
            stages.insert(min(1, len(stages)), [Lookup(
                'gold_trajectory', [gold], fields={'keys': keys},
                flags={'gold': True, 'receding_weight': 1.0})])
            self.pending_gold = gold
        if spare and rng.random() < self.opt.distractor_rate:
            rec = spare[rng.randrange(len(spare))]
            info = {'keys': keys, **fields_of(rec['kind'], rec['text'])}
            templates = TEMPLATES['tool_doc'] if 'tool' in info else TEMPLATES['distractor']
            stages.insert(rng.randrange(len(stages) + 1), [Lookup(
                rec['kind'], [rec], fields=info, templates=templates, flags={'distractor': True})])
            counts['distractors'] += 1
        for lookups in stages:
            site(lookups)

        # the answer / trajectory
        turns = episode.get('turns') or []
        if family == 'function_call':
            calls = (episode.get('verify') or {}).get('gold')
            if calls is None:
                calls = json.loads(answer)
            messages.append({'role': 'assistant', 'content': '',
                             'tool_calls': [call(c['name'], c.get('arguments') or {}) for c in calls]})
        elif turns:
            docs = {fields_of('tool_doc', r['text']).get('tool'): r for r in wanted
                    if r['kind'] == 'tool_doc'}
            searched: set[str] = set()
            last_call = None
            for turn in turns:
                text = turn['text']
                if turn['role'] == 'assistant' and text.startswith('Call: ') and \
                        family == 'policy_tool_agent':
                    parsed = _parse_call(text[6:])
                    if parsed is None:
                        return None, 'unparsable_call', counts
                    name = parsed['function']['name']
                    if name in docs and name not in searched:
                        searched.add(name)
                        info = {'keys': keys, **fields_of('tool_doc', docs[name]['text'])}
                        site([Lookup('tool_doc', [docs[name]], fields=info)])
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [parsed]})
                    last_call = name
                elif turn['role'] == 'assistant':
                    messages.append({'role': 'assistant', 'content': text})
                elif family == 'policy_tool_agent' and text.startswith('Result: '):
                    messages.append({'role': 'tool', 'name': last_call, 'content': text[8:]})
                elif family == 'policy_tool_agent' and text.startswith('Customer: '):
                    messages.append({'role': 'user', 'content': text[10:]})
                else:
                    messages.append({'role': 'user', 'content': text})
            unsearched = [r for n, r in docs.items() if n not in searched]
            if unsearched and family == 'policy_tool_agent':
                counts['tool_docs_never_called'] += len(unsearched)
        else:
            messages.append({'role': 'assistant', 'content': answer})

        write_sites = []
        if writes:
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call(
                'memory_write', {'content': write_content(episode, family, user, dataset)})]})
            write_sites.append({'message': len(messages) - 1, 'call': 0, 'site': 'episode_end',
                                'source': 'own_trajectory'})
            messages.append({'role': 'tool', 'name': 'memory_write',
                             'content': {'write_result': {'kb': kb, 'status': 'stored'}}})

        # queries, in causal order
        answers = [answer] + [a for a in prov.get('answer_aliases') or [] if isinstance(a, str)]
        tools_text = json.dumps(tools, ensure_ascii=False)
        texts = {s['record_id']: s['text'] for s in supports.values()}
        if self.pending_gold is not None:
            texts[self.pending_gold['record_id']] = self.pending_gold['text']
        search_sites = []
        for (i, j), lk in sorted(plans.items()):
            prefix, future, allowed = call_context(messages, i, j, tools_text, texts)
            query, how = self._choose(lk, rng, prefix, future, allowed, answers)
            counts[f'query_{how}'] += 1
            messages[i]['tool_calls'][j]['function']['arguments']['query'] = query
            result = i + 1 + j
            search_sites.append({'message': i, 'call': j, 'result': result, 'kind': lk.kind,
                                 'records': len(lk.records), **lk.flags})

        row = {'episode_id': episode['episode_id'], 'kb': kb, 'split': split,
               'task_family': episode.get('task_family'), 'messages': messages, 'tools': tools,
               'answer': answer, 'verify': episode.get('verify'),
               'provenance': {**prov, 'source_query_time': qt},
               'search_sites': search_sites, 'write_sites': write_sites,
               'loss_mask': LOSS_POLICY}
        for key in ('capability', 'allowed_capability', 'choices', 'restore'):
            if key in episode:
                row.setdefault('episode_meta', {})[key] = episode[key]
        problems = self.audit(row, texts, answers, qt)
        if problems:
            counts.update({f'audit_{k}': v for k, v in problems.items()})
            return None, 'audit_failed:' + ','.join(sorted(problems)), counts
        counts['searches'] += len(search_sites)
        counts['search_sites'] += len({s['message'] for s in search_sites})
        counts['slot_records'] += sum(s['records'] for s in search_sites)
        counts['writes'] += len(write_sites)
        return row, None, counts

    def _choose(self, lk: Lookup, rng, prefix, future, allowed, answers):
        known = set(prefix)
        candidates = [fill(t, lk.fields) for t in lk.templates]
        candidates = list(dict.fromkeys(c for c in candidates if c))
        rng.shuffle(candidates)
        title = lk.fields.get('title')
        if title and not grounded(title, known):
            # an ungrounded title is allowed (the record's header) but tried last
            candidates.sort(key=lambda c: title.lower() in c.lower())
        for c in candidates:
            if self.hook is not None:
                c = clean(self.hook(query=c, kind=lk.kind, need=lk.fields.get('need', '')) or c)
            if leak_reason(c, prefix=prefix, future=future, allowed=allowed, answers=answers,
                           n=self.opt.ngram) is None:
                how = 'template'
                if title and title.lower() in c.lower() and not grounded(title, known):
                    how = 'title_ungrounded'
                elif any(contains(tokens(c), tokens(a)) for a in answers if a and len(a) <= SHORT_ANSWER):
                    how = 'answer_from_prefix'
                return c, how
        generic = GENERIC.get(lk.kind, lk.kind.replace('_', ' '))
        return generic, 'generic'

    def audit(self, row: dict, texts: dict, answers: list[str], query_time: int) -> Counter:
        """Independent re-check of a finished transcript; returns violations."""
        bad: Counter = Counter()
        messages = row['messages']
        tools_text = json.dumps(row['tools'], ensure_ascii=False)
        for i, m in enumerate(messages):
            for j, tc in enumerate(m.get('tool_calls') or []):
                if tc['function']['name'] != 'memory_search':
                    continue
                result = messages[i + 1 + j]
                body = result['content']['slot'] if isinstance(result.get('content'), dict) else None
                if result['role'] != 'tool' or body is None:
                    bad['slot_not_after_call'] += 1
                    continue
                ids = body['record_ids']
                if body.get('gold'):
                    if row['split'] != 'train':
                        bad['gold_outside_train'] += 1
                    if not all(r in self.gold_ids for r in ids) or 'receding_weight' not in body:
                        bad['gold_unflagged'] += 1
                else:
                    for r in ids:
                        if r not in self.index:
                            bad['record_not_in_kb'] += 1
                        elif self.index[r][0] >= query_time:
                            bad['record_after_query'] += 1
                        elif self.kb_of(r) != row['kb']:
                            bad['record_other_kb'] += 1
                        if r in self.gold_ids:
                            bad['gold_record_unflagged'] += 1
                query = tc['function']['arguments'].get('query')
                if not query:
                    bad['empty_query'] += 1
                    continue
                prefix, future, allowed = call_context(messages, i, j, tools_text, texts)
                reason = leak_reason(query, prefix=prefix, future=future, allowed=allowed,
                                     answers=answers, n=self.opt.ngram)
                if reason:
                    bad[f'query_leak_{reason}'] += 1
        return bad


# -- corpus runner ------------------------------------------------------------------------
def episode_files(corpus: Path) -> dict[str, Path]:
    out = {}
    for split in ('train', 'validation', 'test'):
        for name in (f'episodes-{split}.jsonl', f'{split}-episodes.jsonl'):
            if (corpus / name).exists():
                out[split] = corpus / name
    return out


def corpus_name(corpus: Path) -> str:
    name = re.sub(r'^tasks-', '', corpus.name)
    return re.sub(r'-20\d{6}', '', name)


def load_index(corpus: Path) -> tuple[dict, dict[str, list[str]], set[str]]:
    """record_id -> (created_at, kind, domain); tool names per area; domains."""
    index, tools, domains = {}, {}, set()
    with (corpus / 'sources.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            rec = json.loads(line)
            index[rec['record_id']] = (int(rec['created_at']), rec['kind'], rec.get('domain', ''))
            domains.add(rec.get('domain', ''))
            if rec['kind'] == 'tool_doc':
                area = (rec.get('provenance') or {}).get('area')
                name = (rec.get('provenance') or {}).get('tool')
                if area and name:
                    tools.setdefault(area, []).append(name)
    return index, {a: sorted(set(n)) for a, n in tools.items()}, domains


def run_corpus(corpus: Path, output: Path, options: Options, *, limit: int | None = None,
               overwrite: bool = False, tokenizer: str | None = None,
               render_count: int = 50) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    kb = corpus_name(corpus)
    index, tool_names, domains = load_index(corpus)
    builder = Builder(kb, index, options, per_domain=len(domains) > 1, tool_names=tool_names)
    tmp = output.with_name(output.name + '.pending')
    tmp.mkdir(parents=True, exist_ok=True)
    summary, digests = {}, {}
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    gold_handle = (tmp / 'gold-records-train.jsonl').open('w', encoding='utf-8') \
        if options.gold_slots else None
    for split, path in episode_files(corpus).items():
        counts, rejects, per_episode, rendered = Counter(), Counter(), Counter(), Counter()
        digest = hashlib.sha256()
        written = seen = 0
        with path.open(encoding='utf-8') as src, \
                (tmp / f'transcripts-{split}.jsonl').open('w', encoding='utf-8') as out:
            for line in src:
                if limit is not None and seen >= limit:
                    break
                seen += 1
                episode = json.loads(line)
                builder.pending_gold = None
                row, reason, got = builder.build(episode, split)
                counts.update(got)
                if row is None:
                    rejects[reason] += 1
                    continue
                if gold_handle is not None and builder.pending_gold is not None:
                    gold_handle.write(json.dumps(builder.pending_gold, ensure_ascii=False) + '\n')
                text = json.dumps(row, ensure_ascii=False) + '\n'
                out.write(text)
                digest.update(text.encode())
                per_episode[len(row['search_sites'])] += 1
                if tok is not None and written < render_count:
                    rendered['checked'] += 1
                    rendered.update(render_check(tok, row))
                written += 1
        digests[f'transcripts-{split}.jsonl'] = digest.hexdigest()
        summary[split] = {'input': seen, 'written': written, 'rejected': dict(rejects),
                          'counts': dict(sorted(counts.items())),
                          'searches_per_episode': dict(sorted(per_episode.items())),
                          'render_check': dict(rendered)}
    if gold_handle is not None:
        gold_handle.close()
        digests['gold-records-train.jsonl'] = hashlib.sha256(
            (tmp / 'gold-records-train.jsonl').read_bytes()).hexdigest()
    source_manifest = corpus / 'manifest.json'
    manifest = {'format': 1, 'input': str(corpus), 'kb': kb,
                'kb_per_domain': builder.per_domain,
                'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                if source_manifest.exists() else None,
                'options': {'seed': options.seed, 'distractor_rate': options.distractor_rate,
                            'gold_slots': options.gold_slots, 'writes': options.writes,
                            'sequential_rate': options.sequential_rate,
                            'query_hook': options.query_hook, 'ngram': options.ngram,
                            'limit': limit},
                'memory_tools': [t['name'] for t in MEMORY_TOOLS],
                'slot_render': f'{SPAN_TOKENS["mem"][0]} latent span {SPAN_TOKENS["mem_end"][0]}',
                'loss_mask': LOSS_POLICY, 'splits': summary, 'sha256': digests}
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if output.exists():
        import shutil
        shutil.rmtree(output)
    os.replace(tmp, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corpora', nargs='+', type=Path, help='corpus directories')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tag', default='20260928', help='date tag of output directories')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--distractor-rate', type=float, default=0.0,
                        help='share of episodes with one unhelpful search (default off)')
    parser.add_argument('--gold-slots', action='store_true',
                        help='B9: add the own gold trajectory as a flagged, receding-weight '
                             'record (train only)')
    parser.add_argument('--writes', choices=('trajectory', 'all', 'none'), default='trajectory')
    parser.add_argument('--sequential-rate', type=float, default=0.3,
                        help='share of multi-call sites split into consecutive single calls')
    parser.add_argument('--query-hook', help='module:function(query=, kind=, need=) -> str, '
                        'e.g. a local-LLM rewriter; its output is audited like any query')
    parser.add_argument('--ngram', type=int, default=NGRAM)
    parser.add_argument('--limit', type=int, help='episodes per split (smoke runs)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--render-check', metavar='TOKENIZER_DIR',
                        help='render the first --render-count transcripts per split through '
                             'this tokenizer\'s chat template and check the protocol')
    parser.add_argument('--render-count', type=int, default=50)
    args = parser.parse_args()
    options = Options(seed=args.seed, distractor_rate=args.distractor_rate,
                      gold_slots=args.gold_slots, writes=args.writes,
                      sequential_rate=args.sequential_rate, query_hook=args.query_hook,
                      ngram=args.ngram)
    jobs = [(c, args.output_root / f'memory-{corpus_name(c)}-{args.tag}') for c in args.corpora]

    def report(corpus, manifest):
        brief = {s: {k: v[k] for k in ('input', 'written', 'rejected')}
                 for s, v in manifest['splits'].items()}
        print(json.dumps({'corpus': corpus.name, 'kb': manifest['kb'], **brief}), flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(run_corpus, c, o, options, limit=args.limit,
                                   overwrite=args.overwrite, tokenizer=args.render_check,
                                   render_count=args.render_count): c for c, o in jobs}
            for future, corpus in futures.items():
                report(corpus, future.result())
    else:
        for corpus, output in jobs:
            report(corpus, run_corpus(corpus, output, options, limit=args.limit,
                                      overwrite=args.overwrite, tokenizer=args.render_check,
                                      render_count=args.render_count))


if __name__ == '__main__':
    main()
