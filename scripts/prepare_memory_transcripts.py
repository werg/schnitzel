"""Memory-protocol chat transcripts for LFM2, version 3 (knowledge-base stack WP3, restart
plan 3.2).

Every episode of an R6 or task corpus becomes one LFM2 chat transcript in the
structured form ``tok.apply_chat_template(messages, tools=tools)`` renders:

- ``system``: a memory-use policy (and the task family's role);
- ``user``: the task, with the corpus prompt's "Use the stored ..." preamble removed
  (the knowledge base is reached by tool calls now, not announced in the prompt);
- search sites: an ``assistant`` message whose ``tool_calls`` are one or more
  ``memory_search()`` calls WITHOUT arguments (owner, 28 Sep: the query is a vector,
  the decoder's hidden state at the call projected by one key head per KB space),
  each followed by a ``tool`` message whose content is a latent SLOT
  ``{"slot": {"kb", "record_ids", ...}}``; the trainer renders it as ``<|mem|>`` +
  latent span + ``<|/mem|>`` (``schnitz.span_tokens``). The target record ids of every
  call are in its slot and in the row's ``search_sites`` entry (with the placement
  ``step`` and ``trigger``);
- redundant copies: an episode with ``alternatives`` (per hop, every record that alone
  states that hop's fact; the synthetic worlds of ``schnitz.synth_world``) names them in
  the slot of the hop's read (``alternatives``, audited like the slot's records), so the
  L1 bank build (``schnitz.kb.bank.needed_records``) stores every copy; the slot's
  ``record_ids`` stay the one copy the transcript reads;
- the answer: an ``assistant`` message (text, or native tool calls for function
  calling); multi-turn corpora keep their ``turns`` (customers and observations as
  ``user``, API results as ``tool``, agent API calls as native tool calls);
- write sites (``--writes``): a final ``assistant`` message with a ``memory_write()``
  call WITHOUT arguments and a ``write_span`` field: in the same assistant turn the
  model generates a ``<|bg|>`` ... ``<|/bg|>`` span itself (owner, 28 Sep: single-pass
  latent writes). The renderer puts the placeholder ``<|bg|><|/bg|>`` right after the
  call, inside the assistant turn (the trainer fills the reps; the open and close
  decisions carry loss, the reps no token loss). The text a write should hold is kept
  only in ``write_sites[i]["teacher_text"]``, a teacher target for B4 distillation that
  is never rendered into the trained tokens. A ``tool`` ack follows.

Which records a search returns comes from the source episode (``required_ids``, the
first valid ``sufficient_groups`` entry for R6, related ``supports`` such as column
values, worked examples and background). ScienceWorld episodes whose source attaches
no worked example get up to three from the KB's held-out pool of training
trajectories of the same task type (``--pool-examples``).

Placement (single-shot tasks): all searches precede the answer, grouped into stages
(rules/protocol, schema, values, evidence, tool docs, know-how, examples, background);
multi-hop passages are searched one hop per site, ordered so that a hop's title is
grounded in the question or in passages read before it; parallel-recall passages (other
translations of the requested verses) are searched with one call per translation.

Placement (trajectories), per record; a site sits right before the agent turn
``step`` (0 = before the first turn):

- protocol, rules, background, the general agent-policy section: step 0 (``start``);
- tool documentation: before the first call of that tool (``action_tool``); a
  policy section on an action (cancel, modify, return, ...) before the first call of
  a state-changing tool whose verb the section names (``action_tool``);
- know-how (ALFWorld floorplans): before the first action that names an object or
  location the record lists (``action_entity``);
- worked examples: before the first action that uses the example's most specific
  command (the example's command that is rarest among the KB's worked examples,
  e.g. ``heat``, ``use``, ``click``, ``argmax``, ``awk``, SQL ``SELECT``; terminal
  commands such as ``answer``/``submit`` never count, exploration commands such as
  ``go``, ``look``, ``open``, ``ls``, ``DESC`` only when nothing else matches), falling
  back to its next most specific command the trajectory uses (``action_command``);
  with ``--example-reads`` 2 (default) it is placed before the first uses of its two
  most specific commands the trajectory uses, the later one a re-read
  (``action_command_reread``);
- a record none of whose cues the trajectory uses is searched at step 0
  (``start_fallback``); records placed at the same step share a site (one call per
  kind, table or tool as in single-shot stages; the call's ``trigger`` joins theirs);
- after an observation that reports a failed action (``--failure-rereads``, at most
  once per episode) the standing records (protocol, policy, rules) are searched again
  (``observation_failure``).

The ``action_*`` triggers place a search before the agent's own next action (as
the tool-doc placement of version 1): the placement is label-side, a demonstration
of when to call, like tool-call placement in SFT (accepted by the owner, 28 Sep);
the call itself carries no content, and everything before it is the unchanged
causal prefix. ``observation_failure`` uses only the prefix.

Enforced checks (counts in ``manifest.json``): every ``memory_search`` and
``memory_write`` call has empty arguments, every write has a ``write_span`` and a
``write_sites`` entry whose teacher text appears in no message; every slot record
exists in the episode's KB (``sources.jsonl`` of the corpus; for R6 each dataset
domain is its own KB) with ``created_at`` before the episode's ``query_time`` and in
the transcript's KB; slots never contain a record
that copies the episode's own long answer (its gold trajectory) unless it is an
explicit ``--gold-slots`` record (train only, flagged ``gold`` with a
``receding_weight``, restart plan B9); validation and test transcripts never
reference gold records; the source messages (request, turns) appear unchanged and in
their source order, so every call's prefix is a prefix of the source episode; the
generated system prompt copies no n-gram of ``--ngram`` tokens from the answer or
later turns that the request lacks. Output is deterministic (per-episode RNG from
``--seed`` and the episode id) and hashed in the manifest.

Loss policy: every assistant message (content and tool calls, including
``memory_search`` and ``memory_write`` calls and the write span's ``<|bg|>`` and
``<|/bg|>``; the reps between them have no token loss) gets loss; system, user and
tool content never does. The LFM2 template marks assistant spans with
``{% generation %}``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schnitz.span_tokens import MEMORY_TOOLS, SPAN_TOKENS  # noqa: E402
# the renderer is shared with the trainers (B4c write sites, L1)
from schnitz.memory_transcripts import WRITE_CALL, WRITE_FILL, render_ids, render_text  # noqa: E402

FORMAT = 3
NGRAM = 5
OWN_GOLD_TOKENS = 12        # answers shorter than this are not checked for copies
OWN_GOLD_OVERLAP = 0.5      # share of a record's 8-grams inside the answer = a copy
COPY_KINDS = {'worked_example', 'know_how', 'protocol', 'policy', 'background', 'rules'}
WRITE_CHARS = 900           # longer write contents are cut (trajectories) or skipped (single-shot)
POOL_EXAMPLES = 3           # pooled worked examples per episode without any
LOSS_POLICY = {'loss_on': 'assistant', 'assistant_parts': ['content', 'tool_calls'],
               'no_loss': ['system', 'user', 'tool'], 'template_generation_tags': True}

SYSTEM = (
    'You can consult a knowledge base. Call memory_search() whenever you need knowledge; '
    'each result is a memory span you read directly. Look things up before you rely on them.',
    'A knowledge base holds the reference material for this task: documentation, schemas, '
    'rules, examples and facts. Call memory_search() when you need some of it, as often as '
    'needed.',
    'When you need knowledge, call memory_search(); results arrive as memory spans.',
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
    'parallel_recall': 'Other translations of the requested text are in the knowledge base.',
}
WRITE_POLICY = 'When the task is done, store reusable results with memory_write().'

# search stages, in order; kinds of one stage share a site
STAGES = (('protocol', 'policy', 'rules'), ('schema',), ('column_values', 'table_rows'),
          ('evidence',), ('tool_doc',), ('know_how',), ('worked_example',), ('parallel_passage',),
          ('background',))
STANDING = ('protocol', 'policy', 'rules')
SPACE_HINT = {'column_values': 'fine', 'table_rows': 'fine', 'schema': 'fine',
              'evidence': 'fine', 'tool_doc': 'fine', 'background': 'coarse',
              'worked_example': 'coarse', 'protocol': 'coarse', 'policy': 'coarse',
              'know_how': 'coarse', 'rules': 'coarse', 'parallel_passage': 'fine'}
HEADERS = {
    'schema': r'^Database (?P<db>\S+), table (?P<table>.+?) \(schema\):',
    'column_values': r'^Database (?P<db>\S+), values of (?P<table>[^.\n]+)\.(?P<column>[^:\n]+):',
    'table_rows': r'^Database (?P<db>\S+), table (?P<table>.+?) \(',
    'evidence': r'^Database (?P<db>\S+), note:',
    'tool_doc': r'^(?:Tool: (?P<tool>\S+)|(?P<area>\w+) tool (?P<tool2>[^:\s]+):)',
    'passage': r'^Title: (?P<title>[^\n]+)',
    'parallel_passage': r'^(?P<title>[^\n]+?) \[(?P<version>[^\]\n]+)\], (?P<ref>[^\n]+)',
}
STOP = set('''a an the of in on at to for from by with and or but is are was were be been being
do does did has have had what which who whom whose when where why how that this these those it
its as into than then there their they he she his her them i you your we our me my not no yes
can could would should will shall may might must if so such any all some each about after
before over under between during also only just very more most other one two'''.split())
MAX_SLOT = {'worked_example': 4, 'background': 4}  # coarse slots keep a sample of their records

# writes: families whose single-shot result is reusable later (see --writes)
TRAJECTORY_FAMILIES = ('agent', 'policy_tool_agent')
REUSABLE_FAMILIES = ('text_to_sql', 'stored_table_qa', 'function_call', 'code',
                     'hotpot_multihop', 'public_multihop_qa', 'public_claim_verification')

# trajectories: commands that end an episode never place a search; exploration commands
# place one only when an example has no other command the trajectory uses
TERMINAL = {'answer', 'submit', 'finish', 'final', 'stop', 'exit'}
EXPLORATION = {'go', 'goto', 'look', 'open', 'close', 'teleport', 'inventory', 'examine',
               'move_ahead', 'turn_left', 'turn_right', 'turn_around', 'wait', 'wait1', 'scroll',
               'back', 'ls', 'cd', 'cat', 'pwd', 'echo', 'show', 'desc', 'describe'}
# read-only API verbs; other tool verbs change state and pull their policy section
READ_ONLY = {'get', 'list', 'search', 'find', 'calculate', 'think', 'transfer', 'check'}
FAILURE = re.compile(r'^(?:Observation:\s*)?(?:Nothing happens|No known action|Invalid action|'
                     r'Error|ERROR|.{0,40}\berror\b|.{0,60}No such file|.{0,40}command not found)',
                     re.IGNORECASE)
ACTION = re.compile(r'(?:^|\n)[ \t]*(?:Action|Act)[ \t]*:[ \t]*', re.IGNORECASE)
CODE = re.compile(r'```(\w*)[ \t]*\n?(.*?)(?:```|$)', re.DOTALL)
SQL_START = re.compile(r'\s*(select|show|desc|describe|insert|update|delete|create|drop|alter|'
                       r'with|replace)\b', re.IGNORECASE)

# ETO's ScienceWorld task indices (the numeric ``group`` of its trajectories), matched to
# the task names of the cp2107 trajectories by the template words of their task texts
# (majority vote over the training episodes; 23 of 24 indices unanimous, 11: 93 of 120)
SCIENCEWORLD_ETO_TASKS = {
    '0': 'boil', '1': 'change-the-state-of-matter-of', '2': 'freeze', '3': 'melt',
    '4': 'measure-melting-point-known-substance', '6': 'use-thermometer',
    '7': 'power-component', '8': 'power-component-renewable-vs-nonrenewable-energy',
    '9': 'test-conductivity', '10': 'test-conductivity-of-unknown-substances',
    '11': 'find-animal', '12': 'find-living-thing', '13': 'find-non-living-thing',
    '14': 'find-plant', '15': 'grow-fruit', '16': 'grow-plant', '17': 'chemistry-mix',
    '18': 'chemistry-mix-paint-secondary-color', '19': 'chemistry-mix-paint-tertiary-color',
    '20': 'lifespan-longest-lived', '21': 'lifespan-longest-lived-then-shortest-lived',
    '22': 'lifespan-shortest-lived', '23': 'identify-life-stages-1',
    '24': 'identify-life-stages-2'}


# -- text helpers -----------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', (text or '').lower())


def ngrams(toks: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def fields_of(kind: str, text: str) -> dict:
    found = re.match(HEADERS.get(kind, r'(?!)'), text or '')
    if not found:
        return {}
    got = {k: v for k, v in found.groupdict().items() if v}
    if 'tool2' in got:
        got['tool'] = got.pop('tool2')
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
    return ' '.join(text.split()[:limit])


# -- agent actions -------------------------------------------------------------------
def action_parts(text: str) -> list[str]:
    """The action parts of a turn or worked example: the text after each ``Action:`` /
    ``Act:`` marker; a turn with none but a final answer counts as ``answer``."""
    parts = ACTION.split(text or '')[1:]
    parts = [p for p in parts if p.strip()]
    if not parts and re.search(r'Final Answer\s*:|^Answer\s*:', text or '', re.MULTILINE):
        return ['answer']
    return parts


def commands_of(part: str) -> list[str]:
    """Command names of one action: the SQL keyword of a SQL block, each command of a
    shell block, else the leading word (``go``, ``click``, ``get_neighbors``, ...)."""
    code = CODE.search(part)
    if code:
        lang, body = code.group(1).lower(), code.group(2)
        if lang == 'sql' or (lang != 'bash' and SQL_START.match(body)):
            found = SQL_START.match(body) or re.match(r'\s*([A-Za-z]+)', body)
            return [found.group(1).lower()] if found else []
        out = []
        for seg in re.split(r'\|\||&&|[|;\n]', body):
            found = re.match(r'\s*(?:sudo\s+)?([A-Za-z_][\w.-]*)', seg)
            if found:
                out.append(found.group(1).lower())
        return out
    found = re.match(r'\s*([A-Za-z_][\w-]*)', part)
    return [found.group(1).lower()] if found else []


def record_commands(text: str) -> list[str]:
    """Commands of a worked example in order of first use (duplicates dropped)."""
    return list(dict.fromkeys(c for part in action_parts(text) for c in commands_of(part)))


def know_how_cues(text: str) -> set[str]:
    """Objects and locations a know-how record lists ("item: place 1, place 2")."""
    cues = set()
    for line in (text or '').split('\n')[1:]:
        if ':' not in line:
            continue
        item, places = line.split(':', 1)
        cues.add(clean(item).lower())
        cues.update(clean(p).lower() for p in places.split(','))
    return {c for c in cues if c}


def has_phrase(text: str, phrase: str) -> bool:
    return re.search(r'(?<![\w])' + re.escape(phrase) + r'(?![\w])', text) is not None


def verb_stem(tool: str) -> str:
    verb = tool.split('_')[0].lower()
    return verb[:-1] if verb.endswith('e') and len(verb) > 4 else verb


def task_type(dataset: str, group: str | None) -> str | None:
    """Task type of a ScienceWorld trajectory group (cp2107 ``gold:<task>:variation:N``,
    ETO numeric index); None elsewhere (no pooling)."""
    if dataset != 'scienceworld' or not group:
        return None
    if group.startswith('gold:'):
        return group.split(':')[1]
    return SCIENCEWORLD_ETO_TASKS.get(group.split('_')[0])


# -- audit helpers -------------------------------------------------------------------
def leak_reason(text: str, *, prefix: list[str], future: list[str], n: int = NGRAM) -> str | None:
    """'ngram' if ``text`` copies an n-gram of a ``future`` text that the prefix lacks."""
    mine = ngrams(tokens(text), n)
    ok = ngrams(prefix, n)
    for other in future:
        if mine & (ngrams(tokens(other), n) - ok):
            return 'ngram'
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


def message_text(message: dict) -> str:
    """Plain text of a message (slots, write acks and memory calls excluded)."""
    parts = []
    content = message.get('content')
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, dict) and 'slot' not in content and 'write_result' not in content:
        parts.append(json.dumps(content, ensure_ascii=False))
    for tc in message.get('tool_calls') or []:
        fn = tc['function']
        if fn['name'] in ('memory_search', 'memory_write'):
            continue
        parts.append(fn['name'] + ' ' + json.dumps(fn['arguments'], ensure_ascii=False))
    return '\n'.join(parts)


def is_memory(message: dict) -> bool:
    """A memory call, a search slot or a write ack."""
    content = message.get('content')
    if isinstance(content, dict) and ('slot' in content or 'write_result' in content):
        return True
    calls = message.get('tool_calls') or []
    return bool(calls) and all(tc['function']['name'] in ('memory_search', 'memory_write')
                               for tc in calls)


def render_check(tok, row: dict) -> Counter:
    """Render ``row`` through the real chat template and check the protocol: one
    tool-call block per calling assistant message, ``memory_search()`` rendered without
    arguments, an empty ``<|mem|><|/mem|>`` pair per slot outside the loss mask, every
    memory call inside it and no system, user or tool token in it."""
    bad: Counter = Counter()
    messages, tools = row['messages'], row['tools']
    ids, mask = render_ids(tok, messages, tools)
    if tok.decode(ids) != render_text(messages, tools, tok):
        bad['render_ids_text_mismatch'] += 1
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
    searches = sum(tc['function']['name'] == 'memory_search' for m in messages
                   for tc in m.get('tool_calls') or [])
    if trained.count('memory_search()') != searches or \
            text.count('memory_search(') != text.count('memory_search()'):
        bad['memory_search_not_empty_in_loss'] += 1
    writes = sum(tc['function']['name'] == 'memory_write' for m in messages
                 for tc in m.get('tool_calls') or [])
    if trained.count(WRITE_CALL + WRITE_FILL) != writes or \
            text.count('memory_write(') != text.count('memory_write()'):
        bad['memory_write_not_empty_with_span_in_loss'] += 1
    spans = [i for i, t in enumerate(ids) if t in (SPAN_TOKENS['bg'][1], SPAN_TOKENS['bg_end'][1])]
    if len(spans) != 2 * writes or not all(mask[i] for i in spans):
        bad['write_span_tokens'] += 1
    for site in row.get('write_sites') or []:
        teacher = site.get('teacher_text') or ''
        if teacher and clean(teacher) in clean(text):
            bad['teacher_text_rendered'] += 1
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


# -- lookups --------------------------------------------------------------------------
class Lookup:
    """One memory_search call to be placed: its records, placement and flags."""

    def __init__(self, kind: str, records: list[dict], *, flags: dict | None = None,
                 step: int = 0, trigger: str = 'start'):
        self.kind, self.records = kind, records
        self.flags = flags or {}
        self.step, self.trigger = step, trigger


def lookups_for_stage(kinds: tuple, records: list[dict], rng: random.Random,
                      counts: Counter, **placement) -> list[Lookup]:
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
        elif rec['kind'] == 'parallel_passage':     # one call per stored translation
            key = (rec['kind'], info.get('version'))
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
        out.append(Lookup(key[0], recs, **placement))
    return out


def staged(records: list[dict], rng, counts, **placement) -> list[list[Lookup]]:
    """Records of one site as stages (each a list of calls), in STAGES order."""
    stages = []
    for kinds in STAGES:
        lookups = lookups_for_stage(kinds, records, rng, counts, **placement)
        if lookups:
            stages.append(lookups)
    return stages


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


# -- trajectory placement ------------------------------------------------------------------
class Step:
    """One agent turn of a trajectory, as the placement rules see it."""

    def __init__(self, text: str, tool: str | None = None):
        parts = action_parts(text) if tool is None else []
        self.commands = {c for p in parts for c in commands_of(p)}
        self.action = '\n'.join(parts).lower()
        self.tool = tool


def place_records(records: list[dict], steps: list[Step], family: str, command_df: Counter,
                  tools: list[str], example_reads: int = 2) -> dict[str, list[tuple]]:
    """record_id -> [(step, trigger), ...] for a trajectory (module docstring); step None
    = the record is never searched (documentation of a tool the episode never calls). A
    worked example is read before the first use of each of its ``example_reads`` most
    specific commands that the trajectory uses, at distinct steps."""
    first_tool: dict[str, int] = {}
    for k, s in enumerate(steps):
        if s.tool:
            first_tool.setdefault(s.tool, k)
    out = {}
    for rec in records:
        kind, text = rec['kind'], rec['text']
        where: tuple[int | None, str] = (0, 'start')
        extra = []
        if kind == 'tool_doc':
            name = fields_of('tool_doc', text).get('tool')
            where = (first_tool[name], 'action_tool') if name in first_tool else (None, 'unused')
        elif kind == 'policy' and family == 'policy_tool_agent' and '\n# ' not in text:
            low = text.lower()
            hits = [first_tool[t] for t in tools if t in first_tool
                    and verb_stem(t) not in READ_ONLY and len(verb_stem(t)) >= 3
                    and re.search(r'\b' + re.escape(verb_stem(t)), low)]
            if hits:
                where = (min(hits), 'action_tool')
        elif kind == 'know_how':
            cues = know_how_cues(text)
            hit = next((k for k, s in enumerate(steps)
                        if s.action and any(has_phrase(s.action, c) for c in cues)), None)
            where = (hit, 'action_entity') if hit is not None else (0, 'start_fallback')
        elif kind == 'worked_example':
            cmds = [c for c in record_commands(text) if c not in TERMINAL]
            order = sorted(range(len(cmds)), key=lambda i: (
                cmds[i] in EXPLORATION, command_df.get(cmds[i], 0), -i))
            hits = []
            for i in order:
                hit = next((k for k, s in enumerate(steps) if cmds[i] in s.commands), None)
                if hit is not None and hit not in hits:
                    hits.append(hit)
                    if len(hits) >= example_reads:
                        break
            hits.sort()
            where = (hits[0], 'action_command') if hits else (0, 'start_fallback')
            extra = [(k, 'action_command_reread') for k in hits[1:]]
        out[rec['record_id']] = [where] + extra
    return out


# -- transcript builder ----------------------------------------------------------------
class Options:
    def __init__(self, seed=0, distractor_rate=0.0, gold_slots=False, writes='reusable',
                 sequential_rate=0.3, ngram=NGRAM, pool_examples=True, failure_rereads=True,
                 example_reads=2):
        self.seed, self.distractor_rate, self.gold_slots = seed, distractor_rate, gold_slots
        self.writes, self.sequential_rate, self.ngram = writes, sequential_rate, ngram
        self.pool_examples, self.failure_rereads = pool_examples, failure_rereads
        self.example_reads = example_reads


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


def write_content(episode: dict, family: str, user: str, dataset: str) -> str | None:
    """Reusable content from the episode's own result (None: nothing reusable fits)."""
    need = need_text(user, dataset)
    turns = episode.get('turns') or []
    answer = episode.get('answer', '')
    if family == 'agent':
        actions = [re.sub(r'^.*?Action:\s*', '', t['text'], flags=re.DOTALL).strip()
                   for t in turns if t['role'] == 'assistant']
        actions = [clean(a)[:120] for a in actions if a]
        if len(actions) > 16:
            actions = actions[:8] + ['...'] + actions[-7:]
        text = f'Task: {words(need, 40)}\nSolved in {sum(t["role"] == "assistant" for t in turns)} ' \
               f'steps: ' + '; '.join(actions)
        return text[:WRITE_CHARS]
    if family == 'policy_tool_agent':
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
        return text[:WRITE_CHARS]
    db = re.match(r'Database (\S+?)\. ', user)
    db = f'Database {db.group(1)}. ' if db else ''
    labels = {'text_to_sql': ('Question', 'SQL'), 'stored_table_qa': ('Question', 'Answer'),
              'function_call': ('Request', 'Calls'), 'code': ('Task', 'Solution'),
              'public_claim_verification': ('Claim', 'Verdict')}
    ask, result = labels.get(family, ('Question', 'Answer'))
    sep = '\n' if family == 'code' else ' '
    text = f'{db}{ask}: {words(need, 60)}\n{result}:{sep}{answer.strip()}'
    return text if len(text) <= WRITE_CHARS else None


class Builder:
    """Turns source episodes of one corpus into transcripts, with checks and counts."""

    def __init__(self, kb: str, index: dict, options: Options, *, per_domain: bool,
                 tool_names: dict[str, list[str]] | None = None,
                 command_df: Counter | None = None, pool: dict | None = None):
        self.kb, self.index, self.opt, self.per_domain = kb, index, options, per_domain
        self.tool_names = tool_names or {}
        self.command_df = command_df or Counter()
        self.pool = pool or {}          # task type -> worked-example records (held-out pool)
        self.gold_ids: set[str] = set()
        self.pending_gold: dict | None = None

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

    def _pooled(self, episode: dict, dataset: str, qt: int, kb: str, answer_grams: set,
                rng: random.Random, counts: Counter) -> list[dict]:
        """Up to POOL_EXAMPLES worked examples of the episode's task type from the KB's
        held-out pool (never the episode's own trajectory)."""
        prov = episode.get('provenance', {})
        own = re.sub(r'^[^-]+-', '', episode['episode_id'])
        kind = task_type(dataset, prov.get('group') or own)
        picks = []
        for rec in self.pool.get(kind, []):
            group = (rec.get('provenance') or {}).get('group')
            if group == own or group == prov.get('group') and not str(group).isdigit():
                counts['pool_own_skipped'] += 1
                continue
            if not self._valid(rec, qt, counts) or self.kb_of(rec['record_id']) != kb:
                continue
            if answer_grams and own_gold(answer_grams, rec['text']):
                counts['pool_own_gold_skipped'] += 1
                continue
            picks.append(rec)
        if len(picks) > POOL_EXAMPLES:
            picks = sorted(rng.sample(picks, POOL_EXAMPLES), key=lambda r: r['record_id'])
        return picks

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
        if self.opt.pool_examples and family == 'agent' and self.pool and \
                not any(r['kind'] == 'worked_example' for r in wanted):
            pooled = self._pooled(episode, dataset, qt, kb, answer_grams, rng, counts)
            if pooled:
                counts['pooled_examples'] += len(pooled)
                counts['pooled_episodes'] += 1
                wanted += pooled

        user = strip_preamble(episode['query'], dataset)
        need = need_text(user, dataset)

        # tools and system prompt
        tools = list(MEMORY_TOOLS)
        area = prov.get('area') or episode.get('verify', {}).get('area') or ''
        if family == 'function_call':
            names = [fields_of('tool_doc', s['text']).get('tool') for s in supports.values()
                     if s['kind'] == 'tool_doc']
            tools += [tool_stub(n) for n in dict.fromkeys(n for n in names if n)]
        elif family == 'policy_tool_agent':
            tools += [tool_stub(n) for n in self.tool_names.get(area, [])]
        writes = self.opt.writes == 'all' or \
            (self.opt.writes in ('trajectory', 'reusable') and family in TRAJECTORY_FAMILIES) or \
            (self.opt.writes == 'reusable' and family in REUSABLE_FAMILIES)
        content = write_content(episode, family, user, dataset) if writes else None
        if writes and content is None:
            counts['write_skipped_long'] += 1
        system = SYSTEM[rng.randrange(len(SYSTEM))]
        role = ROLE.get(family, '').format(area=area)
        system = ' '.join(p for p in (system, role, WRITE_POLICY if content else '') if p)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        source_index = [None, -1]    # per message: index of its source turn (-1 = request)
        plans: dict[tuple[int, int], Lookup] = {}

        def site(lookups: list[Lookup]):
            sequential = len(lookups) > 1 and rng.random() < self.opt.sequential_rate
            batches = [[lk] for lk in lookups] if sequential else [lookups]
            for batch in batches:
                messages.append({'role': 'assistant', 'content': '',
                                 'tool_calls': [call('memory_search', {}) for _ in batch]})
                source_index.append(None)
                at = len(messages) - 1
                for j, lk in enumerate(batch):
                    plans[at, j] = lk
                    messages.append(slot(kb, [r['record_id'] for r in lk.records],
                                         space_hint=SPACE_HINT.get(lk.kind), **lk.flags))
                    source_index.append(None)

        # placement
        turns = episode.get('turns') or []
        steps: list[Step] = []
        for turn in turns:
            if turn['role'] != 'assistant':
                continue
            text = turn['text']
            if family == 'policy_tool_agent' and text.startswith('Call: '):
                parsed = _parse_call(text[6:])
                if parsed is None:
                    return None, 'unparsable_call', counts
                steps.append(Step('', tool=parsed['function']['name']))
            else:
                steps.append(Step(text))
        start_records, later = [], {}
        if turns:
            where = place_records([r for r in wanted if r['kind'] != 'passage'], steps, family,
                                                  self.command_df, self.tool_names.get(area, []),
                                  self.opt.example_reads)
            for rec in wanted:
                if rec['kind'] == 'passage':
                    start_records.append(rec)
                    continue
                for step, trigger in where[rec['record_id']]:
                    counts[f'trigger_{trigger}'] += 1
                    if step is None:
                        counts['tool_docs_never_called'] += 1
                    else:
                        later.setdefault(step, []).append((rec, trigger))
        else:
            start_records = wanted

        # start sites: passages hop by hop, then stages
        stages: list[list[Lookup]] = []
        for rec in hop_order(need, [r for r in start_records if r['kind'] == 'passage']):
            stages.append([Lookup('passage', [rec])])
        stages += staged([r for r in start_records if r['kind'] != 'passage'], rng, counts)
        by_step: dict[int, list[list[Lookup]]] = {}
        for step, placed in sorted(later.items()):
            triggers = {r['record_id']: t for r, t in placed}
            by_step[step] = staged([r for r, _ in placed], rng, counts, step=step)
            for lookups in by_step[step]:
                for lk in lookups:
                    lk.trigger = '+'.join(sorted({triggers[r['record_id']] for r in lk.records}))
        stages += by_step.pop(0, [])
        # redundant copies (``alternatives``, per hop: every record that alone states that
        # hop's fact): a slot names its hop's copies too, so the bank build stores them all
        hops = [[r for r in hop if r in supports and self._valid(supports[r], qt, Counter())
                 and self.kb_of(r) == kb] for hop in episode.get('alternatives') or []]
        for lookups in [*stages, *(ls for v in by_step.values() for ls in v)]:
            for lk in lookups:
                ids = {r['record_id'] for r in lk.records}
                alts = [r for hop in hops if ids & set(hop) for r in hop]
                if alts:
                    lk.flags['alternatives'] = list(dict.fromkeys(alts))
                    counts['alternative_records'] += len(lk.flags['alternatives'])
        if self.opt.gold_slots and split == 'train' and len(tokens(answer)) >= OWN_GOLD_TOKENS:
            gold = gold_record(kb, episode)
            self.gold_ids.add(gold['record_id'])
            counts['gold_slots'] += 1
            stages.insert(min(1, len(stages)), [Lookup(
                'gold_trajectory', [gold], flags={'gold': True, 'receding_weight': 1.0})])
            self.pending_gold = gold
        if spare and rng.random() < self.opt.distractor_rate:
            rec = spare[rng.randrange(len(spare))]
            stages.insert(rng.randrange(len(stages) + 1), [Lookup(
                rec['kind'], [rec], flags={'distractor': True})])
            counts['distractors'] += 1
        for lookups in stages:
            site(lookups)

        # the answer / trajectory
        if family == 'function_call':
            calls = (episode.get('verify') or {}).get('gold')
            if calls is None:
                calls = json.loads(answer)
            messages.append({'role': 'assistant', 'content': '',
                             'tool_calls': [call(c['name'], c.get('arguments') or {}) for c in calls]})
            source_index.append(0)
        elif turns:
            standing = [r for r in wanted if r['kind'] in STANDING]
            reread = False
            step = 0
            last_call = None
            for t, turn in enumerate(turns):
                text = turn['text']
                if turn['role'] == 'assistant':
                    for lookups in by_step.get(step, []):
                        site(lookups)
                    step += 1
                if turn['role'] == 'assistant' and text.startswith('Call: ') and \
                        family == 'policy_tool_agent':
                    parsed = _parse_call(text[6:])
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [parsed]})
                    last_call = parsed['function']['name']
                elif turn['role'] == 'assistant':
                    messages.append({'role': 'assistant', 'content': text})
                elif family == 'policy_tool_agent' and text.startswith('Result: '):
                    messages.append({'role': 'tool', 'name': last_call, 'content': text[8:]})
                elif family == 'policy_tool_agent' and text.startswith('Customer: '):
                    messages.append({'role': 'user', 'content': text[10:]})
                else:
                    messages.append({'role': 'user', 'content': text})
                source_index.append(t)
                if turn['role'] != 'assistant' and self.opt.failure_rereads and standing and \
                        not reread and FAILURE.match(text) and \
                        any(u['role'] == 'assistant' for u in turns[t + 1:]):
                    reread = True
                    counts['trigger_observation_failure'] += len(standing)
                    by_step.setdefault(step, []).insert(0, lookups_for_stage(
                        STANDING, standing, rng, counts, step=step,
                        trigger='observation_failure'))
        else:
            messages.append({'role': 'assistant', 'content': answer})
            source_index.append(0)

        write_sites = []
        if content:
            messages.append({'role': 'assistant', 'content': '',
                             'tool_calls': [call('memory_write', {})],
                             'write_span': {'kb': kb, 'write_site': len(write_sites)}})
            source_index.append(None)
            write_sites.append({'message': len(messages) - 1, 'call': 0, 'site': 'episode_end',
                                'source': 'own_trajectory' if turns else 'own_result',
                                'teacher_text': content})
            messages.append({'role': 'tool', 'name': 'memory_write',
                             'content': {'write_result': {'kb': kb, 'status': 'stored'}}})
            source_index.append(None)

        search_sites = []
        for (i, j), lk in sorted(plans.items()):
            search_sites.append({'message': i, 'call': j, 'result': i + 1 + j, 'kind': lk.kind,
                                 'records': len(lk.records),
                                 'record_ids': [r['record_id'] for r in lk.records],
                                 'step': lk.step, 'trigger': lk.trigger, **lk.flags})
            counts[f'searches_{"mid" if lk.step > 0 else "start"}'] += 1

        answers = [answer] + [a for a in prov.get('answer_aliases') or [] if isinstance(a, str)]
        row = {'episode_id': episode['episode_id'], 'kb': kb, 'split': split, 'format': FORMAT,
               'task_family': episode.get('task_family'), 'messages': messages, 'tools': tools,
               'answer': answer, 'verify': episode.get('verify'),
               'provenance': {**prov, 'source_query_time': qt},
               'search_sites': search_sites, 'write_sites': write_sites,
               'loss_mask': LOSS_POLICY}
        for key in ('capability', 'allowed_capability', 'choices', 'restore'):
            if key in episode:
                row.setdefault('episode_meta', {})[key] = episode[key]
        problems = self.audit(row, answers, qt, source_index,
                              n_source=len(turns) if turns else 1)
        if problems:
            counts.update({f'audit_{k}': v for k, v in problems.items()})
            return None, 'audit_failed:' + ','.join(sorted(problems)), counts
        counts['searches'] += len(search_sites)
        counts['search_sites'] += len({s['message'] for s in search_sites})
        counts['slot_records'] += sum(s['records'] for s in search_sites)
        counts['writes'] += len(write_sites)
        return row, None, counts

    def audit(self, row: dict, answers: list[str], query_time: int,
              source_index: list[int | None] | None = None, n_source: int | None = None) -> Counter:
        """Independent re-check of a finished transcript; returns violations."""
        bad: Counter = Counter()
        messages = row['messages']
        # writes: memory_write() without arguments and a write span; the teacher text
        # lives only in write_sites and appears in no message
        visible = json.dumps(messages, ensure_ascii=False)
        for site in row.get('write_sites') or []:
            m = messages[site['message']]
            calls = m.get('tool_calls') or []
            if len(calls) != 1 or calls[0]['function']['name'] != 'memory_write' or \
                    calls[0]['function']['arguments'] or 'write_span' not in m or \
                    m['role'] != 'assistant':
                bad['write_site_malformed'] += 1
            teacher = site.get('teacher_text') or ''
            if teacher and json.dumps(teacher, ensure_ascii=False)[1:-1] in visible:
                bad['teacher_text_in_messages'] += 1
        n_writes = sum(tc['function']['name'] == 'memory_write' for m in messages
                       for tc in m.get('tool_calls') or [])
        if n_writes != len(row.get('write_sites') or []):
            bad['write_without_site'] += 1
        for i, m in enumerate(messages):
            for j, tc in enumerate(m.get('tool_calls') or []):
                if tc['function']['name'] != 'memory_search':
                    continue
                if tc['function']['arguments']:
                    bad['search_has_arguments'] += 1
                if i + 1 + j >= len(messages):
                    bad['slot_not_after_call'] += 1
                    continue
                result = messages[i + 1 + j]
                body = result['content'].get('slot') if isinstance(result.get('content'), dict) \
                    else None
                if result['role'] != 'tool' or body is None:
                    bad['slot_not_after_call'] += 1
                    continue
                ids = body['record_ids']
                if not ids:
                    bad['empty_slot'] += 1
                if body.get('gold'):
                    if row['split'] != 'train':
                        bad['gold_outside_train'] += 1
                    if not all(r in self.gold_ids for r in ids) or 'receding_weight' not in body:
                        bad['gold_unflagged'] += 1
                else:
                    alternatives = body.get('alternatives') or []
                    if alternatives and not set(ids) <= set(alternatives):
                        bad['slot_not_in_alternatives'] += 1
                    for r in dict.fromkeys([*ids, *alternatives]):
                        if r not in self.index:
                            bad['record_not_in_kb'] += 1
                        elif self.index[r][0] >= query_time:
                            bad['record_after_query'] += 1
                        elif self.kb_of(r) != row['kb']:
                            bad['record_other_kb'] += 1
                        if r in self.gold_ids:
                            bad['gold_record_unflagged'] += 1
        # the source messages keep their order: every call's prefix is a source prefix
        if source_index is not None:
            seen = [s for s in source_index if s is not None]
            if seen != list(range(-1, (n_source or 0))):
                bad['source_order'] += 1
            for i, m in enumerate(messages):
                if (source_index[i] is None) != (i == 0 or is_memory(m)):
                    bad['generated_message_outside_memory'] += 1
        # the generated system prompt copies nothing from the answer or later turns
        request = tokens(messages[1]['content']) if len(messages) > 1 else []
        future = [message_text(m) for m in messages[2:]] + answers
        if leak_reason(messages[0]['content'], prefix=request, future=future, n=self.opt.ngram):
            bad['system_leak'] += 1
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


def load_index(corpus: Path) -> tuple[dict, dict[str, list[str]], set[str], Counter, dict]:
    """record_id -> (created_at, kind, domain); tool names per area; domains; command
    document frequencies over the worked examples; worked examples per task type."""
    index, tools, domains = {}, {}, set()
    command_df: Counter = Counter()
    pool: dict[str, list[dict]] = {}
    with (corpus / 'sources.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            rec = json.loads(line)
            index[rec['record_id']] = (int(rec['created_at']), rec['kind'], rec.get('domain', ''))
            domains.add(rec.get('domain', ''))
            prov = rec.get('provenance') or {}
            if rec['kind'] == 'tool_doc':
                if prov.get('area') and prov.get('tool'):
                    tools.setdefault(prov['area'], []).append(prov['tool'])
            elif rec['kind'] == 'worked_example':
                command_df.update(set(record_commands(rec['text'])))
                kind = task_type(prov.get('dataset', ''), prov.get('group'))
                if kind:
                    pool.setdefault(kind, []).append(rec)
    return index, {a: sorted(set(n)) for a, n in tools.items()}, domains, command_df, pool


def run_corpus(corpus: Path, output: Path, options: Options, *, limit: int | None = None,
               overwrite: bool = False, tokenizer: str | None = None,
               render_count: int = 50) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    kb = corpus_name(corpus)
    index, tool_names, domains, command_df, pool = load_index(corpus)
    builder = Builder(kb, index, options, per_domain=len(domains) > 1, tool_names=tool_names,
                      command_df=command_df, pool=pool)
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
        steps = Counter()
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
                steps.update(min(s['step'], 10) for s in row['search_sites'])
                if tok is not None and written < render_count:
                    rendered['checked'] += 1
                    rendered.update(render_check(tok, row))
                written += 1
        digests[f'transcripts-{split}.jsonl'] = digest.hexdigest()
        summary[split] = {'input': seen, 'written': written, 'rejected': dict(rejects),
                          'counts': dict(sorted(counts.items())),
                          'searches_per_episode': dict(sorted(per_episode.items())),
                          'search_step': {('10+' if k == 10 else str(k)): v
                                          for k, v in sorted(steps.items())},
                          'render_check': dict(rendered)}
    if gold_handle is not None:
        gold_handle.close()
        digests['gold-records-train.jsonl'] = hashlib.sha256(
            (tmp / 'gold-records-train.jsonl').read_bytes()).hexdigest()
    source_manifest = corpus / 'manifest.json'
    manifest = {'format': FORMAT, 'input': str(corpus), 'kb': kb,
                'kb_per_domain': builder.per_domain,
                'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                if source_manifest.exists() else None,
                'options': {'seed': options.seed, 'distractor_rate': options.distractor_rate,
                            'gold_slots': options.gold_slots, 'writes': options.writes,
                            'sequential_rate': options.sequential_rate, 'ngram': options.ngram,
                            'pool_examples': options.pool_examples,
                            'failure_rereads': options.failure_rereads,
                            'example_reads': options.example_reads, 'limit': limit},
                'memory_tools': [t['name'] for t in MEMORY_TOOLS],
                'memory_search_arguments': 'none (query = hidden state at the call)',
                'memory_write_arguments': 'none (the model generates a <|bg|> span in the '
                                          'same turn; teacher_text in write_sites only)',
                'write_render': f'[memory_write()] {SPAN_TOKENS["bg"][0]} latent span '
                                f'{SPAN_TOKENS["bg_end"][0]} (same assistant turn)',
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
    parser.add_argument('--tag', default='20260928v3', help='tag of output directories')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--distractor-rate', type=float, default=0.0,
                        help='share of episodes with one unhelpful search (default off)')
    parser.add_argument('--gold-slots', action='store_true',
                        help='B9: add the own gold trajectory as a flagged, receding-weight '
                             'record (train only)')
    parser.add_argument('--writes', choices=('reusable', 'trajectory', 'all', 'none'),
                        default='reusable',
                        help='reusable: trajectories plus single-shot families with a reusable '
                             'result (SQL, table answers, tool calls, code, multi-hop answers); '
                             'all: every episode')
    parser.add_argument('--sequential-rate', type=float, default=0.3,
                        help='share of multi-call sites split into consecutive single calls')
    parser.add_argument('--no-pool-examples', dest='pool_examples', action='store_false',
                        help='do not add same-task worked examples to agent episodes without any')
    parser.add_argument('--no-failure-rereads', dest='failure_rereads', action='store_false',
                        help='do not search the protocol again after a failed action')
    parser.add_argument('--example-reads', type=int, default=2,
                        help='trajectories: reads of a worked example, before the first use of '
                             'each of its most specific commands (distinct steps)')
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
                      sequential_rate=args.sequential_rate, ngram=args.ngram,
                      pool_examples=args.pool_examples, failure_rereads=args.failure_rereads,
                      example_reads=args.example_reads)
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
