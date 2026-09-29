"""Text-embedding teacher for the retrieval heads (docs/knowledge-base-stack.md, 5.1 step 4
(K2) and 5.2): a precomputed cache of teacher embeddings, used only as training
supervision.

``build`` embeds, with a frozen text-embedding model (default Qwen3-Embedding-0.6B):

- every record of the KBs (the records an L1 bank holds, read from the banks'
  provenance, or ``schnitz.kb.bank.record_sources`` of the transcripts), as text;
- every search site of the transcripts: the episode's causal prefix at the
  ``memory_search()`` call (``site_prefix``: system, user and earlier turns, earlier
  memory results as a placeholder - the model reads latent spans there, never the
  records' text - and write calls left out, as L1 renders them without ``--writes``;
  nothing at or after the call: invariant 2), under a retrieval instruction.

The cache (``TeacherKeys``) keys queries by (episode id, call index), the call index
being the slot's position in the episode (``Episode.calls``/``slots`` order), and
stores each site's top-k records of its own KB (records no later than the episode's
query time) for teacher-mined positives. Embeddings are unit-normalized float16.

Inference never uses the teacher: reads query from the decoder state and search stored
latent keys (invariant 1); the text enters training only through this cache.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from schnitz.kb.bank import Transcripts, read_sources, record_sources, slots_of

DEFAULT_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
INSTRUCTION = ('Given the conversation so far, retrieve the knowledge-base record the '
               'assistant needs at its next memory search')
MEMORY_RESULT = '[memory result]'
FORMAT = 'schnitz.kb.teacher-keys/1'


# -- query text ----------------------------------------------------------------------
def _is_write(m: dict) -> bool:
    from schnitz.kb.stages.l1 import _is_write as is_write
    return is_write(m)


def site_prefixes(row: dict) -> list[str]:
    """The causal prefix text at each ``memory_search()`` call of a transcript, in call
    order (the slots' order): every message before the call's message, then the call's
    own message up to and including the call. Earlier memory results are the
    placeholder ``[memory result]`` (the model sees latent spans there, not text); write
    calls and their acknowledgements are left out; tool arguments of memory calls are
    empty (the current schema)."""
    out: list[str] = []
    lines: list[str] = []
    for m in row['messages']:
        if _is_write(m):
            continue
        role, content = m['role'], m.get('content')
        if isinstance(content, dict) and 'slot' in content:
            lines.append(f'{role}: {MEMORY_RESULT}')
            continue
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        if content:
            lines.append(f'{role}: {content}')
        for tc in m.get('tool_calls') or []:
            name = tc['function']['name']
            if name == 'memory_search':
                lines.append(f'{role}: memory_search()')
                out.append('\n'.join(lines))
            elif name != 'memory_write':
                arguments = tc['function'].get('arguments') or {}
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                lines.append(f'{role}: {name}({arguments})')
    if len(out) != len(slots_of(row)):
        raise ValueError(f'{row["episode_id"]}: {len(out)} memory_search calls, '
                         f'{len(slots_of(row))} slots')
    return out


# -- the embedding model -------------------------------------------------------------
class TextEmbedder:
    """A Qwen3-Embedding style model: last-token pooling with left padding, queries as
    ``Instruct: ...\\nQuery:...``, unit-normalized outputs. Query prefixes longer than
    ``query_tokens`` keep their end (the latest context); records are cut at
    ``record_tokens``."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = 'cuda',
                 dtype=torch.float16, batch_tokens: int = 16384,
                 query_tokens: int = 1024, record_tokens: int = 512,
                 instruction: str = INSTRUCTION):
        from transformers import AutoModel, AutoTokenizer
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model, padding_side='left')
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(device).eval()
        self.device = device
        self.batch_tokens = batch_tokens
        self.query_tokens, self.record_tokens = query_tokens, record_tokens
        self.instruction = instruction

    def _tail(self, text: str) -> str:
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        if len(ids) <= self.query_tokens:
            return text
        return self.tok.decode(ids[-self.query_tokens:])

    @torch.no_grad()
    def encode(self, texts: Sequence[str], query: bool = False) -> torch.Tensor:
        if query:
            texts = [f'Instruct: {self.instruction}\nQuery:{self._tail(t)}' for t in texts]
        limit = self.query_tokens + 64 if query else self.record_tokens
        lengths = [len(self.tok(t, add_special_tokens=False)['input_ids']) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lengths[i])
        out = torch.zeros(len(texts), self.model.config.hidden_size, dtype=torch.float16)
        start = 0
        while start < len(order):
            width = min(lengths[order[start]] + 2, limit)
            count = max(1, self.batch_tokens // max(width, 1))
            picked = order[start:start + count]
            enc = self.tok([texts[i] for i in picked], padding=True, truncation=True,
                           max_length=limit, return_tensors='pt').to(self.device)
            hidden = self.model(**enc).last_hidden_state
            pooled = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
            out[torch.tensor(picked)] = pooled.half().cpu()
            start += count
        return out


# -- the cache ---------------------------------------------------------------------------
def bank_records(banks: Path) -> tuple[dict[str, dict], dict]:
    """The records an L1 bank holds (the sources of its codec items), with text, KB and
    time from the transcripts' corpora; and the banks' manifest."""
    from schnitz.kb.read import producer_index
    from schnitz.kb_store import KnowledgeBase
    manifest = json.loads((banks / 'banks.json').read_text())
    wanted: set[str] = set()
    for info in manifest['kbs'].values():
        kb = KnowledgeBase(banks / info['dir'])
        for producer, sources in producer_index(kb, 'D').values():
            if producer == 'codec':
                wanted.update(sources)
        kb.close()
    records = read_sources(manifest['transcripts'], wanted)
    missing = wanted - set(records)
    if missing:
        raise ValueError(f'{len(missing)} banked records are not in the corpus sources')
    return records, manifest


def build(output: Path, embedder, transcript_dirs: Sequence[Path],
          splits: Mapping[str, int | None], records: Mapping[str, dict] | None = None,
          top: int = 256, meta: dict | None = None, log: Callable[[dict], None] | None = None
          ) -> dict:
    """Write the teacher cache for the search sites of the first ``splits[split]``
    transcripts per directory against ``records`` (record id -> {'text', 'kb',
    'created_at'}; default ``record_sources`` of the same transcripts). ``embedder``
    has ``encode(texts, query=False)`` -> (n, d) unit rows. Returns the manifest."""
    started = time.time()
    records = dict(records if records is not None
                   else record_sources(transcript_dirs, splits))
    record_ids = sorted(records, key=lambda r: (records[r]['kb'], r))
    sites, texts = [], []
    for split, limit in splits.items():
        for row in Transcripts(transcript_dirs, split, limit):
            prefixes = site_prefixes(row)
            for j, text in enumerate(prefixes):
                sites.append({'episode_id': row['episode_id'], 'call': j, 'kb': row['kb'],
                              'split': split, 'query_time': _query_time(row)})
                texts.append(text)
    keys = [(s['episode_id'], s['call']) for s in sites]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate (episode id, call) search sites')
    rec = embedder.encode([records[r]['text'] for r in record_ids])
    if log:
        log({'records': len(record_ids), 'elapsed_s': round(time.time() - started)})
    qry = embedder.encode(texts, query=True)
    if log:
        log({'sites': len(sites), 'elapsed_s': round(time.time() - started)})
    kb_of = [records[r]['kb'] for r in record_ids]
    time_of = torch.tensor([int(records[r]['created_at']) for r in record_ids])
    tops = torch.full((len(sites), top), -1, dtype=torch.int32)
    by_kb: dict[str, list[int]] = {}
    for i, kb in enumerate(kb_of):
        by_kb.setdefault(kb, []).append(i)
    for kb, rows in by_kb.items():
        idx = torch.tensor(rows)
        emb = rec[idx].float()
        for n, site in enumerate(sites):
            if site['kb'] != kb:
                continue
            cos = emb @ qry[n].float()
            cos[time_of[idx] > site['query_time']] = -float('inf')
            k = min(top, int(torch.isfinite(cos).sum()))
            if k:
                tops[n, :k] = idx[cos.topk(k).indices].to(torch.int32)
    output.mkdir(parents=True, exist_ok=True)
    save_file({'records': rec.contiguous(), 'queries': qry.contiguous(),
               'top': tops.contiguous()}, str(output / 'teacher.safetensors'))
    (output / 'records.json').write_text(json.dumps(
        [[r, records[r]['kb'], int(records[r]['created_at'])] for r in record_ids]) + '\n')
    (output / 'sites.json').write_text(json.dumps(sites) + '\n')
    manifest = {'format': FORMAT, 'model': getattr(embedder, 'name', None),
                'instruction': getattr(embedder, 'instruction', None),
                'query_tokens': getattr(embedder, 'query_tokens', None),
                'record_tokens': getattr(embedder, 'record_tokens', None),
                'transcripts': [str(d) for d in transcript_dirs], 'splits': dict(splits),
                'records': len(record_ids), 'sites': len(sites), 'top': top,
                'dim': int(rec.shape[1]), 'kbs': {kb: len(r) for kb, r in by_kb.items()},
                'elapsed_s': round(time.time() - started), **(meta or {})}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def _query_time(row: dict) -> int:
    prov = row.get('provenance') or {}
    return int(prov.get('source_query_time', prov.get('query_time', 2)))


class TeacherKeys:
    """A teacher cache written by ``build``: unit record and site embeddings (float16)
    and each site's top records. ``query`` refuses unknown sites."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('format') != FORMAT:
            raise ValueError(f'{root} is not a teacher-keys cache ({FORMAT})')
        tensors = load_file(str(self.root / 'teacher.safetensors'))
        self.records, self.queries, self.top = (tensors['records'], tensors['queries'],
                                                tensors['top'])
        rows = json.loads((self.root / 'records.json').read_text())
        self.record_ids = [r for r, _, _ in rows]
        self.record_kb = [kb for _, kb, _ in rows]
        self.record_row = {r: i for i, r in enumerate(self.record_ids)}
        self.sites = json.loads((self.root / 'sites.json').read_text())
        self.site_row = {(s['episode_id'], int(s['call'])): i for i, s in enumerate(self.sites)}

    def site(self, episode_id: str, call: int) -> int:
        row = self.site_row.get((episode_id, int(call)))
        if row is None:
            raise KeyError(f'no teacher query for search site ({episode_id!r}, {call}) '
                           f'in {self.root}')
        return row

    def query(self, episode_id: str, call: int) -> torch.Tensor:
        return self.queries[self.site(episode_id, call)]

    def record(self, record_id: str) -> torch.Tensor | None:
        row = self.record_row.get(record_id)
        return None if row is None else self.records[row]

    def mined(self, episode_id: str, call: int, k: int, kb: str | None = None,
              skip: Iterable[str] = ()) -> list[str]:
        """The site's teacher top-``k`` record ids of its KB (``kb`` checks it), leaving
        out ``skip`` (e.g. neutral records) before counting."""
        n = self.site(episode_id, call)
        if kb is not None and self.sites[n]['kb'] != kb:
            raise PermissionError(f'teacher site of {self.sites[n]["kb"]!r}, not {kb!r}')
        skip = set(skip)
        out = []
        for i in self.top[n].tolist():
            if i < 0 or len(out) >= k:
                break
            if self.record_ids[i] not in skip:
                out.append(self.record_ids[i])
        return out


def teacher_recall(cache: TeacherKeys, transcript_dirs: Sequence[Path], split: str = 'validation',
                   limit: int | None = None, ks: Sequence[int] = (1, 5, 20, 64)) -> dict:
    """The teacher's own retrieval at the search sites of ``split``: per site, the
    cache's records of the site's KB no later than its query time, ranked by teacher
    cosine with the slot's neutral records left out; a hit at k when any of the slot's
    records or alternatives is in the top k. Returns recall@k over the sites (and per
    task family) and the mean candidate count."""
    times = torch.tensor([t for _, _, t in json.loads(
        (cache.root / 'records.json').read_text())])
    kb_rows: dict[str, list[int]] = {}
    for i, kb in enumerate(cache.record_kb):
        kb_rows.setdefault(kb, []).append(i)
    hits: dict[str, list[list[float]]] = {}
    sizes = []
    for row in Transcripts(transcript_dirs, split, limit):
        for j, slot in enumerate(slots_of(row)):
            positive = {*slot['record_ids'], *(slot.get('alternatives') or ())}
            neutral = set(slot.get('neutral') or ()) - positive
            rows = [i for i in kb_rows.get(row['kb'], ())
                    if cache.record_ids[i] not in neutral and times[i] <= _query_time(row)]
            idx = torch.tensor(rows, dtype=torch.long)
            cos = cache.records[idx].float() @ cache.query(row['episode_id'], j).float()
            order = idx[cos.argsort(descending=True)].tolist()
            first = next((n for n, i in enumerate(order) if cache.record_ids[i] in positive),
                         len(order))
            at = [float(first < k) for k in ks]
            sizes.append(len(order))
            hits.setdefault('all', []).append(at)
            hits.setdefault(row.get('task_family', '?'), []).append(at)
    out: dict = {name: {'sites': len(v), **{f'recall@{k}': round(sum(x[n] for x in v) / len(v), 4)
                                            for n, k in enumerate(ks)}}
                 for name, v in hits.items()}
    out['candidates_mean'] = round(sum(sizes) / max(len(sizes), 1), 1)
    return out
