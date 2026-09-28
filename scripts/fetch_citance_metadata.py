"""Metadata for the citance-recall corpus (``schnitz.citances``), fetched once into the raw dir.

Input: the unarXive citation-recommendation release (``saier/unarXive_citrec``) under
``RAW/unarxive-citrec`` (``data/{train,dev,test}.jsonl``, ``license_info.jsonl``).
Steps, streaming, no credentials:

1. ``license_info.jsonl`` maps sample ids to citing arXiv papers (with their licence);
   one pass over the data counts distinct citing papers per cited OpenAlex work and
   writes ``metadata/cited-candidates.jsonl`` for works with at least ``--min-citers``
   citing papers outside the held-out set.
2. Candidates in ``citances.paper_order`` are resolved with the OpenAlex API (100 ids
   per request, free tier without a key) to an arXiv id; works without an arXiv version
   are skipped (``metadata/openalex-works.jsonl``). Stops at ``--target`` arXiv works.
3. The arXiv API (``export.arxiv.org``, CC0 metadata) gives titles and abstracts of the
   cited works (``metadata/cited-works.jsonl``) and titles of the citing papers that the
   build can use (``--citers`` stored and ``--heldout-citers`` held out per cited work,
   in ``citances.citer_order``; ``metadata/citing-papers.jsonl``).

Reruns reuse what is already fetched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from schnitz import citances

try:
    import orjson
    loads = orjson.loads
except ImportError:  # pragma: no cover
    loads = json.loads

ATOM = {'a': 'http://www.w3.org/2005/Atom'}
ARXIV_ID = re.compile(r'arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:v\d+)?(?:\.pdf)?$', re.I)
ARXIV_DOI = re.compile(r'10\.48550/arxiv\.(.+)$', re.I)


def get(url: str, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={'User-Agent': 'schnitz-citance-fetch/1'}), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as error:
            if error.code == 400:
                raise
            wait = 5 * 2 ** attempt
            print(f'retry {attempt + 1} in {wait}s: {error}', file=sys.stderr, flush=True)
            time.sleep(wait)
        except Exception as error:  # noqa: BLE001 - network retries
            wait = 5 * 2 ** attempt
            print(f'retry {attempt + 1} in {wait}s: {error}', file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f'failed: {url}')


def sample_citers(raw: Path) -> tuple[dict[str, str], dict[str, str]]:
    sample, licence = {}, {}
    with (raw / 'license_info.jsonl').open('rb') as handle:
        for line in handle:
            row = loads(line)
            licence[row['paper_arxiv_id']] = row['license']
            for s in row['sample_ids']:
                sample[s] = row['paper_arxiv_id']
    return sample, licence


def data_rows(raw: Path):
    for name in ('train', 'dev', 'test'):
        with (raw / 'data' / f'{name}.jsonl').open('rb') as handle:
            for line in handle:
                yield loads(line)


def candidates(raw: Path, out: Path, sample: dict, min_citers: int, rate: float) -> dict:
    path = out / 'cited-candidates.jsonl'
    if path.exists():
        return {r['cited']: r['citers'] for r in map(loads, path.open('rb'))}
    citers: dict[str, set] = defaultdict(set)
    for i, row in enumerate(data_rows(raw)):
        c = sample.get(row['_id'])
        if c is not None:
            citers[row['label']].add(c)
        if i % 500000 == 0:
            print(f'pass 1: {i} rows, {len(citers)} cited', flush=True)
    kept = {k: sorted(v) for k, v in citers.items()
            if sum(not citances.heldout_citer(c, rate) for c in v) >= min_citers}
    with path.open('w') as handle:
        for k in citances.paper_order(kept):
            handle.write(json.dumps({'cited': k, 'citers': kept[k]}) + '\n')
    print(f'{len(citers)} cited works, {len(kept)} with >= {min_citers} citing papers', flush=True)
    return kept


def arxiv_of(work: dict) -> str | None:
    doi = (work.get('ids') or {}).get('doi') or ''
    m = ARXIV_DOI.search(doi)
    if m:
        return m.group(1)
    for loc in work.get('locations') or []:
        for key in ('landing_page_url', 'pdf_url'):
            m = ARXIV_ID.search(loc.get(key) or '')
            if m:
                return m.group(1)
    return None


def resolve(out: Path, order: list[str], target: int) -> dict[str, dict]:
    path = out / 'openalex-works.jsonl'
    done = {r['cited']: r for r in map(loads, path.open('rb'))} if path.exists() else {}
    found = sum(r['arxiv_id'] is not None for r in done.values())
    todo = [c for c in order if c not in done]
    with path.open('a') as handle:
        for start in range(0, len(todo), 100):
            if found >= target:
                break
            batch = todo[start:start + 100]
            ids = '|'.join(c.rsplit('/', 1)[-1] for c in batch)
            url = ('https://api.openalex.org/works?per-page=100&select=id,title,ids,locations,'
                   f'publication_year&filter=openalex_id:{ids}')
            works = {w['id']: w for w in loads(get(url))['results']}
            for c in batch:
                w = works.get(c)
                row = {'cited': c, 'arxiv_id': arxiv_of(w) if w else None,
                       'openalex_title': (w or {}).get('title'),
                       'year': (w or {}).get('publication_year'), 'found': w is not None}
                found += row['arxiv_id'] is not None
                done[c] = row
                handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(f'openalex: {len(done)} looked up, {found} on arXiv', flush=True)
            time.sleep(0.2)
    return done


def _arxiv_query(ids: list[str]) -> dict[str, dict]:
    """Titles and abstracts of ``ids``; a batch the API rejects (HTTP 400, an id it
    does not accept) is split in halves down to the rejected id, which gets nothing."""
    url = ('https://export.arxiv.org/api/query?' + urllib.parse.urlencode(
        {'id_list': ','.join(ids), 'max_results': len(ids)}))
    try:
        try:
            root = ET.fromstring(get(url))
        except urllib.error.HTTPError:
            time.sleep(30)            # the API also answers 400 when briefly overloaded
            root = ET.fromstring(get(url))
    except urllib.error.HTTPError:
        print(f'arXiv rejected a batch of {len(ids)} ({ids[0]}...)', file=sys.stderr, flush=True)
        if len(ids) == 1:
            return {}
        time.sleep(3.1)
        half = len(ids) // 2
        out = _arxiv_query(ids[:half])
        time.sleep(3.1)
        return {**out, **_arxiv_query(ids[half:])}
    got = {}
    for entry in root.findall('a:entry', ATOM):
        ident = (entry.findtext('a:id', '', ATOM) or '').split('/abs/')[-1]
        ident = re.sub(r'v\d+$', '', ident)
        title = citances.clean(entry.findtext('a:title', '', ATOM))
        if ident and title and title != 'Error':
            got[ident] = {'arxiv_id': ident, 'title': title,
                          'abstract': citances.clean(entry.findtext('a:summary', '', ATOM))}
    return got


def arxiv_meta(ids: list[str], path: Path, batch: int = 200) -> dict[str, dict]:
    done = {r['arxiv_id']: r for r in map(loads, path.open('rb'))} if path.exists() else {}
    todo = sorted(set(ids) - done.keys())
    with path.open('a') as handle:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            got = _arxiv_query(chunk)
            for ident in chunk:
                row = got.get(ident) or {'arxiv_id': ident, 'title': None, 'abstract': None}
                done[ident] = row
                handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(f'arxiv {path.name}: {len(done)} / {len(done) + len(todo) - start - len(chunk)}',
                  flush=True)
            time.sleep(3.1)       # arXiv API terms: one request every three seconds
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=Path('/archive/raw/citances-20260928'))
    parser.add_argument('--min-citers', type=int, default=10)
    parser.add_argument('--heldout-rate', type=float, default=0.1)
    parser.add_argument('--target', type=int, default=4500, help='cited works on arXiv')
    parser.add_argument('--citers', type=int, default=48)
    parser.add_argument('--heldout-citers', type=int, default=4)
    args = parser.parse_args()
    out = args.raw / 'metadata'
    out.mkdir(parents=True, exist_ok=True)
    sample, licence = sample_citers(args.raw / 'unarxive-citrec')
    cands = candidates(args.raw / 'unarxive-citrec', out, sample, args.min_citers,
                       args.heldout_rate)
    del sample
    works = resolve(out, citances.paper_order(cands), args.target)
    on_arxiv = {c: w['arxiv_id'] for c, w in works.items() if w['arxiv_id']}
    cited = arxiv_meta(list(on_arxiv.values()), out / 'arxiv-cited.jsonl')
    with (out / 'cited-works.jsonl').open('w') as handle:
        for c, a in on_arxiv.items():
            meta = cited.get(a) or {}
            handle.write(json.dumps({'cited': c, 'arxiv_id': a, 'title': meta.get('title'),
                                     'abstract': meta.get('abstract'),
                                     'openalex_title': works[c]['openalex_title']}) + '\n')
    need = set()
    for c in on_arxiv:
        order = citances.citer_order(c, cands[c])
        held = [x for x in order if citances.heldout_citer(x, args.heldout_rate)]
        kept = [x for x in order if not citances.heldout_citer(x, args.heldout_rate)]
        need.update(kept[:args.citers], held[:args.heldout_citers])
    citing = arxiv_meta(sorted(need), out / 'arxiv-citing.jsonl')
    with (out / 'citing-papers.jsonl').open('w') as handle:
        for a in sorted(need):
            handle.write(json.dumps({'arxiv_id': a, 'title': citing[a]['title'],
                                     'license': licence.get(a)}) + '\n')
    print(json.dumps({'candidates': len(cands), 'looked_up': len(works),
                      'on_arxiv': len(on_arxiv),
                      'with_abstract': sum(bool((cited.get(a) or {}).get('abstract'))
                                           for a in on_arxiv.values()),
                      'citing_needed': len(need),
                      'citing_with_title': sum(bool(citing[a]['title']) for a in need)}))


if __name__ == '__main__':
    main()
