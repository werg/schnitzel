"""Inverse-cloze retrieval transcripts (K2 data) from task corpora (``schnitz.inverse_cloze``).

For every ``tasks-*`` corpus directory given, writes
``<output-root>/memory-inverse-cloze-<corpus>-<tag>/`` with ``transcripts-{train,validation}.jsonl``
and ``manifest.json`` in the memory-transcript format 3 of
``scripts/prepare_memory_transcripts.py``: one ``memory_search()`` per episode, its cue in
the user turn, the slot naming the source record with its redundant copies as
``alternatives`` and near-duplicates as ``neutral``, no answer turn
(``task_family: inverse_cloze``).

Why transcript level, not a new task corpus: the transcripts name the source corpus's
KB (the manifest's ``input`` is that corpus and ``kb`` its corpus name, per domain for
multi-domain corpora), so ``l1 build`` banks these records into the same KB, reusing
the corpus's span cache (a bank that already holds them writes nothing new), and K2 can
mix inverse-cloze with the corpus's own transcripts over one bank. A new ``tasks-``
corpus would be a new KB, and its bank would re-run the writer over the same records.

Every row passes the transcript builder's audit (``Builder.audit``: slot records exist
in the KB before the query time, alternatives contain the slot's records, neutral
records are never positives, one KB per transcript, empty call arguments, no system
prompt leak); ``--render-check`` also renders the first rows through the tokenizer's chat
template (``render_check``). Output is deterministic in ``--seed``.

Example (in the container)::

    python scripts/prepare_inverse_cloze.py /archive/corpora/tasks-recall-text-r8-20260928 \\
        /archive/corpora/tasks-citance-recall-20260928 --output-root /archive/corpora \\
        --train 4000 --validation 300 --render-check /runs/hf-models/LFM2.5-350M
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schnitz import inverse_cloze as ic  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    'prepare_memory_transcripts', Path(__file__).resolve().parent / 'prepare_memory_transcripts.py')
mt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mt)


def load_corpus(corpus: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    records = {}
    with (corpus / 'sources.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            rec = json.loads(line)
            records[rec['record_id']] = rec
    episodes = {}
    for split, path in mt.episode_files(corpus).items():
        with path.open(encoding='utf-8') as handle:
            episodes[split] = [json.loads(line) for line in handle]
    return records, episodes


def run_corpus(corpus: Path, output: Path, sizes: dict[str, int], *, seed: int = 0,
               overwrite: bool = False, tokenizer: str | None = None, render_count: int = 200,
               citance_kinds: dict[str, float] | None = None) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    kb = mt.corpus_name(corpus)
    index, _, domains, _, _ = mt.load_index(corpus)
    builder = mt.Builder(kb, index, mt.Options(seed=seed), per_domain=len(domains) > 1)
    records, episodes = load_corpus(corpus)
    rows, summary = ic.build(records, episodes, builder.kb_of, sizes, seed=seed,
                             corpus=kb, source_corpus=str(corpus),
                             citance_kinds=citance_kinds or ic.CITANCE_KINDS)
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    tmp = output.with_name(output.name + '.pending')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    digests, splits = {}, {}
    for split, split_rows in rows.items():
        rejected, rendered = Counter(), Counter()
        digest = hashlib.sha256()
        written = 0
        with (tmp / f'transcripts-{split}.jsonl').open('w', encoding='utf-8') as out:
            for row in split_rows:
                problems = builder.audit(row, [], int(row["provenance"]["source_query_time"]))
                if problems:
                    rejected.update({f'audit_{k}': v for k, v in problems.items()})
                    continue
                if tok is not None and written < render_count:
                    rendered['checked'] += 1
                    rendered.update(mt.render_check(tok, row))
                text = json.dumps(row, ensure_ascii=False) + '\n'
                out.write(text)
                digest.update(text.encode())
                written += 1
        digests[f'transcripts-{split}.jsonl'] = digest.hexdigest()
        splits[split] = {**summary.get(split, {}), 'written': written,
                         'rejected': dict(rejected), 'render_check': dict(rendered)}
    source_manifest = corpus / 'manifest.json'
    manifest = {'format': ic.FORMAT, 'input': str(corpus), 'kb': kb,
                'kb_per_domain': builder.per_domain, 'generator': 'schnitz.inverse_cloze',
                'task_family': ic.FAMILY, 'corpus_type': summary['corpus_type'],
                'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                if source_manifest.exists() else None,
                'options': {'seed': seed, 'sizes': sizes, 'ngram': ic.NGRAM,
                            'ambiguous_share': ic.AMBIGUOUS,
                            'sentence_words': list(ic.SENTENCE_WORDS),
                            'span_words': list(ic.SPAN_WORDS),
                            'paraphrase_words': ic.PARAPHRASE_WORDS,
                            'citance_kinds': citance_kinds or ic.CITANCE_KINDS},
                'groups': summary['groups'], 'filters': summary['filters'],
                'memory_tools': [t['name'] for t in mt.MEMORY_TOOLS],
                'memory_search_arguments': 'none (query = hidden state at the call)',
                'answer': 'none (retrieval only: the call is the only assistant turn)',
                'slot_render': mt.SPAN_TOKENS['mem'][0] + ' latent span '
                + mt.SPAN_TOKENS['mem_end'][0],
                'loss_mask': ic.LOSS_POLICY, 'splits': splits, 'sha256': digests}
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if output.exists():
        shutil.rmtree(output)
    os.replace(tmp, output)
    return manifest


def _weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    out = {k: float(v) for k, v in (p.split('=') for p in text.split(',') if p)}
    unknown = set(out) - set(ic.CITANCE_KINDS)
    if unknown:
        raise ValueError(f'unknown citance cue kinds {sorted(unknown)}')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corpora', nargs='+', type=Path, help='tasks-* corpus directories')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tag', default='20260929v3')
    parser.add_argument('--train', type=int, default=4000)
    parser.add_argument('--validation', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--citance-kinds',
                        help='cue kind weights for citance corpora, e.g. '
                             'abstract=0.4,heldout=0.3,sibling=0.3 (the default)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--render-check', metavar='TOKENIZER_DIR')
    parser.add_argument('--render-count', type=int, default=200)
    args = parser.parse_args()
    for corpus in args.corpora:
        output = args.output_root / f'memory-{ic.FAMILY.replace("_", "-")}-' \
                                    f'{mt.corpus_name(corpus)}-{args.tag}'
        manifest = run_corpus(corpus, output, {'train': args.train,
                                               'validation': args.validation},
                              seed=args.seed, overwrite=args.overwrite,
                              tokenizer=args.render_check, render_count=args.render_count,
                              citance_kinds=_weights(args.citance_kinds))
        print(json.dumps({'corpus': corpus.name, 'output': str(output), 'kb': manifest['kb'],
                          **{s: {k: v[k] for k in ('written', 'rejected', 'cue_kinds')}
                             for s, v in manifest['splits'].items()}}), flush=True)


if __name__ == '__main__':
    main()
