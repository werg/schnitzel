"""Bank creation's write step as a stage (docs/knowledge-base-stack.md, 5.1 step 6):
the frozen writer of a writer-stage state writes every record that the memory
transcripts' slots name, one span per record (the record in context under the
memory prompt, at a ratio level of the B1 length schedule), into one span cache per
dataset KB (``schnitz.kb.bank``). The caches fill the memory slots of B4 prefixes
(``train.py writer --slot-spans``) and feed L1's KB build. Resumable; a cache is
tied to one writer state and level, so a newer writer writes a new cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from schnitz.kb.bank import build_caches, record_sources
from schnitz.kb.decoder import frozen_reader


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--writer-state', type=Path, required=True,
                        help='writer-stage state (writer.pt) whose frozen writer writes')
    parser.add_argument('--transcripts', type=Path, nargs='+', required=True)
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--limit', type=int, help='first N transcripts per split and directory')
    parser.add_argument('--with-writes', action='store_true',
                        help="only the records of transcripts that have write sites (B4c's)")
    parser.add_argument('--level', default='s1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cuda-fraction', type=float, default=0.2)


@torch.no_grad()
def run(args) -> None:
    records = record_sources(args.transcripts, {s: args.limit for s in args.splits.split(',')},
                             with_writes=args.with_writes)
    model = frozen_reader(args.checkpoint, args.experiment, args.writer_state, args.cuda_fraction)
    state = torch.load(args.writer_state, map_location='cpu', weights_only=False)
    meta = {'writer_state': str(args.writer_state), 'step': int(state.get('step', -1))}
    del state
    counts = build_caches(args.output, model, records, args.level, args.batch_size, meta)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'caches.json').write_text(json.dumps(
        {'transcripts': [str(d) for d in args.transcripts], 'level': args.level, **meta,
         'records': counts}, indent=2) + '\n')
    print(json.dumps({'records': counts}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    run(parser.parse_args())
