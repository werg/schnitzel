"""One entry point for every training stage (docs/knowledge-base-stack.md, section 5).

    python scripts/train.py <stage> [stage arguments]

Stages live in ``schnitz.kb.stages``; each exposes ``add_args(parser)`` and
``run(args)`` and builds on the shared modules (``schnitz.kb.decoder``,
``schnitz.kb.stack``, ``schnitz.kb.losses``, ``schnitz.kb.loop``).
"""
from __future__ import annotations

import argparse
import importlib
import sys

STAGES = {
    'writer': 'schnitz.kb.stages.writer',   # B2, B3, B4 (+ soft I/O port)
    'bank': 'schnitz.kb.stages.bank',       # bank creation's write step (span caches)
    'k1': 'schnitz.kb.stages.k1',           # autoencoding through the spaces
    'l1': 'schnitz.kb.stages.l1',           # live items end to end (K2: --retrieval-only)
    'l2': 'schnitz.kb.stages.l2',           # producers reproduce the L1a items
    'b9': 'schnitz.kb.stages.b9',           # learning by experience over rounds
}
# K3 (the write fit to read-phase rows) is the L2 stack mode
ALIASES = {'k3': ('l2', ['train', '--producer', 'stack'])}


def main() -> None:
    names = list(STAGES) + list(ALIASES)
    if len(sys.argv) < 2 or sys.argv[1] not in names:
        raise SystemExit(f'usage: train.py {{{",".join(names)}}} [arguments]')
    name, argv = sys.argv[1], sys.argv[2:]
    if name in ALIASES:
        name, prefix = ALIASES[name]
        argv = prefix + argv
    stage = importlib.import_module(STAGES[name])
    parser = argparse.ArgumentParser(prog=f'train.py {sys.argv[1]}', description=stage.__doc__)
    stage.add_args(parser)
    stage.run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
