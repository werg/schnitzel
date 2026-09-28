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
    'k1': 'schnitz.kb.stages.k1',           # autoencoding through the spaces
    'k3': 'schnitz.kb.stages.k3',           # superposition-operator warm-up (K3a/K3b)
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f'usage: train.py {{{",".join(STAGES)}}} [arguments]')
    stage = importlib.import_module(STAGES[sys.argv[1]])
    parser = argparse.ArgumentParser(prog=f'train.py {sys.argv[1]}', description=stage.__doc__)
    stage.add_args(parser)
    stage.run(parser.parse_args(sys.argv[2:]))


if __name__ == '__main__':
    main()
