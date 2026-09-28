"""Time row placement and field assignment of the superposed KB on synthetic unit keys
(CPU, exact scans): ``python scripts/bench_superpose_build.py --leaves 200000``."""
from __future__ import annotations

import argparse
import json
import resource
import time

import torch
from torch import nn

from schnitz.kb import superpose as sp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leaves', type=int, default=200000)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--field', type=float, default=8.0)
    parser.add_argument('--overlap', type=int, default=3)
    parser.add_argument('--clusters', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)
    centres = nn.functional.normalize(torch.randn(args.clusters, args.width, generator=gen), dim=-1)
    keys = nn.functional.normalize(
        centres[torch.randint(args.clusters, (args.leaves,), generator=gen)]
        + 0.3 * torch.randn(args.leaves, args.width, generator=gen), dim=-1)
    m = max(1, round(args.overlap / args.field * args.leaves))
    out = {'leaves': args.leaves, 'rows': m}
    t0 = time.time()
    rows = sp.place_rows(keys, m, seed=args.seed)
    out['place_s'] = round(time.time() - t0, 2)
    t0 = time.time()
    fields, candidates = sp.fields_of(keys, rows, args.overlap, 10.0)
    out['fields_s'] = round(time.time() - t0, 2)
    fill = [len(f) for f in fields]
    out['fill_mean'] = round(sum(fill) / len(fill), 2)
    out['empty_rows'] = sum(1 for f in fill if f == 0)
    out['max_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
