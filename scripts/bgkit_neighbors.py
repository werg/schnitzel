"""Semantically related records for combiner training (restart plan B7).

Embeds every bank source as the centred, normalized mean of its S2 teacher reps
in one space of the B1 cache and finds each source's top-k neighbours by an exact
cosine scan on the GPU (no ANN). Output ``neighbors.pt``: ``record_ids`` (list,
cache order), ``neighbors`` (int32 [N, k], indices into record_ids, self and
exact duplicates excluded) and ``scores`` (float16 [N, k]). Training-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from schnitz.kb.decoder import TeacherCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--space', default='s1')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=4096)
    args = parser.parse_args()

    cache = TeacherCache(args.cache, args.sources)
    ids, rows = [], []
    for shard in sorted(cache.handles):
        reps = cache.handles[shard].get_tensor(f'{args.space}_reps').float()
        counts = cache.counts[shard, args.space]
        means = torch.stack([chunk.mean(0) for chunk in torch.split(reps, counts.tolist())])
        rows.append(means)
    ids = [item[2] for item in cache.items]
    embed = torch.cat(rows)
    # centred: mean-pooled reps share a large common direction
    embed = torch.nn.functional.normalize(embed - embed.mean(0), dim=-1).half().cuda()
    assert embed.shape[0] == len(ids)
    neighbors = torch.empty(len(ids), args.k, dtype=torch.int32)
    scores = torch.empty(len(ids), args.k, dtype=torch.float16)
    for start in range(0, len(ids), args.chunk):
        sim = embed[start:start + args.chunk] @ embed.T
        sim[sim > 0.9995] = -2  # self and exact duplicates
        top = sim.topk(args.k, dim=-1)
        neighbors[start:start + args.chunk] = top.indices.int().cpu()
        scores[start:start + args.chunk] = top.values.cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'record_ids': ids, 'neighbors': neighbors, 'scores': scores,
                'space': args.space}, args.output)
    print({'records': len(ids), 'mean_top1': float(scores[:, 0].float().mean()),
           'mean_topk': float(scores[:, -1].float().mean())})


if __name__ == '__main__':
    main()
