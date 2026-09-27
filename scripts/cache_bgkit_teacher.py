"""B1 of the BGKit restart: cache S2 teacher encodings of every bank source.

Each source is encoded by the frozen S2 encoder with the question-free
reconstruct prompt, through BGKit's own batch path (``Collator`` then
``core.encode``, as in its evaluation), at each requested ratio. Survivors are
kept in document order with the summary slots ``finish_reps`` appends.

Output: ``<output>/shard-NNNNN.safetensors`` holding, per ratio tag, the
concatenated reps (bf16, [total, 1024]) and per-source counts, plus
``shard-NNNNN.json`` with the source IDs, token counts and ratios. Shards are
written atomically and skipped when present, so the job resumes. A training-time
teacher only: SDKB inference never runs this encoder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import save_file


def _tag(ratio: float) -> str:
    return f'x{round(1 / ratio)}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratios', nargs='+', type=float,
                        default=[1 / 16, 1 / 32, 1 / 64, 1 / 128])
    parser.add_argument('--shard-size', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--cuda-fraction', type=float, default=0.10)
    args = parser.parse_args()

    from bgkit_core.host_memory_guard import cap_cuda
    cap_cuda(args.cuda_fraction)
    from bgkit2.data.autoencode import Collator, Sample
    from bgkit2.data.templates import Templates
    from bgkit2.training.standalone import load_models

    rows = []
    with args.sources.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((row['record_id'], row['text']))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    identity = {'checkpoint': str(args.checkpoint), 'experiment': args.experiment,
                'sources': str(args.sources), 'source_count': len(rows),
                'ratios': args.ratios, 'shard_size': args.shard_size,
                'prompt': 'reconstruct', 'max_tokens': args.max_tokens}
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('Existing cache was built with different settings')
    manifest.write_text(json.dumps({'identity': identity, 'complete': False}, indent=2) + '\n')

    core = load_models(args.experiment, str(args.checkpoint))
    templates = Templates.from_tokenizer(core.tok, style=core.cfg2.data.prompt_style)
    collate = Collator(templates, enc_pad_id=None if core.ctx_tok is None
                       else core.chat_renderer().enc_pad_id)
    prompt = Templates.compression_prompt_ids(core.ctx_tok or core.tok)['reconstruct']
    tok = core.ctx_tok or core.tok
    shards = (len(rows) + args.shard_size - 1) // args.shard_size
    truncated, started = 0, time.time()
    for shard in range(shards):
        path = args.output / f'shard-{shard:05d}.safetensors'
        if path.exists():
            continue
        chunk = rows[shard * args.shard_size:(shard + 1) * args.shard_size]
        encoded = []
        for record_id, text in chunk:
            ids = tok(text, add_special_tokens=False)['input_ids']
            truncated += len(ids) > args.max_tokens
            encoded.append((record_id, torch.tensor(ids[:args.max_tokens], dtype=torch.long)))
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][1]))
        tensors, counts = {}, {}
        for ratio in args.ratios:
            reps, lengths = [None] * len(encoded), [0] * len(encoded)
            for start in range(0, len(order), args.batch_size):
                picked = order[start:start + args.batch_size]
                samples = [Sample(ctx_ids=encoded[i][1], target_ids=encoded[i][1], task='reconstruct',
                                  store=0, doc=i, prompt_ids=prompt) for i in picked]
                batch = collate(samples).to(core.device)
                with torch.no_grad(), core.autocast():
                    out = core.encode(batch, ratio)
                for row, i in enumerate(picked):
                    kept = out.reps[row][out.rep_mask[row]].to(torch.bfloat16).cpu()
                    reps[i], lengths[i] = kept, kept.shape[0]
            tensors[f'{_tag(ratio)}_reps'] = torch.cat(reps).contiguous()
            tensors[f'{_tag(ratio)}_counts'] = torch.tensor(lengths, dtype=torch.int32)
            counts[_tag(ratio)] = sum(lengths)
        pending = path.with_suffix('.pending')
        save_file(tensors, str(pending))
        (args.output / f'shard-{shard:05d}.json').write_text(json.dumps({
            'record_ids': [record_id for record_id, _ in encoded],
            'tokens': [int(ids.shape[0]) for _, ids in encoded], 'reps': counts}) + '\n')
        os.replace(pending, path)
        print(json.dumps({'shard': shard, 'of': shards, 'sources': len(chunk), 'reps': counts,
                          'truncated_so_far': truncated,
                          'elapsed_s': round(time.time() - started)}), flush=True)
    manifest.write_text(json.dumps({'identity': identity, 'complete': True, 'shards': shards,
                                    'truncated_sources': truncated}, indent=2) + '\n')


if __name__ == '__main__':
    main()
