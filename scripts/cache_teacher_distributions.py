"""Cache a teacher model's next-token distributions on corpus text
(docs/knowledge-base-stack.md, section 1.1; the later distillation phase).

For every record of a ``sources.jsonl`` corpus the text is tokenized exactly as
the decoder reads it (``add_special_tokens=False``, at most ``--max-tokens``
tokens), the teacher reads BOS + text, and for each text token t_i the teacher's
distribution over t_i given BOS and t_<i is stored as its ``--top-k`` most likely
token ids (stored as int16 bits of the uint16 id: read back with
``x.to(torch.int32) & 0xFFFF``; the vocabulary has 65,536 entries), their
log-probabilities (fp16) and the log of the remaining probability mass.

The teacher must share the decoder's tokenizer: the regular vocabularies' token-to-id
maps must be identical (the run stops otherwise). Added special tokens may differ
(e.g. LFM2.5-1.2B-Base places ``<think>``/``</think>`` at other ids than our
decoder); teacher ids whose token differs from ours are never stored among the
top-k, their probability stays in the remaining mass.

Output: resumable shards ``shard-NNNNN.safetensors`` (``ids`` (T, k) int16,
``logprobs`` (T, k) fp16, ``rest`` (T,) fp16, ``tokens`` (T,) int16 bits: the
text tokens themselves, ``lengths`` (R,) int32) with ``shard-NNNNN.json`` (record
ids, held-out flags) and a ``manifest.json`` with the teacher's identity. Records
keep corpus order; held-out records (the B2 split) are flagged, not skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def _heldout(record_id: str) -> bool:
    return int(hashlib.sha256(('b2-heldout:' + record_id).encode()).hexdigest()[:8], 16) % 200 == 0


def _regular_vocab(tok) -> dict[str, int]:
    added = set(tok.get_added_vocab())
    return {t: i for t, i in tok.get_vocab().items() if t not in added}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True, help='Hugging Face id or local path')
    parser.add_argument('--reference-tokenizer', default='LiquidAI/LFM2.5-350M',
                        help='the decoder\'s tokenizer; the teacher must match it')
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--batch-tokens', type=int, default=32768)
    parser.add_argument('--shard-size', type=int, default=8192, help='records per shard')
    parser.add_argument('--limit', type=int, default=0, help='first N records only (0 = all)')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    ref = AutoTokenizer.from_pretrained(args.reference_tokenizer)
    ours, theirs = _regular_vocab(ref), _regular_vocab(tok)
    different = [t for t, i in ours.items() if theirs.get(t) != i]
    if different or len(ours) != len(theirs):
        raise SystemExit(f'teacher vocabulary differs from the decoder\'s on {len(different)} '
                         f'tokens, e.g. {different[:5]}')
    ours_by_id = {i: t for t, i in ref.get_vocab().items()}
    theirs_by_id = {i: t for t, i in tok.get_vocab().items()}
    if max(ours_by_id) >= 65536:
        raise SystemExit('vocabulary does not fit uint16 ids')
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=getattr(torch, args.dtype))
    model.to(device).eval()
    # every output id whose meaning differs (including ids unused by one side)
    mismatched = [i for i in range(model.config.vocab_size)
                  if ours_by_id.get(i) != theirs_by_id.get(i)]
    bos = tok.bos_token_id if tok.bos_token_id is not None else ref.bos_token_id
    if bos != ref.bos_token_id:
        raise SystemExit('teacher and decoder BOS ids differ')
    excluded = torch.tensor(mismatched, dtype=torch.long, device=device)

    records = []
    with args.sources.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            records.append((row['record_id'], row['text']))
            if args.limit and len(records) >= args.limit:
                break
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'teacher': args.teacher, 'reference_tokenizer': args.reference_tokenizer,
                'top_k': args.top_k, 'max_tokens': args.max_tokens, 'bos': bos,
                'excluded_teacher_ids': mismatched,
                'sources': str(args.sources), 'records': len(records),
                'shard_size': args.shard_size, 'dtype': args.dtype,
                'layout': 'for each text token t_i: teacher distribution given BOS + t_<i'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    from safetensors.torch import save_file
    started = time.time()
    for shard in range(0, (len(records) + args.shard_size - 1) // args.shard_size):
        path = args.output / f'shard-{shard:05d}.safetensors'
        if path.exists():
            continue
        chunk = records[shard * args.shard_size:(shard + 1) * args.shard_size]
        encoded = [tok(text, add_special_tokens=False)['input_ids'][:args.max_tokens]
                   for _, text in chunk]
        results: list[tuple] = [None] * len(chunk)
        order = sorted(range(len(chunk)), key=lambda i: len(encoded[i]))
        start = 0
        while start < len(order):
            end = start
            while end < len(order) and (end - start + 1) * (len(encoded[order[end]]) + 1) <= args.batch_tokens:
                end += 1
            end = max(end, start + 1)
            batch = order[start:end]
            width = max(len(encoded[i]) for i in batch) + 1
            ids = torch.zeros(len(batch), width, dtype=torch.long)
            mask = torch.zeros(len(batch), width, dtype=torch.long)
            for row, i in enumerate(batch):
                seq = [bos] + encoded[i]
                ids[row, :len(seq)] = torch.tensor(seq)
                mask[row, :len(seq)] = 1
            with torch.no_grad():
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                logp = torch.log_softmax(logits.float(), -1)
                if len(excluded):  # ids that mean another token for our decoder
                    logp[..., excluded] = float('-inf')
                top, top_ids = logp.topk(args.top_k, dim=-1)
                rest = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
            for row, i in enumerate(batch):
                n = len(encoded[i])  # positions 0..n-1 predict t_0..t_{n-1}
                results[i] = (top_ids[row, :n].cpu(), top[row, :n].cpu(), rest[row, :n].cpu())
            start = end
            del logits, logp
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.int32)
        tokens = torch.tensor([t for e in encoded for t in e], dtype=torch.int32)
        save_file({
            'ids': torch.cat([r[0] for r in results]).to(torch.int32).to(torch.int16),
            'logprobs': torch.cat([r[1] for r in results]).to(torch.float16),
            'rest': torch.cat([r[2] for r in results]).to(torch.float16),
            'tokens': tokens.to(torch.int16),
            'lengths': lengths}, str(path.with_suffix('.pending')))
        path.with_suffix('.pending').replace(path)
        (args.output / f'shard-{shard:05d}.json').write_text(json.dumps({
            'record_ids': [r for r, _ in chunk],
            'heldout': [_heldout(r) for r, _ in chunk],
            'tokens': int(lengths.sum())}) + '\n')
        print(json.dumps({'shard': shard, 'records': len(chunk), 'tokens': int(lengths.sum()),
                          'elapsed_s': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
