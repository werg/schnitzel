"""B0 of the BGKit restart: reproduce S2's reconstruct quality in our harness.

Loads the S2 encoder/decoder pair exactly as BGKit's trainer does
(``bgkit2.training.standalone.load_models``) and scores teacher-forced
reconstruction through BGKit's own evaluation arm (``core._arm``) on

- BGKit's own corpus eval split (same stores, seed and sampler as the trainer;
  compare with the checkpoint's ``metadata.json``), and
- SCHNITZELJAGD passages: bank sources and invented passages, with the question-free
  reconstruct prompt.

Arms: no context, full text, zeroed x4, and x4/x8/x16/x32/x64 reps. Reports mean
NLL, token accuracy and the captured fraction of the full-text gain. Runs in the
``schnitz-bgkit`` container (BGKit image; BGKit source and checkpoints mounted
read-only). Training-only teacher: nothing here is an SCHNITZELJAGD inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

RATIOS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)


def _captured(floor: float, value: float, full: float) -> float:
    return (floor - value) / max(floor - full, 1e-9)


def _arms(core, batches) -> dict:
    floor = core.teacher_forced(batches, None)
    full = core.teacher_forced(batches, 'full')
    zero = core.teacher_forced(batches, 0.25, zero_reps=True)
    out = {}
    for task in floor:
        entry = {'noctx': {'nll': floor[task][0], 'acc': floor[task][1]},
                 'full': {'nll': full[task][0], 'acc': full[task][1]},
                 'zeroed_x4': {'nll': zero[task][0]}}
        out[task] = entry
    for ratio in RATIOS:
        arm = core.teacher_forced(batches, ratio)
        for task, (nll, acc) in arm.items():
            out[task][f'x{round(1 / ratio)}'] = {
                'nll': nll, 'acc': acc,
                'captured': _captured(out[task]['noctx']['nll'], nll, out[task]['full']['nll'])}
    return out


def _passage_samples(core, texts: list[str], prompts: dict):
    from bgkit2.data.autoencode import Sample
    tok = core.tok
    samples = []
    for index, text in enumerate(texts):
        ids = torch.tensor(tok(text, add_special_tokens=False)['input_ids'], dtype=torch.long)
        samples.append(Sample(ctx_ids=ids, target_ids=ids.clone(), task='reconstruct',
                              store=0, doc=index, prompt_ids=prompts['reconstruct']))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--experiment', default='bgkit2_s2_showcase')
    parser.add_argument('--sources', type=Path, action='append', default=[],
                        help='SCHNITZELJAGD source manifests (JSONL with "text"); one set each')
    parser.add_argument('--per-set', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cuda-fraction', type=float, default=0.10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from bgkit_core.host_memory_guard import cap_cuda
    cap_cuda(args.cuda_fraction)
    from bgkit2.data.autoencode import AutoencodeDataset, Collator
    from bgkit2.data.templates import Templates
    from bgkit2.data.token_store import TokenStore
    from bgkit2.data.tasks import ChatRenderer
    from bgkit2.training.standalone import load_models

    core = load_models(args.experiment, str(args.checkpoint))
    d = core.cfg2.data
    core.templates = Templates.from_tokenizer(core.tok, style=d.prompt_style)
    core.collate = Collator(core.templates, enc_pad_id=None if core.ctx_tok is None
                            else core.chat_renderer().enc_pad_id)
    prompts = Templates.compression_prompt_ids(core.ctx_tok or core.tok)
    results = {'checkpoint': str(args.checkpoint), 'experiment': args.experiment, 'sets': {}}

    with torch.no_grad():
        # BGKit's own eval split, built as BgKIT2Core.build_datasets builds it.
        evald = AutoencodeDataset([TokenStore(p) for p in d.eval_stores], ctx_min=d.ctx_min,
                                  ctx_max=d.ctx_max, cont_min=d.cont_min, cont_max=d.cont_max,
                                  p_continue=d.p_continue, seed=10_007,
                                  epoch_size=d.eval_samples, task_prompt_ids=prompts)
        if core.ctx_tok is not None:
            evald.dec_view = ChatRenderer.from_tokenizer(core.tok, enc_tok=core.ctx_tok).dec_view
        batches = core.eval_batches(evald, d.eval_batch_size)
        results['sets']['bgkit_eval'] = _arms(core, batches)
        print(json.dumps({'bgkit_eval': results['sets']['bgkit_eval']}), flush=True)

        for path in args.sources:
            rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
            random.Random(0).shuffle(rows)
            texts = [row['text'] for row in rows[:args.per_set]]
            samples = _passage_samples(core, texts, prompts)
            batches = [core.collate(samples[i:i + args.batch_size])
                       for i in range(0, len(samples), args.batch_size)]
            results['sets'][path.name] = _arms(core, batches)
            print(json.dumps({path.name: results['sets'][path.name]}), flush=True)

    meta = json.loads((args.checkpoint / 'metadata.json').read_text())['metrics']
    results['metadata_reference'] = {k: v for k, v in meta.items()
                                     if k.startswith('eval/') and '/reconstruct/' in k
                                     and k.split('/')[-1] in ('loss', 'token_acc', 'captured')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
