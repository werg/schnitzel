"""Prepare a pinned 10k-source passage bank and article-disjoint QA episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from schnitz.corpus_data import prepare_squad
from schnitz.operations import atomic_json
from schnitz.trajectories import file_sha256


REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(root: Path, cache: Path, raw: Path):
    if not root.parent.is_dir() or not cache.is_dir() or not raw.is_file():
        raise FileNotFoundError('External output parent, cache and raw corpus required')
    if root.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Corpus and source manifest must live on the external disk')
    if shutil.disk_usage(root.parent).free < 10 * 1024**3:
        raise OSError('Preparation requires a 10 GiB disk reserve')
    from transformers import AutoTokenizer
    snapshot = cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer_files = {p.name: file_sha256(p) for p in snapshot.iterdir()
                       if p.name in {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    declaration = {'source': str(raw), 'source_sha256': file_sha256(raw),
                   'preparer_sha256': file_sha256(Path(__file__)),
                   'corpus_module_sha256': file_sha256(Path(prepare_squad.__code__.co_filename)),
                   'tokenizer_revision': REVISION, 'tokenizer_files': tokenizer_files,
                   'max_train_sources': 10000, 'max_validation_sources': 1024,
                   'max_source_tokens': 512, 'max_target_tokens': 128,
                   'notice': 'Public SQuAD v2.0 recorded passages/questions; no live teacher. Raw contents and sources stay external.'}
    root.mkdir(exist_ok=True)
    lock = root / 'inputs.json'
    if lock.exists():
        if json.loads(lock.read_text()) != declaration:
            raise ValueError('Corpus preparation declaration changed')
    else:
        atomic_json(lock, declaration)
    output = root / 'data'
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if (manifest['source_sha256'] != declaration['source_sha256']
                or any(file_sha256(output / name) != digest for name, digest in manifest['sha256'].items())):
            raise ValueError('Completed corpus bytes differ from manifest')
        return manifest
    return prepare_squad(raw, output, tokenizer,
                         max_train_sources=declaration['max_train_sources'],
                         max_validation_sources=declaration['max_validation_sources'],
                         max_source_tokens=declaration['max_source_tokens'],
                         max_target_tokens=declaration['max_target_tokens'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.cache, args.raw), indent=2), flush=True)
