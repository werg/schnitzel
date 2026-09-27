"""Materialize immutable source/prompt/target token IDs for training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from schnitz.config import load_config
from schnitz.episode_index import EpisodeIndex
from schnitz.text import render_prompt


def build(config_path: Path, episodes_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    episodes = EpisodeIndex(episodes_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id,
        revision=config.model.revision, trust_remote_code=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + '.pending')
    digest = hashlib.sha256()
    with pending.open('wb') as handle:
        for episode in episodes:
            support_text = ('\n'.join(source.text for source in episode.supports
                            if source.record_id in episode.required_ids)
                            if config.train.arm == 'oracle_text' else '')
            prompt = tokenizer.encode(render_prompt(tokenizer, episode.query, support_text),
                                      add_special_tokens=False)
            target = tokenizer.encode(episode.answer, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                target.append(tokenizer.eos_token_id)
            sources = [tokenizer.encode(source.text, add_special_tokens=True)
                       for source in episode.supports]
            if any(len(ids) > config.train.max_source_tokens for ids in sources):
                raise ValueError(f'Source token limit exceeded in {episode.episode_id}')
            if len(prompt) > config.train.max_prompt_tokens or len(target) > config.train.max_target_tokens:
                raise ValueError(f'Prompt/target token limit exceeded in {episode.episode_id}')
            row = {'episode_id': episode.episode_id,
                   'source_record_ids': [source.record_id for source in episode.supports],
                   'source_ids': sources, 'prompt_ids': prompt, 'target_ids': target}
            line = (json.dumps(row, separators=(',', ':')) + '\n').encode()
            handle.write(line)
            digest.update(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, output)
    manifest = {'format': 1, 'rows': len(episodes), 'sha256': digest.hexdigest(),
                'episode_sha256': episodes.sha256, 'model_id': config.model.model_id,
                'revision': config.model.revision, 'arm': config.train.arm,
                'max_source_tokens': config.train.max_source_tokens,
                'max_prompt_tokens': config.train.max_prompt_tokens,
                'max_target_tokens': config.train.max_target_tokens}
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    pending_manifest = manifest_path.with_suffix(manifest_path.suffix + '.pending')
    pending_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(pending_manifest, manifest_path)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.episodes, args.output), indent=2))
