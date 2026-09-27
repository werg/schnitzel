"""Random-access, versioned token IDs derived from an immutable episode file."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json


class TokenIndex:
    def __init__(self, path: str | Path, *, episode_sha256: str,
                 model_id: str, revision: str, arm: str):
        self.path = Path(path)
        manifest_path = self.path.with_suffix(self.path.suffix + '.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        expected = {'episode_sha256': episode_sha256, 'model_id': model_id,
                    'revision': revision, 'arm': arm}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('Tokenized episode artifact does not match data/model/prompt contract')
        self.offsets, digest = [], hashlib.sha256()
        with self.path.open('rb') as handle:
            while True:
                offset, line = handle.tell(), handle.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    self.offsets.append(offset)
        if len(self.offsets) != manifest.get('rows') or digest.hexdigest() != manifest.get('sha256'):
            raise ValueError('Tokenized episode artifact is incomplete or changed')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())
