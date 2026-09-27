"""Dry-run or apply verified deduplication of immutable completed checkpoint weights."""
import argparse
import json
from pathlib import Path

from schnitz.checkpoint_dedup import deduplicate_weights
from schnitz.operations import atomic_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    root = args.root.resolve(strict=True)
    checkpoints = [p.parent for p in root.rglob('manifest.json')
                   if p.parent.parent.name == 'checkpoints' and p.parent.name.startswith('step-')
                   and p.parent == p.parent.resolve()]
    report = deduplicate_weights(checkpoints, apply=args.apply)
    atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'replacements'}, indent=2))
