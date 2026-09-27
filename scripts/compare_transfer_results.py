"""Compare full results.json files on the same held-out questions."""
import argparse
import hashlib
import json
from pathlib import Path

from schnitz.comparison import compare_transfer_results, compare_generation_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metric', choices=['choice', 'generation'], default='choice')
    args = parser.parse_args()
    inputs = {name: path.read_bytes() for name, path in
              [('baseline', args.baseline), ('candidate', args.candidate)]}
    compare = compare_transfer_results if args.metric == 'choice' else compare_generation_results
    report = compare(*(json.loads(inputs[n]) for n in ('baseline', 'candidate')))
    report['inputs'] = {name: {'path': str(getattr(args, name)),
                              'sha256': hashlib.sha256(data).hexdigest()}
                        for name, data in inputs.items()}
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
