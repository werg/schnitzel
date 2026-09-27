"""Check individual/bulk frozen-bank equivalence on the configured real model."""
import argparse
import json
from pathlib import Path
import time

import torch

from schnitz.agent import SchnitzelAgent
from schnitz.config import load_config
from schnitz.data import make_multiuse_world
from schnitz.evaluation import build_shared_bank, stored_transfer_evaluation
from schnitz.runtime import configure_memory
from schnitz.store import DiskStore
from schnitz.training import resource_report


class IndividualStore(DiskStore):
    def put_many(self, records):
        for record in records:
            self.put(record)


def validate(config_path, output):
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    configure_memory(config.train)
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(23)
    agent = SchnitzelAgent(config).to(config.train.device).eval()
    episodes = [e for i in range(2) for e in
                make_multiuse_world(i, split='batch-parity-20260919', bindings=2)]
    stores = {'individual': IndividualStore(output / 'individual.sqlite'),
              'bulk': DiskStore(output / 'bulk.sqlite')}
    timings = {}
    for name, store in stores.items():
        started = time.perf_counter()
        build_shared_bank(agent, store, episodes)
        timings[name] = time.perf_counter() - started
    with stores['individual'].connect() as a, stores['bulk'].connect() as b:
        assert a.execute('SELECT * FROM records ORDER BY namespace,record_id,space').fetchall() == b.execute(
            'SELECT * FROM records ORDER BY namespace,record_id,space').fetchall()

    def forbidden(*args, **kwargs):
        raise AssertionError('Writer called during reads')
    agent.produce = forbidden
    reports = {name: stored_transfer_evaluation(agent, store, episodes, drop_supports=True)
               for name, store in stores.items()}
    assert reports['individual']['rows'] == reports['bulk']['rows']
    result = dict(model=config.model.model_id, revision=config.model.revision,
                  precision=config.train.precision, storage_dtype=config.memory.storage_dtype,
                  records=stores['bulk'].sizes()['records'], queries=len(episodes),
                  serialized_rows_identical=True, all_evaluation_rows_identical=True,
                  evaluation_rows=len(reports['bulk']['rows']), writer_disabled_during_reads=True,
                  write_seconds=timings, resources=resource_report(),
                  notice='Small correctness check under shared-device/disk contention. Individual '
                         'writes run first; these are not randomized or cold-I/O throughput measurements.')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.config, args.output)
