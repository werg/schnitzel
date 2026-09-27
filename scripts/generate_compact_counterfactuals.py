"""Generate changed action answers from existing compact banks, without candidates."""
import argparse
import json
from pathlib import Path
import re

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.cluster_store import ClusterBank, state_fingerprint
from schnitz.data import load_episodes
from schnitz.evaluation_adapter import load_frozen_agent
from schnitz.frozen_scoring import FrozenScorer
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


@torch.no_grad()
def run(source, confirmation, episodes_file, output, methods, worlds=8):
    if (worlds < 1 or not methods or len(methods) != len(set(methods))
            or any(not re.fullmatch(r'[a-z][a-z0-9_]*', m) for m in methods)):
        raise ValueError('Positive world count and safe method names required')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(output.parent, clear_stop=False):
        inputs = json.loads((confirmation / 'inputs.json').read_text())
        if worlds > inputs['generation_worlds']:
            raise ValueError('Requested worlds exceed the existing original-generation reference')
        checkpoint = resolve_checkpoint(source, verify=True)
        if (file_sha256(checkpoint / 'manifest.json') != inputs['source_manifest_sha256'] or
                file_sha256(episodes_file) != inputs['episodes_sha256']):
            raise ValueError('Reference source or episode identity changed')
        episodes = load_episodes(episodes_file)
        selected = set(list(dict.fromkeys(e.environment for e in episodes))[:worlds])
        original = {e.episode_id: e for e in episodes if e.environment in selected and e.task_family == 'multiuse/action'}
        if not original:
            raise ValueError('No action questions in the requested worlds')
        reports = {method: json.loads((confirmation / f'{method}.json').read_text()) for method in methods}
        for method, report in reports.items():
            if report['inputs'] != inputs or report['method'] != method:
                raise ValueError('Reference arm identity changed')
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Stored generation invoked a writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer = FrozenScorer(agent)
        reader_hash = state_fingerprint(agent.reader)
        rows = []
        bank_hashes = {}
        with autocast_context(config):
            for kind in ('permission', 'restoration'):
                store = DiskStore(confirmation / f'{kind}.sqlite')
                bank_hashes[kind] = file_sha256(store.path)
                changed = [e for e in load_episodes(confirmation / f'{kind}.jsonl') if e.episode_id in original]
                if len(changed) != len(original):
                    raise ValueError('Counterfactual action grid differs')
                for method in methods:
                    baseline = {r['episode']: r for r in reports[method]['generation_rows'] if r['condition'] == 'all'}
                    codes = None if method == 'raw' else ClusterBank(store, view=method, reader_hash=reader_hash)
                    for e in changed:
                        if stop_requested(output.parent):
                            raise RuntimeError('Counterfactual generation stopped')
                        old = original[e.episode_id]
                        if e.query != old.query or e.required_ids != old.required_ids or e.query_time != old.query_time:
                            raise ValueError('Counterfactual changed the query or evidence identities')
                        previous = baseline[e.episode_id]
                        if previous['answer'] != old.answer:
                            raise ValueError('Original generated answer identity differs')
                        prompt = agent.prompt_ids(e.query)
                        session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                               query_time=e.query_time, oracle_ids=e.required_ids, cluster_bank=codes)
                        if codes is not None and not any(a['clusters'] for a in session.payload_accounting):
                            raise ValueError('Expected a full-cluster compact read for this action')
                        prediction = scorer.generate(prompt, session.memory, max_new_tokens=inputs['max_new_tokens'])
                        rows.append({'method': method, 'variant': kind, 'episode': e.episode_id, 'environment': e.environment,
                                     'answer': e.answer, 'prediction': prediction, 'exact_match': prediction == e.answer,
                                     'original_answer': old.answer, 'original_prediction': previous['prediction'],
                                     'should_change': old.answer != e.answer,
                                     'both_correct': previous['exact_match'] and prediction == e.answer,
                                     'prediction_changed': prediction != previous['prediction'],
                                     'payload_accounting': session.payload_accounting})
                    print(json.dumps({'completed_method': method, 'variant': kind}), flush=True)
        summary = {}
        for method in methods:
            summary[method] = {}
            for kind in ('permission', 'restoration'):
                group = [r for r in rows if r['method'] == method and r['variant'] == kind]
                changed, unchanged = [r for r in group if r['should_change']], [r for r in group if not r['should_change']]
                summary[method][kind] = {'queries': len(group), 'exact_correct': sum(r['exact_match'] for r in group),
                    'changed_pairs': len(changed), 'both_correct_on_change': sum(r['both_correct'] for r in changed),
                    'unchanged_pairs': len(unchanged), 'false_changes': sum(r['prediction_changed'] for r in unchanged)}
        atomic_json(output, {'source_inputs': inputs, 'worlds': len(selected), 'bank_sha256': bank_hashes,
                            'reference_report_sha256': {m: file_sha256(confirmation / f'{m}.json') for m in methods},
                            'by_method': summary, 'rows': rows, 'decoder_reuse': scorer.report(),
                            'notice': 'Greedy counterfactual action strings; no candidate answers, writer, compactor or agent execution.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['raw', 'mean', 'trained'])
    parser.add_argument('--worlds', type=int, default=8)
    args = parser.parse_args()
    run(args.source, args.confirmation, args.episodes, args.output, args.methods, args.worlds)
