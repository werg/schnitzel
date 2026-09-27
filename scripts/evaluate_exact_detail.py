"""Compare exact identifier generation from oracle-selected latent and text evidence."""
import argparse
import json
from pathlib import Path

import torch

from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from schnitz.frozen_scoring import FrozenScorer
from schnitz.operations import atomic_json, run_lock, stop_requested
from schnitz.sessions import read_session
from schnitz.store import DiskStore
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def selected_text(episode):
    sources = {s.record_id: s for s in episode.supports}
    if len(sources) != len(episode.supports) or not set(episode.required_ids) <= sources.keys():
        raise ValueError('Ambiguous or missing oracle source')
    selected = [s for s in episode.supports if s.record_id in episode.required_ids]
    if any(s.created_at >= episode.query_time for s in selected):
        raise ValueError('Future source in text control')
    return '\n'.join(s.text for s in selected)


@torch.no_grad()
def run(source, episodes_file, bank_path, reference_path, output, routing_probe, count_policy):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        reference = json.loads(reference_path.read_text())
        identity = {'source_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_file), 'bank_sha256': file_sha256(bank_path),
                    'reference_sha256': file_sha256(reference_path), 'script_sha256': file_sha256(__file__),
                    'max_new_tokens': 24}
        if any(reference['inputs'][k] != identity[k] for k in
               ('source_manifest_sha256', 'episodes_sha256', 'bank_sha256')):
            raise ValueError('Reference source/data/bank differs')
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Exact-detail diagnostic identity changed')
        atomic_json(output/'inputs.json', identity)
        if (output/'results.json').exists():
            if json.loads((output/'results.json').read_text())['inputs'] != identity:
                raise ValueError('Completed diagnostic identity changed')
            return
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        config.memory.neighbors = [2]
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=routing_probe,
                                          independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        if adapter != reference['inputs']['routing_probe'] or count != reference['inputs']['read_count_policy']:
            raise ValueError('Reference routing/count adapter differs')
        agent.requires_grad_(False)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Exact-detail inference invoked a writer')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer, store = FrozenScorer(agent), DiskStore(bank_path)
        originals = {(r['episode'], r['condition']): r for r in reference['generation_rows']}
        rows = []
        with autocast_context(config):
            for e in load_episodes(episodes_file):
                if e.task_family != 'multiuse/identifier':
                    continue
                if stop_requested(output):
                    raise RuntimeError('Stopped exact-detail evaluation; restart recomputes this diagnostic')
                text = selected_text(e)
                prompt = agent.prompt_ids(e.query)
                session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                       query_time=e.query_time, oracle_ids=e.required_ids)
                for condition in ('stored_required', 'text_required', 'none'):
                    context = text if condition == 'text_required' else ''
                    tokens = agent.prompt_ids(e.query, context)
                    memory = session.memory if condition == 'stored_required' else None
                    prediction = scorer.generate(tokens, memory, max_new_tokens=24)
                    if condition != 'text_required':
                        old = originals[(e.episode_id, 'selected_pair' if condition == 'stored_required' else 'none')]
                        if prediction != old['prediction']:
                            raise ValueError('Stored/no-memory generation differs from frozen reference')
                    rows.append({'episode': e.episode_id, 'environment': e.environment,
                                 'condition': condition, 'answer': e.answer, 'prediction': prediction,
                                 'exact_match': prediction.strip() == e.answer.strip(),
                                 'prompt_tokens': tokens.numel(), 'source_text_bytes': len(context.encode()),
                                 'selected_ids': list(e.required_ids) if condition != 'none' else []})
        summary = {c: {'n': sum(r['condition'] == c for r in rows),
                       'correct': sum(r['exact_match'] for r in rows if r['condition'] == c)}
                   for c in ('stored_required', 'text_required', 'none')}
        atomic_json(output/'results.json', {'inputs': identity, 'summary': summary, 'rows': rows,
            'notice': 'Known-corpus diagnostic; oracle selection and identical source information, unequal text/latent input budgets. '
                      'Text is a separate control, not an enabled exact-detail inference path. No candidate answers passed to generation.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'bank', 'reference', 'output', 'routing-probe', 'count-policy'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.episodes, args.bank, args.reference, args.output, args.routing_probe, args.count_policy)
