"""Frozen-writer, identical-bank evaluation across recurrent depths."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_model

from .agent import SchnitzelAgent
from .checkpoints import resolve_checkpoint
from .data import load_episodes
from .evaluation import build_shared_bank, stored_transfer_evaluation
from .operations import atomic_json, run_lock, stop_requested
from .store import DiskStore
from .training import config_from_run, resource_report
from .trajectory_eval import build_teacher_bank, stored_teacher_evaluation
from .trajectories import file_sha256
from .checkpoints import stop_on_signal


def evaluate_depths(run: str | Path, episodes_file: str | Path, output: str | Path, *,
                    depths: tuple[int, ...] = (1, 2, 3, 4), protocol: str = 'transfer',
                    max_episodes: int = 32) -> dict:
    """Materialize once with writer depth one, never regenerate between depths.

    Wall time covers prefix-plan construction, I/O, and all scored conditions; it
    is not a serving-latency benchmark. R=1 intentionally cannot read between loops.
    """
    if not depths or any(d < 1 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError('Depths must be distinct positive integers')
    if protocol not in {'transfer', 'teacher'} or max_episodes < 1:
        raise ValueError('Invalid evaluation protocol/limit')
    config = config_from_run(Path(run))
    if config.model.recurrence_mode != 'middle_block' or config.model.writer_loops != 1:
        raise ValueError('Depth isolation requires middle-block recurrence and fixed writer_loops=1')
    if config.train.arm != 'memory':
        raise ValueError('This sweep evaluates a trained latent-memory student')
    if any(1 < d <= config.memory.read_steps for d in depths):
        raise ValueError('A depth would truncate the scheduled read plan; use deeper passes')
    from .runtime import configure_memory
    configure_memory(config.train)
    checkpoint = resolve_checkpoint(run, verify=True)
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes:
        raise ValueError('Empty evaluation set')
    if protocol == 'transfer' and any(not e.choices for e in episodes):
        raise ValueError('Transfer accuracy requires annotated choices; use --protocol teacher')
    directory = Path(output)
    identity = dict(version=1, checkpoint_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    checkpoint_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file), evaluator_sha256=file_sha256(__file__),
                    protocol=protocol, depths=list(depths), max_episodes=max_episodes)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory), stop_on_signal() as signals:
        identity_file = directory / 'inputs.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text()) != identity:
                raise ValueError('Depth evaluation identity changed')
        elif any(directory.iterdir()):
            raise ValueError('Unidentified depth evaluation directory; use a new output path')
        else:
            atomic_json(identity_file, identity)
        summary_file = directory / 'summary.json'
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        from .runtime import available_host_memory
        def want_stop():
            available = available_host_memory()
            return bool(signals['signal'] or stop_requested(directory) or
                        (available is not None and available < config.train.min_system_available_bytes) or
                        shutil.disk_usage(directory).free < config.train.min_free_disk_bytes)
        if want_stop():
            return dict(status='checkpointed', directory=str(directory))
        torch.set_num_threads(config.train.threads)
        torch.manual_seed(config.train.seed)
        agent = SchnitzelAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        store = DiskStore(directory / 'bank.sqlite')
        bank_file = directory / 'bank-progress.json'
        current_write = (build_shared_bank(agent, store, episodes,
                         writer_identity=identity['checkpoint_model_sha256']) if protocol == 'transfer'
                         else build_teacher_bank(agent, store, episodes,
                         writer_identity=identity['checkpoint_model_sha256'], want_stop=want_stop))
        if not current_write.get('complete', True) or want_stop():
            return dict(status='checkpointed', directory=str(directory))
        if bank_file.exists():
            bank_progress = json.loads(bank_file.read_text())
            if bank_progress['inputs'] != identity:
                raise ValueError('Depth bank progress identity changed')
            write_phase = bank_progress['write_phase']
        else:
            write_phase = current_write
            atomic_json(bank_file, dict(inputs=identity, write_phase=write_phase))
        def forbidden(*args, **kwargs):
            raise AssertionError('The depth sweep must not call the writer after materialization')
        agent.produce = forbidden
        reports = []
        for depth in depths:
            report_file = directory / f'depth-{depth}.json'
            if report_file.exists():
                report = json.loads(report_file.read_text())
                reports.append({k: v for k, v in report.items() if k != 'rows'})
                continue
            if want_stop():
                return dict(status='checkpointed', directory=str(directory), depth=depth)
            agent.backbone.loops = depth
            if config.train.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            progress_file = directory / f'depth-{depth}-progress.json'
            prior = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            if prior and prior.get('inputs') != identity:
                raise ValueError('Depth scoring progress identity changed')
            elapsed_before = prior.get('elapsed_seconds', 0.)
            start = time.perf_counter()
            def save_progress(rows, plans):
                atomic_json(progress_file, dict(inputs=identity, rows=rows, plans=plans,
                    elapsed_seconds=elapsed_before + time.perf_counter() - start))
            report = (stored_transfer_evaluation(agent, DiskStore(store.path), episodes, drop_supports=True)
                      if protocol == 'transfer' else stored_teacher_evaluation(
                          agent, DiskStore(store.path), episodes,
                          rows=prior.get('rows', []), saved_plans=prior.get('plans', {}),
                          progress=save_progress, want_stop=want_stop))
            if report is None:
                return dict(status='checkpointed', directory=str(directory), depth=depth)
            if config.train.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = elapsed_before + time.perf_counter() - start
            report.update(recurrence=agent.backbone.manifest(), evaluation_wall_seconds=elapsed,
                          depth_exceeds_final_training_max=depth > max(config.train.loop_counts or [config.model.loops]),
                          depth_in_final_main_objective=depth in (config.train.loop_counts or [config.model.loops]),
                          memory_enabled=depth > 1, resources=resource_report())
            atomic_json(report_file, report)
            reports.append({k: v for k, v in report.items() if k != 'rows'})
        result = dict(status='complete', protocol=protocol, writer_depth=1, write_phase=write_phase,
                      same_serialized_bank=True, checkpoint=str(checkpoint), bank_sizes=store.sizes(), depths=reports,
                      timing_scope='All conditions, fixed-plan candidate scoring, prefix recomputation and I/O included after bank materialization. OS page-cache state is uncontrolled; this is neither cold-NVMe nor serving latency.',
                      retrieval_scope=('Oracle source IDs; no ANN search' if config.train.retrieval == 'oracle'
                                       else 'Exact stored-key scan; no ANN index'),
                      comparison='R=1 is a no-in-loop-memory control. Compare R>=2 at fixed read budget; no claim of extrapolation gains.')
        atomic_json(summary_file, result)
        return result
