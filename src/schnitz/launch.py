"""Synchronous staged SCHNITZELJAGD training with immutable inputs and resumable stages."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil
import tempfile
import time

import yaml
from .backbones import ByteTokenizer
from .config import load_config, TrainConfig
from .data import make_boolean_world, make_multiuse_world, save_episodes, evidence_ids
from .episode_index import EpisodeIndex
from .text import render_prompt
from .trajectories import prepare_trajectories, pinned_spec, digest, file_sha256


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text())
    allowed = {'base_config', 'protocol', 'seed', 'validation_fraction', 'sources',
               'preparation', 'stages', 'evaluation', 'causal_train_worlds', 'bindings'}
    if not isinstance(recipe, dict) or set(recipe) - allowed:
        raise ValueError('Unknown recipe fields')
    if recipe.get('protocol') not in {'prefix', 'cross_experience', 'causal', 'binding'}:
        raise ValueError('Invalid protocol')
    if not isinstance(recipe.get('bindings', 2), int) or recipe.get('bindings', 2) < 1:
        raise ValueError('Positive integer bindings required')
    names = []
    if not recipe.get('stages'):
        raise ValueError('At least one stage required')
    for stage in recipe['stages']:
        if set(stage) - {'name', 'arm', 'steps', 'freeze_backbone', 'init_from', 'oracle_anchor_weight', 'compaction',
                         'loops', 'loop_counts', 'backbone_train_scope', 'oracle_anchor_loops', 'parent_kl_weight', 'compact_records'}:
            raise ValueError('Unknown stage fields')
        if not stage['name'].replace('_', '').isalnum() or stage['name'] in names:
            raise ValueError('Stage names must be unique safe identifiers')
        if stage.get('init_from') and stage['init_from'] not in names:
            raise ValueError('Stage dependency must precede stage')
        if stage['steps'] < 1:
            raise ValueError('Positive stage steps required')
        names.append(stage['name'])
    return recipe


def _tokenizer(config):
    if config.model.backend == 'tiny':
        return ByteTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.model.model_id, revision=config.model.revision, trust_remote_code=False)


def prepare_launch(recipe_path, output, *, resume=False):
    recipe = load_recipe(recipe_path)
    output = Path(output).resolve()
    base = load_config(recipe['base_config'])
    identity_config = asdict(base)
    # Defaults preserving old behavior must not invalidate an existing 0.4 recipe.
    defaults = TrainConfig()
    for name in ('archive_dir', 'archive_keep_checkpoints', 'min_free_disk_bytes',
                 'wandb_mode', 'wandb_project', 'wandb_entity', 'wandb_group', 'evidence_scope',
                 'optimizer', 'weight_decay', 'adam_betas', 'adam_eps', 'muon_momentum', 'muon_ns_steps',
                 'cuda_memory_fraction', 'min_system_available_bytes', 'stall_timeout_seconds'):
        if identity_config['train'][name] == getattr(defaults, name):
            identity_config['train'].pop(name)
    identity = digest({'recipe': recipe, 'base_config': identity_config})
    manifest_path = output / 'launch.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Launch exists; use --resume')
        old = json.loads(manifest_path.read_text())
        if old['input_identity'] != identity:
            raise ValueError('Recipe/base configuration changed; use a new output directory')
        for name, expected in old['prepared_checksums'].items():
            if file_sha256(output / name) != expected:
                raise ValueError(f'Prepared input changed: {name}')
        return old
    if output.exists() and not resume:
        raise FileExistsError('Output exists; use a fresh directory or resume interrupted preparation')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'input-lock.json'
    if lock.exists() and json.loads(lock.read_text())['input_identity'] != identity:
        raise ValueError('Interrupted preparation recipe/base configuration changed')
    atomic_json(lock, {'input_identity': identity})
    lock = output / 'model-lock.json'
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous['model_id'] != base.model.model_id:
            raise ValueError('Model lock mismatch')
        base.model.revision = previous['revision']
    else:
        if base.model.backend == 'hf':
            from huggingface_hub import HfApi
            info = HfApi().model_info(base.model.model_id, revision=base.model.revision)
            if not info.sha:
                raise RuntimeError('Model revision could not be resolved')
            base.model.revision = info.sha
        atomic_json(lock, {'model_id': base.model.model_id, 'revision': base.model.revision, 'backend': base.model.backend})
    tokenizer = _tokenizer(base)
    base.train.seed = recipe.get('seed', 17)
    data = output / 'data'
    dataset_lock = output / 'datasets-lock.json'
    if recipe['protocol'] not in {'causal', 'binding'}:
        if dataset_lock.exists():
            locked_sources = json.loads(dataset_lock.read_text())
        else:
            locked_sources = [pinned_spec(request) for request in recipe['sources']]
            atomic_json(dataset_lock, locked_sources)
    if not (data / 'manifest.json').exists():
        with tempfile.TemporaryDirectory(prefix='.data-', dir=output) as temp:
            pending = Path(temp) / 'data'
            if recipe['protocol'] in {'causal', 'binding'}:
                pending.mkdir()
                worlds = recipe.get('causal_train_worlds', 128)
                for split, count in [('train', worlds), ('validation', max(1, recipe.get('evaluation', {}).get('causal_worlds', 32)))]:
                    namespace = f"{recipe['protocol']}-{split}-{base.train.seed}"
                    examples = [e for i in range(count) for e in (
                        make_boolean_world(i, split=namespace, operations=('a', 'b', 'xor'))
                        if recipe['protocol'] == 'causal' else
                        make_multiuse_world(i, split=namespace, bindings=recipe.get('bindings', 2)))]
                    save_episodes(pending / f'{split}.jsonl', examples)
                atomic_json(pending / 'manifest.json', {'protocol': recipe['protocol'], 'worlds': worlds})
            else:
                budgets = recipe.get('preparation', {}) | {k: getattr(base.train, k) for k in
                    ('max_source_tokens', 'max_prompt_tokens', 'max_target_tokens')}
                prepare_trajectories(locked_sources, tokenizer, budgets, pending, protocol=recipe['protocol'],
                                     seed=base.train.seed, validation_fraction=recipe.get('validation_fraction', .1))
            if data.exists():
                raise ValueError('Incomplete data directory; inspect it before reusing output')
            shutil.move(str(pending), str(data))
    for split in ('train', 'validation'):
        for e in EpisodeIndex(data / f'{split}.jsonl'):
            if len(tokenizer.encode(e.answer, add_special_tokens=False)) + int(tokenizer.eos_token_id is not None) > base.train.max_target_tokens:
                raise ValueError('Prepared target overflow')
            visible = evidence_ids(e, base.train.evidence_scope)
            support_text = '\n'.join(s.text for s in e.supports if s.record_id in visible)
            if len(tokenizer.encode(render_prompt(tokenizer, e.query, support_text), add_special_tokens=False)) > base.train.max_prompt_tokens:
                raise ValueError('Prepared oracle-text overflow')
    stages = []
    for stage in recipe['stages']:
        config = deepcopy(base)
        config.train.episodes_file = str(data / 'train.jsonl')
        config.train.arm, config.train.steps = stage['arm'], stage['steps']
        config.model.freeze_backbone = stage.get('freeze_backbone', False)
        config.model.loops = stage.get('loops', base.model.loops)
        config.model.backbone_train_scope = stage.get('backbone_train_scope', base.model.backbone_train_scope)
        config.train.loop_counts = stage.get('loop_counts', [])
        config.train.oracle_anchor_loops = stage.get('oracle_anchor_loops', base.train.oracle_anchor_loops)
        config.train.oracle_anchor_weight = stage.get('oracle_anchor_weight', 0.)
        config.train.parent_kl_weight = stage.get('parent_kl_weight', 0.)
        if stage.get('compaction'):
            config.memory.compaction = 'synthetic'
            config.memory.compact_records = stage.get('compact_records', config.memory.compact_records)
            config.memory.compaction_probability, config.memory.compaction_warmup = 1., 0
            config.train.optimization_scope = 'compactor'
        config.validate()
        from .routing import validate_routing_dataset
        validate_routing_dataset(config, EpisodeIndex(data / 'train.jsonl'))
        name = stage['name']
        path = output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        stages.append(dict(name=name, config=str(path), run=str(output / name),
                           init_from=str(output / stage['init_from']) if stage.get('init_from') else None))
    files = ['input-lock.json', 'model-lock.json', 'data/train.jsonl', 'data/validation.jsonl', 'data/manifest.json']
    if dataset_lock.exists():
        files.append('datasets-lock.json')
    if (data / 'normalized.jsonl').exists():
        files.append('data/normalized.jsonl')
    files += [s['name'] + '.yaml' for s in stages]
    manifest = dict(format=1, project='SCHNITZELJAGD', input_identity=identity, recipe=recipe, stages=stages,
                    resolved_model_revision=base.model.revision, backend=base.model.backend,
                    prepared_checksums={f: file_sha256(output / f) for f in files}, prepared_at_unix=time.time())
    atomic_json(manifest_path, manifest)
    return manifest


def launch(recipe_path, output, *, resume=False, prepare_only=False, preserve_stop=False):
    from .operations import run_lock
    from .checkpoints import stop_on_signal
    with run_lock(output, clear_stop=not preserve_stop), stop_on_signal() as stop:
        return _launch(recipe_path, output, resume=resume, prepare_only=prepare_only, stop=stop)


def _launch(recipe_path, output, *, resume, prepare_only, stop):
    from .operations import stop_requested
    def stopping():
        return stop['signal'] is not None or stop_requested(output)
    manifest = prepare_launch(recipe_path, output, resume=resume)
    output = Path(output).resolve()
    if prepare_only:
        return {'status': 'prepared', 'manifest': str(output / 'launch.json'), 'stages': manifest['stages']}
    if stopping():
        return {'status': 'checkpointed', 'output': str(output), 'boundary': 'prepared'}
    from .probes import model_probe
    from .training import train
    from .checkpoints import resolve_checkpoint
    from .trajectory_eval import evaluate_teacher_run
    probe = output / 'model-probe.json'
    if not probe.exists():
        atomic_json(probe, model_probe(load_config(manifest['stages'][0]['config'])))
    for stage in manifest['stages']:
        if stopping():
            return {'status': 'checkpointed', 'output': str(output), 'boundary': 'between_stages'}
        run = Path(stage['run'])
        config, completed = load_config(stage['config']), 0
        if (run / 'CURRENT').exists():
            checkpoint = resolve_checkpoint(run, verify=True)
            completed = json.loads((checkpoint / 'manifest.json').read_text())['step']
            if completed > config.train.steps:
                raise ValueError('Stage advanced outside its immutable launch plan')
        if completed < config.train.steps:
            if run.exists() and not (run / 'CURRENT').exists():
                raise RuntimeError(f'Uncommitted training directory {run}; inspect before retrying')
            result = train(config, run, resume=(run / 'CURRENT').exists(),
                           init_from=None if (run / 'CURRENT').exists() else stage['init_from'],
                           stop_output=output)
            if result['stopped_early'] or result.get('stop_requested'):
                return dict(status='checkpointed', stage=stage['name'], steps=result['steps'])
        if stopping():
            return dict(status='checkpointed', stage=stage['name'])
        report = output / (stage['name'] + '-evaluation.json')
        if not report.exists():
            evaluation = evaluate_teacher_run(run, output / 'data/validation.jsonl',
                max_episodes=manifest['recipe'].get('evaluation', {}).get('max_episodes', 64),
                output=output / (stage['name'] + '-evaluation'))
            if evaluation.get('status') == 'checkpointed':
                return dict(status='checkpointed', stage=stage['name'], boundary='teacher_evaluation')
            atomic_json(report, evaluation)
    last = manifest['stages'][-1]
    last_config = load_config(last['config'])
    compact_evaluation = (last_config.memory.compaction != 'none' and last_config.train.arm == 'memory')
    count = manifest['recipe'].get('evaluation', {}).get('causal_worlds', 32)
    split = 'post-freeze-' + manifest['input_identity'][:12]
    if count:
        if stopping():
            return dict(status='checkpointed', boundary='before_causal_evaluation')
        from .evaluation import evaluate_transfer_run
        if not (output / 'causal-evaluation.json').exists():
            path = output / 'fresh-causal.jsonl'
            save_episodes(path, [e for i in range(count) for e in make_boolean_world(i, split=split, operations=('a', 'b', 'xor'))])
            result = evaluate_transfer_run(last['run'], path, boolean_counterfactuals=True, drop_supports=True,
                                           persistent_compact=compact_evaluation)
            atomic_json(output / 'causal-evaluation.json', {k: v for k, v in result.items() if k != 'rows'})
        # Independent marker: a failed binding evaluation does not silently get skipped on resume.
        if not (output / 'multiuse-evaluation.json').exists():
            if stopping():
                return dict(status='checkpointed', boundary='before_binding_evaluation')
            path = output / 'fresh-multiuse.jsonl'
            binding_protocol = manifest['recipe']['protocol'] == 'binding'
            save_episodes(path, [e for i in range(count) for e in make_multiuse_world(
                i, split=split, bindings=manifest['recipe'].get('bindings', 2))])
            result = evaluate_transfer_run(last['run'], path, persistent_compact=compact_evaluation,
                                           drop_supports=binding_protocol, binding_counterfactuals=binding_protocol)
            atomic_json(output / 'multiuse-evaluation.json', {k: v for k, v in result.items() if k != 'rows'})
    return dict(status='complete', output=str(output), stages=[s['name'] for s in manifest['stages']],
                scientific_claim='Inspect stored-memory and counterfactual effects; execution alone is not evidence of transfer.')
