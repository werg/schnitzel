from dataclasses import asdict
from pathlib import Path
import json
import pytest
import yaml
from schnitz.launch import prepare_launch, launch
from schnitz.agent import SchnitzelAgent
from schnitz.data import load_episodes
from schnitz.store import DiskStore
from schnitz.trajectory_eval import build_teacher_bank, stored_teacher_evaluation

ROOT = Path(__file__).parents[1]


def make_recipe(tmp_path, tiny_config):
    tiny_config.train.max_prompt_tokens, tiny_config.train.max_source_tokens = 1400, 128
    config = tmp_path / 'base.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    recipe = yaml.safe_load((ROOT / 'recipes/offline_smoke.yaml').read_text())
    recipe['base_config'] = str(config)
    recipe['sources'][0]['local_file'] = str(ROOT / 'tests/fixtures/trajectories.jsonl')
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_preparation_resume_and_input_integrity(tmp_path, tiny_config):
    recipe, out = make_recipe(tmp_path, tiny_config), tmp_path / 'launch'
    m = prepare_launch(recipe, out)
    assert m['backend'] == 'tiny' and len(m['stages']) == 3
    assert prepare_launch(recipe, out, resume=True) == m
    with pytest.raises(FileExistsError):
        prepare_launch(recipe, out)
    (out / 'data/train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Prepared input changed'):
        prepare_launch(recipe, out, resume=True)


def test_full_three_stages_and_idempotent_resume(tmp_path, tiny_config):
    recipe, out = make_recipe(tmp_path, tiny_config), tmp_path / 'run'
    result = launch(recipe, out)
    assert result['status'] == 'complete' and launch(recipe, out, resume=True)['status'] == 'complete'
    for name in result['stages']:
        assert (out / name / 'CURRENT').exists()
        report = json.loads((out / (name + '-evaluation.json')).read_text())
        assert 'token_weighted_nll' in report['summary']['all']


def test_teacher_evaluation_stop_does_not_publish_stage_marker(tmp_path, tiny_config, monkeypatch):
    import schnitz.trajectory_eval as teacher_eval
    recipe, out = make_recipe(tmp_path, tiny_config), tmp_path / 'run'
    calls = []
    def evaluator(run, episodes, **kwargs):
        calls.append(kwargs['output'])
        return {'status': 'checkpointed' if len(calls) == 1 else 'complete'}
    monkeypatch.setattr(teacher_eval, 'evaluate_teacher_run', evaluator)
    assert launch(recipe, out)['status'] == 'checkpointed'
    assert not (out/'text_bootstrap-evaluation.json').exists()
    assert launch(recipe, out, resume=True)['status'] == 'complete'
    assert calls[0] == calls[1] == out/'text_bootstrap-evaluation'


def test_stored_teacher_reads_cannot_call_producer(tmp_path, tiny_config, monkeypatch):
    recipe = make_recipe(tmp_path, tiny_config)
    prepare_launch(recipe, tmp_path / 'prepared')
    es = load_episodes(tmp_path / 'prepared/data/validation.jsonl')[:2]
    agent, store = SchnitzelAgent(tiny_config).eval(), DiskStore(tmp_path / 'bank.sqlite')
    assert build_teacher_bank(agent, store, es)['writer_calls'] > 0
    def forbidden(*args, **kwargs):
        raise AssertionError('Writer invoked on read path')
    monkeypatch.setattr(agent, 'produce', forbidden)
    result = stored_teacher_evaluation(agent, DiskStore(store.path), es)
    assert set(result['summary']) == {'all', 'none', 'zero_values', 'wrong_values'}
    assert result['paired_memory_nll_benefit']['trajectories'] >= 1


def test_explicit_target_limit(tiny_config):
    agent = SchnitzelAgent(tiny_config)
    agent.config.train.max_target_tokens = 5
    with pytest.raises(ValueError, match='Target uses'):
        agent.target_ids('This is a long complete answer')


def test_namespace_scripts_and_documented_docker():
    assert not (ROOT / 'src/elm').exists()
    assert '25.11-py3' in (ROOT / 'docker/Dockerfile.spark').read_text()
    for name in ('spark.sh', 'start_spark.sh', 'install_in_ngc.sh'):
        assert (ROOT / 'scripts' / name).stat().st_mode & 0o100


def test_interrupted_preparation_rejects_changed_recipe(tmp_path, tiny_config, monkeypatch):
    import importlib
    module = importlib.import_module('schnitz.launch')
    recipe = make_recipe(tmp_path, tiny_config)
    with monkeypatch.context() as m:
        def fail(*args, **kwargs):
            raise RuntimeError('interrupted')
        m.setattr(module, '_tokenizer', fail)
        with pytest.raises(RuntimeError, match='interrupted'):
            prepare_launch(recipe, tmp_path / 'run')
    r = yaml.safe_load(recipe.read_text())
    r['seed'] += 1
    recipe.write_text(yaml.safe_dump(r))
    with pytest.raises(ValueError, match='Interrupted preparation'):
        prepare_launch(recipe, tmp_path / 'run', resume=True)


def test_causal_and_binding_evaluations_resume_independently(tmp_path, tiny_config, monkeypatch):
    import schnitz.evaluation as evaluation
    recipe = make_recipe(tmp_path, tiny_config)
    r = yaml.safe_load(recipe.read_text())
    r['evaluation']['causal_worlds'] = 1
    recipe.write_text(yaml.safe_dump(r))
    calls = []
    def evaluate(run, path, **kwargs):
        calls.append(Path(path).name)
        if Path(path).name == 'fresh-multiuse.jsonl' and len(calls) == 2:
            raise RuntimeError('interrupted binding')
        return {'test_only': True, 'rows': []}
    monkeypatch.setattr(evaluation, 'evaluate_transfer_run', evaluate)
    with pytest.raises(RuntimeError, match='interrupted binding'):
        launch(recipe, tmp_path / 'run')
    assert (tmp_path / 'run/causal-evaluation.json').exists()
    assert launch(recipe, tmp_path / 'run', resume=True)['status'] == 'complete'
    assert calls == ['fresh-causal.jsonl', 'fresh-multiuse.jsonl', 'fresh-multiuse.jsonl']


def test_causal_curriculum_actual_cpu_path(tmp_path, tiny_config):
    recipe = make_recipe(tmp_path, tiny_config)
    r = yaml.safe_load(recipe.read_text())
    r.update(protocol='causal', sources=[], causal_train_worlds=4)
    r['evaluation'] = dict(max_episodes=2, causal_worlds=2)
    recipe.write_text(yaml.safe_dump(r))
    assert launch(recipe, tmp_path / 'run')['status'] == 'complete'
    assert (tmp_path / 'run/multiuse-evaluation.json').exists()


def test_learned_causal_launcher_completes(tmp_path, tiny_config):
    tiny_config.train.retrieval = 'learned'
    recipe = make_recipe(tmp_path, tiny_config)
    r = yaml.safe_load(recipe.read_text())
    r.update(protocol='causal', sources=[], causal_train_worlds=4)
    r['stages'] = [dict(name='memory', arm='memory', steps=1)]
    r['evaluation'] = dict(max_episodes=1, causal_worlds=1)
    recipe.write_text(yaml.safe_dump(r))
    assert launch(recipe, tmp_path / 'run')['status'] == 'complete'
    assert (tmp_path / 'run/causal-evaluation.json').exists()


def test_launcher_reports_actual_persistent_compaction(tmp_path, tiny_config):
    recipe = make_recipe(tmp_path, tiny_config)
    r = yaml.safe_load(recipe.read_text())
    r.update(protocol='causal', sources=[], causal_train_worlds=4)
    r['stages'] = [dict(name='memory', arm='memory', steps=1),
                   dict(name='compact', arm='memory', steps=1, init_from='memory',
                        compaction=True, compact_records=1)]
    r['evaluation'] = dict(max_episodes=12, causal_worlds=1)
    recipe.write_text(yaml.safe_dump(r))
    out = tmp_path / 'run'
    assert launch(recipe, out)['status'] == 'complete'
    teacher = json.loads((out / 'compact-evaluation.json').read_text())
    causal = json.loads((out / 'causal-evaluation.json').read_text())
    assert teacher['persistent_codes']['codes'] > 0 and 'compact' in teacher['summary']
    assert causal['persistent_codes']['codes'] > 0 and 'persistent' in causal['summary']
