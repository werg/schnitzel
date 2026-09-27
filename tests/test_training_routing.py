import math
import pytest
import torch

from schnitz.agent import SchnitzelAgent
from schnitz.config import load_config
from schnitz.data import make_episode
from schnitz.routing import group_plan_loss, utility_ranking_loss, complete_support_recall
from schnitz.store import DiskStore
from schnitz.training import train, build_evaluation_store, stored_evaluation


def test_group_starters_get_nonzero_gradient():
    scores = torch.zeros(4, requires_grad=True)
    loss = group_plan_loss(scores, [(0, 1)])
    assert abs(loss.item() - math.log(6)) < 1e-6
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[1] < 0
    assert scores.grad[2] > 0


def test_group_probabilities_include_all_valid_orders():
    scores = torch.zeros(3)
    assert group_plan_loss(scores, [(0, 1), (0, 2), (1, 2)]).abs() < 1e-6


def test_flat_utility_does_not_penalize_starters():
    scores = torch.randn(3, requires_grad=True)
    utility_ranking_loss(scores, torch.zeros(3)).backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_any_sufficient_group_counts():
    assert complete_support_recall({"a", "b"}, [{"a", "b"}, {"c"}]) == 1
    assert complete_support_recall({"a"}, [{"a", "b"}, {"c"}]) == 0


def test_train_smoke_save_resume(tmp_path, tiny_config):
    path = tmp_path / "run"
    result = train(tiny_config, path)
    assert result["steps"] == 2
    assert (path / "model.safetensors").exists()
    tiny_config.train.steps = 3
    resumed = train(tiny_config, path, resume=True)
    assert resumed["last"]["step"] == 3


def test_native_training_executes_two_examples_as_one_batch(tmp_path, tiny_config):
    tiny_config.model.recurrence_mode = 'middle_block'
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.model.loops = 2
    tiny_config.model.writer_loops = 1
    tiny_config.memory.read_timing = 'loop_boundary'
    tiny_config.memory.compaction = 'none'
    tiny_config.train.retrieval = 'learned'
    tiny_config.train.routing_warmup = 2
    tiny_config.train.live_fraction = 1.0
    tiny_config.train.gradient_accumulation = 1
    tiny_config.train.batch_size = 2
    tiny_config.train.sampling_policy = 'shuffled_passes'
    tiny_config.train.steps = 1
    result = train(tiny_config, tmp_path / 'batched')
    assert result['steps'] == 1
    assert result['last']['sampling']['position_in_pass'] == 0


def test_stored_only_eval_never_calls_writer(tmp_path, tiny_config, monkeypatch):
    agent = SchnitzelAgent(tiny_config).eval()
    episodes = [make_episode(0, split="unseen", distractors=1)]
    store = DiskStore(tmp_path / "evaluation.sqlite")
    build_evaluation_store(agent, store, episodes, "frozen")
    def forbidden(*args, **kwargs):
        raise AssertionError("Writer called on normal inference path")
    monkeypatch.setattr(agent, "produce", forbidden)
    result = stored_evaluation(agent, DiskStore(store.path), episodes, "frozen")
    assert len(result["rows"]) == 8
    assert result["summary"]["all"]["complete_support_recall"] == 1
    assert result["summary"]["A"]["complete_support_recall"] == 0


def test_unknown_config_fields_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  typo: 123\n")
    with pytest.raises(TypeError):
        load_config(path)


def test_general_support_query_training_and_stored_eval(tiny_config, tmp_path):
    import json
    from schnitz.training import train, evaluate_episode_file
    from schnitz.data import load_episodes
    row = {"episode_id": "new-task", "query_time": 10,
           "supports": [{"record_id": "experience-a", "text": "The API takes a snapshot before retry.", "created_at": 1}],
           "query": "How do I retry safely?", "answer": "Take a snapshot first.",
           "required_ids": ["experience-a"]}
    dataset = tmp_path / 'tasks.jsonl'
    dataset.write_text(json.dumps(row) + '\n')
    assert load_episodes(dataset)[0].answer == row['answer']
    tiny_config.train.episodes_file = str(dataset)
    tiny_config.train.steps = 1
    run = tmp_path / 'run'
    train(tiny_config, run)
    result = evaluate_episode_file(run, dataset)
    assert len(result['rows']) == 3
    # Numerical integration test only: production evaluations must use held-out tasks.
