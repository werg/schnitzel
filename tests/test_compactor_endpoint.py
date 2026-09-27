import importlib.util
from pathlib import Path

import pytest
import torch


def test_compact_confirmation_requires_complete_source_bound_equal_budget_state(tmp_path, tiny_config, monkeypatch):
    from schnitz.agent import SchnitzelAgent
    from schnitz.cluster_store import state_fingerprint
    from schnitz.compaction import SyntheticCompactor
    path = Path(__file__).parents[1] / 'scripts/evaluate_compact_transfer.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('compact_confirmation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = SchnitzelAgent(tiny_config).eval().requires_grad_(False)
    compactor = SyntheticCompactor(24, 24, 1)
    reader_hash = state_fingerprint(agent.reader)
    state = {'step': 2, 'identity': {'steps': 2, 'checkpoint_manifest_sha256': 'source',
             'reader_hash': reader_hash, 'compactor': {'width': 24, 'records': 1}},
             'compactor': compactor.state_dict()}
    endpoint = tmp_path / 'resume.pt'
    torch.save(state, endpoint)
    loaded, _ = module.load_compactor(agent, endpoint, 'source', reader_hash)
    assert state_fingerprint(loaded) == state_fingerprint(compactor)
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match='source/reader/code budget'):
        module.load_compactor(agent, endpoint, 'other', reader_hash)
    with pytest.raises(ValueError, match='source/reader/code budget'):
        module.load_compactor(agent, endpoint, 'source', 'other')
    state['step'] = 1
    torch.save(state, endpoint)
    with pytest.raises(ValueError, match='incomplete'):
        module.load_compactor(agent, endpoint, 'source', reader_hash)
    state['step'] = 2
    state['identity']['compactor']['records'] = 2
    torch.save(state, endpoint)
    with pytest.raises(ValueError, match='code budget'):
        module.load_compactor(agent, endpoint, 'source', reader_hash)
