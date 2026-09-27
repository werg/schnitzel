import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import pytest
import torch

from schnitz.data import make_multiuse_world

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
try:
    from probe_routing_features import AddressProbe
    spec = importlib.util.spec_from_file_location('global_probe', scripts / 'probe_global_routing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture():
    episodes = make_multiuse_world(0, bindings=2) + make_multiuse_world(1, bindings=2)
    keys = {s.record_id: torch.randn(6) for e in episodes for s in e.supports}
    features = {'key': torch.stack([torch.stack([keys[s.record_id] for s in e.supports]) for e in episodes]),
                'raw_query': torch.randn(len(episodes), 6),
                'required': torch.tensor([([next(i for i,s in enumerate(e.supports) if s.record_id==rid)
                                              for rid in e.required_ids]*2)[:2] for e in episodes]),
                'lengths': torch.tensor([len(e.required_ids) for e in episodes])}
    return features, episodes


def test_global_feature_identity_and_label_mapping():
    features, episodes = fixture()
    data, ids = module.global_features(features, episodes)
    assert len(ids) == 8
    for i,e in enumerate(episodes):
        assert [ids[j] for j in data['required'][i][:len(e.required_ids)]] == list(e.required_ids)
        for local, global_index in enumerate(data['support_indices'][i]):
            torch.testing.assert_close(data['key'][global_index], features['key'][i,local], rtol=0, atol=0)


@pytest.mark.parametrize('corruption', ['future', 'feature', 'label'])
def test_global_feature_rejects_causal_or_identity_drift(corruption):
    features, episodes = fixture()
    if corruption == 'future':
        episodes[0] = replace(episodes[0], query_time=1)
    elif corruption == 'feature':
        features['key'][1,0] += 1
    else:
        features['required'][0] = (features['required'][0] + 1) % 4
    with pytest.raises(ValueError):
        module.global_features(features, episodes)


def test_world_mask_preserves_pair_loss_gradients_and_global_scope_adds_competitors():
    from types import SimpleNamespace
    from torch import nn
    features, episodes = fixture()
    data, _ = module.global_features(features, episodes)
    agent = SimpleNamespace(key_head=nn.Linear(6,3,bias=False), address_maps=[nn.Linear(3,3,bias=False)],
                            query_maps=[nn.Linear(3,3,bias=False)], query_head=nn.Linear(6,3,bias=False))
    model = module.StopProbe(AddressProbe(agent, True).state_dict(), False)
    indices = torch.arange(len(episodes))
    scores = module.address_scores(model, data, indices, 'world')
    assert torch.isfinite(scores).sum(-1).tolist() == [4]*len(episodes)
    assert torch.isfinite(module.address_scores(model, data, indices, 'global')).all()
    local, _ = model(features)
    torch.testing.assert_close(scores.gather(1,data['support_indices']),local,rtol=1e-5,atol=1e-5)
    loss = module.pair_loss(scores,data['required'],data['lengths'])
    expected = module.pair_loss(local,features['required'],features['lengths'])
    grads = torch.autograd.grad(loss,tuple(model.parameters()))
    reference = torch.autograd.grad(expected,tuple(model.parameters()))
    for a,b in zip(grads,reference):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    # Even a whole-bank feature comparison excludes sources at or after query time.
    data['created_at'][0] = data['query_time'].max()
    assert torch.isneginf(module.address_scores(model,data,indices,'global')[:,0]).all()


def test_bf16_proxy_uses_serialized_key_normalization_and_float32_search_scores():
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    features, episodes = fixture()
    data, _ = module.global_features(features, episodes)
    agent = SimpleNamespace(key_head=nn.Linear(6,3,bias=False), address_maps=[nn.Linear(3,3,bias=False)],
                            query_maps=[nn.Linear(3,3,bias=False)], query_head=nn.Linear(6,3,bias=False))
    model = module.StopProbe(AddressProbe(agent, True).state_dict(), False)
    indices = torch.arange(len(episodes))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        # Match produce -> stored_channel: final normalized key is BF16 before FP32 storage.
        serialized = F.normalize(model.address(F.normalize(model.key(data['key']),dim=-1)),dim=-1).float()
        query = model.query_map(F.normalize(model.query_head(data['raw_query']),dim=-1)).float()
        actual = module.address_scores(model,data,indices,'global')
    expected = F.normalize(query,dim=-1) @ F.normalize(serialized,dim=-1).T / .1
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
