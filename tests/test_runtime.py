from types import SimpleNamespace

import pytest

from schnitz import runtime


def test_unified_memory_uses_available_host_memory(monkeypatch):
    monkeypatch.setattr(runtime, 'available_host_memory', lambda: 100)
    config = SimpleNamespace(device='cpu', cuda_memory_fraction=None, min_system_available_bytes=101)
    with pytest.raises(MemoryError, match='MemAvailable'):
        runtime.configure_memory(config)
    config.min_system_available_bytes = 50
    assert runtime.configure_memory(config)['host_available_bytes'] == 100


def test_watchdog_is_disarmed_on_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.faulthandler, 'dump_traceback_later', lambda *a, **kw: calls.append('armed'))
    monkeypatch.setattr(runtime.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancelled'))
    with pytest.raises(ValueError):
        with runtime.compute_watchdog(60):
            raise ValueError('compute failed')
    assert calls == ['armed', 'cancelled']


def test_watchdog_covers_cuda_completion_before_disarming(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.faulthandler, 'dump_traceback_later', lambda *a, **kw: calls.append('armed'))
    monkeypatch.setattr(runtime.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancelled'))
    monkeypatch.setattr(runtime.torch.cuda, 'synchronize', lambda device: calls.append(('completed', device)))
    with runtime.compute_watchdog(60, device='cuda'):
        calls.append('queued')
    assert calls == ['armed', 'queued', ('completed', 'cuda'), 'cancelled']


@pytest.mark.parametrize('seconds,device', [(0, 'cuda'), (60, 'cpu')])
def test_watchdog_does_not_add_unrequested_device_wait(monkeypatch, seconds, device):
    monkeypatch.setattr(runtime.torch.cuda, 'synchronize', lambda *_: pytest.fail('Unexpected CUDA wait'))
    with runtime.compute_watchdog(seconds, device=device):
        pass


def test_cpu_resource_reset_never_initializes_cuda(monkeypatch):
    from schnitz import training
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training.torch.cuda, "synchronize",
                        lambda: pytest.fail("CPU training synchronized CUDA"))
    monkeypatch.setattr(training.torch.cuda, "reset_peak_memory_stats",
                        lambda: pytest.fail("CPU training initialized CUDA metrics"))
    training.reset_resource_peaks("cpu")


def test_cuda_cache_reclaimed_only_above_allowance(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.torch.cuda, 'memory_allocated', lambda _device: 2)
    monkeypatch.setattr(runtime.torch.cuda, 'memory_reserved', lambda _device: 12 - 6 * len(calls))
    monkeypatch.setattr(runtime.torch.cuda, 'empty_cache', lambda: calls.append('reclaimed'))
    metrics = runtime.reclaim_cuda_cache('cuda', 4)
    assert calls == ['reclaimed']
    assert metrics == {
        'cuda_allocated_bytes': 2,
        'cuda_reserved_before_reclaim_bytes': 12,
        'cuda_reserved_bytes': 6,
        'cuda_cache_reclaimed_bytes': 6,
        'cuda_cache_host_pressure': False,
    }


def test_cuda_cache_below_allowance_is_retained(monkeypatch):
    monkeypatch.setattr(runtime.torch.cuda, 'memory_allocated', lambda _device: 8)
    monkeypatch.setattr(runtime.torch.cuda, 'memory_reserved', lambda _device: 10)
    monkeypatch.setattr(runtime.torch.cuda, 'empty_cache',
                        lambda: pytest.fail('cache below allowance was reclaimed'))
    assert runtime.reclaim_cuda_cache('cuda', 4)['cuda_reserved_bytes'] == 10


def test_cuda_cache_reclaimed_under_host_pressure(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.torch.cuda, 'memory_allocated', lambda _device: 4)
    monkeypatch.setattr(runtime.torch.cuda, 'memory_reserved',
                        lambda _device: 10 - 6 * len(calls))
    monkeypatch.setattr(runtime.torch.cuda, 'empty_cache', lambda: calls.append('reclaimed'))
    monkeypatch.setattr(runtime, 'available_host_memory', lambda: 7)
    metrics = runtime.reclaim_cuda_cache(
        'cuda', 100, min_host_available_bytes=8)
    assert calls == ['reclaimed']
    assert metrics['cuda_cache_host_pressure'] is True
