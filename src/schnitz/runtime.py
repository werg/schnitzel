"""Portable opt-in resource rails and stack diagnostics for stalled compute."""
from contextlib import contextmanager
from pathlib import Path
import faulthandler

import torch


def available_host_memory():
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def configure_memory(config):
    available = available_host_memory()
    if available is not None and available < config.min_system_available_bytes:
        raise MemoryError('Host MemAvailable is below the configured reserve; '
                          'on unified-memory systems this includes CUDA and filesystem-cache pressure')
    if config.cuda_memory_fraction is not None and config.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(config.cuda_memory_fraction)
    return dict(host_available_bytes=available, cuda_memory_fraction=config.cuda_memory_fraction,
                min_system_available_bytes=config.min_system_available_bytes)


def memory_metrics(device):
    """Allocator counters without device synchronization; never sum host and CUDA RAM."""
    result = {}
    available = available_host_memory()
    if available is not None:
        result['host_available_bytes'] = available
    if torch.device(device).type == 'cuda':
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_attempt_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


def reclaim_cuda_cache(device, max_unused_bytes: int, *, force: bool = False,
                       min_host_available_bytes: int = 0):
    """Return unused allocator blocks after a completed optimizer boundary.

    The caller must not use this inside a forward/backward graph. On unified
    memory, cached CUDA blocks consume the same physical pool as checkpoint
    staging and CPU work even though no live tensor owns them.
    """
    if max_unused_bytes < 0 or min_host_available_bytes < 0:
        raise ValueError('CUDA cache allowance and host reserve must be nonnegative')
    if torch.device(device).type != 'cuda':
        return {}
    allocated = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    host_available = available_host_memory()
    pressure = (host_available is not None
                and host_available < min_host_available_bytes)
    if force or pressure or reserved_before - allocated > max_unused_bytes:
        torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved(device)
    return {
        'cuda_allocated_bytes': allocated,
        'cuda_reserved_before_reclaim_bytes': reserved_before,
        'cuda_reserved_bytes': reserved_after,
        'cuda_cache_reclaimed_bytes': max(0, reserved_before - reserved_after),
        'cuda_cache_host_pressure': pressure,
    }


@contextmanager
def compute_watchdog(seconds, *, device=None):
    """Dump stacks on a stall; never hard-kill a process with unsaved work.

    Armed around the declared compute region, excluding setup and saves.
    With a CUDA device, completion is awaited before disarming an enabled guard.
    Cooperative stopping still needs the current native kernel to return.
    """
    if seconds:
        faulthandler.dump_traceback_later(seconds, repeat=True)
    try:
        yield
        if seconds and device is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        if seconds:
            faulthandler.cancel_dump_traceback_later()
