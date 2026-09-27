"""Small hardware, storage and compaction checks, not benchmark claims."""
from __future__ import annotations

import json
import platform
from pathlib import Path
import time

import torch

from .compaction import SyntheticCompactor, contribution_loss, mean_and_mass
from .readers import SetReader
from .store import DiskStore, StoredRecord
from .training import environment_report, resource_report


def doctor(require_spark: bool = False) -> dict:
    report = environment_report()
    from importlib.metadata import PackageNotFoundError, version
    report['packages'] = {}
    for name in ('schnitz', 'transformers', 'datasets', 'safetensors'):
        try:
            report['packages'][name] = version(name)
        except PackageNotFoundError:
            report['packages'][name] = 'not installed'
    image = Path('/opt/schnitz-base-image.txt')
    report['container_base_image'] = image.read_text().strip() if image.exists() else None
    report['torch_import_path'] = torch.__file__
    if torch.cuda.is_available():
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = a @ a.T
        torch.cuda.synchronize()
        report["bf16_matmul_finite"] = bool(torch.isfinite(b).all())
        del a, b
    report.update(resource_report())
    report["unified_memory_note"] = (
        "On DGX Spark, CPU allocations, GPU allocations, and host caches share physical memory. "
        "Do not add reported CUDA total and host RAM as independent capacity."
    )
    if require_spark:
        problems = []
        if platform.machine().lower() not in {"aarch64", "arm64"}:
            problems.append("native ARM64 process required")
        if not torch.cuda.is_available():
            problems.append("CUDA unavailable")
        elif torch.cuda.get_device_capability(0)[0] < 12:
            problems.append("GB10-class CUDA capability expected")
        if not report.get("bf16_matmul_finite", False):
            problems.append("BF16 device computation did not pass")
        report["spark_checks"] = "passed" if not problems else problems
        if problems:
            raise RuntimeError(json.dumps(report, indent=2))
    return report


def io_benchmark(path: str | Path, *, records: int = 1000, payload_dim: int = 256,
                 key_dim: int = 64, reads: int = 20, neighbors: int = 16,
                 cache_mib: int = 32) -> dict:
    """Exact key scan and payload timing; leaves OS page cache untouched."""
    if min(records, payload_dim, key_dim, reads, neighbors, cache_mib) < 1:
        raise ValueError("Benchmark sizes must be positive")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Use a fresh database path to avoid mixing benchmark runs")
    torch.manual_seed(123)
    torch.set_num_threads(2)
    store = DiskStore(path, cache_mib=cache_mib)
    start = time.perf_counter()
    for i in range(records):
        store.put(StoredRecord(f"item-{i:09d}", torch.randn(key_dim),
                               torch.randn(payload_dim).bfloat16()))
    write_seconds = time.perf_counter() - start
    searches, fetches = [], []
    for _ in range(reads):
        q = torch.randn(key_dim)
        start = time.perf_counter()
        plan = store.search(q, top_k=neighbors)
        searches.append(time.perf_counter() - start)
        start = time.perf_counter()
        store.fetch(plan)
        fetches.append(time.perf_counter() - start)

    def quantiles(samples):
        x = torch.tensor(samples, dtype=torch.float64)
        return {"p50_ms": float(x.quantile(.5) * 1000),
                "p95_ms": float(x.quantile(.95) * 1000)}

    return {"environment": environment_report(), "records": records,
            "payload_dim": payload_dim, "payload_dtype": "bfloat16",
            "key_dim": key_dim, "neighbors": neighbors, "reads": reads,
            "sqlite_cache_mib_per_connection": cache_mib,
            "write_seconds": write_seconds, "search": quantiles(searches),
            "fetch": quantiles(fetches), "storage": store.sizes(),
            "ideal_value_bytes_per_read": min(neighbors, records) * payload_dim * 2,
            "notice": "Exact CPU key scan; freshly populated/warm OS cache, NOT a cold-NVMe or ANN result.",
            "resources": resource_report()}


def compact_probe(*, steps: int = 100, seed: int = 7, reader_kind: str = "mlp") -> dict:
    """Fit an amortized compactor against a fixed random reader; held-out tensors.

    Validates an optimization path, NOT semantic compactability of a trained writer.
    Clusters contain two repeated/perturbed prototypes (or independent records).
    """
    if steps < 1 or reader_kind not in {"mlp", "attention"}:
        raise ValueError("Invalid probe settings")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    dim, qdim, n = 12, 8, 8
    reader = SetReader(dim, qdim, 16, width=24, slots=2, rounds=2, kind=reader_kind,
                       chunk_size=16, checkpoint_chunks=False)
    for p in reader.parameters():
        p.requires_grad_(False)
    reader.eval()
    compactor = SyntheticCompactor(dim, width=48, records=2)
    optimizer = torch.optim.AdamW(compactor.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed + 1)

    def sample(batch: int, mode: str):
        if mode == "redundant":
            centers = torch.randn(batch, 2, dim, generator=generator)
            x = centers.repeat_interleave(n // 2, dim=1)
            x = x + .03 * torch.randn(batch, n, dim, generator=generator)
        else:
            x = torch.randn(batch, n, dim, generator=generator)
        return x, torch.ones(batch, n), torch.randn(batch, qdim, generator=generator)

    heldout = {mode: sample(32, mode) for mode in ("redundant", "independent")}

    @torch.no_grad()
    def measure():
        result = {}
        for mode, (x, w, q) in heldout.items():
            mean, mass = mean_and_mass(x, w)
            code, multiplicity = compactor(x, w)
            result[mode] = {
                "mean_contribution_loss": float(contribution_loss(reader, x, w, mean, mass, q)),
                "synthetic_contribution_loss": float(contribution_loss(reader, x, w, code, multiplicity, q)),
            }
        return result

    before = measure()
    first = last = None
    for _ in range(steps):
        x, w, q = sample(16, "redundant")
        code, multiplicity = compactor(x, w)
        loss = contribution_loss(reader, x, w, code, multiplicity, q)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    return {"seed": seed, "steps": steps, "reader": reader_kind,
            "original_records": n, "synthetic_records": 2, "before": before, "after": measure(),
            "first_training_loss": first, "last_training_loss": last,
            "notice": "Random fixed reader/tensor probe only; no language-task quality or capacity claim."}
