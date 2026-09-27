# SCHNITZELJAGD on NVIDIA DGX Spark

See [portable operations](operations.md) for named detached containers, W&B,
configurable run/archive mounts and checkpoint retention. These features also work
through the Python CLI on other CPU/CUDA hosts.


> **v0.4 update:** the recommended entry points are `recipes/looped_smoke.yaml`,
> `recipes/looped_starter_muon.yaml` and `recipes/looped_causal.yaml`. They add native
> middle-block recurrence and in-loop reads. See [recurrent conversion](recurrence.md)
> for the four-stage protocol. The one-pass recipes below remain control experiments.

## Chosen runtime

Run natively on Spark's ARM64 host with its supported NVIDIA driver, Docker and NVIDIA
Container Toolkit working. SCHNITZELJAGD does not modify host drivers, install host Python
packages, run privileged containers, or use x86 emulation.

The default base is **`nvcr.io/nvidia/pytorch:25.11-py3`**. NVIDIA's
[Spark Unsloth instructions](https://build.nvidia.com/spark/unsloth/instructions)
explicitly use it. Its [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html)
document the CUDA 13 / PyTorch 2.10 development stack. This is a supported baseline,
not a claim that later NVIDIA releases are unsuitable. The previous unvalidated
26.08 default is replaced; the cited instructions do not require a separate `-igpu`
image for this path.

We use the NVIDIA runtime, not Unsloth's model rewrites. SCHNITZELJAGD needs ordinary
`inputs_embeds`, shared producer/consumer parameters, exact replay and reference
comparisons. An inference-serving image is not the training environment.

## Build, inspect, test

```bash
./scripts/spark.sh build
./scripts/spark.sh run schnitz doctor --require-spark
./scripts/spark.sh run python -m pytest -q
```

The build checks native architecture, pulls `linux/arm64`, verifies the image,
resolves its digest, and records that digest in `.schnitz/base-image.txt` and inside
the image at `/opt/schnitz-base-image.txt`. Dockerfile builds made directly by an editor
use the documented tag; use the script for a digest-recorded scientific run.

The installer creates `/opt/schnitz-venv` with system site packages, then adds pinned
[Transformers 5.17.0](https://pypi.org/project/transformers/5.17.0/) and
[datasets 5.0.1](https://pypi.org/project/datasets/5.0.1/). LFM's
[model card](https://huggingface.co/LiquidAI/LFM2.5-230M) specifies Transformers >=5.0.
Vendor torch/torchvision/torchaudio/triton/NVIDIA versions are first recorded as
explicit constraints. General vendor pip pins are lifted only for the isolated
project install; GPU-runtime constraints stay enforced. The build fails if the
imported torch version, CUDA runtime or torch file path changes. It checks LFM class
imports and records `/opt/schnitz-python-freeze.txt`.

No FlashAttention, bitsandbytes, mamba or custom causal-convolution package is
mandatory. Default attention is SDPA. Parameters and optimizer state are FP32 with
BF16 autocast and BF16 stored payloads. Model preflight checks the actual configured
backbone, not merely that torch can import.

## Start and resume

```bash
./scripts/start_spark.sh --recipe recipes/spark_smoke.yaml --output /runs/spark-smoke
./scripts/start_spark.sh --recipe recipes/starter.yaml --output /runs/starter
./scripts/start_spark.sh --recipe recipes/starter.yaml --output /runs/starter --resume
```

Container output paths should use `/runs/NAME`, which maps to the chosen host
storage. Relative training output paths are rejected to prevent bypassing that
mount and writing into the repository.

Choose an existing run directory with `SCHNITZELJAGD_RUNS_DIR`, or persist it locally:

```bash
mkdir -p .schnitz
printf '%s\n' /path/to/existing/external/schnitz-runs > .schnitz/runs-dir
printf '%s\n' /path/to/existing/external/archive > .schnitz/archive-dir # optional mount
```

These ignored files contain paths only; they are not shell scripts. Environment
variables override them. Configured directories must exist, so a missing disk does
not silently cause a new internal directory to be created. On the current machine,
`.schnitz/runs-dir` selects `/mnt/external/schnitz-archive/runs`. Other machines choose
their own storage. Mounting an archive does not itself enable asynchronous copies;
that remains an explicit training checkpoint policy.

`start_spark.sh` builds only when the local image is absent, checks the device, then
executes `schnitz launch` in the foreground. Rebuild explicitly after changing pinned
Python dependencies or the Dockerfile. Ordinary source edits are bind-mounted.
The model preflight checks ordinary one-pass behavior, zero-gated loop identity,
causal isolation and nonzero finite soft-memory gradients before training.

`doctor --require-spark` requires native ARM64, CUDA, GB10-class capability and a
finite BF16 device matmul; it records package versions and memory counters. It does
not prove the learned model solves tasks. The separate controlled curriculum is:

```bash
./scripts/start_spark.sh --recipe recipes/causal.yaml --output /runs/causal
```

## Resource and access details

The container uses the caller's UID/GID, a mounted checkout at `/workspace/schnitz`,
8 GiB shared memory, unlimited memlock, a 64 MiB stack limit, an init process, and a
600-second graceful stop interval. No service port is opened. A hard kill resumes
from the last committed checkpoint, not a partially written optimizer state.
Only one launcher should use a given output directory.

`SCHNITZELJAGD_CACHE_DIR` overrides the ignored, checkout-local `.schnitz/cache-dir` setting.
With neither configured, cache storage defaults to `~/.cache/schnitz`. It is mounted
at `/cache`, with Hugging Face model/dataset caches under `/cache/huggingface`.
An explicitly configured cache directory must already exist; an unavailable path
fails before Docker launch rather than being created on an unintended disk.
On this machine `.schnitz/cache-dir` now selects `/mnt/external/schnitz-archive/cache`.
Other machines choose their own path; no external-disk location is hardcoded.
Downloads require network and adequate storage capacity. Streaming
bounds selected rows/buffering, not necessarily remote Parquet bytes transferred.
Check actual cache/checkpoint disk usage before scaling sources. Every saved optimizer
checkpoint can be significantly larger than model weights alone.

Spark has [unified CPU/GPU memory](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
Dataset buffers, page cache, parameters, optimizer state and GPU allocations share
physical capacity. Record host RSS/system availability and CUDA allocated/reserved
peaks; do not add host total and CUDA total as independent memory pools.

An existing `HF_TOKEN` is forwarded by environment variable name only; it is not put
in build arguments, Git or commands printed by the script. Default public sources
can be used anonymously when their hosts permit it. Optional gated datasets require
upstream access; no mirrors or access-control workarounds are selected automatically.

```bash
SCHNITZELJAGD_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3 ./scripts/spark.sh build
SCHNITZELJAGD_CACHE_DIR=/path/to/ssd/cache ./scripts/start_spark.sh \
  --recipe recipes/coding.yaml --output /runs/coding
```

`SCHNITZELJAGD_IMAGE` changes the local tag. Another NVIDIA runtime must pass the same checks.
CUDA configurations never silently fall back to CPU. The VS Code devcontainer uses
the same Dockerfile. The manual Spark workflow requires an owner-provided
`[self-hosted, linux, ARM64, spark]` runner and is not run on arbitrary pull requests.

## Validation boundary

The release was developed and tested on x86 CPU. Actual Spark/Docker/LFM execution
and upstream dataset downloads have not been performed in this environment. The
remaining real-machine checks are implemented in the launch command and their
results are written before the main curriculum. See [validation](validation-v0.3.md).


The wrapper also directs XDG, Triton, CUDA driver, TorchInductor, Torch and W&B
artifact caches to subdirectories of `/cache`, and creates `/cache/tmp` before
setting `TMPDIR`. This prevents runtime compilation and temporary files from
silently returning to container-local storage. These are cache-location settings;
they do not replace or patch the NVIDIA software stack.
