# Portable training operations

The Python runner supports CPU and CUDA PyTorch environments without Docker,
Spark, W&B or an external disk. Spark's ARM64 image and driver checks remain in
`scripts/spark.sh`. Do not install a replacement Torch into a working vendor
runtime merely to use the optional logging extra.

## Start, stop, inspect and resume

Foreground execution works with a terminal, a scheduler, or a container:

```bash
schnitz launch --recipe recipes/looped_causal.yaml --output /fast/schnitz-runs/causal
# Resume the identical recipe and output:
schnitz launch --recipe recipes/looped_causal.yaml --output /fast/schnitz-runs/causal --resume
```

For a local detached process, use the same Python environment that will train:

```bash
schnitz runs start --recipe recipes/looped_causal.yaml --output /fast/schnitz-runs/causal
schnitz runs status --output /fast/schnitz-runs/causal
schnitz runs stop --output /fast/schnitz-runs/causal
schnitz runs start --recipe recipes/looped_causal.yaml --output /fast/schnitz-runs/causal --resume
```

Status, cooperative stop, archive/restore and checkpoint relocation inspect metadata
and copy opaque files; they do not import Torch or Safetensors. From a source
checkout, a host with Python and PyYAML can inspect container runs through their
shared host paths:

```bash
PYTHONPATH=src python3 -m schnitz runs status --output /mnt/external/schnitz-archive/runs/STUDY/STAGE
PYTHONPATH=src python3 -m schnitz runs stop --output /mnt/external/schnitz-archive/runs/STUDY/STAGE
```

Training and model save/load still require the native ML environment. For jobs
launched with `docker exec` into a persistent container, request each job's
cooperative stop and wait for its checkpoint commit and ownership lock release
before stopping the container. The foreground container launcher below owns its
training process directly.

Status gives a persistent console-log path, resolved checkpoint locations and free space.
`checkpoint_step` is the committed recovery position; `latest_logged_step` reports
the newest complete training metric in a bounded log tail. With sparse saves these
can differ substantially. A logged update is not a durable checkpoint. Interrupted
last log lines are ignored; missing recent metrics report null. `step` remains an
alias for the checkpoint position for existing callers.

Storage status includes the resolved path, filesystem device ID and whether it
shares the code filesystem. `checkpoint_directory_is_symlink` describes relocation;
the older `externalized` alias has the same meaning. Direct output on an external
disk can have a regular checkpoint directory, so the symlink flag alone does not
identify its physical storage medium. Device IDs are local operating-system values.
Optional archives expose active/pending copies, last completion and errors in
`archive-status.json`. A stop request is a control file,
not a signal sent to a potentially recycled PID. Training finishes the current microbatch and its complete producer replay, then
commits model, optimizer, RNG and cache together. If accumulation is incomplete,
the checkpoint also includes accumulated gradients, microbatch position, sampled
depth and running loss totals; resume finishes that same optimizer update. Preparation
and preflight stop at a safe boundary. The real-trajectory teacher evaluator checks
between committed offline-bank namespaces and scored conditions, preserving its
original read plans; it never publishes a partial namespace or stage marker.
SIGINT/SIGTERM also request a graceful stop. A hard kill resumes the
last committed checkpoint. Detached runs do not automatically restart failures.

OS file locks reject concurrent writers to a launch or training directory and
release on process death. Control files live in `.sdkb-control` beside the run;
their presence alone does not imply a live process. A new explicit invocation
acknowledges an earlier stop request. Use one canonical run path through shared
mounts. These controls have been exercised on Linux ARM64; other platforms need
their own runtime validation.

Target-curriculum supervisors acknowledge the earlier stop immediately before they
spawn a stage. The stage itself preserves any stop written during model, index, or
checkpoint initialization, so a startup-time request cannot be cleared when the run
lock is acquired later.

Inside Docker, run `schnitz launch` in the foreground and let Docker own process
lifetime; do not detach a child inside an ephemeral `docker run --rm` container.
On Spark:

```bash
export SCHNITZELJAGD_RUNS_DIR=/mnt/external/schnitz-runs       # existing mounted external storage
export SCHNITZELJAGD_ARCHIVE_DIR=/mnt/external/schnitz-archive # optional separate archive directory
export SCHNITZELJAGD_CONTAINER=schnitz-causal
./scripts/spark.sh build
./scripts/spark.sh start --recipe recipes/looped_causal.yaml --output /runs/causal
./scripts/spark.sh status
./scripts/spark.sh logs
./scripts/spark.sh stop
```

The detached container is retained for inspection. After it exits, remove that
specific stopped container with `docker rm "$SCHNITZELJAGD_CONTAINER"`, then repeat `start`
with `--resume`. Logs can be detached without stopping training. Docker stop allows
600 seconds by default (`SCHNITZELJAGD_STOP_TIMEOUT` overrides this); an expired grace
period becomes a hard kill. Existing unrelated GPU services are never stopped.

## Checkpoint frequency and storage placement

Choose checkpoint placement explicitly with `--output`. On this machine retained
checkpoints belong on the external disk. New runs can live entirely there, with
`archive_dir: null` to avoid a redundant second copy. Existing stopped runs can
move their checkpoint directories without changing dataset or result paths:

```bash
mkdir -p /mnt/external/schnitz-checkpoints/my-stage
schnitz relocate-checkpoints --run /fast/run/my-stage \
  --destination /mnt/external/schnitz-checkpoints/my-stage
schnitz runs configure --output /fast/run --checkpoint-every 1000 --no-archive
```

Relocation verifies every retained checkpoint, publishes the external directory,
replaces the old checkpoint directory with a link, then removes redundant local
files. A corrupt destination or a running stage fails closed. Optional
`--archive-hint PATH/TO/checkpoints` reuses verified existing archive files with
hardlinks when possible. Source/destination paths must be visible in the same
location inside and outside containers; the Spark wrapper exposes the archive at
both `/archive` and its host path. Run parents must also be stopped during a
manual stage relocation. New direct-external runs do not need symlinks.

Periodic saves default to **1,000 optimizer updates**, plus initial, final and
emergency saves. `runs configure` changes only checkpoint operations and records
`checkpoint-policy.json`; it does not rewrite immutable scientific inputs. Short
200–400-update stages therefore normally write initial/final checkpoints only.
Stop and final checkpoints remain mandatory regardless of periodic cadence.

Console logs mark `checkpoint_start` and `checkpoint_committed`, with elapsed
seconds and serialized bytes. These separate storage waits from compute stalls.
If a stop arrives during an initial or periodic save, the trainer reuses that just
committed state instead of writing it a second time. This reuse is limited to the
current invocation; a stopped resume still commits allowed configuration changes
such as an extended total step budget.
Direct external saves can take minutes under disk contention, including graceful
stop saves. The asynchronous archive mode below provides a faster local save path
when explicitly configured; direct external output does not silently stage an
emergency copy on NVMe. Wait for the commit event or process exit before removing
the disk or container.

A fast local staging directory and asynchronous archive remain optional when
external-write latency is unacceptable. They require deliberate retirement of
completed runs: keeping two local checkpoints **per stage forever** is not a
space-management policy. The external-first setup avoids that accumulation.

Copy a base config/recipe for the intended run and set these fields under `train`:

```yaml
archive_dir: /archive             # Docker mount; use the actual path outside Docker
checkpoint_every: 1000           # emergency and final saves are independent
keep_checkpoints: 2               # recent complete sets on the output filesystem
archive_keep_checkpoints: 3       # recent archived sets per unique run identity
min_free_disk_bytes: 10737418240  # optional 10 GiB reserve; default is 1 GiB
```

`archive_dir` must already exist on the intended mounted disk. Leaving it null
disables archival. Each stage has a persistent UUID and a separate archive
subdirectory. Archives cannot be mixed across run identities. W&B shares that
identity, and it is included in checkpoint recovery data.

Copies run in a background thread with 8 MiB buffers and fsync every 64 MiB,
bounding dirty-page pressure. Every immutable checkpoint file is checked against
its recorded SHA256 before atomic publication of the archive's `CURRENT` pointer.
An incomplete copy is never a recovery checkpoint. Where supported, archive pages
are advised out of the OS cache after flush; this is not a cold-cache benchmark.

There is at most one active and one pending copy. If storage falls behind, the
pending copy advances to the newest checkpoint: **the archive is a bounded recovery
history, not a promise to retain every training step**. Local pruning protects
active/pending copies in addition to the configured recent sets, so transient local
retention can exceed `keep_checkpoints` by up to two. Completed runs drain the
archive; graceful stops prioritize the local checkpoint and do not wait for the
external disk. Resume queues the current committed checkpoint again.

Before saves, free-space checks budget for weights, Adam state, the SQLite snapshot,
and the reserve. Archive errors surface as errors while leaving the local recovery
checkpoint intact. Reserve checks reduce disk-exhaustion risk; they cannot reserve
space against other processes writing to the same disk.

```bash
schnitz storage --path /fast/schnitz-runs/causal
schnitz storage --path /mnt/external/schnitz-archive
# Manual copy of a stopped stage into an existing dedicated archive directory:
schnitz archive --run /fast/schnitz-runs/causal/recurrent_joint --destination /archive/manual-stage
# Restore into a NEW directory, then resume using its committed config:
schnitz restore --archive /archive/RUN_UUID --output /fast/recovered-stage
schnitz train --output /fast/recovered-stage --resume
```

Archives hold checkpoint sets, including the stale training cache and run identity.
They are not full backups of downloaded datasets, prepared launch inputs, console
logs or W&B history. Preserve the prepared launch directory separately; exact
resume still requires the unchanged dataset at the path recorded in the config.
The restore command never overwrites an existing directory. For a new experiment
from an old checkpoint, use warm-start (`--init-from`) and a new run identity.

Storage inspection reports bytes and abandoned `.pending-*` directories; it does
not delete shared Hugging Face caches, other runs, Docker images, or archives.
Inspect interrupted copies before manually removing them. No whole-disk cleanup
command runs automatically.

## Optional W&B

Install `schnitz[tracking]` into the project environment; the Spark installer includes
it while constraining the NVIDIA runtime. Logging is disabled by default. Configure:

```yaml
wandb_mode: offline # disabled / offline / online
wandb_project: schnitz
wandb_entity: null
wandb_group: causal-mlp-seed17
```

Online mode uses credentials from the environment/login, never YAML. The Spark
wrapper forwards `WANDB_API_KEY` by name and optionally `WANDB_BASE_URL`. No upload
was required to validate the offline path. Source text, model weights and latent
payloads are not logged; automatic console and Git capture are disabled. Config
and scalar optimizer metrics are logged, including the actual optimizer step.

Online resumes use the same explicit ID with `resume="allow"`, following the
[W&B resume contract](https://docs.wandb.ai/models/runs/resuming). Offline restarts
produce separate local segments with the same identity; the SDK does not merge
offline history automatically. After a crash, W&B can contain uncheckpointed
metrics, so each attempt is tagged and local checkpoint-consistent JSONL remains
authoritative. W&B is telemetry, never a source for optimizer recovery.

A raised runtime `log` error disables W&B logging for that attempt while JSONL,
training and checkpoints continue. Error types/steps are recorded under
`tracking/failure-ATTEMPT.json` and announced in the local console; private SDK
exception text is not copied into that record. A raised `finish` error cannot mask
an original training failure or invalidate a completed checkpoint. A new explicit
attempt retries tracking with the same run identity. Requested tracking setup
errors still fail before training. This handles raised errors; it does not add a
timeout to an SDK call that never returns.

## Lessons applied from bgkit

Bgkit's runbook, trainer and checkpoint archiver document failures from writing
large checkpoints synchronously to slow external storage, treating an archive's
existence as proof of completeness, mixing retention across runs, and losing run
identity during resume. SCHNITZELJAGD adopts bounded flushing, verified atomic publication,
independent retention, durable identities and graceful stopping. It keeps the
implementation independent of bgkit's model code, Hydra configuration, absolute
mount paths and Spark-only optimizations.

## Optimizers and exact resume

`train.optimizer: muon` uses native `torch.optim.Muon` on eligible 2D matrix
transforms, with AdamW on embedding/output tables, learned slots, vectors,
non-matrix tensors and LoRA factors. It requires a runtime with native Muon;
older Torch runtimes fail clearly without installing a replacement. The saved
configuration controls momentum, Newton–Schulz steps, weight decay, Adam betas
and epsilon. Muon uses `adjust_lr_fn: match_rms_adamw`; learning rates still need
experimental validation and are not assumed equivalent to AdamW.

Every trainable parameter must have exactly one optimizer owner. Checkpoints save
both optimizers, actual per-group learning rates and settings, momentum/moments,
parameter names/order and all accumulation state. `optimizer.json` reports the
settings actually loaded after resume. Changed training hyperparameters are
rejected on exact resume; an optimizer switch is an explicit warm-start into a
new run. `schnitz train --output RUN --resume` loads the saved config automatically.
There is no LR scheduler or early-stopping controller in SCHNITZELJAGD yet; no scheduler
state is implied. W&B identity and random episode-sampling position also survive.

Resume into the mutable run directory, never a `checkpoints/step-*` directory.
Training rejects an immutable checkpoint as its output before creating run state.
Use `schnitz restore` to recover an archive into a new mutable run directory. Direct
checkpoint paths remain valid for evaluation, warm-starts and read-only state loading;
the latter still verifies the dataset and base-model revision.

A 20 September ownership correction moves reader `null_tokens` fallback tables
from Muon to AdamW, consistent with the embedding/slot rule. This changes named
group membership for future runs. Older Muon checkpoints still resume exactly
from their original frozen checkout; current code rejects their former group
ordering. Use an explicit weight warm-start for the corrected policy. Do not
reinterpret old momentum as AdamW state or alter an active run's optimizer.

Current resume requires a manifest, explicit optimizer type and named ownership,
global Python RNG, episode-sampler RNG and Torch/CUDA RNG state. Older states
missing these fields are rejected instead of substituting another RNG or guessing
momentum ownership. Resume those with their original checkout and mutable run
directory, or use an explicit warm-start with fresh optimizer state. Current
complete checkpoints and partial-accumulation recovery retain their exact semantics.

## Runtime rails and profiling

Logged training updates include host `MemAvailable` where supported and CUDA
allocated/reserved bytes plus the current attempt's allocation peak. These appear
under `memory/` in W&B and `memory` in JSONL. Host and CUDA readings must not be
added on unified-memory machines. CUDA allocator counters do not measure every
driver or library allocation. Exact resume restores training state, not historical
allocator peaks; the peak starts over for each attempt.

The sustained spatial supervisor defaults to PyTorch expandable allocator segments,
an 0.8 garbage-collection threshold, and a 512 MiB maximum split size. This follows
the working Spark policy in `bgkit` and bounds cached-segment growth from variable
trajectory and writer shapes. A caller-supplied `PYTORCH_ALLOC_CONF` or legacy
`PYTORCH_CUDA_ALLOC_CONF` takes precedence, and the effective setting and allocator
backend are recorded in the run environment. `nvidia-smi` process attribution on
Spark includes CUDA-side use that ordinary process RSS and Docker memory accounting
omit; identify the PID from the container process tree before assigning it to a
co-tenant.

The spatial trainer also bounds unused cached blocks at completed optimizer
boundaries (4 GiB by default) and force-reclaims them before checkpoint
serialization. It clears completed-step gradients first. This prevents variable
shapes from retaining tens of GiB and prevents checkpoint CPU staging from
temporarily stacking on top of that unused CUDA cache. Metrics record the
pre-reclaim reservation, post-reclaim reservation, and bytes returned.

The target spatial supervisor enables native decoder-layer activation
checkpointing. Retrieval and payload I/O remain between recurrent core passes and
outside checkpoint closures, so recomputation cannot repeat a read. This is an
execution-only resume setting, recorded in the run environment and each subsequent
checkpoint configuration; it reduces the active activation peak at additional
compute cost.

Optional `cuda_memory_fraction` limits this process's CUDA allocator;
`min_system_available_bytes` checks host `MemAvailable` before allocation and
between updates, requesting a checkpointed stop under pressure. Neither adds host
RAM and CUDA totals together. Unsupported host memory reporting is recorded as
unknown. These controls are portable opt-ins rather than Spark assumptions.

`stall_timeout_seconds` dumps all-thread stacks during stalled compute. It is
disarmed for checkpoint writes and never hard-kills a process with unsaved work.
A hung native kernel must return before cooperative checkpointing is possible;
SIGKILL or power loss can only recover the last committed state.

Benchmark representative shapes before changing activation checkpointing or
batch size. GPU memory headroom alone is not throughput evidence. Keep cold-disk,
warm-cache, exact-scan and ANN claims separate. Model-specific bgkit DeltaNet
kernels do not apply to this LFM2 convolution/attention backbone.

The detailed adoption inventory is in [the bgkit audit](bgkit-audit.md).


## Deduplicate immutable warm-start weights

Completed warm starts can retain byte-identical weight files in separate checkpoint
sets. `scripts/deduplicate_checkpoint_weights.py --root RUNS_ROOT --report PLAN.json`
is a dry run. Add `--apply` with a new report path to replace verified duplicate
weight files with hard links on the same filesystem. Every candidate SHA256 is
checked before any replacement; stage and parent run locks must be available.
Permissions, ownership and extended attributes must match. Symlinked checkpoint
aliases and already-shared destination files are excluded. Failed atomic publication
preserves the original destination. Optimizer/RNG/cache/config files and manifests
are unchanged; ordinary independent checkpoint retention still works.

This relies on the existing immutable-checkpoint contract: never edit checkpoint
files in place. Hard-link support is required; unsupported filesystems fail without
silently copying or deleting payloads. Reported bytes are deduplicated logical file
lengths; open readers can delay physical release. This is explicit maintenance,
not an automatic background mutation of running training checkpoints.


### Long oracle confirmations

`scripts/evaluate_oracle_transfer.py` commits each completed scoring variant and
its captured causal read plans, then saves generation progress every 32 strings.
SIGINT/SIGTERM or a control-file stop preserves all completed strings at the next
generation boundary. Resume with the same command after clearing the stop request.
It verifies checkpoint, corpus, evaluator and bank identities before reusing work;
changed inputs require a new output directory. A hard kill may repeat at most the
uncommitted generation block or the currently incomplete scoring variant.

Progress is small JSON containing scores/plans/predictions, never model weights.
It uses atomic fsynced publication and the configured free-disk reserve. Banks are
still written offline and inference still forbids writers/compactors. The progress
files retain their original phase resource observations; they are not fresh whole-
run timing measurements. Historical frozen evaluators retain their earlier restart
behavior and must be resumed using their own checkout.


The bounded compact/paired confirmation controller also watches stop requests at
its study root and `confirmation-queue` directory, both while waiting for training
and while children are running. It stops launching pending arms and sends SIGTERM
to active evaluators, whose current implementation flushes generation progress.
This controller stop does not stop a separately running training process; target
that training stage explicitly when needed. Clear the requested stop before
restarting the same frozen controller.


Successful pytest temporary checkpoint fixtures are retired automatically. One failed
session is retained for debugging. This policy applies to disposable test directories,
not experiment checkpoints or their recovery state.
