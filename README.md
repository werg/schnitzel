# SCHNITZELJAGD

> Formerly SDKB (`sdkb`). Historical experiment records retain their original names.

[Research status](docs/research-status.md) summarizes what is established and what
remains open, including the completed reader-capacity and real-trajectory studies.
The next target combines the [trajectory memory v0.5 plan](docs/trajectory-memory-v0.5.md)
with the [mutable-bank v0.8 contract](docs/mutable-bank-v0.8.md):
frequent visible memory search/write tool calls at distinct causal positions,
length-scaled write-call counts, recursive read-augmented writes into one
continuously learned bank, and
corpus-scale storage. The first compatibility stage now implements multi-site stored
reads; learned call placement, multi-site writes, recursive authored bank evolution,
and corpus scale remain planned and are not attributed to it.
The multi-site target runs the whole teacher-forced trajectory at each recurrent
level, retrieves all active spatial sites together, and scatters results into blank
site workspaces for the following level.

Run ownership, detached start/stop/resume, optional W&B and verified external-disk
archives are documented in [portable training operations](docs/operations.md).
See [current Spark validation](docs/validation-spark.md) for tested behavior and remaining limits.
The [Muon binding study](experiments/binding-muon-20260919/README.md) and
[longer MLP continuation](experiments/binding-continuation-20260919/README.md) confirm
narrow stored-memory action composition with both attention and the pooled MLP.
Reliable competing-entity composition and exact unseen identifiers remain unresolved;
the first [learned routing intervention](experiments/binding-routing-20260919/README.md)
learned source types but did not solve entity selection.
Later [global retrieval tests](experiments/binding-global-routing-20260919/README.md)
exposed the dependence on supplied world membership. Later
[cross-world training](experiments/binding-global-stored-long-20260919/README.md)
reaches 101/128 generated actions on a fresh 128-record bank, but
[adding distractors](experiments/binding-bank-scale-20260919/README.md) reduces this
to 76/128 at 4,096 records, with uncertain benefit over no memory.
[Corrected scoring precision](experiments/binding-routing-precision-20260919/README.md)
preserves this pattern (102/128 and 78/128). An
[exact-detail control](experiments/binding-exact-detail-20260920/README.md)
copies 64/64 unseen identifiers from selected text but 0/64 from latent payloads;
perfect retrieval alone does not solve this representation/training failure.
The [matched breadth](experiments/binding-detail-breadth-20260920/README.md) and
[alternate-history](experiments/binding-endpoint-views-20260920/README.md) continuations
preserve oracle action composition but still give 0/64 exact identifiers.
[Temporary-compaction training](experiments/binding-compact-aware-20260920/README.md)
now makes a single mean-plus-mass MLP code preserve 128/128 actions and the tested
rule-change pairs; its matched raw-trained mean-code control gets 81/128. The raw
path regresses to 126/128. The completed [paired-objective comparison](experiments/binding-paired-compaction-20260920/README.md)
has mixed raw-path results and no compact advantage over interleaving. Raw fallback
records remain stored; net disk savings are not established.
A [lexical selection control](experiments/binding-lexical-routing-20260920/README.md)
recovers 128/128 stored-memory actions at 4,096 records using an additional source-token
index; this exposes the literal-name fixture and does not improve learned keys.
A [focused copy diagnostic](experiments/binding-copy-fit-20260920/README.md) fits
64/64 training identifiers but fails changed and held-out endpoints, with rule/action
regressions. The completed [endpoint freshness study](experiments/binding-endpoint-freshness-20260920/README.md)
reduces old-target reuse with broader training but still generates 0/64 original
held-out endpoints. [Frozen readouts](experiments/binding-reader-stages-20260920/README.md)
find weaker character access through the reader than in its input payloads. The
[reader-capacity study](experiments/binding-reader-capacity-20260920/README.md) is
now training matched reset readers of two widths; its capability results are pending.
The [bgkit audit](docs/bgkit-audit.md) records storage, recovery and runtime adoption.

**Stigmergic Compactable Holographic Neural Indexed Trajectory Zettelkasten with Evolving Latents, Jointly Adapted by Gated Decoders**

SCHNITZELJAGD trains a small language model to write experiences into stored latent values,
compose selected values through a recurrent set reader, and use the result on later
questions or actions. Selective replay trains the writer without retaining every
source graph. Compaction learns to replace groups with synthetic records while
preserving their conditional contributions.

**Version 0.4.0 · Student `LiquidAI/LFM2.5-230M` · Initial GPU validation: DGX Spark.**
The Python runtime supports CPU/CUDA; Spark-specific checks live in the container wrapper.
The Python package, CLI, agent class, scripts, container and active documentation now
use SCHNITZELJAGD (`schnitz`, `SchnitzelAgent`). Historical experiment logs remain immutable evidence.

## Start on the Spark

Clone the upstream repository using your existing access:

```bash
git clone git@github.com:werg/schnitzel.git schnitzel
cd schnitzel
./scripts/start_spark.sh --recipe recipes/looped_smoke.yaml --output /runs/looped-smoke
```

This **real-model integration run** builds the native ARM64 image when absent,
checks CUDA/BF16, pins model and dataset revisions, prepares real teacher traces,
checks memory gradients and runs two updates in each of four stages. It is not
a capability experiment. After inspecting its outputs, start the main curriculum:

```bash
./scripts/start_spark.sh --recipe recipes/looped_starter_muon.yaml --output /runs/looped-starter-muon
```

| Stage | Optimizer updates | Training behavior |
|---|---:|---|
| `text_bootstrap` | 200 | Adapt the student to the recorded target format with the same selected evidence rendered as text. |
| `recurrence_bridge` | 200 | Two middle-core passes, frozen parent, live bridge, and one-pass parent distribution anchoring. |
| `latent_warmup` | 400 | Two passes with an actual inter-pass SCHNITZELJAGD read; freeze the base while training the latent interface. |
| `recurrent_joint` | 400 | Sample 2/3 passes; update shared core + SCHNITZELJAGD modules, retain one-pass text anchoring. |

The Muon starter uses native Muon for eligible matrices and AdamW for embedding,
output, slot and other excluded tensors. It requires a runtime providing native
Muon; keep the NVIDIA Torch installation intact. Checkpoints default to every
1,000 updates plus initial, final and graceful-stop saves; configure output storage
through the operations guide.

These are initial run budgets, not promised convergence thresholds. The starter scans
up to 2,000 Hermes and 2,000 UltraChat rows with seeded source shuffling. It records
how many complete, budget-fitting examples survive preparation. No live teacher API,
experiment-tracking account or paid inference endpoint is required.

Resume the identical recipe/output:

```bash
./scripts/start_spark.sh --recipe recipes/looped_starter_muon.yaml --output /runs/looped-starter-muon --resume
```

The job runs synchronously in the foreground. A graceful interruption checkpoints
after a complete microbatch/replay, preserving partial gradient accumulation for
exact resume. Completed stages and evaluations are not repeated. Prepared
input/config changes are rejected on resume. Run only one launcher per output directory.

To download, pin and inspect data without loading the training model:

```bash
./scripts/start_spark.sh --recipe recipes/looped_starter_muon.yaml --output /runs/prepared --prepare-only
# Inspect data/manifest.json and launch.json; then continue:
./scripts/start_spark.sh --recipe recipes/looped_starter_muon.yaml --output /runs/prepared --resume
```

## Training recipes

| Recipe | Purpose |
|---|---|
| `looped_starter_muon.yaml` | Hermes tool conversations + UltraChat with Muon/AdamW parameter split, sparse saves and offline W&B; initial budgets. |
| `looped_starter.yaml` | Historical AdamW starter control. |
| `tools.yaml` / `chat.yaml` | Separate tool-use and general conversational continuation runs. |
| `coding.yaml` | Successful SWE-smith traces; earlier-prefix memory. |
| `openhands.yaml` | Successful Nebius SWE-rebench/OpenHands coding traces. |
| `cross_experience.yaml` | Different prior SWE-smith instances in the same repository; repository-held-out validation. |
| `looped_causal.yaml` | **Real LFM student** text-to-latent training on controlled rules, then fresh-world counterfactual and binding evaluation. |
| `looped_binding.yaml` | Selected-pair permission, restoration, action and identifier curriculum with rule counterfactuals; oracle routing. |
| `looped_binding_muon_selected.yaml` / `looped_binding_muon_all.yaml` | Native Muon MLP curricula with selected versus all-world evidence; external-storage operating policy and sparse checkpoints. |
| `looped_binding_muon_attention.yaml` | Matched attention control; demonstrated selected-support action composition, with explicit distractor/identifier limits. |
| `looped_smoke.yaml` | Real model and starter sources, two updates per stage, native recurrence preflight. |
| `tiny_looped_smoke.yaml` | Tiny CPU recurrent curriculum and fresh causal worlds, no downloads. |

The previous `starter.yaml`, `causal.yaml`, and dataset-specific recipes remain one-pass controls; the `looped_` recipes explicitly enable recurrent conversion. `smollm2_looped_causal.yaml` is the attention-only conversion comparison. Paths are under `recipes/`. Sources/configurations/licenses and exact parsing are in
[the dataset guide](docs/datasets.md); `schnitz datasets` prints the executable catalog.
See [training](docs/training.md) and [Spark setup](docs/spark.md) for operations.

**Two separate data protocols are implemented.** Prefix memory encodes earlier
context before a later assistant target. Cross-experience memory encodes different
completed instances, never the query's own future solution. Supplied context is not
falsely labeled as a verified sufficient support set. The controlled causal suite
provides that stronger test.

## Docker choice

The default is **`nvcr.io/nvidia/pytorch:25.11-py3`**, explicitly used in
[NVIDIA's Spark fine-tuning instructions](https://build.nvidia.com/spark/unsloth/instructions).
The build script pulls `linux/arm64`, verifies architecture, resolves an immutable
image digest and preserves NVIDIA's torch/CUDA installation. An isolated venv adds
Transformers 5.17.0 and datasets 5.0.1 without replacing the GPU runtime.

SCHNITZELJAGD does not require Unsloth model patches, FlashAttention, bitsandbytes or custom
convolution wheels. It uses the public Transformers embedding interface and the
ordinary pretrained path as its one-pass baseline. A documented base image is not
proof that the complete assembled stack has run here: build-time imports and the
real-device model/gradient preflight verify it on the Spark. `SCHNITZELJAGD_BASE_IMAGE` is an
explicit override for another tested NVIDIA image.

## Implemented architecture

```text
source experience -> shared LFM + learned write slots
  -> canonical key and fixed values -> per-space codecs -> stored payloads

prompt -> prelude -> shared core -> query/read -> bridge + soft result
  -> same shared core again -> coda -> later teacher target

training: consumer cotangents -> selective replay of live source producers
inference: stored payloads only; no producer encoding on the read path
```

The new main experiment retains LFM's 14 layers, repeats layers `[4:10]`, and
reads SCHNITZELJAGD between core passes. The one-pass plain-input path exactly preserves the
parent at installation. The writer stays at one pass while consumer depth changes.
Native masks and positional conventions are retained; no KV/conv cache crosses a
loop. The initial gates are small but live, not zeroed across all extra computation.

The recurrent set readers, multiscale transforms, selective producer replay, and
checkpointing remain. In-loop raw reads and persisted full-cluster codes are
implemented, with complete cumulative aggregation at each boundary and raw fallback
for partial clusters. Native single-read training supports temporary mean/synthetic
compaction with interleaved or paired raw/compact objectives, including replay and
emergency resume. Streaming in-loop reads and multi-read compaction remain deferred.
Offline code fitting is separate from stored-only reads.

[The research and conversion note](docs/recurrence.md) explains the choice, equations,
2025–September 2026 primary evidence, conversion curriculum, limitations, and exact
commands. The legacy full-stack adapter remains available as a control.

## What is measured

Every stage writes held-out supports once, serializes/reopens the bank, and performs
stored-only teacher evaluation. Metrics include full-target NLL, token-weighted
summaries, paired trajectory intervals, no memory, zero payloads, and fixed-key/ID
payload permutation. Reference likelihood is **not** an agent execution success rate.

After the last checkpoint freezes, new Boolean and multi-binding worlds test support
removal, adjusted-answer source counterfactuals, joint evidence and exact identifiers.
For the starter this is an out-of-domain diagnostic; `looped_causal.yaml` supplies the relevant
controlled training family. No evaluation threshold silently changes the recipe.

Artifacts stay under the run directory: revision locks, prepared JSONL, model probe,
metrics, optimizer/model/cache checkpoints, stored banks and evaluation JSON. Input
files are hashed and indexed by offsets rather than all retained in training RAM.

## Offline development

With Python 3.11+ and CPU PyTorch installed:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
schnitz launch --recipe recipes/tiny_looped_smoke.yaml --output runs/offline
schnitz launch --recipe recipes/tiny_looped_smoke.yaml --output runs/offline --resume
```

Core tests do not download models/data. Current execution and capability evidence
is in [Spark validation](docs/validation-spark.md). The earlier
[0.4 handoff](docs/validation-v0.4.md) remains historical CPU evidence.

[Recurrent conversion](docs/recurrence.md) · [Positional memory interface](docs/positional-memory-v0.7.md) · [Architecture](docs/architecture.md) · [Implementation](docs/implementation.md) ·
[Dataset guide](docs/datasets.md) · [Training](docs/training.md) ·
[Migration](docs/migration.md) · [Publication/handoff](docs/handoff.md)
