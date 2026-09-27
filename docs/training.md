# SCHNITZELJAGD training and causal-transfer protocol

Detached runs, graceful stops, W&B and external-disk checkpoint archives are
described in [portable operations](operations.md).

## Adaptive spatial bank training (v0.6)

The target spatial trainer may enable `memory.distance_gating` and
`train.writer_replay_records_per_site`. It then requires the exact standalone source
manifest through `scripts/train_spatial_bank.py --sources ...`. Selected records are
encoded from prompted causal prefixes ending at `memory.write`, cast to actual
storage precision, and captured as replay leaves. Consumer backward is followed by
writer replay before clipping and the optimizer step. Updated keys and payloads are
then regenerated into the checkpointed training overlay.

Within one optimizer step, a record selected at several read waves shares one
serialized replay leaf, so every consumer use accumulates into the same producer
cotangent. Post-update refresh groups records by source length and may use a larger
no-grad batch; it still regenerates every touched key and payload before the next
step. Overlay payload reads are batched by space while retaining the authorization
checks performed by the immutable base store.

In the overlapped pipeline, independent microbatches at the same recurrent level
combine their selected writer records into one producer batch. Their consumer graphs
remain separate and execute in deterministic round-robin order; shared replay leaves
sum every use before the single producer replay.

`--profile-steps N` synchronizes phase boundaries for the first `N` steps of a
process and records forward, consumer-backward, writer-backward, optimizer, refresh,
and cache-reclamation wall times. It is a diagnostic mode because those additional
synchronizations can reduce overlap; target supervisors use three steps after launch.

`--retain-writer-replay-activations` disables layer recomputation only while an
individual producer replay batch runs forward and backward. Consumer graphs keep
their configured checkpointing policy, and the writer policy is restored afterward.
This is a platform-neutral throughput option for machines with measured activation
headroom; it does not change selected records, stored precision, gradients, or refresh.

The spatial trainer may retain bounded unused CUDA allocator blocks between steps to
avoid repeated native allocations. `--max-unused-cuda-gib` sets that cache allowance,
while `--cache-reclaim-host-reserve-gib` forces release when host-available memory
falls below its reserve. This applies to both unified and discrete-memory hosts;
checkpoint boundaries always force release. The Spark target supervisor uses a
28 GiB unused-cache ceiling and a 16 GiB host reserve based on measured peaks.

The source manifest digest must equal the digest recorded by the verified base
snapshot and its logical-record catalog.
The overlay is restored from `training_cache.sqlite`; deleting or omitting it changes
the next retrieval plan and is not an exact resume. Hard search still uses a bounded
candidate field. `train.routing_weight` controls the gentle support anchor, while
the ordinary task loss trains continuous gates and selected keys/payloads directly.

### Full-bank address correction after v0.6 R3

The first 100,000-source R3 run showed unassisted top-256 positive recall near the
random expectation, despite improving task NLL with supplied supports. Its writer
keys retain article-level structure; the causal query-to-key alignment is the
immediate bottleneck. A continuation can use `--routing-episodes` with the original
episode JSONL and `--sources` to train the same packed trajectories with a stronger
address objective. It samples causally eligible random and other in-batch verified
sources as early negatives, then ramps the weight of globally mined exact-search
negatives over 3,000 optimizer steps. Verified supports remain the main positive
signal. A weak source-derived lexical similarity distribution supplies an auxiliary
gradient over the *same candidate field*. It does not choose records, inject text,
or enter inference. Unassisted top-k recall remains the primary address metric.

`--routing-weight` sets the new stage's address-loss weight. Changing it or the
curriculum starts a new warm-start stage. `--inherit-bank` copies the committed
parent journal into that stage after checking it matches the parent checkpoint;
the model and mutable keys/payloads therefore begin at the same causal revision.
The new stage resets optimizer state deliberately and saves its own step-zero
checkpoint. `--maintenance-records-per-step` refreshes a bounded additional set
of old records, while writer replay refreshes the selected records after every
step. The offline bank remains a physical bootstrap, not a frozen logical target.

For large candidate fields, `--routing-logit-scale` sharpens the training
softmax over cosine scores without changing exact-search order or the reader's
adaptive distance gates. `--routing-live-weight` adds a separate address loss
through replayed writer keys. The primary routing loss stays on serialized
stored keys, so its forward result matches search. Live-key loss reaches the
writer through replay; key-stability loss limits the discrepancy until the
updated keys are published. Both settings are stage fingerprint inputs.

The curriculum should be judged on source-disjoint validation queries using
unassisted positive recall at the actual neighborhood budgets, then with correct,
wrong, and zeroed latent payloads on the same task examples. Lower routing loss or
teacher NLL alone does not establish usable retrieval or information transfer.
If address learning stalls, a small embedding teacher can supply a richer soft
similarity target; each space should have its own trainable teacher projection and
the additional teacher compute and key-refresh cost must be reported. The
[keyspace distillation plan](keyspace-distillation-plan.md) records the causal
teacher inputs, direct per-space head conversion and validation gate.

### Staged whole-bank refresh after measured key drift

The first R3 continuation revealed that selected-record refresh alone can leave a
large bank incoherent: about 58,757 records had journal heads and 41,243 still used
the original base keys, while sampled head/base key cosines were near zero in all
four spaces. Hard search then compares vectors produced by substantially different
writer states. See [the key-coherence validation record](validation-key-coherence-2026-09-23.md).

`scripts/refresh_training_bank.py` stages a complete all-space rewrite from a
committed writer checkpoint. It copies the stopped run's journal to external
storage, encodes sources using the same prompted `memory.write` prefixes as the
trainer, records chunk progress, and publishes a manifest only after every source
has a new key and payload in each space. The original run and checkpoint are left
intact. A new `schnitz` spatial stage can use `--bank-journal` with this complete
manifest and `--init-from` the matching model checkpoint. The warm start resets
optimizer state and saves its own bank-aligned step-zero checkpoint.

`--writer-key-learning-rate` can keep the key head and per-space writer address
maps trainable at a smaller rate than the query maps and reader. This reduces
global key rotation while preserving continuous key learning. It is not a
substitute for measuring drift and refreshing the full bank again. The first
coherent stage should check head age, key similarity, unassisted retrieval, and
task controls before choosing a refresh cadence.

For coherent-bank training, the contrastive address loss scores the **stored**
keys that exact search actually ranks. Selected live writer keys still affect
the reader's continuous gate and task loss. With `--key-stability-weight`, an
additional cosine penalty keeps replayed writer keys near their current stored
revisions, accumulating its producer gradient before the optimizer step. This
allows writer keys to move while preventing the address objective from chasing
newly encoded positives that cannot yet be found in the persistent index.
`--routing-hard-ramp-steps` controls how quickly full-bank mined negatives gain
weight. The logs separate easy, hard, lexical, and key-stability terms.

Document ingestion data should use the helpers in `document_ingestion.py`. Provide
both complete-document prompts with several model-chosen writes and streaming
natural/manual parts with one visible write after each part. Store exact source spans,
chunk policy, call IDs, and parent read IDs in the prepared artifact.


> **v0.4 update:** the recommended entry points are `recipes/looped_smoke.yaml`,
> `recipes/looped_starter_muon.yaml` and `recipes/looped_causal.yaml`. They add native
> middle-block recurrence and in-loop reads. See [recurrent conversion](recurrence.md)
> for the four-stage protocol. The one-pass recipes below remain control experiments.
>
> [Trajectory memory v0.5](trajectory-memory-v0.5.md) is the next curriculum:
> visible memory tool calls, frequent multi-site reads, length-scaled writes, and
> repeated read-then-write evolution of one logical bank. Source events and causal
> frontiers are immutable, while stored keys, payloads, indices, and compact codes
> continue to learn. [Mutable bank v0.8](mutable-bank-v0.8.md) is the canonical
> lifecycle contract. The current support/query trainer is Phase 0 interface
> pretraining for that program.

## Foreground staged launcher

The entry point is `schnitz launch`, normally invoked through `scripts/start_spark.sh`.
The same Python launcher is exercised in CPU integration tests; it is not a placeholder
shell command. It resolves the model revision, loads its tokenizer, prepares immutable
inputs, runs the model/gradient preflight, executes stage dependencies and evaluates
each committed stage. `--prepare-only` stops before model/GPU allocation.

The default student is the native instruction-tuned `LiquidAI/LFM2.5-230M`, not its
Base, GGUF or inference-server variant. The initial memory configuration uses one
space, oracle retrieval, one backbone pass, eight write/read slots and three pooled
reader rounds. Existing recurrence and attention/multiscale alternatives remain
independent configurable experiments rather than a compulsory bundle of changes.

## Objectives and stage transitions

`text_bootstrap` adapts the interpreter to the normalized recorded-target task using
exactly the selected support text available to the memory arm. `latent_warmup` loads
that checkpoint, freezes backbone parameters, and trains write slots, query/codecs
and reader through the frozen network. Gradient propagation into soft inputs still
works with frozen weights. `latent_joint` unfreezes with a smaller backbone learning
rate and optionally retains a same-example oracle-text loss.

For memory stages the recorded target NLL flows through the fixed soft response and
reader into the selected live source producers. Selective replay recomputes those
producers after accumulating consumer cotangents. Shared parameter gradients are
all accumulated before the optimizer step. Cached values remain their actual stored
encodings; replay is not falsely described as an unbiased all-fresh gradient.
Fresh encoding remains a **training** operation. Evaluation reads never invoke it.

The optional `oracle_anchor_weight` adds text-path target NLL on the same selected
evidence. Logs distinguish the raw memory `loss`, `oracle_anchor_nll`, and actual
weighted `optimization_loss`. A stage warm-start uses a new optimizer/cache; exact
resume restores model, optimizer, RNG and intentionally stale cache instead.

`oracle_alignment_weight` is an experimental, default-zero training objective for
native in-loop oracle memory without compaction. It adds mean `1 - cosine` between
the latent decoder's final answer-prediction states and detached selected-text
states at the corresponding next-token positions. The text path shares current
model weights and must retain a positive `oracle_anchor_weight`; it is an anchored
moving teacher, not a separately frozen model. Its depth is `oracle_anchor_loops`.
Both paths receive the same selected evidence and preceding answer tokens only;
the text states supervise a loss and never enter a memory query, writer or read.
Training rejects a schedule that has not actually read the declared evidence.
The text forward is reused for its NLL anchor. `oracle_alignment_loss` is logged
separately, included in `optimization_loss`, and its accumulated total survives
partial-update emergency recovery. Changing its weight requires a warm-start fork.
This is a learning intervention, not evidence of improved generation or an
inference-time text bypass.

`oracle_distillation_weight` is a separate, default-zero experiment for the
native R=2 memory path. It minimizes next-token KL from the selected-text R=1
distribution to the latent path's answer distribution, alongside latent target
NLL. The one-pass teacher has a frozen parent backbone and is evaluated under
`no_grad`; its answer-position outputs use only the same query, selected source
text and preceding answer tokens. Teacher states and logits never enter a query,
writer or stored read. Both paths use the same selected source IDs, and training
rejects a missed read. The supported configuration is one in-loop oracle read,
writer R=1, fully live payloads, no compaction or other text objective, and no
sampled depth. The loss adds gradients to the consumer, reader and replayed
writer, while the teacher remains detached. `oracle_distillation_kl` and the
weighted `optimization_loss` are logged separately. Emergency checkpoints retain
the partial KL total, optimizer, gradients and RNG; changing its weight starts a
new warm-start fork. A lower KL or NLL alone does not establish payload use or
agent success; compare frozen stored-only real, zero, wrong and text conditions.

`warmstart_memory_gate` explicitly replaces only the native bridge's memory
injection gate after loading a parent model into a **new** run. It requires a
middle-block warm-start checkpoint and a probability strictly between zero and
one. The old and requested gate values are recorded in `initialization.json`;
the initial checkpoint contains the changed value and a fresh optimizer. Exact
resume loads the checkpointed gate and optimizer without reapplying the override,
and rejects a changed gate setting as a hyperparameter mismatch. The one-pass
parent path still bypasses the bridge. This option is an experimental starting
condition, not an automatic inference-time adjustment or evidence of better
memory use.

Default budgets: four source chunks, 512 source tokens each, 4096 outer-prompt tokens,
512 complete target tokens, a 2048-dimensional per-source storage payload, eight
returned slots. Canonical write, storage-code and returned-read capacities are separate
config axes. Targets are never silently truncated. These bounded windows are not
an architectural limit on trajectory read counts; producer replay addresses graph
storage independently.

The active v0.6 compatibility generation has eight canonical writer value slots and
eight returned reader slots. Phase 2 follows `positional-memory-v0.7.md`: first
distill an eight-position structured interface, then expand both counts to 32 while
retaining per-position channel widths `[32, 64, 128, 256]`. This yields total stored
widths `[1024, 2048, 4096, 8192]` without a 503-million-parameter dense codec.
Use a new optimizer, regenerated trajectories, and rebuilt banks. Never label this
transition an exact resume.

Gradient accumulation is four examples per optimizer update. Parameters/optimizer
state remain FP32, forward operations use BF16 autocast, and stored values use BF16.
Reader chunk checkpointing and source replay reduce activation retention, not total
parameter storage or arbitrary serving-context memory.

## Run layout and resume

```text
runs/starter/
  input-lock.json                 # recipe/base-config identity
  model-lock.json                 # actual model revision
  launch.json                     # dependencies and prepared-file checksums
  data/
    normalized.jsonl              # audit messages, source provenance and split
    train.jsonl
    validation.jsonl
    manifest.json                 # source revisions, filters, licenses and budgets
  model-probe.json
  text_bootstrap.yaml
  text_bootstrap/                 # CURRENT, checkpoint sets, metrics/cache
  text_bootstrap-evaluation.json
  latent_warmup.yaml
  latent_warmup/
  latent_joint.yaml
  latent_joint/
  fresh-causal.jsonl
  causal-evaluation.json
  fresh-multiuse.jsonl
  multiuse-evaluation.json
```

Use one immutable recipe/output pair. A changed recipe, base configuration or prepared
file hash requires a fresh output directory. `CURRENT` names a fully committed atomic
checkpoint set. Never use partially written temporary files as a resumable state.
A hard kill returns to the last committed checkpoint. A graceful stop saves after
the current complete microbatch/replay, including any partial gradient accumulation;
resume finishes that same optimizer update. Runs with no committed checkpoint are
not silently overwritten.

Completed stage/evaluation markers are independent. In particular, a failed binding
evaluation after causal evaluation is retried without retraining or being skipped.
The high-level recipe has fixed step counts; the lower-level `schnitz train --steps`
can extend a compatible run deliberately. Do not advance a stage outside its sealed
launch plan and then expect the launcher to reinterpret it as unchanged.

## Real-student causal transfer

```bash
./scripts/start_spark.sh --recipe recipes/causal.yaml --output /runs/causal
```

This uses LFM, not the tiny backend. It learns on controlled Boolean support worlds
through the same text/latent stages, then freezes all weights before preparing new-world
supports. Evaluation writes each new source once, serializes/reopens the store, and
asks multiple later questions. Conditions include both supports, no memory, zeroed
payloads, dropping A or B, and content counterfactuals with adjusted correct answers.
Multi-binding rules and exact endpoint identifiers provide another held-out family.

Known sufficient support groups belong to these generated tasks. The teacher datasets
lack that annotation, so complete-support recall is not invented for them. A larger
model excluded from post-freeze random facts is not a fair capacity-substitution
control; use equal source access when comparing the deployment frontier.

The starter recipe's causal suite is an out-of-domain diagnostic, while `causal.yaml`
trains that task family before holding out new worlds. Failure can reflect interpreter
or format mismatch as well as memory quality. No scientific metric threshold changes
or stops a recipe silently. The goal is to obtain interpretable measurements.

## Real-trajectory evaluation

### Controlled procedural curriculum

`recipes/looped_binding.yaml` uses `protocol: binding` to train on permission,
restoration, action and exact-identifier queries from generated multi-entity
worlds. The four stages and stored-only evaluation are the same as the Boolean
curriculum. `bindings` sets entities per world; `causal_train_worlds` retains its
existing name and controls the generated training world count. Train, validation
and post-freeze worlds have separate namespaces.

Oracle routing supplies the query's relevant source pair. This isolates action
interpretation and latent transfer; it does not establish learned entity selection
or correct binding among all competing records. Text controls receive exactly
the same selected sources. Both rule records are supplied for action queries,
including STOP branches, so selection cardinality cannot reveal the action.

`train.evidence_scope: available` supplies every candidate in the episode to
both text and latent arms. The original `required_ids` and `sufficient_groups`
remain the ground truth for ablations and support metrics; distractors are never
relabeled as sufficient. This option requires oracle candidate access and does
not claim learned routing. For generated binding worlds it exposes all entities
in a world, while world membership is still supplied. The default `required`
preserves existing recipes. Use `recipes/looped_binding_all.yaml` for the matched
all-context curriculum. Producer replay and serialized value precision are unchanged.

The current native-Muon variants are `recipes/looped_binding_muon_selected.yaml`
and `recipes/looped_binding_muon_all.yaml`. They use Muon for eligible matrices
and AdamW for embeddings, output heads and other excluded parameters. They start
fresh optimizer state; an AdamW checkpoint cannot become a Muon exact resume.
See [optimizer ownership and resume](operations.md#optimizers-and-exact-resume) for details.

`schnitz evaluate-transfer --binding-counterfactuals` flips permission or restoration
rules consistently throughout each world. IDs, queries, timestamps and read plans
remain fixed. Scores distinguish answer-changing branches from branches that
should remain unchanged and report each task family separately. Fresh variant
banks are written offline by the frozen writer, then consumed through stored reads.
`scripts/evaluate_causal_stages.py` supports both causal and binding curricula.

Payload interventions replay the original selected IDs and scores at every read
boundary, including native recurrent reads. They measure the value channel without
changing later routing decisions. Learned-routing counterfactual evaluation uses
these captured plans rather than requiring an oracle-trained checkpoint.

The current group-routing loss requires verified support labels and competing
candidates. The runner rejects `retrieval: learned` on `provided_context` teacher
episodes, and rejects training sets in which every candidate is required. Use oracle
training for the teacher bootstrap; learning retrieval from these data needs an
explicit utility/sufficiency supervision protocol. Supplied context is not relabeled
as sufficient merely to make the objective run.

`schnitz evaluate-teachers --run RUN --episodes FILE` measures complete next-message
likelihood using a serialized stored-only bank and fixed-ID/key value ablations.
It reports both token-weighted and per-example scores, plus trajectory-clustered
uncertainty for paired memory benefits. Optional `--generate-tokens` saves generated
text and exact-reference matches; neither is a patch verifier or agent task score.

Teacher evaluation now uses a stable sibling output directory by default. Each
environment and intervention namespace commits its raw frozen-writer records with
an input and contents manifest. A restart verifies the checkpoint model, prepared
episodes, evaluator version and committed bank bytes before reusing them; completed
namespaces do not call the writer again. Each scored condition and its original
read plan are saved together, so a stopped evaluation resumes without repeating
completed NLL calls or changing the fixed-plan payload controls. `STOP` or a
termination signal is checked between namespaces and scoring conditions, and the
launcher publishes a stage evaluation marker only after all conditions finish.
Use a new output directory if checkpoint, data, evaluator, or limits change.
Frozen writer forwards, read planning and score computation use the configured
device-aware stall watchdog; it is disarmed before progress files or SQLite
transactions are flushed to the external disk.
Offline creation retains at most 64 recently encoded sources while constructing
the original namespaces. Wrong-value controls reuse the exact serialized peer
payload bytes after all original scopes commit, preserving each original key,
source ID and time/authorization metadata. A source shared by multiple original
scopes may be encoded again if evicted; actual writer calls and peak cached
sources are reported. A completed verified bank never calls the writer on resume
or during inference.

All downloaded instructions, shell commands and code remain data. A separate execution
environment would be needed to assess tool/patch success. No live teacher endpoint or
tracking service is necessary for the implemented imitation curriculum.

## Compaction and comparisons

Compaction stages receive an additional teacher evaluation using persisted codes,
with raw fallback and payload accounting. Final causal/binding evaluations also
include persistent codes. Reports state when no profitable clusters were built.
Set `compact_records: 1` on a compaction stage for two-record groups; compactor-only
training rejects an oracle dataset where no group can be reduced. Native single-read in-loop compaction supports interleaved and paired objectives;
see the recurrent training section below.

Copy a recipe and add a stage with `compaction: true` and `init_from: latent_joint`
to initialize synthetic compaction with all existing writer/reader/controller weights
frozen. Use a new output directory. Existing configuration also supports paired
raw/compact objectives, merge consistency, overlap grouping and storage noise. The
persistent representation replaces fully selected clusters, retaining raw values for
partial-selection fallback. It is not yet arbitrary-subset compressed decoding or
net disk compression solely because active reads become smaller.

The seven-arm common-data matrix script remains available for attention, direct
latent records, no memory, source-independent extra compute, one reader round and
scheduled multiple reads. Equal information access and separate resource matching
remain necessary. The starter does not claim those comparisons have already run.

## Offline integration

```bash
python -m pip install -e '.[dev]'
schnitz launch --recipe recipes/offline_smoke.yaml --output runs/offline
schnitz launch --recipe recipes/offline_smoke.yaml --output runs/offline --resume
python -m pytest -q
```

This exercises the same preparation, three stages, stored-only likelihood and resume
using authored fixtures and a tiny byte-token model. It verifies execution, not learned
real-student capability. [Validation](validation-v0.3.md) records the tested scope.


## Isolated address training

`train.optimization_scope: routing` is an optional memory-stage scope for learned
routing with positive routing weight and raw records. Only `key_head`,
`address_maps` and `query_maps` train; payload generation, the query feature
extractor, reader and backbone remain frozen. Frozen child modules run in evaluation
mode while the top-level training flag preserves the configured routing warmup.
This preserves oracle-supported behavior and payload content while changing stored
keys and their ranking. The same replay, Muon ownership, checkpoint and exact-resume
contracts apply. Use a new warm-start fork when changing this scientific setting.


## Temporary compaction inside the recurrent reader

For a single-space, single-read native recurrent memory run, set `memory.compaction`
to `mean` or `synthetic`, `compact_records: 1`, and
`compaction_objective: interleaved`. `compaction_probability` chooses compact task
examples after `compaction_warmup`; other examples preserve the raw task objective.
Start an explicit warm-start fork when changing this objective. New synthetic
compactor parameters initialize separately; the source writer/reader state is retained.

The compact read consumes all selected records before its shared-state update and
matches the configured serialized value precision plus FP32 multiplicities. Its
auxiliary loss preserves conditional pre-normalization numerator and mass. Native
scheduled multi-read compaction remains unsupported and fails validation. Both MLP and attention readers support this same intervention. This
implementation and its gradient/storage checks do not establish useful compression;
compare held-out raw/compact counterfactual behavior at a declared update budget.

Set `compaction_objective: paired` to retain raw task loss on each compact-selected
example and add `compact_task_weight * compact_nll`, plus the contribution loss
and optional `behavior_kl_weight * KL(raw.detach() || compact)`. Both paths use the
same causal first-boundary query, selected sources and noisy values. This computes
two decoder paths and changes the loss scale; declare both when comparing budgets.
Groups too small to reduce still receive the paired objective, as in prefix mode.
Use a fresh warm-start fork; an exact resume must keep its saved objective.

## Explicit reader-capacity forks

`train.reinitialize_reader: true` is an opt-in warm-start experiment. It replaces
all reader weights with their seeded initialization while loading every compatible
non-reader weight from `--init-from`. Reader kind, width and round count may change;
stored payload, key and slot interfaces may not. Coupled compactor states are
rejected. The initialization manifest lists every reset reader parameter. Use a
matched reset-reader control when testing a larger reader, since resetting itself
changes behavior. This option is never applied during exact resume; the saved
reader, optimizer ownership and RNG state are restored normally.

## Selected producer computation for fully live oracle training

`train.selected_producers_only: true` is an explicit compute-policy fork. Only
sources already selected by the oracle evidence scope are materialized; every
selected source and scheduled read remains present. Python live/cache draws are
still consumed for all declared sources, preserving the episode/depth sampler.
The policy requires memory training, oracle selection and `live_fraction: 1`.
Cached history and learned routing require their existing full candidate behavior.
Inference and offline bank creation are unaffected.

The default remains false. Omitting an unused stochastic writer forward can change
Torch RNG consumption, so this is a recorded training configuration, not a silent
optimization or an allowed exact-resume override. Full-graph and replay execution
remain available under either policy. A native numerical/timing profile is recorded
separately before adopting the policy in future experiments.

## Four-space passage curriculum and snapshot-backed mutable training bank

`configs/lfm25_230m_four_space_reconstruction_spark.yaml` starts a fresh interface
around the pinned parent. It uses four MLP spaces with payload widths
256/512/1024/2048, maximum neighborhoods 128/64/32/16, two consumer passes,
one writer pass and Muon. A single-space checkpoint is structurally incompatible
and cannot be resumed into this configuration.

`scripts/prepare_squad_bank.py` writes an external, article-disjoint 10,000-source
train manifest, 1,024-source validation manifest and verified passage QA episodes.
The source manifest has no future question or answer. The pinned tokenizer and raw
corpus checksums are recorded. `scripts/prepare_four_space_pilot.py` makes a bounded
reconstruction/QA mixture; it checks every complete target against the configured
token budget, including EOS. Preparation writes to a fresh directory on an external
disk. The first stage uses supplied one-record evidence in all four spaces and
shuffled passes with resumable coverage accounting. Its loss is teacher-forced NLL;
held-out free generation and source interventions are separate evaluations.

`scripts/build_source_bank.py --run RUN --sources SOURCES --output BANK
--max-sources 10000 --shard-size 64` encodes the standalone manifest with a frozen
checkpoint. Each SQLite shard commits all space views of its logical records
atomically. Completed shards are byte-verified and skipped on resume; the complete
generation is published only after every declared shard and record is verified.
The manifest binds source order and bytes, writer checkpoint bytes, model and memory
configuration, storage precision, and the generation. Its values use BF16 while
keys remain FP32. Raw records and manifests remain outside Git.

For a bank-training fork, set `train.bank_dir` to a verified base snapshot,
`train.bank_read_limits` to explicit counts within the per-space neighbor caps,
`train.retrieval: learned`, `train.live_fraction: 0`, and keep the backbone frozen.
Start with `schnitz train --config ... --output ... --init-from RUN` using the exact
writer checkpoint named by the bank. The current bank path supports the native
two-pass, single-read model. It searches each space independently over the global
bank, fixes the eligible read plan before backward, fetches only selected stored
payloads, and trains query/address routing against verified positives and global
candidates. During this initial mixed-selection phase, known positives are always
delivered with retrieved distractors; telemetry reports their supplied status and
the learned retriever's unaided positive recall separately. The generic
corpus-training path freezes the writer and its stored transforms for that controlled
fork. The spatial trainer instead appends touched key/payload revisions to the
mutable-bank journal after optimizer steps. Resume checks the base fingerprint and
restores optimizer, sampler, RNG, partial-microbatch state, journal cursor, and
maintenance position without copying the bank into the checkpoint.

Published bank training loads a resident, contiguous FP32 key array for each
space (10.24 MB at 10,000 sources and four 64-dimensional keys). It performs an
exact CPU scan with the same domain/time/exclusion ordering as the SQLite
reference at index load; selected payloads still come from SQLite with fresh
visibility checks. A later deletion therefore fails closed at fetch time.
The resident array is rebuilt from the verified base snapshot on each start and
then patched from active journal heads. It is never copied into optimizer
checkpoints. A warm-cache Spark probe
measured about 0.05–0.06 seconds per space for the SQLite scan and about
0.0006–0.0007 seconds per space for the resident array on the 10,000-source bank.
These timings exclude cold NVMe behavior and do not claim ANN or million-record
throughput. This path establishes corpus-backed reader/address training;
joint live-writer replay against a refreshed corpus, learned-only delivery and
full-budget neighborhood training still need controlled validation.
Snapshot readers verify shard bytes under a shared SQLite read transaction, so
concurrent trainers and evaluations do not contend for a write lock merely to
recheck the base. The target mutable store uses pinned read epochs and atomic
post-step revision publication rather than a permanently frozen semantic bank.

Use `python scripts/gc_mutable_bank.py --journal RUN/training_cache.sqlite` to
inspect cursor retention and checkpoint pins. Adding `--before-cursor N --apply`
removes superseded logical revisions only when every retained checkpoint permits
that floor. SQLite reuses those pages for later revisions; add `--vacuum` only in a
maintenance window with enough scratch space when the filesystem must receive the
free pages immediately.

The optional `train.payload_contrast_weight` fork uses episodes with one verified
source and an earlier independent distractor (`--with-distractor` in
`scripts/prepare_four_space_pilot.py`). It adds a second native read using the
distractor for the *same* causal query and target, then penalizes a correct-source
NLL that fails to beat the distractor NLL by the declared margin. Both writers
are fully live and both consumer graphs contribute before selective replay and
the optimizer step. The fork requires one R=2 oracle read, no compaction and
`live_fraction: 1`; `payload_contrast_loss` and `swapped_source_nll` are logged.
The target NLL remains an anchor. Compare stored all/zero/wrong payload conditions
on held-out sources before treating a lower training loss as memory use.
Use `schnitz evaluate-teachers --output <fresh-directory>` for each checkpoint in a
fork; the result identity includes the checkpoint bytes and cannot be reused for
a different step.

`scripts/prepare_bank_training.py` creates title-located, answer-filtered QA
from a declared source offset. The current Spark bank stage uses sources
1,024 onward, disjoint from the first 1,024 reconstruction sources.
`scripts/prepare_bank_queries.py` selects a later disjoint block for evaluation;
the current source offset is 3,072. Article titles are present in stored source
text, and normalized answer strings are rejected if they occur in the query.
`scripts/evaluate_published_bank.py` verifies the
published generation, uses exact learned selection at explicit per-space budgets,
and reports NLL, support recall, zero-payload and removed-support controls. It
never calls the writer during inference. Both published-bank evaluators reject a
model whose actual serialized writer parameters differ from the writer snapshot
that created the selected bank. A bank-trained reader may change its recurrent
bridge and query/reader weights while retaining the one-pass writer. A first
small run is a mechanics check; source-held-out behavior and later-task
transfer require larger controlled runs.
`scripts/evaluate_bank_ranks.py` measures the complete verified-support rank
from the same causal native prefix, without fetching payloads. It reports
median rank, reciprocal rank and recall cutoffs for each space. This makes
addressing progress visible before a positive enters the small read budget.

When a frozen physical base snapshot has near-chance addressing, the controlled
`configs/lfm25_230m_four_space_local_routing_spark.yaml` stage trains source
keys and query maps together before any bank refresh. Its input comes from
`scripts/prepare_local_routing.py`: one verified earlier passage and three
distinct-article prior candidates per title-located question. On Spark the
training distractor pool is bounded to the first 3,072 source rows, leaving
the later heldout positive source block unseen in every training support role.
The target is kept out of the causal query. During the declared 512-update stage, oracle
delivery keeps the answer objective anchored while the differentiable routing
loss compares all four live candidates. A new bank must be explicitly rebuilt
from the resulting frozen writer checkpoint; an existing generation cannot
adopt its changed keys or values.

The optional `train.bank_payload_contrast_weight` stage uses one verified
stored source per question and a different causally eligible source outside
the selected neighborhood. It replaces the verified value in each space while
keeping the native query, selected counts and other fetched values fixed. The
correct and swapped teacher NLLs share the captured causal prefix and both
backpropagate before the optimizer step. `swapped_source_nll` and
`payload_contrast_loss` are logged. This supplies a direct value-use objective
for a frozen bank; it does not make retrieval successful by itself. The
unselected swap is an eligible comparison source, not a verified insufficient
record. Check source ambiguity and heldout interventions before interpreting
its training loss.
