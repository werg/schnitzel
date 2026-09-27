# Implementation specification — 0.3


> **v0.4 update:** the recommended entry points are `recipes/looped_smoke.yaml`,
> `recipes/looped_starter_muon.yaml` and `recipes/looped_causal.yaml`. They add native
> middle-block recurrence and in-loop reads. See [recurrent conversion](recurrence.md)
> for the four-stage protocol. The one-pass recipes below remain control experiments.
>
> The multi-site memory-tool interface is specified in
> [trajectory memory v0.5](trajectory-memory-v0.5.md). The first stored-read training
> slice and fixed-trajectory `memory.write` execution are implemented; learned call
> placement and generated write arguments remain later releases.
>
> **v0.6 update:** [adaptive memory](adaptive-memory-v0.6.md) specifies and tracks
> density-adaptive continuous read gates, support anchors across spaces, serialized
> key-and-payload writer replay, a checkpointed mutable training-bank overlay, and
> agentic document-ingestion prefixes. Consult its status table before treating a
> target behavior as implemented.

`trajectory_memory.py` defines and validates the first canonical transcript contract.
It requires each memory call to occupy its own assistant event, one logical record
per write call, paired tool results, monotonic optional timestamps, stored-only
latent attachment references, and causally earlier completed-read lineage for
read-dependent writes. Multi-site stored-read training and its bounded asynchronous
microbatch scheduler are implemented. Learned call placement and execution of
prompted writes remain subsequent releases.

The legacy recurrent consumer is spatially single-site. The new
`spatial_recurrent_hidden` path processes a complete teacher-forced trajectory in
parallel, gathers all query positions active at one recurrent boundary, calls one
provider batch, and injects the returned values into distinct blank workspace spans
before the next shared-core pass. `read_steps` still count depth boundaries; the
packed row manifest records the separate number of transcript sites.

The recurrent bridge accepts a `LoopWrites` bundle and atomically validates and
scatters several non-overlapping result spans before one shared core update.
`spatial_data.py` packs structured `memory.search` calls, result envelopes, and blank
workspaces using the real chat template. It can also pack a separately prompted,
supervised `memory.write` call after each task segment, one logical record per call,
with causal lineage to that segment's completed read. `spatial_training.py` batches all same-level
queries into one resident exact key scan per space, fetches pinned serialized payload
revisions, trains global routing against verified positives, and supervises assistant
tokens. The latent writer is absent from ordinary inference reads. During spatial
training, selected producers are replayed and touched keys and payloads are regenerated
after the optimizer step into the checkpointed training overlay. An
optional pipeline splits an optimizer batch into several retained microbatch graphs,
runs their search and serialized payload reads on CPU workers, and resumes their
recurrent boundaries in a deterministic order. Its microbatch size and in-flight
limit are recorded in the run fingerprint.

The implemented target run has two microbatches in flight. Live Spark sampling with
the local resident-key/NVMe bank shows continuous SCHNITZELJAGD GPU activity and low sustained
I/O wait, so this is adequate for the current regime. It is not evidence that remote
or network-bank latency is hidden. The pipeline still has a same-wave join before
combined writer capture, a bounded two-state queue, and a synchronous post-update
refresh/publication boundary.

Network-backed training needs a bounded completion-driven scheduler: keep a larger
configurable pool of paused causal trajectory states, issue reads asynchronously,
batch ready continuations by recurrent level and compatible shape, and apply
backpressure by retained activation bytes as well as request count. Preserve each
captured plan, causal metadata, RNG/autocast state, serialized precision, and all
producer cotangents through the optimizer boundary. Add issued/ready/consumed
timestamps, retrieval wait, ready-queue depth, continuation batch fill, GPU idle
intervals, and p50/p95/p99 latency telemetry before claiming latency hiding.
The current bounded pipeline now records request and service latency, ready-queue
delay, main-thread blocking time, and maximum pending reads. It still consumes a
deterministic same-level wavefront and therefore does not yet establish that remote
latency can be hidden.
Execution and publication of the already prompted write sites is implemented by
`build_prequential_bank.py`. Learned placement and network retrieval remain later
work; this executor does not claim that the model chose the calls or generated their
visible arguments.

`TrainingBank.update(..., event=...)` supplies the mutable prequential persistence
primitive. In one SQLite transaction it verifies that every logical write has all
configured space views, appends their physical payload revisions, advances logical
heads and index revisions, records a content-addressed event commit, persists
evidence lineage and write metadata, and advances a monotonic stream frontier. Exact
retries are idempotent. Because
ordinary search requires `created_at < query_time`, a record committed at an event's
visibility time cannot appear in that event's own reads. The executor follows file
order, resumes from the transactional frontier, runs every trajectory against the
same mutable catalog, encodes all prompted write arguments with the final writer,
and publishes those writes only after the trajectory finishes. It records selected
read lineage and bank size after every event.

The physical bootstrap snapshot remains read-only. Journal heads supersede it and
new records extend its resident index, so the two files implement one logical bank.
Checkpoints store a journal cursor and digest instead of copying the SQLite payload
database. Retained checkpoints pin history; explicit GC releases superseded rows
only below every pin. A conforming remote backend, learned call placement,
utility-driven maintenance, and training sampled from logged size bands remain planned.

The stored read path now depends on the public protocols in
`storage_contract.py`: stored payload fetch/lookup, scoped search, batched key
search, and mutable recovery control. SQLite implements those protocols without
exposing its connection to model code. This is an executable adapter boundary, not
a network backend result; the remote transaction and failure contract is specified
in [scale-out v0.9](scale-out-v0.9.md).

This file maps the research plan to executable behavior. `architecture.md` remains
the design document; the table in the root README is the implementation inventory.
Operational details are in [development-v0.2.md](development-v0.2.md).

## Shared writer, query and decoder

`SchnitzelAgent` uses a shared pretrained backbone (or the offline tiny model). A
source is tokenized using the student tokenizer, followed by `write_slots + 1`
learned input embeddings. The first resulting state produces the single canonical
key; the remaining states produce a fixed sequence of canonical value vectors.
Per-space address maps and codecs produce stored keys and payloads. Historical
checkpoints flatten those vectors before a dense codec. New positional checkpoints
apply a projection shared over value positions and retain a declared
`[positions, channels]` layout even when serialized contiguously. The canonical
Phase 2 operator and migration contract are in `positional-memory-v0.7.md`.

The writer receives only the source. The query head receives only the current
query prompt and, on follow-up reads, the preceding memory state. Required-record annotations, target answers and opaque IDs are not
neural input features. Keys are used for retrieval and routing loss, not silently
concatenated into the reader payload. Future tool observations are not available.
Source and prompt limits fail explicitly instead of silently truncating rules.

The read result has fixed output slots, normalized and gated into embeddings after
the prompt. Teacher-forcing loss predicts the first answer token from the last
context state, then each next token from preceding context. Only answer positions
contribute to the supervised loss. Greedy reference decoding recomputes prefixes;
it is not an optimized serving engine.

The legacy runner supports scheduled causal multi-read tasks. Each follow-up query uses
the previous soft working state; newly retrieved records are added to cumulative
evidence and recomposed into fixed slots. Complete-group supervision updates the
remaining-support target. The schedule is not an adaptive invocation policy or a
replay memory cap. Its single-site path remains synchronous; the spatial curriculum
runner provides the implemented asynchronous training path described above.

## Reader equations

For record x_i, query q, residual slot r_j and round t, the MLP evaluates

```
h_ij = activation(U_t x_i + V_t r_j + Q_t q + e_j)
g_ij = sigmoid(G_t h_ij)     # or unit gate in the lower-level API
numerator_j = P_t sum_i weight_i * g_ij * h_ij
mass_j = sum_i weight_i * g_ij
```

`P_t` has no bias and is applied after pooling. A bias could instead be included
per record with its mass; adding one once after pooling would change the function.
The update reads normalized numerator, log(1 + mass) and residual state, performs a
nonlinear update and fixed-slot mixing, and repeats. Record ordering is invariant;
output-slot and within-record information are not treated as exchangeable.

The attention baseline has the same update interface with exponential scores and
value projections. Its statistics carry a detached numerical log shift to allow
stable merging. The mass is never discarded. Tests include derivative parity at
zero multiplicity, attention chunk merging, query/state/gate gradients and
permutation invariance. Double-precision tests establish algebraic parity; BF16
rounding can vary with chunk shape and is not claimed bit-identical.

Chunk checkpointing does not build an all-pairs neighborhood attention matrix. It
still retains the small statistic graph per chunk/round and the selected payload
inputs; it is not a constant-memory training streamer. A separate single-space stored-only
inference streamer stages one payload chunk at a time and rereads the captured
plan each round. It does not overlap transfers with kernels or measure disk latency. All chunks complete a round before
the shared residual advances. Processing each chunk through all rounds independently
would be a different computation.

The multi-space branch evaluates each space against one common residual state and
merges its proposed update. Different widths and neighbor counts are explicit.
The main test corpus has few records and oracle routing, so merely enabling four
spaces does not constitute an experiment in large-neighborhood multiscale retrieval.

## Exact first-order producer replay

`ReplayTape.capture` runs a producer without retaining its graph, returning detached
floating leaves. The consumer accumulates cotangents on these leaves, including
multiple uses of the same leaf. `backward` replays only records with a cotangent
and backpropagates their vector–Jacobian products.

The tape preserves PyTorch CPU/CUDA RNG, autocast mode/dtype/weight-cache policy, parameter versions,
module training flags and buffers. It rejects parameter updates, changed modes
and mutable buffers before replay. Buffer-mutating producers are unsupported rather
than approximately replayed. Inputs/closure contents must remain immutable until
replay; the training runner captures each source tensor explicitly.

All producer and consumer parameter gradients accumulate before the optimizer
step. Live outputs include storage codecs and precision casts in the captured
forward, so the replayed Jacobian belongs to the values actually consumed. Cached
entries are detached and absent from the tape. This is the exact gradient of the
chosen mixed cached/live computation, not an unbiased all-fresh objective.

Reader dropout/noise is replayed by checkpointing where applicable; producer RNG
belongs to the producer tape. Native BF16 full-graph and replay accumulation can differ slightly when a shared
autocast weight cast collects contributions before conversion to FP32. The native
compaction diagnostic measured at most 0.24% per-parameter relative L2 difference;
disabling the autocast weight cache reduced the maximum absolute difference to
3.7e-9. This is not a claim of bit-identical BF16 accumulation across graph layouts.
The tape restores each producer's captured cache policy, even if the caller changes
it before replay. Default training cache policy is unchanged. A further shared-linear-layer regression
places capture and consumer forward inside the same BF16 autocast context. CPU and
CUDA losses/RNG match exactly; gradients match exactly with the cache disabled and
show at most 0.0009765625 absolute difference with it enabled in that operator test.
This is additional coverage of shared consumer paths, not another full-LFM parity
claim. See `experiments/operations-20260920/shared-autocast-replay.json`.

The first implementation does not replay nested
historical read dependencies, support higher-order gradients or sharded distributed
optimizers, or differentiate through historical tool actions. No such guarantees
are implied by the first-order tests.

## Training cache and stored-only evaluation

The legacy episode training cache stores the first payload produced for a stable
source ID.
A fully live run (`live_fraction: 1`) skips unused cache reads, writes and the
extra population forward. Mixed cached/live runs retain the original stale-cache
semantics and checkpointed recovery behavior.
A selected live read re-encodes the source with current parameters; an unselected
cached read truly uses the old serialized value. This intentionally exposes a
stale/live mixture. It is not a sophisticated generational refresh policy.

During frozen evaluation, an offline write phase encodes unseen support experiences
once into a new bank. That bank is reopened. The read phase consumes only the stored
payloads; a regression test makes writer calls raise during evaluation. Original
and counterfactual worlds have separate namespaces, preserving identical record
IDs and routing metadata without colliding in the store.

The configured precision (`bfloat16` by default) is used in the actual serialized
payloads. Compute may cast those values back to FP32 under autocast. JSON metadata
and safetensors payloads avoid arbitrary pickle loading. Only locally generated
optimizer checkpoints are loaded through `torch.load(weights_only=True)`.

## Routing and compaction

The exact CPU reference searches only records eligible under namespace, model
space/generation, authorization domain and time. Its bounded key chunks avoid
loading the whole key array into VRAM. It is a full scan, not ANN. The initial
learned router trains complete sufficient sets under a Plackett–Luce selection
objective, marginalized over valid orderings for small groups. This directly
rewards group starters even when isolated utility is zero. Single-utility ranking
is a separate utility that skips flat targets. Top-k itself is still discrete.

Temporary compaction supports whole/random/local/overlap groups. The synthetic
writer preserves total multiplicity and is shared across clusters. Contribution
loss matches each round's numerator/mass at raw-reader states and also a free
compact rollout. The paired objective computes raw and compact task losses on the
same noisy values, optionally adding detached-teacher KL; the original interleaved
objective remains available. Compactor-only optimization freezes all other state.

Local merge regularization and regrouping are integrated. Overlap splits both
contributions and mass with responsibilities summing to one per record. All shares
participate in the training read; independently retrieving only some shares would
require different semantics. Noise is not assumed to make independent facts
compressible by itself.

`ClusterBank` persists full-cluster codes and connects them to original keys.
Full selections use the stored code; partial selections fetch raw values. Reader
fingerprints and domain/time checks are enforced; deletions invalidate derivatives.
The inference API never calls the writer or compactor. Retaining fallback records
means this prototype does not establish net disk savings. Learned arbitrary-subset
or persistent overlapping-field decoding is not implemented.

The mutable bank also maintains leased rebuild jobs for invalid compact descendants.
Only parents whose compact children are current can be claimed, so recursive
compactions rebuild from the bottom up. Successful all-space journal publication
checks the live lease owner and captured child cursors, then clears the lease
atomically; failed or stale compactor inference releases it for retry.
`compaction_quality.py` separately gates promotion on held-out behavior and measured
serialized byte reduction. Passing contribution loss alone is insufficient.

Stored single-space codes also work at native recurrent read boundaries. Each
boundary fetches the complete cumulative selection, including code multiplicity,
before updating shared reader state; a partial cluster still uses raw fallback.
Per-boundary accounting records codes, fallback IDs and payload bytes. Runtime re-compaction and streaming in-loop reads remain unsupported.
Single-read, single-space recurrent training now supports interleaved temporary
mean or synthetic compaction: the complete selected group is replaced before the
shared reader update, with a differentiable storage-precision value cast and FP32
multiplicity. Uncompacted examples retain their task objective; compact examples
add conditional numerator/mass and rollout matching. Logs distinguish raw and
compact NLL. The paired objective uses one causal first-boundary query, selection
and noisy value set for separate raw/compact decoder paths, with optional detached
raw-teacher KL. Both cotangents reach the shared query graph before producer replay.
Multi-read compaction remains rejected. Offline creation of persistent codes is still separate from inference;
partial cluster selections retain exact raw fallback.

## Persistence and concurrency

SQLite rows are immutable physical revisions within namespace/ID/space/generation;
this is an MVCC/storage property, not a requirement that a logical record remain
unchanged. A read plan captures IDs and scores; values are rechecked for visibility
when fetched. Tombstones
prevent resurrection under the same logical ID, and deletion invalidates transitive
compact parents. Regeneration needs a new logical record ID after deletion.

Writes reserve the SQLite writer before checking tombstones or child visibility;
deletions reserve it before collecting the transitive lineage snapshot. This closes
check/insert and lineage/delete races between concurrent writers. Offline frozen-bank
builders stream records through one atomic transaction, preserving the same record
bytes while avoiding a durable commit for every payload. A failed batch rolls back;
readers never see a partial batch. Training-cache writes retain their existing
per-record publication. Bulk raw insertion does not replace the lineage-aware
compaction API. Large offline builds hold the writer reservation until completion,
so they should use their own bank rather than share an active write destination.

Authorization domains are caller-provided research boundaries, not user
authentication: a production service must bind caller identity to permitted domains.
Do not compact incompatible domains. Deletion is logical invalidation, not certified
physical erasure from WAL, backups or previous experiment artifacts.

AsyncRetriever provides a narrow CPU future API. Separately, the spatial curriculum
runner now pauses whole-sequence graphs at recurrent boundaries and overlaps several
CPU search and payload-read waves with GPU work. It uses deterministic bounded local
workers; remote request handling, failure injection and asynchronous inference remain
unimplemented.

## Run artifacts

A run records config, dependency/model revision, dataset content hash, metrics,
resource counters, a safetensors checkpoint and complete optimizer state. Resume
allows changing total step count and the explicit operational checkpoint policy;
scientific hyperparameter changes require a new warm-start fork. Changing episode content is rejected before
reusing stale cache data. Periodic immutable checkpoint sets include model, optimizer, RNG, config, and the
stale training-cache snapshot. An atomic CURRENT pointer commits the set. Resume
verifies hashes, restores that snapshot, and truncates later log fragments. The
full-copy SQLite snapshot remains a small-scale reference, not an incremental
distributed checkpoint system.

A mutable `model.revision: main` is logged as the actual resolved commit but should
be pinned before training. Different runs can otherwise fetch different initial
weights/tokenizers even though their YAML strings look identical.


## SCHNITZELJAGD 0.3 trajectory implementation

Current operations: [training](training.md), [datasets](datasets.md), [Spark](spark.md).
`launch.py` pins/prepares inputs, verifies the backbone, and runs independent resumable
stages and evaluations. `trajectories.py` provides explicit upstream adapters and
causal source/target construction. `episode_index.py` validates identities/hashes
while keeping byte offsets rather than all episode text. `trajectory_eval.py` separates
source materialization from writer-free stored reads and scores full target likelihood.
The command/package is `schnitz` and the model class is `SchnitzelAgent`.

Real-data supports are marked provided context, not verified sufficient groups. Loss
is recorded-assistant imitation, not environment reward or unavailable teacher logits.
The initial fixed-chunk producer is a bootstrap interface, not yet an adaptive learned
whole-trajectory lesson extractor. Stage evidence and exact tested hardware scope are
recorded in [validation-v0.3.md](validation-v0.3.md).


## Optional independent routing query

`memory.independent_routing_query: true` adds a second projection of the same
causal prefix state for address search. The existing query continues to condition
the reader; its behavior is preserved when only address parameters train. Stored
records still have one key per space and unchanged payload shapes. This is an
experimental addressing variant, not established general entity retrieval.

The default shares the original query and adds no parameters. A warm-start from
that default copies the loaded, trained query-head weights into the new head.
Routing-only optimization then includes the new head while freezing the reader
query, writer payload path and backbone. Resume retains the flag and named Muon
ownership. Prefix, multiread and loop-boundary training and stored sessions use
the separate routing key; target tokens remain excluded from both queries.
Full-graph/replay, stored-plan parity, frozen payload/oracle outputs and exact
Muon resume are covered by regressions.

The diagnostic evaluators also accept a separately trained, frozen reader-query
count classifier through `--read-count-policy`. It can request one or two records
within the configured search cap, using only the current causal reader query.
The policy is bound to its source checkpoint; keys and payloads are unchanged.
It is a single-space, single-read inference adapter, not a general learned stopping
policy or an added standard-training objective. Fixed plans and oracle controls
retain their declared selections. Default models have no count parameters.
Integrated/full-graph, replay and stored-session paths share the count decision;
the classifier itself is trained separately on frozen features and remains frozen.

### Resumable offline bank publication

`build_shared_bank(..., writer_identity=...)` accepts a caller-verified frozen
checkpoint identity and atomically commits raw records with a manifest of sources,
model/interface settings, precision and stored bytes. Repeated calls validate and
reuse those bytes without encoding sources. Derived code views do not change the
raw-bank digest. Unmanifested existing banks fail closed and need an explicit new
output path for offline rebuilding. This optional path is used by the standalone
compactor probe and confirmation builders; ordinary immutable inserts retain their
previous behavior. Compact-code markers, lineage and membership publish in one
writer transaction, including the disjointness and live-child checks.

Evaluation can also substitute only explicitly designated payloads from an offline
counterfactual bank. `TargetedPayloadStore` requires captured fixed plans, validates
original visibility first, and checks alternative visibility, source/time provenance,
shape and serialized precision before replacement. Every unselected or untargeted
record remains the original payload. Target IDs are intervention annotations used
only by the diagnostic, not model inputs, routing hints or authorization decisions.
