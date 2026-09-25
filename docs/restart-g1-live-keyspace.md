# Restart from g1-live: keyspace pretraining and Phase 2 merge

**Status:** proposed recipe, 23 September 2026. Nothing here is trained yet.
It supersedes the R3 continuation chain (`spatial-r2-g1` through
`spatial-r3-livekey-v11`) and the head-only
[keyspace distillation plan](keyspace-distillation-plan.md) as the next training
path. Those runs and their artifacts stay on the external disk unchanged.

## 1. What we learned

Each finding is measured, and the evidence is recorded in
`/archive/runs/four-space-target-curriculum-v2-20260921/keyspace-warmup-20260923`.

1. **The key slot carries almost no retrievable content in R3.** Writer key-slot states
   of the R3 checkpoint have mean cosine 0.9988 with their common direction.
   Linear 64-d, linear 256-d and 2-layer MLP heads trained over the full
   100k field all stall near 2% held-out recall@16 (about 12% at 256). The same
   linear head over mean-pooled source-token states and a pooled 64-token causal
   query window reaches 64% recall@16, 89% at 256, and median rank 5. The
   information exists in the backbone. The key slot never learned to collect it.
   On `g1-live` itself (22k-source sampled field, 1,213 held-out level-1 sites,
   the same linear probe), the key slot reaches 7.1% recall@16 (median rank
   519), and pooled token states reach 61.9% (median rank 6). The slot holds a
   little content, but far less than the backbone does. On the same field, the
   stopped R3 checkpoint's slot is worse (2.7% recall@16, median rank 1,862),
   while its pooled states are about the same (58.5%). The R3 chain degraded
   the key slot, which confirms `g1-live` as the rollback point.
2. **BF16 key heads destroy ranking.** Over those collinear states, a BF16
   query scores all 100,000 stored keys with about five distinct values. Every
   autocast recall metric in R3 and in early warmup evaluations was either
   tie-inflated or meaningless. Key heads must run in fp32, and evaluation must
   rank ties pessimistically.
3. **Routing never had to work.** `g1-live` and its parent used oracle-selected
   reads for their whole stages (`routing_warmup` equal to steps). Its key loss
   only separated each episode's few local candidates. Every spatial stage
   supplied the verified supports to the reader. Nothing made global addressing
   load-bearing.
4. **Stored keys drifted.** The mutable bank mixed revisions from many writer
   states (median cosine near 0 against base keys). Head changes without
   coherent refresh make the index compare incomparable vectors.
5. **Full-field objectives need their own scale and conditioning.** A cosine
   ×10 softmax over 100k records cannot concentrate probability. Minibatch Adam
   on raw collinear states scrambles rankings before the loss moves.
   Standardized fp32 heads with a learned logit scale do learn.
6. **Teacher signal is strong and diverse.** On held-out training sites, a
   lexical index already gets 98% any-support recall@16 but only 63%
   all-support. Local dense teachers reach up to 80% all-support. Best per
   teacher (source view : query view):
   - EmbeddingGemma-300m, `chunk_neighbor:content`: 0.80
   - e5-base-v2, `chunk_neighbor:arguments`: 0.75
   - granite-english-r2, `chunk:content`: 0.73
   - LFM2.5-Embedding, `chunk:arguments`: 0.68

   Causal prefix windows are poor query views, because the prefix mostly holds
   earlier episodes.

## 2. Principles for the new recipe

- Train addressing before relying on it, and measure it with an unassisted,
  fp32, tie-pessimistic gate on source-disjoint validation sites.
- The writer must be trained to fill the key slot. A key head cannot recover
  content that the slot state does not contain.
- Key training must not silently change payload semantics. Payload-preserving
  structure or distillation is part of the recipe.
- Every evaluation and every training search uses one coherent key generation.
- Reads become load-bearing gradually: supplied supports are annealed away, so
  a wrong retrieval costs task loss.

## 3. Interface changes (merged with Phase 2)

Phase 2 ([positional memory v0.7](positional-memory-v0.7.md)) already requires a
versioned migration, regenerated trajectories and rebuilt banks. The keyspace
changes ride the same migration, so we pay that cost once.

| Component | g1-live (Phase 1) | Restart target |
| --- | --- | --- |
| Writer value slots | 8 | **64** |
| Returned read slots | 8 | **64** |
| Payload layout | flat `[256,512,1024,2048]` (47% of 8 × 1,024) | **joint full-width tokens**: `[4,8,16,36]` tokens × 1,024 per space, 65,536 scalars = 64 × 1,024 (1:1) |
| Codec | one dense `Linear(8·1024 → d_s)` per space | per-space joint cross-attention: P_s learned queries over all 64 slot states |
| Reader | pooled MLP set reader | positional block-operator reader, per-space source-token counts |
| Key slot position | first write slot, before values | **last** write slot, after the 32 value slots |
| Key heads | shared `key_head` + per-space maps, BF16 | per-space direct heads with bias, **fp32** |
| Query heads | shared `query_head` + per-space maps | per-space direct query heads with bias, fp32; reader keeps its own `query_head` |
| Key width | 64 in every space | **256 in every space** (section 5); key width is independent of payload width |
| Training logit scale | fixed | learned per space for full-field objectives; ranking and gates use raw cosine |

**Joint token layout (owner decision, 23 September 2026).** The writer's 64
slot states (64 × 1,024) are reshuffled rather than compressed. Every space is
a joint mapping of all 64 slots into a few full-width tokens, with counts
ascending so the four spaces sum to the source size:

| Space | Stored tokens | Scalars | BF16 bytes | Records read (16/8/4/4) | Source tokens per read |
| ---: | ---: | ---: | ---: | ---: | ---: |
| s0 | 4 | 4,096 | 8 KB | 16 | 64 |
| s1 | 8 | 8,192 | 16 KB | 8 | 64 |
| s2 | 16 | 16,384 | 32 KB | 4 | 64 |
| s3 | 36 | 36,864 | 72 KB | 4 | 144 |

A record is 128 KB (12.8 GB per 100k records), and a query reads about
670 KB of payload. A joint dense map at this size would need billions of
parameters. Learned-query cross-attention is joint and content-adaptive at
about 2M parameters per space, and it is independent of the writer slot count.
This supersedes v0.7's per-position `[32,64,128,256]`-channel codec, which
stored about 47% of the slot states. A record read in only some spaces
contributes just those spaces' shares, by design.

Moving the key slot after the value slots does two things:
- Under the causal writer, value states no longer depend on the key slot, so
  training the key slot's embedding and heads cannot change payloads.
- The key slot can attend to the source and to every value slot it summarizes.

## 4. Stages

### R0 — Root and diagnostics
Root: `g1-live` step 39,504, the bank's exact writer with healthy oracle-read
payloads. Record its slot-versus-pooled diagnostic, fp32 score resolution and
payload-dependence baseline.

### R1 — Joint-token distillation at 8 slots

**Result (23 September 2026):** `/archive/runs/phase2-restart-20260923/r1-joint-8`,
2,000 steps. Token distillation loss fell from 0.72 to 0.07 and downstream loss
from 1.26 to 0.09. On 128 source-disjoint validation episodes with oracle
supports, answer NLL was 2.944 with correct payloads (teacher 2.949). Swapping
in another episode's supports added 0.417 (teacher 0.429), zeroing added 0.267
(teacher 0.329), and correct beat swapped in 74.2% of episodes (teacher 74.2%).
Gate passed.

From `g1-live`, keep 8 writer and read slots. Freeze the copied writer and
backbone, and train the new joint codecs and the per-space operator reader.
Payload layouts differ from the flat teacher, so distillation matches reader
states, returned tokens and fixed-plan downstream outputs rather than raw
payloads. Gate: payload removal and replacement change outputs at least as much
as the flat teacher's do.

### R2 — Expand to 64 writer and read slots

**Result (23 September 2026):** `/archive/runs/phase2-restart-20260923/r2-joint-64`,
1,000 slice-frozen steps. Old-function preservation held (payload loss 0.04,
old-slot token loss 0.004). On the same 128 validation episodes, answer NLL
was 3.021 with correct payloads, the swap gap was +0.317 (74% of the teacher's),
the zero gap was +0.192, and correct beat swapped in 68.0% of episodes. The
decoder now receives 56 read tokens that the reader has not learned to use.
R5 must recover this and then exceed it; R3 does not depend on it.

Use slice-frozen expansion for writer slots, reader target positions and
recurrent workspaces. The joint codec's learned queries do not depend on the
slot count. R2 changes nothing on the key path. Rebuild trajectory packing
with 64 read slots.

### R3 — Keyspace pretraining (the new core stage)
R3 owns every key-path change. Its initialization:
- moves the key slot to the last write position (`key_slot_position: last`);
- converts the shared 64-d key path into 256-d direct fp32 heads with bias
  (`convert_to_direct` with widths). The first 64 rows are folded exactly, the
  added rows start small, and distance-gate inputs are widened with zeros.

Moving the key slot changes what value slots attend to. R3's
payload-preservation loss against frozen R2 payloads absorbs that shift.

Train the writer to fill the key slot, and train the query side to match it.
Trainable parameters are the key-slot embedding, the direct writer and query
key heads, and the recurrent core at a low learning rate (owner decision,
section 5). The core also produces value states, so a payload-preservation loss
against frozen R2 payloads keeps value semantics.

Per-step losses:
1. **Teacher alignment (per source, no bank field).** Live writer keys are
   aligned with each space's frozen projected teacher vector: cosine plus an
   in-batch similarity-distribution KL. Four teachers, each with a distinct
   source and query view per space (section 1, item 6).
2. **Query alignment.** Live query keys at causal `memory.search` sites are
   aligned with the projected teacher query vectors in the same space.
3. **Support contrast over the full field.** Verified supports against all
   eligible records, scored with a key cache recomputed from the live writer
   every N steps (a coherent generation, not a mutable mix), with a learned
   logit scale.
4. **Lexical auxiliary** at small weight.

Data: Hotpot v2's 100k sources and 39.5k training episodes, plus the approved
public retrieval corpora in the local HF cache (MS MARCO, NQ-open, TriviaQA,
SQuAD, SearchQA) for query and source diversity. Validation stays Hotpot's
source-disjoint gate.

R3 result so far (r3-keyspace-b), on 500 level-1 validation sites against a
field of 12,130 sources, recall at 16/8/4/4:

| Step | s0 | s1 | s2 | s3 | median best rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | about 4,000 |
| 500 | 0.26 | 0.19 | 0.13 | 0.14 | about 55 |
| 1000 | 0.56 | 0.44 | 0.33 | 0.33 | about 11 |
| 1500 | 0.71 | 0.61 | 0.52 | 0.52 | 4 |

At step 1500, recall@256 is about 0.95 in every space. The payload-preservation
loss stays at about 0.015.

Gate before R4, on the packed validation sites and a complete key refresh:
unassisted selected-support recall at 16/8/4/4 of at least 5% in every space
and at least 10% union (the plan's defaults). The stretch target is the pooled
probe's level (at least 50% recall@16 in s0).

### R4 — Coherent bank build
Build the 100k bank with the R3 writer: fp32 keys, positional BF16 payloads,
and one generation. Verify writer reproduction of stored keys, and report
reproduction noise across batch shapes.

### R5 — Spatial trajectory training with load-bearing reads
Train the spatial curriculum (2 loops, then 3) from R3/R4 with:
- supplied-support probability annealed from 1 to 0;
- unassisted reads always searched in fp32;
- writer replay with a nonzero key learning rate;
- a measured coherent-refresh cadence;
- the teacher-alignment loss kept at a low weight.

Evaluate the gate every 50 steps, plus stored-only transfer and payload
interventions.

R5 read limits follow the constant read-width rule (architecture section 3.2):
limit × tokens per record is the same in every space. With stored tokens
`[4, 8, 16, 36]`, a budget of C tokens per space gives limits of about
`[C/4, C/8, C/16, C/36]`. The R3 gate cutoff of 16/8/4/4 corresponds to C = 64,
with s3 rounded up from about 2. That budget is too small: the widest space
reads so few records that unassisted positives are rarely delivered, which
starved the earlier learned gates of any signal. C is chosen from the R3
read-count recall curve (any and every support within k records, per space;
logged from step 2500) together with the reader's measured cost per token.

#### R5 first launch: writer drift against a 218k-record bank (24 September 2026)
The first R5 launch (`r5-spatial`, `restart_r5_joint_spark.yaml`) trained the
recurrent core (Muon, 5e-6), the key-slot embedding (1e-4) and the key heads
(1e-5), with writer replay re-encoding about 64 records per step. Within 120 steps
the live keys of replayed records fell to about 0.9 cosine against their stored
keys (key-stability loss 0.10–0.14). Queries followed the live keys: routing loss
against live keys fell from 6.6 to about 3–5 while loss against stored keys rose
to 9–16. Unassisted item recall against the bank fell (s1 0.71 → 0.47, s3 0.64
→ 0.47), while answer NLL dropped quickly (1.55 → 0.85). This is finding 4 again:
per-step refresh of a few records cannot keep a 218k-record index coherent while
the writer moves.

R5a therefore freezes the writer and the bank
(`restart_r5a_frozen_writer_spark.yaml`: frozen backbone apart from the reading-side
loop bridge, no writer replay). The reader, gates, query heads and bridge learn
against a coherent R3 key space. Key-side learning (R5b) needs a coherent
full-bank refresh cadence (about 2.4 hours per 218k re-encode) or a writer
learning rate small enough that stored keys stay within tolerance between
refreshes. The key-stability loss is the drift gauge.

#### R5b: error-driven writer learning with coherent keys (owner design, 25 September 2026)
Owner direction: refresh through actual error flow rather than by lowering
learning rates, and save computation by routing the writer backward
selectively. Implemented in `src/sdkb/record_gradients.py`:

1. **Key-state cache.** Each record's writer key-slot state is cached
   (`scripts/cache_bank_key_states.py`, fp32, about 0.9 GB for 218k records).
   Stored keys are the direct heads applied to these states. Candidate keys are
   computed live, so the heads get exact gradients from every retrieved record.
   After every optimizer step the whole index is recomputed, so head updates
   never leave keys stale.
2. **Decayed per-record gradients.** Consumer reads use leaf tensors for cached
   states and stored payloads. Their cotangents accumulate per record, with
   per-step decay and a last-updated step. The writer backward runs for a
   generous close neighbourhood of each step's reads (a quarter of each read
   limit, plus gold), using everything accumulated for those records. A small
   extra budget flushes the largest accumulated gradients outside the
   neighbourhood, so large signals do not just decay away.
3. **Rolling refresh.** A forward-only rolling refresh re-encodes the stalest
   records each step. Each refresh reports the cosine between a record's old and
   new keys, which serves as the drift gauge.

Accumulated cotangents are applied at the writer's current parameters, so older
contributions are delayed gradients: a deliberate estimator, not the full-graph
replay reference. With one step of cotangents the writer backward equals direct
backpropagation (tested). Key search can run exactly on the device
(`--device-search`, on by default with record gradients) with the same
eligibility and scores as the NumPy reference; an ANN index is deferred until the
bank reaches millions of records and keys change more slowly.
Config: `configs/restart_r5b_record_gradients_spark.yaml`.

**R5b result (stopped at step 63).** At the R5 learning rates, the key a record
would get from the current writer moved fast. By the record's age in steps, the
cosine between its cached key and its current key was:

| Age (steps) | Mean cosine | Minimum cosine |
| ---: | ---: | ---: |
| 10 | .995 | .986 |
| 20 | .964 | .844 |
| 40 | .865 | .573 |
| 60 | .836 | .468 |

At about 550 refreshes per step, the average record in a 218k bank would be
about 400 steps old. Every key was a function of the shared, training decoder,
so no affordable refresh rate kept the index coherent. The profile at about
40 s per step:

| Phase | Time |
| --- | ---: |
| Forward | 3 s |
| Consumer backward | 4 s |
| Writer backward, 288 records | 12 s |
| Refresh, 544 records | 21 s |

#### R5c: terminal per-record keys (owner design, 25 September 2026)
Keys are treated like an embedding table, not as a decoder output that is
backpropagated into. Implemented as `KeyTable` and `writer_pass` in
`src/sdkb/record_gradients.py`:

1. **Terminal keys.** Search and gates use one trainable unit key per record and
   space, initialized from the published R4 keys. Retrieval, gold-contrast and
   spreading gradients update only the rows touched in a step, with a per-row
   Adam and a last-updated step. The whole table is written to the index after
   every step. Decoder training never moves a key, so the index is exact on the
   full bank.
2. **The decoder learns to produce the keys.** Whenever a record is encoded
   (a gradient flush or a stalest-first refresh), the decoder's predicted keys
   regress onto the table rows (`key_prediction_weight`). New records at
   inference still take their keys from the decoder.
3. **Commitment.** Each encode moves the record's rows a fixed fraction toward
   the prediction (`key_commitment_weight`, 0.05). The move is an
   interpolation, not a gradient, so its size does not depend on the adaptive
   step. It keeps rows reproducible from content.
4. **Payloads as in R5b.** Payload cotangents accumulate and flush. Each encode
   carries its record's payload cotangents and key loss in a single backward,
   before the optimizer step, and republishes the payloads, which are therefore
   one step old. Flushed records no longer need a second forward pass.

Gauges:
- `*_key_prediction_cosine_*`: how well the decoder reproduces the table.
- `*_value_drift_cosine_*`: s0 payload drift since the last publish.
- `table_step_cosine_*`: how far touched rows move per step.

Owner direction: if values drift too far on the full bank, shrink the bank. A
smaller bank then needs ramp stages back to the target size.
Config: `configs/restart_r5c_key_table_spark.yaml`.

#### Moving from teacher distillation to utility-driven keys (owner decision, 24 September 2026)
Teacher distillation is a bootstrap only. The spaces the system needs will
differ from the reference embedding spaces, so distillation is removed as early
as possible. The answer loss is the main teacher: continuous distance gates
carry its gradient into queries and stored keys, and writer replay carries it
into the writer (adaptive-memory v0.6 §2, mutable-bank v0.8 §4).

- **R3 stops at its first clear plateau.** It does not run to its nominal step
  count.
- **R5 carries no teacher-distillation term.** R3 is the whole bootstrap, which
  is the earliest possible phase-out. If unassisted recall at the limits
  collapses (falls more than a set margin below its R3 value), fall back to a
  short, decaying distillation term rather than restarting R3.
- **Gold retrieval is kept wherever a dataset provides source documents.** A
  permanent gold-support contrast loss (verified supports against the full
  eligible field) remains at a moderate weight in every phase.
- **Forced gold retrieval gets gradient flowing.** With some probability per read
  site, verified supports that were not retrieved replace the lowest-ranked
  retrieved records, so reads keep their size. The probability starts high and
  anneals to a small floor instead of zero, so the gates always see gold records
  and their keys keep receiving utility gradient. Telemetry separates supplied
  from unassisted positives.
- **Answer targets are restored** for the public corpora that have them
  (MS MARCO, SQuAD, TriviaQA and SearchQA answers), as answer-NLL targets only,
  never in query text. This makes the utility signal denser than Hotpot's alone.

Implemented in `scripts/train_spatial_bank.py` and `src/sdkb/spatial_training.py`:
`--gold-force START FLOOR ANNEAL_STEPS` sets the linear forced-gold schedule. Each
read site takes one draw, reproducible from the episode, call, position and step,
and the draw is shared across spaces. Forced supports displace the lowest-ranked
non-support records. `learned_positive_*` metrics count unassisted retrieval
only. `gold_forced_fraction` reports the draw rate, and `gate_support_share`
against `gate_uniform_share` reports gate selectivity. The gold contrast (the
routing loss over the retrieved field plus the supports) is unchanged and always
on. Public-corpus answer targets need spatial trajectories built from those
corpora; that is still to do.

#### Gate fixes so injected gold receives utility gradient (owner decision, 24 September 2026)
The adaptive distance gate weights a record by `sigmoid((cos - radius) / T)`.
The radius sits at the k-th nearest candidate (k = 8) plus a bounded learned
adjustment. Three properties kept utility gradient away from distant gold:

1. The support floor was `max(w, floor)`, so a floored forced record passed no
   gradient to its key or query.
2. With k = 8 and reads of up to 256 records, almost every record read sits
   below the radius.
3. The key gradient through the gate is `w(1 - w)/T`, which dies exponentially
   with distance at the minimum temperature of 0.02.

Distant gold was reached only through the gold contrast loss (a label signal),
never through task utility. Fixes, all read-side settings outside bank identity:

- `gate_floor_mode: additive`: `floor + (1 - floor) · w`, which keeps the
  gradient everywhere. The legacy default stays `max`.
- `gate_density_fraction: 0.25`: the radius sits at `max(k, 0.25 × records
  read)`, which is 64/32/16/8 at the R5 limits. The gate computes this itself, so
  inference and training agree.
- `gate_min_temperature: 0.05`. If forced records still starve, the fallback is
  a heavy-tailed gate.
- Logged per space for forced gold, retrieved gold and other records:
  `gate_weight_*` and `gate_slope_*` (the dw/ds that reaches keys and queries).

#### Even spatial distribution (owner decision, 24 September 2026)
Permanent regularizers at small weights, kept through every R5 phase and
especially after the teachers are removed:

1. **Dimensional collapse.** VICReg-style variance and covariance terms on the
   keys and queries of each space's materialized field. The earlier failure (a
   mean-direction cosine of 0.9988) was this kind.
2. **Local clumping.** A KoLeo nearest-neighbour term on field keys, which evens
   out the local density that top-k retrieval sees.
3. **Rich-get-richer.** A running retrieval count per record, a load penalty on
   records far above their expected load (as in mixture-of-experts load
   balancing), and exploration slots in each read for rarely retrieved records,
   together with maintenance re-encoding. The density-relative gates already keep
   dense regions from automatically winning gate mass.

Perfect uniformity is random keys, so recall and teacher agreement decide the
weights. These act within a space; they do not optimize differences between
spaces. Cross-space convergence without teachers is an open risk: measure
cross-space neighbourhood overlap and per-space drop ablations, and if the spaces
merge, keep a very small per-space teacher anchor.

Implemented (`src/sdkb/key_geometry.py`, wired into the spatial trainer):
`--spread-variance-weight` and `--spread-covariance-weight` act on each wave's
query addresses. `--koleo-weight` acts on each read's candidate keys and reaches
live replayed keys. `--load-penalty-weight`, `--load-decay` and
`--load-threshold` configure a per-space retrieval-count tracker; retrievals
recorded during a step are committed after it, and the tracker is checkpointed
beside each model checkpoint. `--exploration-fraction` replaces that share of
each read's tail with eligible records drawn in inverse proportion to their
load. Stored bank keys are detached, so key-side spreading reaches only
replayed records; query-side spreading reaches every read.

Metrics, per space: effective rank of keys and queries (entropy of singular
values), mean-direction cosine, the uniformity statistic, hubness (skew of top-k
occurrence counts), the Gini of retrieval counts, and the fraction of the bank
never retrieved. Baseline them on the R3 keys at the R4 bank build.

Owner decision, 24 September 2026: the limits are **256/128/64/32** for s0–s3,
which is C of about 1,024 tokens per space (s3 reads 1,152) and about 4,200
stored tokens per read site. The recall curve and the reader's cost per token
confirm that this is affordable in R5; they do not reopen the choice.

## 5. Decisions

Owner decisions, 23 September 2026:

- **Writer adaptation in R3:** the recurrent core trains at a low learning rate
  together with the key-slot embedding and heads. A payload-preservation loss
  against frozen R2 payloads (serialized precision) keeps value semantics.
  Payload removal and replacement checks run at the R3 gate.
- **Public retrieval corpora:** approved, including their licenses, for R3 query
  and source diversity. The local cache has MS MARCO, NQ-open, TriviaQA, SQuAD
  and SearchQA. Each corpus keeps its own source namespace, so causal and
  authorization boundaries stay per corpus.
- **Order:** the Phase 2 migration (R1–R2) comes before R3.

- **Key width: 256 in every space.** Changing the width later means retraining
  the heads and rebuilding the bank, so the generous choice is made now. The
  source-disjoint gate still decides whether the wider keys generalize.

Key-width evidence, measured on 1,750 held-out Hotpot training sites, by
projecting a frozen teacher to width `d` and measuring all-support recall@16
(raw 768-d teacher in parentheses):

| Teacher | 32 | 64 | 128 | 256 | raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbeddingGemma `chunk_neighbor:content` | 0.44 | 0.61 | 0.72 | 0.75 | (0.80) |
| e5-base-v2 `chunk_neighbor:arguments` | 0.43 | 0.60 | 0.69 | 0.73 | (0.75) |

Going from 64 to 128 recovers about ten points; 128 to 256 adds about four.
Cost per record: an fp32 key is 256 bytes at 64 and 512 bytes at 128, against
2–16 KB of Phase 2 payload per space. Exact search over 100k keys in a space is
26 MB at 64 and 51 MB at 128. A backlog caution: on an earlier synthetic
fixture, wider addresses improved training fit but reduced held-out pair
retrieval, so the width choice is confirmed on the source-disjoint gate. At
256, fp32 keys take 4 KB per record across four spaces (against about 30 KB of
Phase 2 payload), and the resident exact index takes about 410 MB per 100k
records (about 4 GB per million).

## 6. Carried-over tooling

Built and tested in this session (not yet committed):
- the direct fp32 per-space key interface;
- state caching and teacher embedding scripts;
- the view benchmark;
- the unassisted gate;
- vectorized full-field losses, pessimistic ranks and standardized heads;
- the trainer's host-memory floor.

The R3 stage needs a new live-writer trainer. The cached head-only warmup is
kept as the frozen-state control.
