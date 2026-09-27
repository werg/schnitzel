# Research status and the remaining claim

> **Phase 2 interface update, 22 September 2026:**
> `positional-memory-v0.7.md` supersedes the dense flattened-codec proposal.
> Position-preserving codecs, MLP operator readers, and an operator compactor are
> implemented as opt-in primitives. Eight-slot teacher distillation, slice-frozen
> 8-to-32 expansion, and their recovery paths have executable runners and CPU
> tests. A pinned LFM2.5-230M numerical preflight on the GB10 passed with exact
> one-loop identity, exact causal-prefix invariance, and nonzero gradients in all
> four positional codec/reader branches. Sustained distillation, task training,
> and persistent recursive compaction remain pending and are not included in the
> capability evidence below.

Updated 20 September 2026. This is a synthesis of completed evidence and current
work. Individual experiment records retain their original protocols and results.
The [architecture](architecture.md) remains the research objective; implementation
and tests alone do not establish that objective.

## Where the project stands

The native pretrained model can use stored latent records to combine two simple
rules into correct actions on new synthetic worlds, when the relevant records
are supplied. Training can also make that behavior survive a mean-plus-mass
compact code. These are useful scoped results. Reliable learned addressing,
exact unseen facts, real agent execution and reduced resident-model requirements
remain open.

| Part of the idea | Executed evidence | Present limit |
|---|---|---|
| Stored-only inference | Frozen evaluations serialize/reopen banks and forbid writer calls during reads. | Offline bank construction still uses a frozen writer; it is a separate operation. |
| Combining separate evidence | The useful MLP source gets 128/128 fresh oracle-supported actions, correct changed-rule pairs and invariant branches. | This is a small authored permission/restoration distribution with oracle selection. |
| Detailed transferable memories | Selected source text copies 64/64 unseen endpoints; the same information through latent payloads yields 0/64. | More worlds, alternate histories and fixed questions have not established exact recall at their tested budgets. |
| Learning versus memorizing | A focused continuation fits all 64 training endpoints through memory. | It fails all replaced and fresh endpoints and degrades rule/action behavior; it is not promoted. |
| Learned global addressing | Corrected keys generate 102/128 actions at 128 records and 78/128 at 4,096 on the measured questions. | Retrieval still misses relevant entities; an extra sparse lexical index gets 128/128 on this literal-name fixture. |
| A compact dense lexical control | Equal 64D key blobs and three fixed seeds were tested, with all results retained. | Full action-pair retrieval is 2/0/0 out of 128; a prior-corpus centering follow-up is 0/0/0. Sparse lexical success is not an established equal-byte dense alternative. |
| Learned compactability | Temporary training lets mean and learned codes retain all tested 128 action answers and rule-change pairs; a raw-trained mean control gets 81/128. | Learned codes do not outperform mean-plus-mass. The raw path can regress; pairing objectives has mixed results. |
| Persistent compaction | Full-cluster codes, mass accounting, visibility and partial-selection raw fallback are implemented and exercised. | Raw fallback records remain stored, so these runs do not establish net disk savings or arbitrary-subset coding. |
| Capacity substitution | The real native LFM student, memory interface and recurrent compute run on Spark. | No matched quality–resident-memory–latency comparison shows that external memory replaces a larger resident model. |
| Agent usefulness | Teacher-data preparation, causal-prefix protocols and imitation evaluation are implemented. | Source commands are inert data. Teacher NLL and authored action strings are not measured tool execution or agent success. |

Sources: [fresh source controls](../experiments/binding-endpoint-freshness-20260920/README.md),
[exact detail](../experiments/binding-exact-detail-20260920/README.md),
[copy fit](../experiments/binding-copy-fit-20260920/README.md),
[routing precision](../experiments/binding-routing-precision-20260919/README.md),
[lexical generation](../experiments/binding-lexical-routing-20260920/README.md),
[dense lexical controls](../experiments/binding-dense-lexical-20260920/README.md),
[centered controls](../experiments/binding-centered-lexical-20260920/README.md),
[temporary compaction](../experiments/binding-compact-aware-20260920/README.md),
[paired compaction](../experiments/binding-paired-compaction-20260920/README.md).

## What the latest completed experiment shows

The [endpoint freshness study](../experiments/binding-endpoint-freshness-20260920/README.md)
completed two 4,000-update Muon continuations from the useful original MLP source.
Both consumed the same task/source-count/recurrent-depth schedule and retained rule
and action tasks. One repeatedly sampled 512 endpoint strings; the other sampled
7,516 distinct strings, mostly once. World content also differed, and equal updates
were not equal token or FLOP budgets. Both endpoint states pass the complete
sampler/depth and checkpoint audit.

Both models still generate **0/64 original held-out endpoints**, while retaining
128/128 original actions and 64/64 selected-text endpoints. The larger arm gets
one changed endpoint right, but neither gets any correct original/changed endpoint
pairs. It also has a few rule-intervention regressions. The small arm fits 45/64
common training-corpus endpoints, then fails all 64 replacements and drops to
26/64 after an irrelevant permission change. Neither is promoted for exact recall.

There is partial progress: identifier teacher NLL falls from 5.1163 in the source
to 3.3872/1.7493. Post-hoc original-generation character agreement is 102/384 and
125/384 versus 29/384 and 35/384 with zero payloads, concentrated in early positions.
The small arm emits old training targets in 53/64 cases; the larger does so only
once. More fresh targets reduces this old-answer reuse pattern at this budget,
but has not produced reliable copying.

The [fixed-query readout follow-up](../experiments/binding-freshness-readout-20260920/README.md)
completed all 18 declared heads. The larger-corpus writer's payload lets a linear
six-position classifier recover 3/64 exact identifiers and 251/384 characters;
the reader output yields 0/64 and 156/384. Zero-value features are exactly constant
under these identical queries and yield 29/384 characters. These supervised heads
supply the output format and have different parameter budgets. They locate a gap
in tested accessibility, not an information-theoretic loss or a language-generation
success. The completed [intermediate-state follow-up](../experiments/binding-reader-stages-20260920/README.md) yields 205/384 characters after input projections and 165/384 in the final shared state with linear heads, with no exact identifiers. The gap precedes the final output transformation; the [reader-capacity intervention](../experiments/binding-reader-capacity-20260920/README.md) completed matched reset 256- and 1024-wide readers from the same writer/controller. Original identifiers are 1/64 and 0/64, with zero correct identifier counterfactual pairs in both arms; selected text remains 64/64. Widening is not promoted as a recall solution.

## What would support the stronger idea

These are requirements for future evidence, not implemented or completed results:

1. **Useful transferred content:** repeat exact and semantic transfer on genuinely
   new experiences, with source-matched text, no/zero/permuted memory controls and
   targeted interventions. Separate retrieval errors from writer/reader errors.
2. **Selection and composition together:** preserve that behavior in larger banks
   with competing entities and permissions, without supplying world membership or
   treating learned selection as authorization. Report the actual selected budget.
3. **A measured compaction benefit:** retain behavior while reducing total stored
   bytes, including exceptions, raw fallback, indices and metadata. Give attention
   the same compaction opportunity. Full-cluster and subset-selective claims differ.
4. **A capacity/compute comparison:** compare smaller-memory-augmented and larger
   resident models under declared information, training and inference budgets.
   Measure the entire resident footprint and end-to-end latency. Warm-cache exact
   scans on this external rotational disk are not cold NVMe or ANN results.
5. **Actual agent outcomes:** evaluate tool execution in a bounded environment
   separately from teacher imitation and synthetic answer generation. No such
   execution success is asserted by the current dataset or binding studies.

The central gap is empirical: the machinery now executes several intended paths,
but the broad capacity-substitution claim still lacks these outcomes. The current
positive results should guide tests without being relabeled as that broader proof.

## Next target: trajectory-native memory

The owner has selected [trajectory memory v0.5](trajectory-memory-v0.5.md) as the
next architecture program, with its older immutable-generation lifecycle
superseded by [the mutable-bank v0.8 contract](mutable-bank-v0.8.md). Long agent traces should contain repeated visible
`memory.search` calls and tool results at distinct causal positions, plus distinct
`memory.write` calls whose site count scales with useful input and events. Batching
several items into one call does not meet that requirement. The evolving bank must
contain substantial contributions authored by trajectories that queried earlier
bank states, while later learning may update the keys, payloads, and compactions of
both old and new logical records.

Its training computation is whole-sequence and recurrent. Each pass processes all
teacher-forced positions in parallel; all query sites active at that level retrieve
together; and latent results are scattered into site-aligned blank workspaces before
the next pass. The first implementation now packs multiple structured search sites,
batches their global exact searches by space and level, fetches stored payloads, and
scatters them into their own workspaces. Current `read_steps` are depth boundaries
and must not be reported as trajectory site count. The packer also emits distinct
prompted write-call targets with read lineage. The growing-bank executor now runs
those fixed trajectories against one revisioned logical catalog and atomically
publishes authored writes, event lineage, dependencies, and the visibility frontier.
Its journal supports checkpoint cursor recovery, index updates, stale-record
maintenance sampling, compaction invalidation, and checkpoint-pinned revision GC.
Learned site placement, generated write arguments, size-band training,
utility-driven maintenance and a conforming network backend remain planned. Leased,
dependency-ordered compact-code rebuild workers and explicit behavior-plus-byte
promotion accounting are implemented but have not established compaction benefit.

This is a course extension, not a description of the current HotpotQA curriculum.
The current run performs one automatic latent read per short episode and writes
source chunks through the live/offline writer. It has no learned call placement,
no memory tool-call syntax, no prompted write policy, and no model-authored recursive
bank content. Its writer/reader/routing learning remains useful Phase 0 preparation.

The target capacity claim also changes the scale requirement. One hundred thousand
records is an integration tier. The program advances through 1M, 10M, and 100M
logical records, then requires scale-out storage for full pretraining/posttraining
corpora. At the current four-space format, array bytes alone are about 8.7 GB, 87 GB,
and 870 GB at those tiers. Payload width, returned slots, records per source, and
selected bytes are measured capacity axes rather than assumed sufficient constants.

## Operational state

The [bgkit audit](bgkit-audit.md) and [operations guide](operations.md) record
external placement, verified relocation/deduplication, sparse saves, emergency
microbatch recovery, complete named optimizer/RNG/config state, offline W&B,
resource reserves and cooperative run control. About 48 GiB was reclaimed from
the internal run tree, which is now about 52 MiB. Current study artifacts go directly
to the external drive.

Muon uses the installed NVIDIA Torch implementation for eligible matrix transforms,
with AdamW for embeddings, slots, fallback tokens and other excluded tensors. The
fallback-token ownership correction changes future optimizer groups; older runs
must resume with their frozen checkout or explicitly warm-start a new run. The
completed freshness study used frozen `c58028c`; completed intermediate readouts used `417c2f4`.

The latest download-free suite passes 501 tests; native pretrained execution and
a real BF16 Muon emergency-resume comparison have also run. CPU/CUDA configuration
and portable operating helpers are separate from the Spark wrapper. Tested native
Spark execution does not imply performance portability to every GPU or backbone.
See [current machine validation](validation-spark.md) for scope and artifacts.


The [selected-text alignment experiment](../experiments/binding-text-alignment-20260920/README.md)
completed its 800-update matched Muon comparison. Both arms recover 0/64 identifiers
in every latent condition and 64/64 from selected text. Alignment slightly reduces
teacher NLL but introduces three false action changes under irrelevant identifier
changes. It is not promoted; the predeclared fresh-corpus gate failed and those
worlds remain unevaluated. Native BF16 partial-microbatch recovery still matches
complete weights, optimizer and RNG exactly. The next readiness check uses causal
public trajectories and first measures the pretrained text path's context utility.


The [public-trajectory readiness check](../experiments/trajectory-readiness-20260920/README.md)
uses all 119 repository-held-out episodes from a pinned, bounded SWE-smith sample.
Untouched pretrained one-pass LFM benefits from prior text: token-weighted NLL
1.490 versus 1.937 without it. This establishes a usable conditioning signal for
the [declared recurrent-bridge curriculum](../experiments/trajectory-prefix-muon-20260920/README.md),
not latent-memory utility, cross-experience composition or agent success.

## Completed real-trajectory pilot

The [native Muon trajectory pilot](../experiments/trajectory-prefix-muon-20260920/README.md)
completed a bridge, latent warmup and 400-update recurrent joint stage on the
Spark. It operates on frozen-bank causal prefixes from 530 training and 119
repository-held-out episodes, with full recovery checkpoints directly on the
external disk. A later matched uniform-data control scored held-out R=2 teacher
NLL at 0.9710 with real stored values, 0.9807 with zeroed values and 1.0896
without memory. The selected-text R=1 path scored 0.8666, under a different
token and compute budget. Thus most of the memory-versus-none improvement is
also present after value ablation; the incremental payload signal is small.

Three matched or bounded interventions have not increased the average
payload-specific held-out effect: [selected-text output KL](../experiments/trajectory-output-distillation-20260920/README.md)
improved real NLL but improved zero-value NLL more; a [training-only high-text-utility
curriculum](../experiments/trajectory-context-curriculum-20260920/README.md)
worsened real NLL; and a [raised initial memory gate](../experiments/trajectory-memory-gate-20260920/README.md)
did not preserve its read-only overlay's slight improvement after 400 updates.
Reader values vary substantially under payload ablation, but their effect is
much smaller at the answer state and next-token distribution. The bridge trace
and gate overlays are exploratory on the reused validation set, not proof of a
specific bottleneck. A [full training-set stored-bank diagnostic](../experiments/trajectory-train-fit-20260920/README.md)
is complete. Its real-versus-zero token-weighted NLL gain is 0.01017 on training
versus 0.00969 heldout, despite real-value NLL of 0.6478 versus 0.9710. This
does not support a large training-only payload effect. Unequal repositories and
target lengths limit the direct split comparison.

These results establish a working native causal-prefix and external-storage
research pipeline and a modest teacher-likelihood payload effect. They do not
establish content-specific generation, source counterfactual success, autonomous
agent outcomes or resident-capacity substitution. Further development should
test source-dependent behavior directly, with a fresh evaluation split for any
new objective chosen after these exploratory probes.

A [fresh trajectory slice](../experiments/trajectory-fresh-holdout-20260920/README.md)
is now prepared from later rows of the same pinned public stream. It excludes all
89 repository groups in the original sample and seals 342 as-yet-unscored episodes
across 30 new groups for evaluation only. This supplies a repository-disjoint
confirmation set for a predeclared next intervention; it does not itself address
source-dependent behavior or establish broader dataset independence.

## Chronology of the binding studies

*Moved from the README on 27 September 2026; the text is unchanged apart from
link paths. It was last updated around 20 September.*

**Version 0.4.0 · Student `LiquidAI/LFM2.5-230M` · Initial GPU validation: DGX Spark.**
The Python runtime supports CPU/CUDA; Spark-specific checks live in the container wrapper.
The Python package, CLI, agent class, scripts, container and active documentation now
use SCHNITZELJAGD (`schnitz`, `SchnitzelAgent`). Historical experiment logs remain immutable evidence.

[Research status](research-status.md) summarizes what is established and what
remains open, including the completed reader-capacity and real-trajectory studies.
The next target combines the [trajectory memory v0.5 plan](trajectory-memory-v0.5.md)
with the [mutable-bank v0.8 contract](mutable-bank-v0.8.md):
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
archives are documented in [portable training operations](operations.md).
See [current Spark validation](validation-spark.md) for tested behavior and remaining limits.
The [Muon binding study](../experiments/binding-muon-20260919/README.md) and
[longer MLP continuation](../experiments/binding-continuation-20260919/README.md) confirm
narrow stored-memory action composition with both attention and the pooled MLP.
Reliable competing-entity composition and exact unseen identifiers remain unresolved;
the first [learned routing intervention](../experiments/binding-routing-20260919/README.md)
learned source types but did not solve entity selection.
Later [global retrieval tests](../experiments/binding-global-routing-20260919/README.md)
exposed the dependence on supplied world membership. Later
[cross-world training](../experiments/binding-global-stored-long-20260919/README.md)
reaches 101/128 generated actions on a fresh 128-record bank, but
[adding distractors](../experiments/binding-bank-scale-20260919/README.md) reduces this
to 76/128 at 4,096 records, with uncertain benefit over no memory.
[Corrected scoring precision](../experiments/binding-routing-precision-20260919/README.md)
preserves this pattern (102/128 and 78/128). An
[exact-detail control](../experiments/binding-exact-detail-20260920/README.md)
copies 64/64 unseen identifiers from selected text but 0/64 from latent payloads;
perfect retrieval alone does not solve this representation/training failure.
The [matched breadth](../experiments/binding-detail-breadth-20260920/README.md) and
[alternate-history](../experiments/binding-endpoint-views-20260920/README.md) continuations
preserve oracle action composition but still give 0/64 exact identifiers.
[Temporary-compaction training](../experiments/binding-compact-aware-20260920/README.md)
now makes a single mean-plus-mass MLP code preserve 128/128 actions and the tested
rule-change pairs; its matched raw-trained mean-code control gets 81/128. The raw
path regresses to 126/128. The completed [paired-objective comparison](../experiments/binding-paired-compaction-20260920/README.md)
has mixed raw-path results and no compact advantage over interleaving. Raw fallback
records remain stored; net disk savings are not established.
A [lexical selection control](../experiments/binding-lexical-routing-20260920/README.md)
recovers 128/128 stored-memory actions at 4,096 records using an additional source-token
index; this exposes the literal-name fixture and does not improve learned keys.
A [focused copy diagnostic](../experiments/binding-copy-fit-20260920/README.md) fits
64/64 training identifiers but fails changed and held-out endpoints, with rule/action
regressions. The completed [endpoint freshness study](../experiments/binding-endpoint-freshness-20260920/README.md)
reduces old-target reuse with broader training but still generates 0/64 original
held-out endpoints. [Frozen readouts](../experiments/binding-reader-stages-20260920/README.md)
find weaker character access through the reader than in its input payloads. The
[reader-capacity study](../experiments/binding-reader-capacity-20260920/README.md) is
now training matched reset readers of two widths; its capability results are pending.
The [bgkit audit](bgkit-audit.md) records storage, recovery and runtime adoption.
