# Experiment protocol

See [v0.2 validation](validation-v0.2.md) for the executed CPU study and
[development guide](development-v0.2.md) for runner commands.

## 1. Establish the information path before retrieval engineering

Run the one-space, one-loop student with oracle support selection. Compare no
memory, oracle text, individually supplied latent records, a one-round MLP reader,
a multi-round MLP reader and a multi-round attention reader. Shared source/training
data access is required; compute, parameters and bytes should be matched in
separate comparisons when simultaneous equality is impossible.

The supplied retry/restoration generator has randomized environment names and
rules. It supplies a restoration rule A and permission rule B, with unrelated-world
distractors. The later task predicts STOP, RETRY or RESTORE_RETRY. Both records are
needed on applicable branches; restoration is irrelevant on a branch where retry
is forbidden. Test A only, B only, A+B, none, irrelevant, zeroed values, and rule
replacement with unchanged IDs/metadata. Increase binding complexity before
claiming general composition.

Every evaluation freezes weights, constructs truly new support worlds, serializes
the payload format/precision, reopens the bank and evaluates stored-only. A unit
test that passes through serialization is necessary but does not establish transfer.
Inspect whether predictions change appropriately under counterfactual replacements.
The original 30-step run did not show this. The v0.2 warm-started toy study shows
partial joint-use improvement, but counterfactual consistency remains insufficient
for a robust-composition claim.

Built-in choice scoring now uses total sequence NLL including EOS. Mean-token
NLL and mean-score predictions remain separately reported. This is not open-ended
generation correctness; `--generate` in the original action evaluator supplies an
additional greedy exact-match diagnostic. Unequal choice lengths/tokenizations can
still favor some answers; equal-length Boolean labels isolate that issue in the
CPU study. Do not compare v0.1 mean-scored action accuracy directly to v0.2 scores.

## 2. Separate write, storage and read bottlenecks

The writer emits a fixed canonical value before the future query is known. The
storage codec can further reduce that value. The reader condenses selected values
after seeing the query. Sweep these capacities separately. The wide-payload config
uses an identity storage codec; the direct-latent config bypasses pooling and sends
individual canonical records. Its decoder token count is larger and must be charged.

Train once, write an experience once, then test multiple distinct future uses:
procedure, exception, failure diagnosis and exact identifier. General support/query
JSONL allows the same immutable source ID to appear in multiple later episodes.
When importing teacher trajectories, keep the query answer and its future
observations out of its support. Split repositories/task families and remove
near-duplicates. Teacher API execution and output-license management are outside
the current runner.

For new held-out JSONL use `schnitz evaluate-transfer`, which writes each source once
in a shared bank and reports choice or target NLL, missing-support/value ablations,
and world-clustered paired metrics. The legacy `evaluate-episodes` remains available. Tool execution, code tests and
reward models must be added as real verifiers rather than renamed likelihoods.

## 3. Learned access

The learned configuration retrieves two records from ten, after oracle warmup.
The scorer receives gradients from known sufficient groups even when each member
is individually unhelpful. The current reference enumerates group orderings up to
size four; it is not an arbitrary group-discovery algorithm. Candidate exploration,
conditional marginal utility and learned invocation remain follow-on experiments.
Scheduled state-dependent multi-read queries and remaining-group supervision are
implemented; do not call a fixed schedule an adaptive stopping policy.

Track complete-support recall, accepting any actually sufficient group, plus
end-task success. The retry benchmark recognizes branch-specific sufficient groups; its oracle read
plan still supplies both rules so target-dependent cardinality cannot leak the answer. Do not treat perfect oracle recall as a
learned-router result. Current learned training is candidate-local; `evaluate-transfer` searches one
heterogeneous global namespace, exposing distractors outside the training episode.
Oracle recall is not evidence that this harder retrieval problem is solved.

## 4. Compaction and compactability

`compact-probe` fits a shared compactor against a fixed random reader on new tensor
clusters; it validates the optimizer/target path only. It is not a language-memory
result, and the random MLP/attention feature scales are not a fair reader ranking.

With a useful trained memory system, compare mean-plus-mass, synthetic records,
merge-consistency regularization, temporary compact reads, alternative grouping
views, and storage-aligned perturbations. Train on related records and test held-out
clusters/queries/states. Separately test independent facts and rare exceptions.
Leave uncompressible exceptions as explicit records rather than forcing uniform
compression. Multi-field responsibilities must conserve evidence mass.

Match conditional pre-normalization contributions and masses for both MLP and
attention readers. Measure downstream quality against both the original useful
model and the regularized model's own uncompacted path. Reducing useful behavior
until it becomes easy to compress is not a compaction improvement.

For a positive, statistically resolved memory benefit, report

```
retained_benefit = (quality_compact - quality_no_memory) /
                   (quality_raw - quality_no_memory)
```

Show absolute scores and uncertainty as well. Fourfold payload reduction with 90%
retained benefit is an illustrative point, not a universal stop/go condition.
Account for keys, overlap assignments, selection metadata, exceptions, shared
compactor parameters and original sources retained for training or regeneration.
Temporary replacement inside a reader does not yet save disk bytes or physical I/O.

## 5. Recurrence, multiscale retrieval and deployment

Compare loops independently of memory. One-loop versus two-loop results are not
compute-matched merely because parameters are shared. Test multiscale processing
against single-space controls at equal retrieved bytes, decoder slots and
appropriate compute budgets. Small oracle neighborhoods do not test the proposed
exponentially varying neighborhoods.

Early-layer query heads, pending-result states and useful loops during I/O need a
separate scheduler. Track end-to-end latency and throughput, not just sum of kernel
times. The current prefix-recomputing decoder and exact CPU scan are correctness
references, not competitive serving baselines.

## 6. Capacity substitution

Two protocols are distinct:

* Common-corpus storage: competing systems have the same information during allowed
  training or memory construction; compare different storage choices on new queries.
* New-experience deployment: all serious alternatives can access new supports,
  including larger models with strong text retrieval, and pay actual context,
  retrieval, encoding and residency costs.

A larger frozen model denied random post-freeze rules cannot be used as proof that
our memory replaced its parameter capacity. The initial comparison arms also retain
unused modules and use diagnostic evaluation passes; they are not yet stripped,
resource-optimal deployment baselines. Build those baselines explicitly for the
frontier experiment.

On Spark the unified physical-memory budget includes host caches and model state.
Measure peak CUDA allocation/reservation, process residency, host/page-cache use,
NVMe bytes, storage layout amplification, latency distributions and throughput at
a specified concurrency. Do not add GPU-visible total and system RAM as independent
capacity. Preserve cold/warm/cache-limited distinctions.
