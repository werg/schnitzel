# SCHNITZELJAGD recurrent decoder conversion

> **v0.5 spatial extension:** the legacy path described below has one workspace and
> one query position. The first spatial path now adds many site-aligned blank
> workspaces to the same whole-sequence recurrent timing. After each core pass, all
> active site queries are gathered together; their stored results are scattered into
> their respective positions before the next pass. Spatial site count and recurrent
> depth are separate axes and must not be inferred from `read_steps`.

The implemented batched boundary representation has query positions
$P\in\mathbb N^{B\times K}$, workspace starts
$S\in\mathbb N^{B\times K}$, an active mask for each recurrence level, and returned
tokens $Z\in\mathbb R^{B\times K\times m\times d}$. The boundary gathers every
active $H_{b,P_{b,k}}$, flattens active sites for batched search and reader work,
then scatters each $Z_{b,k}$ into its non-overlapping span beginning at $S_{b,k}$.
Every span follows its query position, and the causal mask controls which later
positions can consume it. Complete all sites' neighborhood aggregates before the
next shared-state update.

The initial provider uses resident contiguous exact key arrays and immutable SQLite
payloads. It batches searches by space and recurrent level. The spatial curriculum
runner can retain a bounded number of differentiable microbatch boundary states,
issue their CPU search and SQLite reads on worker threads, and continue the states in
deterministic round-robin order as results become available. This overlaps local
retrieval waves with useful GPU work; it is not yet a remote database latency
measurement or a general asynchronous inference scheduler.

Autoregressive post-training may use a faster schedule: after a generated call has a
result, inject its workspace after the fixed prelude before the recurrent core. This
is a distinct measured schedule, not evidence that blank-first-pass training and
immediate-result inference are automatically equivalent.

**Decision and implementation note · 19 September 2026 · SCHNITZELJAGD 0.4**

## Decision

Make a **prelude / shared recurrent core / coda** decoder the main SCHNITZELJAGD experiment.
Retain LFM2.5-230M as the initial student, retain every pretrained layer, and initially
repeat the middle six layers. Read memory at actual boundaries between core passes.
Keep the writer at one pass while varying consumer depth. This is an implemented
research conversion, not a claim that LFM was originally trained recurrently or
that extra depth has already improved its capabilities.

The v0.3 repository already contained an optional, weight-shared full-stack adapter.
Describing it as “not looped at all” was inaccurate. It did not, however, offer the
native middle-block conversion, conversion curriculum, or inter-pass retrieval now
implemented. Its default was one pass and its zero-initialized update gate preserved
the initial function at all depths. The gate itself had a gradient; it was not a
mathematically dead gate. That adapter remains a historical/control option.

There are two different transformations worth distinguishing:

1. **Extend effective depth using the existing weights.** Keep the original blocks,
   repeat a subset, add a small bridge. This is our first conversion.
2. **Reduce the number of unique blocks by tying or merging formerly different
   layers.** This removes stored parameter capacity and generally needs recovery
   training. It is a later parameter-reduction experiment, not a prerequisite.

The first transformation lets us investigate repeated interpretation of external
knowledge without simultaneously throwing away part of the pretrained controller.
The SCHNITZELJAGD capacity-substitution claim must still be measured separately.

## 1. What the current literature supports

The relevant distinction is evidence about **converting pretrained models**, versus
successful **pretraining of recurrent models from scratch**. Both inform design;
they do not establish that the same training budget or normalization trick transfers.

| Work | Relevant finding or mechanism | Use in SCHNITZELJAGD |
|---|---|---|
| McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*, November 2025 [1] | Direct conversions of TinyLlama, OLMo and Llama; prelude/core/coda, persistent prelude reinjection, recurrent-depth curriculum and recovery training. | Main methodological precedent. Adopt reinjection and gradual depth; do not copy its pruning or large-depth training schedule into the first 230M hybrid experiment. |
| *Retrofitting Recurrent Depth into a Pretrained Language Model*, August 2026 [2] | A keep-all-layers Qwen retrofit with a one-pass identity path; bridge installation, retention and shared-adapter experiments. | Supports treating identity, bridge trainability and retention as distinct tests. This is a recent, narrow study, not established LFM validation. |
| *Relaxed Recursive Transformers*, ICLR 2025 [3] | Parameter sharing supplemented with layer-specific low-rank residuals and recovery/distillation. | Useful later for reducing unique resident layers. Per-depth specialization is a different trade from a uniformly shared recurrent operator. |
| Huginn / *Scaling by Thinking in Continuous Space* [4] | Recurrent-depth pretraining with prelude anchoring, sampled depth and explicit gradient/cache design. | Architectural reference; its pretrained model and pretraining budget are not the proposed 230M checkpoint conversion. |
| Ouro / *Scaling Latent Reasoning via Looped Language Models* [5] | Large-scale recurrent pretraining, adaptive depth, and controlled comparisons of knowledge storage versus manipulation. | Supports the motivation for combining external knowledge with repeated computation. Its norms and exit objectives are not drop-in LFM surgery. |
| SMELT, September 2026 [6] | Compute-matched MoE scaling experiments include two passes through a middle segment. | Recent support for evaluating middle-block recurrence. Repeating our unchanged LFM core does **not** inherit that paper's compute matching or reported gains. |
| *How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents*, September 2026 [7] | Studies recurrence alongside growth and boundary operators in scaling experiments. | Reinforces measuring the re-entry operator as an architectural choice, rather than assuming raw repeated layer calls are sufficient. |
| *DeepLoop*, July/August 2026 [8] | Depth-aware residual/initialization analysis in a post-normalized setting. | Do not transplant its rescaling formulas into already-trained LFM pre-normalized blocks. Preserve the parent's block conventions. |
| *Mixture-of-Recursions* [9] | Adaptive token-level recurrence and cache-aware execution. | A later serving/halting direction. Start with sequence-level depths and explicit cache semantics, especially for a convolution/attention hybrid. |

The conversion experiments in [1] include training on tens of billions of tokens.
The small budgets in our recipes are **pilot budgets for the conversion interface
and task family**, not a reproduction of those experiments. Increase budgets based
on measured conversion and transfer curves, rather than interpreting a working
forward pass as a learned recurrent solver.

The research justification does not depend on rumors about proprietary frontier
models. Open conversion work is sufficient to start this experiment.

## 2. Exact LFM partition

The official configuration has 14 layers, hidden width 1,024, and this zero-indexed
sequence [10]:

```text
layer:     0 1 2 3 | 4 5 6 7 8 9 | 10 11 12 13
operator:  C C A C | A C A C A C | A  C  A  C
role:      prelude | shared core | coda
```

Here `A` is attention and `C` is the native short-convolution block. The runtime
partition is `[0:4]`, `[4:10]`, `[10:14]`. The split is a documented initial choice,
not an empirically established optimum. All normal feed-forward sublayers remain.

| Core passes R | Decoder-layer visits | Changed resident decoder blocks |
|---:|---:|---:|
| 1 | 14 | None |
| 2 | 20 | None |
| 3 | 26 | None |
| 4 | 32 | None |

These are **layer visits, not FLOP or latency ratios**. Different block types cost
different amounts, and planning, the reader and the bridge add work.

`MiddleBlockBackbone` retains a single owner for each native block. Increasing R
changes neither parameter identity nor count. The bridge adds `3d² + 3d + 3`
parameters: 3,148,803 at width 1,024. With eight blank workspace slots and a query
normalization, the new recurrence-specific additions total 3,158,019 parameters.
This excludes the existing writer, codecs and set reader, whose costs are reported
separately by the model manifest.

The attention-only comparison is SmolLM2-135M-Instruct, whose official configuration
uses Llama blocks, width 576 and 30 layers [12]. Its initial split is `[0:8]`,
`[8:22]`, `[22:30]`. This is a debugging/comparison route, not a conclusion that it
will outperform LFM. Neither real checkpoint has been executed in this CPU-only
handoff environment; their actual native paths are checked on the Spark.

## 3. Re-entry and the exact one-pass path

Let `P`, `F`, `C` denote prelude, shared core and coda; let `N` be the original final
normalization. The initial pass is exactly:

```text
p = P(embedded_input)
h1 = F(p)
output_at_R1 = N(C(h1))
```

No new bridge, extra final normalization, or soft-memory workspace is used by the
plain one-pass identity control. At initialization, it is the parent's function.
When parent weights later change, the one-pass path remains structurally identical
to the **updated** parent, not a guarantee of unchanged original capabilities.

For each subsequent pass, the implemented SCHNITZELJAGD bridge is:

```text
scale(p) = stopgrad(max(RMS_per_token(p), epsilon))
u_r = p + sigmoid(a) * scale(p) * W([RMSNorm(h_r), RMSNorm(p)])
u_r[memory_span] += sigmoid(c) * scale(p[memory_span]) * V(RMSNorm(z_r))
proposal = F(u_r)
h_(r+1) = h_r + sigmoid(b) * (proposal - h_r)
```

`W` starts as `[I, -I]`, `V` as `I`, and the three sigmoid coefficients start at 0.1.
The first core pass bypasses these parameters. Their initialization makes subsequent
updates conservative but live; it does not make all depths mathematically identical.
There are no per-depth copies of W, V or the core. The coda and original final norm
run only once after the last core pass.

This particular bridge is **our adaptation**, not a verbatim reproduction of a
published recipe. Its parameters, gradient norms, update sizes and eventual depth
benefit need measurement. Initial identity alone is insufficient, and an apparently
stable adapter that learns to ignore the memory does not meet the goal.

Normalization and scale are strictly **per token**. Pooling across sequence positions
would let teacher-forced future targets affect the prefix scale. There is no such
operation in the bridge.

### Preserve native hybrid semantics

The adapter calls the original decoder blocks, with the pinned Transformers 5.17
mask/RoPE interface. LFM's `embedding_norm` is its **final** norm, despite the name.
Convolution and attention receive their different native masks. Position indices
are sequence positions and are reused across depth; looping does not advance them.
The implementation reference is the model source [11].

All passes use `past_key_values=None`. Convolution histories and attention caches
must not silently survive as if one depth pass were another segment of sequence.
This reference recomputes prefixes at inference. It is suitable for validating the
training architecture, not an optimized serving engine. Layer checkpointing stays
inside the native block calls and does not repeat retrieval side effects. For native
middle-block training, activation checkpointing applies to the repeated shared block;
the one-pass prelude and coda retain their activations. This spends bounded memory to
avoid recomputing layers that execute only once.

## 4. SCHNITZELJAGD reads are part of the recurrence

The decoder input for a memory-conditioned decision is:

```text
prompt | fixed blank working slots | teacher-forced continuation, during training
```

The working slots are present before the answer, but initially contain no retrieved
payload. After a core pass, the query head reads a prefix-only workspace state.
Retrieval and the set reader produce a fixed-size result, inserted into that span
before the next core pass. The recurrence is therefore:

```text
prelude -> core 1 -> query/read -> bridge + inject -> core 2 -> ... -> coda -> answer
```

With multiple scheduled reads, the next query is derived from the newly computed
state. It can depend on earlier memory. Previously retrieved IDs are excluded from
new selection; accumulated evidence is recomposed by the reader. After the last
scheduled read, the current soft result can be reinjected without another disk query.

The current first query is after native layer 9, not in the first half of the initial
14-layer stack. That is deliberately the simplest actual core-boundary integration.
A prelude query after layer 3 and asynchronous prefetch is a separate latency
optimization; it has **not** been implemented or benchmarked in this release.

### Training and inference have the same conditional computation

During teacher-forced training the entire causal sequence can be processed in
parallel. Queries take only prefix states, so changing a future target does not
change the read plan. Tests explicitly replace the target and check every query.

At inference, `read_session` executes the prefix through the necessary boundaries,
captures `LoopMemory` events and their selected records, and reuses those fixed events
when scoring candidate answers or decoding a continuation. No candidate answer
participates in planning. Recomputing the prefix with those events reproduces the
integrated consumer graph in deterministic evaluation. The current implementation
pays this planning/recomputation cost explicitly.

Reads are scheduled rather than learned to halt. Native in-loop compaction and
streaming combinations are rejected rather than silently using a different graph.
The existing prefix-mode compaction experiments are still available. The initial
native experiment uses raw stored values, one space and one read; a second read is
supported when at least three core passes leave a pass to consume it.

### Producer replay

Producer depth is explicitly configured by `writer_loops`; changing or sampling
consumer depth does not silently change the serialization function. Integrated
read-before-write trajectories set it to the trajectory depth so their write-slot
states can depend on earlier injected results. Training can freshly regenerate selected values;
normal inference only reads stored values. Core and reader gradients accumulate
through the full consumer trajectory, then producer cotangents are replayed before
any optimizer update. This is not truncated backpropagation across depth.

Checkpointing reduces consumer activation retention; selective producer replay
reduces source-graph retention. Weight sharing alone does not make either activation
cost disappear. There is no read-count restriction imposed by the replay engine.

## 5. Conversion curriculum

The new recommended starter and causal recipes use four stages. Counts are initial
optimizer-update budgets, not promises of convergence.

| Stage | Core passes | Updated parts | Objective |
|---|---|---|---|
| `text_bootstrap`, 200 | 1 | Ordinary backbone | Teacher targets with the same support information rendered as text. |
| `recurrence_bridge`, 200 | 2 | Bridge; parent backbone frozen | Text target loss plus 0.1 forward KL from the fixed one-pass parent on identical inputs. |
| `latent_warmup`, 400 | 2 | Writer slots/heads, codecs, reader, bridge and workspace; backbone frozen | Downstream target loss through an actual in-loop read. |
| `recurrent_joint`, 400 | Sample 2 or 3 per optimizer step | Shared native core plus SCHNITZELJAGD modules; prelude/coda/embeddings frozen | Latent task loss plus a 0.1 one-pass oracle-text task anchor. |

The frozen-parent KL is available without another resident model because the R=1
path bypasses all bridge parameters and the base weights are frozen. It is computed
under `no_grad` on exactly the same text. Once native blocks are unfrozen, the code
refuses to call that path a fixed-parent teacher. The later text anchor is supervised
regularization, not fixed-parent distillation or proof of general retention.

Early memory-free passes are **not** trained to solve tasks whose crucial information
arrives later. The first-pass text objectives contain the evidence; the latent task
loss is evaluated after the read. Sampling depths never removes the last opportunity
to consume a scheduled result.

One depth is sampled per optimizer step and held through microbatches and producer
replay. Its RNG is included in atomic checkpoints. The interrupted/resumed test
checks exact model equality with an uninterrupted run. Gates and layer-visit counts
are logged along with the actual sampled depth.

The initial optimizer remains AdamW with the existing low core learning rate. [1]
provides a reason to compare Muon, but a valid hybrid parameter-group implementation
and controlled optimizer comparison are better than applying a matrix optimizer
indiscriminately to every convolution/norm/embedding tensor.

### Existing checkpoint conversion

A v0.3 full-stack checkpoint can be warm-started with a middle-block configuration
and `train.allow_recurrence_conversion: true`. Native parent parameter names are
preserved. Only the legacy gate/norm may be discarded and only the explicit new
bridge/workspace parameters may be missing. Any other unexpected state fails.
Changing an existing middle-block partition through ordinary warm-start also fails.
Use a fresh output directory: conversion is not resuming the old optimizer/cache.

The new staged recipes start directly from the pretrained Hugging Face checkpoint;
no old SCHNITZELJAGD checkpoint is required.

## 6. Running the experiments

On the Spark, start with a short real-model integration run:

```bash
./scripts/start_spark.sh --recipe recipes/looped_smoke.yaml --output /runs/looped-smoke
```

Then run the teacher-trajectory curriculum or controlled causal curriculum:

```bash
./scripts/start_spark.sh --recipe recipes/looped_starter.yaml --output /runs/looped-starter
./scripts/start_spark.sh --recipe recipes/looped_causal.yaml --output /runs/looped-causal
```

These are independent foreground runs, not intended to launch simultaneously by
pasting the whole block into background jobs. Resume a particular run with the same
recipe/output plus `--resume`. Revision locks and native NVIDIA ARM64 image digest
pinning are unchanged. The local image tag is now `schnitz-spark:0.4`.

Use an independent attention-only comparison with:

```bash
./scripts/start_spark.sh --recipe recipes/smollm2_looped_causal.yaml --output /runs/smol-looped
```

The preflight checks the actual loaded checkpoint's parent identity, finite recurrent
states, target-prefix causality, and live writer/reader/bridge gradients before any
stage runs. Optional random native LFM/Llama tests are also included for environments
with Transformers installed. Do not override a failing preflight to continue training.

After the causal run:

```bash
./scripts/spark.sh run schnitz evaluate-depths \
  --run runs/looped-causal/recurrent_joint \
  --episodes runs/looped-causal/fresh-causal.jsonl \
  --output runs/looped-depths --depths 1 2 3 4 --max-episodes 32
```

For recorded teacher data use `--protocol teacher` and the prepared validation JSONL.
The command materializes the bank **once**, then forbids writer calls across the
entire sweep. R=1 is the exact no-in-loop-read control; comparisons among R>=2 use
the same read budget. More passes can reduce quality; R=4 is an explicit test beyond
the joint stage's main training depths, not an assumed improvement.

The sweep pins its checkpoint, episode file, requested depths and evaluator code
to a stable output directory. Its frozen bank is verified on resume without source
encoding, and completed depth reports are reused. Teacher-protocol scoring saves
each condition with its original read plan, so STOP or a signal can resume within
a depth without rerouting payload interventions. Transfer scoring stops between
completed depths; the active depth finishes before a cooperative stop returns.

The timing includes prefix planning, payload access, and all candidate/condition
scoring after bank materialization. OS page-cache state is uncontrolled; it is
neither cold-NVMe timing nor a tokens-per-second serving result. Oracle routing
uses declared source IDs; learned routing uses the exact stored-key scan, not ANN.

Offline execution without downloads:

```bash
python -m pytest -q
schnitz launch --recipe recipes/tiny_looped_smoke.yaml --output runs/tiny-looped
schnitz launch --recipe recipes/tiny_looped_smoke.yaml --output runs/tiny-looped --resume
```

## 7. What would establish useful conversion?

Distinguish four results, rather than requiring a single headline benchmark:

**Installation:** one-pass equivalence, live recurrent parameters, native masks,
correct replay, and stored-only execution. The CPU tests establish these for the
test backbone; the actual LFM preflight remains an on-device check.

**Retention:** evaluate the one-pass and recurrent models on held-out teacher text
before and after each stage, including inputs outside the narrow causal task family.
Frozen-parent KL is a training aid, not a replacement for these measurements.

**Causal transfer:** newly written stored supports affect appropriate decisions;
changing a relevant source changes the later answer; removing either necessary
support impairs composition. Compare zero-payload and wrong-payload controls with
unchanged read metadata. The no-memory condition also removes working slots, so
zero/wrong-payload conditions provide a closer layout-matched intervention.

**Compute benefit:** compare 2/3/4 passes at fixed weights, fixed writer and fixed
bank; compare against separately trained one-pass text and latent baselines. Compare
quality against actual time/memory and equal information access. Layer visits are
not substitute measurements. A better recurrent agent with a changed writer is not
by itself evidence that extra inference loops caused the improvement.

The next extensions are to vary the core split and bridge, add a genuinely frozen
retention reference when more base weights are unfrozen, and integrate prelude
prefetch with explicit result availability. Learned stopping, per-loop caches,
token-level schedules, and parameter-reducing block merging follow the validated
reference. Their postponement does not block training the recurrent SCHNITZELJAGD now.

## Sources

[1]: https://arxiv.org/html/2511.07384v1
[2]: https://arxiv.org/html/2608.11233v1
[3]: https://arxiv.org/html/2410.20672v3
[4]: https://arxiv.org/html/2502.05171v2
[5]: https://arxiv.org/html/2510.25741v1
[6]: https://arxiv.org/html/2609.01343v1
[7]: https://arxiv.org/html/2609.19107v1
[8]: https://arxiv.org/html/2607.13491v2
[9]: https://arxiv.org/html/2507.10524v2
[10]: https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/config.json
[11]: https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/lfm2/modeling_lfm2.py
[12]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json

- [1] [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence][1].
- [2] [Retrofitting Recurrent Depth into a Pretrained Language Model][2].
- [3] [Relaxed Recursive Transformers][3].
- [4] [Scaling by Thinking in Continuous Space][4].
- [5] [Scaling Latent Reasoning via Looped Language Models][5].
- [6] [SMELT][6].
- [7] [How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents][7].
- [8] [DeepLoop][8].
- [9] [Mixture-of-Recursions][9].
- [10] [Official LFM2.5-230M configuration][10].
- [11] [Pinned Transformers LFM2 implementation][11].
- [12] [Official SmolLM2-135M-Instruct configuration][12].
