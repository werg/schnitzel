# Stigmergic Compactable Holographic Neural Indexed Trajectory Zettelkasten with Evolving Latents, Jointly Adapted by Gated Decoders (SCHNITZELJAGD)

> **Current memory design, 28 September 2026:** the
> [knowledge-base stack](knowledge-base-stack.md) specifies the superposed spaces,
> forward codecs, per-space compactor-combiners and recombiner, built from a dense
> MLP-matrix operator, on the BGKit restart ([restart plan](bgkit-restart-plan.md)).
> The research objective below stands; where the two differ, the stack document is current.

> **Implementation update, 19 September 2026:** the decoder now has a native
> prelude/shared-core/coda conversion with in-loop reads and a conversion curriculum.
> [Recurrent conversion specification](recurrence.md) gives the executable v0.4
> design, rather than treating the looped backbone as a deferred prerequisite.
>
> **Target update, 21 September 2026:** [trajectory memory v0.5](trajectory-memory-v0.5.md)
> specifies frequent transcript-level `memory.search` and `memory.write` tool calls,
> length-scaled multi-record writes, recursive read-then-write bank evolution,
> and corpus-scale capacity targets. These are planned behaviors, not properties of
> the current one-read HotpotQA run.
>
> **Target update, 22 September 2026:** [adaptive memory v0.6](adaptive-memory-v0.6.md)
> specifies density-adaptive continuous contribution gates, gentle support anchors,
> selective key-and-payload writer replay, continuously revised stored records,
> and holistic plus streaming document-ingestion trajectories. Its status table
> distinguishes implemented primitives from curriculum and compaction work still pending.
>
> **Lifecycle update, 22 September 2026:** [mutable bank v0.8](mutable-bank-v0.8.md)
> supersedes the earlier immutable-generation target. Source trajectories, event
> order, provenance, and historical physical revisions remain immutable. The logical
> records they ground have continuously trainable keys, payloads, index membership,
> utility state, and compact representations.
>
> **Scale-out preparation, 22 September 2026:** [scale-out storage and recursive
> compaction v0.9](scale-out-v0.9.md) defines the public network-backend contract,
> leased dependency-ordered rebuilds, and behavior-plus-byte promotion gates. The
> contracts and local worker machinery are implemented; a network service and a
> measured learned compaction benefit remain future work.

## Read-time superposition, selective replay, and learned cluster compaction

**Research architecture and experimental specification — 19 September 2026**  
**Revision 2:** Review-integrated plan and compactability-aware training.  
**Status:** Proposed design; no model-training or deployment results are asserted.

**Revision summary.** The core architecture is retained. Oracle transfer and multi-record composition now precede learned routing. The attention comparator receives the same contribution-compaction treatment as the pooled MLP. Write and read bottlenecks, information access, and actual stored-payload execution are isolated explicitly. New optional mechanisms encourage compactability before persistent compaction is built: temporary surrogate replacement, locally overlapping compaction fields, controlled latent perturbations, and a mergeable shared component with separate exceptions. Full compaction infrastructure remains a later stage; a cheap training intervention does not need to wait for it. Appendix C records which review points were accepted, narrowed, or treated as optional project criteria.

### Abstract

This document specifies a language-model architecture that trades selectively accessed external storage and computation for lower resident neural-network capacity. A small controller reads and writes a knowledge base through a fixed interface: one continuous key vector and a fixed number of soft value tokens. Internally, the knowledge base maps records into multiple retrieval spaces with increasing payload width and decreasing neighborhood size. A query-conditioned pooled-MLP reader repeatedly processes individual records, linearly aggregates their contributions, and feeds the aggregate back through a shared residual state. Its output is a fixed-size latent context that subsequent decoder computation can consume.

Training emphasizes transfer between experiences rather than isolated reconstruction. Distinct support trajectories generate memories whose usefulness is measured on later query tasks. Selected values are regenerated from their sources during training, while others are read from stored encodings. Selective producer replay routes downstream gradients into memory-writing computations without retaining all producer activation graphs. Normal inference reads stored payloads; source-trajectory regeneration is not part of its read path.

Read-time superposition is the initial learning mechanism. A later amortized compaction encoder replaces clusters with fewer synthetic records or conditional codes, trained to preserve their query-, selection-, and state-dependent contributions. Compactability is a separate, distribution-dependent hypothesis, not a consequence of additive pooling. Temporary compression during training can encourage useful representations to become compactable without requiring a persistent compacting store. The research program tests transfer, composition, compaction, and the resulting quality–VRAM–latency trade separately before combining them.

### Contents

1. [Research thesis and design commitments](#1-research-thesis-and-design-commitments)
2. [System organization and interfaces](#2-system-organization-and-interfaces)
3. [Multiscale external memory](#3-multiscale-external-memory)
4. [Dense pooled-MLP reader](#4-dense-pooled-mlp-reader)
5. [Looped controller and asynchronous execution](#5-looped-controller-and-asynchronous-execution)
6. [Learning from experience and teacher trajectories](#6-learning-from-experience-and-teacher-trajectories)
7. [Selective replay and gradient accounting](#7-selective-replay-and-gradient-accounting)
8. [From read-time superposition to cluster compaction](#8-from-read-time-superposition-to-cluster-compaction)
9. [Memory lifecycle, staleness, and online learning](#9-memory-lifecycle-staleness-and-online-learning)
10. [Resource model and engineering constraints](#10-resource-model-and-engineering-constraints)
11. [Evaluation: proving transfer, composition, and substitution](#11-evaluation-proving-transfer-composition-and-substitution)
12. [Implementation roadmap and reference configuration](#12-implementation-roadmap-and-reference-configuration)
13. [Main risks and unresolved choices](#13-main-risks-and-unresolved-choices)
14. [Relationship to prior work](#14-relationship-to-prior-work)

[Appendix A: Selective replay update](#appendix-a-selective-replay-update) · [Appendix B: Pooled reader and compaction targets](#appendix-b-pooled-reader-and-compaction-targets) · [Appendix C: Review disposition](#appendix-c-review-disposition) · [Appendix D: Minimal compactability experiment](#appendix-d-minimal-compactability-experiment) · [References](#references)

---

## 1. Research thesis and design commitments

### 1.1 The objective

The project asks whether increasing selectively accessed, disk-resident memory can reduce the resident neural-network capacity required to achieve a given level of language-model and agentic-task performance. The target is a small decoder, initially below one billion resident backbone parameters, whose capability grows through an external knowledge base (KB), learned memory operations, and additional recurrent computation.

The system does not eliminate stored information or claim that an arbitrary large model can be reproduced by an arbitrarily small controller. It redistributes capacity across resident weights, external latent records, and computation. A successful result must improve a measured quality–VRAM–latency frontier, not merely improve a small model after giving it more data or more inference time.

**Central research hypothesis.** A compact learned controller can write transferable representations of experience, retrieve multiple relevant records, and synthesize them into a fixed-size latent context. Training this synthesis first, then optionally exposing it to temporary compression during training, may yield representations that can be compacted into fewer external bytes while retaining their useful behavior on the task distribution.

### 1.2 Three coupled hypotheses

**Capacity substitution:** external memory replaces some task-relevant resident capacity under explicit latency and resource constraints. **Read-time composition:** multiple experiences jointly support new tasks that no individual experience solves. **Behavior-preserving compaction:** an amortized cluster encoder learns to replace groups of records while preserving their conditional contributions to the reader.

A fourth, systems-level hypothesis is that a recurrent decoder can overlap useful computation with retrieval and use additional shared-weight depth to interpret the resulting latent context. Recurrence is a backbone capability to adopt and integrate, not the primary architectural research contribution of this project.

### 1.3 What is fixed, and what remains experimental

The defining interface is one continuous key vector and a fixed number of continuous value-token positions per KB operation. The stored representation may differ from that interface. The primary reader is a dense, query-conditioned pooled-MLP network with repeated aggregation and residual feedback. Neighborhood self-attention is not required.

Superposition means **composition at read time**, not initially writing multiple memories into the same storage location. Persistent cluster compaction is a later stage, trained against the behavior of this reader. Small in-memory compression probes and compactability-aware training can begin once useful composition has appeared; they do not require cluster-level indexing, disk deployment, or background maintenance. Fresh value encoding from source trajectories is a training-only mechanism; deployment reads stored latent payloads. Training may mix stored values with selectively regenerated values.

Selective replay, activation checkpointing, and offloading address training-memory pressure. The architecture does not impose a small read-count ceiling merely to keep all producer graphs resident. Read frequency may still be optimized for actual compute, bandwidth, or latency costs, and finite deployments still require admission control.

Experimental choices include the number and geometry of retrieval spaces, their width/neighborhood schedule, the number of reader rounds and decoder loops, address-space plasticity, the compaction representation, and the exact training curriculum. All numerical configurations in this document are proposed starting points, not measured results.

## 2. System organization and interfaces

### 2.1 Components and data flow

The resident system contains a decoder/controller, a query head, a writer mode with continuous output heads, per-space projections, and a fixed-size set reader. The persistent system contains a versioned retrieval index, latent value payloads, and metadata. A source archive supplies selected trajectories or documents during training and regeneration jobs; it is not on the normal inference read path.

| Stage | Operation | Output or state |
|---|---|---|
| Experience | Run a task, observe tool results and outcomes, or ingest a document. | A source record with provenance and causal boundaries. |
| Write | Encode the source using the controller in writer mode. | One key and a fixed sequence of soft value tokens. |
| Store | Map the key and value into multiple retrieval spaces. | Versioned keys, transformed payloads, and source handles. |
| Query | Extract a continuous key from a causally available decoder state. | An asynchronous read request. |
| Compose | Retrieve neighborhoods and repeatedly pool conditional MLP contributions. | A fixed number of decoder-shaped soft tokens. |
| Continue | Insert the returned tokens at a defined computation boundary. | Revised reasoning, an action, or a follow-up query. |
| Compact | Distill groups of records against the trained reader. | Fewer synthetic records or a compact conditional code. |

### 2.2 Notation

| Symbol | Meaning |
|---|---|
| $D_\theta$, $W_\phi$ | Decoder and writer; their backbone parameters may be shared. |
| $q,k_i\in\mathbb R^{d_k}$ | One read key and the canonical write key of record $i$. |
| $V_i,Z\in\mathbb R^{m\times d_D}$ | Canonical write values and the returned decoder-shaped soft tokens. |
| $\ell=0,\ldots,L-1$ | Retrieval-space index. |
| $d_\ell,n_\ell$ | Stored payload width and target retrieved count in space $\ell$. |
| $\mathcal N_\ell(q)$ | Actual retrieved neighborhood after eligibility filtering. |
| $R^{(t)}\in\mathbb R^{m\times d_R}$ | Shared reader residual state at round $t$. |
| $T,K$ | Number of reader rounds and decoder loops, respectively. |
| $\tau_i,\bar V_i$ | Recorded source trajectory and an already stored value encoding. |
| $c_C,\rho_C$ | Compact representation of cluster $C$ and its selection metadata. |

### 2.3 The soft-token contract

A write operation produces $(k_i,V_i)=W_\phi(\tau_i)$. The key is a single vector, not a vocabulary token. The $m$ value positions are also continuous vectors. A shared writer backbone can expose a key head and $m$ learned output slots; a separate large encoder is not required. A long experience can yield several records through several fixed-shape writes.

In the target agent interface, those operations are visible transcript events.
The controller emits `memory.search` and `memory.write` tool calls, receives ordinary
tool-result envelopes, and attaches latent read results at the corresponding causal
result sites. Each result site has blank learned workspace positions on the initial
pass. After a recurrent core pass, all active same-level queries are retrieved
together and their results are scattered into their respective workspaces for the
next pass. Repeated tool calls distribute sites through a long trajectory. See the
v0.5 plan for the protocol, site density, recursive bank evolution, and scale budget.

One site is one visible tool call at one causal position. Multiple records batched
inside a call or multiple recurrent reads used to compute one result do not create
multiple trajectory sites. Recursive evolution adds repeated read-author-write
cycles through bank time; it does not replace repeated calls within each trajectory.

A read operation returns $Z\in\mathbb R^{m\times d_D}$ plus discrete status and provenance metadata. An input projection, normalization, and learned gate place $Z$ at the decoder's expected scale. Empty, pending, failed, and completed reads have distinct states. The API does not silently interpret an unavailable result as useful zero-valued evidence.

Record order is exchangeable inside a neighborhood. Positions within each stored value sequence and positions in the returned sequence are not exchangeable. They need explicit positional or slot identity. Read handles, source identifiers, and timestamps are not additional semantic query tokens; they are operational metadata.

### 2.4 An illustrative learning cycle

Consider a synthetic coding environment with unfamiliar transaction and retry conventions. One support task reveals that a failed attempt must restore a state snapshot before retrying. A separate support task reveals that a particular error is retryable only when a capability flag is present. Neither support contains the solution to the later evaluation task.

After each support task, the writer receives the observed trajectory and emits a key plus $m$ latent value tokens. The storage transforms create each space's addressing vector and payload. The original trajectories remain available in the training archive; inference stores and reads the generated payloads.

A new task requires a correct retry wrapper. The controller's initial pass issues a query while beginning its attempt. Broad spaces can return related procedural patterns; narrower spaces can return the particular conventions. The reader's first pooled round may establish the relevant stateful-operation interpretation. Later rounds can extract and combine the preconditions and exceptions, returning $m$ soft tokens. A subsequent decoder loop uses those tokens to complete the wrapper or issues a more specific follow-up query.

During training, the target task's loss flows through the reader into selected support values. Selective replay then trains the writer to encode those support trajectories more usefully for this later task. The architecture need not identify a human-readable sentence corresponding to every latent feature. The evaluation question is whether changing or withholding the appropriate support changes the correct behavior.

Once this composition is reliable, a compactor can replace groups of related records with a few synthetic records or a conditional code. Its training target is the group's contribution across queries and reader states, including interactions with other groups. At deployment, both new individual records and compacted records are read as stored values. The completed task may itself create a new write, making experience available to later tasks without requiring a weight update.

This is an illustrative behavior and test fixture, not a predicted internal decomposition. The model might distribute the information across slots and rounds differently. What must hold is the causal sequence: **experience → write → later read and composition → downstream utility → training credit → later consolidation**.

## 3. Multiscale external memory

### 3.1 Address spaces and stored values

For each canonical record, a space-specific key transform and value transform produce

$$
k_i^{(\ell)}=K_\ell(k_i),\qquad u_i^{(\ell)}=C_\ell(V_i),\qquad q^{(\ell)}=Q_\ell(q).
$$

Here $u_i^{(\ell)}$ denotes the whole stored payload in that space and its total scalar width is $d_\ell$. Historical Phase 1 records treat it as a flat vector. The Phase 2 interface defined in `positional-memory-v0.7.md` requires the logical shape `[stored_positions, channels_per_position]`; contiguous serialization does not remove that structure. Position count and channel width must both be recorded in the bank identity and byte budget. Key dimensions are separate from value dimensions and need not follow the same schedule.

Each space retrieves a neighborhood using its own score and eligibility rules. The reader receives values, key-derived relevance features, space identity, and selected provenance features. Values from the same source can appear in several spaces; this is intentional multi-view processing, not independent supporting evidence. Maintain canonical record identifiers so that duplicates and correlated sources can be recognized.

### 3.2 Width–neighborhood schedule

The initial multiscale hypothesis is

$$
d_\ell=d_0\,2^\ell,\qquad n_\ell\approx n_0\,2^{-\ell}.
$$

Broad neighborhoods therefore contribute many narrow representations, while narrow neighborhoods contribute fewer detailed representations. Before integer rounding, the payload count per space is constant: $n_\ell d_\ell=n_0d_0$. With $b$ bytes per scalar, the idealized read payload is

$$
B_{\mathrm{values}}\approx b\sum_\ell n_\ell d_\ell=bL n_0d_0.
$$

This is a bandwidth allocation heuristic, not a theorem about information or speed. Index traversal, key payloads, metadata, record alignment, cache misses, and per-record compute remain additional costs. Returning many tiny records can cause more I/O operations than returning a few larger records with the same total payload.

If all $N$ original records are stored in every space, storage grows as $bN\sum_\ell d_\ell$. Constant read payload across levels does not imply constant total storage. The project intentionally permits more external storage, but must measure duplication and later compaction rather than hide them.

### 3.3 Diversity, granularity, and topology

The desired diversity is operational: spaces should expose different useful relationships, such as task intent, failure mechanism, procedural structure, or interface compatibility. Different dimensions or metrics do not by themselves create complementary neighborhoods. An orthogonal rotation with Euclidean distance, for example, preserves all distances and therefore preserves nearest-neighbor rankings.

The first implementation should use learned projections and ordinary normalized similarity, with geometry recorded as an experimental variable. Alternative geometries or topologies can be added when a task distribution motivates them. Compare downstream marginal utility, neighborhood overlap, and coverage of jointly necessary records; do not optimize geometric difference as an end in itself.

Coarse spaces may learn shared patterns and fine spaces may supply exceptions, but this division must emerge or be tested. Scale dropout and resource-matched ablations can expose scale collapse. Auxiliary scale-specific objectives are optional regularizers, not requirements that every scale be independently capable of solving every task.

### 3.4 Retrieval plans and exact-detail access

A read plan records selected canonical IDs, space-local scores, the index generation, and any filtering or deduplication. Preserve it through replay. Approximate-nearest-neighbor membership is a discrete decision; the ordinary task gradient through selected payloads does not differentiate through all possible index outcomes.

The baseline uses fixed-size latent responses throughout. An optional exact-detail path may expose source spans for tasks requiring literal names, constants, or code fragments. Evaluate this as a separate variant and charge its bytes and decoder tokens. Otherwise it can obscure whether the latent memory itself is carrying the relevant capability.

## 4. Dense pooled-MLP reader

### 4.1 Local contributions and shared residual feedback

The reader maintains $m$ fixed output slots, initialized from the query and learned slot embeddings. A record participates in every output position through a shared conditional MLP. Each round aggregates all records before updating the shared state:

$$
a_{\ell j}^{(t)}=\sum_{i\in\mathcal N_\ell(q)}w_{\ell i}^{(t)}\,\Phi_{\ell t}(x_{\ell i},q,r_j^{(t)},e_j),
$$

$$
R^{(t+1)}=R^{(t)}+\Psi_t\!\left(R^{(t)},\{a_\ell^{(t)}\}_\ell,q,\nu^{(t)}\right),\qquad Z=P_{\mathrm{out}}\operatorname{Norm}(R^{(T)}).
$$

$x_{\ell i}$ combines the stored payload with allowed key and metadata features. $w_{\ell i}^{(t)}$ is an optional nonnegative relevance gate; setting it to one gives the simplest dense reader. $\nu$ contains counts or total weights used for normalization. More expressive slot-dependent gates are possible, but complicate bookkeeping and are not needed for the initial design.

The contribution vectors themselves can have signed components. Consequently, the aggregate is not restricted to a convex combination of stored value vectors. $\Psi_t$ can mix the fixed output slots and spaces using an MLP. Cross-record interaction occurs through the residual state, which is broadcast into the next round's local computations.

This is related to pooled permutation-invariant and equivariant processing in Deep Sets [^4], but the proposed query-conditioned recurrent reader and its training objectives are design choices here. It is not a claim of unlimited expressivity: results on set-function approximation show that latent dimensionality can be a genuine bottleneck [^5].

### 4.2 Why feedback matters

Without feedback, each memory contributes a representation conditioned only on the original query. With feedback, one round can establish a shared interpretation and the next can extract information relevant to it. In a coding example, one memory may establish that an operation is stateful, causing another memory's retry rule to contribute a different exception or precondition in the next round.

This creates dense communication through a limited shared channel. It avoids explicit all-pairs record interactions, but it can lose distinctions that require a larger shared state, more rounds, or explicit binding. Tasks involving role assignment, contradictory evidence, and many interacting entities should therefore be deliberate stress tests. The output slots should preserve typed roles or learn slot specialization rather than collapse into repeated copies.

### 4.3 Efficient conditional MLP blocks

A full width-$d$ MLP evaluated separately for every record–slot pair can cost approximately $O(nmd^2)$. A useful first implementation factorizes the local computation:

$$
h_{\ell ij}^{(t)}=\sigma\!\left(U_{\ell t}x_{\ell i}+V_{\ell t}r_j^{(t)}+Q_{\ell t}q+e_{\ell j}^{(t)}\right),
$$

$$
s_{\ell j}^{(t)}=\sum_i w_{\ell i}^{(t)}h_{\ell ij}^{(t)},\qquad a_{\ell j}^{(t)}=P_{\ell t}s_{\ell j}^{(t)}.
$$

The final linear projection moves outside the sum exactly. An output bias must either be omitted locally or multiplied by the corresponding total weight; adding it once after pooling is otherwise a different function. Record projections and slot/query projections can be computed once per round, leaving a broadcast addition, nonlinearity, and accumulation per record–slot pair.

If $x$ has width $d_x$ and the intermediate width is $h$, the main terms per space and round are $O(nd_xh+md_Rh+nmh+mhd_R)$, plus gate and residual-update costs. These are derived operation-count estimates, not hardware measurements. Adding additional nonlinear hidden layers per pair restores more expensive pairwise matrix multiplications; this is an expressivity/compute ablation.

### 4.4 Normalization and evidence multiplicity

Accumulate an additive numerator and additive mass separately, then normalize only after the relevant contributions have been merged. For example, $\mu_\ell=\sum_iw_{\ell i}$ and $a_\ell/(\epsilon+\mu_\ell)$ give a weighted mean, with $\log(1+\mu_\ell)$ and distinct-source count also exposed to the residual update.

A raw sum changes scale with neighborhood size. A mean alone discards multiplicity. Counting every duplicate as independent evidence can inflate confidence. Compare sum, mean-plus-count, and learned normalization while tracking canonical duplicates and source-family correlation. If confidence is reported, calibrate it against outcomes rather than derive it mechanically from record count.

The null-read case uses an explicit status embedding and well-defined zero mass. Stable accumulation precision and a deterministic reduction order are especially useful when testing replay parity. Chunking can change floating-point summation order even when the mathematical function is unchanged.

### 4.5 Comparison with fixed-slot cross-attention

The main alternative is a small latent array cross-attending to the retrieved records, as in the latent-bottleneck approach of Perceiver IO [^6]. It also avoids neighborhood-wide quadratic self-attention. Standard attention is not inherently sparse: both readers can aggregate every record.

The distinction is inductive bias. A simple attention head changes scalar selection weights over projected values; the conditional MLP can change the vector contribution of a record before pooling. That distinction motivates a comparison, not a presumption of superiority.

**An additive compaction boundary is available to both readers.** For one attention head and a selected cluster $C$, define

$$
N_C(h)=\sum_{i\in C\cap\mathcal N(q)}\exp(s(h,k_i))v_i,
\qquad D_C(h)=\sum_{i\in C\cap\mathcal N(q)}\exp(s(h,k_i)).
$$

The global output is $\sum_C N_C/\sum_C D_C$. This follows directly from softmax attention [^17]. Train an attention compactor to preserve these conditional numerators and denominators, just as the MLP compactor preserves contributions and mass. Both remain query-dependent functions; additivity alone does not ensure either admits a smaller representation.

Use per-head, per-slot targets and identical selection plans. For numerical stability, a cluster may store or return a log-scale alongside scaled numerator/denominator statistics; merge using a common scale, not independently normalized outputs. Any record dropout masks are part of the fixed training read plan.

Compare both uncompacted and compacted readers at matched output slots and retrieved information, with separate bytes-, parameter-, and measured-compute-matched comparisons. The relevant outcome is a quality–bytes–compute curve, not a single attention-versus-MLP score.

### 4.6 Streaming semantics

For a given round, stream chunks of records, accumulate their contributions, and only then update $R$. The next round sees the completed shared state. Processing each chunk through all rounds independently and combining the final outputs does not implement the same model.

Chunked replay need not imply rereading disk at every round. Small retrieved payloads or their projections can stay in a bounded GPU/host buffer while large producer and reader activation graphs are discarded. If even the payloads cannot stay resident, account for the additional host transfers or disk passes explicitly.

## 5. Looped controller and asynchronous execution

### 5.1 The role of recurrence

The decoder processes the complete available training trajectory at every recurrent
level. Tool-result locations contain learned blank workspace embeddings until memory
is available. After a shared core pass, query heads gather states at every active
tool-call position, retrieval and reader composition run for all those sites, and
their fixed-size results are scattered into the corresponding blank positions. The
next core pass processes the complete trajectory again with those latent results.
Shared-weight depth therefore supplies both additional computation and ordered
memory-dependency levels without adding decoder parameter copies. Public
recurrent-depth language-model work provides a relevant implementation precedent
[^12].

Spatial site count and recurrent depth are independent. Many sites can activate in
one level. A site assigned to a later level can use earlier-position results injected
at a prior level, producing read-after-read composition through the causal mask. The
legacy Phase 0 path is the one-site special case: one workspace after the prompt and
one query per boundary. The spatial Phase 0 path now executes many sites and batched
stored reads. A prequential executor can encode and publish the packer's fixed
prompted writes after each completed trajectory. Learned call placement and generated
write arguments remain later work.

### 5.2 Query timing and causality

For site $j$ at token position $p_j$ and recurrent level $r_j$, the query head reads
$H^{(r_j)}_{p_j}$. The normal causal mask permits positions at or before $p_j$ and
excludes later positions even though all positions in the pass execute in parallel.
Its result workspace lies after the call position. A same-level result cannot affect
another query from that level; a dependency must be assigned to a later recurrent
level. Teacher-forced targets and future observations never enter an earlier query.

Each request carries a unique site handle, issuing level, model/index generation,
authorization scope, time boundary, and causally valid state identity. Results are
consumed only by the declared site workspace on the following pass. The request
state machine is issued, pending, ready, consumed, failed, or cancelled. Train with
variable delays and failures. A bounded number of in-flight requests is operational
backpressure, not a bound on spatial sites or total reads.

### 5.3 Overlap and cache semantics

Without inter-batch overlap, an idealized recurrent trajectory latency is

$$
T_{\mathrm{trajectory}}\approx T_{\mathrm{prelude}}+T_{\mathrm{coda}}+
\sum_r\left(T_{\mathrm{core},r}+\max_{j:r_j=r}T_{\mathrm{retrieve},j}
+T_{\mathrm{compose/scatter},r}\right).
$$

Same-level sites overlap with one another. Across training batches, retain the
differentiable whole-sequence boundary state and run another batch while a retrieval
wave is pending. Level-boundary checkpoint/recomputation is the lower-memory fallback.
Evaluate latency and throughput under concurrency rather than treating overlap as
free computation. Engram demonstrates a related prefetching opportunity using
deterministic token-derived addressing; hidden-state-dependent semantic queries do
not inherit its predictability automatically [^14].

Injection deliberately revises the next recurrent computation of the whole sequence.
No attention or convolution cache is shared across loop depths. A retained state or
recomputed boundary must reproduce the uninterrupted graph, RNG, autocast, selected
plans, and serialized payload precision. Retrieval callbacks never run during
checkpoint recomputation.

### 5.4 Frequent reads without conflating memory budgets

The total number of reads, the number of outstanding requests, the number of retained read tokens, and the number of resident producer graphs are distinct quantities. Selective replay targets the last one. It does not require a small value for the first.

For autoregressive inference, only the generated prefix is available, so it cannot
use training's parallel access to future positions. Once the model emits a call and
retrieval completes, an optimized path can populate that site's workspace after the
fixed prelude and enter the recurrent core with the result already present. This
avoids spending a blank recurrent pass on a query already produced by generation.
Post-training or policy optimization should mix this immediate-result schedule with
the blank-first-pass reference and measure whether behavior transfers. Serving can
overlap independent requests. Offloading or consolidation policies may change
retention costs, but each changes model semantics and must be evaluated separately.

## 6. Learning from experience and teacher trajectories

### 6.1 Support/query episodes

The main training unit is a family of related tasks. Support experiences $S_z$ populate an episode-local memory, and distinct query tasks $Q_z$ measure whether that memory transfers:

$$
\mathcal M_\phi(S_z)=\{W_\phi(\tau):\tau\in S_z\},
$$

$$
\mathcal L_{\mathrm{task}}=\mathbb E_{z,S_z,Q_z}\left[\sum_{(x,y)\in Q_z}-\log p_\theta(y\mid x,\operatorname{Read}(x,\mathcal M_\phi(S_z)))\right].
$$

This equation suppresses the sequence of adaptive reads for clarity. In an agentic implementation, the policy conditions on its current state and all causally available results, and task loss may combine action-token likelihood with verified environmental outcomes.

A support trajectory may include prompts, actions, observations, failures, repairs, and verifier results. The writer learns representations useful to subsequent tasks rather than merely compressing the wording of the source. Query solutions must not appear in the support memory or in a write that is visible before its causal completion.

### 6.2 Data construction

Start with controlled task families whose API names, conventions, constraints, and combinations can be randomized. For example, separate support episodes can teach a state-restoration requirement and a retry condition, while the query requires both. Hold out combinations and task families, not just surface strings.

Then use larger-teacher agentic trajectories, including successful and failed coding attempts with observable outcomes. Distill behavior into the student and its memory interface; no alignment between teacher and student hidden-state coordinates is assumed. Teacher traces that use information unavailable to the student should be marked or excluded from the relevant supervision.

Document-derived memory can be produced from selected Wikipedia snapshots, FineWeb material, or other documented corpora; FineWeb provides a published account of its curation approach [^16]. Treat document sources and experiential sources as different provenance types. Record snapshot, processing recipe, source identifiers, use permissions, and exclusion rules. Source deletion and benchmark contamination controls must propagate to derived records.

Mix episode-local memory with a persistent distractor corpus as training progresses. Otherwise an encoder may work only when every retrieved record belongs to the current task family. Include irrelevant, contradictory, duplicated, stale, and null-memory conditions deliberately.

### 6.3 Bootstrap the interface, then optimize transfer

Initially train the soft interface with oracle-selected evidence, reconstruction of selected details, continuation prediction, and question answering about the support. AutoCompressors and ICAE establish precedents for learned soft representations that a language model can consume [^7][^8]. They motivate a bootstrap, not the final objective.

Shift emphasis toward cross-experience query performance once communication works. Preserve this as the main objective: reconstruction is useful only insofar as it enables later decisions.

Writing and reading are separate information bottlenecks. A write is formed before the future query is known; a read condenses selected stored records after the query is known. Add a write-once/many-uses test: encode a support experience once, serialize it, and reuse that same payload for procedure, exception, failure-diagnosis, and exact-detail queries without rewriting. Vary write capacity separately from read capacity. In diagnostic variants denote their fixed slot counts by $m_W$ and $m_R$; the default architecture uses $m_W=m_R=m$. Also sweep stored payload width/precision separately from the final response width. Every run still has a fixed per-operation interface. A narrow task-conditioned code may be useful, but its scope should not be mistaken for general retention. If oracle text helps but latent evidence does not, investigate compression and decoding. If oracle latent evidence helps but learned routing does not, investigate addressing. If neither helps, test controller training and recurrent computation before attributing failure to the index.

The writer may share the decoder backbone with a distinct write mode, normalization, and output heads. The initial implementation should avoid independently trainable free payloads as its main mechanism: those can memorize training records without teaching a writer to encode unseen experiences. Directly optimized payloads remain an informative comparison.

### 6.4 Query-space learning from downstream utility

Learned routing follows successful oracle transfer and composition. A single-record utility objective is useful for bootstrap, but not sufficient for the intended system. Given already obtained support $S$ and a candidate $i\notin S$, use conditional marginal utility

$$
\Delta_i(S)=\ell(y\mid x,S)-\ell(y\mid x,S\cup\{i\}).
$$

The loss comparison uses the same target continuation or verifier task. Sample $S$ from partial useful groups, ordinary retrieved sets, and irrelevant/empty contexts. Include complete useful groups as candidate actions in a subset of training episodes: an empty-set marginal objective alone misses pure complementarity.

For an explored candidate set, one implementable ranking target is

$$
p_i^*(S)=\operatorname{softmax}_i(\Delta_i(S)/T_u),\qquad
\mathcal L_{\mathrm{route}}=-\sum_i\operatorname{stopgrad}(p_i^*(S))\log\operatorname{softmax}_i(s(q_S,k_i)/T_s).
$$

$q_S$ is extracted from the state after consuming $S$. An explicit no-additional-read candidate has zero incremental utility before any read-cost penalty. Single-record utility is the special case $S=\varnothing$, not the main definition of usefulness. REALM supplies a precedent for retrieval trained by downstream language-model supervision, not for this particular proposed estimator [^3].

Candidate construction mixes known useful groups, hard negatives, random exploration, and ordinary nearest neighbors. Recompute selected scores in a differentiable reranker when query/key gradients are needed, while freezing discrete index membership for that update. End-task reward can train group selection and invocation later; verifier-based and likelihood-based estimates are the initial lower-complexity alternatives.

### 6.5 Operational group retrieval and invocation

The learned-access milestone must demonstrate that routing actually supplies jointly necessary evidence. In tasks with a known support group $G_x$, report

$$
\operatorname{CompleteSupportRecall}=\mathbb E_x\left[\mathbf 1\{G_x\subseteq\mathcal R_x\}\right],
$$

where $\mathcal R_x$ is the union of records obtained before the decisive action. When there are several valid minimal groups, count success when at least one is fully covered; redundant supporting records need not all be retrieved. Also report ordinary useful-record recall, retrieval calls, and extra records fetched. High average recall does not imply complete coverage.

Train both one-shot group retrieval and state-conditioned follow-up retrieval. Sample conditional marginals and leave-one-out removal within useful groups; use full groups when individual marginals are uninformative. This is required in the learned-routing experiment, not deferred to a diagnostic appendix. Keep the exhaustive subset calculation confined to small synthetic fixtures; sampled comparisons are sufficient for the initial larger experiment.

Invocation can begin at task/action boundaries and later become a learned policy. Its penalty represents actual bandwidth, compute, or latency, not the inability to retain producer graphs. A write policy can use future utility and storage cost; append-only verified writes are simpler than jointly learning deletion and admission.

### 6.6 Combined objective and staged optimization

A practical objective is

$$
\mathcal L=\mathcal L_{\mathrm{task}}+\lambda_r\mathcal L_{\mathrm{route}}+\lambda_b\mathcal L_{\mathrm{bootstrap}}+\lambda_c\mathcal L_{\mathrm{cost}}+\lambda_s\mathcal L_{\mathrm{stability}}.
$$

Full compaction adds a later objective. After an uncompacted system learns useful composition, a cheap temporary-compaction objective may be introduced as a controlled training arm (Section 8.9), before persistent compaction exists. Keep an uncompacted task loss and a fixed or lagged reference so the system is not rewarded merely for making all memories uninformative. Loss coefficients are tuning parameters. Begin with oracle transfer **and composition**, then learned group-aware routing and adaptive invocation; preserve the oracle diagnostic throughout.

### 6.7 Teacher supervision and student-generated experience

Teacher trajectories supply ordinary action and response targets, support experiences, and outcome labels. A basic behavioral objective is

$$
\mathcal L_{\mathrm{imit}}
=
-\sum_t \log p_\theta(a_t^{\mathrm{teacher}}\mid h_t,\text{causally available memory}),
$$

where $h_t$ includes only observations available at that step. Use an explicit mask for observations and tool outputs that the controller is not being asked to predict. Auxiliary supervision for first-loop attempts and query heads can be added, but soft values are principally trained through their consequences on later tasks, not by matching a teacher's hidden-state coordinates.

When compatible output distributions are available, distribution matching is an additional option; different tokenizers require an explicit alignment scheme rather than comparing unrelated vocabulary indices. A teacher's completed trace is not proof that the student would reach the same intermediate state, and a teacher may succeed using information the student does not receive.

For that reason, introduce student-generated rollouts and verified outcomes after bootstrapping. Let the writer observe the student's own failures and repairs, then evaluate transfer to separate subsequent tasks. The first pass can be supervised to make a genuine attempt while learning to issue useful queries; the final objective should reward the entire attempt–retrieve–refine process. Successful memory writing should not require access to a privileged solution at inference.

There is no requirement to run online weight updates in order to test this loop. With all weights frozen, a student can complete a support task, write its experience, and improve on a later query. That is the first clean test of learning stored in the KB. Training the writer from the later query loss is the separate offline or episodic mechanism that teaches it how to make such writes useful.

## 7. Selective replay and gradient accounting

### 7.1 Mixed stored and freshly generated values

During training, let $b_i\in\{0,1\}$ select whether an entry is regenerated from its source for the current update:

$$
V_i=b_iW_\phi(\tau_i)+(1-b_i)\operatorname{stopgrad}(\bar V_i).
$$

The forward pass uses the selected value consistently across all dependent transforms. The reader and decoder train on the mixture. The writer receives task gradients through freshly generated entries; stored-only entries are constant inputs at that boundary. Downstream trainable transforms may still receive gradients when they are applied after that boundary.

This is an explicit mixed-live/cached objective, not automatically an unbiased estimate of an all-fresh objective. Selective regeneration can be sampled by age, expected utility, source diversity, and cost. Avoid selecting only high-frequency records, which would starve rarely retrieved but important experiences of writer training.

At inference, $b_i=0$: read stored payloads. Writing a new experience invokes the writer once to create its stored representation. This is distinct from re-encoding old source trajectories on every read.

The cache boundary must match the chosen storage layout. If a training read loads a canonical cached value, current per-space transforms can run on that detached value and receive their own gradients. If it loads an already transformed payload from disk, its producer-side transform is cached too: it receives no gradient unless that payload is regenerated and included in replay. Mixing these two paths without recording the boundary would silently change which parameters are being trained.

A production-oriented variant therefore defines the detached object as the tuple of actual stored payloads, while a simpler reference implementation detaches at the canonical value. Both are valid. Stored-only validation should use the deployment layout and precision, rather than silently applying current transforms to old canonical values when deployment will not do so.

### 7.2 The replay boundary

Choose and document the tensor boundary at which producer graphs are detached. One simple choice is the canonical value $V_i$; per-space transforms then remain on the consumer side. Another is the tuple of stored space-specific payloads, in which case replay must include their production transforms. The boundary determines which parameters receive replay gradients and what must be reproduced exactly.

For a canonical-value boundary, collect the downstream cotangent $g_i=\partial\mathcal L/\partial V_i$. The producer contribution is

$$
\nabla_\phi\mathcal L=\sum_{i:b_i=1}\left(\frac{\partial W_\phi(\tau_i)}{\partial\phi}\right)^\top g_i.
$$

This is a vector–Jacobian product. The representation-gradient separation is closely related to gradient caching in contrastive learning [^10]; activation checkpointing supplies the broader computation-for-memory precedent [^11]. The trajectory-backed memory application and its replay schedule are specified here.

### 7.3 Execution schedule

Generate selected values without retaining producer activations, recording parameter version and stochastic state. Run the consumer using detached leaf values, then accumulate gradients at those leaves while releasing consumer activations according to its checkpoint schedule. Replay selected producers in small batches and apply the saved cotangents. Take the optimizer step only after all consumer-side and producer-side contributions for the update have accumulated.

If an entry is used several times with the same produced tensor and parameter snapshot, sum its cotangents before replay. If separate calls used different stochastic realizations or different conditioning, retain separate replay identities. Sharing decoder and writer parameters is valid: their direct consumer gradient and their replayed producer gradient are both accumulated before the update.

### 7.4 Conditions for exactness

Replay must reproduce the computation that generated the value used in the forward pass: source tokens, masks, memory inputs, random-number states, adapters, quantization behavior, normalization state, and parameter snapshot. Compactability interventions add payload-noise seeds, temporary cluster assignments, field responsibilities, surrogate choices, and compression masks to this record when they lie across the replay boundary. Freeze the discrete read plan for that update. Avoid an optimizer or stateful normalization update between forward generation and replay.

Using an old stored value in the forward pass and a fresh writer Jacobian in the backward pass is a surrogate gradient. Likewise, replay can reproduce a chosen straight-through rule for quantization, but that rule remains a surrogate; it is not the mathematical derivative of the discontinuous quantizer. Label these choices when used; do not conflate them with exact replay.

Replay treats the recorded historical trajectory as data. It does not differentiate through past discrete actions, environment transitions, or an earlier training process that created the trajectory. Learning the historical action policy requires a separate imitation or policy-gradient objective.

### 7.5 Replay within the reader and nested dependencies

For an additive reader aggregate, replay chunks to compute the local vector–Jacobian products while retaining only the small shared residual checkpoints and required buffers. Backpropagate rounds in reverse; each replayed round uses the same completed shared state as its forward computation. This controls per-record activation memory without restricting the neighborhood itself.

Producer trajectories may themselves reference earlier reads. The first implementation should replay the writer conditioned on recorded read values and stop gradients at those historical boundaries. This is a defined training objective, not a limit on the number of reads the agent can make. Later experiments can replay selected dependency subgraphs with explicit versioning and measured cost. Unbounded historical differentiation is not assumed.

The consumer trajectory still has its own activation and cache requirements. Checkpoint, recompute, or offload those independently. Report all replay I/O, retained cotangents, host buffers, and repeated compute; a small producer-activation peak is not the whole training footprint.

## 8. From read-time superposition to cluster compaction

### 8.1 What compaction should preserve

After the writer and reader have learned useful composition, train a cluster encoder to replace many records with a smaller representation. The target is not a prose summary, an isolated cluster answer, or merely the mean of its value vectors. It is the cluster's contribution to the reader under relevant queries, selection plans, and shared states.

**Compactability hypothesis.** Related experiences have sufficiently redundant conditional contribution functions over the relevant query/state/selection distribution that an amortized encoder can represent them in fewer stored bytes. Training can encourage this redundancy where task distinctions permit it. Successful composition alone does not establish it.

A simple counterexample clarifies the boundary. For $M_C(q)=\sum_{i\in C}\operatorname{ReLU}(q-x_i)$ with distinct $x_i$, the slope changes at each $x_i$. Fewer weighted synthetic records evaluated by the same one-breakpoint function have fewer breakpoint locations and cannot reproduce the sum exactly for every real query. This is a valid obstruction to that exact restricted representation, not to approximate distributional compaction, richer codes, or a reader trained to be compactable. Test redundant clusters against independent facts, incompatible rules, and rare exceptions rather than requiring every cluster to compress equally.

For cluster $C$ in one space, define the additive contribution and normalization mass at round $t$:

$$
M_{Cj}^{(t)}(q,R,\rho_C)=\sum_{i\in C\cap\mathcal N(q)}w_i^{(t)}\Phi_t(x_i,q,r_j,e_j),\qquad \mu_C^{(t)}=\sum_{i\in C\cap\mathcal N(q)}w_i^{(t)}.
$$

$\rho_C$ carries the relevant selection information. Summing contributions and masses over disjoint clusters reconstructs the original pre-normalization aggregate. The nonlinear residual update occurs after that merge. Cross-cluster interaction is preserved through subsequent shared-state feedback if the conditional contributions are preserved sufficiently well.

### 8.2 Query-dependent membership is part of the target

A query may select only part of a cluster. Replacing it with the contribution of every member changes the read operation even if each member is encoded perfectly. Moreover, exact top-$k$ membership depends on the wider index, not just the query and the contents of one cluster.

Initially preserve original routing keys and the original selection plan. Supply selected child handles/scores or a sufficient selection descriptor to the compact reader, and measure the cost of that metadata. A compact reader that cannot distinguish selected subsets can only approximate their behavior over its training distribution. Alternatively, introduce cluster-level retrieval explicitly and evaluate it as a separate change in addressing semantics.

Preserving many keys while sharing a compact payload can save value bytes without saving comparable index bytes. Deduplicate repeated payload fetches, but do not discard the selected-child information or multiply a cluster's contribution by the number of keys that happened to match.

### 8.3 Two compact representations

**Synthetic-record representation.** An amortized encoder maps the original cluster to a small set of weighted synthetic records:

$$
\operatorname{Compact}_\eta(C)=\{(\widetilde x_a,\alpha_a)\}_{a=1}^{r},\qquad r\ll |C|.
$$

The existing reader consumes these records with selection-aware gating. Train their summed contributions and normalization statistics to match the original cluster. Nonnegative weights can represent effective multiplicity; signed information remains inside the contribution vectors. Full-cluster reads provide the cleanest initial setting, followed by partial-selection tests.

**Conditional-code representation.** The encoder produces $c_C$, and a small learned function predicts $\widetilde M_t(c_C,q,R,\rho_C)$ and normalization mass. This can express a richer conditional contribution than a few ordinary synthetic records, but adds an interface and resident model parameters. Compare both at matched payload bytes and reader compute.

A code optimized separately for every training cluster is an informative high-capacity baseline, not a formal upper bound and not evidence that the compaction encoder generalizes. Evaluate amortized encoding on previously unseen clusters without per-cluster gradient optimization.

### 8.4 Distillation objective

Freeze a teacher reader snapshot initially, and sample queries, retrieval plans, and residual states from its executions. A candidate loss is

$$
\mathcal L_{\mathrm{compact}}=\mathbb E\left[\sum_t\|\widetilde M_t-M_t\|_F^2+\lambda_\mu\|\widetilde\mu_t-\mu_t\|^2\right]+\lambda_D\mathcal L_{\mathrm{decoder}}+\lambda_B\mathcal L_{\mathrm{bytes}}.
$$

The decoder term can combine task supervision and matching the uncompressed system's output distribution. Contribution matching is a convenient intermediate target; downstream behavior remains decisive. Byte cost should count the actual stored representation, including weights and metadata, rather than an arbitrary latent norm.

Train with mixed compacted and uncompacted clusters. Include states reached by the partly compacted reader to reduce exposure mismatch. Early-round approximation errors change later shared states, so small per-round error alone does not guarantee a small final task error. Validate full rollouts, not only teacher-forced intermediate-state matching.

### 8.5 An exactly mergeable baseline

If the per-record contribution factorizes as $\Phi_t(x_i,q,R)=A_t(q,R)b_t(x_i)$ and the selected cluster members are fixed for the read, then

$$
\sum_{i\in C}\Phi_t(x_i,q,R)=A_t(q,R)\left(\sum_{i\in C}b_t(x_i)\right).
$$

The statistic in parentheses is exactly additive and mergeable. Store the statistics needed for each round, and apply the nonlinear residual update after merging. Arbitrary query-dependent subset selection or nonlinear pre-pooling gates break this simple sufficient-statistic argument unless their effects are also represented.

This restricted baseline measures how much benefit the richer reader obtains from query-dependent nonlinear processing before pooling. The compaction encoder then tests whether much of that extra behavior can be approximated with a compact code on the actual task distribution.

### 8.6 Cluster construction and reversible rollout

Cluster by conditional response similarity as well as address distance. One inexpensive diagnostic is to evaluate records at a small shared set of query/state probes, concatenate their contributions and masses, and compare these response signatures. Key proximity is a proposal mechanism, not proof of behavioral redundancy. Probe distributions should include exceptions; a low average error can otherwise hide a small but decisive response.

Use disjoint clusters as the exact reference. Compare them with the local overlapping fields in Section 8.7, rather than treating overlap as an automatic improvement. An experience may participate in multiple training clusters without being copied into multiple inference payloads.

Retain source lineage, membership, teacher version, and quality checks. New observations can live in an uncompacted delta store until recompression. Validate compact candidates in a shadow read path, then replace payloads with rollback available. Test repeated compaction separately. Preserve indivisible facts as originals, or allocate a larger code, when a small compact representation loses useful distinctions; this is adaptive compression rather than a failure of the entire memory architecture.

### 8.7 Overlapping local compaction fields

A **field** here is a local domain with a shared compact representation, not a new topology or physical field. Nearby records can participate in a few overlapping domains. Partition-of-unity networks provide a relevant precedent for combining local approximation models; their results do not establish latent-memory compactability [^22].

There are two distinct implementations.

**Training-only multi-membership.** Re-form local clusters across minibatches, or evaluate two alternative clusterings of the same selected records. Use a shared compactor and the same uncompacted target. Each record is trained in several neighboring groupings, but a deployment generation may still use one disjoint partition. This is the simplest overlap experiment: it encourages group-independent compact behavior without committing to extra stored replicas. Average losses over views or correct sampling weights so frequently included records do not silently dominate training. Changing groups without applying a bottleneck or compaction loss creates no compactability pressure by itself.

**Persistent overlapping fields.** Let $\alpha_{ic}\ge0$ be the responsibility of field $c$ for record $i$, with a small number of nonzero entries and

$$
\sum_c\alpha_{ic}=1.
$$

For the original read-selection indicator $z_i(q)$, combine the numerator and mass into an augmented record contribution

$$
g_{tj}(x_i;q,R)=\big[w_i^{(t)}\Phi_t(x_i,q,r_j,e_j),\;w_i^{(t)}\big].
$$

The field target is

$$
F_{ctj}(q,R,z)=\sum_i\alpha_{ic}z_i(q)g_{tj}(x_i;q,R).
$$

Then

$$
\sum_cF_{ctj}=\sum_i z_i(q)g_{tj}(x_i;q,R)
$$

exactly, before approximation. Split both contribution and normalization mass with the same responsibilities. Overlap must not multiply evidence. Distinct retrieval spaces remain intentional distinct views; this conservation rule is for splitting one space's original contribution across fields.

A fixed two-nearest-center soft assignment in a normalized address or response-feature space is a sufficient starting point. Retain the same canonical value for all memberships. Query-conditioned field decoders may use local coordinates internally while targeting the same original $g$. Additional memberships must be evaluated at a stated total storage/read budget. More total disk storage can still be worthwhile if it reduces read cost or resident memory at the required quality: the overarching project explicitly permits that trade. Report such a result as a better resource allocation, not as net store compression when bytes actually increase.

**Field participation is not a free retrieval change.** The exact identity sums all field shares of the selected records. Fetching only some fields can omit part of their mass. An active-field renormalization would have to occur per record, and a code built for fixed responsibilities does not generally implement that transformation from one scalar correction. Initially fetch all needed field shares under the frozen read plan. Later evaluate field-level retrieval as its own approximation. The partial-selection descriptors from Section 8.2 remain necessary; overlap does not reconstruct arbitrary excluded or included children automatically.

Use authorization-homogeneous fields. Record any multi-membership lineage so deletion invalidates every affected field. At first these can be simple structural invariants in a single-domain prototype rather than a separate security infrastructure project.

#### Hierarchical halo reads

The corpus-scale design should also evaluate a multiresolution read: a small raw
core near the query plus a bounded halo of overlapping compact fields. Coarser halo
levels cover more logical descendants while exposing only a fixed number of
synthetic records to the main reader. Construct fields with density-adaptive centers
or a metric cover in the learned address/response space; a uniform grid is only a
low-dimensional control because its cell count grows exponentially with dimension.
Train with the discrete field plan fixed for each forward pass, backpropagate the
task loss through every temporary compactor level, and later serve immutable
materialized codes without fetching or re-encoding their descendants.

Representational redundancy is desirable: one raw point may influence many
overlapping fields, alternative clusterings, and hierarchy levels. The reader does
not receive provenance or decide whether records are statistically independent;
dependence is pervasive. Provenance remains operational metadata for deletion,
authorization, contamination analysis, and rebuilding descendants.

There are two different experimental semantics. In **equivalence compaction**, an
overlapping construction is trained to approximate one declared raw aggregation.
Partition-of-unity responsibilities and mass preservation make the approximation
invariant to an arbitrary refinement of that construction. These coefficients are
numerical aggregation weights, not epistemic confidence or independence estimates.
Across levels, equivalence compaction uses a replacement frontier or residual codes
so its target remains well defined.

In **computational expansion**, many overlapping compact records deliberately enter
the downstream computation. They need not sum to one globally or pretend to be
independent evidence. Their multiplicity, hierarchy level, and retrieval schedule
are part of the learned architecture, and downstream task training determines how
the reader uses the redundant states. Compare this against a matched-compute control
and keep the schedule stable enough that changes in database density do not silently
change the model. A conservative first implementation can normalize within each
local compaction and then fuse several redundant views; it does not need to expose
source provenance to the neural reader.

Condition the shared compactor on simple local geometry. Each positioned input can
carry its payload and normalized offset from the overlapping neighborhood's center.
The neighborhood can carry its radius or density and hierarchy level. Compacting the
set of these tuples remains permutation invariant. Begin with this information only;
do not prescribe roles for neighboring summaries or add specialization losses unless
measured collapse makes them necessary. Different centers and overlapping input sets
already provide the basic asymmetry.

The initial payload compactor does not receive the absolute center coordinate. Its
published summary receives a separate absolute retrieval key, but summary formation
uses payload content and relative geometry. This avoids coupling one shared local
operator to an arbitrary coordinate gauge or writer-generation drift. Compare an
explicit center-conditioned arm only if the invariant baseline shows that different
semantic regions need different compression behavior; pin the key-space generation
for that comparison.

Prefer relative coordinates in a versioned learned metric over raw absolute key
coordinates. The present keys are normalized cosine addresses and can drift between
writer generations; a hierarchy must pin its coordinate transform or rebuild its
neighborhoods. A compact record needs its own learned/versioned region key rather
than borrowing an arbitrary child's key. Train the retrieved set of positional
summaries under task and raw-descendant objectives. Add dropout, neighbor context, or
explicit diversity pressure only in response to a demonstrated failure.

Train every level against raw descendants as well as immediate child codes so
recursive error does not become the teacher. Add direct-versus-recursive consistency,
downstream task loss, and contribution-and-mass losses when an equivalence target is
used. Keep raw exceptions where a field exceeds its behavioral error budget.

Published codes carry computation forward even though publication detaches their
historical gradient graph. Train the shared transition on shallow sampled subtrees
with raw-descendant, task, and consistency targets at every depth, analogous to
learning local steps of a long iterative process rather than backpropagating through
its entire history. This makes task-derived computation portable in external state,
but codes remain versioned against the reader and compactor that interpret them.

The hierarchy is a retrieval structure, not an authorization mechanism. A compact
node may be used only when its complete support is eligible for the query's scope and
time, unless an explicitly selection-conditioned representation has been trained and
validated. Keys or region descriptors for compact nodes need their own versioned
index and routing evaluation. At training time, dynamic descendant graphs are bounded
samples; at inference time, the stored code is the read payload. Report logical
descendant fan-in, physical payload count, reader and compactor compute, total bytes,
and quality by level so a larger receptive field is not mistaken for free capacity.

### 8.8 A mechanical reason locality can help

For fixed query and state, consider a field with static nonnegative weights $a_i$, total mass $A=\sum_i a_i$, and continuous record coordinates with weighted mean

$$
\bar x=\frac{1}{A}\sum_i a_i x_i.
$$

Assume the vector function $g(x;q,R)$, including any nonlinear gates and normalization-mass component, has second-derivative operator norm at most $H$ throughout the convex neighborhood. A Taylor expansion at $\bar x$ gives

$$
\left\|\sum_i a_i g(x_i;q,R)-A g(\bar x;q,R)\right\|
\le\frac{H}{2}\sum_i a_i\|x_i-\bar x\|^2.
$$

The first-order term cancels because $\sum_i a_i(x_i-\bar x)=0$. This is a conditional derivation, not a guarantee that a learned reader satisfies a useful bound. It identifies two actionable levers: make fields locally coherent in the coordinates the reader uses, and reduce unnecessary curvature before pooling. Smoothness in the original key alone is insufficient when values or applicability conditions differ.

A simple compaction baseline therefore stores field mass and mean, optionally a diagonal or low-rank covariance, and learns a conditional decoder from those moments. Full covariance can be too expensive; charge its actual width. If query selection changes which records count, the corresponding moments change too. The static moment baseline applies first to full-field reads or an explicitly supported family of selections, not arbitrary child subsets.

For the factorized local MLP, a further exact special case is useful. With unit gates and a shared ReLU activation pattern over a field for a particular query/state, the per-record map is affine there. A weighted mean plus mass exactly preserves that round's contribution. Different activation patterns, nonlinear gates, and later states break the special case. ReLU is consequently not categorically incompatible with compaction, and merely replacing it with a smooth activation is not a sufficient strategy.

The exactly mergeable model in Section 8.5 supplies an even stronger structural bias: move most conditional nonlinearity after pooled features. A hybrid can use a mergeable component plus an expressive per-record component. Start with the two separate baselines before adding an elaborate learned division of capacity.

### 8.9 Compactability-aware training without a persistent compactor

The most direct intervention is **temporary compressed reads**. Once the uncompacted system uses memory, select a small portion of training reads, form local groups, replace each with a smaller surrogate, and continue the actual reader and decoder. Original records remain available; no index rewrite is required.

The cheapest version needs no learned compactor. For a small coherent group with normalized nonnegative weights $a_i$ satisfying $\sum_i a_i=1$, compute $\bar x_C=\sum_i a_i x_i$ and penalize the normalized contribution mismatch

$$
\mathcal L_{\mathrm{merge}}
=\mathbb E_{C,q,R,t,j}\left[
\left\|g_{tj}(\bar x_C;q,R)-\operatorname{sg}\!\left(\sum_i a_i g_{tj}(x_i;q,R)\right)\right\|^2
\right].
$$

Carry the group's total multiplicity separately. This local merge-consistency penalty is zero for affine contributions and directly encourages replaceability at the intended boundary. Average only continuous representation coordinates, not identifiers, authorization labels, or categorical metadata. Begin with the same small groups used for full-group compaction tests, and evaluate the actual mean-replacement path as well as the penalty. With few sampled groups, the existing raw contribution evaluations are reusable and only the mean's contribution needs an extra local MLP evaluation at each sampled state.

Keep loss scales tied to a reference and preserve both raw and replaced-path task utility. A decreasing latent mismatch by itself could reflect rescaling or lost information rather than useful compaction. Apply the penalty locally, not as a demand that unrelated or contradictory records become interchangeable. The richer two-synthetic-record variant tests how much useful behavior a single mean cannot preserve.

A small prototype compactor can then return moments or two synthetic records from a group of eight. Use local pooling as an initialization to compare against unrelated random vectors. Vary grouping and compression ratio across training so the interface does not depend on one permanent cluster layout. Dataset-condensation work is a useful analogy for matching a smaller synthetic set in feature space; it is not evidence that the proposed conditional reader can be compressed at a particular ratio [^24].

A candidate objective is

$$
\mathcal L_{\mathrm{aware}}
=\mathcal L_{\mathrm{task}}^{\mathrm{raw}}
+\lambda_a\mathcal L_{\mathrm{task}}^{\mathrm{temporary\ compact}}
+\lambda_f\mathcal L_{\mathrm{field\ contribution}}
+\lambda_d\operatorname{KL}\!\left(\operatorname{sg}(p_{\mathrm{reference}})\,\|\,p_{\mathrm{temporary\ compact}}\right).
$$

The field term matches unnormalized contributions and masses at corresponding states; the compact task term runs through the compact system's own state evolution. Initially freeze a reference writer/reader snapshot while training the prototype compactor. Then optionally adapt the live writer and reader under both raw and compact task losses, keeping the frozen or lagged reference as an anchor. Do not merely shrink the targets until they are easy to match.

The compactor receives the source payloads and legal metadata, not the future answer, query-specific labels, or a precomputed target contribution disguised as a stored code. A reusable compact code is formed before the evaluated query; only its readout is query/state-conditioned. Validate on held-out clusters and query uses. Materialize evaluation clusters and codes independently of the held-out query batch; query-selected regrouping is allowed as a training augmentation but is not the persistent-compaction evaluation. Deployment reads the actual serialized code, never the original cluster hidden behind an inference-only teacher branch.

**Recommended initial sweep centers, not established optima:** groups of two or four for the mean baseline, and groups of eight for two synthetic records; a 10% or 25% temporary-compaction fraction; two alternative local groupings; and an unmodified control. Start by changing one factor. Persistent overlapping payloads are a later comparison, not a prerequisite.

### 8.10 Noise, precision, and representation dropout

Noise is useful when it models the information the deployed representation should not depend on. It is less direct than actually training through a smaller representation.

**Storage-aligned payload perturbation.** Apply modest noise after controlled normalization/scaling of the stored-value representation, or simulate the chosen quantizer. For a uniform quantizer with step $\Delta$, additive noise in $[-\Delta/2,\Delta/2]$ is a familiar training surrogate in learned compression [^23]. Actual stored-code evaluation must still use the real quantizer, precision, scale metadata, and decoder. A smaller nominal bit width is not a measured storage saving until serialized bytes are counted. Noise variance is defined relative to a fixed or constrained signal scale so the writer cannot trivially evade it by increasing its amplitude.

Use the same perturbation policy on live and cached payloads. A Gaussian-noise arm can sweep zero and small RMS-relative amplitudes before trying aggressive corruption. Bishop's small-noise analysis provides a regularization precedent under its assumptions, not a theorem that noise reduces cluster size or conditional response rank [^21]. Independent records can remain independent after their responses are smoothed.

**Query/state coverage.** Sample additional legitimate queries and on-policy reader states for compaction distillation. Small perturbations can broaden local coverage, but do not impose invariance across changes that alter the correct response. Initially freeze hard retrieval membership during such a reader-only experiment. Query jitter that changes which support is found tests routing robustness as well as compaction and should be labeled separately.

**Representation dropout, not arbitrary evidence deletion.** Alternate raw and compact representations, or alternate local grouping views while preserving the relevant source support. Do not drop one of two uniquely necessary experiences and insist on the same correct answer; that trains the system to ignore missing evidence or exploit shortcuts. A duplicate field view may be dropped only with explicitly correct responsibility/mass handling. Grouping augmentation is the cleaner first implementation.

Noise addresses sensitivity and bit-level robustness; local fields address approximation domain; temporary compaction directly addresses replaceability. Keep those effects separable in the ablations rather than bundling them as one unexplained regularizer.

### 8.11 Shared structure, exceptions, and compaction criteria

Compact common structure while keeping genuinely distinct residual information available. The simplest implementation partitions a candidate group into a compacted subset and a small uncompacted exception subset. Exclude exceptions from the compact target, then add their ordinary contributions. This avoids double-counting and requires no inference-time reconstruction of the original group. A later residual-code design must subtract the contribution already represented by the shared component rather than adding a second full copy.

Use held-out behavioral error to allocate capacity: retain a small code when sufficient, use more synthetic records or split the field when useful distinctions are lost, and leave difficult records uncompacted. Rare exceptions need dedicated probes. Uniform compression ratios across heterogeneous clusters are not a requirement.

For a quality score where higher is better, report retained memory benefit

$$
\operatorname{RMB}
=\frac{Q_{\mathrm{compact}}-Q_{\mathrm{no\ memory}}}
{Q_{\mathrm{uncompacted}}-Q_{\mathrm{no\ memory}}}.
$$

Use this ratio only when the uncompacted benefit is positive and statistically resolved; report the underlying scores and paired task-family uncertainty intervals. Do not clip values above one or below zero. For a compactability-aware model, compare against both its own raw path and the original unregularized raw baseline so becoming easier to compress by becoming less useful cannot look like progress.

A useful aspirational checkpoint is fourfold payload reduction retaining 90% of memory-derived benefit. These are proposed project criteria, not a feasibility theorem or an automatic stop rule. The primary output is a quality–bytes–compute curve with tail/exception results. Report net savings including all memberships, keys, selection descriptors, synthetic weights, quantization scales, exception payloads, and additional resident parameters. A profitable modest compaction result remains worth keeping.

## 9. Memory lifecycle, staleness, and online learning

### 9.1 Record schema and visibility

Each logical record needs a canonical identifier; source/trajectory handle; source span and availability time; outcome and provenance type; canonical key; per-space keys and payload locations; writer, reader-compatibility, transform, and index versions; data-use metadata; and any parent-cluster or dependency links. Keep operational fields separate from learned value dimensions so bookkeeping remains inspectable.

Writes become visible through an atomic commit after their payloads and index entries are ready. A trajectory sees a declared snapshot plus explicitly permitted subsequent writes. Concurrent readers must not observe a key pointing to a partially written payload. Tombstones and deletion lineage apply to keys, latent payloads, compacted derivatives, and caches.

### 9.2 Address drift and value compatibility

Key staleness, value staleness, and reader compatibility are distinct. Freshly generating a selected value during training does not repair an address that can no longer be found. A new reader may also interpret an old latent code incorrectly even when its key remains useful.

The target continuously trains query and write keys. Each optimizer step pins one
bank epoch, then publishes touched key revisions and corresponding index deltas only
after replay and the optimizer update complete. Representation versions and adapters
remain explicit so a reader never silently interprets incompatible stored bytes.
Background audits regenerate stale and low-traffic records; raw scores across
incompatible revisions are not assumed calibrated.

At training time, regenerate sampled keys and compare retrieval against a fresh reference subset. Include random audits of records not currently retrieved: refresh-on-hit alone cannot discover entries that have become unreachable. Measure stale-index recall, downstream utility, and key displacement, not just mean embedding similarity.

For a controlled bootstrap experiment, “stable addressing” means stabilizing the
complete mapping into the indexed key space. Freezing only the final key projection
does not prevent drift when its upstream writer features change. That bootstrap is
not the lifecycle target: jointly learned query and record keys, revisioned index
updates, drift audits, and scheduled regeneration are.

### 9.3 Memory growth versus online weight updates

With weights frozen, the trained writer can encode a newly completed experience and insert its value into the KB. This is online learning in external state and should be evaluated independently from online gradient updates. Later weight adaptation creates an additional nonstationarity problem and requires version transitions for stored codes and keys.

Treat proposed lessons, observed facts, and verifier-supported procedures differently. Preserve applicability conditions and failures. Confidence should derive from evidence and measured performance, not from how assertively a source trajectory states its conclusion. Repetition of an unverified lesson should not silently become independent corroboration.

Regeneration and compaction are maintenance operations, not the inference read path. Prioritize them using age, utility, drift, and storage pressure, with some exploration of low-traffic records. Preserve rare useful exceptions rather than deleting solely by read frequency.

The corpus-scale curriculum uses a **prequential event stream**, not a bank that is
fully populated before every training example. For event (t), retrieval is limited
to an explicitly committed visibility frontier strictly before (t). The trajectory
and its loss complete against that prefix; only then may its atomic set of authored
records become visible to later events. This exposes the controller to growing bank
sizes and prevents the current target or its derived memory from contaminating its
own evaluation.

Event time, ingestion time, and original source availability time are separate
fields. Visibility uses the earliest admissible causal boundary and retains the
source snapshot, authorization scope, writer version, and parent-read lineage.
Curriculum reports stratify performance by bank-size band and include stale,
redundant, conflicting, low-value, and subsequently invalidated records. A small
declared core may seed a stream, but heldout target material cannot enter it
before its evaluation event. Its learned representations can subsequently change
under the same revision and causality rules as other logical records.

Garbage collection publishes tombstone, supersession, deduplication, or
contribution-and-mass preserving compaction revisions to the mutable bank. It never
changes immutable event history or the physical revisions pinned by a logged read.
Training retains raw evidence links and samples both retained and collected material
so deletion by popularity does not silently erase rare useful cases.

### 9.4 Trust boundaries

External experiences and documents may contain conflicting, misleading, or instruction-like content. Treat memory as evidence, not as a higher-priority instruction channel. Preserve source permissions and tenant boundaries before retrieval; a latent encoding is not a substitute for access control. Do not compact incompatible authorization domains into a shared code and rely on a learned selection mask to prevent information flow. Use structurally authorization-homogeneous clusters and fields. Deleting a child key does not remove its contribution from a shared code: invalidate every affected compact derivative and cache, then regenerate from authorized surviving sources before reuse. In a single-domain research prototype, these can begin as schema and test invariants rather than a full multi-tenant service. Evaluate poisoned-record injection, cross-task contamination, and deletion propagation as explicit robustness tests.

Because latent records are difficult to inspect directly, retain a provenance view and optional diagnostic decoder for auditing. A plausible textual decoding is not proof of the latent record's actual influence; use intervention tests and outcome measurements as well.

## 10. Resource model and engineering constraints

### 10.1 What counts as resident memory

Inference VRAM must include all simultaneously resident components:

$$
M_{\mathrm{VRAM}}=M_{\mathrm{weights}}+M_{\mathrm{KV}}+M_{\mathrm{working}}+M_{\mathrm{buffers}}+M_{\mathrm{index,GPU}}+M_{\mathrm{runtime}}.
$$

$M_{\mathrm{weights}}$ includes the decoder, writer-specific modules, query projections, reader, and any conditional compaction decoder. $M_{\mathrm{working}}$ denotes their non-KV activation working sets, not another copy of the reader weights. $M_{\mathrm{buffers}}$ covers fetched values and transfer staging, and $M_{\mathrm{runtime}}$ covers additional framework/workspace overhead. Assign each allocation once; do not count shared parameters or buffers twice. Training additionally includes gradients, optimizer state, checkpoints, and replay working sets. Report host RAM and disk storage separately; moving everything from VRAM into host RAM is a different result from demonstrating a predominantly disk-resident store.

The resident parameter count, the number of trainable external scalars, generated latent payload bytes, source-archive bytes, and accessed bytes per task are different quantities. A small backbone with a large learned KB is not a small-total-capacity model. It is a selectively resident system.

### 10.2 Illustrative payload calculation

Assume eight value positions, decoder width 1,024, and two bytes per stored scalar. A canonical value occupies 16,384 bytes, or 16 KiB. One million such values occupy 16.384 GB (about 15.26 GiB), excluding keys and metadata.

For a four-space toy schedule, take total payload widths $(256,512,1024,2048)$ scalars and retrieved counts $(128,64,32,16)$. Every space contributes 64 KiB; together they fetch 256 KiB of idealized value payload per read. The response to the decoder is still only 16 KiB. Storing every one of a million records in all four spaces costs 7.68 GB for those transformed payloads alone.

These figures are arithmetic examples, not accuracy or latency estimates. They exclude the optional canonical copy, compression scales, alignment, indices, metadata, and the source archive. They also say nothing about how much information those widths can preserve. Measure actual serialized bytes and physical I/O, especially for small records and cold caches.

An early storage-only harness can already test the intended record sizes. In the example, 240 selected records each requiring a distinct 4 KiB block would cause 960 KiB of physical payload reads, 3.75 times the 256 KiB logical payload, before index traffic. This is a conditional arithmetic scenario, not a claim about a particular filesystem or a latency prediction. Packing, cache reuse, coalescing, alignment, and metadata can change it. Run the harness alongside the learning prototype rather than first building asynchronous decoder integration. Enforce a host-cache budget or a sufficiently large working set before calling a result predominantly disk-resident.

### 10.3 Indexing, caching, and concurrency

Disk-backed approximate search is a viable systems starting point; DiskANN provides a published SSD-oriented precedent [^15]. Its benchmark results are not latency forecasts for learned keys, frequent updates, multiscale payloads, or this reader. Use a simple exact index first for model debugging, then introduce approximate search and storage with a measured recall/latency curve.

A practical layout shards the mutable index and groups payload revisions for
efficient fetches. A hot delta can sit over compacted base arrays, but both are
physical parts of one logical bank and consolidation must preserve revision history.
Cache keys must include representation version. Query coalescing and shared payload
fetches can reduce repeated I/O, but must preserve each request's selection plan and
authorization filters.

Report warm-cache and cold-cache tests, storage hardware, page-cache state, batch size, concurrency, and queue depth. Include p50/p95 latency and throughput, not only single-query averages. Maintenance traffic competes with reads and should be included in sustained-load experiments.

### 10.4 Compression, precision, and bandwidth

Soft-token compression can reduce decoder sequence length while using more disk bytes than the original text. Quantized payloads may reduce transfer volume but introduce another representation mismatch; train or calibrate for the deployed precision. Compare stored-value and fresh-value quality directly to detect a training/deployment gap.

Budget the reader as well as the decoder. If dense composition dominates wall time, factorize local MLPs, cache record projections, reduce unnecessary rereads, or compact clusters. Do not attribute an improvement to disk storage if it actually comes from an unreported large resident encoder.

## 11. Evaluation: proving transfer, composition, and substitution

### 11.1 Experimental tracks

**Interface track:** with oracle routing, determine whether newly written soft memories improve a frozen-weight student's performance on new related tasks. Compare oracle text, oracle latent memory, and no memory. This isolates whether information is usable before testing approximate retrieval.

**Composition track:** from the first oracle experiment, construct tasks requiring two or more separately learned factors. Hold out factor combinations and environments; ensure no source contains the full evaluation solution. Remove memories individually and in groups, and counterfactually replace one support with an otherwise matched experience that reverses the relevant rule. Correct behavior should change in the predicted direction. Compare separately supplied latent records, one-round pooled synthesis, multi-round pooled synthesis, and multi-round fixed-slot cross-attention. Keep exact routing and one space until this mechanism is measured.

**Substitution track:** sweep backbone size, external memory size, read frequency, reader rounds, and recurrent compute. Measure the minimum resident footprint meeting a quality target and a latency ceiling. A useful formal target is

$$
M^*(Q_0,T_0)=\min M_{\mathrm{VRAM}}\quad\text{subject to}\quad Q\ge Q_0,\;T_{95}\le T_0,
$$

with host RAM, hardware, and workload fixed or explicitly constrained. Plot this frontier against disk budget. Improvement of one fixed small model is evidence of augmentation, not yet evidence of capacity substitution.

New random rules generated after weights are frozen are a strong transfer test but not, on their own, a fair substitution test against a larger model denied those rules. In the substitution protocol, every practical comparator receives a legitimate route to the same relevant information: strong support-text retrieval, equivalent context, or an explicitly matched knowledge-acquisition procedure. Measure which system meets the workload's quality and latency targets with less resident memory, not which system was given additional facts. Separate support-access and controller-execution limitations with oracle text.

**Compaction track:** replace increasing fractions of memory with compacted representations, including unseen clusters and unseen mixtures. Measure retained quality, contribution error, payload/index bytes, read latency, and maintenance cost. Evaluate both original-key and cluster-level retrieval to separate value loss from addressing changes.

### 11.2 Required baselines

| Baseline or intervention | Main question it answers |
|---|---|
| Same student without memory; same student with extra loops only | Are gains attributable to memory rather than additional computation? |
| Strong text retrieval with the same allowed sources | Does the latent interface outperform ordinary evidence delivery? |
| Oracle text and oracle latent reads | Is the bottleneck retrieval, representation, or controller execution? |
| Individual latent records without learned composition | Does pooled synthesis add value beyond supplying compressed examples? |
| One-round versus multi-round pooled reader, with an extra-compute control | Does residual feedback improve composition rather than just add compute? |
| Single-space pooled reader and fixed-slot cross-attention, both before and after compaction | Which reader offers the better quality–bytes–compute trade? |
| Single-space versus multiscale retrieval | Does the retrieval hierarchy add value beyond the composition mechanism? |
| Same selected IDs, scores, keys, and metadata with nulled or shuffled values | Is useful information carried through the intended payload channel? |
| Directly trainable external values or memory layers | Does amortized experience writing generalize beyond stored training slots? |
| Larger quantized or offloaded decoders with access to the same relevant information | Is the deployment trade better than practical information-matched alternatives? |
| Uncompacted memory, synthetic records, conditional codes | What quality–bytes–latency trade does compaction achieve? |

No single resource match answers every question. Provide quality at matched VRAM and latency, quality at matched inference compute, and quality with matched source/training-data access. Charge teacher generation and replay to training cost. Memory Layers at Scale is a relevant external-capacity comparator, not a substitute for these controls [^13].

### 11.3 Metrics and leakage controls

Task metrics include verified success, exactness where required, tool efficiency, and calibration. Memory metrics include complete-support recall, useful-record recall, null-read behavior, writes per solved task, conditional marginal utility, source diversity, and measured multi-record interaction. Compaction metrics include retained memory benefit, actual net storage, and error on explicitly sampled exceptions. Systems metrics include peak VRAM, host memory, stored bytes, physical bytes read, cache hit rate, latency distribution, throughput, and replay overhead.

Stored-only evaluation is an invariant from the first successful prototype: serialize payloads in the chosen deployment layout and precision, reload them, and use only that read boundary. Current producer transforms may not be silently reapplied unless deployment actually performs them.

Freeze all weights while growing the evaluation KB to measure learning in external state. Separately evaluate adaptation with weight updates. Split by repository or task family, enforce chronological visibility, remove near-duplicate solutions, and document whether evaluation memories may accumulate across tasks. Keep metadata from leaking labels or future verifier outcomes into query-time inputs. Randomize opaque IDs and freeze the same routing plan during value ablations. Nulled and shuffled values are complementary probes; null vectors alone can be an out-of-distribution intervention. Learned keys may legitimately carry semantic information, so a surviving key-only effect is not automatically leakage. Attribute that effect correctly rather than calling it value composition.

Use multiple seeds and uncertainty intervals at the task-family or repository level when examples share context. Include failures, not just successful trajectories. A reported quality advantage should survive cold-cache evaluation and a stored-value-only inference path before supporting the deployment claim.

### 11.4 Compaction and replay correctness tests

On a small deterministic setup, compare full-graph gradients with selective replay across shared parameters, repeated reads, and multiscale transforms. Check chunked versus unchunked reader outputs and gradients. Test empty sets, duplicate records, different permutation orders, dropped scales, stale versions, and failed requests.

For compaction, test full and partial cluster selection, deliberately redundant clusters, independent facts, contradictory members with applicability conditions, rare exceptions, unseen cluster sizes, overlapping-responsibility conservation, and repeated recompression. Compare behavior after deleting a source and its derivatives. Contribution matching must not be accepted as sufficient when final task quality or causal selection semantics change.

## 12. Implementation roadmap and reference configuration

### 12.1 Milestones with decision gates

| Stage | Deliverable | Evidence sought |
|---|---|---|
| A. Small reference | Fixed-shape writer, one-space reader, oracle plans, full autograd, actual payload serialization. | Shape, permutation, causal-visibility, and stored-boundary tests; small reproducible training run. |
| B. Oracle transfer and composition | Counterfactual support/query tasks; write-once/many-uses tests; pooled and attention controls. | New stored memories help a frozen-weight student; individually necessary supports and counterfactual changes affect behavior correctly. |
| C. Scale and learned access | Selective replay, distractors, conditional group utility, and adaptive reads. | Full-graph gradient parity and complete-support recall sufficient to exercise the measured composition. |
| D. Compaction | Temporary-compression training arms and amortized persistent compaction; attention compaction comparator. | Held-out contribution and task behavior preserved on a measured net-bytes/compute curve. |
| E. Deployment frontier | Real disk retrieval, recurrent backbone integration, asynchronous overlap where useful. | Information-matched quality–VRAM–latency comparison under stated host-cache and workload constraints. |

The order expresses experimental dependencies, not a requirement to serialize all engineering. The tiny reference and replay implementation can be developed in parallel; replay correctness does not need to wait for a large learning result. A storage-only cost harness also starts early. Once B establishes a useful raw model, cheap temporary-compaction probes can begin alongside learned access. Persistent multiscale retrieval is an optional branch tested against the single-space system, not bundled into the definition of composition.

Keep the full architecture as the target while ensuring each experiment answers a distinct question. Partial success is actionable: a useful uncompressed memory learner, an effective compactor, or an efficient recurrent integration need not wait for every other component to work. Quantitative milestones guide effort allocation, not universal scientific vetoes.

### 12.2 Proposed starting configuration

The early compatibility experiments used a sub-billion-parameter student with eight canonical writer slots, eight returned reader slots, and flat payload widths `[256, 512, 1024, 2048]`. That interface remains valid for historical checkpoints but is not the Phase 2 target. Phase 2 follows `positional-memory-v0.7.md`: distill an eight-position structured interface and then expand both slot counts to 32 while retaining per-position channel widths `[32,64,128,256]`, yielding stored widths `[1024,2048,4096,8192]`. The reader and compactor use repeated shared input-position-to-target-position MLP operators. Do not instantiate a dense `Linear(32*decoder_width, payload_width)` codec; at decoder width 1,024 it would add roughly 503 million codec parameters and contradict the small-resident-model objective.

For multiscale experiments, test four spaces with the illustrative payload/count schedule in Section 10 against single-space readers matched for total bytes and output slots. Preserve all eligibility and provenance rules. Prefer factorized local MLPs before adding deeper per-pair networks.

Use a recurrent implementation as a replaceable backbone dependency. Validate a small number of loops, then sweep additional loops and early query positions. Use observed resource costs to train invocation; do not set a global read cap to accommodate producer activations. For compaction, sweep two, four, and eight synthetic records per cluster and report actual byte ratios rather than just record-count ratios. Add the compactability-aware arms only after obtaining a raw reference: temporary surrogate replacement first, storage-aligned noise second, and overlapping grouping as a separate factor.

### 12.3 Implementation boundaries

Separate modules for the controller, writer, transforms, retriever, read planner,
pooled reader, replay scheduler, persistent store, and compactor. A training
checkpoint should record the model state, bank journal cursor, pinned index revision,
payload compatibility versions, maintenance-sampler state, optimizer state, random
state, and data/split recipes. Deterministic test fixtures should work without disk
search or asynchronous scheduling.

The first end-to-end implementation should be small enough to compare against a fully differentiable reference. Optimization comes after numerical and causal parity. In particular, custom backward implementations must include query/state gradients through local contributions and gates, not only gradients to stored values.

### 12.4 First decisive experiment

Use a randomized transaction/retry environment. Support A establishes whether state must be restored before retrying; support B establishes a capability-dependent retry rule. Query tasks require the correct joint action. Create matched environments in which restoration is required versus forbidden, and vary the capability condition independently. Generate fresh test environments after freezing all weights. Increase the number of objects and bindings so the result is not restricted to an easy two-bit rule lookup.

Compare no memory with useful extra computation, oracle support text, individually supplied latent records, one-round pooled MLP, multi-round pooled MLP, and multi-round fixed-slot cross-attention. Match sources and training allowance, then separately match read bytes and compute where appropriate. Test no support, A, B, A+B, irrelevant support, and counterfactual replacement of A or B. Include value-channel ablations with routing and metadata held fixed.

Write each support once, save and reload the actual stored payload, and evaluate several query uses. First train through the full graph on small episodes. The same fixtures become replay-parity tests. The initial learning result should not depend on learned search, multiple retrieval geometries, asynchronous scheduling, or persistent compaction.

After that result, run the compactability matrix in Appendix D on the same task family plus deliberately redundant and deliberately independent cluster populations. This moves from "does the reader use the information?" to "can we make its useful behavior cheaper to store?" without changing the task and storage system simultaneously.

## 13. Main risks and unresolved choices

| Risk | Diagnostic | Initial response |
|---|---|---|
| The controller cannot execute retrieved procedures. | Oracle evidence fails despite useful source content. | Improve behavioral training or recurrence; compare controller sizes. |
| Routing never explores useful combinations. | Oracle sets help, learned candidate sets do not. | Add group exploration and known-positive supports. |
| Dense pooling loses bindings or exceptions. | Composition fails as interacting factors grow. | Increase slots/rounds; test typed roles and an attention comparator. |
| One space or output slot dominates. | Scale/slot ablations have no effect. | Inspect utility; simplify or use targeted dropout. |
| Training relies on freshly encoded values. | Stored-only evaluation drops sharply. | Train mixed ages/precisions and control version compatibility. |
| Compaction changes subset-selection semantics. | Full-cluster tests pass, partial reads fail. | Preserve selection metadata or isolate cluster-level routing. |
| Key drift hides valuable old records. | Fresh audits recover useful unreachable entries. | Stable addresses, audited regeneration, versioned indices. |
| Retrieval/replay cost overwhelms savings. | Frontier loses under cold-cache or full-cost accounting. | Optimize buffers and compaction; retain simpler viable variants. |

Open research choices include whether reader rounds should share weights, whether spaces should share local MLP parameters, how much query-dependent gating is needed, how to select replay entries without long-tail starvation, and whether compact codes should preserve the original reader interface or use a specialized conditional decoder.

The architecture provides no guarantee of lossless finite-width storage or universal replacement of dense model capacity. Its testable claim is narrower: on a specified distribution, learned external representations and composition may offer a better resource allocation. The strongest result would show new-experience transfer, held-out composition, successful compaction, and a measured deployment advantage in the same system.

## 14. Relationship to prior work

RETRO integrates large-scale retrieval into language modeling, and Memorizing Transformers retrieve stored internal representations [^1][^2]. They motivate external access but do not by themselves demonstrate the proposed experience-writer training and later cluster-contribution distillation.

Memory³ is a particularly close precedent for explicit memory outside model weights and a smaller resident language model [^9]. This project should distinguish itself through the combination of fixed-shape writes, dense read-time composition, selective producer replay, and amortized behavior-preserving compaction, rather than claim novelty for external latent memory alone.

Deep Sets and Perceiver IO provide alternative architectural foundations for fixed-size processing of large sets [^4][^6]. AutoCompressors and ICAE motivate soft-token communication [^7][^8]. Gradient caching and activation checkpointing motivate the replay implementation [^10][^11]. Recurrent-depth language models supply a backbone precedent [^12]. Memory layers and Engram are relevant capacity and systems comparators [^13][^14].

Three additional 2026 papers sharpen the positioning. **LatentMem** stores raw experience trajectories and uses a learned agent-conditioned composer to generate fixed-length latent memory; the ordinary read path therefore differs from this proposal's prewritten stored payloads [^18]. **ElasticMem** uses an offline latent bank, reasoner-state retrieval, learned latent budgets, and soft-token injection; its bank is constructed with a frozen encoder and stays fixed during memory-use training [^19]. **FocusMem** separates content formation, decision-conditioned readout, and trust while keeping the GUI policy frozen [^20]. Its original-keys/shuffled-values diagnostic is also relevant to channel attribution. These are overlapping components, not demonstrated equivalents of the proposed persistent writer–composition–compaction system.

The narrower contribution to investigate is **transfer-trained stored payloads; controlled comparison of recurrent set-composition mechanisms; and amortized preservation of conditional cluster contributions**. Selective replay makes this trainable at larger scales, while optional local fields and compactability-aware training shape the representation for consolidation. None of these phrases alone is an established novelty claim.

Noise regularization, rate–distortion training, local partition-of-unity approximation, and feature-matching synthetic datasets provide precedents for individual compactability interventions [^21][^22][^23][^24]. Applying them to this memory system is a proposed experiment, not a transfer of their published guarantees.

These connections support feasibility of components, not the success or novelty of their proposed combination. The bibliography is a verified starting set of primary sources, not an exhaustive novelty review. No benchmark result for the proposed architecture is asserted in this document.

## Appendix A. Selective replay update

The following pseudocode specifies the canonical-value boundary and a small fully checkable reference execution. Sources and historical producer-side read inputs are fixed data for this update. Exact replay is conditional on those inputs and the frozen discrete read plan. Pinning parameter versions prevents updates; it does not disable gradient accumulation. Operational error handling, distributed reduction, and checkpoint placement are implementation responsibilities.

```text
snapshot = pin_parameter_and_index_versions()
zero_parameter_gradients()
leaf_registry = {}
read_plan_registry = {}

on_consumer_read(read_id, query, causal_state):
    # The same logical read_id is reused during consumer recomputation.
    plan = get_or_create_frozen_plan(
        read_plan_registry, read_id, query, causal_state, snapshot
    )
    values = []
    for record in plan.records:
        if frozen_fresh_selection(record, snapshot):
            key = get_or_create_replay_identity(record, snapshot)
            if key not in leaf_registry:
                with no_grad(), recorded_producer_state(key):
                    value = writer(record.source, record.read_inputs)
                leaf_registry[key] = make_gradient_leaf(value)
            values.append(leaf_registry[key])
        else:
            values.append(load_stored_value(record).detach())
    return reader(query, plan, values)

loss = run_consumer_episode(on_consumer_read)
backward_with_consumer_checkpointing(loss)

for key, leaf in leaf_registry.items():
    cotangent = accumulated_gradient(leaf)
    with restore_exact_producer_state(key):
        live_value = writer(key.source, key.recorded_read_inputs)
    backward_vector_jacobian_product(live_value, cotangent)
    release_producer_activations()

# Shared writer/decoder parameters already contain both contributions.
# No optimizer step occurs between value generation and these replays.
optimizer_step_once()
```

The registry expresses logical reuse, not a requirement to retain every leaf tensor and cotangent on the GPU. At scale, stage values and accumulated cotangents through host or disk storage, or regenerate them while replaying consumer segments; materialize only the currently needed tensors. These schedules must preserve the reference execution and account for their transfer cost. Recomputed consumer segments reuse recorded selection masks, request timing, and read plans rather than searching again.

If keys or space-specific payloads are included in the replay boundary, register their cotangents as well. If reads of the same source use different stochastic producer states, they receive different replay identities. A deployment-precision surrogate must be present consistently in both generation and replay.

## Appendix B. Pooled reader and compaction targets

```text
R = initialize_slots(query)
for t in range(reader_rounds):
    aggregates = zeros_for_all_spaces_and_slots()
    masses = zeros_for_all_spaces()
    for space in spaces:
        for chunk in payload_chunks(read_plan[space]):
            local = conditional_mlp(chunk, query, R, slot_ids, t)
            weights = relevance_gate(chunk, query, R, t)
            aggregates[space] += sum_records(weights * local)
            masses[space] += sum_records(weights)
    # Barrier: every chunk in this round has contributed.
    R = residual_update(R, aggregates, masses, query, t)
return output_projection(normalize(R))
```

For compaction, record each original cluster's unnormalized contribution, normalization mass, current shared state, query, and selection descriptor at every round. Train the amortized compact representation against those targets and the downstream decoder. During mixed compact/uncompacted execution, merge numerators and masses before normalization. The compactor must not normalize each cluster independently and then average cluster outputs as though that were equivalent to the original reader.

## Appendix C. Review disposition

The review's mathematical and architectural corrections are largely valid. The following dispositions preserve their diagnostic value without turning every concern into an additional prerequisite.

| Review point | Validation and change |
|---|---|
| Composition does not establish compactability. | Accepted. The ReLU breakpoint example is correct for the stated restricted representation. Section 8.1 states a distributional hypothesis; Sections 8.7–8.11 add interventions that can actively improve the representation rather than merely wait to discover compressibility. |
| Attention also has additive conditional statistics. | Accepted. Section 4.5 now compacts its numerator and denominator and compares both readers after compaction. The MLP argument rests on conditional vector transformations and measured resource cost. |
| Write and read bottlenecks must be separated. | Accepted. Write-once/many-uses evaluation and separate capacity sweeps are added. A narrow useful memory is not a failure, but claims must match its tested scope. |
| Group utility must be operational. | Accepted. Conditional marginal and group-level training are part of learned access; complete-support recall is a primary metric. All useful groups need not be exhaustively enumerated outside small fixtures. |
| Composition should precede learned routing and multiscale retrieval. | Accepted. The core experiment uses oracle plans and one space. Multiscale processing remains an optional architectural branch. |
| Build a small full-graph implementation before depending on replay. | Accepted as a correctness reference, not a reason to abandon replay or cap read count. Full graph and replay can be implemented in parallel against the same fixtures. |
| Defer compaction until the full early program is complete. | Narrowed. Persistent compaction is deferred, but inexpensive temporary-compression probes begin after raw composition works. Shaping compactability is an experiment in its own right. |
| Use counterfactual support and value-payload interventions. | Accepted. Randomized environments and fixed-plan value ablations strengthen causal attribution. Legitimate key-carried knowledge is measured, not automatically classified as a shortcut. |
| Stored-only evaluation from the outset. | Accepted. Serialization, precision, and cache boundary are explicit invariants of the first successful experiment. |
| Fourfold compression with 90% retained benefit. | Retained as an illustrative target, not a universal go/no-go rule. Report curves, uncertainty, net bytes, and tails; avoid ratios with unresolved denominators. |
| Fresh random rules are not a fair standalone substitution test. | Accepted. Larger alternatives get a legitimate route to the same information. Transfer and capacity substitution remain different protocols. |
| Early storage harness and enforced host-cache budget. | Accepted. The 960 KiB conditional page-read calculation is arithmetically correct. Run a small independent harness; it is not a reason to build full asynchronous serving before learning results. |
| Structural authorization and deletion rules for compact codes. | Accepted. Homogeneous domains and derivative invalidation are schema-level requirements; multi-tenant hardening is not a prerequisite for a single-domain learning prototype. |
| Add LatentMem, ElasticMem, and FocusMem. | Verified against primary-source descriptions. Added with explicit distinctions; no reproduction of their headline performance claims and no exhaustive novelty claim. |
| Compaction is necessarily the largest uncertainty. | Not adopted as an objective ranking. Its difficulty depends on workload redundancy; small-controller execution and transfer may be equally important. The experiment order localizes both. |

The review does not establish that the architecture will fail or that a resource advantage will occur. Its most useful effect is to improve identification of mechanisms while preserving the research direction.

## Appendix D. Minimal compactability experiment

This is an in-memory training/evaluation experiment, not a storage-service project. Its objective is to distinguish naturally compactable representations from representations made more compactable by training.

### D.1 Arms and controls

Use the same source data, fixed soft-token interface, single retrieval space, reader rounds, and held-out queries across arms.

| Arm | Intervention | Question |
|---|---|---|
| Raw reference | No compactability intervention. | What useful composition is available before compression? |
| Mean-plus-mass / merge consistency | Replace small groups with weighted means, with and without the local consistency loss. | How much compaction can the cheapest explicit bias support? |
| Post-hoc surrogate | Freeze the raw model; fit a small amortized compactor. | How compactable is the existing function? |
| Temporary compression | Train through a small surrogate on a fraction of reads, keeping raw task training. | Does direct exposure improve quality at the same compressed byte budget? |
| Storage perturbation | Add storage-aligned noise or quantization simulation. | Does precision robustness help byte reduction independently of grouping? |
| Regrouping/overlap | Train the compactor on alternative local clusterings of the same records. | Does multi-group participation improve held-out compaction without inference replication? |
| Persistent fields, optional | Two nonnegative responsibility shares per record, normalized to unit total. | Does extra local modeling justify its storage and read amplification? |
| Mergeable-feature comparator | Pool query-independent sufficient features, then apply query-conditioned nonlinear updates. | How much useful behavior can a structurally compactable reader preserve? |

Do not initially train every combination. Establish the post-hoc and temporary-compression arms, then add one factor at a time. Repeat the strongest small comparison with fixed-slot attention and its conditional numerator/mass targets.

### D.2 Reference execution

```text
# One useful uncompacted checkpoint already exists.
reference = frozen_copy(useful_raw_checkpoint)

for support_episode, query_batch in training_episodes:
    payloads = write_sources_once(support_episode)
    plan = fixed_oracle_plan(query_batch, payloads)
    raw_loss = task_loss(run_raw_reader(payloads, plan), query_batch.targets)
    extra_loss = 0

    if sample_temporary_compaction():
        groups = make_local_groups(payloads, plan, recorded_grouping_seed)
        # The encoder cannot see query answers or teacher response vectors.
        compact_codes = compact_encoder(groups)
        compact_codes = chosen_storage_surrogate(compact_codes)
        compact_run = run_compact_reader(compact_codes, plan)
        teacher_run = run_reference_with_legal_support(reference, support_episode, plan)
        extra_loss = compact_task_loss(compact_run, query_batch.targets)
        extra_loss += scaled_behavior_distillation(teacher_run, compact_run)
        # State alignment is explicit: teacher contributions are evaluated
        # at corresponding probe states; compact rollout loss uses its own states.
        extra_loss += scaled_field_contribution_loss(groups, compact_codes, probe_states)

    backward(raw_loss + extra_loss)
    # At scale replace retained producer graphs with parity-tested replay.
    optimizer_step_after_all_shared_parameter_gradients()
```

The pseudocode expresses objective boundaries, not a library implementation. A frozen reference's payloads are generated in its own compatible representation; they are not silently interpreted by an unrelated current reader. If the reference and live writer are updated jointly, pin or explicitly bridge versions. Record all stochastic choices needed for replay.

For a pure compactor diagnostic, first freeze the writer, reader, and decoder. For compactability-aware training, unfreeze selected components while retaining the uncompacted objective. Evaluate a held-out serialized store produced by the resulting writer/compactor with all weights frozen. The compact path must not read the originals in order to reconstruct a supposedly stored code.

### D.3 Readout of the experiment

Evaluate redundant paraphrases and repeated procedures; independently useful facts; compatible facts that must be bound together; incompatible rules with explicit scope; and rare exceptions. Write supports once and ask several future questions, then test unseen groups and new combinations of groups.

Report raw quality, compact quality, retained memory benefit, payload and total-store bytes, read compute, complete-support recall where applicable, and exception success. Compare the compactability-aware raw path against the original raw baseline. Repeatedly returning the same easier answer or reducing sensitivity to counterfactual evidence is not an acceptable compaction gain.

The preferred first outcome is modest and concrete: temporary compression improves held-out quality at a fixed compact byte budget while retaining the original model's useful dependence on stored evidence. Persistent overlap, deeper hierarchy, and asynchronous disk execution can then be evaluated with a known working learning mechanism.

## References

Original primary-source bibliography retained from the first specification; review-specific sources and compactability precedents were checked on 19 September 2026. Publication years identify the cited work, not a claim that it is the latest work in its area. References [18]–[20] are research reports whose architectural descriptions are used here; their performance claims are not adopted as results for this project.

[^1]: Borgeaud, S., et al. (2022). [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426). ICML. arXiv:2112.04426.

[^2]: Wu, Y., et al. (2022). [Memorizing Transformers](https://arxiv.org/abs/2203.08913). ICLR. arXiv:2203.08913.

[^3]: Guu, K., et al. (2020). [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909). ICML. arXiv:2002.08909.

[^4]: Zaheer, M., et al. (2017). [Deep Sets](https://arxiv.org/abs/1703.06114). NeurIPS. arXiv:1703.06114.

[^5]: Wagstaff, E., et al. (2021). [Universal Approximation of Functions on Sets](https://arxiv.org/abs/2107.01959). arXiv:2107.01959.

[^6]: Jaegle, A., et al. (2021; ICLR 2022). [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795). arXiv:2107.14795.

[^7]: Chevalier, A., et al. (2023). [Adapting Language Models to Compress Contexts](https://arxiv.org/abs/2305.14788). EMNLP. arXiv:2305.14788.

[^8]: Ge, T., et al. (2023; ICLR 2024). [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945). arXiv:2307.06945.

[^9]: Yang, H., et al. (2024). [Memory³: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178). Journal of Machine Learning, 3, 300–346. arXiv:2407.01178.

[^10]: Gao, L., Zhang, Y., Han, J., and Callan, J. (2021). [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://aclanthology.org/2021.repl4nlp-1.31/). RepL4NLP, 316–321.

[^11]: Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016). [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). arXiv:1604.06174.

[^12]: Geiping, J., et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171). arXiv:2502.05171.

[^13]: Berges, V.-P., et al. (2024). [Memory Layers at Scale](https://arxiv.org/abs/2412.09764). arXiv:2412.09764.

[^14]: Cheng, X., et al. (2026). [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372). arXiv:2601.07372.

[^15]: Subramanya, S. J., et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/). NeurIPS.

[^16]: Penedo, G., et al. (2024). [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557). arXiv:2406.17557.


[^17]: Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS. arXiv:1706.03762.

[^18]: Fu, M., et al. (2026). [LatentMem: Customizing Latent Memory for Multi-Agent Systems](https://arxiv.org/abs/2602.03036). arXiv:2602.03036. The reviewed v1 methodology and the current abstract agree on the raw-trajectory bank and conditional composer distinction.

[^19]: Feng, T., et al. (2026). [ElasticMem: Latent Memory as a Learnable Resource for LLM Agents](https://arxiv.org/abs/2605.30690). arXiv:2605.30690, v1. See Section 3.2 and the training description for the frozen bank.

[^20]: Zhang, Z., et al. (2026). [FocusMem: Factorizing Content, Readout, and Trust in Latent GUI Memory](https://arxiv.org/abs/2608.04530). arXiv:2608.04530, v1. See method and evidence-dependence diagnostics.

[^21]: Bishop, C. M. (1995). [Training with Noise is Equivalent to Tikhonov Regularization](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/). Neural Computation, 7(1), 108–116. The regularization interpretation has small-noise and loss-function assumptions; it is not a compactability theorem.

[^22]: Trask, N., Henriksen, A., Martinez, C., and Cyr, E. (2022). [Hierarchical partition of unity networks: fast multilevel training](https://proceedings.mlr.press/v190/trask22a.html). PMLR 190, 271–286. A precedent for local approximation and hierarchical decomposition, not for the proposed memory representation.

[^23]: Ballé, J., Laparra, V., and Simoncelli, E. P. (2017). [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704). ICLR. See also the [authors' description of the quantization surrogate](https://www.cns.nyu.edu/~lcv/iclr2017/).

[^24]: Zhao, B., and Bilen, H. (2021 preprint; CVPR 2023). [Dataset Condensation with Distribution Matching](https://arxiv.org/abs/2110.04181). A precedent for synthetic-set feature-distribution matching, not for preserving this reader's full conditional behavior.


## SCHNITZELJAGD 0.3: implemented real-student curriculum

The project is now **Stigmergic Compactable Holographic Neural Indexed Trajectory Zettelkasten with Evolving Latents, Jointly Adapted by Gated Decoders (SCHNITZELJAGD)**.
The full research hypotheses above remain intact. The current runnable operational
plan is in [training.md](training.md), with researched source choices in
[datasets.md](datasets.md) and Spark runtime setup in [spark.md](spark.md).

The default real student remains LiquidAI/LFM2.5-230M. A documented NVIDIA 25.11
ARM64 PyTorch image replaces the previously untested latest-tag default while
preserving the vendor runtime. The pretrained one-pass path is the starting control;
recurrence, multiscale geometry and learned routing remain independent variants.

Teacher data now have two explicit executable protocols: earlier-prefix compression
and prior different-instance experience. Neither lets a prior write see a later
target. Preparation uses actual tokenizer budgets, complete targets, group-held-out
splits, source revisions/provenance and matched evidence for the text comparison.
The causal recipe uses the real student on controlled supports before fresh-world
stored-only evaluation with frozen weights.

The implemented curriculum stages text bootstrap, frozen-backbone latent warmup and
low-rate joint adaptation. A same-example oracle-text anchor is optional and logged
separately from the memory objective. This tests the coordination issue observed in
the CPU study without claiming all students require the same curriculum. Existing
compaction, replay and set-reader mechanisms remain available. A completed script run
is not automatically a successful transfer/composition/substitution result.
