# The knowledge-base stack: superposed spaces and MLP-matrix operators

**Status: design, 28 September 2026 (owner direction).** The writer (restart
plan B2/B3), the MLP-matrix operator and the K1 training script exist; section 9
tracks what is built and running. This document replaces restart plan B7 ("combiners") and the
restart plan's reading of "spaces" as BGKit compression ratios.

## 1. Goal

The knowledge base (KB) is meant to hold knowledge that would otherwise live in
the parameters of a much larger language model, so that the KB plus a small
decoder can replace that model in a modular way. It trades disk storage and some
latency for resident memory. The items are parameters (trained, or extracted
from text) that act **only in context**: a read places a short latent span in
the decoder's context, which the decoder reads like any other input. They are not
expert outputs mixed into the hidden states (owner, 28 September).

**Extreme sparsity, breadth over intensity.** Reads are frequent and as cheap as
possible: each read touches a handful of items out of a very large KB and costs a
short span. Capability is meant to come from the breadth of the KB, not from
how much is read per sample; the experiment tests how far that goes (quality
against KB size at a fixed per-sample read budget). Retrieval tasks are the
starting point for training this kind of continual-learning system, not its
purpose.

For that, stored items must not be isolated copies of source documents.
Retrieval-augmented QA works fine with localized records (retrieve the right
passage, read it), and our training tasks started there, which biases intuition
towards "find the gold record". A parameter store needs the opposite: every
item carries parts of many sources, every source is spread over many items, and
the few items one read touches each carry many sources. **Superposition is an
explicit goal**, and objectives are chosen to require it.

Earlier SDKB runs showed how hard it is to get an end-to-end lift from learned
KB representations at all (R5d5 retrieved the gold record 89-99% of the time,
yet its stored values carried 2-5% of the text arm's gain). The BGKit decoder is
therefore used as a scaffold: its compressed spans are a representation the
decoder already reads well. The stack below starts from that localized
representation and diffuses it into superposed spaces.

### 1.1 Benchmark and teachers (owner, 28 September)

The measure of success is the small decoder plus the KB against larger models,
on held-out text and tasks from domains whose corpus is in the KB, with the
decoder alone, the same retrieval as plain text, and a shuffled KB as controls.

| Model | Params (total / active) | Tokenizer | Role |
|---|---|---|---|
| LFM2.5-350M (our decoder) | 0.35B | ours | |
| LFM2.5-1.2B | 1.2B dense | identical | first distillation teacher (cheap) |
| LFM2-8B-A1B | 8.3B / about 1B | identical except 2 special tokens | distillation teacher |
| LFM2-24B-A2B | 23.8B / about 2B | identical (all 64,400 + 509 added tokens) | main distillation teacher |
| LFM2.5-8B-A1B | 8.5B / about 1B | different (128k vocabulary) | benchmark |
| Ling-3.0-tiny | 7.9B, 128 experts / 8 active | different (157k vocabulary) | benchmark (stronger on code) |

Training signal (owner, 28 September): while basic function and capability are
being built, the stack trains by SFT on existing trajectories and corpora (the B9
task corpora, KB-domain text): one forward and backward pass of our own model per
token, no teacher cost. Teacher distillation comes later, once the system works,
to hammer in capability for real tasks: next-token distributions can be distilled
only from models with our tokenizer, cached offline (top-k log-probabilities plus
the remaining mass per token, `scripts/cache_teacher_distributions.py`, smoke-
tested with LFM2.5-1.2B-Base); the KB stack is then trained so that the decoder
plus the KB matches them. That objective is dense and spread out, with no "gold
record"; the gap between the small and the large model is what the KB has to
supply.

## 2. What drifted, and what is kept

Between 27 and 28 September the restart plan's "spaces" became four BGKit
compression ratios of the same record (s0-s3, all 1024 wide), the codecs were
distilled to reproduce BGKit's encoding at a coarser ratio, and the combiner
merged whole "gold records" of one space. That is a localized design with no
superposition, and per-space retrieval (a record found in one space and not
another) had dropped out. It is superseded here.

Kept:

- **The writer** (restart plan B2/B3): the decoder writes a BGKit-style span
  `<|bg|>(rho) r_1 ... r_k` of variable length for a record. Its training still
  uses BGKit's four length-scaled ratios, which the B1-B3 code and metrics call
  `s0`-`s3`; those are ratio levels of the writer's output, **not KB spaces**.
- The invariants: stored payloads only at inference (1), causal prefixes (2),
  numerator-and-mass aggregation (5), responsibilities summing to one (7).

## 3. Architecture

```text
write:  source ──writer──▶ span x (n reps × 1024)
        span ──forward codec F_s──▶ item in space s (m_s positions × d_s), for s = A..D
        item ──key head──▶ key k_s ; store (item, key, mass, provenance) per space

read:   query ──query head──▶ q_s per space
        space s: retrieve the neighbourhood N_s(q_s) (exact scan first)
                 superposition operator S_s(neighbourhood | target key q_s) ──▶ one item
        recombiner R(items of all spaces) ──▶ span (decoder input space) ──▶ decoder reads

rewrite: S_s(neighbourhood | target keys) ──▶ items in space s, written back (recursively)
```

- **Spaces** differ in granularity and width. An item's position count scales
  linearly with the source span, m_s = ceil(r_s · n), and all spaces together are
  about the size of the input (sum of r_s · d_s = 1024 per input rep): the stack
  as a whole does not compress, but each space alone is a bottleneck, so no
  single space can carry a record and the recombiner must combine spaces. Every
  space holds the same amount of information per input rep (owner, 28
  September): width grows exactly as positions shrink, so a coarse item carries
  as much as the fine positions it stands for.

  | Space | r_s (positions per input rep) | width d_s | r_s · d_s |
  |---|---|---|---|
  | A (fine) | 1 | 256 | 256 |
  | B | 1/2 | 512 | 256 |
  | C | 1/4 | 1024 | 256 |
  | D (coarse) | 1/8 | 2048 | 256 |
  | total | | | 1024 |

  K1 trains every space to carry content of its own: space dropout per space
  (the fine space dropped most often) and a term reconstructing from one random
  space alone. Reconstruction alone gives coarse spaces no advantage; they pay
  off where one item must stand for a wide input at a low read cost, so the
  training regimes include such inputs (long and multi-record spans in K1/K3,
  sparse read budgets and directly trained items in L1).

  Retrieval neighbourhoods grow with coarseness: fine spaces retrieve few items,
  coarse spaces many.
- **Forward codecs** F_s map the writer's span to each space.
- **Superposition operator, per space.** S_s maps a neighbourhood of items of
  space s to items of space s, conditioned on target keys. Because input and
  output live in the same space (closure), the same operator serves reads
  (combine a retrieved neighbourhood into one item for a query) and rewriting the
  store (replace a neighbourhood by new items, written back). Rewriting is meant
  to increase superposition and the availability of knowledge, not to reduce the
  amount of representation: *compaction* (fewer outputs than inputs) is only its
  special case. Applied recursively, it spreads each source over more items and
  lets each item carry more sources. (Earlier drafts called it the compactor.)
- **Recombiner (reverse codec)** R reads the items of all spaces and produces a
  span in the decoder's input space. Spaces may be missing (a space may not
  retrieve anything relevant), so R is trained with spaces dropped.
- **Output positions** are variable. In pre-training the target count is given by
  the target (the original span's n). At inference S_s emits the average
  position count of its inputs, and R emits a span whose length follows from the
  spaces' position counts (n ≈ m_s / r_s).

## 4. The MLP-matrix operator

All three components (codecs, superposition operators, recombiner) use one operator family,
built for dense joint recombination of inputs. Attention retrieves sparsely at
each layer; here every source position contributes to every target position
through its own learned function of both positions.

Sources j = 1..n carry content x_j, a normalized position p_j = (j + 0.5)/n within
their item, a kind (which space or input they come from, each kind with its own
input projection, since widths differ) and the gate of their item. Targets i = 1..m
carry a residual state h_i, a normalized position t_i and an optional condition c
(the target key). One layer:

```text
z_ij = W_src[kind_j] x_j + P φ(p_j) + T LN(h_i) + Q φ(t_i) + D φ(p_j − t_i) + C c
a_i  = Σ_j w_j σ(z_ij) / Σ_j w_j            (numerator and mass)
       w_j = gate of j's item / its length    (each item's total mass is its gate)
h_i ← h_i + O a_i
h_i ← h_i + FFN(LN(h_i))
```

- σ(z_ij) followed by O is a two-layer MLP per (source, target) pair, indexed by
  both positions and conditioned on the target's residual state; because O is
  linear it commutes with the weighted sum, so the per-pair cost is one hidden
  vector (size H), not a full output vector: O(n · m · H) per layer.
- φ are Fourier features. The relative term φ(p_j − t_i) lets a codec align
  source and target positions; a neighbourhood read by S_s has no meaningful
  cross-item order, so its sources carry only their within-item position.
- **Locality kernel** (codecs and recombiner, where positions align): the pair
  weight is w_ij = w_j · exp(−(p_j − t_i)² / 2σ²), σ = β · max(1/m, 1/n_item), β
  learnable per layer and input kind, starting at one position spacing. Without
  it, every target averages all sources equally and one source's signal is
  diluted by their number: the first K1 run (uniform weights) learned a
  position-only average span and ignored the spaces (content 0.01 nats at step
  1000, every space ablation identical). β can grow until the combination is
  flat, so dense superposition stays reachable; training decides how local each
  layer is. S_s over unordered neighbourhoods uses no kernel.
- **Gates only modulate mass.** An item's gate scales its positions'
  contributions to the numerator and the mass; it is not an input feature. A gate
  of 0 removes an item exactly, and scaling all gates together changes only the
  returned total mass. Each item's mass is its gate spread evenly over its
  positions, so a long item (e.g. fine space A) does not outweigh a short one by
  length alone; the per-pair MLPs can still learn to weight contributions. Each
  target's weights over the sources sum to one, and the operator returns the
  total input mass, so compacted items can be weighed against others
  (invariant 5).
- Several layers; each later layer's contributions depend on the target's current
  state h_i, so targets can specialize what they draw from each source.
- Target initial state: an MLP of φ(t_i), the log size ratio and the condition c.
  Output: a linear map of LN(h_i) to the target width, then a fixed-norm rms
  normalization (for the recombiner, BGKit's interface norm into the decoder's
  input space).
- Keys: S_s's condition is the target key. In its first, short pre-training
  phase S_s sees no input keys (it must combine by content); afterwards the input
  items' keys are added as source features.

## 5. Training plan (owner review, 28 September)

Every stage keeps the controls: the same information as text, no memory,
shuffled neighbourhoods or items, and an equal-byte baseline; content is
measured in nats over the shuffled control, not only as captured fractions.

There is one stack (spaces, operators, key and query heads) and one KB per
dataset (also the authorization boundary). KBs differ only in their content and
in the tasks trained over them: the R6 passage corpora (reconstruction, QA,
text continuation) and the task corpora (tool docs, schemas, background, worked
examples; trajectory SFT, later B9 loops). Training mixes tasks over all KBs.

### 5.1 Order (owner review, 28 September; concrete steps)

Data principle: targeted corpora and targeted tasks, supervised by trajectories of
more capable models (the SFT corpora we hold; local larger models such as
LFM2.5-8B-A1B or Ling-3.0-tiny can generate more; paid teacher collection only by
owner decision). A much larger KB run comes only after this works. Code: every
stage is `scripts/train.py <stage>` over the shared package `schnitz.kb`
(decoder, stack, losses, loop, and one producer path `schnitz.kb.producer` for bank
creation's codec step, in-context writes, L1b and B9 replay and L2), so a change
applies to every stage.

1. **Writer, B3** (running): the decoder writes BGKit-style spans.
2. **B4, decoder capabilities** (the last stage in which the decoder trains):
   - *B4a* (built, starts after B3): protocol tokens with LM-head rows,
     ratio-stated prompts, `<|mem|>` delimiters in reads, soft input port for
     questions (ramped to half).
   - *B4b, soft output port:* on tasks whose answer is text (QA, summaries,
     tool results), the decoder answers with a `<|port|>` … `<|/port|>` span
     from its own rep and stop heads (not the memory writer's). Target: the
     frozen S2 encoder's x1 encoding of the answer; losses: KL of a frozen S2
     reader reading the port span against reading the target encoding, a light
     reconstruction NLL, a loose cosine; rollout passes as for the writer.
     Starts once the input port holds, ramps to half of those tasks; evaluated
     over text/soft input × text/soft output.
   - *B4d, the memory protocol itself:* whole memory transcripts (v3) trained on
     their assistant turns, so the decoder learns when to call `memory_search()`
     and `memory_write()` and how to continue after a tool result; the slots are
     filled from the span caches once they exist. From L1 on the decoder is frozen,
     so call behaviour has to be learned here (the B3 decoder never calls on its
     own: 0 of 44 held-out calls).
   - *B4c, in-context writes:* at the write sites of the memory transcripts
     (v3) the model calls `memory_write()` and generates the span itself; the
     span is distilled toward the writer's span of the site's teacher text
     (teacher-fed, then free-running) and checked by a reader reconstructing
     that text.
3. **K1 - Autoencoding through the spaces** (running). Codecs F_s and
   recombiner R: span → spaces → span, read by the frozen decoder; space dropout.
4. **K2 - Keys.** Item-key heads per space (on the items) and query heads (on the
   decoder's middle-layer state at `memory_search()`), trained with the
   retrieval loss (5.2) on the transcripts' search sites against the items of
   each slot's records, with in-batch and KB negatives. It is the L1 stage with
   only the retrieval loss (`--retrieval-only`); no separate key-table
   distillation.
5. **K3 - Superposition operator warm-up** (drop-one: a neighbourhood of items in
   space s with the target removed; S_s produces an item from which R
   reconstructs the target's span).
   - *K3a:* S_s conditioned on the target key only.
   - *K3b:* additionally each neighbour item's key enters at each of its
     positions (continues K3a; the key weights start at zero, so K3b begins as
     K3a).
   - *Depth (owner, 28 September):* the warm-up also runs S_s recursively, at
     least two levels (a neighbourhood of level-1 items, each itself S_s over a
     neighbourhood of records), so the operator works on its own outputs before
     L1 uses it that way.
6. **Bank creation** (offline): the model with a record in context calls
   `memory_write()`; the span, the per-space heads and the key heads give the
   items, one KB per dataset. The same path builds a user's KB from their own
   corpus, so its throughput is a product property (measured). A bank that L1b
   replays is written one span at a time (`l1 build --span-batch-size 1`): the GPU
   free run depends on its batch composition, and batch 1 costs about 3x the
   batched throughput (2.1 vs 5-7 records/s), so bank-scale builds stay batched.
7. **L1 - End to end over the KBs** (decoder frozen, so knowledge has to land in
   the KB). Reads are `memory_search()` calls (vector queries, sparse gates,
   S_s → R → a short span in the tool result); writes are `memory_write()`.
   Tasks: trajectory SFT over the task KBs, QA and reconstruction over the R6
   KBs, continuation of KB-domain text. Two gradient regimes, run as
   alternating phases:
   - *L1a, superposed KB trained in place (owner, 28 September):* the trainable
     parameters sit at the source level (one item per record per space, starting
     from the codecs' output, live items with sparse optimizer state), and reads
     never see them directly: what a read retrieves are items of level L >= 2
     (default 2), each S_s over a neighbourhood of level L-1 items, down to the
     sources. The task loss trains the source items, S_s, keys, query heads and
     R together, so every source is shaped by all the items it feeds.
     Mechanics: level items are computed lazily (only those a read retrieves,
     their level-1 inputs from a cache refreshed on a schedule, gradients through
     the read level and a sampled part of the level below); fixed neighbourhood
     graphs per level from keys, rebuilt periodically; against the identity,
     fewer top-level items than sources (the storage budget) and drop-one at
     level 1; each source's contributions normalized to its mass (invariants 5
     and 7), recorded as rewrite shares; an item's time is the latest of its
     sources' (invariant 2); mixing only within one KB (invariant 6). Control:
     depth 0 (the items themselves) at the same storage and read budget.
   - *L1b, through the sources:* for the items a read retrieves, their write is
     recomputed from the stored source with gradients (selective producer
     replay, the serialized forward exactly: invariant 3), so the task loss
     trains the writer's span heads, codecs, keys, S_s and R end to end.
     Training-only; inference still reads stored payloads (invariant 1).
8. **L2 - Producers reproduce the L1a items** (two-step): the trained top-level
   items are the targets; several S_s layers learn to map the original KB (the
   codecs' items of the sources) to them, with the writer and codecs trainable
   too (the rewrite-then-recover objective with L1a's items as targets). This
   S_s stack is also how a new corpus becomes a superposed KB without its own
   L1 run.
9. **B9 - Learning by experience** (restart plan B9): the model works on a task
   over its KB for several rounds; at the end of each round it writes what it
   learned with `memory_write()` (single pass, its attempt in context), the
   items enter the dataset's KB, and later rounds and later tasks read them. It
   trains on these trajectories that read and write: SFT toward trajectories of
   more capable models and verifiable outcomes, the gradient reaching earlier
   rounds' writes (truncated over 2-3 rounds, the L1b regime), gold records at
   a receding weight. This loop is how a user adapts the model to their use
   case, repeated for several rounds over their KB.
10. **Later:** a much larger KB when the targeted setting works; distillation
    from larger models; joint co-training of the decoder with the stack under
    replay.

### 5.2 Standing requirements

- **Storage budget.** A fixed item budget per space, below one item per source in
  the coarse spaces, so rewriting with sharing is necessary, not optional.
- **Superposition metrics** at every evaluation: sources served per item, items
  per source, the effective number of items carrying mass in a read (from the
  gate masses), and retention of old knowledge after new items are written in.
- **The knowledge must be in the KB (invariant 9).** Edit a fact in the KB and
  the output must follow the KB, not the decoder's weights; removing a domain's
  items must remove the capability; inserting new knowledge must keep old
  knowledge.
- **Context cost of reads.** Reads stay in context (owner, 28 September), so
  their cost is span length times frequency: a 30-rep span every 64 tokens
  lengthens a sequence by about 47%. Read spans are therefore kept short (a few
  reps per read, the recombiner's count a trained choice under a budget), and a
  read's gates are sparse (a handful of items with nonzero mass out of the
  candidate set). The standing breadth experiment measures quality against KB
  size at a fixed per-sample read budget.
- **Gradients into producers (owner: both routes).** Two-step: L1a trains items
  in place with the writer detached, L2 trains the producers to reproduce them.
  Direct: L1b and B9 backpropagate through the retrieved items' sources, with the
  writes recomputed by selective producer replay (invariant 3: every value, key,
  gate and shared-parameter path, the serialized forward, RNG and autocast).
- **Store contracts.** Per-space variable-width items with masses, keys and
  rewrite lineage extend the mutable-bank (v0.8) and scale-out (v0.9) contracts;
  exact-scan index first, ANN measured separately (invariant 8).
- **Writes during trajectories** (B9) are `memory_write` calls: supervised at
  episode or round ends first, learned write sites later.

- **Read count (owner).** R's output length: during pretraining (K1, K3) the
  target's own count; at read time the gate-mass-weighted mean of the retrieved
  items' lengths in decoder reps (an item of m positions in space s stands for
  m / r_s reps), capped by the per-read budget (`schnitz.kb.stack.read_count`).
- **Retrieval auxiliary loss.** Per space, over a read's candidates:
  −log Σ_{positive} softmax(score / τ), positives being the items of the slot's
  records (after rewriting, their descendants weighted by responsibility share),
  plus the same loss's recall@k in the logs. It starts routing (K2) and is
  annealed in L1 as the task loss takes over (`schnitz.kb.losses`).
- **Rewarding spread-out use.** Breadth needs many items to be useful, not a few
  popular ones: a balance loss n · Σ_j f_j · p_j over the items a batch touches
  (f_j an exponential moving average of item j's share of read mass, p_j its
  mean gate in the batch), which penalizes routing mass onto already heavily
  used items; the share of the KB read at least once per evaluation window, and
  dead items, are logged. Per read, gates stay sparse (a handful of items);
  spreading happens across reads. Sources spread over items through the budget
  and the rewriting passes.

## 6. Relation to other plan stages

- B2/B3 (writer) feed K1; B4 and the soft I/O port train the decoder before L1.
- B5 (keys, query diversity) becomes K2 and L1's routing, per space; the query
  diversity measures (several queries per site, coverage discount, repulsion,
  exploration) carry over unchanged.
- B6 (reads) is L1's read path. Recurrence (reads between looped core passes) is
  dropped (owner, 28 September): it costs 37-75% more training FLOPs per token plus
  its own conversion phase, and buys quality per parameter rather than per
  training FLOP; between-token reads with a mid-layer query already overlap
  retrieval with computation, and multi-hop happens across read sites.
- B9 (recursive improvement) stores its trajectories through the writer and this
  stack; one persistent KB per dataset. Rewriting only mixes items within one KB
  (one authorization domain), and learned selection is never used as
  authorization (invariant 6).

## 7. Costs

Per layer and item the operator costs O(n · m · H): with H = 256, a 60-rep span
reconstructed from about 110 space positions is about 1.7M hidden activations
per layer. Widths, neighbourhood sizes (about 16-64 items) and position counts
are the levers; the dense per-pair form is kept on purpose.

## 8. Open questions

- Space count, widths and position ratios (the table above is a starting point).
- Neighbourhood sizes per space, and M/N in rewrite-then-recover.
- Whether R also receives the query (question-conditioned recombination).
- How rewriting levels are scheduled once the store is large.
- Top-k size for cached teacher distributions, and the distillation corpus.

## 9. Implementation work packages (28 September)

Built in parallel while B3 and K1 train, so each stage is ready when its
inputs are. Training data is regenerated where the format changes (owner:
"not afraid to regenerate training data").

- **WP1 - Span protocol, B4 and the soft I/O port** (`scripts/train_bgkit_b4.py`).
  Reserved tokens become `<|bg|>`, `<|rep|>`, `<|/bg|>`, `<|mem|>`, `<|/mem|>`,
  `<|port|>`, `<|/port|>` with LM-head rows (restart plan 3.2); the two-way stop
  head is replaced by the `<|rep|>`/`<|/bg|>` rows; a ratio head at `<|bg|>`.
  Training mixes BGKit general compression at all ratios, interleaved
  text-span-text sequences, memory tool-call transcripts (WP3) with latent
  results, and the soft input then output port, with the replay KL.
- **WP2 - Knowledge-base store** (`src/schnitz/kb_store.py`). Per KB (one per
  dataset) and space: variable-length items of the space's width, keys, masses,
  provenance, versions and rewrite lineage; append, supersede and rewrite
  operations; exact top-k scan per space over memory-mapped keys; live-item mode
  with per-item optimizer state for sparse in-place updates (L1).
- **WP3 - Memory-protocol data** (`scripts/prepare_memory_transcripts.py`). Every
  episode of the R6 corpora and the task corpora rewritten as an LFM2 chat
  transcript with argument-free `memory_search()` calls (target record ids per
  call), tool results as latent slots filled at training time, and
  `memory_write` calls; checks that results only contain records that exist
  before the call and that nothing from the answer or later turns precedes it.
- **WP4 - Evaluation**. Superposition metrics (sources per item, items per
  source, effective items per read, retention), KB-dependence tests (edited,
  removed and inserted knowledge) and content-over-shuffled helpers are built in
  `src/schnitz/kb_eval.py` (pure, tested); the benchmark runner for
  LFM2.5-8B-A1B, Ling-3.0-tiny and LFM2-24B-A2B on the task corpora with their
  verifiers is `scripts/benchmark_models.py` (smoke-tested with LFM2.5-350M and
  1.2B-Instruct; the large-model runs are pending). Harness version 2 (28
  September): per-task context budgets that cut no validation episode, per-turn
  APIGen-MT scoring, Reasoning Gym normalizations, SynLogic with its verifiers,
  sandboxed code execution.
- **WP5 - Stack training after K1**: K2 key and query heads, K3a warm-up of S_s,
  the L1 read path (retrieve per space, gates from similarity, S_s, R, `<|mem|>`
  span). Built (28 September): the read path `src/schnitz/kb/read.py` and the stage
  `scripts/train.py l1 build|train` (`src/schnitz/kb/stages/l1.py`), on the shared
  `schnitz.kb.stack`/`losses`/`bank` modules.
  - *Build* (offline bank creation, invariant 1): the records the transcripts'
    slots name (plus `--distractors` per KB), their writer spans through
    `schnitz.kb.bank` (per-KB span caches shared with `train.py bank`; B3 free run
    at level s0, or `--span-source teacher`), the stack's codecs (K1 `stack.pt`, or
    random init with span statistics from the records), keys from the initial
    item-key heads; one `kb_store` KB per dataset, item time = `created_at`.
  - *Read*: query state = the frozen decoder's state after `--query-layer` (8 of
    16) layers at the token holding the call's closing parenthesis; query heads per
    space; exact top-k over the live keys of the episode's own KB only (candidates
    A/B/C/D 8/16/32/64), time <= the episode's query time; scores from the item-key
    heads applied to the candidates' current values; gates
    `sigmoid(scale (cos - b_s))`; sparse reads (top 2/2/3/4 per space enter, the rest
    gate exactly 0); S_s per space on the query key; R over the space reads with
    their masses; `read_count` capped at 16 reps; span spliced between `<|mem|>`
    and `<|/mem|>`.
  - *Causality and gradients*: the query at call k comes from a pass over the prefix
    up to the call with all earlier reads' spans spliced in (one pass per site,
    truncated at the query layer, recomputed in backward); retrieval (discrete) runs
    outside the recomputed function. Gradients reach earlier reads through later
    queries (tested). The decoder is frozen.
  - *Training (L1a)*: task NLL on assistant tokens plus the retrieval loss (weight
    0.5, `--retrieval-anneal`) and the balance loss (0.01); reader parameters by
    AdamW, item values by one sparse `live_step` per touched (KB, space) with the
    live state resident (`load_live`, CPU by default); stored search keys refreshed
    from the item-key heads every `--rekey-every` steps; each reader checkpoint is
    paired with `checkpoint_live` of every KB, restored on resume (the store's
    restore is tested bit-exact; the trainer resumed cleanly in the GPU smoke, but a
    resumed run was not compared with an uninterrupted one).
    `--retrieval-only` is K2 (only the retrieval loss; spans enter later prefixes
    detached; item values fixed); `--init-reader` starts L1a from a K2 run's key heads
    and gate offsets and re-keys the KBs. The retrieval loss adds in-batch negatives:
    the target items of the batch's other slots in the same KB (never another KB's),
    up to `--inbatch-negatives` (64) per space. A read item's gate is multiplied by its
    stored mass and by an optional per-item weight (`weights=`, B9's gold weight).
  - *L1b* (`--phase l1b`, or alternating with `--phase-schedule a:2000,b:500`): the
    items a read keeps are recomputed from their stored sources (`Producers`): a bank
    item by the writer's free run of its record under the memory prompt, then the
    codecs; a written item by the free run at its write site from the logged prefix
    and reads (`WriteLog`), then the codecs. Span and item enter at their serialized
    precision (bf16, straight-through), in the producer's batch composition (bank
    items one at a time; writes with their writer batch). The gradients of all reads
    of a step accumulate on the recomputed items, then each replay unit is recomputed
    with gradients (per-rep checkpointing) and backpropagated into the writer's rep
    head and ratio code and the codecs before the optimizer step (`--l1b-train`,
    default also keys, S_s, R). Live item values are not updated in L1b; an item L1a
    moved in place is read as its producers' recomputation there (L2 reconciles).
    `--l1b-replay teacher` feeds the stored span instead (one pass; not exact).
  - *Writes* (`--writes`, v3 transcripts): the render keeps each `memory_write()`
    call with an empty `<|bg|><|/bg|>` pair (not targets, empty in every arm); after a
    step's reads the frozen writer generates each site's span in place from the
    episode's causal prefix with its reads spliced in (B4c's length schedule), the
    codecs and item-key heads make the items, and they enter the episode's own KB
    (time = query time, producer `write`, source = the write site; a revisited site is
    superseded). Only later steps read them; an episode never reads its own writes.
  - *Evaluation arms* (invariant 9): no memory (empty `<|mem|><|/mem|>`), the slot
    records' text as the tool result (the information-matched text control, the
    `full` of `kb_eval.nll_summary`), the retrieved read, another episode's reads
    (shuffled), the gold items at gate 1 (no retrieval) and its shuffled control;
    content nats over shuffled, captured fractions, recall@k, effective items per
    read, and the share of reads (and of read mass) on items earlier episodes wrote.
  - *Not in L1 yet*: tasks beyond the transcripts' SFT; the decoder's write path in
    L1b (needs replay of the merged decoder; only the span heads and codecs train).

## 10. Status

| Part | State |
|---|---|
| Writer (B2/B3) | training (restart plan B3) |
| B4 (B4a protocol and input port, B4b soft output port, B4c in-context writes, B4d memory-transcript SFT) | built, not trained (`train.py writer`; `schnitz.span_protocol`, `PortHeads`, `schnitz.memory_transcripts`); starts automatically when B3 ends (`b4.sh`: B4a, B4c and B4d from the start, B4b from step 4000; streams QA 0.25, SFT 0.15, writes 0.15, classical 0.35, bank 0.10); memory slots filled from span caches (`--slot-spans`) once `train.py bank` has written them with the final B3 writer |
| MLP-matrix operator | built (`src/schnitz/mlp_matrix.py`, 9 property tests): locality kernel, per-kind input normalization, optional per-item extra features (zero-initialized; neighbour keys in K3b) |
| K1 codecs and recombiner | training (`train.py k1`, run `kb-k1`, third start 28 Sep with standardized spans: content over shuffled 0.40 nats at step 500, spaces differ under ablation; earlier runs `kb-k1-uniform`, `kb-k1-kernel-unnormed`, `kb-k1-inputnorm` had position-only codes) |
| KB store (WP2) | built (`src/schnitz/kb_store.py`, 29 tests, schema `schnitz.kb/2`): per-dataset KBs, per-space items with keys, masses, provenance, versions and lineage; cursor-pinned commits; exact chunked scan over memory-mapped keys. Rewrites carry per-(output, input) responsibility shares (each input's shares sum to one, output mass = share-weighted input mass, stored exactly; several outputs require explicit shares), and `lineage()`/`source_composition()` resolve share x mass through `kb_eval.source_composition`. Per-commit segment checksums (xxh3-128, else blake2b), hash-chained with the head in the manifest; `verify()`, optional on open. Live mode: per-item Adam state, live keys separate from the immutable stored keys (cursor-pinned and other-process reads never see live state), `pin_live()` snapshots of a live generation by in-memory copy on write, `checkpoint_live`/`restore_live` with bit-identical resume (optionally discarding later commits), export as a frozen KB. Resident live mode (`load_live(device, sync_every)`): values, Adam moments and live keys held as fp32 tensors in RAM or on the GPU, so `live_step` touches only memory; disk is written only by `sync_live()` (every `sync_every` updates, `enable_live`, export, compaction, `unload_live`, clean `close`), which journals the changed items in bounded parts with one commit point, or rewrites and swaps the live files when most items changed; `checkpoint_live` writes straight from memory. A crash reopens at the last sync (`synced_live_updates`); exact resume is `restore_live` of the `checkpoint_live` taken with the model checkpoint. Resident and per-step journaled modes run the same vectorized per-item Adam and are bit-identical on CPU, and both match `torch.optim.Adam`/`AdamW` bitwise. `compact()` writes a KB without superseded rows, their metadata in `history.jsonl`. 200k-item check (4 x 384 positions each): append 12 s, verify 0.3 s, 64-query scan 0.2 s, composition 3 s, checkpoint 4 s, compaction 19 s. Live step at 3000 touched items (28 Sep, machine shared with training runs): journaled 0.2-0.9 s; resident CPU 19-27 ms (1000/5000 items: 10-14/31-40 ms, memory-bandwidth bound); resident CUDA 13 ms median, 19 ms p90 (1000/5000: 4/28 ms; queued without host sync 3/13 ms per step, 2-11 ms host time). Disk-bound, varying with other I/O: sync after one 3000-item step 0.2-0.5 s, after 10 such steps 4-22 s (journal), of the whole state 15-27 s (swap); checkpoint from memory 3-25 s; restore 11-26 s. Resident state 3.9 GB; peak RSS 5.4 GB on CPU, 3.7 GB plus 5.0 GB CUDA. Not yet used by a trainer; live mode is single-process |
| Memory-protocol transcripts (WP3) | version 3 generated (`scripts/prepare_memory_transcripts.py`, 17 tests): 526,709 of 526,802 episodes of 22 corpora in `/archive/corpora/memory-<name>-20260928v3` (v1 and v2 dirs kept). Reads are `memory_search()` without arguments (record ids per call in the slot and in `search_sites` with `step` and `trigger`); writes are `memory_write()` without arguments, rendered with the model's own `<|bg|>`…`<|/bg|>` span in the same assistant turn (placeholder `<|reserved_20|><|reserved_22|>`, open and close in the loss); the content text is kept only as `write_sites[i].teacher_text` for B4 distillation and never rendered (audited). Agent trajectories search mid-episode (protocol at the start; tool docs and action-specific policy sections before the first call; ALFWorld know-how before the first action naming a listed object or place; worked examples before the first use of their most specific command and again before the next; protocol again after a failed action; placement is label-side, accepted by the owner). ScienceWorld episodes without examples get three same-task examples from the KB's held-out pool (3.2 searches per train episode, v1 1.3). Writes (`--writes reusable`): trajectories plus SQL, table answers, tool calls, code up to 900 characters and multi-hop answers (281,875). Audit clean; LFM2.5-350M render check clean on the first 200 transcripts of every split (9,085). Open: 93 reasoning-gym episodes have no records; no trainer reads them yet (B4 write sites, L1) |
| Evaluation harness (WP4) | built (`scripts/benchmark_models.py`, `src/schnitz/kb_eval.py`, verifiers in `src/schnitz/task_verifiers.py`; 57 tests): benchmark of reference models on 9 task corpora with oracle context and closed book; superposition metrics, counterfactual edits, removal and insertion reports. Harness 2 (28 Sep; results of harness 1 are redone on rerun): required records are never cut and per-task context budgets (8k to 80k characters) cover every validation episode (0 truncated; longest full-context prompt 20.6k LFM2.5 / 25.4k Ling tokens, spider_memory; knights 18.1k; all fit the 32k LFM2.5 window with their generation budget, `--measure` reproduces this); APIGen-MT scores every tool-call turn on its gold causal prefix (1,228 calls in 274 validation episodes instead of 19 opening calls; first-call and opening-call rates kept); Reasoning Gym per-family normalizations (`reasoning_gym` is not installed; 521 of 549 validation episodes have a unique stored answer, 28 in 9 families whose scorer accepts any valid solution still undercount; +7 of 400 answers on the 350M run); SynLogic added with the repository's verifiers (local checkout, run sandboxed; 23 families, 3 `math_verify` families reimplemented; 286 validation episodes, `futoshiki` excluded because 12 of 20 stored puzzles have no solution); model code and SynLogic verifiers run in a sandbox (time, CPU, heap and file-size limits, own temporary directory, Internet sockets blocked in Python; KodCode gold pass rate unchanged, 179/200). `kb_eval.nll_summary(..., gain=False)` reproduces the K1 evaluation record exactly (tested) for the K1 stage to report through. Smoke-tested on 350M (5 episodes per task, harness 1; harness 2 on 3 episodes of 5 tasks); reference runs (8B-A1B, Ling-3.0-tiny, 24B-A2B) pending |
| Bank creation (write step) | built (`schnitz.kb.bank`, `train.py bank`): records the transcripts name, the frozen writer's span of each (memory prompt, a ratio level), one resumable span cache per dataset KB; shared by L1's KB build and B4c slot filling. Measured about 2 records/s at level s0 during a shared-GPU smoke; the write-site caches (136,676 records) are scheduled after B3 |
| K2 keys | built as `train.py l1 train --retrieval-only` (item-key and query heads of `schnitz.kb.stack`, retrieval loss of `schnitz.kb.losses`); GPU smoke only, not trained |
| K3a/K3b superposition operator | built, not run (`train.py k3`: drop-one or target-present neighbourhoods from the neighbour table, target key, `--neighbour-keys` for K3b, key-only control); waits for a K1 whose spaces carry content |
| L1 | built, not trained: `train.py l1 build|train` (`src/schnitz/kb/read.py`, `src/schnitz/kb/stages/l1.py`; 31 read-path and trainer tests, 1 store test for batched reads). L1a, L1b (producer replay) alternating by `--phase-schedule`, K2 -> L1a chaining (`--init-reader`), same-KB in-batch negatives, in-context writes from v3 write sites. Live reads gather once per (KB, space) and call on the resident state, onto the reader's device. GPU smoke 28 Sep (bird, alfworld and r6-mixed v3, 80 train transcripts each, 20 KBs, 718 records plus 16 distractors per KB, K1 codecs and B3 writer snapshots of 16:52, batch 4, shared GPU at 0.15): K2 6 steps, then L1 12 steps `a:4,b:2` from K2's heads with writes; 2-4 writes per step, 0.6-4.8 s; after 4 steps training reads start retrieving earlier episodes' writes, and at the step-12 evaluation 78% of validation reads in space C had a written item among the items read (41% of C's read mass; A, B, D 0%), so written items attract routing without carrying content (retrieved 1.7278 vs shuffled 1.7275 nats, noctx 2.02, text 1.58); L1a steps 6-17 s with the live state on the CPU, 7-18 s on CUDA (no gain: the decoder passes dominate; the former 22 s per step of per-item GPU gathers is gone). L1b replay is bit-exact on the GPU at its first step (`l1b_match_exact` 1.0, drift 0, bank and written items) when the bank's spans were written one at a time (`build --span-batch-size 1`) and replayed one at a time; spans written in batches of 16 differ from a batch-1 replay by 0.4-3.5% (padding changes the free run's numerics). Cost per L1b step (free replay): 50-130 s, of which the producers' backward 30-86 s for 26-60 sources (about 1.5 s per source) against 6-17 s for L1a; `--l1b-replay teacher` 14-23 s (backward 4-6 s), 0.56% from the stored items. One L1b step at the former rates moved the recomputed items by 25-45% relative (lr 3e-4 on the codecs), so the producers now have their own AdamW groups (`--l1b-codec-lr` 3e-5, `--l1b-writer-lr` 3e-6) and every L1b step logs the relative change of the recomputed items after the optimizer step (`l1b_change_rel`, `--l1b-change-units`). The replay is the shared `schnitz.kb.producer.Producers` (also L2's and B9's); GPU smoke after the merge (the batch-1 banks of the 28 Sep smoke, K2 heads, batch 4): first L1b step still bit-exact (`l1b_match_exact` 1.0, drift 0, 26 sources); one step moves the items by 2.7% (max 4.0%) at the new rates against 25.3% (max 37%) at the old; the second step 2.5% (max 16%), its match to the stored items 2.8% after one update. Bank write throughput (96 bird records, level s0, shared GPU at 0.12): 2.1 records/s at span batch 1, 5.2-6.7 at 16; batch-16 spans differ from batch-1 spans by 1.1% on average, up to 29%, so L1b needs `--span-batch-size 1` banks (documented in `l1 build --help`; the default stays batched) |
| L2 | built, not trained: `train.py l2 export|train` (`src/schnitz/kb/stages/l2.py`, producer path `src/schnitz/kb/producer.py`, the replay L1b and B9 use: one `Producers` class, one `WriteLog`, one `Writer`; 5 CPU tests). Targets: the live KBs of an L1 run at its reader checkpoint's live tag, restored in a scratch copy and exported frozen (`export_live`; the L1 run is not touched), or given exports. Producer per record: the writer's span of the source under the memory prompt at the banks' level, teacher-fed with the bank's cached span (`--feed teacher`), free-running without gradients then one gradient pass (`self`; was `free` before the producer merge) or the free run replayed with gradients through every step (`free`, L1b's replay), rounded to the span cache's bf16 like L1b's replay, then the codecs; rewrite outputs through their lineage (S_s over the produced inputs at gate share x mass, conditioned on the item's key; one level, since an export keeps only metadata of superseded rows). Losses: per space 1 - cosine and MSE over the target's mean square, key cosine (L1 item-key heads on the produced values vs the live key), KL of the frozen decoder reading R(produced) vs R(live items) on the source's reconstruction (R from the L1 reader, frozen). Trained: the writer's rep head and ratio code (`--train writer`), optionally codecs and S_s. Eval: held-out records and a training sample (reproduction error per space, key cosine, NLL of R(produced) / R(live) / R(bank item) / no memory), and `--eval-episodes N` re-runs L1's arms with a KB written from the producers. GPU smoke (28 spider records built by `l1 build` from 40 v3 transcripts with the K1 codecs, B3 writer): after 8 L1a steps at item lr 1e-2 the live items had barely moved (bank vs live MSE 0.001, the teacher-fed reproduction floor), so L2 had nothing to learn; after 16 L1a steps at item lr 0.1 (bank vs live relative MSE 0.16 in A, 0.21-0.23 in D), 60 L2 steps (batch 4, lr 1e-4) lowered the functional KL to the live items from 0.018 to 0.006 on the training sample and from 0.036 to 0.023 on 7 held-out records, key cosine in D 0.90 to 0.93, the value MSE only slightly (A 0.157 to 0.150); L1's arms with produced items: retrieved 0.827 vs 0.823 nats with live items (6 validation episodes, content over shuffled at noise level in both). About 1 s per step of 4 records |
| B9 | built, not trained: `train.py b9` (`src/schnitz/kb/stages/b9.py`, building blocks `src/schnitz/kb/experience.py`; 12 CPU tests). Per episode R rounds (default 3): the attempt generated with the KV cache, each emitted `memory_search()` executed as an L1 read (query at the call's closing parenthesis from a pass over the exact prefix, the span in the tool message; the protocol tokens match the template's rendering), scored by `schnitz.task_verifiers`; the SFT pass on the teacher transcript with its reads from the KB as it is at round t (task NLL plus retrieval loss; trains the L1 reader, items in place with `--item-lr`); then the single-pass write (`memory_write()` after the attempt, the frozen writer free-running in place, codecs, item-key heads), appended the first time and superseded after (one current record per task). Gold records (teacher text by the bank writer) read at w = schedule x decay^supersedes, applied as a gate factor where S_s combines the read items (gates scale mass, so w is its exact share; hidden at 0). Registry of records (model / hinted / gold, tasks whose gold is in the lineage); held-out rounds read only records without gold for their task (`--heldout-lineage any`: no gold at all), write their own records (never visible to training or later evaluations) and run the `removed` and `swapped` controls. Gradients into earlier rounds' writes (`--backprop-rounds k`, default 2): each training round's write is logged as a write source (site token ids, the attempt's read spans, the generated span and items, and per read its query state and items read); the SFT pass of round t reads the task's own record as its producers' recomputation (the shared L1b replay, bit-exact at the forward), so round t's losses reach the writer's span heads and codecs through round t-1's write, and the replay of that write recomputes its reads of the own record from round t-2's write (`L1Reader.reread`: forward the logged span exactly, gradient through the recomputed read), truncated after k writes; producers at their own rates (`--l1b-codec-lr`, `--l1b-writer-lr`); k = 0 keeps writes detached and writer and codecs frozen (CPU tests: gradient reaches the previous round's write at k >= 1 and not at k = 0; a depth-2 chain at k = 2 and not at k = 1; replay exact, reread equal to the logged read). The write site's cached prefix (`Model.prefix`) is a no-gradient cache, so a chained replay computes it with gradients (`producer.prefix_for`; same forward bit for bit, gradient reaches the site's inputs). GPU smoke (spider, 20 transcripts, random-init codecs at the new widths, B3 snapshot, 3 rounds, 2 steps x 2 episodes, k = 2): per episode 3 replays (rounds 0 and 1 at depth 1, round 0 again at depth 2 through round 1's read of it), bit-exact (`replay_match_exact` 1.0, drift 0, reread equal to the logged read), 18-26 s per step. Single-answer verifiers only (agent trajectories and APIGen-MT turns are skipped). GPU smoke (spider, 2 rounds, 6 steps x 2 episodes, 4 held-out episodes, `--open-with-search` because the B3 decoder never emits the call on its own): runs end to end with resume (restore of the live checkpoint with later commits discarded); accuracy 0 in every round and arm (the 350M answers `SELECT 1` or a guessed table); with the 8-step L1 reader, B9 records were 7-38% of the scored candidates but never among the items read, so round 1 read exactly what round 0 read; with the banks' initial heads the gold record took 6-10% of read mass at step 1 and held-out round 1 read its own record (4.5% of mass), 2 records became `hinted`. About 13-40 s per step |
| Teacher distributions | later phase; cache script smoke-tested (LFM2.5-1.2B-Base, 300 records: mass sums to 1, true token in the top 32 for 84% of positions); models in `/home/werg/sdkb-runs/hf-models` |
