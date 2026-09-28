# The knowledge-base stack: superposed spaces and MLP-matrix operators

**Status: design, 28 September 2026 (owner direction).** The writer (restart
plan B2/B3), the MLP-matrix operator and the K1 training script exist; section 9
tracks what is built and running. This document replaces restart plan B7 ("combiners") and the
restart plan's reading of "spaces" as BGKit compression ratios.

## 1. Goal

The knowledge base (KB) is meant to hold knowledge that would otherwise live in
the parameters of a much larger language model, so that the KB plus a small
decoder can replace that model in a modular way. It trades disk storage and some
latency for resident memory: a very large, dynamic mixture of experts whose
"experts" are stored items, selected per query.

For that, stored items must not be isolated copies of source documents.
Retrieval-augmented QA works fine with localized records (retrieve the right
passage, read it), and our training tasks started there, which biases intuition
towards "find the gold record". A parameter store needs the opposite: every
item carries parts of many sources, every source is spread over many items, and
a query is answered by combining many items densely. **Superposition is an
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
  about the size of the input (sum of r_s · d_s ≈ 1024 per input rep): the stack
  as a whole does not compress, but each space alone is a bottleneck, so no
  single space can carry a record and the recombiner must combine spaces.
  Starting point (to be tuned):

  | Space | r_s (positions per input rep) | width d_s | r_s · d_s |
  |---|---|---|---|
  | A (fine) | 1 | 384 | 384 |
  | B | 1/2 | 512 | 256 |
  | C | 1/4 | 768 | 192 |
  | D (coarse) | 1/8 | 1024 | 128 |
  | total | | | 960 |

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

### 5.1 Order

1. **Writer, B3** (running): the decoder writes BGKit-style spans.
2. **Decoder capabilities, B4 and the soft I/O port** (restart plan), while the
   decoder still trains: general compression at all ratios, then the soft input
   port and the soft output port ramped to half of the tasks.
3. **K1 - Autoencoding through the spaces** (running, in parallel). Forward
   codecs F_s and recombiner R: span → spaces → span. Losses: the frozen decoder
   reads R's output and reconstructs the source text (NLL) with a KL to reading
   the original span, plus a light cosine to the original span. Space dropout
   forces every space to carry part of the content and R to work with spaces
   missing. Inputs: first the cached S2 teacher spans (B1), then the writer's
   own spans (offline generation by the frozen writer; invariant 1). Gate:
   reconstruction through the stack close to reading the span itself; each
   space's ablation costs something.
4. **K2 - Keys.** A key head per space and query heads, initialized by
   distillation from the R5d5 key table and routing addresses; only a starting
   point, since L1 trains routing end to end.
5. **K3a - Superposition operator warm-up** (no keys, drop-one). A neighbourhood
   of items in space s, the target item removed; S_s, conditioned on the target
   key only, produces an item from which R reconstructs the target's span. Only
   needs to be roughly right, so that reads work in L1.
6. **L1 - Live items, end to end** (the decoder frozen, so knowledge has to land
   in the KB). Items start as the codecs' output and are then updated in place by
   gradients from reads (sparse updates, optimizer state per item), a fast loop
   to real superposition. Jointly trained: items, S_s (as the read-time
   combiner), R, key and query heads.
   - *Reads* are `memory_search` tool calls in LFM2's native format (trajectory
     memory v0.5; restart plan 3.2): the query is formed from a middle layer's
     state at the call, so retrieval can start while the call finishes; the
     result is a tool message whose content is the latent span between `<|mem|>`
     and `<|/mem|>`; everything earlier keeps its cache (causal). Writes are
     `memory_write` calls. A parameter store is consulted often, so SFT and
     B9 data carry many calls per trajectory (several queries per site, query
     diversity per B5); each call costs its few envelope tokens plus the span.
   - *Routing through gates:* each space retrieves a generous candidate set and
     every candidate's gate comes from its query-key similarity; gates scale mass
     exactly, so the task loss trains keys and query heads (the routing half of
     a mixture of experts).
   - *Tasks:* reconstruction and QA over the R6 KBs, trajectory SFT over the task
     KBs, continuation of KB-domain text.
   - *Superposition pressure:* the storage budget and recursive rewriting passes
     with S_s between updates (section 5.2).
7. **L2 - Learn to reproduce the live items.** The writer, forward codecs and
   *recursive* applications of S_s are trained to produce the L1 items from the
   sources alone: the in-place-trained items are the targets. This is where the
   rewrite-then-recover objective lives (a neighbourhood rewritten by S_s into M
   items, M < N compaction, M = N pure superposition; every original recoverable
   at its own key), with L1's items as targets instead of self-reconstruction.
   Afterwards the producers derive superposed items live from a new corpus.
   S_s can also be distilled on L1's rewriting trajectories.
8. **B9 loops and continual learning.** Multi-round attempts written back into
   the per-dataset KBs; new corpora enter through the producers; periodic
   rewriting; the KB-dependence tests (section 5.2).
9. **Later:** teacher distillation (section 1.1) and joint co-training of the
    decoder with the stack under replay.

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
- **Context cost of spliced reads.** A 30-rep span every 64 tokens lengthens a
  sequence by about 47%; injecting read results into the hidden states, like
  expert outputs, is the later alternative.
- **Gradients into producers.** In L1 items are detached from the writer; L2
  reconnects producers by distillation. Where task gradients must reach a
  producer directly (B9 across rounds), selective producer replay applies
  (invariant 3).
- **Store contracts.** Per-space variable-width items with masses, keys and
  rewrite lineage extend the mutable-bank (v0.8) and scale-out (v0.9) contracts;
  exact-scan index first, ANN measured separately (invariant 8).
- **Writes during trajectories** (B9) are `memory_write` calls: supervised at
  episode or round ends first, learned write sites later.

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
  transcript with `memory_search` calls (query text, target record ids), tool
  results as latent slots filled at training time, and `memory_write` calls;
  checks that results only contain records that exist before the call and that
  query text never leaks the answer.
- **WP4 - Evaluation** (`scripts/evaluate_kb.py`). Superposition metrics (sources
  per item, items per source, effective items per read), KB-dependence tests
  (edited, removed and inserted knowledge), and a benchmark runner for
  LFM2.5-8B-A1B, Ling-3.0-tiny and LFM2-24B-A2B on the task corpora with their
  verifiers.
- **WP5 - Stack training after K1**: K2 key and query heads, K3a warm-up of S_s,
  the L1 read path (retrieve per space, gates from similarity, S_s, R, `<|mem|>`
  span).

## 10. Status

| Part | State |
|---|---|
| Writer (B2/B3) | training (restart plan B3) |
| B4 general compression, soft I/O port | built, not trained: `--protocol` in `scripts/train_bgkit_reps.py` (`src/schnitz/span_protocol.py`: protocol input/output rows via hooks on the untied embedding and LM head; the writer's stop head becomes the `<|rep|>`/`<|/bg|>` rows; losses for opening `<|bg|>` and ending the turn after `<|/bg|>`; ratio-stated prompts with a ratio head; memory spans between `<|mem|>`/`<|/mem|>` in reads; QA questions through the soft input port, ramped). GPU smoke test from B3 step ~14000 passed, including resume; the untrained port already nearly matches the text question (s0 answer NLL 1.21 vs 1.20). Launch after B3 |
| MLP-matrix operator | built (`src/schnitz/mlp_matrix.py`, 8 property tests), locality kernel since 28 Sep |
| K1 codecs and recombiner | training (`scripts/train_kb_codecs.py`, run `kb-k1`, restarted 28 Sep with the locality kernel: B1 teacher spans, B3 reader at step 11500, 28M parameters; the uniform-weight run is kept as `kb-k1-uniform`) |
| K2 keys, K3a superposition operator | not built (R5d5 key table exists) |
| L1, L2 | not built |
| Teacher distributions | later phase; cache script smoke-tested (LFM2.5-1.2B-Base, 300 records: mass sums to 1, true token in the top 32 for 84% of positions); models in `/home/werg/sdkb-runs/hf-models` |
