# The knowledge-base stack: superposed spaces and MLP-matrix operators

**Status: design, 28 September 2026 (owner direction). The writer (restart plan
B2/B3) exists; everything else here is not yet implemented.** Section 9 tracks
what is built. This document replaces restart plan B7 ("combiners") and the
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
                 compactor C_s(neighbourhood | target key q_s) ──▶ one item in space s
        recombiner R(items of all spaces) ──▶ span (decoder input space) ──▶ decoder reads

compact: C_s(neighbourhood | cluster key) ──▶ fewer items in space s, written back
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
- **Compactor = combiner, per space.** C_s maps a neighbourhood of items of space
  s to an item of space s, conditioned on a target key. Because input and output
  live in the same space (closure), the same operator serves reads (combine a
  retrieved neighbourhood for a query) and compaction (replace a neighbourhood by
  fewer items, written back). Compaction can be repeated at several levels, and
  may add re-representation and superposition even where it does not shrink the
  store.
- **Recombiner (reverse codec)** R reads the items of all spaces and produces a
  span in the decoder's input space. Spaces may be missing (a space may not
  retrieve anything relevant), so R is trained with spaces dropped.
- **Output positions** are variable. In pre-training the target count is given by
  the target (the original span's n). At inference a compactor emits the average
  position count of its inputs, and R emits a span whose length follows from the
  spaces' position counts (n ≈ m_s / r_s).

## 4. The MLP-matrix operator

All three components (codecs, compactors, recombiner) use one operator family,
built for dense joint recombination of inputs. Attention retrieves sparsely at
each layer; here every source position contributes to every target position
through its own learned function of both positions.

Sources j = 1..n carry content x_j, a normalized position p_j = (j + 0.5)/n within
their item, a kind (which space or input they come from, each kind with its own
input projection, since widths differ) and a gate g_j >= 0. Targets i = 1..m
carry a residual state h_i, a normalized position t_i and an optional condition c
(the target key). One layer:

```text
z_ij = W_src[kind_j] x_j + P φ(p_j) + T LN(h_i) + Q φ(t_i) + D φ(p_j − t_i) + C c
a_i  = Σ_j g_j σ(z_ij) / Σ_j g_j            (numerator and mass; mass_i = Σ_j g_j)
h_i ← h_i + O a_i
h_i ← h_i + FFN(LN(h_i))
```

- σ(z_ij) followed by O is a two-layer MLP per (source, target) pair, indexed by
  both positions and conditioned on the target's residual state; because O is
  linear it commutes with the weighted sum, so the per-pair cost is one hidden
  vector (size H), not a full output vector: O(n · m · H) per layer.
- φ are Fourier features. The relative term φ(p_j − t_i) lets a codec align
  source and target positions; a compactor's neighbourhood has no meaningful
  cross-item order, so its sources carry only their within-item position.
- **Gates only modulate mass.** g_j scales source j's contribution to the
  numerator and the mass; it is not an input feature. A gate of 0 removes a
  source exactly; masses are kept so compacted items can be weighed against
  others (invariant 5), and a source's contributions sum to its mass over targets
  with normalized responsibilities (invariant 7).
- Several layers; each later layer's contributions depend on the target's current
  state h_i, so targets can specialize what they draw from each source.
- Target initial state: an MLP of φ(t_i), the log size ratio and the condition c.
  Output: a linear map of LN(h_i) to the target width, then a fixed-norm rms
  normalization (for the recombiner, BGKit's interface norm into the decoder's
  input space).
- Keys: the compactor's condition is the target key. In its first, short
  pre-training phase the compactor sees no input keys (it must combine by
  content); afterwards the input items' keys are added as source features.

## 5. Training stages

Every stage keeps the controls: the same information as text, no memory,
shuffled neighbourhoods or items, and an equal-byte baseline; content is
measured in nats over the shuffled control, not only as captured fractions.
The decoder that reads is the B3 decoder, frozen, unless stated.

- **K1 - Autoencoding through the spaces.** Forward codecs F_s and recombiner R:
  span → spaces → span. Losses: the frozen decoder reads R's output and
  reconstructs the source text (NLL) with a KL to reading the original span, plus
  a light cosine to the original span. Space dropout (each space removed with
  some probability, at least one kept) forces every space to carry part of the
  content and R to work with spaces missing. Inputs: first the cached S2 teacher
  spans of the bank (B1, available now), then the B3 writer's own spans (offline
  generation by the frozen writer; invariant 1). Gate: reconstruction through the
  stack close to reading the span itself; each space's ablation costs something.
- **K2 - Keys.** A key head per space, initialized by distillation from the R5d5
  key table; query heads likewise from R5d5's routing addresses.
- **K3a - Compactor warm-up, no keys, drop-one.** Neighbourhood of items in space s
  (nearest neighbours by key), the target item removed; C_s, conditioned on the
  target key only, produces an item from which R (with the other spaces)
  reconstructs the target's span. Only needs to be roughly right.
- **K3b - Compact, then recover (the superposition objective).** A neighbourhood
  of N items is compacted by C_s into M < N items (conditioned on cluster keys);
  then every one of the N originals must be recovered from the M compacted items
  queried at its own key (C_s again), through R and the decoder. Several records
  must share each stored item. Input keys become available to C_s here.
- **K4 - End-to-end reconstruction through retrieval.** Neighbourhoods retrieved
  from the stored KB per space, compactors, R, decoder: reproduce the original.
- **K5 - End-to-end tasks, decoder frozen.** Only the KB stack trains (codecs,
  compactors, recombiner, key and query heads). Tasks:
  - *spread-out use:* next-token prediction on held-out text of a domain whose
    corpus is in the KB, with many items each contributing a little; measured as
    NLL reduction per resident parameter, against no memory, shuffled KB, BM25
    top-k as text, and a larger model;
  - QA and the B9 task corpora as secondary checks.
- **K6 - Compaction levels.** Periodic compaction passes with C_s on stored
  neighbourhoods, written back; the store is evaluated before and after.

## 6. Relation to other plan stages

- B2/B3 (writer) feed K1. B4 (general compression) is unchanged.
- B5 (keys, query diversity) becomes K2 and applies per space; the query
  diversity measures (several queries per site, coverage discount, repulsion,
  exploration) carry over unchanged.
- B6 (reads at loop boundaries) reads through this stack.
- B9 (recursive improvement) stores its trajectories through the writer and this
  stack; one persistent KB per dataset still holds. Compaction only merges items
  within one authorization domain (one KB), and learned selection is never used
  as authorization (invariant 6).

## 7. Costs

Per layer and item the operator costs O(n · m · H): with H = 256, a 60-rep span
reconstructed from about 110 space positions is about 1.7M hidden activations
per layer. Widths, neighbourhood sizes (about 16-64 items) and position counts
are the levers; the dense per-pair form is kept on purpose.

## 8. Open questions

- Space count, widths and position ratios (the table above is a starting point).
- Neighbourhood sizes per space, and M/N in compact-then-recover.
- Whether R also receives the query (question-conditioned recombination).
- How compaction levels are scheduled once the store is large.

## 9. Status

| Part | State |
|---|---|
| Writer (B2/B3) | training (restart plan B3) |
| MLP-matrix operator | not built |
| K1 codecs and recombiner | not built |
| K2 keys | not built (R5d5 key table exists) |
| K3 compactor | not built |
| K4-K6 | not built |
