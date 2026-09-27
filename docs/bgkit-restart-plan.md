# SDKB restart on the BGKit S2 decoder: plan

**Status: proposal, 27 September 2026. Nothing here is implemented.**
Owner direction (27 September): start fresh from the jointly trained BGKit 350M
decoder; the decoder emits BGKit-format representations as its memory output,
with variable length; spatial operations work on variable-sized records; the
model can also produce compressed output when prompted. Keys are distilled from
the current R5d5 model; payloads from the BGKit S2 encoder; combiners from
multi-level BGKit. Unfreeze step by step.

## 1. Why restart

R5d5 retrieves well but its stored values carry almost none of their content.
Probes on checkpoint 1300 (`scripts/evaluate_key_table.py --conditions`,
`scripts/make_memory_probe_sets.py`, answer tokens only):

| Probe (256 each) | no read | normal read | gold forced | shuffled | gold as text |
|---|---|---|---|---|---|
| Invented passages, exact 8-word spans | 9.08 | 8.93 | 8.92 | 8.98 | 1.97 |
| Mixed Hotpot/public validation | 2.768 | 2.686 | 2.683 | 2.702 | 1.141 |

Memory recovers 2% (invented) and 5% (real questions) of the text arm's gain,
although search delivers the gold record 89–99% of the time. Earlier gates only
compared correct against zeroed/swapped values or against a weak teacher, never
against the same information as text. Every gate below includes the text arm.

## 2. Base and teacher

- **Base decoder:** `bgkit2_s2_showcase` step 19999
  (`/mnt/external/bgkit-checkpoints/bgkit2_step19999_20260925_103158_274684_run-bgkit2_s2_showcase/decoder.pt`):
  `LiquidAI/LFM2.5-350M` (snapshot `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`,
  hidden 1024, 16 layers, vocab 65,536) with a rank-32 LoRA on layers 0–7. The
  owner confirmed this is the jointly trained decoder; no later 350M variant exists.
- **Teacher encoder (training only, never part of inference):** the same
  checkpoint's `encoder.pt`, `LiquidAI/LFM2.5-Encoder-350M` (snapshot
  `b886781f7c6f10ca9b7096e21b83e30a073c2f39`). Load both with
  `bgkit2.training.standalone.load_models`, strictly.
- The SDKB model moves from LFM2.5-230M (`40cb2ad3…`) to LFM2.5-350M. Existing
  230M checkpoints remain for comparison; tiny-model core tests stay download-free.

BGKit S2 facts this plan relies on (BGKit repo, checkpoint `metadata.json`):
reps are the encoder's last-layer states at top-k survivors (k = ceil(r·N),
variable), passed through a projection block, an MLP projection and
`DecoderInterfaceNorm` into the decoder's input-embedding space, and spliced into
the decoder input in place of a sentinel token. Which positions survive barely
matters (learned vs random: SQuAD 0.615 vs 0.615); the ratio and the prompt do.
Generic reconstruct-prompt quality (fraction of full-text gain captured):
x4 0.87, x8 0.68, x16 0.47, x32 0.30, x64 0.19. A wrong question prompt is worse
than no context.

## 3. Representation format

### 3.1 Write prompt and ratio
Records are written before any question exists and inference never re-encodes
a source (invariant 1), so the teacher encodes with BGKit's question-free
reconstruct prompt ("Compress the following text so that it can be reproduced
verbatim."). Question-conditioned BGKit encoding is not used for memory.

### 3.2 Variable-length autoregressive output
The decoder emits a compressed span:

```
<text …> <|bg|> r_1 r_2 … r_k <|/bg|>
```

- Two new control tokens; inside a span the LM head predicts `<|rep|>` (emit a
  representation here) or `<|/bg|>` (stop). The span length is a decision of the
  ordinary head, trained to k = ceil(r·N) for the requested ratio r.
- A rep head (MLP on the final hidden state, then a copy of BGKit's interface
  norm) produces r_i in the decoder's input-embedding space; r_i is the next
  position's input embedding. The ratio is stated in the prompt, so one model
  can compress at x4, x8, x16, …
- Reps are ordered as the teacher's survivors (document order).

### 3.3 Spaces become ratios (proposed)
R5d5's four spaces were four widths of one record (4/8/16/36 tokens). Here each
space is the same record compressed at a different ratio (e.g. x4, x8, x16, x32),
each with its own variable-length payload and its own key. Coarse spaces give
cheap wide reads; fine spaces give detail. This keeps R5d5's four key spaces as
distillation targets.

### 3.4 Storage
Payloads become variable-length sequences of 1024-d bf16 vectors per record and
space (about 2 KB per rep). A 700-character passage (~160 tokens) is about 40 reps
at x4 and 20 at x8, 80 KB and 40 KB. The store and key index keep one key per
record and space; the payload blob gains a rep count. Offline bank creation is
generation by a frozen writer.

## 4. Training stages

Each stage names what trains, what is frozen, and its gate. Gates always report
the text arm, no-read and shuffled controls, on held-out and invented passages.

**B0 — Harness parity.** Load S2 into SDKB's harness (separate container with
the BGKit source and checkpoint mounts; the current container lacks both).
Reproduce BGKit's own numbers when the decoder reads *teacher* reps: reconstruct
token accuracy at x4/x8/x16 on BGKit's eval split and on our passages, and QA
from teacher reps. Gate: within noise of `metadata.json`.

**B1 — Teacher cache.** Encode every bank source with the frozen S2 encoder, the
reconstruct prompt, at each space ratio; store survivors in document order.
Invented-passage and held-out sets are encoded the same way. (GPU job; announce
to the BGKit session. Roughly 470k sources × 4 ratios.)

**B2 — Rep generation, rep head only.** Decoder frozen (including its LoRA);
train the rep head and the stop decision, teacher-forced. Losses: cosine and norm
against teacher reps; functional loss — the frozen decoder reading student reps
reconstructs/continues the text (NLL) and matches its outputs when reading
teacher reps (KL). Gate: fraction of the teacher's reconstruct gain recovered
from student reps, per ratio.

**B3 — Free-running and write adapter.** Add a write-side adapter active only
inside compressed spans (reading stays exactly S2). Train on the model's own
generated spans (scheduled sampling to full rollouts) with the functional loss.
Then unfreeze further in steps (write adapter → upper layers), with a replay KL
to S2 on ordinary text and reading tasks. Gate: own-rep reading within a stated
margin of teacher-rep reading; S2 reading and chat ability unchanged.

**B4 — General compression capability.** Mix BGKit's autoencode and task data
with prompted compression ("compress at x16") and reading tasks over the model's
own spans, so compressed output is a general skill, not only a memory write.

**B5 — Key distillation.** Key heads read the state at `<|/bg|>` per space;
query heads read the query state. Targets: R5d5's key table rows (records) and
R5d5's routing addresses (queries), cached offline. Gate: retrieval recall with
distilled keys on R5d5's validation sites close to R5d5's (0.94 union at
checkpoint 1000).

**B6 — Reads by splicing.** A read inserts retrieved records' rep sequences
(each wrapped in its markers) into the read workspace under a rep budget per
space, instead of the MLP reader's fixed slots. Build the bank by frozen-writer
generation; train reading with retrieval (gold-forced at first, then annealed).
Gate: memory-vs-text probe fraction far above R5d5's 5%; invented-passage
fraction above 50% at x4.

**B7 — Combiners.** Multi-record compaction by the same mechanism: the decoder
generates a node span from its children's spans at 1/4 (BGKit tree nodes). First
distill BGKit's multi-level node encodings (`encode_tree`, identity bridge), then
train on task loss. Mass/responsibility invariants (5, 7) carry over to how
children are weighted.

**B8 — Spatial training.** Resume the bank curriculum, key table and record
gradients on the new format, with the R6 corpus (187,813 episodes, 13 new
datasets) and periodic memory-use evaluation.

## 5. Open decisions

1. Space ratios (x4/x8/x16/x32 proposed) and whether x4 is affordable for the
   full bank (about 80 KB per passage per space).
2. Whether to keep the recurrent middle-block loops (R5) in the first B6 runs
   or start one-pass and reintroduce loops after reading works.
3. Whether reads carry per-record relevance (gates) when splicing; BGKit has no
   weighting, and scaling reps would leave the interface-norm distribution.
4. Whether R5d5 keeps training until B5 needs its keys (it is the key teacher).
