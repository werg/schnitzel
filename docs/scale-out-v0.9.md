# Scale-out storage and recursive compaction v0.9

**Status, 22 September 2026:** executable contracts and local reference machinery
are implemented. No network database backend, ANN performance result, or learned
compaction benefit is claimed.

This document prepares the mutable bank for network storage and persistent recursive
compaction. It extends the v0.8 lifecycle; it does not restore immutable bank
generations. Source trajectories and causal event history remain stable evidence.
Keys, payloads, index membership, compact codes, and their dependency graph continue
to change through journal revisions.

## 1. Network storage boundary

The model process uses three narrow capabilities:

1. `StoredReadBackend` fetches stored payloads for captured `ReadPlan`s, streams
   bounded chunks, and looks up stored keys and payloads. It has no source-text API.
2. `KeySearchBackend` adds scoped search for inference. `BatchKeyIndexBackend` adds
   batched search and exact selected-key lookup for training waves. Exact scan and ANN
   services may both implement this interface, but their telemetry must identify the
   search method.
3. `MutableRecoveryBackend` exposes the small cursor token, exact rollback, checkpoint
   pins, and revision garbage collection.

These protocols are defined in `schnitz.storage_contract`. The SQLite implementation is
the correctness reference. Stored sessions, corpus training, and spatial training now
depend on the public read/search protocols rather than a SQLite connection.

### Required remote semantics

* Search returns opaque record IDs, scores, full causal scope, and one bank cursor.
  Fetch uses that cursor. It must not silently read newer heads.
* Fetch rechecks namespace, authorization domain, generation, visibility time,
  deletion, invalidation, and cursor retention. A captured selection is not an
  authorization grant.
* A logical write publishes every configured space or publishes nothing. The commit
  advances one monotonic cursor and has an idempotency key plus content digest.
* Event writes publish record revisions, immutable provenance, lineage, dependencies,
  and the event visibility frontier in the same transaction.
* Updating a child invalidates every transitive compact descendant before the new
  child becomes searchable. Rebuilt descendants become searchable atomically.
* Checkpoint restore verifies store identity and commit digest before truncating an
  uncheckpointed branch. Garbage collection cannot pass retained cursor pins.
* Payload tensors use a declared dtype, shape, and checksummed serialization. RPC
  envelopes bound request count and bytes; they do not contain source trajectories.
* Timeouts and retries retain request identity. An uncertain write is resolved by
  reading its idempotency result, never by issuing a semantically new write.

### Conformance and performance gates

A candidate backend must run the core store, temporal-event, mutable-bank, checkpoint,
and rebuild-worker tests against its adapter. Add failure injection before commit,
after durable commit but before response, during batched fetch, during cursor restore,
and while a rebuild lease expires. Compare returned IDs and payload bytes with the
SQLite exact reference on fixed queries.

Performance reports separate exact from ANN search and cold from warm storage. Report
search, payload fetch, and end-to-end latency p50/p95/p99; bytes per request; candidate
recall; stale-cursor failures; retry counts; queue depth; retained activation bytes;
and GPU idle time. Network testing starts only after the completion-driven recurrent
microbatch scheduler can work on another ready continuation during retrieval latency.

## 2. Persistent recursive compaction

Compaction is an offline/background writer. Inference only reads published compact
payloads. A compact parent records the exact child revisions and overlapping-field
responsibilities that produced it. Responsibilities, including mass, sum to one for
each child contribution. Absolute key position is not an MLP input; local relative
geometry and density-scaled distances may be.

When a child changes, the mutable bank exposes invalid descendants as a durable
dependency queue. `claim_rebuilds` leases only the lowest ready dependency layer: a
compact parent cannot rebuild while one of its compact children remains invalid.
`rebuild_invalidated` performs explicit compactor inference and publishes the rebuilt
all-space parent through the ordinary journal transaction. Success clears its lease
atomically; failure releases the job for retry. Publication verifies the live lease
owner and compares every child cursor captured at claim time, so a slow worker cannot
publish stale output after a child changes or its lease is reassigned. This supports
compactions of compactions without pretending gradients pass through old bank history.

Overlapping local charts remain allowed. Several parents may include shares of one
raw record, provided the declared responsibilities conserve its total numerator and
mass. Full-cluster codes remain valid only for their full declared child selection.
Partial reads continue to use exact raw fallback until a separately trained and
validated selection-conditioned representation exists.

## 3. Quality and promotion

Contribution loss trains compactors, but it does not prove useful compaction.
`schnitz.compaction_quality.assess_compaction` records a held-out raw/compact comparison:

* raw-to-compact output KL;
* argmax disagreement;
* optional downstream NLL increase;
* serialized byte reduction; and
* record-count reduction.

A persistent view is promotable only when it passes configured behavior thresholds
and achieves measured serialized savings. The comparison must use the same queries,
reader state, selected evidence, serialized precision, and authorization scope.
Report raw fallback bytes, key/index bytes, compact-code bytes, rebuild compute, and
read compute. Keeping all raw payloads for arbitrary subset fallback is a useful
latency experiment but is not a net storage reduction.

Promotion proceeds by density and difficulty bands. Start with one-level compactors,
exceptions, and independent-fact controls. Add overlapping charts, then compacted
children. Each deeper level retains direct task supervision and raw neighborhood
teacher targets; recursive reconstruction error is not its sole teacher. Widen code
count or channels as density rises when held-out quality requires it.

## 4. Delivery sequence

1. Run local rebuild workers in shadow mode and record quality reports without making
   compact parents eligible for ordinary reads.
2. Add the completion-driven scheduler and storage latency telemetry.
3. Implement a remote adapter behind the storage protocols and pass semantic/failure
   conformance against SQLite.
4. Benchmark exact batched keys and contiguous payload objects. Introduce ANN only
   after measurements identify exact search as the limiting component.
5. Promote one-level compact views that pass behavior and byte gates; retain automatic
   raw fallback and counterfactual controls.
6. Enable recursive levels with leased dependency rebuilds, bounded maintenance work,
   and explicit garbage collection of unpinned superseded revisions.

The current Phase 1 run remains on raw local payloads. These additions prepare later
stages and do not alter its model or data fingerprint.
