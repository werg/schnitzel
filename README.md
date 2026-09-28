# SCHNITZELJAGD

*Schnitzeljagd* is German for a paper chase. One group sets off first and leaves
a trail: chalk arrows, scraps of paper (*Schnitzel*), little notes with hints. A
second group follows later, reads the signs and finds its way to the goal.

SCHNITZELJAGD (short: **schnitz**) teaches a small language model to play both
parts. As it works through a task, it leaves compact notes about what it saw
and learned. Later, perhaps in a completely different task, it finds the notes
that matter and follows them.

> Formerly **SDKB**. Older experiment records and links still use that name.

## Why

Small models can't hold everything in their weights, and they forget whatever
drops out of their context window. The usual fix is a bigger model. We want to
test a different trade: keep the model small and give it an external memory it
can write to, search and read from. More storage and a little more computation,
instead of more resident parameters.

The long-term aim is for the memory to stand in for much of what a larger
model keeps in its weights: a huge, dynamic mixture of experts, where the
"experts" are stored pieces of knowledge picked per query. That is why the
memory is not just a pile of separate documents. Each stored item should carry
parts of many sources, and each source should be spread over many items.

Plenty of systems already bolt on a memory by retrieving text. SCHNITZELJAGD
differs in three ways:

- **The notes are latent, not text.** A memory is a short sequence of vectors
  that the model reads straight into its input, like a compressed passage.
  Fifty tokens of experience might become a dozen of these soft tokens, or just
  a couple.
- **The model learns what to write.** Training rewards a note for how much it
  helps later, on a *different* task, not only for how faithfully it summarizes
  its source. The writer learns to leave the clues a future reader needs.
- **Reading happens while thinking.** The model can look things up in the middle
  of its own computation, between passes of its layers, not just once before it
  starts.

## How it works

```text
          ┌──────────── writing ────────────┐
 a task ──▶ model works on it ──▶ writes a few latent notes ──┐
                                                             ▼
                                              ┌──────────────────────────┐
                                              │  the bank (Zettelkasten) │
                                              │  many small records with │
                                              │  IDs, time, provenance   │
                                              └──────────────────────────┘
                                                             │
 a new task ──▶ model thinks ──▶ asks the bank ──▶ combines ─┘
                   ▲                                 │  the hits
                   └───── reads them, thinks on ◀────┘
```

**1. Writing.** While the model processes a trajectory (a conversation, a tool
session, a document), it emits memory records: variable-length runs of soft
tokens in the same space the model uses for its own input embeddings.
Compression can be light (about x4) for details that must survive, or heavy
(x32 and beyond) for the gist. Records carry an opaque ID, a timestamp and where
they came from.

**2. Storing.** Notes go into a bank that behaves like a *Zettelkasten*
(Niklas Luhmann's slip box): lots of small units, retrieved and recombined as
needed, never read front to back. Each note is spread over several spaces of
different granularity and width: fine spaces with many narrow items, coarse
spaces with few wide ones. No single space holds a whole note, and every item
has its own key, so one space may find a note that another misses.

**3. Reading.** When the model needs something, it forms a query from what it
has seen so far, never from the answer it is about to produce. In each space it
retrieves a neighbourhood of items, and a combiner for that space blends them
into one item for the query. A recombiner then joins the spaces into one span
that the model reads, much as it would read text. Items are weighted by gates,
and a gate can turn an irrelevant item all the way down.

**4. Learning end to end.** At training time, gradients flow from a later
task's loss back through the read, the combiner and the notes, into the
computation that wrote them. So the writer is shaped by how useful its notes
turn out to be. Inference is kept honest: it reads only the stored notes and
never goes back to re-encode the original experience.

**5. Tidying up, and superposing.** The combiner of a space works the same way
whether it answers a query or tidies the bank: it turns a neighbourhood of items
into new items of the same space. So as the bank grows, neighbourhoods can be
*compacted* into fewer items, which must still let every original be recovered
when queried by its own key, with the weight it carried. That is the
"holographic" part of the name: many memories superposed in one representation.
Training asks for this superposition directly rather than hoping it appears.

**6. Getting better by leaving better trails.** Agents don't message each other.
One writes, and later ones are steered by what they find, the way ants follow
and reinforce pheromone trails (*stigmergy*). The bank for a domain keeps
evolving: each new attempt at a task reads earlier attempts and supersedes its
own old record. Over time the bank accumulates know-how that any task in that
domain can draw on.

## Where things stand

This is an active research project, not a finished system. Some parts are well
tested, some are being built, and some are still plans.

- The pipeline runs end to end on a real small model (LiquidAI LFM2.5) on an
  NVIDIA DGX Spark. It covers writing, storing, serializing a bank, reopening it
  and reading from it.
- On controlled synthetic tasks, the model can combine separate stored notes
  into correct new actions. It keeps that ability when the notes are compacted.
- Precise recall of unseen details through latent notes is not solved yet. The
  first generation of notes turned out to carry too little of their content.
- The project is therefore restarting on a decoder that already reads
  compressed text well (a BGKit-style compressor). On that decoder the model
  already writes its own compressed notes well. Next comes the memory stack
  that spreads them over superposed spaces and recombines them, then real
  tasks: tool calling, SQL, code and text-world agents.

The details, with every number and its caveats, are in
[research status](docs/research-status.md). The current work plan is the
[restart plan](docs/bgkit-restart-plan.md), and the memory design is the
[knowledge-base stack](docs/knowledge-base-stack.md).

## Try it

The core tests and a tiny recurrent curriculum run on a laptop CPU and need no
downloads (Python 3.11+ with PyTorch):

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
schnitz launch --recipe recipes/tiny_looped_smoke.yaml --output runs/offline
```

Real-model training runs in an NVIDIA container on a DGX Spark. The
[getting started guide](docs/getting-started.md) walks through the Spark launch,
the training recipes and what each run measures.

## Where to read next

| If you want… | Read |
|---|---|
| to run things | [Getting started](docs/getting-started.md) · [Training](docs/training.md) · [Spark setup](docs/spark.md) · [Operations](docs/operations.md) |
| the full design | [Knowledge-base stack](docs/knowledge-base-stack.md) · [Architecture](docs/architecture.md) · [Recurrent reading](docs/recurrence.md) · [Implementation](docs/implementation.md) |
| what has been shown | [Research status](docs/research-status.md) · [Spark validation](docs/validation-spark.md) · [`experiments/`](experiments/) |
| what's happening now | [Restart plan](docs/bgkit-restart-plan.md) · [Backlog](docs/backlog.md) |
| the data | [Dataset guide](docs/datasets.md) |
| the name | [Naming](docs/naming.md) |

Contributors: please read [AGENTS.md](AGENTS.md) first. It lists the invariants
that keep the experiments honest.

## The acronym

**S**tigmergic **C**ompactable **H**olographic **N**eural **I**ndexed
**T**rajectory **Z**ettelkasten with **E**volving **L**atents, **J**ointly
**A**dapted by **G**ated **D**ecoders.
