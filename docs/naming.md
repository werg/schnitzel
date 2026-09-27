# Project name: SCHNITZELJAGD

**Status, 27 September 2026:** repository renamed; package, CLI, imports, agent class and
active documentation now use `schnitz` / `SchnitzelAgent`. Legacy `sdkb` import and CLI
aliases remain for compatibility with existing training processes and checkpoints.

The project is renamed from SDKB (Spatially Superposed Differentiable Knowledge
Base) to **SCHNITZELJAGD**:

> **S**tigmergic **C**ompactable **H**olographic **N**eural **I**ndexed
> **T**rajectory **Z**ettelkasten with **E**volving **L**atents,
> **J**ointly **A**dapted by **G**ated **D**ecoders

The standalone short form (commands, package, imports) is **`schnitz`**. In
compound identifiers the fuller *Schnitzel* reads better, e.g. `SchnitzelAgent`.

## Schnitzeljagd

*Schnitzeljagd* is the German children's version of a treasure hunt or paper chase.
One group sets off first and leaves a trail of signs behind: arrows chalked on the
path, paper scraps (*Schnitzel*), notes with hints. A second group follows later,
reads the signs and tries to reach the goal. The name means "scrap hunt".

This is the same pattern as the system: an agent working through a trajectory
leaves latent records behind, and later agents that never saw that trajectory
follow those records to act. The writer lays the trail; the reader hunts along it.

## Stigmergic

Stigmergy is coordination through traces left in a shared environment. Pierre-Paul
Grassé coined the term in 1959 for termite nest building, from Greek *stigma*
(mark) and *ergon* (work): the work leaves a mark, and the mark directs the next
worker. Ants following and reinforcing pheromone trails are the familiar example.

Agents here do not message each other. One writes records into the bank; later
agents are steered by what they retrieve from it. The bank is the shared
environment, and the records are the pheromone.

## Zettelkasten

A *Zettelkasten* ("slip box") is a note-taking system made famous by the sociologist
Niklas Luhmann: a box of small, self-contained note slips (*Zettel*), each with its
own identifier, retrieved and combined as needed rather than read as one document.

The bank is a Zettelkasten of latent records: many small units with opaque IDs and
provenance, retrieved by key and combined at read time. *Zettel* are also the paper
scraps of the Schnitzeljagd, which is how the two halves of the name meet.

## Holographic

Holographic memory stores many items superposed in one representation and recalls
from it by probing with a query. Compaction is the holographic part: a cluster of
records is replaced by a compact code, the original records are discarded for
inference, and the reader recovers the cluster's query-dependent contribution and
mass from that code alone. Unlike a holographic reduced representation, the
encoding and readout are learned rather than a fixed binding algebra.

## Naming conventions for the rename

| Use | Name |
| --- | --- |
| Full name (papers, README title) | SCHNITZELJAGD |
| Package, CLI command, import | `schnitz` |
| Project in prose | Schnitz |
| Compound identifiers, e.g. agent class (replacing `SDKBAgent`) | `Schnitzel…`, e.g. `SchnitzelAgent` |

When renaming, do not edit historical result records (AGENTS.md); add a
"formerly SDKB" note in the README so older records and links remain
understandable. Check PyPI, GitHub and arXiv for collisions with `schnitz` before
publishing.
