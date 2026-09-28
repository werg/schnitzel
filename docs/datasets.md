# Teacher datasets and episode construction

Source documentation checked September 19, 2026. Upstream names/schemas/licenses
are from the linked primary cards. The parser tests use project-authored fixtures;
live downloads happen on the training machine. Model/data revisions and accepted or
filtered counts are recorded by preparation. No teacher weights or raw external
corpus is redistributed in this repository.

## Selected datasets

| Catalog key | Original source / configuration / split | Role in the curriculum |
|---|---|---|
| `hermes` | [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), `func_calling`, `train` | Multi-turn function calls, schemas and observations. Starter and tools recipe. |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), `default`, `train_sft` | Conversational continuation and use of earlier context. Starter and chat recipe. |
| `swe_smith` | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), `default`, `tool` | Coding-agent action/observation traces with resolved labels and instance identity. Prefix and cross-experience recipes. |
| `openhands` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), `default`, `train` | OpenHands traces of Qwen3-Coder-480B-A35B-Instruct with resolved labels. Coding comparison. |
| `xlam` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), default, `train` | Optional mostly single-turn schema/argument curriculum; not a cross-experience benchmark. |

The cards declare Apache-2.0 for Hermes, MIT for UltraChat/SWE-smith and CC-BY-4.0
for Nebius/xLAM. These are source-card licenses, not a blanket relicensing of
repository code embedded in trajectories. Preserve source/teacher/repository
provenance and required attribution in derivatives. The official xLAM source is
gated; it is excluded from the default recipe. LFM weights retain their own
[lfm1.0 license](https://huggingface.co/LiquidAI/LFM2.5-230M).

LFM's authors position the 230M instruction checkpoint for lightweight tool use and
data extraction. This motivates tools/chat first, followed by bounded coding
commands—not a claim that coding is impossible or that a model-card recommendation
establishes a fundamental capacity limit.

## What is distilled

The teacher target is the **recorded next assistant message**, not logits or hidden
states that the dataset does not provide. No live teacher or paid API is called.
The available per-row `model`/`teacher` identity is retained; otherwise it is recorded
as `not_recorded`. The normalized transcript and hard target drive supervised NLL
through the soft-memory interface.

Successful SWE/OpenHands rows are filtered using `resolved=true/1`. Outcome columns
only affect selection; they are not neural input features. Final `patch` and
`model_patch` fields are never injected into earlier writes/queries. Commands remain
inert text: no shell, browser, file edit, patch application or test suite is executed.
Lower teacher likelihood is not a measured software-engineering success rate.

## Explicit adapters

| Adapter | Input schema |
|---|---|
| Hermes | `conversations` containing `from`/`value`; optional `tools`, `id`, `model`. |
| UltraChat | `messages` with `role`/`content`; `prompt_id`. |
| SWE-smith | `messages` list or JSON, including double-serialized strings; `instance_id`, `traj_id`, `resolved`, `model`. |
| OpenHands | `trajectory`, `repo`, `instance_id`, `trajectory_id`, `tools`, `resolved`; structured calls and observations. |
| xLAM | `query`, `tools`, `answers`; converted to schema/user/assistant. |

Human/gpt/function aliases map to standard roles. OpenAI call argument strings and
Anthropic text/tool blocks are handled explicitly. Calls become canonical JSON
inside `<tool_call>` tags and retain linking call IDs. Tool observations retain
name/call ID where supplied. Source/target whitespace and Unicode code identifiers
are preserved. Unknown roles and unsupported visual blocks are rejected with
filter counts rather than guessed into text.

The outer prompt uses the student's actual chat template. Recorded call targets
use an explicit JSON wrapper. This is not a claim that the wrapper equals LFM's
native Pythonic tool-call format, which is documented in its model card. Native
format conversion and execution validation can be separate experiments.

## Prefix-memory protocol

For assistant target at turn `t`, choose a cutoff before `t`. Older messages are
written as independent fixed-budget source chunks. The query gets an original-task
excerpt and only recent **preceding** messages. The target and subsequent observations
are excluded from both writer and query. Assistant greetings before a first user
message are not eligible targets.

Actual student tokenization determines source chunks and prompt/answer budgets.
Source chunks preserve original characters rather than decoding split UTF-8 tokens.
A few target turns are sampled across each trajectory. Oversized complete assistant
answers are **skipped**, not truncated with a false EOS target.

The text control receives exactly the same selected chunks. If the shared text
prompt is too long, earlier selected chunks are removed and counted for both arms.
The original task/recent query can be explicitly excerpted. These are bounded
training windows, not a requirement that the eventual architecture limit its reads
because producer graphs do not fit. Nor is this already a learned whole-trajectory
lesson extractor: it is a concrete source-to-latent bootstrap curriculum.

## Cross-experience protocol

Support is taken from different already ordered instances in the same repository
(or explicit tool-schema group). Multiple attempts at the **same** issue cannot
serve as each other's support. Support draws bounded chunks from a prior task and
its recorded tail. Round-robin selection preserves multiple prior experiences when
the representation budget permits; provenance lists only retained producers.

The ordering is deterministic and explicitly **experimental**, not asserted to be
the original collection chronology. Prior experiences can include their own completed
solutions because they precede this query in the experiment. The query instance's
future solution is never a prior source. Source IDs incorporate the experimental
time so immutable identity remains consistent across serialized episodes.

Related experiences are not assumed sufficient for every later query. Real-data
episodes label them `provided_context`, with no invented sufficient-group annotations.
The controlled causal suite provides known support groups and source replacements.

## Splits and contamination controls

Split before windowing. All SWE trajectories from one repository stay in one split.
Tool schemas or normalized first-user prompt identities group the other sources.
Exact normalized transcript duplicates are removed across selected sources. This
is not semantic deduplication or external benchmark decontamination; those limitations
are stated in the manifest rather than hidden behind a generic clean-data claim.

Only UltraChat `train_sft` is used; the test and generation-ranking splits are not
training inputs. Validation is a group-held-out subset of the requested upstream
split. A deterministic 256-row shuffle buffer improves local source mixing but is
not a uniform sample of an entire large corpus. `max_rows` limits scanned rows, not
accepted examples or exact bytes downloaded. A source/split yielding no usable
examples fails before training with filter counts: increase the sample budget or
choose a new seed rather than weakening splits silently.

Each prepared episode records original dataset SHA, declared license, known teacher,
trajectory/instance identity, target index/hash, complete-target flag, source indices,
retained producer IDs and experimental/original ordering type. These metadata are
not writer inputs. Training validates unique episode IDs and consistent source IDs,
hashes input files, then uses offsets rather than keeping every episode's text in RAM.

## Redundant knowledge tasks (28 September)

These two task corpora are for the L1 read phase. The answer can only come from the
KB, and the KB is highly redundant, so finding a useful record is easy. Both are
subcommands of `scripts/prepare_task_corpora.py` in the tasks-* format. Their
generators are importable modules with tests that need no downloads. Records are
`kind: passage` with `provenance.record_type`. An episode lists every record that
states its answer as a support. It also carries `alternatives`: per hop (people) or
per target segment (recall), every record that holds that part whole.
`prepare_memory_transcripts.py` names these in the slot of the read
(`slot.alternatives`, audited like `record_ids`). `schnitz.kb.bank.needed_records`
banks them, so the L1 KB contains every copy, not just the one a transcript reads.
The L1 retrieval loss still takes only `record_ids` as positives. A copy that is
merely in `alternatives` can be scored as a negative among the candidates.

- **`recall-text`** (`schnitz.recall_text`). The sources are Wikipedia articles from
  `/archive/raw/background-20260927/wikipedia-{sql-domains,household,logic}`, with
  chunks joined in page order. Articles of 200 to 1500 LFM2.5 tokens are kept (1066).
  Records are `--window` token windows every `window / --redundancy` tokens, widened
  to whole words. Shorter head and tail windows keep the document ends as redundant
  as the middle. Each document also gets a title record (title and lead paragraph).
  The episodes are `continuation` (1 to 2 sentences given, then the next 64 to 256
  tokens), `title` (the first 128 tokens of a named article) and `middle` (one
  sentence given, then the following 64 to 192 tokens). Targets end at a sentence
  end when one leaves at least 64 tokens. Sufficient groups are the covers by one
  residue class of windows, so there are `redundancy` covers. The split is by
  document. Corpora: `tasks-recall-text-r8-20260928` (71,807 records, 3800/464
  episodes; target tokens median 117, p90 188; about 18 supports per episode) and
  `-r2-` (18,353 records), each with `memory-recall-text-r{8,2}-20260928v3`.
- **`synth-people`** (`schnitz.synth_world`). This is a seeded fictional world with
  unique natural names, built-in vocabularies of real cities, plausible
  universities, majors, companies and job titles, and mentor and sibling relations.
  Every asked fact is stated in exactly `--redundancy` distinct records, counted
  across bios (templated and permuted, with pronouns in a share of sentences),
  company rosters, city birth registers and alumni lists. Companies have their own
  profile records for 2-hop questions (`--hops 2`). Episodes ask one attribute of one
  person, with the short value as the answer. The split is by person: validation
  people's records are in the KB, but no training question asks about them. The
  purpose is a controlled redundancy knob; contamination is not a concern at 350M.
  Corpora (seed 0, 2000 people, 1-hop): `tasks-synth-people-r32-20260928` (66,867
  records, 15100/1426 episodes) and `-r4-` (8,627 records), each with
  `memory-synth-people-r{32,4}-20260928v3`.
## Task corpus: parallel-version recall

`scripts/prepare_task_corpora.py parallel-recall` builds `tasks-parallel-recall-<date>`
in the task-corpus schema. `scripts/prepare_memory_transcripts.py` turns it into
`memory-parallel-recall-<tag>` transcripts (format 3). The generator is
`schnitz.parallel_recall` and its tests are in `tests/test_parallel_recall.py`. The task
is a knowledge probe whose KB holds many *versions of the same content*. The target
has high entropy for the model alone but is nearly determined by the KB.

- **Data.** Verse-aligned CSVs of public-domain (or CC0) Bible translations from
  [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases):
  `master` at `e1b254c` (2026-07-10), and the legacy `2024` branch at `19e9663` for
  the World English Bible. The raw files are in `/archive/raw/parallel-20260928/`
  (host `/mnt/external/sdkb-archive/raw/...`). Its README lists each version's
  declared licence and the versions skipped (non-PD, or PD status unclear).
  `schnitz.parallel_recall.VERSIONS` lists the 25 English and 22 French, German,
  Spanish, Chinese and Japanese versions obtained. No Arabic version is available
  in the source.
- **KB records** (`kind: parallel_passage`, `created_at` 1). Each record is a run
  of 4-10 consecutive verses of one chapter, at most `RECORD_CHARS` (1500)
  characters, headed `<title> [<code>], <Book> <c>:<v1>-<v2>`, one numbered verse
  per line. Chunk boundaries are drawn per version. Text cleaning removes footnotes
  (`{...}`), tags and italic brackets. A version keeps a book only if at least 90%
  of the book's chapters have the reference (KJV) verse count. It keeps a chapter
  only if the chapter's count matches, and for English only if the text lines up
  verse by verse. This removes, for example, Vulgate-numbered Psalms (DRC, CPDV) and
  the JPS books with Hebrew numbering. The stored versions come from `--versions`,
  or else all versions of `--kb-languages` (default `en`). The KB stores the whole
  text of every stored version, including passages no episode uses.
- **Targets** (`--target-versions`, default `WEB BBE`) are modern, less-memorized
  translations. They are **never stored**, so no record contains a target. An
  episode is also rejected (`target_in_kb`) if any stored version equals the target
  word for word over the whole range.
- **Episodes.** A target is a run of consecutive verses in one chapter of 64-200
  tokens (LFM2.5 tokenizer with `--tokenizer`, otherwise characters/4), one numbered
  verse per line. There are two query modes. `reference`: "Give <Book c:v1-v2> in the
  <title> (<code>) wording". `continuation`: the target version's preceding verse is
  quoted and the next N verses are asked for, without a reference. In `supports`,
  each covering stored version's overlapping records form one `sufficient_groups`
  entry. `required_ids` is their union. `redundancy` is the number of covering
  versions (`--min-versions`, default 3). `provenance` records `covering_versions`,
  `verbatim_verses` (target verses that some stored version has word for word),
  `nearest_version` / `nearest_similarity` (word-sequence ratio) and `target_tokens`.
- **Split** by chapter: about 10% of chapters (hashed) are validation. Validation
  passages are never trained on, although their other versions are in the KB.
- **Transcripts.** The builder searches once per covering translation (`kind`
  `parallel_passage`, space hint `fine`), all before the answer. With the v3
  defaults, 30% of the multi-call sites are split into sequential calls. There are no
  write sites, and the role line is "Other translations of the requested text are in
  the knowledge base."

Built 28 September (`tasks-parallel-recall-20260928`, `memory-parallel-recall-20260928v3`,
same transcript options as the other v3 corpora, LFM2.5-350M render check on 200 per
split):

- The KB has 23 English versions and 90,651 records.
- There are 4000 train episodes (712 chapters) and 300 validation episodes (54 of the
  119 validation chapters), half WEB and half BBE. 47% are continuation queries.
- Redundancy is 11-22 (median 17). Old-Testament passages are covered by about 16-18
  versions and New-Testament passages by 20-22.
- Target length: quartiles of 92/111/131 tokens.
- How close the nearest stored version comes differs by target. For WEB the median
  similarity is 0.97, because its derivative NHEB is stored. For BBE it is 0.60,
  because no stored version is close to it. Use `--versions` without `NHEB-JE
  NHEB-ME` for a harder WEB condition. 19% of target verses appear verbatim in some
  stored version.
- Transcripts average 17.7 `memory_search()` calls per episode, with about 23 slot
  records.

## Stored-only measurement

Evaluation has a separate write phase, serializes payload precision, reopens the bank,
and reads with per-trajectory namespaces and causal timestamps. It includes no-memory,
zero-value and fixed-key/ID payload-permutation controls. Distinct records can share
semantics, so permutation is not mislabeled a guaranteed contradictory counterfactual.
Causal source changes with adjusted targets belong to the controlled family.

The first real-data comparison is teacher NLL at equal source access. The next is
fresh-experience causal transfer. Agent execution/task success remains a separate
future measurement requiring an environment; it is not silently inferred from either.

## Recipe example

```yaml
sources:
  - name: swe_smith
    max_rows: 4000
    shuffle_buffer: 256
preparation:
  max_supports: 4
  max_targets_per_trajectory: 3
  recent_messages: 1
  recent_tokens: 512
  task_tokens: 160
```

The base YAML separately controls source/prompt/target limits (512/4096/512 initially),
write/code/read capacity, and training behavior. Inspect `data/manifest.json` before
scaling. A supplied source revision is honored; otherwise preparation resolves a
commit SHA. Local fixtures are content-hashed and use the same conversion path.
