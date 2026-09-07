# ACE-3 Documentation

This index separates orientation, implementation contracts, reproducibility,
and accepted results so readers can quickly identify what ACE-3 does and does
not establish.

## Start here

| Document | Purpose |
| --- | --- |
| [README](../README.md) | Project overview, execution boundary, and common commands |
| [Current status](STATUS.md) | Accepted, active, planned, and explicitly unclaimed boundaries |
| [Getting started](GETTING_STARTED.md) | Dependencies, model assets, and reproducible commands |
| [Architecture](ARCHITECTURE.md) | Arithmetic and execution architecture |
| [Roadmap](ROADMAP.md) | Ordered milestones from W4A16 to implementation evidence |

## Runtime and integration

| Document | Boundary |
| --- | --- |
| [First Voice Hybrid RTL](FIRST_VOICE_HYBRID_RTL.md) | Persistent per-layer RTL state, trusted lineage, compact builds, and token-major execution |
| [RTL traceability](../design/RTL_TRACEABILITY.md) | Requirement-to-source-to-evidence mapping |
| [RTL manifest](../design/RTL_MANIFEST.json) | Machine-readable RTL inventory |
| [Contracts index](../ace3/contracts/README.md) | Machine-readable arithmetic and interface contracts |
| [Model/oracle index](../ace3/model/README.md) | Independent reference and vector-generation tools |

## Reviewed results

Result notes are scope-bounded. A note about one layer, operator, or simulator
does not certify a later integration level.

| Result | What it establishes |
| --- | --- |
| [AWQ W4A16 G128](results/AWQ_W4A16_G128_CF01.md) | Native G128 arithmetic and protocol boundary |
| [Full-input projection](results/AWQ_W4A16_PROJECTION_CF02.md) | Complete 896-input reduction for selected official outputs |
| [Q-projection single-round RTL](results/ACE3_Q_PROJECTION_RTL_20260906.md) / [Compact summary](results/ACE3_Q_PROJECTION_RTL_20260906.json) | Bounded reviewed Icarus component: 16 official channels plus 13 directed cases; opt-in dot+bias single rounding, with full-model wiring still pending |
| [Q-only RoPE bounded RTL agreement](results/ACE3_Q_ONLY_ROPE_BOUNDED_RTL_20260907.md) / [Compact summary](results/ACE3_Q_ONLY_ROPE_BOUNDED_RTL_20260907.json) | Independently reviewed experimental candidate: 27 actual RTL transactions across layers 0–8 / positions 0–2, 513 passing FP16-interstage comparisons, and 5,640 operator cases; not full-model generation or binary64 certification |
| [Model24 systematic continuations](../results/model24-systematic-continuations/) | Independently reviewed software/oracle continuation evidence |
| [Selected engineering highlights](results/ACE3_ENGINEERING_HIGHLIGHTS_20260906.md) / [Numeric summary](results/ACE3_ENGINEERING_HIGHLIGHTS_20260906.json) | Completed, bounded decoder/head/host milestones and one token producing `Hello world!` |
| [Software continuation preview](results/ACE3_SOFTWARE_CONTINUATION_PREVIEW_20260906.md) / [Numeric summary](results/ACE3_SOFTWARE_CONTINUATION_PREVIEW_20260906.json) | Separate CPU-software reference example: all 32 generated tokens from a fixed plain-text seed, capped without EOS; not RTL-generated |

The 2026-09-06 highlights select completed local engineering results, with
their actual precision and fixture boundaries. They are not a comprehensive
evaluation or a self-contained runtime source/reproduction release. Component
agreement and one host-token append do not establish full-model reference
equivalence, persistent-KV second-token completion, dialogue, or hardware results.
The separate software preview demonstrates its own bounded software K/V
continuation; it does not promote the RTL execution boundary.

## Bounded execution examples

| Example | Execution and certification boundary |
| --- | --- |
| [RTL second-token feedback](results/ACE3_RTL_TOKEN_FEEDBACK_20260906.md) / [Compact summary](results/ACE3_RTL_TOKEN_FEEDBACK_20260906.json) | Actual local RTL selection of token `358`: `Hello world! I`; retained decoder work and a fresh norm/head tail, with independent whole-trajectory certification still pending |

This records completed bounded execution, not an accepted continuous-generation
seed, full independent numerical certification, or deployment of a continuous
RTL driver.

## Evidence ladder

ACE-3 uses the following ordering:

```text
contract
  → independent oracle
  → authenticated vectors
  → primitive RTL simulation
  → integrated RTL simulation
  → full model/runtime acceptance
  → synthesis and timing
  → FPGA deployment
  → measured hardware
```

Passing one step never implies a later step. The current project boundary is
summarized in [STATUS.md](STATUS.md).

## Contribution and governance

- [Contributing](../CONTRIBUTING.md)
- [Apache-2.0 license](../LICENSE)
- [Pull request template](../.github/PULL_REQUEST_TEMPLATE.md)

Generated model assets, build products, simulator traces, and local agent state
must remain outside source control.
