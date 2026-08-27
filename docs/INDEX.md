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
| [Model24 systematic continuations](../results/model24-systematic-continuations/) | Independently reviewed software/oracle continuation evidence |

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
