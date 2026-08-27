# ACE-3 MP: ARGUS Mixed-Precision Engine

[简体中文](README.zh-CN.md) · [Documentation](docs/INDEX.md) ·
[Current status](docs/STATUS.md) · [Getting started](docs/GETTING_STARTED.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md)

ACE-3 MP is a standalone, evidence-first RTL project for mixed-precision
transformer inference. The first implementation profile targets native AWQ
W4A16 execution for the official
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
checkpoint: packed asymmetric INT4 weights, group size 128, and FP16
activations, residuals, and K/V state.

The repository follows one rule: **a result is only as strong as its execution
boundary**. Software-oracle, RTL-simulation, synthesis, FPGA, and
measured-hardware results are reported separately.

ACE-3 is developed with **[Argus](https://github.com/lbx154/Argus)**, the
open-source long-running agent harness used to plan, execute, review, and
preserve evidence across this engineering campaign.

> **Research preview:** the repository contains verified RTL building blocks,
> a controller-driven 24-layer RTL cascade, and the infrastructure for
> persistent-state Hybrid RTL generation. It does not yet publish an accepted
> readable RTL dialogue, synthesis result, PPA result, FPGA bitstream, timing
> closure, or measured hardware performance.

## Project contract

| Item | Target |
| --- | --- |
| Model | Official Qwen2.5-0.5B-Instruct-AWQ, batch 1 |
| Initial precision | Native asymmetric AWQ W4A16, G128 |
| Decoder shape | 24 layers, hidden size 896, intermediate size 4,864 |
| Execution rule | Every represented token traverses indexed RTL layers 0–23 |
| Host boundary | Tokenizer, embeddings, final RMSNorm, tied `lm_head`, greedy selection, decode |
| Verification | Independent oracle, authenticated inputs, Icarus and Verilator |
| Delivery level | Reproducible RTL simulation before synthesis/PPA/FPGA claims |

## Current status

| Layer | Published boundary | Status |
| --- | --- | --- |
| Native AWQ arithmetic | G128 W4A16 dot lane, exact packing and FP16 rounding | Accepted RTL simulation |
| Projection | 896/128/4864 geometries and complete 896-input official `q_proj` reductions | Accepted RTL simulation |
| FP16 adaptation | Residual, RMSNorm, SiLU/gate, RoPE, and FP16 K/V state | Accepted bounded RTL simulation |
| Attention | Scaled QK, causal softmax approximation, and cached-value composition | Accepted bounded RTL simulation |
| Decoder | Indexed decoder execution with independent reference comparison | Accepted bounded RTL simulation |
| Model24 | Arithmetic-free 24-layer controller and layer-indexed Verilator cascade | Accepted bounded RTL simulation |
| Host decision | Final RMSNorm and tied-head top-10/argmax over the accepted fixture | Accepted host/oracle boundary |
| First Voice | Savable RTL state, authenticated lineage, trusted tips, and compact indexed-layer builder | Infrastructure accepted; all-layer runtime evidence and full traversal in progress |
| Implementation | Synthesis, timing, PPA, FPGA, and measured performance | Not claimed |

See [Current status](docs/STATUS.md) for exact claim boundaries and the active
milestone.

## Execution model

ACE-3 separates the host boundary from the RTL decoder boundary:

```text
Host
  chat template → tokenizer → embedding lookup
                         │
                         ▼
RTL
  layer 0 → layer 1 → ... → layer 23
     │          │                  │
     └── persistent authenticated FP16 K/V state ──┘
                         │
                         ▼
Host
  final RMSNorm → tied lm_head → greedy token → feedback
```

In the First Voice profile, every prompt token and every fed-back generated
token must traverse all 24 indexed RTL decoder layers. The host is limited to
serialization, tokenization, embeddings, final normalization, tied-head
selection, decoding, and feedback. A software-only hidden-state path cannot be
reported as RTL dialogue.

## Repository map

```text
ace3/
  contracts/   Machine-readable arithmetic, interface, lineage, and evidence contracts
  model/       Independent bit-level oracles, vector tools, and host/runtime drivers
  rtl/         Synthesizable SystemVerilog modules
  tb/          Icarus and Verilator testbenches
design/        RTL manifest and requirement-to-evidence traceability
docs/
  results/     Reviewed, scope-bounded result notes
  INDEX.md     Documentation map
  STATUS.md    Current accepted and active boundaries
  ROADMAP.md   Ordered development plan
```

Generated vectors, simulator objects, traces, model weights, and local agent
state are intentionally excluded from source control.

## Reproducible entry points

Requirements:

- Python 3.10 or newer;
- GNU Make;
- Icarus Verilog;
- Verilator and a C++ compiler.

List supported entry points:

```sh
make help
```

Run the standalone arithmetic oracle:

```sh
make oracle
```

Run the source-controlled AWQ fixture regression:

```sh
make test
```

Run the reviewed Model24 controller/publication checks. This target validates
the published controller and source/unit evidence; it does not rerun the sealed
full-24 numerical cascade:

```sh
make model24-publication-tests
```

Rerun the checkpoint-bound full-24 RTL cascade only after preparing the
official assets described in [Getting started](docs/GETTING_STARTED.md):

```sh
make model24-controller-rtl-cascade
```

Run focused First Voice state-lineage and compact-builder checks:

```sh
make model24-first-voice-hybrid-tests
make model24-first-voice-compact-builder-tests
```

Model-bound full execution additionally requires the official checkpoint and
tokenizer. Those assets are not redistributed by this repository. See
[Getting started](docs/GETTING_STARTED.md) for paths, overrides, and
target-specific prerequisites.

## Verification model

1. **Contract:** packing, widths, rounding, reset, stream behavior, and claim
   scope are machine-readable.
2. **Independent oracle:** Python reference models do not derive expected
   results from DUT implementation logic.
3. **Authenticated inputs:** checkpoint revision, tensor payloads, serialized
   vectors, binaries, and state transitions are SHA-256 bound.
4. **Independent simulators:** Icarus provides bounded four-state checks;
   Verilator provides full two-state numerical execution where documented.
5. **Fail-closed lineage:** persistent RTL state is restored only against a
   caller-held trusted commitment, not a self-authenticating mutable envelope.
6. **Explicit non-claims:** simulation cycles are not hardware latency, and
   software execution is not RTL, FPGA, or silicon evidence.

The fixed model revision is
`db09cd27ead7fee40cdee309693cf83601b9c899`.

## Precision roadmap

ACE-3 is developed as one standalone architecture line:

1. native AWQ W4A16;
2. W8A16;
3. BF16/FP16;
4. larger 1.5B and 3B model tiers;
5. reproducible synthesis, PPA, and FPGA evidence.

Modes are added only after a real datapath and verification contract exist.
There are no placeholder precision claims.

## Documentation

Start with [Documentation index](docs/INDEX.md). The most useful pages are:

- [Current status and claim matrix](docs/STATUS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Getting started](docs/GETTING_STARTED.md)
- [First Voice Hybrid RTL](docs/FIRST_VOICE_HYBRID_RTL.md)
- [RTL traceability](design/RTL_TRACEABILITY.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Argus

Argus provides the long-horizon engineering loop behind ACE-3: backlog and
budget supervision, reusable skill matching, engineer execution, independent
review, checkpoints, and evidence-aware replanning.

- Source: <https://github.com/lbx154/Argus>
- ACE-3 remains the standalone hardware project; Argus is the general agent
  harness used to develop and supervise it.

## License

ACE-3 source is licensed under the [Apache License 2.0](LICENSE). Qwen model
assets remain subject to their upstream license and are not included in this
repository.
