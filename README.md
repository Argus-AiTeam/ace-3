# ACE-3 MP: ARGUS Mixed-Precision Engine

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

The design covers native AWQ unpacking and dequantization, complete projection
reductions, FP16 normalization and nonlinear operators, RoPE, causal K/V state,
attention and value composition, decoder-layer integration, indexed 24-layer
execution, authenticated persistent simulator state, and the host boundary
needed for readable autoregressive generation.

## Current status

ACE-3 is active research, not a synthesized implementation, FPGA deployment,
fabricated chip, or measured-performance result. The repository contains
independently reviewed RTL and evidence through a controller-driven 24-layer
decoder cascade. A genuine Hybrid RTL dialogue traversal is currently running,
but it is not an accepted readable-dialogue result until the complete
transaction chain and generated output are independently reviewed.

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

The current Hybrid RTL boundary keeps chat serialization, tokenization,
embedding lookup, final RMSNorm, tied `lm_head`, greedy selection, decoding,
and token feedback on the host. Every represented prompt or generated token
must traverse indexed RTL decoder layers 0–23 with authenticated persistent
per-layer K/V state. A software-only hidden-state path is not completion.

## Repository map

| Path | Contents |
| --- | --- |
| `ace3/contracts/` | Machine-readable arithmetic, interface, lineage, and evidence contracts |
| `ace3/model/` | Independent bit-level oracles, vector tools, and host/runtime drivers |
| `ace3/rtl/` | Synthesizable SystemVerilog implementation |
| `ace3/tb/` | Icarus and Verilator testbenches |
| `ace3/fixtures/` | Small source-controlled model fixtures with provenance |
| `design/` | RTL manifest and requirement-to-evidence traceability |
| `docs/results/` | Reviewed, scope-bounded result notes |

Generated vectors, simulator objects, traces, model weights, and local agent
state are intentionally excluded from source control.

Start with [the documentation map](docs/INDEX.md),
[current status](docs/STATUS.md), [architecture](docs/ARCHITECTURE.md), and
[getting started](docs/GETTING_STARTED.md). The status page is authoritative
for accepted, active, and explicitly unclaimed results.

## Reproducible entry points

```sh
# List supported validation and Model24 entry points.
make help

# Run the standalone arithmetic oracle.
make oracle

# Run the source-controlled AWQ fixture regression.
make test

# Validate the published Model24 controller and source/unit evidence.
# This does not rerun the sealed full-24 numerical cascade.
make model24-publication-tests

# Run focused First Voice state-lineage and compact-builder checks.
make model24-first-voice-hybrid-tests
make model24-first-voice-compact-builder-tests

# Rerun the checkpoint-bound full-24 RTL cascade after preparing the
# official checkpoint and tokenizer described in docs/GETTING_STARTED.md.
make model24-controller-rtl-cascade
```

The basic regressions require Python 3.10 or newer, GNU Make, Icarus Verilog,
Verilator, and a C++ compiler. Model-bound execution additionally requires the
official checkpoint revision
`db09cd27ead7fee40cdee309693cf83601b9c899` and its tokenizer; these assets are
not redistributed by this repository.

The validation flow binds contracts, official tensor payloads, serialized
vectors, simulator binaries, and persistent state transitions with SHA-256.
Icarus provides bounded four-state checks while Verilator provides the
documented full numerical execution. Simulation cycles are not hardware
latency, and software execution is not RTL, FPGA, or silicon evidence.

Some flows require model assets, synthesis tools, or FPGA hardware that are not
bundled with this repository. Missing prerequisites are reported explicitly
rather than represented as successful synthesis, PPA, FPGA, or hardware runs.

For the ordered W4A16, W8A16, BF16/FP16, larger-model, and implementation
milestones, see the [roadmap](docs/ROADMAP.md).

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
