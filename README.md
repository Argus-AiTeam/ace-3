# Argus Compute Engine 3 Mixed-Precision (ACE-3 MP)

ACE-3 MP is an evidence-driven, pre-synthesis accelerator project for
mixed-precision transformer inference. It develops a complete native-AWQ
system boundary for the official
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
checkpoint: packed asymmetric INT4 weights, group size 128, FP16 activations
and residuals, causal K/V state, decoder execution, host integration, and
reproducible verification.

ACE-3 MP is a standalone successor in the ACE hardware line. It does not
depend on an ACE-2 source tree, build directory, fixture path, runtime, or
evidence store. Reused architectural ideas are reimplemented or copied into
ACE-3-owned, provenance-tracked assets.

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

The design covers model-bound tensor loading, native AWQ unpacking and
dequantization, complete projection reductions, FP16 normalization and
nonlinear operators, RoPE, causal K/V cache, attention and value composition,
decoder-layer integration, indexed 24-layer execution, authenticated persistent
simulator state, and the tokenizer/host/generation boundary needed for readable
autoregressive dialogue.

## Current status

ACE-3 is active research, not a synthesized implementation, FPGA deployment,
fabricated chip, or measured-performance result. The repository contains
independently reviewed RTL and evidence from the native G128 W4A16 arithmetic
lane through complete official projection reductions, FP16 residual/RMSNorm/
SiLU/RoPE operators, causal K/V state, attention, one integrated decoder layer,
and indexed execution across all 24 decoder layers.

The accepted full-24 fixture consumes all 624 official decoder tensors. Its
post-layer-23 Token 1 hidden state has maximum absolute error
`0.08988498970425507`, within the published `0.125` bound. Host final RMSNorm
and the tied software `lm_head` reproduce the independent reference Top-10
ordering and select token ID `0` (`!`) for the fixed `Hello world` fixture.
The Token 0/global maximum error remains `2.3170627008770595`; this FP16
boundary behavior is disclosed rather than hidden.

The current First Voice milestone extends that reviewed decoder into an
autoregressive system. Twenty-four compact indexed Verilator binaries have
been built operationally with savable state, authenticated predecessor
lineage, and caller-held trusted commitments. A genuine Hybrid RTL dialogue
traversal is in progress: every prompt token and every fed-back generated token
must pass through RTL layers 0–23 while preserving causal per-layer K/V state.
The host performs only chat serialization, tokenization, embedding lookup,
final RMSNorm, tied-head selection, decode, and feedback.

No readable RTL dialogue has been accepted yet. RTL final RMSNorm, a streaming
tied `lm_head`/Top-K unit, W8A16, BF16/FP16, larger-model tiers, synthesis,
timing closure, PPA, FPGA deployment, and measured hardware performance remain
future milestones.

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
