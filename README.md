# ACE-3 MP

ACE-3 MP is an evidence-first mixed-precision accelerator project for
transformer inference. Its first implementation profile targets the official
Qwen2.5-0.5B-Instruct AWQ checkpoint with W4A16 execution.

> **Private pre-release:** this repository does not yet claim a complete
> accelerator, synthesized design, FPGA bitstream, measured hardware
> performance, or fabricated chip.

## Why ACE-3

ACE-2 explores a strict integer path based on signed INT4 weights, INT8
activations, and Scale32 metadata. ACE-3 is a separate architecture line that
adds mixed-precision execution while preserving ACE-2 as an independent,
unchanged baseline.

The initial AWQ software qualification established:

- the official AWQ tensor contract: G128, packed INT4 `qweight` and `qzeros`,
  FP16 scales, and native GEMM ordering;
- 168 reconstructed quantized transformer Linear modules;
- viable CPU-reference dialogue, instruction following, multi-turn memory,
  translation, summarization, safety refusal, and simple code generation;
- remaining model weaknesses in exact JSON formatting, one factual explanation,
  and a longer algebra problem.

These are software-reference findings, not RTL or hardware evidence.

## Initial profiles

| Profile | Intent | Status |
| --- | --- | --- |
| `AWQ_W4A16` | Native AWQ G128 weights with FP16 activations | Full-input sequential projection RTL verified |
| `ACE_W4A8` | Compatibility with the existing strict integer line | Planned |

The implemented RTL boundary now includes the accepted G128 primitive and a
parameterized sequential engine that composes every input group for a tiled
set of output channels.

## Repository layout

```text
ace3/
  rtl/         Synthesizable ACE-3 RTL
  tb/          RTL testbenches
  model/       Bit-level software oracles and vector generation
  contracts/   Implemented precision and interface contracts
docs/          Architecture and roadmap
```

Generated logs, model weights, build outputs, and local evidence bundles are not
source files and must not be committed by default.

## Standalone validation

The accepted G128 dot lane and full-input projection engine share a
repository-root validation entry point using only Python, GNU Make, Icarus
Verilog, and Verilator:

```sh
make clean
make OFFICIAL_TENSOR_DIR=/home/argustest/ace-2/build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official test
```

`OFFICIAL_TENSOR_DIR` is explicit and configurable; the default is the path
shown above. The generator reads the three official layer-0 `q_proj` sample
files in place, verifies their frozen SHA256 values, and never writes to that
directory. Vectors, simulator objects, binaries, and logs are created only
under ignored `build/`. `build/logs/` records each command and result.

Every `make test` invocation deletes and regenerates `build/vectors/`, reruns
the oracle and JSON validation, recompiles and reruns both Icarus tests, and
rebuilds and reruns Verilator. It separately regenerates, authenticates, and
simulates full-input projection vectors before printing the aggregate pass
line. No semantic check is stamp-cached.

The historical frozen manifest remains byte-identical. A separate
source-controlled standalone binding contract authenticates SHA256, byte
count, and line count for all five serialized artifacts consumed by the
validator or simulators: `manifest.json`, `meta.hex`, `pairs.hex`, `cases.txt`,
and `vector_params.svh`. Validation occurs before simulation, and `make test`
also confirms that a tampered copy of `meta.hex` is rejected.

The target checks the integer-only oracle, deterministic seed `0xACE3CF01`, 30
cases and 3,840 G128 pairs, exact accumulator values, zero-ULP binary16
results, protocol invariants, Icarus four-state X/Z probes, and an independent
Verilator run. Verilator is a two-state simulator in this configuration, so
X/Z claims come only from the bounded Icarus test. These are dynamic
simulation checks, not formal verification.

## Full-input projection boundary

`ace3_awq_w4a16_projection_engine` is parameterized by `IN_FEATURES` and
`OUT_FEATURES`. It sequences a contiguous output tile, consumes one metadata
record and 128 activation/qweight pairs per AWQ group, sign-extends each exact
96-bit Q47.48 group accumulator into a 102-bit Q53.48 cross-group accumulator,
and rounds once after all groups. It never adds the primitive's already-rounded
FP16 group outputs.

| Modules | Input features | Output features | Groups |
| --- | ---: | ---: | ---: |
| q/o projections | 896 | 896 | 7 |
| k/v projections | 896 | 128 | 7 |
| gate/up projections | 896 | 4864 | 7 |
| down projection | 4864 | 896 | 38 |

Official-tensor numerical evidence uses authenticated layer-0 `q_proj`
qweight, qzeros, and scales from
`Qwen/Qwen2.5-0.5B-Instruct-AWQ@db09cd27ead7fee40cdee309693cf83601b9c899`
with deterministic generated FP16 activations. It covers channels 4 through 11
over all 896 inputs. Directed outputs cover cross-group round-once
cancellation, saturation, subnormal, zero, and invalid operands. Other
geometries are elaborated and linted with both simulators; they are not claimed
as official-tensor numerical matches.

Measured RTL-simulation latency for this intentionally sequential single-lane
engine is 910 cycles from accepted start or previous output acceptance to
`out_valid` for 896 inputs, and 4,940 cycles for a 4,864-input synthetic-zero
output. Output acceptance takes one additional cycle. These are simulation
cycle counts, not synthesis, timing, or performance measurements.

## Evidence policy

Every published claim must identify its execution boundary:

- software reference;
- RTL simulation;
- synthesis and timing;
- FPGA deployment;
- or measured hardware.

Software fallback is never reported as RTL or hardware completion. Unsupported
precision modes remain absent rather than being represented by placeholders.

## Relationship to ACE-2

ACE-2 remains a separate project and continues its strict W4A8 productization
path. ACE-3 does not move, rename, or overwrite ACE-2 RTL. Reuse must happen
through explicit, reviewed interfaces and independently reproducible evidence.

See [Architecture](docs/ARCHITECTURE.md),
[Roadmap](docs/ROADMAP.md),
[projection result](docs/results/AWQ_W4A16_PROJECTION_CF02.md), and
[Contributing](CONTRIBUTING.md).
