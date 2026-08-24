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
| `AWQ_W4A16` | Native AWQ G128 weights with FP16 activations | In development |
| `ACE_W4A8` | Compatibility with the existing strict integer line | Planned |

The first RTL milestone is a reusable AWQ W4A16 projection primitive with an
independent bit-level oracle and open-source simulation evidence.

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
[Roadmap](docs/ROADMAP.md), and [Contributing](CONTRIBUTING.md).
