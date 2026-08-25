# ACE-3 MP

[简体中文](README.zh-CN.md) · [Getting started](docs/GETTING_STARTED.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) ·
[Contributing](CONTRIBUTING.md)

ACE-3 MP is an open research RTL project for mixed-precision transformer
inference. The first profile targets native AWQ W4A16 execution for
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ):
packed INT4 weights, group size 128, and FP16 activations.

The repository is organized around one rule: **every result must name its
execution boundary**. Software-oracle, RTL-simulation, synthesis, FPGA, and
measured-hardware results are reported separately.

> **Research preview:** the published source contains verified arithmetic and
> decoder building blocks, not a complete accelerator. There is currently no
> synthesis, timing, PPA, FPGA bitstream, or measured-hardware claim.

## What is available

| Area | Published boundary |
| --- | --- |
| Native AWQ G128 dot lane | Synthesizable RTL; independent bit oracle; Icarus and Verilator simulation |
| Full-input AWQ projection | Parameterized 896/128/4864 geometries; complete 896-input official `q_proj` reductions; bounded simulation |
| FP16 adaptation | Residual, RMSNorm, and SiLU/gate operators; bounded simulation |
| QKV path | Q/K/V projection geometry, Qwen2 RoPE, and indexed FP16 K/V cache; bounded simulation |
| Attention | Scaled QK score, causal softmax approximation, and cached-FP16 value composition; bounded simulation |
| Model24 software schedule | Deterministic reduced-geometry 24-layer software/oracle execution |

Not yet published as accepted hardware evidence:

- an integrated decoder-layer RTL result;
- all 24 decoder layers and the tied language-model head in RTL;
- RTL-backed readable multi-token dialogue;
- synthesis, timing closure, PPA, FPGA deployment, or hardware performance.

## Repository map

```text
ace3/
  contracts/   Machine-readable arithmetic, interface, and evidence contracts
  model/       Independent bit-level oracles and deterministic vector tools
  rtl/         Synthesizable SystemVerilog modules
  tb/          Icarus and Verilator testbenches
design/        RTL manifest and requirement-to-evidence traceability
docs/
  results/     Reviewed, scope-bounded result notes
  ARCHITECTURE.md
  GETTING_STARTED.md
  ROADMAP.md
```

Generated vectors, logs, simulator objects, model files, and local state belong
under ignored directories and are not source artifacts.

## Quick start

Requirements:

- Python 3.10 or newer;
- GNU Make;
- Icarus Verilog;
- Verilator and a C++ compiler.

List the supported entry points:

```sh
make help
```

Run the self-contained Model24 reduced-geometry software/oracle smoke test. It
does not download or require model weights:

```sh
make model24-smoke
```

Run the standalone arithmetic oracle:

```sh
make oracle
```

The complete RTL regression uses small, hash-authenticated samples extracted
from the official checkpoint. Model files are intentionally not redistributed.
Place the required files in `official_tensors/`, or point
`OFFICIAL_TENSOR_DIR` at your read-only fixture directory:

```sh
make OFFICIAL_TENSOR_DIR="$PWD/official_tensors" test
```

The command regenerates vectors, validates every serialized input, runs tamper
rejection, recompiles Icarus and Verilator, executes both simulators, and checks
that the source tree remains unchanged. See
[Getting started](docs/GETTING_STARTED.md) for the expected fixture names and
target-specific commands.

## Implemented projection geometry

`ace3_awq_w4a16_projection_engine` consumes all AWQ groups for each selected
output channel and rounds only after the complete reduction.

| Qwen2.5 projection | Input features | Output features | AWQ groups |
| --- | ---: | ---: | ---: |
| Q / O | 896 | 896 | 7 |
| K / V | 896 | 128 | 7 |
| Gate / Up | 896 | 4864 | 7 |
| Down | 4864 | 896 | 38 |

The published official-tensor numerical evidence covers eight layer-0
`q_proj` outputs over all 896 inputs. Other geometries have elaboration and
bounded interface coverage; they are not presented as full official-tensor
matches.

The intentionally sequential reference engine takes 910 simulated cycles for
one 896-input output and 4,940 simulated cycles for one synthetic-zero
4,864-input output, plus one output-acceptance cycle. These are RTL simulation
cycle counts, not synthesized frequency, latency, or throughput.

## Verification approach

- Arithmetic contracts define packing, widths, rounding, reset, and stream
  behavior.
- Python oracles are independent executable specifications, not generated from
  the RTL.
- Serialized vectors are bound by SHA-256 and validated before simulation.
- Icarus provides four-state X/Z checks; Verilator provides an independent
  two-state run.
- Software fallback is never counted as RTL or hardware completion.

The model-bound artifacts identify the official checkpoint revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. Users are responsible for
obtaining model assets under the upstream model license and terms.

## License

ACE-3 source is licensed under the [Apache License 2.0](LICENSE). The Qwen model
and any extracted checkpoint samples are separate upstream assets and are not
included in this repository.
