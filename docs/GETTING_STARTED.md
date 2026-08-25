# Getting started

ACE-3 separates self-contained software/oracle checks from model-bound RTL
regressions. This keeps the public repository useful without redistributing
third-party model weights.

## 1. Install tools

The current flow uses:

- Python 3.10 or newer (standard library only for the published oracle tools);
- GNU Make;
- Icarus Verilog (`iverilog` and `vvp`);
- Verilator;
- a C++ compiler supported by Verilator;
- Git for the repository-hygiene checks.

On Ubuntu or Debian:

```sh
sudo apt-get update
sudo apt-get install -y build-essential git iverilog make python3 verilator
```

Check the available entry points:

```sh
make help
```

## 2. Run checks that need no model files

The arithmetic oracle:

```sh
make oracle
```

The deterministic reduced-geometry, 24-layer Model24 software/oracle schedule:

```sh
make model24-smoke
```

Outputs are written below ignored `build/` directories.

## 3. Prepare official checkpoint samples

The full RTL regression uses small read-only samples derived from
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. The repository does not distribute
the checkpoint or extracted tensors.

Provide a directory containing:

```text
autoawq-v0.2.9-packing_utils.py
config.json
model-api.json
sample-model_layers_0_self_attn_q_proj-qweight.bin
sample-model_layers_0_self_attn_q_proj-qzeros.bin
sample-model_layers_0_self_attn_q_proj-scales.bin
```

The generators contain the accepted SHA-256 for every consumed file and fail
before vector generation if any file differs. Keep this directory read-only;
the build never writes to it.

By default, Make looks in `official_tensors/` at the repository root. Override
the location explicitly when needed:

```sh
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors check-official-tensors
```

## 4. Run targeted RTL regressions

```sh
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors projection
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors fp16-adaptation
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors qkv-rope-cache
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors attention
```

Run the complete published regression:

```sh
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors test
```

The aggregate target:

1. regenerates vectors from authenticated inputs;
2. independently validates serialized vectors and SHA-256 bindings;
3. confirms tampered inputs are rejected;
4. rebuilds and runs Icarus tests;
5. rebuilds and runs Verilator tests;
6. checks that generated output remains ignored and the source tree is
   unchanged.

No semantic test is accepted from a previous stamp or cached build.

## 5. Read results correctly

The result vocabulary is intentionally strict:

- **software/oracle** means an executable reference model;
- **RTL simulation** means a named simulator ran a bounded test;
- **synthesis/timing** requires tool reports for a declared target;
- **FPGA** requires a built image and deployment evidence;
- **measured hardware** requires the physical platform and measurement method.

Simulation cycle counts are not wall-clock latency or synthesized performance.
See [Architecture](ARCHITECTURE.md), [Roadmap](ROADMAP.md), and
[`docs/results/`](results/) for the current evidence boundary.
