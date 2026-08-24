# Reference models

Bit-level arithmetic oracles and vector generators live here. These models are
verification references, not accelerator implementations or performance paths.

`generate_vectors.py` requires explicit `--official-tensor-dir` and
`--output-dir` arguments. The root `Makefile` binds the official read-only
sample location and places all generated files under `build/vectors/`.
`validate_vectors.py` authenticates every serialized output against the
source-controlled standalone binding contract before any simulator runs.

`projection_oracle.py`, `generate_projection_vectors.py`, and
`validate_projection_vectors.py` extend the same integer-only arithmetic to
complete input reductions while keeping the historical primitive artifacts
unchanged. Generation re-authenticates the fixed model revision, config,
native-GEMM packing source, and sampled qweight/qzero/scale payloads; validation
recomputes every serialized output from the simulator streams.

`attention_oracle.py` independently specifies exact FP16-to-Q24 decoding,
scaled Q48 score and value accumulation, and the frozen Q0.24 softmax
approximation. Its generator re-authenticates the accepted checkpoint samples;
its validator recomputes all serialized stage results and enforces SHA-256
bindings before simulation.
