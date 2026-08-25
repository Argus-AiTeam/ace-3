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

`model24_execution_oracle.py` executes the complete 483-event, 24-layer
software/oracle schedule with deterministic reduced geometry. It uses native
asymmetric packed-INT4 AWQ G128 projections, FP16 activations and layer-owned KV
state, residual handoffs, final RMSNorm, and a grouped lm_head tied to the
embedding values. Generate and independently validate canonical evidence with:

```sh
python3 ace3/model/generate_model24_execution_vectors.py --output-dir build/model24_execution/vectors
python3 ace3/model/validate_model24_execution_vectors.py --vector-dir build/model24_execution/vectors
```

The evidence is bounded to reduced-geometry software/oracle execution. It does
not establish official-checkpoint logits or dialogue, decoder RTL acceptance,
accelerator latency, synthesis, PPA, or FPGA behavior.

## Official Model24 continuation suite

`official_model24_systematic_continuations.py` runs the authenticated official
checkpoint and tokenizer through a deterministic 32-case English and Chinese
suite. Every row preserves the serialized prompt, raw decoded output, token IDs,
stop reason, per-step Primary/PyTorch comparisons, FP16 KV lineage, failures,
and source/checkpoint/tokenizer hashes.

The checkpoint and tokenizer are not distributed by this repository. Supply the
authenticated `Qwen/Qwen2.5-0.5B-Instruct-AWQ` assets when regenerating:

```sh
make OFFICIAL_MODEL24_CHECKPOINT=/path/to/model.safetensors \
  OFFICIAL_MODEL24_TOKENIZER_DIR=/path/to/tokenizer \
  official-model24-systematic-continuations
```

The Python CLIs accept `--official-checkpoint` and
`--official-tokenizer-dir`; their portable defaults can also be configured with
`ACE3_OFFICIAL_MODEL24_CHECKPOINT` and
`ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR`. Without configuration, the tokenizer
defaults to the repository-relative `model24_execution_vectors/tokenizer/` and
fails with an explicit configuration error when that directory is absent.

The independently reviewed result set is tracked under
[`results/model24-systematic-continuations/`](../../results/model24-systematic-continuations/).
It is bounded software/oracle evidence, not RTL, synthesis, PPA, FPGA, latency,
throughput, or broad model-quality evidence.
