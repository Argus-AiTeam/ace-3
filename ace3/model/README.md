# Reference models

`final_rmsnorm_model.py` authenticates the official checkpoint and emits four
integer-exact 896-element vector cases. `final_rmsnorm_compare.py` gates raw
simulator evidence before opening the generated oracle artifacts.

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

`decoder_layer0_oracle.py` composes those independent integer oracles with the
native-AWQ projection oracle for two sequential tokens. The generated trace
retains stage and tensor indices, including Qwen half-split RoPE handoffs, and
the validator reruns the composition before either RTL simulator executes.
The integrated RTL milestone compares all 46,676 trace rows and 1,792 final
hidden rows under Verilator after authenticating every generated artifact.
Icarus remains the four-state authority for the focused width, reset, clear,
fault, preload, SiLU-streaming, and qzeros-address boundaries; a full two-token
Icarus trace is not claimed because a fault-free bounded run reached its
5,400-second limit after 7,000,000 controller cycles.

`model24_execution_oracle.py` also provides the fixed tokenizer/host profile for
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. It hash-authenticates
`tokenizer.json` and `tokenizer_config.json`, serializes and round-trips one
fixed system/user chat prompt, and steps every prompt and generated token
through the published 483-event structural `ExecutionMachine`. The host emits
the reduced fixture sequence `Hello world` followed by `<|im_end|>`, records
cache-slot/position reuse and stop handling, and rejects IDs outside its
explicit reduced execution vocabulary. The reduced logits are deterministic
test fixtures.

The Model24 Make targets expect the pinned checkpoint at
`model24_execution_vectors/model.safetensors` and its authenticated tokenizer
files under `model24_execution_vectors/tokenizer/`. These paths are
repository-relative and the assets are not committed. Override them with
`OFFICIAL_MODEL24_CHECKPOINT` and `OFFICIAL_MODEL24_TOKENIZER_DIR` when the
official assets are stored elsewhere. Direct Python invocations may use
`ACE3_OFFICIAL_MODEL24_CHECKPOINT` and
`ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR`, or pass the corresponding CLI options.

The same oracle separately authenticates the complete fixed-revision
`model.safetensors` object and the `model.norm.weight`, `lm_head.weight`, and
`model.embed_tokens.weight` payloads. It verifies the distinct tied-weight
ranges byte-for-byte, applies the integer-only final RMSNorm to a fixed FP16
structural terminal-state fixture, and computes all 151,936 tied-head logits
with exact Q47.48 accumulation and binary16 round-to-nearest-even. The host
records stable top-10/argmax selection and decodes the selected token with the
authenticated tokenizer. This establishes the official final-projection
token-decision slice only: the fixture is not an official numerical layer-23
output, and full-model numerical execution and readable dialogue remain
unclaimed.

`controller_model24_cascade.py` authenticates the controller launch transcript
and all 624 per-layer tensor bindings before applying the accepted decoder
arithmetic in controller order. `controller_model24_rtl_cascade.py` applies the
same gates to separately compiled, layer-indexed Verilator decoder instances.
Layers 0 through 2 preserve the reviewed rational SiLU profile; layers 3 through
23 select the range-reduced exponential profile in both the oracle and RTL.
`layer3_token0_diagnostic.py` binds the same layer-2 handoff and classifies the
single Token 0 final outlier at dimension 62 as a bounded FP16 reference
boundary while independently checking Token 1 and K/V causality.

`official_single_decoder_layer.py` authenticates all 26 consumed layer-0 tensors
and two official embedding rows directly from the pinned checkpoint. It executes
the two-token `Hello world` slice through native-AWQ projections, accepted
bit-level FP16 adaptation/attention operators, and an FP16 K/V cache. Generated
evidence records official-shape intermediate hashes, 42 sampled exact projection
bit-oracle checks, bounded comparisons to an independent PyTorch Qwen2 path, and
an FP16 post-layer handoff. Layers 1 through 23 and a valid terminal hidden state
remain explicitly outside this boundary.

`official_model24_dialogue.py` runs the authenticated fixed chat prompt through
all 24 layers and the tied head in a deterministic greedy loop. It extends the
FP16 K/V cache at each generated position, records per-step hidden/logit hashes
and cache parentage, and compares every selected token with a PyTorch CPU
float64 dequantized-AWQ reference. Acceptance authenticates and preserves the
actual official-tokenizer output instead of requiring a canned lexical result.

`official_model24_showcase.py` reuses that authenticated 24-layer executor for
six raw-continuation and official-chat-template prompts in English, Chinese,
and Python. The adjacent JSON evidence preserves every prompt and raw output,
all generated token IDs, stop reasons, full-vocabulary tied-head decisions,
per-token Primary/PyTorch errors, incremental FP16 K/V lineage, and failures.
`SHOWCASE.md` is a non-authoritative readable projection of those complete rows.

`official_model24_systematic_continuations.py` extends the reviewed
`showcasecontinuations15c` baseline to a fixed, checked-in 32-case English and
Chinese first batch spanning continuation, chat, factual, commonsense, code,
and short reasoning prompts. The generator always executes the complete ordered
suite and emits authenticated JSONL, JSON summary, Markdown report, and run log
artifacts under `build/official_model24_systematic_continuations/`. Every step
states ACE-versus-PyTorch agreement or mismatch and retains FP16 K/V parentage.
Diagnostic host wall time is preserved but is not product latency or throughput
evidence. Unreviewed ancestry `486e5d848245` is excluded from acceptance and
claim-bearing evidence.

The independently reviewed bounded software/oracle result set is tracked under
[`results/model24-systematic-continuations/`](../../results/model24-systematic-continuations/).
It is not RTL, synthesis, PPA, FPGA, latency, throughput, or broad
dialogue-quality evidence.
