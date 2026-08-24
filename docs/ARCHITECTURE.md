# ACE-3 MP Architecture

## Scope

ACE-3 MP is a mixed-precision transformer accelerator architecture. The first
implemented target is AWQ W4A16; broader precision support is added only when a
real datapath and verification contract exist.

## AWQ W4A16 contract

The initial profile consumes:

- packed INT4 `qweight` in INT32 containers;
- packed INT4 `qzeros` in INT32 containers;
- AWQ GEMM channel ordering;
- group size 128;
- no qzero plus-one adjustment;
- FP16 group scales;
- FP16 activations and outputs.

For each logical weight:

```text
dequantized_weight = (iweight - izero) * fp16_scale
```

The complete architecture must eventually cover embedding, RMSNorm, Q/K/V/O
projections, RoPE, attention, softmax, KV cache, gated MLP, residual paths, and
the tied FP16 language-model head.

## First RTL milestone

The first bounded primitive is one G128 logical-output AWQ dot lane:

1. unpack native AWQ nibbles;
2. restore logical channel ordering;
3. subtract the group zero point;
4. combine the signed weight with FP16 scale and FP16 activation;
5. accumulate under a declared bit-level contract;
6. round once to FP16 using round-to-nearest-even.

The current candidate uses an exact signed Q47.48 internal accumulator. It is a
milestone contract, not a claim that the final architecture has been selected.

The synthesizable DUT must not use SystemVerilog `real`, DPI-based arithmetic,
or proprietary floating-point IP as a substitute for the datapath.

### Streaming and four-state hardening boundary

For legal binary-valued traffic, the accepted arithmetic and transaction
interface are unchanged. `start_ready_o` and `pair_ready_o` are suppressed
during reset or synchronous clear so a producer cannot count a transaction
that the DUT is aborting. Output acceptance is likewise blocked unless clear
is known deasserted.

The standalone Icarus test adds bounded four-state probes for X and Z on
`start_valid_i`, `pair_valid_i`, `out_ready_i`, and `clear_i`, including
unknown payload while no legal handshake exists. A property-style monitor
requires accepted payload to be fully known, exactly 128 accepted pairs before
each output, known output/status whenever `out_valid_o` is asserted, and stable
output/status/accumulator while backpressured. Reset, clear, completed, and
aborted transaction counts are checked explicitly.

This is runtime simulation coverage, not formal verification. Clock and reset
must be driven as known binary values. Accepted X/Z payload is a testbench
protocol violation rather than a synthesized hardware error-recovery mode.
Verilator 4.038 is two-state and repeats the legal protocol and arithmetic
checks but cannot substantiate X/Z behavior.

The generated case manifest retains its historical CF01 hash. Standalone
serialization is bound separately by
`ace3/contracts/awq_w4a16_g128_standalone_vector_bindings.json`, which records
SHA256 for every file consumed by Icarus or Verilator. The root test
regenerates and authenticates those files on every invocation before either
simulator runs; a tampered build-local copy is required to fail validation.

## Full-input projection engine

`ace3_awq_w4a16_projection_engine(IN_FEATURES, OUT_FEATURES)` is a tiled,
single-lane engine. One start selects a contiguous output-channel range, and
the engine sequences every output channel and every G128 input group. The
producer supplies metadata and pairs under ready/valid backpressure while the
engine exposes the exact indices being requested.

For output channel `o`, group `g`, and input `i`:

```text
packed output word = floor(o / 8)
logical lane       = o mod 8
qweight word       = qweight[i][floor(o / 8)]
qzero word         = qzeros[g][floor(o / 8)]
scale              = scales[g][o]
```

The accepted native-AWQ physical nibble order remains
`[0, 4, 1, 5, 2, 6, 3, 7]`. Each group is executed by the unchanged accepted
G128 primitive. Only its signed 96-bit Q47.48 accumulator and invalid flag feed
the projection engine; its FP16 group result and group saturation flag are
ignored.

### Cross-group precision

The maximum supported input dimension is 4,864, or 38 G128 groups. The
cross-group accumulator is signed 102-bit Q53.48:

```text
96 + ceil(log2(38)) = 102 bits
```

Every group result is sign-extended and added exactly. This width contains the
sum of any 38 values representable by the signed 96-bit group contract, so
there is no internal wrap or saturation. Non-finite input flags are ORed across
groups. One round-to-nearest-ties-to-even binary16 conversion occurs only at
the final output boundary; only that conversion may saturate.

### Authenticated geometries and measured cycle model

The fixed official model revision
`Qwen/Qwen2.5-0.5B-Instruct-AWQ@db09cd27ead7fee40cdee309693cf83601b9c899`
and config hash
`bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090`
declares hidden size 896, intermediate size 4,864, 14 attention heads, two KV
heads, and AWQ G128. One parameterized module covers q/o (896x896), k/v
(896x128), gate/up (896x4864), and down (4864x896). Both Icarus and Verilator
elaborate or lint all four parameter sets.

With no input stalls, each group takes one metadata cycle, 128 pair cycles, and
one group-result collection cycle. RTL simulation measures 910 compute cycles
for a seven-group 896-input output and 4,940 compute cycles for a 38-group
4,864-input output. `out_valid` is then held until accepted; output acceptance
costs one additional cycle. These numbers describe this sequential RTL only,
not synthesized frequency, throughput, or PPA.

## ACE-2 evolutionary reuse audit

The projection milestone was compared against the ACE-2 Alpha 2 certified
snapshot revision `8edf99e16633e7877f6be357cae4ddc4b9a8fb97`. The audited files
were present and clean at that revision:

- `rtl/ace2_w4a8_proj_core.sv`
- `rtl/ace2_shell.sv`
- `tools/ace2_projection_reference.py`
- `verification/tb/ace2_w4a8_proj_tb.sv`
- `verification/verilator/ace2_shell_oproj_harness.sv`
- `verification/verilator/ace2_shell_oproj_main.cpp`

The reuse decision is explicit:

| Classification | ACE-2 structure | ACE-3 projection treatment |
| --- | --- | --- |
| Reused | Separate start, pair, metadata, and output ready/valid channels; handshake-gated state changes; held output until acceptance | Preserved as the native projection stream protocol and stable-output backpressure rule |
| Adapted | Shell output-index sequencing and variable-ready verification harnesses | Output-major then G128-group-major sequencing with exposed tensor indices; deterministic metadata/pair stalls and output backpressure checks |
| Adapted | Independent fixed-point projection reference | Independent native-AWQ oracle with exact FP16 inputs, G128 accumulation, cross-group summation, and one final FP16 rounding |
| Replaced | Signed symmetric INT4 x INT8 MAC, 32-bit accumulator, integer multiplier/right shift, output zero point, and INT8 saturation | Asymmetric packed AWQ qweight/qzeros, FP16 scales/activations, unchanged 96-bit G128 results, exact 102-bit cross-group accumulation, and final FP16 RNE |
| Deferred | ACE-2 shell memory requests, SRAM scheduling, fused command flow, and writeback | Not copied into this bounded stream-engine milestone; SRAM/DMA and decoder integration remain unsupported |

No ACE-2 arithmetic RTL is copied. Compatible interface and control patterns
are re-expressed for the W4A16 profile, while incompatible W4A8 assumptions are
replaced and independently verified. ACE-2 is therefore a provenance-tracked
architectural baseline, not a build dependency or a numerically equivalent
projection baseline. Cross-profile comparisons may compare protocol invariants
or identically defined cycle boundaries, but must not present differing
precision, tensor format, or memory scope as apples-to-apples PPA data.

## Verification levels

1. **Bit oracle:** independent packed-weight and FP16 arithmetic model.
2. **Primitive RTL:** reset, backpressure, edge vectors, and seeded random
   vectors.
3. **Projection:** official layer tensor slices and complete dot products.
4. **Decoder layer:** mixed-precision residual, normalization, attention, and
   MLP integration.
5. **Model execution:** readable multi-token generation with tokenizer and KV
   cache.
6. **Implementation:** synthesis, timing, FPGA build, deployment, and measured
   performance.

Passing one level does not certify a later level.

The current evidence reaches level 3 for eight complete official layer-0
`q_proj` outputs and directed arithmetic cases. It does not numerically certify
the other projection modules or any decoder-layer behavior.
