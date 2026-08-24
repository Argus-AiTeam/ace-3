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

## Isolation from ACE-2

ACE-2 source paths are not part of this repository. ACE-3 may reproduce an
interface only after documenting the dependency and verifying both sides.
ACE-2 W4A8 and ACE-3 AWQ W4A16 remain separately selectable profiles; neither
is silently converted into the other.

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
