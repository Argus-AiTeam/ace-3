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
