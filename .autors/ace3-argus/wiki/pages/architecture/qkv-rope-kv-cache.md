---
title: QKV, RoPE, and K/V cache RTL boundary
description: Fixed-checkpoint projection geometry, rotary arithmetic, cache indexing, and claim limits.
---

# QKV, RoPE, and K/V cache RTL boundary

The ACE-3 QKV candidate composes native-AWQ projection engines for
896x896 Q and 896x128 K/V. Q selects `SINGLE_ROUND_BIAS=1`: exact FP16
activation-scale products times native asymmetric INT4 deltas, exact group
and cross-group sums, and exact FP16 bias addition precede one FP16 RNE.
K/V and the shared engine default retain legacy dot-round-then-bias behavior.
The accumulator output remains the pre-bias dot in both modes. Native mode
preserves signed underflow zero and reports overflow as signed infinity with
the overflow flag; nonfinite inputs retain the invalid/zero-output protocol.
No change to packed nibble order, qzero interpretation, or stream ports is made.

The authenticated Qwen2.5 geometry is 14 query
heads, two K/V heads, 64 elements per head, theta 1,000,000, and a 32,768-token
position field.

RoPE uses Qwen's half split: pair `x[p]` with `x[p+32]`. Binary16
multiplication rounds before binary16 addition. Cosine and sine arrive through
an indexed coefficient interface; no transcendental ROM or generator is
claimed.

The parameterized cache stores rotated FP16 K and unrotated FP16 V by cache
slot, position, K/V head, and dimension. Reset and clear invalidate entries
without claiming that SRAM data bits are zeroed. Reads are retained under
backpressure, misses return positive zero with `hit=0`, and same-cycle
read/write to one address is write-through.

This is an RTL-simulation boundary. It does not include attention scores,
softmax, value composition, decoder execution, synthesis/PPA, FPGA evidence,
or dialogue. The Q arithmetic change requires independent review before
actual-output-fed full-model validation and any re-prefill. Previous compiled
binaries, K/V state and software-injected diagnostic trajectories do not
constitute corrected-Q RTL evidence.
