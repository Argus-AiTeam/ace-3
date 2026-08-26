---
title: Attention RTL boundary
description: Fixed GQA score, causal softmax, cached-value composition, and claim limits.
---

# Attention RTL boundary

The ACE-3 attention candidate maps each of 14 query heads to one of two K/V
heads with `kv_head = query_head / 7`. Each score accumulates 64 exact products
of FP16 values decoded to Q24, scales the Q48 sum by 1/8, and returns FP16.
Only keys at or before the query position are eligible.

Softmax performs max subtraction in Q24. Its synthesizable exponential uses a
17-point Q0.24 base-2 table, linear interpolation, explicit RNE, and an
underflow threshold. A causally required cache miss or invalid row returns
zeros rather than renormalizing a partial context. Value composition accumulates
FP16 probability/value products and reports required V misses.

The evidence boundary is authenticated Icarus and Verilator dynamic simulation.
The aggregate regenerates and independently recomputes 20 score cases, 20
softmax rows containing 140 probabilities, and 20 value-composition cases. It
requires serialized SHA-256 authentication and tamper rejection before either
simulator runs. Both simulators reject known-invalid score and value GQA starts;
Icarus additionally covers six bounded X/Z non-acceptance probes.
Official-derived operands are deterministic selections from an authenticated
layer-0 q-projection FP16 scale sample, not captured runtime Q/K/V activations.

It does not claim decoder or full-model execution, dialogue, formal proof,
synthesis, timing, PPA, FPGA, or silicon behavior.
