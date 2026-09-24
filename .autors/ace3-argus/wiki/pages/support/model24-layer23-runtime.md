---
title: Model24 layer-23 W4A16 runtime boundary
description: Scope of the computer-local layer-23 RTL simulation accepted from the sealed layer-22 continuation.
---

# Model24 layer-23 W4A16 runtime boundary

ACE-3 has a computer-local layer-23 W4A16 RTL simulation pass for two token
positions. The run consumes only the sealed layer-22 FP16 residual handoff,
uses the official layer-23 AWQ tensors, reaches a natural simulator terminal,
matches the independently generated integer trace and final output bit for
bit, and has zero failures against the independent FP16-interstage policy
comparison.

The accepted evidence is sealed under
`build/model24_layer23_attempt001/`. Its predecessor-preservation comparison
also passes.

This establishes the isolated layer-23 continuation boundary. It does not by
itself establish one-process execution of all 24 layers, final RMSNorm,
tied-lm-head/top-k integration, tokenizer/host integration, persistent
multi-token K/V behavior, readable dialogue, synthesis, PPA, FPGA, or silicon.
