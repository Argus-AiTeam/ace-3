---
title: Streaming tied lm_head and deterministic Top-K
description: Standalone ACE-3 boundary for exact FP16 tied-vocabulary projection without storing the complete logits vector.
---

# Streaming tied lm_head and deterministic Top-K

The standalone boundary accepts one 896-element finite FP16 vector after final RMSNorm, then consumes `model.embed_tokens.weight` in token-major, feature-minor order for the pinned `Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision `db09cd27ead7fee40cdee309693cf83601b9c899`.

The official geometry is 151936 vocabulary rows by 896 features. The checkpoint is bound by SHA-256 `c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b`; both `model.embed_tokens.weight` and `lm_head.weight` must have tied value digest `d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0`.

Each FP16 operand is decoded exactly to signed Q16.24. Products are exact signed Q32.48 values accumulated without intermediate rounding in a signed 96-bit Q47.48 accumulator. Each completed row is rounded once to finite FP16 using round-to-nearest, ties-to-even. NaN and infinity are rejected; overflow saturates to signed maximum finite FP16 and raises saturation.

The boundary retains the decoded hidden vector, one current accumulator, one held logit, and ten winners. It never stores the complete logits vector. Top-K compares rounded finite FP16 logits by descending numeric value and breaks equal-logit ties by ascending token ID. Rank zero is the selected token.

Hidden indices, token IDs, feature indices, last markers, and explicit end markers must be exact and ordered. Early explicit termination, malformed framing, unknown accepted payload bits, nonfinite operands, and index-order violations enter sticky error state until clear or reset. Logit and Top-K payloads remain stable under backpressure.

This boundary does not establish decoder-to-lm_head integration, dialogue generation, synthesis, PPA, FPGA execution, latency, or throughput.

## Accepted repair8 handoff preparation

The bounded repair8 handoff preparation authenticates result SHA-256 `da2c696b9701e86944ddfb22a28bb51f99f64a55bbe6e325aa58da7bc7d420c9` and its layer-23 terminal FP16 row SHA-256 `3e99fe3e1d30e6350d32f3ef45311f4a81114b27c13a5e01ef04f1a1ca0bf429` without modifying the accepted run. The accepted Q26 final-RMSNorm boundary feeds a full 151,936-row exact-integer head oracle and a path-distinct PyTorch CPU float64 head oracle. They have zero rounded-FP16 integer mismatches and identical deterministic Top-K ordering; the predicted selected token is 271 with FP16 logit bits `0x4c0f`.

The preparation also binds all 24 repair8 `position002` state envelopes and simulator states as the position-2 K/V lineage. Focused Icarus four-state checks and a Verilator numerical/tie-break check pass, and an official-shape Verilator binary is built from the accepted commit's sealed source blobs. Official lm_head execution and position-2 traversal authority remain withheld pending independent review; no dialogue, synthesis, PPA, FPGA, latency, throughput, or broad generation-quality claim is made.

## Accepted official v4 execution

The independently accepted v4 package was consumed through its canonical launcher exactly once on 2026-08-28. The immutable review seal SHA-256 is `69c7333312a383f20ca6f1e45fe4d0083ba948249b60853f795cb3fa508717bf`, the package-manifest SHA-256 is `64ad19a80e237e5be2c689ff854b341c60afc8038a18f8818ea9c4dc63fe71e4`, and the sealed `COMPLETE` terminal SHA-256 is `588de02dee141f75b2ef02f81841dcc6bda6426c0e2e2dfe8ef579d2f132e7a1`.

The official Verilator harness accepted all 151936 logits and 136134656 streamed weights, matched 12 selected checks, and returned token 271 with FP16 logit bits `0x4c0f`. The receipt binds independent exact-logit SHA-256 `0cdc0f71e32614abd3a798645db98326fa05d75d3c705d5f9bdb16686b507cb9`, zero integer mismatches, and the accepted deterministic Top-K. The process exited 0 without timeout or signal after 115.000086 seconds of monotonic launcher timing and 136287550 reported simulation cycles.

This evidence is limited to the accepted repair8 position-1 terminal hidden state through the standalone tied `lm_head`. It did not materialize a raw full-logit vector, did not execute position-2 traversal, and does not establish dialogue, synthesis, PPA, FPGA, latency, throughput, or broad generation quality.

## Host selected-token receipt boundary

The host receipt bridge consumes no hardware transaction. It accepts an
already-produced terminal-evidence binding plus the established ten-entry
Top-K shape (`rank`, `token_id`, `logit_f16_bits`, and exact decoded
`logit_q24`). Rank zero must match the separately carried selected token ID and
FP16 logit. All entries must be finite, unique, and ordered by the same
descending-logit/ascending-token-ID policy as the streaming RTL.

Checkpoint, tokenizer, prompt, terminal-evidence, and receipt-use authority
lineages are validated before the selected token is appended to the next host
token history. The receipt-use authority must be explicitly authorized and
unconsumed, but the bridge only reads that state; it neither creates nor
consumes any authority. This boundary does not invoke the streaming
`lm_head`, Model24 controller/model/oracle/RTL, lifecycle, or durable
submission, and does not extend the accepted hardware claims above.

An ordered host chain adds no hardware behavior. Each receipt starts at
generation ordinal zero and advances without a gap, repeats the same
authenticated prompt record, binds the token history produced by prior
rank-zero selections, and binds the prior accepted terminal-evidence record.
The complete caller-supplied terminal-evidence chain and one authorized,
unconsumed receipt-use authority lineage per receipt are checked before the
host returns the assembled token history and tokenizer-decoded transcript.
