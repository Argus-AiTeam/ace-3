# ACE-3 MP Roadmap

## Phase 0: software qualification

- [x] Recover the official AWQ G128 tensor contract.
- [x] Validate packed-weight reconstruction with independent implementations.
- [x] Run a CPU FP16 reference over the reconstructed official checkpoint.
- [x] Establish basic dialogue viability.
- [ ] Run native AutoAWQ or vLLM inference when a compatible GPU is available.

## Phase 1: W4A16 projection

- [ ] Freeze the implemented arithmetic and interface contract.
- [ ] Implement a synthesizable unpack/dequant/product or dot-lane primitive.
- [ ] Verify reset and backpressure.
- [ ] Pass deterministic edge vectors.
- [ ] Pass seeded random vectors against an independent bit oracle.
- [ ] Bind vectors to an official Qwen projection tensor slice.

## Phase 2: decoder arithmetic

- [ ] FP16 residual path.
- [ ] FP16 RMSNorm with declared reduction precision.
- [ ] FP16 RoPE.
- [ ] Mixed-precision attention score and composition.
- [ ] Softmax with explicit internal precision.
- [ ] FP16 KV-cache prefill and incremental decode.
- [ ] FP16 SiLU and gated MLP.
- [ ] Tied FP16 embedding and language-model head.

## Phase 3: end-to-end execution

- [ ] Integrate all 24 Qwen2.5-0.5B decoder layers.
- [ ] Match software-reference logits and greedy tokens.
- [ ] Generate readable multi-token dialogue.
- [ ] Add a one-command host/runtime path.
- [ ] Record honest software and RTL latency separately.

## Phase 4: implementation evidence

- [ ] Synthesize with a reproducible toolchain.
- [ ] Close timing for a declared target.
- [ ] Build and deploy an FPGA image when hardware is available.
- [ ] Measure throughput, latency, memory traffic, resource use, and power.

## Release

ACE-3 remains private until the repository contains reproducible source,
contracts, tests, and evidence appropriate to the claims being published.
