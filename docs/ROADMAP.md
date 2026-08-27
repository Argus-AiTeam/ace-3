# ACE-3 MP Roadmap

The roadmap is ordered by evidence dependency. A checked item means the
corresponding bounded result is published; it does not imply completion of a
later integration or implementation level.

## 1. Native AWQ W4A16 foundation

- [x] Authenticate the official AWQ G128 tensor contract.
- [x] Implement packed asymmetric INT4 unpacking and zero-point handling.
- [x] Implement the synthesizable G128 W4A16 arithmetic lane.
- [x] Verify reset, clear, ready/valid backpressure, and four-state boundaries.
- [x] Bind deterministic vectors to official checkpoint tensor samples.
- [x] Implement complete cross-group accumulation with one final FP16 rounding.
- [x] Elaborate Q/O, K/V, Gate/Up, and Down projection geometries.
- [x] Numerically verify selected complete 896-input official `q_proj` outputs.
- [ ] Extend official-tensor numerical coverage across every projection family.

## 2. Decoder arithmetic and state

- [x] FP16 residual path.
- [x] FP16 RMSNorm with declared reduction precision.
- [x] FP16 SiLU and gated MLP.
- [x] Q/K/V projection cluster.
- [x] Qwen2 half-split RoPE.
- [x] Indexed FP16 K/V cache with overwrite and isolation checks.
- [x] Scaled QK score accumulation.
- [x] Causal softmax with frozen internal precision.
- [x] Cached-value composition.
- [x] Integrated indexed decoder execution with independent reference checks.

## 3. Model24 execution

- [x] Add an arithmetic-free 24-layer launch/completion controller.
- [x] Compile and execute layer-indexed decoder RTL for layers 0 through 23.
- [x] Authenticate all consumed layer tensors and compiled binaries.
- [x] Compare post-layer-23 hidden state and Host top-K with an independent
  official-checkpoint reference.
- [x] Add persistent Verilator save/restore.
- [x] Add canonical state envelopes and predecessor lineage.
- [x] Require caller-held trusted chain-tip commitments before restore.
- [x] Implement and test the compact authenticated indexed-layer builder.
- [ ] Publish independently reviewed all-24 compact-build manifests and
  reproducible evidence. An operational build exists outside source control.
- [ ] Complete and review the first readable Hybrid RTL dialogue.

## 4. Runtime completion

- [ ] Publish a one-command accepted First Voice execution path.
- [ ] Implement RTL final RMSNorm.
- [ ] Implement streaming tied `lm_head` and Top-K.
- [ ] Add layer-major prompt prefill without changing token-major generation.
- [ ] Add durable checkpoint/resume across host restarts.
- [ ] Report software, RTL simulation, and future hardware latency separately.

## 5. Precision and model scaling

- [x] Establish native AWQ W4A16 as the first complete profile.
- [ ] Add W8A16.
- [ ] Add BF16/FP16.
- [ ] Scale the architecture to a 1.5B model tier.
- [ ] Scale the architecture to a 3B model tier.
- [ ] Define larger tiers only after memory, bandwidth, and implementation
  contracts are evidence-backed.

## 6. Implementation evidence

- [ ] Freeze a reproducible synthesis toolchain and source manifest.
- [ ] Synthesize a declared configuration.
- [ ] Report timing and area without extrapolating unsupported PPA.
- [ ] Close timing for a declared target.
- [ ] Package and deploy an FPGA image when hardware is available.
- [ ] Measure latency, throughput, memory traffic, resources, power, and energy.

## Release policy

- [x] Publish source, contracts, tests, and scope-bounded result notes.
- [x] Keep model assets, generated traces, build products, and local state out
  of source control.
- [x] State software, RTL simulation, synthesis, FPGA, and hardware boundaries
  separately.
- [ ] Promote First Voice dialogue only after reproducible independent review.
- [ ] Promote synthesis/PPA/FPGA claims only with the corresponding artifacts.
