# ACE-3 Current Status

Technical status reflects the implementation accepted through source revision
`f8270fa1a2601fb57e4c50e08372bcbce09d862d`.

This page distinguishes accepted source-level evidence from active operational
work. Active work is not promoted to an accepted claim until its artifacts are
complete, independently reviewed, and published.

## Accepted boundaries

| Boundary | Evidence level | Reproducible entry point |
| --- | --- | --- |
| Native asymmetric AWQ W4A16 G128 arithmetic | Icarus four-state and Verilator two-state RTL simulation | `make test` |
| Parameterized full-input projection | Official layer-0 `q_proj` samples plus geometry checks | `make projection` |
| FP16 residual, RMSNorm, SiLU/gate, and RoPE operators | Bounded independent-oracle RTL simulation | See `make help` |
| Indexed FP16 K/V cache and attention composition | Bounded independent-oracle RTL simulation | `make attention` |
| Indexed decoder execution | Verilator numerical comparison plus focused Icarus checks | `make decoder-layer0` |
| Model24 controller | Authenticated controller schedule and source/unit publication checks | `make model24-publication-tests` |
| Layer-indexed full-24 cascade | Sealed checkpoint-bound RTL cascade evidence | `make model24-controller-rtl-cascade` |
| Host final RMSNorm and tied-head top-10/argmax | Authenticated checkpoint and independent host/oracle comparison | Model24 publication targets |
| First Voice save/restore and trusted state lineage | Focused tests, real savable self-test, tamper rejection | `make model24-first-voice-hybrid-tests` |
| Compact indexed-layer builder | Cleanup, manifest, tamper, and retained-executable test coverage | `make model24-first-voice-compact-builder-tests` |

## Active milestone

The active milestone is the first genuine readable Hybrid RTL dialogue for the
fixed official chat-template fixture.

An operational set of 24 compact indexed binaries has been built and
reauthenticated outside source control. Publishing that all-layer build as
accepted repository evidence remains part of this milestone.

Required acceptance conditions:

1. every prompt position traverses RTL layers 0 through 23;
2. every layer restores and advances its own authenticated causal K/V state;
3. the first generated token is selected only after the final prompt position;
4. if generation continues, that token is fed back through all 24 RTL layers;
5. selected tokens agree with the independent PyTorch CPU reference;
6. the final artifact records all transaction, cache-lineage, and non-claim
   boundaries.

The traversal is operational work, not yet an accepted repository claim.

## Explicit non-claims

The repository does not currently claim:

- an RTL implementation of final RMSNorm or the full tied language-model head;
- an accepted readable multi-token RTL dialogue artifact;
- a monolithic all-24-layer hardware image;
- formal verification of the complete design;
- synthesis, timing closure, area, power, or PPA;
- FPGA deployment or bitstream;
- measured hardware latency, throughput, memory bandwidth, or energy;
- silicon implementation.

## Next milestones

1. complete and independently authenticate First Voice dialogue;
2. implement RTL final RMSNorm;
3. implement streaming tied `lm_head`/Top-K;
4. add a layer-major prompt-prefill path while retaining token-major generation;
5. extend the standalone datapath to W8A16 and BF16/FP16;
6. scale to 1.5B and 3B model tiers;
7. begin reproducible synthesis/PPA/FPGA work only when the required tools and
   hardware are available.

See [ROADMAP.md](ROADMAP.md) for the complete ordered plan.
