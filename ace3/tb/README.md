# Testbenches

RTL testbenches, deterministic vectors, and harness source live here. Each
testbench must identify the independent oracle and arithmetic contract it uses.

The vector testbench checks all 30 frozen cases. The focused protocol testbench
adds bounded Icarus X/Z probes and property-style transaction checks; the C++
harness repeats legal arithmetic and protocol checks under two-state Verilator.
None of these dynamic simulations is a formal proof.

Projection testbenches verify eight complete official q-projection outputs,
six directed outputs, 896- and 4,864-input cycle models, tiled output
sequencing, indices, stalls, reset/clear, backpressure, and bounded Icarus X/Z
behavior. Other projection geometries receive elaboration/lint coverage only.

QKV tests authenticate 512 RoPE pairs spanning every one of 14 query heads and
two key heads, then exercise every dimension of both value heads through indexed
K/V writes and reads. Icarus adds bounded inactive X/Z probes; both simulators
check retained outputs, overwrite, cache-slot/token isolation, reset, and clear.

The attention harness is shared by Icarus and Verilator and checks every
authenticated score, probability, and value-composition record. Both runs cover
retained backpressure, reset/clear aborts, causal masking, GQA mapping, and
cache-miss propagation; bounded X/Z rejection is claimed only for Icarus.

`make decoder-layer0` regenerates and authenticates the two-token layer-0
oracle stream, rejects a tampered trace, runs the fast index/reset/clear/fault
boundary, and compares all stage and final records under Icarus and Verilator.
`make decoder-preload-micro` runs a vector-free, bounded preload-only lifecycle
under both simulators, including expected timeout failures and the non-vacuous
`S_IDLE` to `S_N1_START` transition. Each preload epoch accepts exactly 2,688
ordered entries (896 each for norm1, norm2, and activation); the clear/reload
lifecycle checks 5,376 accepted entries across two epochs.
