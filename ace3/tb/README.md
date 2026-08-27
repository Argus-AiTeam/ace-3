# Testbenches

Generation-feedback benches cover reduced tie/backpressure/fault protocol cases
and the official 151936-by-896 Verilator traversal.

`ace3_final_rmsnorm_tb.sv` captures four-state Icarus output evidence without
opening oracle files. `ace3_final_rmsnorm_main.cpp` provides the corresponding
Verilator capture harness.

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
and qzeros boundaries under Icarus, and compares all stage and final records
under Verilator. The full Icarus target remains available explicitly, but a
pre-launch canonical-path gate rejects direct and symlink raw/vector aliases
before cleanup or simulator access. Every Icarus fatal after raw-path binding
writes and closes an abnormal terminal before stopping; a focused injected
failure checks this without restarting the full Icarus trace. A
fault-free run reached the 5,400-second bound at 7,000,000 of the required
controller cycles, so no full-trace Icarus comparison is claimed.
`make decoder-preload-micro` runs a vector-free, bounded preload-only lifecycle
under both simulators, including expected timeout failures and the non-vacuous
`S_IDLE` to `S_N1_START` transition. Each preload epoch accepts exactly 2,688
ordered entries (896 each for norm1, norm2, and activation); the clear/reload
lifecycle checks 5,376 accepted entries across two epochs.

`make decoder-layer01-verilator-cascade` compiles separate layer-0 and layer-1
instances of the parameterized decoder engine. It gates the official layer-0
handoff on natural completion before materializing layer-1 vectors, then
compares every layer-1 trace and final row after a second natural completion.
`make decoder-layer1-iverilog-boundary` is the focused four-state counterpart:
it covers layer-1 elaboration, official vector loading, reset, and abnormal
terminal closure without claiming a full Icarus numerical run.

`make model24-layer-controller` regenerates and authenticates the fixed 24-layer
checkpoint sequence, then runs the controller protocol under both Icarus and
Verilator. The tests cover strict ordering, retained backpressure, layer-23-only
completion, malformed or mismatched transactions, latched fault suppression,
and clear recovery.
