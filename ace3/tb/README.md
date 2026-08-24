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
