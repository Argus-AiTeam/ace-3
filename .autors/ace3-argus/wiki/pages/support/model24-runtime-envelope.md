---
title: Model24 RTL cascade CPU-contention runtime envelope
description: Read-only partial-run timing evidence and the bounded authorization envelope for a later full cascade.
---

# Model24 RTL cascade CPU-contention runtime envelope

The runtime envelope is captured from existing `build/model24_rtl_cascade`
artifacts and does not launch the 24-layer cascade. The capture requires at
least three contiguous layer records, authenticates every consumed compile
log, simulation log, and layer record by SHA-256, and accepts only simulation
logs with one non-vacuous layer PASS. File modification transitions separate
pre-simulation, simulator, and post-simulation phases.

On August 28, 2026, the retained partial run completed layers 0 through 7 and
stopped without a natural terminal during layer 8. Across completed layers 1
through 7, inter-record time ranged from 389.833 to 585.461 seconds with a
401.061-second median. Verilator simulation accounted for 90.44% of their
aggregate inter-record time, so simulator execution materially explains the
completed-layer elapsed time. This does not attribute the interrupted layer-8
cause.

The capture-time machine snapshots had 96 logical CPUs, one-minute load
between 65.97 and 71.91, and 38 to 41 runnable processes. These are
contention-context snapshots after the partial run, not a historical load
trace. The completed-layer range, including the observed initial-to-layer-0
interval, projects to 2 hours 44 minutes through 3 hours 55 minutes. A 25%
margin and whole-hour rounding yield a five-hour durable-run timeout. A layer lacking a natural
terminal after 20 minutes requires review rather than silently extending the
run.

L2 may authorize a later run with the five-hour timeout only when the
scale-aware comparator is enabled after every layer and a true comparison
failure stops before the next layer is materialized. This envelope is host
simulation scheduling evidence only; it is not hardware latency, throughput,
synthesis, PPA, FPGA, or completed full-model evidence.
