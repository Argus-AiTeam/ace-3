# RTL

Synthesizable ACE-3 modules live here. Generated netlists and simulator output
belong under ignored build directories, not in this source tree.

The full-input projection engine composes the accepted G128 lane and the
parameterized Q53.48-to-binary16 final rounder. It is a sequential/tiled
datapath, not a parallel performance implementation.

The QKV milestone adds a fixed three-engine projection cluster, Qwen2.5
half-split FP16 rotary pair datapath, and parameterized indexed FP16 K/V cache.
The cache arrays are SRAM-oriented data stores with separately invalidated
validity metadata; no synthesis or physical-memory mapping is claimed.

The attention candidate adds separate retained-handshake score, softmax, and
value-composition cores. They implement the fixed 14-to-2 GQA mapping, causal
eligibility, an explicitly bounded Q0.24 exponential approximation, and FP16
cached-V accumulation.

The layer-0 token engine serially composes the accepted projection, FP16
adaptation, half-split RoPE, K/V cache, and attention cores for the fixed
Qwen2.5 geometry. The bounded evidence here is Icarus simulation of its width,
reset, clear, and fail-closed boundary plus model24 structural vectors bound to
the exact RTL and testbench hashes. Full numerical token execution, all 24
layers, dialogue, synthesis, PPA, and FPGA behavior remain outside this claim.
