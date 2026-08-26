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
Qwen2.5 geometry. The bounded evidence includes a complete two-token Verilator
comparison of 46,676 intermediate rows and 1,792 post-layer hidden rows plus
focused Icarus width, reset, clear, fail-closed, preload, streaming, and qzeros
address checks. A fault-free full Icarus trace exceeded the 5,400-second bound
after 7,000,000 controller cycles and is not claimed. All 24 layers, the tied
language-model head, dialogue, synthesis, PPA, and FPGA behavior remain outside
this claim.
