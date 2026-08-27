# RTL

`ace3_generation_feedback_chain.sv` connects the unchanged tied lm_head directly
to the selected-token feedback stage; no Host-selected token port exists.

`ace3_final_rmsnorm.sv` is the parameter-free 896-element final-layer wrapper
around the accepted FP16 RMSNorm arithmetic core.

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
after 7,000,000 controller cycles and is not claimed.

The arithmetic-free `ace3_model24_layer_controller` launches one reusable layer
boundary in strict index order from 0 through 23, retaining every checkpoint
until acceptance and allowing terminal completion only after layer 23. The
controller-driven cascade uses separately compiled decoder instances; it is not
a monolithic controller-plus-decoder RTL image. All 24 indexed layers select
the range-reduced exponential SiLU profile. The tied language-model head,
dialogue, synthesis, PPA, and FPGA behavior remain outside this claim.
The position-1 causal traversal reuses the existing indexed decoder RTL without
changing its public ports. Its Verilator `--savable` image is restored only
after checkpoint, binary, layer, position, envelope, state, and parent K/V
lineage authenticate against the pinned ordered parent set.
The independent reference imports the same layer-indexed inherited K/V values
before evaluating position 1; it does not regenerate position 0.
