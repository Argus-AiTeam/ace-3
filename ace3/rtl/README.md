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
