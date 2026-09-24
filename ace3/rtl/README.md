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
address checks. Its default SiLU profile preserves the accepted layer-indexed
regressions; the controller-driven full-model run explicitly selects the
range-reduced exponential profile for every layer. A fault-free full Icarus
trace exceeded the 5,400-second bound
after 7,000,000 controller cycles and is not claimed. The completed
controller-driven Verilator harness executes separately compiled decoder RTL
for layers 0 through 23 and compares the post-layer-23 hidden state to the
independent official oracle. The tied language-model head, dialogue,
synthesis, PPA, and FPGA behavior remain outside this claim.

The Model24 layer controller is a separate arithmetic-free scheduler. It launches
one reusable layer boundary in strict index order from 0 through 23 and exposes a
retained checkpoint after every accepted completion. The next layer cannot launch
until that checkpoint is accepted, and terminal completion follows only the
layer-23 checkpoint. The numerical harness authenticates this launch transcript
before dispatching the separately indexed decoder instances; it is not a
monolithic controller-plus-decoder RTL image.
