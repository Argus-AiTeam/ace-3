# FP16 model-adaptation RTL milestone

ACE-3 now has bounded, projection-compatible binary16 stream operators for
residual addition, RMSNorm, and SiLU/gating. The implemented default dimensions
are Qwen2.5-0.5B hidden size 896 and MLP intermediate size 4864. This milestone
does not integrate a decoder or claim full-model execution.

The arithmetic contract is frozen in
`ace3/contracts/fp16_adaptation_operators.json`. Finite binary16 inputs decode
exactly into Q24 integers. Named division, fixed-point, and binary16 boundaries
use round-to-nearest, ties-to-even. Non-finite inputs produce an invalid,
positive-zero result; finite binary16 overflow saturates to maximum finite and
raises a status flag. RMSNorm uses epsilon `1e-6`, represented by the nearest
Q48 integer 281474977. SiLU uses the documented rational Q0.24 sigmoid
approximation rather than claiming a correctly rounded exponential.

The reuse ledger records the exact ACE-2 RTL snapshots and the inspected Python
reference and testbench patterns. Ready-valid channel separation,
handshake-gated advancement, retained outputs, reset/clear aborts, two-pass
RMSNorm scheduling, and generated self-checking tests are reused structurally.
INT8, Scale32, requantization, LUT clipping, and INT8 saturation are replaced by
the explicit FP16/Q24 contract.

`make test` regenerates and authenticates directed plus checkpoint-derived
vectors, independently recomputes every serialized result, rejects a corrupted
copy, and runs Icarus and Verilator. The Icarus test alone makes bounded
four-state claims for inactive X/Z input data. The measured unstalled
simulation boundaries are one cycle from accepted scalar input to output for
residual and SiLU/gate, and 46 square-root cycles after RMSNorm collection
before its first output. These are RTL simulation cycle counts, not timing or
performance measurements.

The official-derived operands come from hash-checked layer-0 `q_proj` FP16
scale samples for the frozen Qwen AWQ revision. They establish authenticated
checkpoint-derived arithmetic cases, but are not captured hidden states,
RMSNorm weights, or end-to-end model evidence.

Excluded claims remain attention, decoder/full-model execution, correctly
rounded transcendental SiLU, synthesis/PPA, FPGA, silicon, and dialogue.
