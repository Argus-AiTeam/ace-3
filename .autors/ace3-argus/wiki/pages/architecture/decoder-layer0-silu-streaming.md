# Decoder layer-0 SiLU streaming contract

The layer-0 controller feeds 4,864 gate/up pairs to a one-entry
`ace3_fp16_silu_gate_core`. The controller must therefore accept and trace SiLU
outputs while it remains in `S_SI_IN`; waiting until `S_SI_OUT` to assert output
readiness creates a circular wait after the first accepted input.

Input and output progress are tracked independently. `intermediate_index_q`
tracks accepted inputs, while `si_output_count_q` authenticates the ordered SiLU
output index and detects drops or duplicates. The last accepted input advances
to `S_SI_OUT`, where the final pending output is consumed before selecting the
down projection.

The focused boundary covers projection kind 5 at phase 32 with 4,864 accepted
inputs, 4,864 ordered nonzero outputs, trace backpressure, an observed phase 33,
and transition to projection kind 6 at phase 4 in Icarus Verilog and Verilator.
It does not claim an integrated decoder-layer completion or an Oracle
comparison.
