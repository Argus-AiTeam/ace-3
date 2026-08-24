# ACE-3 FP16 adaptation RTL traceability

## Authority and scope

The frozen numerical and protocol authority is
`ace3/contracts/fp16_adaptation_operators.json`. This note and
`design/RTL_MANIFEST.json` describe the implementation; they do not amend,
relax, or replace that contract.

This trace covers the two shared helper modules in
`ace3/rtl/ace3_fp16_fixed.sv` and the residual-add, RMSNorm, and SiLU/gate
cores. The pre-existing `ace3_q47_48_to_f16_rne` projection converter remains
part of the reviewed full-projection baseline and is not an adaptation-delta
module. Claims remain limited to authenticated Icarus and Verilator RTL
simulation. Attention, decoder/full-model behavior, correctly rounded
transcendental SiLU, synthesis/PPA, FPGA, silicon, dialogue, and model quality
are outside this trace.

## First-party source provenance

All five modules are first-party ACE-3 RTL authored against the native-AWQ
W4A16 FP16 contract. No ACE-2 arithmetic source is copied. The ACE-2 snapshots
and reference/test hashes remain recorded under `/ace2_reuse_audit` in the
frozen contract.

| Source | Module | SHA256 | ACE-2 relationship |
| --- | --- | --- | --- |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_fp16_to_q24` | `896c0e4c1405bf1cb80141a69259a94af905833ae45a8c7f1338664b4c0552d9` | Replaces INT8/Scale32 decode arithmetic with exact finite binary16-to-Q24 conversion. |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_q24_to_fp16_rne` | `896c0e4c1405bf1cb80141a69259a94af905833ae45a8c7f1338664b4c0552d9` | Replaces requantization and INT8 saturation with the frozen binary16 RNE/saturation boundary. |
| `ace3/rtl/ace3_fp16_residual_add_core.sv` | `ace3_fp16_residual_add_core` | `335954e0bf6909f3aa27330c241cc002f777e6eb5ef483e0b1f684c2fe35ba89` | Re-expresses ready-valid, handshake-gated advancement, retained output, reset, and clear structure; replaces Scale32 and INT8 residual arithmetic. |
| `ace3/rtl/ace3_fp16_rmsnorm_core.sv` | `ace3_fp16_rmsnorm_core` | `f302975fa91aefc20bb48f768fc08ffbddc088e83ca9cdb23b219c2f70d9fc2a` | Re-expresses two-pass scheduling and stream control; adapts sum/square-root dataflow to the frozen Q24/Q48 contract. |
| `ace3/rtl/ace3_fp16_silu_gate_core.sv` | `ace3_fp16_silu_gate_core` | `528b931da1f201c941e7f713a6ba0418f950bd44b64fbf10d92cb5a80f386997` | Re-expresses stream scheduling and retained output; replaces LUT/clipping and fixed INT8-domain arithmetic with the frozen rational sigmoid and wide product. |

The reviewed full-projection baseline is commit
`d6f37f1c3bfcce0c9c71f7d28cd1cd5b97ef0ad6`
(`feat(rtl): add native AWQ full projection`). The adaptation sources are an
uncommitted delta on that baseline.

## Exact module interfaces

### `ace3_fp16_to_q24`

Parameters: none. This is combinational.

```systemverilog
input  wire [15:0]        f16_i
output reg  signed [40:0] q24_o
output wire               finite_o
output wire               sign_o
```

### `ace3_q24_to_fp16_rne`

Parameter: `parameter integer WIDTH = 84`. This is combinational.

```systemverilog
input  wire signed [WIDTH-1:0] q24_i
input  wire                    zero_sign_i
output wire [15:0]             f16_o
output wire                    saturation_o
```

Core instances set `WIDTH=42` for residual addition, `WIDTH=84` for RMSNorm,
and `WIDTH=64` for SiLU/gate.

### `ace3_fp16_residual_add_core`

Parameter: `parameter integer VECTOR_SIZE = 896`.

```systemverilog
input  wire         clk_i
input  wire         rst_ni
input  wire         clear_i
input  wire         start_valid_i
output wire         start_ready_o
input  wire [12:0]  element_count_i
input  wire         in_valid_i
output wire         in_ready_o
input  wire [15:0]  projection_f16_i
input  wire [15:0]  residual_f16_i
output wire         out_valid_o
input  wire         out_ready_i
output wire [15:0]  out_f16_o
output wire [12:0]  out_index_o
output wire         out_last_o
output wire         invalid_operand_o
output wire         saturation_o
output wire         busy_o
```

### `ace3_fp16_rmsnorm_core`

Parameters: `parameter integer HIDDEN_SIZE = 896` and
`parameter [63:0] EPSILON_Q48 = 64'd281474977`.

```systemverilog
input  wire         clk_i
input  wire         rst_ni
input  wire         clear_i
input  wire         start_valid_i
output wire         start_ready_o
input  wire [12:0]  element_count_i
input  wire         in_valid_i
output wire         in_ready_o
input  wire [15:0]  activation_f16_i
input  wire [15:0]  weight_f16_i
output wire         out_valid_o
input  wire         out_ready_i
output wire [15:0]  out_f16_o
output wire [12:0]  out_index_o
output wire         out_last_o
output wire         invalid_operand_o
output wire         saturation_o
output wire         busy_o
output wire [45:0]  rms_q24_o
```

### `ace3_fp16_silu_gate_core`

Parameter: `parameter integer INTERMEDIATE_SIZE = 4864`.

```systemverilog
input  wire         clk_i
input  wire         rst_ni
input  wire         clear_i
input  wire         start_valid_i
output wire         start_ready_o
input  wire [12:0]  element_count_i
input  wire         in_valid_i
output wire         in_ready_o
input  wire [15:0]  gate_f16_i
input  wire [15:0]  up_f16_i
output wire         out_valid_o
input  wire         out_ready_i
output wire [15:0]  out_f16_o
output wire [12:0]  out_index_o
output wire         out_last_o
output wire         invalid_operand_o
output wire         saturation_o
output wire         busy_o
```

For all three cores, `rst_ni` is an active-low asynchronous abort and
`clear_i` is an active-high synchronous abort. Start, input, and output
transactions occur only on their respective ready-valid handshakes.

## Requirement-to-implementation mapping

The requirement IDs below are defined machine-readably in
`design/RTL_MANIFEST.json`; contract references are JSON Pointers into
`ace3/contracts/fp16_adaptation_operators.json`.

| Requirement | Contract authority | RTL implementation | Executable check |
| --- | --- | --- | --- |
| `FP16-ADAPT-FORMAT-001` | `/projection_compatibility/data_format`, `/projection_compatibility/awq_profile` | All five module interfaces carry binary16 bit patterns or exact fixed-point intermediates; no Scale32 input exists. | Independent oracle, authenticated vectors, Icarus and Verilator comparisons. |
| `FP16-ADAPT-DECODE-001` | `/numerical_contract/decode` | `ace3_fp16_to_q24` exactly maps finite normals and subnormals to signed Q40.24. | Directed edge vectors are independently recomputed by `fp16_adaptation_oracle.py` and checked in both simulators. |
| `FP16-ADAPT-ENCODE-001` | `/numerical_contract/rounding`, `/numerical_contract/finite_overflow`, `/numerical_contract/signed_zero` | `ace3_q24_to_fp16_rne` is instantiated at widths 42, 84, and 64. | Directed tie, signed-zero, and overflow cases require nonzero saturation coverage. |
| `FP16-ADAPT-EXCEPTION-001` | `/numerical_contract/nonfinite` | Each core gates non-finite decoded operands to positive zero and asserts `invalid_operand_o` without saturation. RMSNorm accumulates invalid status across its transaction. | Both harnesses require nonzero invalid-output coverage and exact status agreement. |
| `FP16-ADAPT-PROTOCOL-001` | `/protocol/input`, `/protocol/output` | Counters and state advance only on handshakes; output value/index/last/status are retained while stalled. | Deterministic input/output stalls and three stable-output checks run in both simulators. |
| `FP16-ADAPT-ABORT-001` | `/protocol/reset`, `/protocol/clear` | All cores implement asynchronous reset and synchronous clear transaction aborts. | Icarus and Verilator require one reset and two clear-abort checks. |
| `FP16-ADAPT-START-001` | `/protocol/start`, `/protocol/invalid_start` | Residual and SiLU accept `1..parameter`; RMSNorm accepts exactly `HIDDEN_SIZE`. | Each harness requires three invalid-start rejections; geometry builds elaborate defaults 896/896/4864. |
| `FP16-ADAPT-RESIDUAL-001` | `/implementations/residual_add` and the shared numerical policies | `ace3_fp16_residual_add_core` adds exact Q24 operands and rounds once through `ace3_q24_to_fp16_rne(WIDTH=42)`. | Directed and official-checkpoint-derived operands are compared bit-for-bit with the oracle. |
| `FP16-ADAPT-RMSNORM-001` | `/numerical_contract/rmsnorm` | Two-pass storage, unsigned 92-bit Q44.48 sum, `EPSILON_Q48`, 46-cycle floor square root, wide divide, and one binary16 output rounding. | Full 896-element transactions compare output/status/index/RMS root in Icarus and Verilator. |
| `FP16-ADAPT-SILU-001` | `/numerical_contract/silu_gate` | Rational Q0.24 sigmoid, exact signed Q72 gate product, one RNE Q24 shift, and `WIDTH=64` binary16 conversion. | Directed and authenticated official-derived operands compare bit-for-bit in both simulators. |
| `FP16-ADAPT-DIMENSION-001` | `/projection_compatibility/residual_vector_size`, `/projection_compatibility/rmsnorm_hidden_size`, `/projection_compatibility/silu_intermediate_size` | Defaults are `VECTOR_SIZE=896`, `HIDDEN_SIZE=896`, and `INTERMEDIATE_SIZE=4864`; the bounds remain parameters. | `fp16-geometry` elaborates/lints 896- and 4864-element parameterizations under Icarus and Verilator. |
| `FP16-ADAPT-CYCLE-001` | `/cycle_boundaries` | Residual and SiLU register outputs one cycle after acceptance; RMSNorm runs 46 square-root iterations after collection. | Harnesses reject any mismatch and report measured `residual_latency=1`, `silu_latency=1`, and `rms_sqrt_cycles=46`. |
| `FP16-ADAPT-PROVENANCE-001` | `/ace2_reuse_audit` | First-party ACE-3 arithmetic with explicit reused/adapted/replaced boundaries; ACE-2 remains a read-only structural baseline. | Source hashes in the manifest bind the reviewed delta; the frozen contract binds all audited ACE-2 source/reference/test snapshots. |

`make test` is the fresh aggregate regression. It regenerates and
authenticates vectors, requires tamper rejection, runs Icarus four-state
protocol probes, runs the legal two-state Verilator cross-check, elaborates the
parameter geometries, and retains the accepted projection regressions.
