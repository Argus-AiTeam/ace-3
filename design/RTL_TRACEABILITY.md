# ACE-3 FP16 adaptation and QKV/RoPE/KV-cache RTL traceability

## Authority and scope

The frozen numerical and protocol authorities are
`ace3/contracts/fp16_adaptation_operators.json` and
`ace3/contracts/qkv_rope_kv_cache.json`. This note and
`design/RTL_MANIFEST.json` describe the implementation; they do not amend,
relax, or replace either contract.

This trace covers the two shared helper modules in
`ace3/rtl/ace3_fp16_fixed.sv` and the residual-add, RMSNorm, and SiLU/gate
cores, plus the fixed Q/K/V projection cluster, one-pair Qwen2.5 RoPE core, and
parameterized FP16 K/V cache. The pre-existing
`ace3_q47_48_to_f16_rne` converter and
`ace3_awq_w4a16_projection_engine` remain accepted dependencies rather than
new milestone modules. Claims remain limited to authenticated Icarus and
Verilator RTL simulation. Attention scores, softmax, value composition,
decoder-layer or full-model execution, correctly rounded transcendental SiLU,
synthesis/timing/area/power, FPGA, silicon, dialogue, and model quality are
outside this trace.

## First-party source provenance

All five FP16-adaptation modules and all three QKV/RoPE/cache modules are
first-party ACE-3 RTL authored against the native-AWQ W4A16 FP16 contracts. No
ACE-2 arithmetic or cache source is copied. The ACE-2 snapshots and
reference/test hashes for the adaptation baseline remain recorded under
`/ace2_reuse_audit` in its frozen contract.

| Source | Module | SHA256 | ACE-2 relationship |
| --- | --- | --- | --- |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_fp16_to_q24` | `896c0e4c1405bf1cb80141a69259a94af905833ae45a8c7f1338664b4c0552d9` | Replaces INT8/Scale32 decode arithmetic with exact finite binary16-to-Q24 conversion. |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_q24_to_fp16_rne` | `896c0e4c1405bf1cb80141a69259a94af905833ae45a8c7f1338664b4c0552d9` | Replaces requantization and INT8 saturation with the frozen binary16 RNE/saturation boundary. |
| `ace3/rtl/ace3_fp16_residual_add_core.sv` | `ace3_fp16_residual_add_core` | `335954e0bf6909f3aa27330c241cc002f777e6eb5ef483e0b1f684c2fe35ba89` | Re-expresses ready-valid, handshake-gated advancement, retained output, reset, and clear structure; replaces Scale32 and INT8 residual arithmetic. |
| `ace3/rtl/ace3_fp16_rmsnorm_core.sv` | `ace3_fp16_rmsnorm_core` | `f302975fa91aefc20bb48f768fc08ffbddc088e83ca9cdb23b219c2f70d9fc2a` | Re-expresses two-pass scheduling and stream control; adapts sum/square-root dataflow to the frozen Q24/Q48 contract. |
| `ace3/rtl/ace3_fp16_silu_gate_core.sv` | `ace3_fp16_silu_gate_core` | `528b931da1f201c941e7f713a6ba0418f950bd44b64fbf10d92cb5a80f386997` | Re-expresses stream scheduling and retained output; replaces LUT/clipping and fixed INT8-domain arithmetic with the frozen rational sigmoid and wide product. |
| `ace3/rtl/ace3_qkv_projection_cluster.sv` | `ace3_qkv_projection_cluster` | `a02880cc69110b226f0121053b3bad72355e33c1e576f7c749f9daf034305de1` | First-party fixed-checkpoint wrapper around three unchanged accepted ACE-3 projection engines; no ACE-2 source is copied. |
| `ace3/rtl/ace3_qwen2_rope_pair.sv` | `ace3_qwen2_rope_pair` | `d6da922485f1f9818a08e604b3559d56fe407ad42fc2f7605d3bfdded3ee36b8` | First-party half-split Qwen2.5 rotary arithmetic using accepted ACE-3 FP16 converters; no ACE-2 W4A8 path is used. |
| `ace3/rtl/ace3_fp16_kv_cache.sv` | `ace3_fp16_kv_cache` | `fa2b30ca6f22fcc0e2f1fb7ac91761c1aa3d2440d3b9abed28bb76a0569ed179` | First-party SRAM-oriented indexed FP16 K/V storage; no ACE-2 source or cache format is copied. |

The reviewed full-projection baseline is commit
`d6f37f1c3bfcce0c9c71f7d28cd1cd5b97ef0ad6`
(`feat(rtl): add native AWQ full projection`). The FP16 adaptation delta is
accepted commit `241c977dda5ae4615681c583eb8301dfe9d3dd05`
(`feat(rtl): add FP16 adaptation operators`). The QKV/RoPE/cache sources are
the uncommitted review delta on that accepted commit.

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
`design/RTL_MANIFEST.json`; FP16 contract references are JSON Pointers into
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

## QKV projection-cluster interface

`ace3_qkv_projection_cluster` has no parameters. Vector lane 0 is Q, lane 1 is
K, and lane 2 is V. Each concatenated bus is three copies of the corresponding
accepted projection-engine port.

```systemverilog
input  wire         clk_i
input  wire         rst_ni
input  wire         clear_i
input  wire [2:0]   start_valid_i
output wire [2:0]   start_ready_o
input  wire [38:0]  first_output_channel_i
input  wire [38:0]  output_count_i
input  wire [2:0]   meta_valid_i
output wire [2:0]   meta_ready_o
output wire [38:0]  meta_output_channel_o
output wire [17:0]  meta_group_index_o
output wire [29:0]  meta_output_word_o
output wire [8:0]   meta_logical_lane_o
input  wire [95:0]  qzeros_i
input  wire [47:0]  scale_f16_i
input  wire [2:0]   pair_valid_i
output wire [2:0]   pair_ready_o
output wire [38:0]  pair_input_index_o
output wire [38:0]  pair_output_channel_o
output wire [17:0]  pair_group_index_o
output wire [29:0]  pair_output_word_o
output wire [8:0]   pair_logical_lane_o
input  wire [47:0]  activation_f16_i
input  wire [95:0]  qweight_i
output wire [2:0]   out_valid_o
input  wire [2:0]   out_ready_i
output wire [38:0]  out_channel_o
output wire [47:0]  out_f16_o
output wire [305:0] acc_q53_48_o
output wire [2:0]   invalid_operand_o
output wire [2:0]   saturation_o
output wire [2:0]   busy_o
output wire         all_idle_o
```

The exact dependencies are
`ace3_awq_w4a16_projection_engine(IN_FEATURES=896, OUT_FEATURES=896)` for Q
and two instances with `(IN_FEATURES=896, OUT_FEATURES=128)` for K and V.
These retain native asymmetric packed INT4 G128 qweight/qzeros, FP16 scales,
FP16 activations, and the accepted projection protocol.

## Qwen2.5 RoPE-pair interface

`ace3_qwen2_rope_pair` has no parameters. A query head is legal for
`head_index_i=0..13`; a key head is legal for `head_index_i=0..1`.
`pair_index_i=0..31` denotes the Qwen half-split pair `x[p], x[p+32]`, and the
15-bit position carries `0..32767`. Cosine and sine are externally supplied
binary16 coefficients; this module does not claim a coefficient ROM or
transcendental generator.

```systemverilog
input  wire        clk_i
input  wire        rst_ni
input  wire        clear_i
input  wire        in_valid_i
output wire        in_ready_o
input  wire        is_key_i
input  wire [3:0]  head_index_i
input  wire [4:0]  pair_index_i
input  wire [14:0] position_i
input  wire [15:0] low_f16_i
input  wire [15:0] high_f16_i
input  wire [15:0] cos_f16_i
input  wire [15:0] sin_f16_i
output wire        out_valid_o
input  wire        out_ready_i
output wire        is_key_o
output wire [3:0]  head_index_o
output wire [4:0]  pair_index_o
output wire [14:0] position_o
output wire [15:0] low_f16_o
output wire [15:0] high_f16_o
output wire        invalid_operand_o
output wire        saturation_o
```

Its dependencies are `ace3_fp16_to_q24`,
`ace3_q47_48_to_f16_rne(ACC_WIDTH=82)`, and
`ace3_q24_to_fp16_rne(WIDTH=42)`. Each multiply rounds to binary16 RNE before
the binary16 RNE add. A non-finite data or coefficient operand produces
positive-zero outputs with `invalid_operand_o=1`; finite overflow saturates to
signed maximum finite binary16. The one-entry ready-valid output and metadata
remain stable while stalled. Active-low asynchronous reset and active-high
synchronous clear abort a pending output.

## FP16 K/V-cache interface

Parameters and defaults are:

```systemverilog
parameter [2:0]  CACHE_SLOTS = 3'd2
parameter [15:0] MAX_TOKENS  = 16'd128
parameter [4:0]  KV_HEADS    = 5'd2
parameter [6:0]  HEAD_DIM    = 7'd64
```

```systemverilog
input  wire        clk_i
input  wire        rst_ni
input  wire        clear_i
input  wire        write_valid_i
output wire        write_ready_o
input  wire [1:0]  write_cache_slot_i
input  wire [14:0] write_position_i
input  wire [3:0]  write_head_i
input  wire [5:0]  write_dimension_i
input  wire [15:0] write_k_f16_i
input  wire [15:0] write_v_f16_i
input  wire        read_valid_i
output wire        read_ready_o
input  wire [1:0]  read_cache_slot_i
input  wire [14:0] read_position_i
input  wire [3:0]  read_head_i
input  wire [5:0]  read_dimension_i
output wire        out_valid_o
input  wire        out_ready_i
output wire        out_hit_o
output wire [1:0]  out_cache_slot_o
output wire [14:0] out_position_o
output wire [3:0]  out_head_o
output wire [5:0]  out_dimension_o
output wire [15:0] out_k_f16_o
output wire [15:0] out_v_f16_o
```

The flattened order is cache slot, position, K/V head, then head dimension.
The module has no RTL-module dependencies. Separate K and V arrays retain
rotated FP16 K and unrotated FP16 V; only validity metadata is reset or
cleared. A legal accepted read produces a retained response, a miss returns
positive-zero K/V with `out_hit_o=0`, a later write overwrites exactly one
index, and a simultaneous accepted read/write to one address is write-through.

## QKV requirement-to-verification mapping

QKV contract references are JSON Pointers into
`ace3/contracts/qkv_rope_kv_cache.json`.

| Requirement | Contract authority | RTL implementation | Executable check |
| --- | --- | --- | --- |
| `QKV-PROJECTION-001` | `/projection_cluster/composition`, `/projection_cluster/engine` | Three unchanged projection engines at Q 896x896 and K/V 896x128, with lane-isolated interfaces and `all_idle_o`. | `ace3_qkv_projection_geometry_tb.sv` executes all three geometries in Icarus; Verilator lint elaborates the same cluster and dependencies. |
| `QKV-ROPE-GEOMETRY-001` | `/model/query_heads`, `/model/key_value_heads`, `/model/head_dim`, `/model/max_position_embeddings`, `/model/rope_theta`, `/rope/pairing`, `/rope/coefficient_interface` | Half-split 64-dimensional pairs with legal 14-Q-head/2-K-head gating and carried pair/position metadata. | Both dynamic harnesses require 32 cases for each query and key head across 512 authenticated cases; Icarus also rejects key head 2. |
| `QKV-ROPE-NUMERICAL-001` | `/rope/equations`, `/rope/operation_rounding`, `/rope/nonfinite`, `/rope/finite_overflow` | Four binary16-rounded products followed by two binary16-rounded sums, with explicit invalid and saturation status. | Independent `qwen2_rope_oracle.py` expected bits are compared exactly by Icarus and Verilator; directed non-finite handling runs in Icarus. |
| `QKV-ROPE-PROTOCOL-001` | `/rope/protocol`, `/rope/clear_reset` | One-entry ready-valid stage with handshake-gated replacement, stable output/metadata/status under stall, asynchronous reset, and synchronous clear. | Both harnesses force output stalls and compare retained bits; reset release and clear abort are checked, with four-state idle X/Z injection in Icarus. |
| `QKV-CACHE-INDEX-001` | `/kv_cache/key_format`, `/kv_cache/value_format`, `/kv_cache/index_order`, `/kv_cache/defaults`, `/kv_cache/storage` | Parameterized separate K/V arrays and validity array use the declared flattened address. | Both harnesses write and read 128 authenticated cases spanning both K/V heads; misses and slot/position/head/dimension isolation are directed. |
| `QKV-CACHE-READWRITE-001` | `/kv_cache/read`, `/kv_cache/same_cycle`, `/kv_cache/overwrite` | Handshake-gated write, retained one-entry read response, exact-index overwrite, and same-address write-through. | Both harnesses check read/write, stalls, overwrite, and isolation; Icarus additionally checks simultaneous write-through. |
| `QKV-CACHE-ABORT-001` | `/kv_cache/storage`, `/kv_cache/clear_reset` | Reset and clear invalidate metadata and abort the pending response without resetting data arrays. | Both harnesses prove prior entries miss after reset and clear; Icarus also rejects out-of-range and idle X/Z addresses. |
| `QKV-EVIDENCE-001` | `/model/repository`, `/model/revision`, `/model/config_sha256`, `/verification_boundary/official_derived`, `/verification_boundary/simulators` | Generator and validator bind official-checkpoint-derived scale samples, model identity, contract, and serialized streams to SHA256. | `qkv-tamper-rejection` requires a changed `rope_cases.hex` to fail authentication before simulation; the aggregate reruns Icarus and Verilator. |
| `QKV-BOUNDARY-001` | `/verification_boundary/excluded` | No attention, decoder, physical-design, FPGA, or dialogue module is included in this manifest. | Manifest scope and aggregate output remain bounded to projection geometry, RoPE, and cache behavior. |
| `QKV-PROVENANCE-001` | `/projection_cluster/implementation`, `/rope/implementation`, `/kv_cache/implementation` | First-party ACE-3 sources with exact accepted-module dependencies and no copied ACE-2 implementation. | Manifest SHA256 entries bind all three review-candidate RTL files. |

## Verification surfaces and claim boundary

The authenticated QKV surfaces are:

| Surface | Path or target |
| --- | --- |
| Frozen contract | `ace3/contracts/qkv_rope_kv_cache.json` (`4451a9044f82781dcfd861f44583b59e006f1044279c744f886c3f880ce7b098`) |
| Serialized bindings | `ace3/contracts/qkv_rope_kv_cache_vector_bindings.json` (`a66bef29a3b97234b62168fe70fbd5393a248e275c888837e38e93696cdffb09`) |
| Independent bit-level oracle | `ace3/model/qwen2_rope_oracle.py` (`59d7508463973f8a92de48c105df1aa26845b0fe4d05ff7cb9b723e5a6708cde`) |
| Official-checkpoint vector generator | `ace3/model/generate_qkv_rope_cache_vectors.py` (`21ffc4de8da1cdd93a34b2ba4f499e8a0fb5053ac227919b6bafbbd674c0cdac`) |
| Authenticated vector validator | `ace3/model/validate_qkv_rope_cache_vectors.py` (`ff3d451d3043101fc2808b4e54ce4ed204fc903b8143672d570cbf47a3b4fae6`) |
| Icarus geometry and four-state protocol | `ace3/tb/ace3_qkv_projection_geometry_tb.sv`, `ace3/tb/ace3_qkv_rope_cache_tb.sv` |
| Verilator two-state cross-check | `ace3/tb/ace3_qkv_rope_cache_verilator_top.sv`, `ace3/tb/ace3_qkv_rope_cache_main.cpp` |
| Fresh aggregate regression | `make OFFICIAL_TENSOR_DIR=/home/argustest/ace-2/build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official test` |

The official-derived QKV streams are deterministic selections from
hash-authenticated layer-0 `q_proj` FP16 scale samples, not captured runtime
Q/K/V activations. Passing these surfaces does not establish attention scores,
softmax, value composition, decoder-layer or full-model execution,
synthesis/timing/area/power, FPGA or silicon behavior, dialogue, or model
quality.

`make test` is the fresh non-clean aggregate regression. It regenerates and
authenticates vectors, requires tamper rejection, runs Icarus four-state
protocol probes, runs the legal two-state Verilator cross-check, elaborates the
Q/K/V and FP16 parameter geometries, and retains the accepted primitive,
projection, and FP16-adaptation regressions.
