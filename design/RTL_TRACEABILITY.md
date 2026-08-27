# ACE-3 FP16 adaptation, QKV/RoPE/KV-cache, and attention RTL traceability

## Authority and scope

The frozen numerical and protocol authorities are
`ace3/contracts/fp16_adaptation_operators.json` and
`ace3/contracts/qkv_rope_kv_cache.json`, plus
`ace3/contracts/attention_block.json`. This note and
`design/RTL_MANIFEST.json` describe the implementation; they do not amend,
relax, or replace those contracts.

This trace covers the two shared helper modules in
`ace3/rtl/ace3_fp16_fixed.sv` and the residual-add, RMSNorm, and SiLU/gate
cores, plus the fixed Q/K/V projection cluster, one-pair Qwen2.5 RoPE core, and
parameterized FP16 K/V cache, and the score, causal-softmax, and
value-composition attention cores. The pre-existing
`ace3_q47_48_to_f16_rne` converter and
`ace3_awq_w4a16_projection_engine` remain accepted dependencies rather than
new milestone modules. Claims remain limited to authenticated Icarus and
Verilator RTL simulation. Decoder-layer or full-model execution, correctly
rounded transcendental SiLU, synthesis/timing/area/power, FPGA, silicon,
dialogue, and model quality are outside this trace.

## First-party source provenance

All five FP16-adaptation modules and all three QKV/RoPE/cache modules are
first-party ACE-3 RTL authored against the native-AWQ W4A16 FP16 contracts. No
predecessor arithmetic or cache source is copied. The predecessor snapshots and
reference/test hashes for the adaptation baseline remain recorded under
`/predecessor_reuse_audit` in its frozen contract.

| Source | Module | SHA256 | predecessor relationship |
| --- | --- | --- | --- |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_fp16_to_q24` | `e7f38a3434b60849896a0a5bab549bd6d4b6a4908280a2860a33fc8e839c86c8` | Replaces INT8/Scale32 decode arithmetic with exact finite binary16-to-Q24 conversion. |
| `ace3/rtl/ace3_fp16_fixed.sv` | `ace3_q24_to_fp16_rne` | `e7f38a3434b60849896a0a5bab549bd6d4b6a4908280a2860a33fc8e839c86c8` | Replaces requantization and INT8 saturation with the frozen binary16 RNE/saturation boundary. |
| `ace3/rtl/ace3_fp16_residual_add_core.sv` | `ace3_fp16_residual_add_core` | `335954e0bf6909f3aa27330c241cc002f777e6eb5ef483e0b1f684c2fe35ba89` | Re-expresses ready-valid, handshake-gated advancement, retained output, reset, and clear structure; replaces Scale32 and INT8 residual arithmetic. |
| `ace3/rtl/ace3_fp16_rmsnorm_core.sv` | `ace3_fp16_rmsnorm_core` | `f302975fa91aefc20bb48f768fc08ffbddc088e83ca9cdb23b219c2f70d9fc2a` | Re-expresses two-pass scheduling and stream control; adapts sum/square-root dataflow to the frozen Q24/Q48 contract. |
| `ace3/rtl/ace3_fp16_silu_gate_core.sv` | `ace3_fp16_silu_gate_core` | `c5f26b50ba2396852e966a430719a6170df9adf5546903ec018049e9e084cf43` | Re-expresses stream scheduling and retained output; supports the reviewed rational sigmoid and the range-reduced exponential profile with the same wide gate product. |
| `ace3/rtl/ace3_qkv_projection_cluster.sv` | `ace3_qkv_projection_cluster` | `a02880cc69110b226f0121053b3bad72355e33c1e576f7c749f9daf034305de1` | First-party fixed-checkpoint wrapper around three unchanged accepted ACE-3 projection engines; no predecessor source is copied. |
| `ace3/rtl/ace3_qwen2_rope_pair.sv` | `ace3_qwen2_rope_pair` | `d6da922485f1f9818a08e604b3559d56fe407ad42fc2f7605d3bfdded3ee36b8` | First-party half-split Qwen2.5 rotary arithmetic using accepted ACE-3 FP16 converters; no predecessor W4A8 path is used. |
| `ace3/rtl/ace3_fp16_kv_cache.sv` | `ace3_fp16_kv_cache` | `fa2b30ca6f22fcc0e2f1fb7ac91761c1aa3d2440d3b9abed28bb76a0569ed179` | First-party SRAM-oriented indexed FP16 K/V storage; no predecessor source or cache format is copied. |
| `ace3/rtl/ace3_attention_score_core.sv` | `ace3_attention_score_core` | `35db39940444c3f286c0110c94f46347e7192511b162a7ef4d10a6b299be5221` | First-party exact-Q24/Q48 scaled-QK reduction; no predecessor W4A8 arithmetic is copied. |
| `ace3/rtl/ace3_attention_softmax_core.sv` | `ace3_attention_softmax_core` | `899220e73557cbbb46f4e409551b2396e59ad230b126b2f3fc4809fd8d9d11ca` | First-party frozen Q0.24 causal-softmax approximation; no predecessor source is copied. |
| `ace3/rtl/ace3_attention_value_core.sv` | `ace3_attention_value_core` | `17beacde641684df7f87c03d62c8187e43f6b8a6a928734cd8a3916ce8d14201` | First-party exact-Q24/Q48 cached-V composition; no predecessor W4A8 arithmetic is copied. |

The reviewed full-projection baseline is commit
`d6f37f1c3bfcce0c9c71f7d28cd1cd5b97ef0ad6`
(`feat(rtl): add native AWQ full projection`). The FP16 adaptation delta is
accepted commit `241c977dda5ae4615681c583eb8301dfe9d3dd05`
(`feat(rtl): add FP16 adaptation operators`). The standalone attention baseline
is accepted commit `645760c1de73c83c44515b6174b03fdfa04ba9bb`
(`feat(rtl): add standalone attention block`).

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
| `FP16-ADAPT-PROVENANCE-001` | `/predecessor_reuse_audit` | First-party ACE-3 arithmetic with explicit reused/adapted/replaced boundaries; predecessor remains a read-only structural baseline. | Source hashes in the manifest bind the reviewed delta; the frozen contract binds all audited predecessor source/reference/test snapshots. |

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
| `QKV-PROVENANCE-001` | `/projection_cluster/implementation`, `/rope/implementation`, `/kv_cache/implementation` | First-party ACE-3 sources with exact accepted-module dependencies and no copied predecessor implementation. | Manifest SHA256 entries bind all three review-candidate RTL files. |

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
| Fresh aggregate regression | `make OFFICIAL_TENSOR_DIR=/path/to/official_tensors test` |

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

## Attention requirement-to-verification mapping

Attention contract references are JSON Pointers into
`ace3/contracts/attention_block.json`.

| Requirement | Contract authority | RTL implementation | Executable check |
| --- | --- | --- | --- |
| `ATTENTION-GEOMETRY-001` | `/geometry` | Score and value starts enforce 14-to-2 GQA with `kv_head=floor(query_head/7)`; score and softmax use 64-element heads and `key_position <= query_position`. | Authenticated cases span all 14 query heads and both K/V groups; invalid GQA starts are rejected in both simulators. |
| `ATTENTION-SCORE-001` | `/numerical_policy/score` | `ace3_attention_score_core` exactly decodes Q/K to Q24, accumulates 64 Q48 products in 90 bits, scales by 1/8 with RNE, and converts once to FP16. | The bit-level oracle recomputes all 20 score cases, including causal masking, K misses, non-finite inputs, saturation, and cancellation. |
| `ATTENTION-SOFTMAX-001` | `/numerical_policy/softmax` | `ace3_attention_softmax_core` buffers a row, performs Q24 max subtraction, uses the frozen 17-point Q0.24 interpolation, and normalizes with RNE. | Both simulators compare all 140 probabilities from 20 rows; directed rows cover future masks, ties, underflow, cache miss, invalid input, and no eligible key. |
| `ATTENTION-VALUE-001` | `/numerical_policy/value_composition` | `ace3_attention_value_core` accumulates exact Q48 probability/V products in 90 bits and rounds once at the final FP16 boundary. | The oracle and both simulators compare 20 value cases with required-V miss, non-finite, saturation, cancellation, and upstream row-error coverage. |
| `ATTENTION-PROTOCOL-001` | `/protocol_policy` | All three cores abort on reset/clear, retain output and metadata under backpressure, and suppress ready for illegal or unknown configuration/payload metadata. | Both simulators check retained stalls, reset, clear, and legal configuration; Icarus adds six bounded X/Z rejection checks. |
| `ATTENTION-EVIDENCE-001` | `/verification_boundary/required` | Generation re-authenticates the fixed model identity, config, and official layer-0 q-projection FP16 scale sample before serializing vectors. | `make attention` regenerates, authenticates, rejects a tampered score stream, builds and runs Icarus, then builds and runs Verilator. |
| `ATTENTION-BOUNDARY-001` | `/verification_boundary/excluded` | Only standalone score, causal-softmax, and value-composition cores are included. | Aggregate output does not claim decoder/full-model execution, dialogue, formal proof, synthesis, timing, PPA, FPGA, silicon, or performance. |

## Attention verification surfaces and claim boundary

| Surface | Path or target |
| --- | --- |
| Frozen contract | `ace3/contracts/attention_block.json` (`9c96b51f89c747b001debb69929940ccd252f39777f8d01a904f2b56aae418ac`) |
| Serialized bindings | `ace3/contracts/attention_vector_bindings.json` (`bb3ea6f8561058c0c1857f41eaa6d32320a1351fc9ba10e1da4cd9a0fdf450d3`) |
| Generated manifest | `build/attention_vectors/manifest.json` (`25cbb9327b4bdcc16fd326e8902ab57daedf15a59b9b1759a0ac14e0c8d05f25`, regenerated and ignored) |
| Shared FP16 RTL | `ace3/rtl/ace3_fp16_fixed.sv` (`e7f38a3434b60849896a0a5bab549bd6d4b6a4908280a2860a33fc8e839c86c8`) |
| Independent bit-level oracle | `ace3/model/attention_oracle.py` (`0d18f27c4c01e4c7f9397958ca52f63960308374a2baf35e96e730cce15a6acd`) |
| Official-checkpoint vector generator | `ace3/model/generate_attention_vectors.py` (`5db02bdf108390e83841dee3d8f55fe28deafe9ec21f6cc20bdf62e8946db8eb`) |
| Authenticated vector validator | `ace3/model/validate_attention_vectors.py` (`64400a6c10b4f1850f9b0623f457fb66aec1f57319f456577987175ec9aabc61`) |
| Icarus four-state testbench | `ace3/tb/ace3_attention_block_tb.sv` (`40fe3d9332684415e01f1fcc5f404dfb26ad644118d014bd532f9854bd982036`) |
| Verilator two-state top/harness | `ace3/tb/ace3_attention_verilator_top.sv` (`83c04b524db8363ac6c99a651dd2e57f9a96d13a0f5a6d488da7dce26af9512e`), `ace3/tb/ace3_attention_main.cpp` (`5f5fa0ecc7b27ccd5b75d663f7d605b51cdca3f98a44b9af38e72bed50cc0ae0`) |
| Fresh aggregate regression | `make attention` |

The generated operands use authenticated official source hashes
`bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090`
for `config.json`,
`9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768`
for `model-api.json`, and
`687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321`
for the layer-0 q-projection FP16 scale sample. These operands are deterministic
official-derived selections, not captured runtime Q/K/V activations. The
evidence establishes standalone attention-core correctness only.

## Decoder layer-0 integration mapping

Decoder references are JSON Pointers into
`ace3/contracts/decoder_layer0_token_engine.json`.

| Requirement | Contract authority | RTL implementation | Executable check |
| --- | --- | --- | --- |
| `DECODER-L0-COMPOSE-001` | `/scope`, `/geometry`, `/numerical_order` | `ace3_decoder_layer0_token_engine` serially composes the accepted projection, RMSNorm, half-split RoPE, FP16 K/V cache, attention, residual, and SiLU operators for the fixed layer-0 geometries. | The complete Verilator run compares 46,676 retained stage rows and 1,792 final hidden rows for positions 0 and 1. |
| `DECODER-L0-CACHE-001` | `/geometry`, `/trace` | RoPE-K completion resets the K/V writeback iterator to head zero, writes both K/V heads, and preserves sequential slot-zero context while admitting an independent slot-one position zero. | The full run requires two sequential token completions and a non-vacuous slot-isolation start; focused Icarus cache and address tests remain retained. |
| `DECODER-L0-STREAM-001` | `/protocol_responsibilities` | Residual-1 and residual-2 consume their one-entry outputs while inputs stream; the final residual waits until the preceding DOWN trace is accepted. | Deterministic output stalls cover the streaming path and every final row is byte-compared after natural simulator completion. |
| `DECODER-L0-EVIDENCE-001` | `/trace`, `/oracle_integer_safety` | Generated tensors, coefficients, traces, and final rows are SHA256-authenticated and independently regenerated before simulation. | Tamper rejection, a strict natural-terminal gate, byte comparison, and injected failure with no Oracle-file open are required. |
| `DECODER-L0-BOUNDARY-001` | `/scope` | The numerical claim is one reduced official layer-0 trace under Verilator plus focused Icarus four-state boundaries. | A fault-free Icarus full-trace attempt reached the 5,400-second bound at 7,000,000 controller cycles, before completing token zero; no Icarus full-trace pass is claimed. |

## Decoder layer-0 verification surfaces and claim boundary

| Surface | Path or target |
| --- | --- |
| Frozen contract | `ace3/contracts/decoder_layer0_token_engine.json` (`7026c694baf8c45ea3808cb10582bdd6884bb85de5c3e1c6e8b5553f1751cf99`) |
| Serialized bindings | `ace3/contracts/decoder_layer0_vector_bindings.json` (`12c433d7d3999d2afdcf9f3424a1340c3ada1ccf1a7983e66f23b2d736769040`) |
| Generated boundary manifest | `build/decoder_layer0_vectors/boundary_manifest.json` (`c0e77c256b5c4ae68de55a8102a949835508f46ab0515d9edd0d9c48739b4089`, regenerated and ignored) |
| Independent oracle | `ace3/model/decoder_layer0_oracle.py` (`33ff7d93ecad077d4d6f88442f70ccc52450892dc65a976b17de882bce953128`) |
| Official vector generator | `ace3/model/generate_decoder_layer0_vectors.py` (`2cbc91f71b56bbf2c1f819642cca02249df3f48d17ac4d05c384692cc0e0bda8`) |
| Authenticated validator | `ace3/model/validate_decoder_layer0_vectors.py` (`851cccaf713029781e9703547d9d9224b81dc3248293574e0235f69eb888701e`) |
| Integrated RTL | `ace3/rtl/ace3_decoder_layer0_token_engine.sv` (`34ba6fdf14079dba327cadab708f8102881551c202fbca1f9d1e233c2f43b986`) |
| Qzeros address RTL | `ace3/rtl/ace3_decoder_qzeros_address.sv` (`cf5f8e9d41cc82eb5082046566665cbd14be48e3c2e7ae60b4753ae9409ec344`) |
| Icarus testbench | `ace3/tb/ace3_decoder_layer0_token_engine_tb.sv` (`5440f1084a54be331713721c57cd696587b4a1e3e6522a6c0169f11be608629c`) |
| Verilator harness | `ace3/tb/ace3_decoder_layer0_token_engine_main.cpp` (`31f312030712b2e30e08baa426a835e74d70d14f577ae4a318cf869301652a22`) |
| Fresh bounded aggregate | `make decoder-layer0 OFFICIAL_TENSOR_DIR=/path/to/authenticated/official` |

This establishes the reduced official layer-0 integrated trace and post-layer
hidden output under Verilator, with focused Icarus four-state boundary coverage.
It does not establish all 24 layers, the tied language-model head, full-model
execution, readable dialogue, formal proof, synthesis, timing, area, power,
FPGA deployment, latency, throughput, or other performance.

## Decoder layer-0 to layer-1 cascade

The parameterized `ace3_decoder_layer0_token_engine` binds its compiled
`LAYER_INDEX` to the runtime vector namespace. The bounded cascade executes
layer 0 first, accepts its two-token final stream only after natural completion
and independent comparison, then materializes layer-1 tensors and oracle rows
from the fixed official checkpoint. Layer 1 is compiled separately with
`LAYER_INDEX=1` and consumes the byte-identical layer-0 final stream.

| Surface | Binding |
| --- | --- |
| Official checkpoint | `model24_execution_vectors/model.safetensors` (`c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b`) |
| Reviewed tensor map | `ace3/contracts/model24_tensor_map.json` (`11a03bed8049cd815ac2c37384a7ba15d71d2f69ee397110d1cd443193474624`) |
| Official layer-1 descriptor | `model.layers.1.` (`c8a037c0043ededc764f02b14671781ceeb1fb5be3fa6b7f8e114d75a98ad8f4`) |
| Layer-0 handoff | 1,792 rows, SHA256 `22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446` |
| Layer-1 vector generator | `ace3/model/generate_decoder_layer1_vectors.py` (`879976cc1465537b98a6b37b40bbfcd74f88ec8450a922252e42cfd1c99f23d7`) |
| Independent indexed oracle | `ace3/model/model24_execution_oracle.py` (`d452dfbd10a6694cd183750bfadd22511097f50a2aec7c6d9f79117394ade55b`) |
| Post-layer-1 oracle | 1,792 rows, SHA256 `2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85` |
| Focused four-state check | `make decoder-layer1-iverilog-boundary` |
| Complete numerical check | `make decoder-layer01-verilator-cascade` |

The cascade gate rejects a modified layer-0 handoff before creating layer-1
vectors. The layer-1 simulator writes only raw rows and a terminal; oracle
files are opened by the later comparison gate only after an exact successful
natural terminal. Injected failure after one raw row must leave no comparison
report and no oracle-file open. Focused Icarus covers layer-1 elaboration,
official vector loading, reset, and abnormal-terminal closure; a full Icarus
layer-1 numerical pass is not claimed.

This evidence is limited to two tokens through official decoder layers 0 and 1.
It excludes layers 2 through 23, the tied language-model head, full-model
execution, readable dialogue, formal proof, synthesis, timing, area, power,
FPGA deployment, latency, throughput, and other performance.

## Decoder layer-0 through layer-2 cascade

The bounded continuation reuses the accepted layer-0/layer-1 cascade and
requires its naturally completed layer-1 raw final stream to match the
independent post-layer-1 Oracle before any layer-2 materialization. The
layer-2 generator then authenticates the predecessor hash, official checkpoint,
reviewed tensor map, layer descriptor, and all 26 consumed tensor values.
Layer 2 is compiled separately with `LAYER_INDEX=2`.

| Surface | Binding |
| --- | --- |
| Official checkpoint | `model24_execution_vectors/model.safetensors` (`c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b`) |
| Reviewed tensor map | `ace3/contracts/model24_tensor_map.json` (`11a03bed8049cd815ac2c37384a7ba15d71d2f69ee397110d1cd443193474624`) |
| Official layer-2 descriptor | `model.layers.2.` (`07b907a2f7a800af011b630ce2a026593f05fbd9447e3f106e8970be7888d916`) |
| Layer-1 handoff | 1,792 rows, SHA256 `2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85` |
| Layer-2 vector generator | `ace3/model/generate_decoder_layer2_vectors.py` (`86eb43c07fe6972c6544274fd5d88385043dd90a801d7ba0091a18d6444fc418`) |
| Independent indexed Oracle | `ace3/model/model24_execution_oracle.py` (`d452dfbd10a6694cd183750bfadd22511097f50a2aec7c6d9f79117394ade55b`) |
| Parameterized RTL | `ace3/rtl/ace3_decoder_layer0_token_engine.sv` (`34ba6fdf14079dba327cadab708f8102881551c202fbca1f9d1e233c2f43b986`) |
| Focused four-state testbench | `ace3/tb/ace3_decoder_layer0_token_engine_tb.sv` (`5440f1084a54be331713721c57cd696587b4a1e3e6522a6c0169f11be608629c`) |
| Two-state harness | `ace3/tb/ace3_decoder_layer0_token_engine_main.cpp` (`31f312030712b2e30e08baa426a835e74d70d14f577ae4a318cf869301652a22`) |
| Post-layer-2 Oracle | 1,792 rows, SHA256 `244c9d1d52923ecfff743c165da563468746f47557284865a4b22910a967c511` |
| Focused four-state check | `make decoder-layer2-iverilog-boundary` |
| Complete numerical check | `make decoder-layer012-verilator-cascade` |

The layer-2 simulator remains Oracle-free and writes raw rows plus a terminal.
Only an exact natural terminal permits comparison of all 46,676 retained stage
rows and all 1,792 final rows. Deterministic backpressure is active throughout
the complete Verilator run. The continuation rejects a modified layer-1
handoff before vector output and proves that an injected simulator failure
after one durable raw row neither opens Oracle outputs nor creates comparison
evidence. Focused Icarus covers layer-2 elaboration, official vector loading,
reset, and abnormal-terminal closure; a full Icarus layer-2 numerical run is
not claimed.

This evidence is limited to two tokens through official decoder layers 0, 1,
and 2. It excludes layers 3 through 23, the tied language-model head,
full-model execution, readable dialogue, formal proof, synthesis, timing,
area, power, FPGA deployment, latency, throughput, and other performance.

## Model24 checkpointed controller and full-24 RTL cascade

The arithmetic-free controller accepts one start, launches layers 0 through 23
in strict order, retains every checkpoint under backpressure, and permits a
natural terminal only after the layer-23 checkpoint. The numerical harness
authenticates that event stream before launching one separately compiled,
layer-indexed Verilator decoder instance for each accepted layer.

| Sealed surface | SHA256 |
| --- | --- |
| Official checkpoint | `c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b` |
| Per-layer binding document | `95a46cfb25d8479a9d9921da9b78e581b1c3746c2645574880dac7ea6825ede0` |
| Ordered controller event stream | `fcb4c9a6458fa141b143d9a4c7dfd10b2d15e703257067d935f635dc1bf9dbf1` |
| Post-layer-23 hidden state | `97e729f6f905ecb62f498a6a144beecf6b695465d84fdbaf1de777ce9f5a39b6` |
| Complete execution document | `9d4e048d1316252d67d7e288fd4de0a2a6360f53ff49854c1127b7462578a5c1` |

The run consumes all 624 decoder tensors. Layers 0 through 2 preserve the
reviewed rational SiLU profile and layers 3 through 23 use the range-reduced
degree-7 Q24 exponential profile. The post-layer-23 decision-token maximum
absolute error is `0.08988498970425507`, below the fixed `0.125` bound.

The layer-3 Token 0 diagnostic binds the same layer-2 handoff and discloses one
material final outlier at down-projection dimension 62. It remains within one
FP16 ULP and below 0.001 relative error after the final residual. Token 1 remains
below 0.01 and preserves two-position K/V causality. The RTL remains bit-exact
to its integer oracle at every retained row.

This is a controller-driven, two-token, layer-indexed RTL simulation result. It
does not execute the tokenizer or tied language-model head and does not
establish a monolithic full-model RTL image, formal proof, synthesis, timing,
area, power, PPA, FPGA execution, latency, throughput, or silicon behavior.
