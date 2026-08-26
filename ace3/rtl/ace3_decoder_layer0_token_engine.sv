`timescale 1ns/1ps
`default_nettype none

/*
 * One deliberately serial Qwen2.5 decoder-layer token engine.  Keeping the
 * projections serial makes the (very large) AWQ tensor port unambiguous:
 * every request describes precisely one word of one tensor.
 */
module ace3_decoder_layer0_token_engine #(
    parameter integer LAYER_INDEX = 0,
    parameter integer ACCURATE_SILU = (LAYER_INDEX >= 3)
) (
    input  wire clk_i, input wire rst_ni, input wire clear_i,

    input wire load_valid_i, output wire load_ready_o,
    input wire [1:0] load_kind_i, input wire [12:0] load_index_i,
    input wire [15:0] load_f16_i,
    input wire start_valid_i, output wire start_ready_o,
    input wire [1:0] start_cache_slot_i, input wire [14:0] start_position_i,

    output wire [2:0] projection_kind_o,
    input wire projection_meta_valid_i, output wire projection_meta_ready_o,
    output wire [12:0] projection_meta_output_channel_o,
    output wire [5:0] projection_meta_group_o,
    output wire [9:0] projection_meta_word_o,
    output wire [2:0] projection_meta_lane_o,
    input wire [31:0] projection_qzeros_i, input wire [15:0] projection_scale_f16_i,
    input wire projection_pair_valid_i, output wire projection_pair_ready_o,
    output wire [12:0] projection_pair_input_o,
    output wire [12:0] projection_pair_output_o,
    output wire [5:0] projection_pair_group_o,
    output wire [9:0] projection_pair_word_o,
    output wire [2:0] projection_pair_lane_o,
    input wire [31:0] projection_qweight_i,
    input wire projection_bias_valid_i, output wire projection_bias_ready_o,
    output wire [12:0] projection_bias_output_channel_o,
    input wire [15:0] projection_bias_f16_i,

    input wire rope_valid_i, output wire rope_ready_o,
    output wire [14:0] rope_position_o, output wire [4:0] rope_pair_o,
    input wire [15:0] rope_cos_f16_i, input wire [15:0] rope_sin_f16_i,

    output wire trace_valid_o, input wire trace_ready_i,
    output wire [4:0] trace_stage_o, output wire [12:0] trace_index_o,
    output wire [15:0] trace_f16_o, output wire [14:0] trace_position_o,

    output wire final_valid_o, input wire final_ready_i,
    output wire [12:0] final_index_o, output wire [15:0] final_f16_o,
    output wire final_last_o,
    output wire done_valid_o, input wire done_ready_i,
    output wire [1:0] done_cache_slot_o, output wire [14:0] done_position_o,
    output wire [31:0] done_cycles_o, output wire [31:0] done_stall_cycles_o,
    output wire busy_o, output wire [5:0] phase_o,
    output wire [4:0] layer_index_o
);
    localparam [4:0] TRACE_NORM1  = 5'd0,  TRACE_Q      = 5'd1;
    localparam [4:0] TRACE_K      = 5'd2,  TRACE_V      = 5'd3;
    localparam [4:0] TRACE_RQ     = 5'd4,  TRACE_RK     = 5'd5;
    localparam [4:0] TRACE_CK     = 5'd6,  TRACE_CV     = 5'd7;
    localparam [4:0] TRACE_SCORE  = 5'd8,  TRACE_PROB   = 5'd9;
    localparam [4:0] TRACE_AV     = 5'd10, TRACE_O      = 5'd11;
    localparam [4:0] TRACE_RES1   = 5'd12, TRACE_NORM2  = 5'd13;
    localparam [4:0] TRACE_GATE   = 5'd14, TRACE_UP     = 5'd15;
    localparam [4:0] TRACE_SILU   = 5'd16, TRACE_DOWN   = 5'd17;
    localparam [4:0] TRACE_RES2   = 5'd18;

    localparam [5:0] S_IDLE=0, S_N1_START=1, S_N1_IN=2, S_N1_OUT=3,
                     S_P_START=4, S_P_RUN=5, S_RQ_REQ=6, S_RQ_LO=7,
                     S_RQ_HI=8, S_RK_REQ=9, S_RK_LO=10, S_RK_HI=11,
                     S_CW_LO=12, S_CW_HI=13, S_SC_START=14, S_SC_READ=15,
                     S_SC_TERM=16, S_SC_OUT=17, S_SM_START=18, S_SM_IN=19,
                     S_SM_OUT=20, S_AV_START=21, S_AV_READ=22, S_AV_TERM=23,
                     S_AV_OUT=24, S_R1_START=25, S_R1_IN=26, S_R1_OUT=27,
                     S_N2_START=28, S_N2_IN=29, S_N2_OUT=30,
                     S_SI_START=31, S_SI_IN=32, S_SI_OUT=33,
                     S_R2_START=34, S_R2_IN=35, S_R2_OUT=36,
                     S_FAULT=63;
    localparam [2:0] PK_Q=0, PK_K=1, PK_V=2, PK_O=3, PK_GATE=4,
                     PK_UP=5, PK_DOWN=6;
    localparam integer HIDDEN_SIZE = 896;
    localparam integer INTERMEDIATE_SIZE = 4864;
    localparam integer CONTEXT_MAX = 128;
    localparam integer QUERY_HEADS = 14;
    localparam integer KV_HEADS = 2;
    localparam integer HEAD_DIM = 64;
    localparam integer Q_FLAT_MAX = QUERY_HEADS * HEAD_DIM - 1;
    localparam integer KV_FLAT_MAX = KV_HEADS * HEAD_DIM - 1;
    localparam [4:0] LAYER_INDEX_VALUE = LAYER_INDEX[4:0];

    assign layer_index_o = LAYER_INDEX_VALUE;

    initial begin
        if ((LAYER_INDEX < 0) || (LAYER_INDEX > 23))
            $error("decoder layer index must be in [0,23]");
        if (Q_FLAT_MAX != 895)
            $error("Q flattened geometry must end at index 895");
        if (KV_FLAT_MAX != 127)
            $error("K/V flattened geometry must end at index 127");
        if ((HIDDEN_SIZE > 1024) || (INTERMEDIATE_SIZE > 8192) ||
            (CONTEXT_MAX > 128))
            $error("decoder index domain is too narrow");
    end

    reg [5:0] state_q;
    reg [2:0] psel_q;
    reg [9:0] hidden_index_q, load_act_q, load_n1_q, load_n2_q;
    reg [12:0] intermediate_index_q, si_output_count_q;
    reg [6:0] context_index_q;
    reg [3:0] head_q;
    reg [5:0] dim_q;
    reg [6:0] key_position_q, token_position_q;
    reg token_slot_q;
    reg fault_q;
    reg [12:0] projection_output_index_q;
    reg softmax_row_error_q;
    reg [7:0] context_len_q [0:1];
    reg activation_loaded_q, n1_loaded_q, n2_loaded_q;
    reg [15:0] activation_mem [0:895];
    reg [15:0] norm1_weight_mem [0:895];
    reg [15:0] norm2_weight_mem [0:895];
    reg [15:0] norm1_mem [0:895], q_mem [0:895], k_mem [0:127];
    reg [15:0] v_mem [0:127], rk_mem [0:127], attention_mem [0:895];
    reg [15:0] o_mem [0:895], res1_mem [0:895], norm2_mem [0:895];
    reg [15:0] gate_mem [0:4863], up_mem [0:4863], silu_mem [0:4863];
    reg [15:0] down_mem [0:895], probability_mem [0:127], score_mem [0:127];
    reg score_causal_mem [0:127];
    reg score_cache_miss_mem [0:127];
    reg score_invalid_mem [0:127];

    reg trace_valid_q;
    reg [4:0] trace_stage_q;
    reg [12:0] trace_index_q;
    reg [15:0] trace_f16_q;
    reg [6:0] trace_position_q;
    reg done_valid_q;
    reg [31:0] cycle_q, stall_q, done_cycle_q, done_stall_q;
    reg done_slot_q;
    reg [6:0] done_position_q;

    function automatic known1; input value; begin
        known1 = (value === 1'b0) || (value === 1'b1);
    end endfunction
    function automatic known2; input [1:0] value; begin
        known2 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known3; input [2:0] value; begin
        known3 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known4; input [3:0] value; begin
        known4 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known6; input [5:0] value; begin
        known6 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known13; input [12:0] value; begin
        known13 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known15; input [14:0] value; begin
        known15 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known16; input [15:0] value; begin
        known16 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known32; input [31:0] value; begin
        known32 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known46; input [45:0] value; begin
        known46 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction
    function automatic known102; input [101:0] value; begin
        known102 = ((^value === 1'b0) || (^value === 1'b1));
    end endfunction

    wire idle_w = state_q == S_IDLE;
    wire final_mode_w = (state_q == S_R2_IN) || (state_q == S_R2_OUT);
    wire trace_free_w = !trace_valid_q || (trace_valid_q && trace_ready_i &&
                                           known1(trace_ready_i));
    wire active_w = !idle_w;
    wire fault_detect_w;
    wire controller_healthy_w = !fault_q && !fault_detect_w;
    wire start_slot_valid_w =
        known2(start_cache_slot_i) && !start_cache_slot_i[1];
    wire start_position_valid_w =
        known15(start_position_i) &&
        (start_position_i[14:7] == 8'd0);
    wire [7:0] start_context_len_w =
        start_cache_slot_i[0] ? context_len_q[1] : context_len_q[0];
    wire [12:0] load_act_index_ext_w = {3'd0, load_act_q};
    wire [12:0] load_n1_index_ext_w = {3'd0, load_n1_q};
    wire [12:0] load_n2_index_ext_w = {3'd0, load_n2_q};
    assign busy_o = active_w;
    assign phase_o = state_q;
    assign projection_kind_o = psel_q;
    assign start_ready_o = rst_ni && !clear_i && controller_healthy_w &&
        idle_w && !done_valid_q &&
        activation_loaded_q && n1_loaded_q && n2_loaded_q &&
        start_slot_valid_w && start_position_valid_w &&
        (start_context_len_w == {1'b0, start_position_i[6:0]});
    assign load_ready_o = rst_ni && !clear_i && controller_healthy_w &&
        idle_w && !done_valid_q &&
        known2(load_kind_i) && known13(load_index_i) && known16(load_f16_i) &&
        ((load_kind_i == 2'd0 &&
          load_index_i == load_act_index_ext_w &&
          load_index_i < 13'd896) ||
         (load_kind_i == 2'd1 &&
          load_index_i == load_n1_index_ext_w &&
          load_index_i < 13'd896) ||
         (load_kind_i == 2'd2 &&
          load_index_i == load_n2_index_ext_w &&
          load_index_i < 13'd896));
    assign done_valid_o = done_valid_q && controller_healthy_w;
    assign done_cache_slot_o = {1'b0, done_slot_q};
    assign done_position_o = {8'd0, done_position_q};
    assign done_cycles_o = done_cycle_q;
    assign done_stall_cycles_o = done_stall_q;

    /* The four engines share the external pull bus; exactly one is started. */
    wire b_start_ready_w, b_meta_ready_w, b_pair_ready_w, b_bias_ready_w;
    wire b_out_valid_w, o_start_ready_w, o_meta_ready_w, o_pair_ready_w;
    wire o_out_valid_w, f_start_ready_w, f_meta_ready_w, f_pair_ready_w;
    wire f_out_valid_w, d_start_ready_w, d_meta_ready_w, d_pair_ready_w;
    wire d_out_valid_w;
    wire [12:0] b_meta_out_w,b_pair_in_w,b_pair_out_w,b_bias_out_w,b_out_ch_w;
    wire [12:0] o_meta_out_w,o_pair_in_w,o_pair_out_w,o_bias_out_w,o_out_ch_w;
    wire [12:0] f_meta_out_w,f_pair_in_w,f_pair_out_w,f_bias_out_w,f_out_ch_w;
    wire [12:0] d_meta_out_w,d_pair_in_w,d_pair_out_w,d_bias_out_w,d_out_ch_w;
    wire [5:0] b_meta_grp_w,b_pair_grp_w,o_meta_grp_w,o_pair_grp_w,
               f_meta_grp_w,f_pair_grp_w,d_meta_grp_w,d_pair_grp_w;
    wire [9:0] b_meta_word_w,b_pair_word_w,o_meta_word_w,o_pair_word_w,
               f_meta_word_w,f_pair_word_w,d_meta_word_w,d_pair_word_w;
    wire [2:0] b_meta_lane_w,b_pair_lane_w,o_meta_lane_w,o_pair_lane_w,
               f_meta_lane_w,f_pair_lane_w,d_meta_lane_w,d_pair_lane_w;
    wire [15:0] b_out_f16_w,o_out_f16_w,f_out_f16_w,d_out_f16_w;
    wire signed [101:0] b_acc_w,o_acc_w,f_acc_w,d_acc_w;
    wire b_invalid_w,b_saturation_w,b_busy_w;
    wire o_bias_ready_w,o_invalid_w,o_saturation_w,o_busy_w;
    wire f_bias_ready_w,f_invalid_w,f_saturation_w,f_busy_w;
    wire d_bias_ready_w,d_invalid_w,d_saturation_w,d_busy_w;
    wire b_pair_address_valid_w =
        known13(b_pair_in_w) && (b_pair_in_w < 13'd896);
    wire o_pair_address_valid_w =
        known13(o_pair_in_w) && (o_pair_in_w < 13'd896);
    wire f_pair_address_valid_w =
        known13(f_pair_in_w) && (f_pair_in_w < 13'd896);
    wire d_pair_address_valid_w =
        known13(d_pair_in_w) && (d_pair_in_w < 13'd4864);
    wire [15:0] b_activation_w =
        b_pair_address_valid_w ? norm1_mem[b_pair_in_w[9:0]] : 16'd0;
    wire [15:0] o_activation_w =
        o_pair_address_valid_w ? attention_mem[o_pair_in_w[9:0]] : 16'd0;
    wire [15:0] f_activation_w =
        f_pair_address_valid_w ? norm2_mem[f_pair_in_w[9:0]] : 16'd0;
    wire [15:0] d_activation_w =
        d_pair_address_valid_w ? silu_mem[d_pair_in_w[12:0]] : 16'd0;
    wire p_b_w = (psel_q==PK_Q)||(psel_q==PK_K)||(psel_q==PK_V);
    wire p_o_w = psel_q==PK_O, p_f_w=(psel_q==PK_GATE)||(psel_q==PK_UP);
    wire p_d_w = psel_q==PK_DOWN;
    wire p_start_w = state_q==S_P_START;
    wire b_output_address_valid_w = known13(b_out_ch_w) &&
        (((psel_q == PK_Q) && (b_out_ch_w < 13'd896)) ||
         (((psel_q == PK_K) || (psel_q == PK_V)) &&
          (b_out_ch_w < 13'd128)));
    wire o_output_address_valid_w =
        known13(o_out_ch_w) && (o_out_ch_w < 13'd896);
    wire f_output_address_valid_w =
        known13(f_out_ch_w) && (f_out_ch_w < 13'd4864);
    wire d_output_address_valid_w =
        known13(d_out_ch_w) && (d_out_ch_w < 13'd896);
    wire selected_pair_address_valid_w =
        p_b_w ? b_pair_address_valid_w :
        p_o_w ? o_pair_address_valid_w :
        p_f_w ? f_pair_address_valid_w :
        p_d_w ? d_pair_address_valid_w : 1'b0;
    wire selected_output_address_valid_w =
        p_b_w ? b_output_address_valid_w :
        p_o_w ? o_output_address_valid_w :
        p_f_w ? f_output_address_valid_w :
        p_d_w ? d_output_address_valid_w : 1'b0;
    wire selected_projection_out_valid_w =
        p_b_w ? b_out_valid_w : p_o_w ? o_out_valid_w :
        p_f_w ? f_out_valid_w : p_d_w ? d_out_valid_w : 1'b0;
    wire b_result_ok_w =
        known13(b_out_ch_w) && known16(b_out_f16_w) && known102(b_acc_w) &&
        known1(b_invalid_w) && known1(b_saturation_w) && known1(b_busy_w) &&
        (b_out_ch_w == projection_output_index_q) &&
        !b_invalid_w && !b_saturation_w && b_busy_w;
    wire o_result_ok_w =
        known13(o_out_ch_w) && known16(o_out_f16_w) && known102(o_acc_w) &&
        known1(o_invalid_w) && known1(o_saturation_w) && known1(o_busy_w) &&
        (o_out_ch_w == projection_output_index_q) &&
        !o_invalid_w && !o_saturation_w && o_busy_w;
    wire f_result_ok_w =
        known13(f_out_ch_w) && known16(f_out_f16_w) && known102(f_acc_w) &&
        known1(f_invalid_w) && known1(f_saturation_w) && known1(f_busy_w) &&
        (f_out_ch_w == projection_output_index_q) &&
        !f_invalid_w && !f_saturation_w && f_busy_w;
    wire d_result_ok_w =
        known13(d_out_ch_w) && known16(d_out_f16_w) && known102(d_acc_w) &&
        known1(d_invalid_w) && known1(d_saturation_w) && known1(d_busy_w) &&
        (d_out_ch_w == projection_output_index_q) &&
        !d_invalid_w && !d_saturation_w && d_busy_w;
    wire selected_projection_result_ok_w =
        p_b_w ? b_result_ok_w : p_o_w ? o_result_ok_w :
        p_f_w ? f_result_ok_w : p_d_w ? d_result_ok_w : 1'b0;
    wire p_output_ready_w =
        controller_healthy_w && trace_free_w && !final_mode_w &&
        selected_output_address_valid_w &&
        (!selected_projection_out_valid_w ||
         selected_projection_result_ok_w);
    wire meta_payload_known_w = known32(projection_qzeros_i) &&
                                  known16(projection_scale_f16_i);
    wire pair_payload_known_w = known32(projection_qweight_i);
    wire projection_input_stall_w =
        (projection_meta_ready_o && !projection_meta_valid_i) ||
        (projection_pair_ready_o && !projection_pair_valid_i) ||
        (projection_bias_ready_o && !projection_bias_valid_i);
    assign projection_meta_ready_o = controller_healthy_w &&
        (p_b_w ? (b_meta_ready_w && meta_payload_known_w) :
        p_o_w ? (o_meta_ready_w && meta_payload_known_w) :
        p_f_w ? (f_meta_ready_w && meta_payload_known_w) :
        p_d_w ? (d_meta_ready_w && meta_payload_known_w) : 1'b0);
    assign projection_pair_ready_o = controller_healthy_w &&
        selected_pair_address_valid_w &&
        (p_b_w ? (b_pair_ready_w && pair_payload_known_w) :
         p_o_w ? (o_pair_ready_w && pair_payload_known_w) :
         p_f_w ? (f_pair_ready_w && pair_payload_known_w) :
         p_d_w ? (d_pair_ready_w && pair_payload_known_w) : 1'b0);
    assign projection_bias_ready_o = controller_healthy_w &&
                                      p_b_w && b_bias_ready_w &&
                                      known16(projection_bias_f16_i);
    assign projection_meta_output_channel_o = p_b_w ? b_meta_out_w :
        p_o_w ? o_meta_out_w : p_f_w ? f_meta_out_w : d_meta_out_w;
    assign projection_meta_group_o = p_b_w ? b_meta_grp_w :
        p_o_w ? o_meta_grp_w : p_f_w ? f_meta_grp_w : d_meta_grp_w;
    assign projection_meta_word_o = p_b_w ? b_meta_word_w :
        p_o_w ? o_meta_word_w : p_f_w ? f_meta_word_w : d_meta_word_w;
    assign projection_meta_lane_o = p_b_w ? b_meta_lane_w :
        p_o_w ? o_meta_lane_w : p_f_w ? f_meta_lane_w : d_meta_lane_w;
    assign projection_pair_input_o = p_b_w ? b_pair_in_w :
        p_o_w ? o_pair_in_w : p_f_w ? f_pair_in_w : d_pair_in_w;
    assign projection_pair_output_o = p_b_w ? b_pair_out_w :
        p_o_w ? o_pair_out_w : p_f_w ? f_pair_out_w : d_pair_out_w;
    assign projection_pair_group_o = p_b_w ? b_pair_grp_w :
        p_o_w ? o_pair_grp_w : p_f_w ? f_pair_grp_w : d_pair_grp_w;
    assign projection_pair_word_o = p_b_w ? b_pair_word_w :
        p_o_w ? o_pair_word_w : p_f_w ? f_pair_word_w : d_pair_word_w;
    assign projection_pair_lane_o = p_b_w ? b_pair_lane_w :
        p_o_w ? o_pair_lane_w : p_f_w ? f_pair_lane_w : d_pair_lane_w;
    assign projection_bias_output_channel_o = b_bias_out_w;

    ace3_awq_w4a16_projection_engine #(.IN_FEATURES(896),.OUT_FEATURES(896),.BIAS_ENABLE(1)) p_bias (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),
      .start_valid_i(p_start_w&&p_b_w&&controller_healthy_w),.start_ready_o(b_start_ready_w),
      .first_output_channel_i(13'd0),.output_count_i((psel_q==PK_Q)?13'd896:13'd128),
      .meta_valid_i(projection_meta_valid_i&&p_b_w&&meta_payload_known_w&&controller_healthy_w),.meta_ready_o(b_meta_ready_w),
      .meta_output_channel_o(b_meta_out_w),.meta_group_index_o(b_meta_grp_w),.meta_output_word_o(b_meta_word_w),.meta_logical_lane_o(b_meta_lane_w),
      .qzeros_i(projection_qzeros_i),.scale_f16_i(projection_scale_f16_i),
      .pair_valid_i(projection_pair_valid_i&&p_b_w&&pair_payload_known_w&&b_pair_address_valid_w&&controller_healthy_w),.pair_ready_o(b_pair_ready_w),
      .pair_input_index_o(b_pair_in_w),.pair_output_channel_o(b_pair_out_w),.pair_group_index_o(b_pair_grp_w),.pair_output_word_o(b_pair_word_w),.pair_logical_lane_o(b_pair_lane_w),
      .activation_f16_i(b_activation_w),.qweight_i(projection_qweight_i),
      .bias_valid_i(projection_bias_valid_i&&p_b_w&&known16(projection_bias_f16_i)&&controller_healthy_w),.bias_ready_o(b_bias_ready_w),.bias_output_channel_o(b_bias_out_w),.bias_f16_i(projection_bias_f16_i),
      .out_valid_o(b_out_valid_w),.out_ready_i(p_output_ready_w&&p_b_w),.out_channel_o(b_out_ch_w),.out_f16_o(b_out_f16_w),.acc_q53_48_o(b_acc_w),.invalid_operand_o(b_invalid_w),.saturation_o(b_saturation_w),.busy_o(b_busy_w));
    ace3_awq_w4a16_projection_engine #(.IN_FEATURES(896),.OUT_FEATURES(896),.BIAS_ENABLE(0)) p_out (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(p_start_w&&p_o_w&&controller_healthy_w),.start_ready_o(o_start_ready_w),.first_output_channel_i(13'd0),.output_count_i(13'd896),
      .meta_valid_i(projection_meta_valid_i&&p_o_w&&meta_payload_known_w&&controller_healthy_w),.meta_ready_o(o_meta_ready_w),.meta_output_channel_o(o_meta_out_w),.meta_group_index_o(o_meta_grp_w),.meta_output_word_o(o_meta_word_w),.meta_logical_lane_o(o_meta_lane_w),.qzeros_i(projection_qzeros_i),.scale_f16_i(projection_scale_f16_i),
      .pair_valid_i(projection_pair_valid_i&&p_o_w&&pair_payload_known_w&&o_pair_address_valid_w&&controller_healthy_w),.pair_ready_o(o_pair_ready_w),.pair_input_index_o(o_pair_in_w),.pair_output_channel_o(o_pair_out_w),.pair_group_index_o(o_pair_grp_w),.pair_output_word_o(o_pair_word_w),.pair_logical_lane_o(o_pair_lane_w),.activation_f16_i(o_activation_w),.qweight_i(projection_qweight_i),
      .bias_valid_i(1'b0),.bias_ready_o(o_bias_ready_w),.bias_output_channel_o(o_bias_out_w),.bias_f16_i(16'd0),.out_valid_o(o_out_valid_w),.out_ready_i(p_output_ready_w&&p_o_w),.out_channel_o(o_out_ch_w),.out_f16_o(o_out_f16_w),.acc_q53_48_o(o_acc_w),.invalid_operand_o(o_invalid_w),.saturation_o(o_saturation_w),.busy_o(o_busy_w));
    ace3_awq_w4a16_projection_engine #(.IN_FEATURES(896),.OUT_FEATURES(4864),.BIAS_ENABLE(0)) p_ffn (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(p_start_w&&p_f_w&&controller_healthy_w),.start_ready_o(f_start_ready_w),.first_output_channel_i(13'd0),.output_count_i(13'd4864),
      .meta_valid_i(projection_meta_valid_i&&p_f_w&&meta_payload_known_w&&controller_healthy_w),.meta_ready_o(f_meta_ready_w),.meta_output_channel_o(f_meta_out_w),.meta_group_index_o(f_meta_grp_w),.meta_output_word_o(f_meta_word_w),.meta_logical_lane_o(f_meta_lane_w),.qzeros_i(projection_qzeros_i),.scale_f16_i(projection_scale_f16_i),
      .pair_valid_i(projection_pair_valid_i&&p_f_w&&pair_payload_known_w&&f_pair_address_valid_w&&controller_healthy_w),.pair_ready_o(f_pair_ready_w),.pair_input_index_o(f_pair_in_w),.pair_output_channel_o(f_pair_out_w),.pair_group_index_o(f_pair_grp_w),.pair_output_word_o(f_pair_word_w),.pair_logical_lane_o(f_pair_lane_w),.activation_f16_i(f_activation_w),.qweight_i(projection_qweight_i),
      .bias_valid_i(1'b0),.bias_ready_o(f_bias_ready_w),.bias_output_channel_o(f_bias_out_w),.bias_f16_i(16'd0),.out_valid_o(f_out_valid_w),.out_ready_i(p_output_ready_w&&p_f_w),.out_channel_o(f_out_ch_w),.out_f16_o(f_out_f16_w),.acc_q53_48_o(f_acc_w),.invalid_operand_o(f_invalid_w),.saturation_o(f_saturation_w),.busy_o(f_busy_w));
    ace3_awq_w4a16_projection_engine #(.IN_FEATURES(4864),.OUT_FEATURES(896),.BIAS_ENABLE(0)) p_down (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(p_start_w&&p_d_w&&controller_healthy_w),.start_ready_o(d_start_ready_w),.first_output_channel_i(13'd0),.output_count_i(13'd896),
      .meta_valid_i(projection_meta_valid_i&&p_d_w&&meta_payload_known_w&&controller_healthy_w),.meta_ready_o(d_meta_ready_w),.meta_output_channel_o(d_meta_out_w),.meta_group_index_o(d_meta_grp_w),.meta_output_word_o(d_meta_word_w),.meta_logical_lane_o(d_meta_lane_w),.qzeros_i(projection_qzeros_i),.scale_f16_i(projection_scale_f16_i),
      .pair_valid_i(projection_pair_valid_i&&p_d_w&&pair_payload_known_w&&d_pair_address_valid_w&&controller_healthy_w),.pair_ready_o(d_pair_ready_w),.pair_input_index_o(d_pair_in_w),.pair_output_channel_o(d_pair_out_w),.pair_group_index_o(d_pair_grp_w),.pair_output_word_o(d_pair_word_w),.pair_logical_lane_o(d_pair_lane_w),.activation_f16_i(d_activation_w),.qweight_i(projection_qweight_i),
      .bias_valid_i(1'b0),.bias_ready_o(d_bias_ready_w),.bias_output_channel_o(d_bias_out_w),.bias_f16_i(16'd0),.out_valid_o(d_out_valid_w),.out_ready_i(p_output_ready_w&&p_d_w),.out_channel_o(d_out_ch_w),.out_f16_o(d_out_f16_w),.acc_q53_48_o(d_acc_w),.invalid_operand_o(d_invalid_w),.saturation_o(d_saturation_w),.busy_o(d_busy_w));

    /* Vector arithmetic engines.  Each is only driven in its owning state. */
    wire n1_start_ready_w,n1_in_ready_w,n1_out_valid_w,n2_start_ready_w,n2_in_ready_w,n2_out_valid_w;
    wire [15:0] n1_out_w,n2_out_w; wire [12:0] n1_idx_w,n2_idx_w; wire n1_last_w,n2_last_w;
    wire n1_invalid_w,n1_saturation_w,n1_busy_w;
    wire n2_invalid_w,n2_saturation_w,n2_busy_w;
    wire [45:0] n1_rms_w,n2_rms_w;
    wire n1_index_valid_w = known13(n1_idx_w) && (n1_idx_w < 13'd896);
    wire n2_index_valid_w = known13(n2_idx_w) && (n2_idx_w < 13'd896);
    wire n1_result_ok_w = n1_index_valid_w && known16(n1_out_w) &&
        known1(n1_last_w) && known1(n1_invalid_w) &&
        known1(n1_saturation_w) && known1(n1_busy_w) &&
        known46(n1_rms_w) && (n1_last_w == (n1_idx_w == 13'd895)) &&
        !n1_invalid_w && !n1_saturation_w && n1_busy_w;
    wire n2_result_ok_w = n2_index_valid_w && known16(n2_out_w) &&
        known1(n2_last_w) && known1(n2_invalid_w) &&
        known1(n2_saturation_w) && known1(n2_busy_w) &&
        known46(n2_rms_w) && (n2_last_w == (n2_idx_w == 13'd895)) &&
        !n2_invalid_w && !n2_saturation_w && n2_busy_w;
    ace3_fp16_rmsnorm_core #(.HIDDEN_SIZE(896)) norm1 (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(state_q==S_N1_START),.start_ready_o(n1_start_ready_w),.element_count_i(13'd896),
      .in_valid_i(state_q==S_N1_IN),.in_ready_o(n1_in_ready_w),.activation_f16_i(activation_mem[hidden_index_q]),.weight_f16_i(norm1_weight_mem[hidden_index_q]),
      .out_valid_o(n1_out_valid_w),.out_ready_i(controller_healthy_w&&(state_q==S_N1_OUT)&&trace_free_w&&(!n1_out_valid_w||n1_result_ok_w)),.out_f16_o(n1_out_w),.out_index_o(n1_idx_w),.out_last_o(n1_last_w),.invalid_operand_o(n1_invalid_w),.saturation_o(n1_saturation_w),.busy_o(n1_busy_w),.rms_q24_o(n1_rms_w));
    ace3_fp16_rmsnorm_core #(.HIDDEN_SIZE(896)) norm2 (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(state_q==S_N2_START),.start_ready_o(n2_start_ready_w),.element_count_i(13'd896),
      .in_valid_i(state_q==S_N2_IN),.in_ready_o(n2_in_ready_w),.activation_f16_i(res1_mem[hidden_index_q]),.weight_f16_i(norm2_weight_mem[hidden_index_q]),
      .out_valid_o(n2_out_valid_w),.out_ready_i(controller_healthy_w&&(state_q==S_N2_OUT)&&trace_free_w&&(!n2_out_valid_w||n2_result_ok_w)),.out_f16_o(n2_out_w),.out_index_o(n2_idx_w),.out_last_o(n2_last_w),.invalid_operand_o(n2_invalid_w),.saturation_o(n2_saturation_w),.busy_o(n2_busy_w),.rms_q24_o(n2_rms_w));
    wire r1_start_ready_w,r1_in_ready_w,r1_out_valid_w,r2_start_ready_w,r2_in_ready_w,r2_out_valid_w;
    wire [15:0] r1_out_w,r2_out_w; wire [12:0] r1_idx_w,r2_idx_w; wire r1_last_w,r2_last_w;
    wire r1_invalid_w,r1_saturation_w,r1_busy_w;
    wire r2_invalid_w,r2_saturation_w,r2_busy_w;
    wire r1_index_valid_w = known13(r1_idx_w) && (r1_idx_w < 13'd896);
    wire r2_index_valid_w = known13(r2_idx_w) && (r2_idx_w < 13'd896);
    wire r1_result_ok_w = r1_index_valid_w && known16(r1_out_w) &&
        known1(r1_last_w) && known1(r1_invalid_w) &&
        known1(r1_saturation_w) && known1(r1_busy_w) &&
        (r1_last_w == (r1_idx_w == 13'd895)) &&
        !r1_invalid_w && !r1_saturation_w && r1_busy_w;
    wire r2_result_ok_w = r2_index_valid_w && known16(r2_out_w) &&
        known1(r2_last_w) && known1(r2_invalid_w) &&
        known1(r2_saturation_w) && known1(r2_busy_w) &&
        (r2_last_w == (r2_idx_w == 13'd895)) &&
        !r2_invalid_w && !r2_saturation_w && r2_busy_w;
    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) res1 (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(state_q==S_R1_START),.start_ready_o(r1_start_ready_w),.element_count_i(13'd896),
      .in_valid_i(state_q==S_R1_IN),.in_ready_o(r1_in_ready_w),.projection_f16_i(o_mem[hidden_index_q]),.residual_f16_i(activation_mem[hidden_index_q]),
      .out_valid_o(r1_out_valid_w),.out_ready_i(controller_healthy_w&&((state_q==S_R1_IN)||(state_q==S_R1_OUT))&&trace_free_w&&(!r1_out_valid_w||r1_result_ok_w)),.out_f16_o(r1_out_w),.out_index_o(r1_idx_w),.out_last_o(r1_last_w),.invalid_operand_o(r1_invalid_w),.saturation_o(r1_saturation_w),.busy_o(r1_busy_w));
    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) res2 (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i((state_q==S_R2_START)&&trace_free_w),.start_ready_o(r2_start_ready_w),.element_count_i(13'd896),
      .in_valid_i(state_q==S_R2_IN),.in_ready_o(r2_in_ready_w),.projection_f16_i(down_mem[hidden_index_q]),.residual_f16_i(res1_mem[hidden_index_q]),
      .out_valid_o(r2_out_valid_w),.out_ready_i(controller_healthy_w&&final_mode_w&&final_ready_i&&trace_ready_i&&known1(final_ready_i)&&known1(trace_ready_i)&&(!r2_out_valid_w||r2_result_ok_w)),.out_f16_o(r2_out_w),.out_index_o(r2_idx_w),.out_last_o(r2_last_w),.invalid_operand_o(r2_invalid_w),.saturation_o(r2_saturation_w),.busy_o(r2_busy_w));
    wire si_start_ready_w,si_in_ready_w,si_out_valid_w; wire [15:0] si_out_w; wire [12:0] si_idx_w; wire si_last_w;
    wire si_invalid_w,si_saturation_w,si_busy_w;
    wire si_index_valid_w = known13(si_idx_w) && (si_idx_w < 13'd4864);
    wire si_result_ok_w = si_index_valid_w && known16(si_out_w) &&
        known1(si_last_w) && known1(si_invalid_w) &&
        known1(si_saturation_w) && known1(si_busy_w) &&
        (si_idx_w == si_output_count_q) &&
        (si_last_w == (si_idx_w == 13'd4863)) &&
        !si_invalid_w && !si_saturation_w && si_busy_w;
    ace3_fp16_silu_gate_core #(
      .INTERMEDIATE_SIZE(4864),
      .ACCURATE_SIGMOID(ACCURATE_SILU)
    ) silu (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(state_q==S_SI_START),.start_ready_o(si_start_ready_w),.element_count_i(13'd4864),
      .in_valid_i(state_q==S_SI_IN),.in_ready_o(si_in_ready_w),.gate_f16_i(gate_mem[intermediate_index_q]),.up_f16_i(up_mem[intermediate_index_q]),
      .out_valid_o(si_out_valid_w),.out_ready_i(controller_healthy_w&&((state_q==S_SI_IN)||(state_q==S_SI_OUT))&&trace_free_w&&(!si_out_valid_w||si_result_ok_w)),.out_f16_o(si_out_w),.out_index_o(si_idx_w),.out_last_o(si_last_w),.invalid_operand_o(si_invalid_w),.saturation_o(si_saturation_w),.busy_o(si_busy_w));

    wire rope_in_ready_w,rope_out_valid_w; wire [15:0] rope_lo_w,rope_hi_w;
    wire rope_is_key_w,rope_invalid_w,rope_saturation_w;
    wire [3:0] rope_out_head_w;
    wire [4:0] rope_out_pair_w;
    wire [14:0] rope_out_position_w;
    wire rope_key_w = (state_q==S_RK_REQ)||(state_q==S_RK_LO)||(state_q==S_RK_HI);
    wire [3:0] rope_head_w = head_q;
    wire [4:0] rope_pair_index_w = dim_q[4:0];
    wire [9:0] q_flat_index_w = {head_q, dim_q};
    wire [9:0] q_flat_high_index_w = q_flat_index_w + 10'd32;
    wire [6:0] kv_flat_index_w = {head_q[0], dim_q};
    wire [6:0] kv_flat_high_index_w = kv_flat_index_w + 7'd32;
    wire q_flat_valid_w =
        known4(head_q) && known6(dim_q) &&
        (head_q < 4'd14) && (q_flat_index_w <= 10'd895);
    wire kv_flat_valid_w =
        known4(head_q) && known6(dim_q) &&
        (head_q < 4'd2);
    wire q_pair_valid_w =
        q_flat_valid_w && (dim_q <= 6'd31) &&
        (q_flat_high_index_w <= 10'd895);
    wire kv_pair_valid_w =
        kv_flat_valid_w && (dim_q <= 6'd31);
    wire [7:0] active_context_count_w =
        {1'b0, token_position_q} + 8'd1;
    wire rope_result_ok_w =
        known1(rope_is_key_w) && known4(rope_out_head_w) &&
        known6({1'b0,rope_out_pair_w}) && known15(rope_out_position_w) &&
        known16(rope_lo_w) && known16(rope_hi_w) &&
        known1(rope_invalid_w) && known1(rope_saturation_w) &&
        (rope_is_key_w == rope_key_w) &&
        (rope_out_head_w == rope_head_w) &&
        (rope_out_pair_w == rope_pair_index_w) &&
        (rope_out_position_w == {8'd0,token_position_q}) &&
        !rope_invalid_w && !rope_saturation_w;
    assign rope_position_o = {8'd0, token_position_q};
    assign rope_pair_o = rope_pair_index_w;
    assign rope_ready_o = controller_healthy_w &&
                          ((state_q==S_RQ_REQ)||(state_q==S_RK_REQ)) &&
                          rope_in_ready_w && known16(rope_cos_f16_i) &&
                          known16(rope_sin_f16_i) &&
                          (rope_key_w ? kv_pair_valid_w : q_pair_valid_w);
    ace3_qwen2_rope_pair rope (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),
      .in_valid_i(rope_valid_i&&rope_ready_o),.in_ready_o(rope_in_ready_w),
      .is_key_i(rope_key_w),.head_index_i(rope_head_w),.pair_index_i(rope_pair_index_w),.position_i({8'd0,token_position_q}),
      .low_f16_i(rope_key_w?k_mem[kv_flat_index_w]:q_mem[q_flat_index_w]),
      .high_f16_i(rope_key_w?k_mem[kv_flat_high_index_w]:q_mem[q_flat_high_index_w]),
      .cos_f16_i(rope_cos_f16_i),.sin_f16_i(rope_sin_f16_i),
      .out_valid_o(rope_out_valid_w),.out_ready_i(controller_healthy_w&&((state_q==S_RQ_HI)||(state_q==S_RK_HI))&&trace_free_w&&(!rope_out_valid_w||rope_result_ok_w)),
      .is_key_o(rope_is_key_w),.head_index_o(rope_out_head_w),.pair_index_o(rope_out_pair_w),.position_o(rope_out_position_w),.low_f16_o(rope_lo_w),.high_f16_o(rope_hi_w),.invalid_operand_o(rope_invalid_w),.saturation_o(rope_saturation_w));

    wire cache_wr_ready_w,cache_rd_ready_w,cache_out_valid_w,cache_hit_w;
    wire [15:0] cache_k_w,cache_v_w;
    wire [1:0] cache_slot_w;
    wire [14:0] cache_position_w;
    wire [3:0] cache_head_w;
    wire [5:0] cache_dimension_w;
    wire cache_write_state_w=(state_q==S_CW_LO);
    wire cache_read_state_w=(state_q==S_SC_READ)||(state_q==S_AV_READ);
    wire [3:0] mapped_kv_head_w = (head_q<4'd7)?4'd0:4'd1;
    wire cache_result_ok_w =
        known1(cache_hit_w) && known2(cache_slot_w) &&
        known15(cache_position_w) && known4(cache_head_w) &&
        known6(cache_dimension_w) && known16(cache_k_w) &&
        known16(cache_v_w) &&
        (cache_slot_w == {1'b0,token_slot_q}) &&
        (cache_position_w == {8'd0,key_position_q}) &&
        (cache_head_w == mapped_kv_head_w) &&
        (cache_dimension_w == dim_q);
    ace3_fp16_kv_cache #(.CACHE_SLOTS(2),.MAX_TOKENS(128),.KV_HEADS(2),.HEAD_DIM(64)) cache (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),
      .write_valid_i(cache_write_state_w&&kv_flat_valid_w&&controller_healthy_w),.write_ready_o(cache_wr_ready_w),.write_cache_slot_i({1'b0,token_slot_q}),.write_position_i({8'd0,token_position_q}),.write_head_i(head_q),.write_dimension_i(dim_q),.write_k_f16_i(rk_mem[kv_flat_index_w]),.write_v_f16_i(v_mem[kv_flat_index_w]),
      .read_valid_i(cache_read_state_w&&controller_healthy_w),.read_ready_o(cache_rd_ready_w),.read_cache_slot_i({1'b0,token_slot_q}),.read_position_i({8'd0,key_position_q}),.read_head_i(mapped_kv_head_w),.read_dimension_i(dim_q),
      .out_valid_o(cache_out_valid_w),.out_ready_i(controller_healthy_w&&((state_q==S_SC_TERM)||(state_q==S_AV_TERM))&&(!cache_out_valid_w||cache_result_ok_w)&&
        ((state_q==S_SC_TERM)?sc_pair_ready_w:av_term_ready_w)),.out_hit_o(cache_hit_w),.out_cache_slot_o(cache_slot_w),.out_position_o(cache_position_w),.out_head_o(cache_head_w),.out_dimension_o(cache_dimension_w),.out_k_f16_o(cache_k_w),.out_v_f16_o(cache_v_w));

    wire sc_start_ready_w,sc_pair_ready_w,sc_out_valid_w; wire [15:0] sc_out_w;
    wire [3:0] sc_query_head_w,sc_key_head_w;
    wire [14:0] sc_query_position_w,sc_key_position_w;
    wire sc_causal_w,sc_cache_miss_w,sc_invalid_w,sc_saturation_w,sc_busy_w;
    wire sc_result_ok_w =
       known16(sc_out_w) && known4(sc_query_head_w) &&
       known4(sc_key_head_w) && known15(sc_query_position_w) &&
       known15(sc_key_position_w) && known1(sc_causal_w) &&
       known1(sc_cache_miss_w) && known1(sc_invalid_w) &&
       known1(sc_saturation_w) && known1(sc_busy_w) &&
       (sc_query_head_w == head_q) &&
       (sc_key_head_w == mapped_kv_head_w) &&
       (sc_query_position_w == {8'd0,token_position_q}) &&
       (sc_key_position_w == {8'd0,key_position_q}) &&
       (sc_causal_w == (key_position_q <= token_position_q)) &&
       sc_busy_w;
    ace3_attention_score_core score (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i((state_q==S_SC_START)&&q_flat_valid_w&&controller_healthy_w),.start_ready_o(sc_start_ready_w),.query_head_i(head_q),.key_head_i(mapped_kv_head_w),.query_position_i({8'd0,token_position_q}),.key_position_i({8'd0,key_position_q}),
      .pair_valid_i((state_q==S_SC_TERM)&&cache_out_valid_w&&cache_result_ok_w&&controller_healthy_w),.pair_ready_o(sc_pair_ready_w),.q_f16_i(q_flat_valid_w?q_mem[q_flat_index_w]:16'd0),.k_f16_i(cache_k_w),.cache_hit_i(cache_hit_w),
      .out_valid_o(sc_out_valid_w),.out_ready_i(controller_healthy_w&&(state_q==S_SC_OUT)&&trace_free_w&&(!sc_out_valid_w||sc_result_ok_w)),.score_f16_o(sc_out_w),.query_head_o(sc_query_head_w),.key_head_o(sc_key_head_w),.query_position_o(sc_query_position_w),.key_position_o(sc_key_position_w),.causal_o(sc_causal_w),.cache_miss_o(sc_cache_miss_w),.invalid_operand_o(sc_invalid_w),.saturation_o(sc_saturation_w),.busy_o(sc_busy_w));
    wire sm_start_ready_w,sm_score_ready_w,sm_out_valid_w; wire [15:0] sm_out_w;
    wire [3:0] sm_query_head_w;
    wire [14:0] sm_query_position_w,sm_key_position_w;
    wire [15:0] sm_index_w;
    wire sm_last_w,sm_row_error_w,sm_cache_miss_w,sm_invalid_w,sm_busy_w;
    wire sm_result_ok_w =
       known16(sm_out_w) && known4(sm_query_head_w) &&
       known15(sm_query_position_w) && known15(sm_key_position_w) &&
       known16(sm_index_w) && known1(sm_last_w) &&
       known1(sm_row_error_w) && known1(sm_cache_miss_w) &&
       known1(sm_invalid_w) && known1(sm_busy_w) &&
       (sm_query_head_w == head_q) &&
       (sm_query_position_w == {8'd0,token_position_q}) &&
       (sm_key_position_w == {8'd0,context_index_q}) &&
       (sm_index_w == {9'd0,context_index_q}) &&
       (sm_last_w == (context_index_q == token_position_q)) &&
       sm_busy_w;
    ace3_attention_softmax_core softmax (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i((state_q==S_SM_START)&&controller_healthy_w),.start_ready_o(sm_start_ready_w),.query_head_i(head_q),.query_position_i({8'd0,token_position_q}),.context_count_i({8'd0,active_context_count_w}),
      .score_valid_i((state_q==S_SM_IN)&&controller_healthy_w),.score_ready_o(sm_score_ready_w),.score_f16_i(score_mem[context_index_q]),.key_position_i({8'd0,context_index_q}),.causal_i(score_causal_mem[context_index_q]),.cache_miss_i(score_cache_miss_mem[context_index_q]),.invalid_operand_i(score_invalid_mem[context_index_q]),
      .out_valid_o(sm_out_valid_w),.out_ready_i(controller_healthy_w&&(state_q==S_SM_OUT)&&trace_free_w&&(!sm_out_valid_w||sm_result_ok_w)),.probability_f16_o(sm_out_w),.query_head_o(sm_query_head_w),.query_position_o(sm_query_position_w),.key_position_o(sm_key_position_w),.out_index_o(sm_index_w),.out_last_o(sm_last_w),.row_error_o(sm_row_error_w),.cache_miss_o(sm_cache_miss_w),.invalid_operand_o(sm_invalid_w),.busy_o(sm_busy_w));
    wire av_start_ready_w,av_term_ready_w,av_out_valid_w; wire [15:0] av_out_w;
    wire [3:0] av_query_head_w,av_value_head_w;
    wire [14:0] av_query_position_w;
    wire [5:0] av_dimension_w;
    wire av_row_error_w,av_cache_miss_w,av_invalid_w,av_saturation_w,av_busy_w;
    wire av_result_ok_w =
       known16(av_out_w) && known4(av_query_head_w) &&
       known4(av_value_head_w) && known15(av_query_position_w) &&
       known6(av_dimension_w) && known1(av_row_error_w) &&
       known1(av_cache_miss_w) && known1(av_invalid_w) &&
       known1(av_saturation_w) && known1(av_busy_w) &&
       (av_query_head_w == head_q) &&
       (av_value_head_w == mapped_kv_head_w) &&
       (av_query_position_w == {8'd0,token_position_q}) &&
       (av_dimension_w == dim_q) && !av_row_error_w &&
       !av_cache_miss_w && !av_invalid_w && !av_saturation_w &&
       av_busy_w;
    ace3_attention_value_core value (
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i((state_q==S_AV_START)&&controller_healthy_w),.start_ready_o(av_start_ready_w),.query_head_i(head_q),.value_head_i(mapped_kv_head_w),.query_position_i({8'd0,token_position_q}),.dimension_i(dim_q),.context_count_i({8'd0,active_context_count_w}),
      .term_valid_i((state_q==S_AV_TERM)&&cache_out_valid_w&&cache_result_ok_w&&controller_healthy_w),.term_ready_o(av_term_ready_w),.probability_f16_i(probability_mem[key_position_q]),.value_f16_i(cache_v_w),.value_hit_i(cache_hit_w),.row_error_i(softmax_row_error_q),
      .out_valid_o(av_out_valid_w),.out_ready_i(controller_healthy_w&&(state_q==S_AV_OUT)&&trace_free_w&&(!av_out_valid_w||av_result_ok_w)),.value_f16_o(av_out_w),.query_head_o(av_query_head_w),.value_head_o(av_value_head_w),.query_position_o(av_query_position_w),.dimension_o(av_dimension_w),.row_error_o(av_row_error_w),.cache_miss_o(av_cache_miss_w),.invalid_operand_o(av_invalid_w),.saturation_o(av_saturation_w),.busy_o(av_busy_w));

    wire disabled_bias_contract_ok_w =
       (o_bias_ready_w === 1'b0) && (f_bias_ready_w === 1'b0) &&
       (d_bias_ready_w === 1'b0) && known13(o_bias_out_w) &&
       known13(f_bias_out_w) && known13(d_bias_out_w) &&
       (o_bias_out_w < 13'd896) && (f_bias_out_w < 13'd4864) &&
       (d_bias_out_w < 13'd896);
    wire child_busy_fault_w =
       (b_busy_w !== ((state_q==S_P_RUN)&&p_b_w)) ||
       (o_busy_w !== ((state_q==S_P_RUN)&&p_o_w)) ||
       (f_busy_w !== ((state_q==S_P_RUN)&&p_f_w)) ||
       (d_busy_w !== ((state_q==S_P_RUN)&&p_d_w)) ||
       (n1_busy_w !== ((state_q==S_N1_IN)||(state_q==S_N1_OUT))) ||
       (n2_busy_w !== ((state_q==S_N2_IN)||(state_q==S_N2_OUT))) ||
       (r1_busy_w !== ((state_q==S_R1_IN)||(state_q==S_R1_OUT))) ||
       (r2_busy_w !== ((state_q==S_R2_IN)||(state_q==S_R2_OUT))) ||
       (si_busy_w !== ((state_q==S_SI_IN)||(state_q==S_SI_OUT))) ||
       (sc_busy_w !== ((state_q==S_SC_READ)||(state_q==S_SC_TERM)||
                       (state_q==S_SC_OUT))) ||
       (sm_busy_w !== ((state_q==S_SM_IN)||(state_q==S_SM_OUT))) ||
       (av_busy_w !== ((state_q==S_AV_READ)||(state_q==S_AV_TERM)||
                       (state_q==S_AV_OUT)));
    wire projection_result_fault_w =
       (state_q==S_P_RUN) && selected_projection_out_valid_w &&
       !selected_projection_result_ok_w;
    wire vector_result_fault_w =
       ((state_q==S_N1_OUT)&&n1_out_valid_w&&!n1_result_ok_w) ||
       ((state_q==S_N2_OUT)&&n2_out_valid_w&&!n2_result_ok_w) ||
       (((state_q==S_R1_IN)||(state_q==S_R1_OUT))&&
        r1_out_valid_w&&!r1_result_ok_w) ||
       (final_mode_w&&r2_out_valid_w&&!r2_result_ok_w) ||
       (((state_q==S_SI_IN)||(state_q==S_SI_OUT))&&
        si_out_valid_w&&!si_result_ok_w);
    wire rope_result_fault_w =
       ((state_q==S_RQ_LO)||(state_q==S_RQ_HI)||
        (state_q==S_RK_LO)||(state_q==S_RK_HI)) &&
       rope_out_valid_w && !rope_result_ok_w;
    wire cache_result_fault_w =
       ((state_q==S_SC_TERM)||(state_q==S_AV_TERM)) &&
       cache_out_valid_w && !cache_result_ok_w;
    wire attention_result_fault_w =
       ((state_q==S_SC_OUT)&&sc_out_valid_w&&!sc_result_ok_w) ||
       ((state_q==S_SM_OUT)&&sm_out_valid_w&&!sm_result_ok_w) ||
       ((state_q==S_AV_OUT)&&av_out_valid_w&&!av_result_ok_w);
    assign fault_detect_w = rst_ni && !clear_i && !fault_q &&
       (!disabled_bias_contract_ok_w || child_busy_fault_w ||
        projection_result_fault_w || vector_result_fault_w ||
        rope_result_fault_w || cache_result_fault_w ||
        attention_result_fault_w);

    assign final_valid_o = controller_healthy_w && final_mode_w &&
                          r2_out_valid_w && r2_result_ok_w;
    assign final_index_o = r2_idx_w;
    assign final_f16_o = r2_out_w;
    assign final_last_o = r2_last_w;
    assign trace_valid_o = controller_healthy_w &&
        (final_mode_w ? (r2_out_valid_w && r2_result_ok_w) : trace_valid_q);
    assign trace_stage_o = final_mode_w ? TRACE_RES2 : trace_stage_q;
    assign trace_index_o = final_mode_w ? r2_idx_w : trace_index_q;
    assign trace_f16_o = final_mode_w ? r2_out_w : trace_f16_q;
    assign trace_position_o =
        final_mode_w ? {8'd0, token_position_q} :
                       {8'd0, trace_position_q};

    always @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        state_q<=S_IDLE; psel_q<=PK_Q; hidden_index_q<=10'd0;
        intermediate_index_q<=13'd0; si_output_count_q<=13'd0;
        context_index_q<=7'd0;
        load_act_q<=10'd0; load_n1_q<=10'd0; load_n2_q<=10'd0;
        activation_loaded_q<=0; n1_loaded_q<=0; n2_loaded_q<=0; context_len_q[0]<=0; context_len_q[1]<=0;
        token_position_q<=7'd0; token_slot_q<=1'b0; head_q<=4'd0;
        dim_q<=6'd0; key_position_q<=7'd0;
        fault_q<=1'b0; projection_output_index_q<=13'd0;
        softmax_row_error_q<=1'b0;
        trace_valid_q<=0; trace_stage_q<=0; trace_index_q<=0; trace_f16_q<=0; trace_position_q<=0;
        done_valid_q<=0; cycle_q<=0; stall_q<=0; done_cycle_q<=0; done_stall_q<=0; done_slot_q<=0; done_position_q<=0;
      end else if (clear_i) begin
        state_q<=S_IDLE; psel_q<=PK_Q; hidden_index_q<=10'd0;
        intermediate_index_q<=13'd0; si_output_count_q<=13'd0;
        context_index_q<=7'd0;
        load_act_q<=10'd0; load_n1_q<=10'd0; load_n2_q<=10'd0;
        activation_loaded_q<=0; n1_loaded_q<=0; n2_loaded_q<=0; context_len_q[0]<=0; context_len_q[1]<=0;
        token_position_q<=7'd0; token_slot_q<=1'b0; head_q<=4'd0;
        dim_q<=6'd0; key_position_q<=7'd0;
        fault_q<=1'b0; projection_output_index_q<=13'd0;
        softmax_row_error_q<=1'b0;
        trace_valid_q<=0; trace_stage_q<=0; trace_index_q<=0;
        trace_f16_q<=0; trace_position_q<=0;
        done_valid_q<=0; cycle_q<=0; stall_q<=0; done_cycle_q<=0;
        done_stall_q<=0; done_slot_q<=0; done_position_q<=0;
      end else begin
        if (trace_valid_q && trace_ready_i && known1(trace_ready_i)) trace_valid_q<=0;
        if (done_valid_q && done_ready_i && known1(done_ready_i)) done_valid_q<=0;
        if (active_w) begin
          cycle_q<=cycle_q+32'd1;
          if ((trace_valid_q && (!known1(trace_ready_i)||!trace_ready_i)) ||
              (final_mode_w && r2_out_valid_w && (!known1(final_ready_i)||!final_ready_i ||
                                                    !known1(trace_ready_i)||!trace_ready_i)) ||
              ((state_q==S_P_RUN) && projection_input_stall_w) ||
              (((state_q==S_RQ_REQ)||(state_q==S_RK_REQ)) &&
               (!rope_valid_i || !known16(rope_cos_f16_i) ||
                !known16(rope_sin_f16_i))))
            stall_q<=stall_q+32'd1;
        end
        if (load_valid_i && load_ready_o) begin
          if (load_kind_i==0) begin activation_mem[load_act_q]<=load_f16_i;
            if (load_act_q==0) activation_loaded_q<=0;
            if(load_act_q==10'd895) begin load_act_q<=0; activation_loaded_q<=1; end else load_act_q<=load_act_q+10'd1; end
          else if(load_kind_i==1) begin norm1_weight_mem[load_n1_q]<=load_f16_i;
            if (load_n1_q==0) n1_loaded_q<=0;
            if(load_n1_q==10'd895) begin load_n1_q<=0; n1_loaded_q<=1; end else load_n1_q<=load_n1_q+10'd1; end
          else begin norm2_weight_mem[load_n2_q]<=load_f16_i;
            if (load_n2_q==0) n2_loaded_q<=0;
            if(load_n2_q==10'd895) begin load_n2_q<=0; n2_loaded_q<=1; end else load_n2_q<=load_n2_q+10'd1; end
        end
        if (fault_detect_w) begin
          fault_q<=1'b1;
          state_q<=S_FAULT;
          trace_valid_q<=1'b0;
          done_valid_q<=1'b0;
        end else case (state_q)
          S_IDLE: if(start_valid_i&&start_ready_o) begin token_slot_q<=start_cache_slot_i[0]; token_position_q<=start_position_i[6:0]; cycle_q<=0; stall_q<=0; hidden_index_q<=0; intermediate_index_q<=0; context_index_q<=0; state_q<=S_N1_START; activation_loaded_q<=0; end
          S_N1_START: if(n1_start_ready_w) begin hidden_index_q<=0; state_q<=S_N1_IN; end
          S_N1_IN: if(n1_in_ready_w) begin if(hidden_index_q==10'd895) begin hidden_index_q<=0; state_q<=S_N1_OUT; end else hidden_index_q<=hidden_index_q+10'd1; end
          S_N1_OUT: if(n1_out_valid_w&&trace_free_w&&n1_result_ok_w) begin norm1_mem[n1_idx_w[9:0]]<=n1_out_w; trace_valid_q<=1;trace_stage_q<=TRACE_NORM1;trace_index_q<=n1_idx_w;trace_f16_q<=n1_out_w;trace_position_q<=token_position_q;
            if(n1_last_w) begin psel_q<=PK_Q;state_q<=S_P_START;end end
          S_P_START: if((p_b_w&&b_start_ready_w)||(p_o_w&&o_start_ready_w)||(p_f_w&&f_start_ready_w)||(p_d_w&&d_start_ready_w)) begin projection_output_index_q<=13'd0;state_q<=S_P_RUN;end
          S_P_RUN: if(trace_free_w && selected_output_address_valid_w && selected_projection_out_valid_w && selected_projection_result_ok_w) begin
            projection_output_index_q<=projection_output_index_q+13'd1;
            trace_valid_q<=1; trace_position_q<=token_position_q;
            if(p_b_w) begin
              if(psel_q==PK_Q) begin q_mem[b_out_ch_w[9:0]]<=b_out_f16_w;trace_stage_q<=TRACE_Q;end
              else if(psel_q==PK_K) begin k_mem[b_out_ch_w[6:0]]<=b_out_f16_w;trace_stage_q<=TRACE_K;end
              else begin v_mem[b_out_ch_w[6:0]]<=b_out_f16_w;trace_stage_q<=TRACE_V;end
              trace_index_q<=b_out_ch_w;trace_f16_q<=b_out_f16_w;
              if((psel_q==PK_Q&&b_out_ch_w==13'd895)||(psel_q!=PK_Q&&b_out_ch_w==13'd127)) begin
                if(psel_q==PK_Q) begin head_q<=0;dim_q<=0;state_q<=S_RQ_REQ;end
                else if(psel_q==PK_K) begin psel_q<=PK_V;state_q<=S_P_START;end
                else begin head_q<=0;dim_q<=0;state_q<=S_RK_REQ;end
              end
            end else if(p_o_w) begin o_mem[o_out_ch_w[9:0]]<=o_out_f16_w;trace_stage_q<=TRACE_O;trace_index_q<=o_out_ch_w;trace_f16_q<=o_out_f16_w;if(o_out_ch_w==13'd895)state_q<=S_R1_START;end
            else if(p_f_w) begin
              if(psel_q==PK_GATE) begin gate_mem[f_out_ch_w[12:0]]<=f_out_f16_w;trace_stage_q<=TRACE_GATE;end else begin up_mem[f_out_ch_w[12:0]]<=f_out_f16_w;trace_stage_q<=TRACE_UP;end
              trace_index_q<=f_out_ch_w;trace_f16_q<=f_out_f16_w;if(f_out_ch_w==13'd4863)begin if(psel_q==PK_GATE)begin psel_q<=PK_UP;state_q<=S_P_START;end else state_q<=S_SI_START;end
            end else begin down_mem[d_out_ch_w[9:0]]<=d_out_f16_w;trace_stage_q<=TRACE_DOWN;trace_index_q<=d_out_ch_w;trace_f16_q<=d_out_f16_w;if(d_out_ch_w==13'd895)state_q<=S_R2_START;end
          end
          S_RQ_REQ: if(rope_valid_i&&rope_ready_o) state_q<=S_RQ_LO;
          S_RQ_LO: if(rope_out_valid_w&&trace_free_w&&q_pair_valid_w&&rope_result_ok_w) begin q_mem[q_flat_index_w]<=rope_lo_w;trace_valid_q<=1;trace_stage_q<=TRACE_RQ;trace_index_q<={3'd0,q_flat_index_w};trace_f16_q<=rope_lo_w;trace_position_q<=token_position_q;state_q<=S_RQ_HI;end
          S_RQ_HI: if(rope_out_valid_w&&trace_free_w&&q_pair_valid_w&&rope_result_ok_w) begin q_mem[q_flat_high_index_w]<=rope_hi_w;trace_valid_q<=1;trace_stage_q<=TRACE_RQ;trace_index_q<={3'd0,q_flat_high_index_w};trace_f16_q<=rope_hi_w;trace_position_q<=token_position_q;
            if(dim_q==6'd31)begin dim_q<=0;if(head_q==4'd13)begin psel_q<=PK_K;state_q<=S_P_START;end else begin head_q<=head_q+4'd1;state_q<=S_RQ_REQ;end end else begin dim_q<=dim_q+6'd1;state_q<=S_RQ_REQ;end end
          S_RK_REQ: if(rope_valid_i&&rope_ready_o) state_q<=S_RK_LO;
          S_RK_LO: if(rope_out_valid_w&&trace_free_w&&kv_pair_valid_w&&rope_result_ok_w) begin rk_mem[kv_flat_index_w]<=rope_lo_w;trace_valid_q<=1;trace_stage_q<=TRACE_RK;trace_index_q<={6'd0,kv_flat_index_w};trace_f16_q<=rope_lo_w;trace_position_q<=token_position_q;state_q<=S_RK_HI;end
          S_RK_HI: if(rope_out_valid_w&&trace_free_w&&kv_pair_valid_w&&rope_result_ok_w) begin rk_mem[kv_flat_high_index_w]<=rope_hi_w;trace_valid_q<=1;trace_stage_q<=TRACE_RK;trace_index_q<={6'd0,kv_flat_high_index_w};trace_f16_q<=rope_hi_w;trace_position_q<=token_position_q;
            if(dim_q==6'd31)begin dim_q<=0;if(head_q==4'd1)begin head_q<=0;state_q<=S_CW_LO;end else begin head_q<=head_q+4'd1;state_q<=S_RK_REQ;end end else begin dim_q<=dim_q+6'd1;state_q<=S_RK_REQ;end end
          S_CW_LO: if(cache_wr_ready_w&&trace_free_w&&kv_flat_valid_w) begin trace_valid_q<=1;trace_stage_q<=TRACE_CK;trace_index_q<={6'd0,kv_flat_index_w};trace_f16_q<=rk_mem[kv_flat_index_w];trace_position_q<=token_position_q;state_q<=S_CW_HI;end
          S_CW_HI: if(trace_free_w&&kv_flat_valid_w) begin trace_valid_q<=1;trace_stage_q<=TRACE_CV;trace_index_q<={6'd0,kv_flat_index_w};trace_f16_q<=v_mem[kv_flat_index_w];trace_position_q<=token_position_q;
            if(dim_q==6'd63)begin dim_q<=0;if(head_q==4'd1)begin head_q<=0;key_position_q<=0;state_q<=S_SC_START;end else head_q<=head_q+4'd1;end else dim_q<=dim_q+6'd1; if(!(dim_q==6'd63&&head_q==4'd1))state_q<=S_CW_LO; end
          S_SC_START: if(sc_start_ready_w) begin dim_q<=0;state_q<=S_SC_READ;end
          S_SC_READ: if(cache_rd_ready_w) state_q<=S_SC_TERM;
          S_SC_TERM: if(cache_out_valid_w&&cache_result_ok_w&&sc_pair_ready_w) begin if(dim_q==6'd63)state_q<=S_SC_OUT;else begin dim_q<=dim_q+6'd1;state_q<=S_SC_READ;end end
          S_SC_OUT: if(sc_out_valid_w&&trace_free_w&&sc_result_ok_w) begin score_mem[key_position_q]<=sc_out_w;score_causal_mem[key_position_q]<=sc_causal_w;score_cache_miss_mem[key_position_q]<=sc_cache_miss_w;score_invalid_mem[key_position_q]<=sc_invalid_w||sc_saturation_w;trace_valid_q<=1;trace_stage_q<=TRACE_SCORE;trace_index_q<={6'd0,key_position_q};trace_f16_q<=sc_out_w;trace_position_q<=token_position_q;
            if(key_position_q==token_position_q)begin context_index_q<=0;state_q<=S_SM_START;end else begin key_position_q<=key_position_q+7'd1;state_q<=S_SC_START;end end
          S_SM_START: if(sm_start_ready_w)begin context_index_q<=0;softmax_row_error_q<=1'b0;state_q<=S_SM_IN;end
          S_SM_IN: if(sm_score_ready_w)begin if(context_index_q==token_position_q)begin context_index_q<=0;state_q<=S_SM_OUT;end else context_index_q<=context_index_q+7'd1;end
          S_SM_OUT: if(sm_out_valid_w&&trace_free_w&&sm_result_ok_w)begin probability_mem[context_index_q]<=sm_out_w;softmax_row_error_q<=softmax_row_error_q||sm_row_error_w||sm_cache_miss_w||sm_invalid_w;trace_valid_q<=1;trace_stage_q<=TRACE_PROB;trace_index_q<={6'd0,context_index_q};trace_f16_q<=sm_out_w;trace_position_q<=token_position_q;
            if(context_index_q==token_position_q)begin dim_q<=0;key_position_q<=0;state_q<=S_AV_START;end else context_index_q<=context_index_q+7'd1;end
          S_AV_START: if(av_start_ready_w)begin key_position_q<=0;state_q<=S_AV_READ;end
          S_AV_READ: if(cache_rd_ready_w)state_q<=S_AV_TERM;
          S_AV_TERM: if(cache_out_valid_w&&cache_result_ok_w&&av_term_ready_w)begin if(key_position_q==token_position_q)state_q<=S_AV_OUT;else begin key_position_q<=key_position_q+7'd1;state_q<=S_AV_READ;end end
          S_AV_OUT: if(av_out_valid_w&&trace_free_w&&q_flat_valid_w&&av_result_ok_w)begin attention_mem[q_flat_index_w]<=av_out_w;trace_valid_q<=1;trace_stage_q<=TRACE_AV;trace_index_q<={3'd0,q_flat_index_w};trace_f16_q<=av_out_w;trace_position_q<=token_position_q;
            if(dim_q==6'd63)begin dim_q<=0;if(head_q==4'd13)begin psel_q<=PK_O;state_q<=S_P_START;end else begin head_q<=head_q+4'd1;key_position_q<=0;state_q<=S_SC_START;end end else begin dim_q<=dim_q+6'd1;key_position_q<=0;state_q<=S_AV_START;end end
          S_R1_START:if(r1_start_ready_w)begin hidden_index_q<=0;state_q<=S_R1_IN;end
          S_R1_IN:begin
            if(r1_in_ready_w)begin if(hidden_index_q==10'd895)state_q<=S_R1_OUT;else hidden_index_q<=hidden_index_q+10'd1;end
            if(r1_out_valid_w&&trace_free_w&&r1_result_ok_w)begin res1_mem[r1_idx_w[9:0]]<=r1_out_w;trace_valid_q<=1;trace_stage_q<=TRACE_RES1;trace_index_q<=r1_idx_w;trace_f16_q<=r1_out_w;trace_position_q<=token_position_q;if(r1_last_w)state_q<=S_N2_START;end
          end
          S_R1_OUT:if(r1_out_valid_w&&trace_free_w&&r1_result_ok_w)begin res1_mem[r1_idx_w[9:0]]<=r1_out_w;trace_valid_q<=1;trace_stage_q<=TRACE_RES1;trace_index_q<=r1_idx_w;trace_f16_q<=r1_out_w;trace_position_q<=token_position_q;if(r1_last_w)state_q<=S_N2_START;end
          S_N2_START:if(n2_start_ready_w)begin hidden_index_q<=0;state_q<=S_N2_IN;end
          S_N2_IN:if(n2_in_ready_w)begin if(hidden_index_q==10'd895)state_q<=S_N2_OUT;else hidden_index_q<=hidden_index_q+10'd1;end
          S_N2_OUT:if(n2_out_valid_w&&trace_free_w&&n2_result_ok_w)begin norm2_mem[n2_idx_w[9:0]]<=n2_out_w;trace_valid_q<=1;trace_stage_q<=TRACE_NORM2;trace_index_q<=n2_idx_w;trace_f16_q<=n2_out_w;trace_position_q<=token_position_q;if(n2_last_w)begin psel_q<=PK_GATE;state_q<=S_P_START;end end
          S_SI_START:if(si_start_ready_w)begin intermediate_index_q<=0;si_output_count_q<=0;state_q<=S_SI_IN;end
          S_SI_IN:begin
            if(si_in_ready_w)begin if(intermediate_index_q==13'd4863)state_q<=S_SI_OUT;else intermediate_index_q<=intermediate_index_q+13'd1;end
            if(si_out_valid_w&&trace_free_w&&si_result_ok_w)begin silu_mem[si_idx_w[12:0]]<=si_out_w;si_output_count_q<=si_output_count_q+13'd1;trace_valid_q<=1;trace_stage_q<=TRACE_SILU;trace_index_q<=si_idx_w;trace_f16_q<=si_out_w;trace_position_q<=token_position_q;end
          end
          S_SI_OUT:if(si_out_valid_w&&trace_free_w&&si_result_ok_w)begin silu_mem[si_idx_w[12:0]]<=si_out_w;si_output_count_q<=si_output_count_q+13'd1;trace_valid_q<=1;trace_stage_q<=TRACE_SILU;trace_index_q<=si_idx_w;trace_f16_q<=si_out_w;trace_position_q<=token_position_q;if(si_last_w)begin psel_q<=PK_DOWN;state_q<=S_P_START;end end
          S_R2_START:if(r2_start_ready_w&&trace_free_w)begin hidden_index_q<=0;state_q<=S_R2_IN;end
          S_R2_IN:begin
            if(r2_in_ready_w)begin if(hidden_index_q==10'd895)state_q<=S_R2_OUT;else hidden_index_q<=hidden_index_q+10'd1;end
            if(r2_out_valid_w&&r2_result_ok_w&&final_ready_i&&trace_ready_i&&known1(final_ready_i)&&known1(trace_ready_i)&&r2_last_w&&context_len_q[token_slot_q]=={1'b0,token_position_q})begin context_len_q[token_slot_q]<={1'b0,token_position_q}+8'd1;done_valid_q<=1;done_slot_q<=token_slot_q;done_position_q<=token_position_q;done_cycle_q<=cycle_q+32'd1;done_stall_q<=stall_q;state_q<=S_IDLE;end
          end
          S_R2_OUT:if(r2_out_valid_w&&r2_result_ok_w&&final_ready_i&&trace_ready_i&&known1(final_ready_i)&&known1(trace_ready_i)&&r2_last_w&&context_len_q[token_slot_q]=={1'b0,token_position_q})begin context_len_q[token_slot_q]<={1'b0,token_position_q}+8'd1;done_valid_q<=1;done_slot_q<=token_slot_q;done_position_q<=token_position_q;done_cycle_q<=cycle_q+32'd1;done_stall_q<=stall_q;state_q<=S_IDLE;end
          S_FAULT:state_q<=S_FAULT;
          default: state_q<=S_IDLE;
        endcase
      end
    end
endmodule

`default_nettype wire
