`timescale 1ns/1ps
`default_nettype none

module ace3_attention_score_core #(
    parameter integer HEAD_DIM = 64
) (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        start_valid_i,
    output wire        start_ready_o,
    input  wire [3:0]  query_head_i,
    input  wire [3:0]  key_head_i,
    input  wire [14:0] query_position_i,
    input  wire [14:0] key_position_i,

    input  wire        pair_valid_i,
    output wire        pair_ready_o,
    input  wire [15:0] q_f16_i,
    input  wire [15:0] k_f16_i,
    input  wire        cache_hit_i,

    output wire        out_valid_o,
    input  wire        out_ready_i,
    output wire [15:0] score_f16_o,
    output wire [3:0]  query_head_o,
    output wire [3:0]  key_head_o,
    output wire [14:0] query_position_o,
    output wire [14:0] key_position_o,
    output wire        causal_o,
    output wire        cache_miss_o,
    output wire        invalid_operand_o,
    output wire        saturation_o,
    output wire        busy_o
);
    localparam integer PAIR_INDEX_WIDTH =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM);
    localparam [PAIR_INDEX_WIDTH-1:0] LAST_PAIR = HEAD_DIM - 1;

    reg active_q;
    reg [PAIR_INDEX_WIDTH-1:0] pair_index_q;
    reg signed [89:0] accumulator_q;
    reg [3:0] query_head_q;
    reg [3:0] key_head_q;
    reg [14:0] query_position_q;
    reg [14:0] key_position_q;
    reg causal_q;
    reg cache_miss_q;
    reg invalid_q;

    reg out_valid_q;
    reg [15:0] score_f16_q;
    reg [3:0] out_query_head_q;
    reg [3:0] out_key_head_q;
    reg [14:0] out_query_position_q;
    reg [14:0] out_key_position_q;
    reg out_causal_q;
    reg out_cache_miss_q;
    reg out_invalid_q;
    reg out_saturation_q;

    wire signed [40:0] q_q24_w;
    wire signed [40:0] k_q24_w;
    wire q_finite_w;
    wire k_finite_w;
    wire unused_q_sign_w;
    wire unused_k_sign_w;
    wire signed [81:0] product_q48_w =
        $signed(q_q24_w) * $signed(k_q24_w);
    wire signed [89:0] product_extended_w =
        {{8{product_q48_w[81]}}, product_q48_w};
    wire signed [89:0] accumulator_next_w =
        accumulator_q + product_extended_w;
    wire cache_miss_next_w =
        cache_miss_q || (causal_q && !cache_hit_i);
    wire invalid_next_w = invalid_q || !q_finite_w || !k_finite_w;

    wire [3:0] mapped_key_head_w =
        (query_head_i < 4'd7) ? 4'd0 : 4'd1;
    wire config_valid_w =
        (query_head_i < 4'd14) &&
        (key_head_i < 4'd2) &&
        (key_head_i == mapped_key_head_w);
    wire start_metadata_known_w =
        known4(query_head_i) && known4(key_head_i) &&
        known15(query_position_i) && known15(key_position_i);
    wire pair_known_w =
        known16(q_f16_i) && known16(k_f16_i) &&
        ((cache_hit_i === 1'b0) || (cache_hit_i === 1'b1));

    wire signed [63:0] scaled_q24_w =
        round_scaled_q48_to_q24(accumulator_next_w);
    wire [15:0] rounded_score_f16_w;
    wire rounded_saturation_w;

    ace3_fp16_to_q24 decode_q (
        .f16_i(q_f16_i),
        .q24_o(q_q24_w),
        .finite_o(q_finite_w),
        .sign_o(unused_q_sign_w)
    );

    ace3_fp16_to_q24 decode_k (
        .f16_i(k_f16_i),
        .q24_o(k_q24_w),
        .finite_o(k_finite_w),
        .sign_o(unused_k_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(64)
    ) round_score (
        .q24_i(scaled_q24_w),
        .zero_sign_i(1'b0),
        .f16_o(rounded_score_f16_w),
        .saturation_o(rounded_saturation_w)
    );

    function automatic known4;
        input [3:0] value;
        begin
            known4 = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic known15;
        input [14:0] value;
        begin
            known15 = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic known16;
        input [15:0] value;
        begin
            known16 = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic signed [63:0] round_scaled_q48_to_q24;
        input signed [89:0] value;
        reg negative;
        reg [89:0] magnitude;
        reg [62:0] base;
        reg [26:0] remainder;
        reg increment;
        reg [63:0] rounded;
        begin
            negative = value[89];
            magnitude = negative ? (~value + 90'd1) : value;
            base = magnitude >> 27;
            remainder = magnitude[26:0];
            increment =
                (remainder > 27'h4000000) ||
                ((remainder == 27'h4000000) && base[0]);
            rounded = {1'b0, base} + {{63{1'b0}}, increment};
            round_scaled_q48_to_q24 =
                negative ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    assign start_ready_o =
        rst_ni && !clear_i && !active_q && !out_valid_q &&
        start_metadata_known_w && config_valid_w;
    assign pair_ready_o =
        rst_ni && !clear_i && active_q && !out_valid_q && pair_known_w;
    assign out_valid_o = out_valid_q;
    assign score_f16_o = score_f16_q;
    assign query_head_o = out_query_head_q;
    assign key_head_o = out_key_head_q;
    assign query_position_o = out_query_position_q;
    assign key_position_o = out_key_position_q;
    assign causal_o = out_causal_q;
    assign cache_miss_o = out_cache_miss_q;
    assign invalid_operand_o = out_invalid_q;
    assign saturation_o = out_saturation_q;
    assign busy_o = active_q || out_valid_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_q <= 1'b0;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            accumulator_q <= 90'sd0;
            query_head_q <= 4'd0;
            key_head_q <= 4'd0;
            query_position_q <= 15'd0;
            key_position_q <= 15'd0;
            causal_q <= 1'b0;
            cache_miss_q <= 1'b0;
            invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            score_f16_q <= 16'd0;
            out_query_head_q <= 4'd0;
            out_key_head_q <= 4'd0;
            out_query_position_q <= 15'd0;
            out_key_position_q <= 15'd0;
            out_causal_q <= 1'b0;
            out_cache_miss_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else if (clear_i) begin
            active_q <= 1'b0;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            accumulator_q <= 90'sd0;
            query_head_q <= 4'd0;
            key_head_q <= 4'd0;
            query_position_q <= 15'd0;
            key_position_q <= 15'd0;
            causal_q <= 1'b0;
            cache_miss_q <= 1'b0;
            invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            score_f16_q <= 16'd0;
            out_query_head_q <= 4'd0;
            out_key_head_q <= 4'd0;
            out_query_position_q <= 15'd0;
            out_key_position_q <= 15'd0;
            out_causal_q <= 1'b0;
            out_cache_miss_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;

            if (start_valid_i && start_ready_o) begin
                active_q <= 1'b1;
                pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
                accumulator_q <= 90'sd0;
                query_head_q <= query_head_i;
                key_head_q <= key_head_i;
                query_position_q <= query_position_i;
                key_position_q <= key_position_i;
                causal_q <= key_position_i <= query_position_i;
                cache_miss_q <= 1'b0;
                invalid_q <= 1'b0;
            end

            if (pair_valid_i && pair_ready_o) begin
                accumulator_q <= accumulator_next_w;
                cache_miss_q <= cache_miss_next_w;
                invalid_q <= invalid_next_w;
                if (pair_index_q == LAST_PAIR) begin
                    active_q <= 1'b0;
                    pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
                    out_valid_q <= 1'b1;
                    score_f16_q <=
                        (!causal_q || cache_miss_next_w || invalid_next_w)
                        ? 16'h0000 : rounded_score_f16_w;
                    out_query_head_q <= query_head_q;
                    out_key_head_q <= key_head_q;
                    out_query_position_q <= query_position_q;
                    out_key_position_q <= key_position_q;
                    out_causal_q <= causal_q;
                    out_cache_miss_q <=
                        causal_q && cache_miss_next_w;
                    out_invalid_q <= invalid_next_w;
                    out_saturation_q <=
                        causal_q && !cache_miss_next_w &&
                        !invalid_next_w && rounded_saturation_w;
                end else begin
                    pair_index_q <= pair_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
