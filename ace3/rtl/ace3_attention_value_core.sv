`timescale 1ns/1ps
`default_nettype none

module ace3_attention_value_core #(
    parameter integer HEAD_DIM = 64,
    parameter integer CONTEXT_MAX = 128
) (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        start_valid_i,
    output wire        start_ready_o,
    input  wire [3:0]  query_head_i,
    input  wire [3:0]  value_head_i,
    input  wire [14:0] query_position_i,
    input  wire [5:0]  dimension_i,
    input  wire [15:0] context_count_i,

    input  wire        term_valid_i,
    output wire        term_ready_o,
    input  wire [15:0] probability_f16_i,
    input  wire [15:0] value_f16_i,
    input  wire        value_hit_i,
    input  wire        row_error_i,

    output wire        out_valid_o,
    input  wire        out_ready_i,
    output wire [15:0] value_f16_o,
    output wire [3:0]  query_head_o,
    output wire [3:0]  value_head_o,
    output wire [14:0] query_position_o,
    output wire [5:0]  dimension_o,
    output wire        row_error_o,
    output wire        cache_miss_o,
    output wire        invalid_operand_o,
    output wire        saturation_o,
    output wire        busy_o
);
    localparam integer CONTEXT_INDEX_WIDTH =
        (CONTEXT_MAX <= 1) ? 1 : $clog2(CONTEXT_MAX);
    localparam integer CONTEXT_COUNT_WIDTH =
        (CONTEXT_MAX <= 1) ? 1 : $clog2(CONTEXT_MAX + 1);

    reg active_q;
    reg [CONTEXT_COUNT_WIDTH-1:0] context_count_q;
    reg [CONTEXT_INDEX_WIDTH-1:0] term_index_q;
    reg signed [89:0] accumulator_q;
    reg row_error_q;
    reg cache_miss_q;
    reg invalid_q;
    reg [3:0] query_head_q;
    reg [3:0] value_head_q;
    reg [6:0] query_position_q;
    reg [5:0] dimension_q;

    reg out_valid_q;
    reg [15:0] out_value_f16_q;
    reg [3:0] out_query_head_q;
    reg [3:0] out_value_head_q;
    reg [6:0] out_query_position_q;
    reg [5:0] out_dimension_q;
    reg out_row_error_q;
    reg out_cache_miss_q;
    reg out_invalid_q;
    reg out_saturation_q;

    wire signed [40:0] probability_q24_w;
    wire signed [40:0] value_q24_w;
    wire probability_finite_w;
    wire value_finite_w;
    wire probability_sign_w;
    wire unused_value_sign_w;
    wire probability_nonzero_w =
        probability_f16_i[14:0] != 15'd0;
    wire signed [81:0] product_q48_w =
        $signed(probability_q24_w) * $signed(value_q24_w);
    wire signed [89:0] product_extended_w =
        {{8{product_q48_w[81]}}, product_q48_w};
    wire signed [89:0] accumulator_next_w =
        accumulator_q + product_extended_w;
    wire current_invalid_w =
        !probability_finite_w ||
        (probability_nonzero_w && probability_sign_w) ||
        (probability_nonzero_w && !value_finite_w);
    wire row_error_next_w = row_error_q || row_error_i;
    wire cache_miss_next_w =
        cache_miss_q || (probability_nonzero_w && !value_hit_i);
    wire invalid_next_w = invalid_q || current_invalid_w;

    wire [3:0] mapped_value_head_w =
        (query_head_i < 4'd7) ? 4'd0 : 4'd1;
    wire [31:0] dimension_ext_w = {26'd0, dimension_i};
    wire [31:0] context_count_ext_w = {16'd0, context_count_i};
    wire [31:0] query_position_ext_w = {17'd0, query_position_i};
    wire [CONTEXT_COUNT_WIDTH-1:0] term_count_index_w =
        {{(CONTEXT_COUNT_WIDTH-CONTEXT_INDEX_WIDTH){1'b0}},
         term_index_q};
    wire config_valid_w =
        (query_head_i < 4'd14) &&
        (value_head_i < 4'd2) &&
        (value_head_i == mapped_value_head_w) &&
        (dimension_ext_w < HEAD_DIM) &&
        (context_count_i != 16'd0) &&
        (context_count_ext_w <= CONTEXT_MAX) &&
        (query_position_ext_w < CONTEXT_MAX);
    wire start_metadata_known_w =
        known4(query_head_i) && known4(value_head_i) &&
        known15(query_position_i) && known6(dimension_i) &&
        known16(context_count_i);
    wire term_known_w =
        known16(probability_f16_i) && known16(value_f16_i) &&
        ((value_hit_i === 1'b0) || (value_hit_i === 1'b1)) &&
        ((row_error_i === 1'b0) || (row_error_i === 1'b1));

    wire signed [66:0] composed_q24_w =
        round_q48_to_q24(accumulator_next_w);
    wire [15:0] rounded_value_f16_w;
    wire rounded_saturation_w;

    ace3_fp16_to_q24 decode_probability (
        .f16_i(probability_f16_i),
        .q24_o(probability_q24_w),
        .finite_o(probability_finite_w),
        .sign_o(probability_sign_w)
    );

    ace3_fp16_to_q24 decode_value (
        .f16_i(value_f16_i),
        .q24_o(value_q24_w),
        .finite_o(value_finite_w),
        .sign_o(unused_value_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(67)
    ) round_composition (
        .q24_i(composed_q24_w),
        .zero_sign_i(1'b0),
        .f16_o(rounded_value_f16_w),
        .saturation_o(rounded_saturation_w)
    );

    function automatic known4;
        input [3:0] candidate_i;
        begin
            known4 = ((^candidate_i === 1'b0) ||
                      (^candidate_i === 1'b1));
        end
    endfunction

    function automatic known6;
        input [5:0] candidate_i;
        begin
            known6 = ((^candidate_i === 1'b0) ||
                      (^candidate_i === 1'b1));
        end
    endfunction

    function automatic known15;
        input [14:0] candidate_i;
        begin
            known15 = ((^candidate_i === 1'b0) ||
                       (^candidate_i === 1'b1));
        end
    endfunction

    function automatic known16;
        input [15:0] candidate_i;
        begin
            known16 = ((^candidate_i === 1'b0) ||
                       (^candidate_i === 1'b1));
        end
    endfunction

    function automatic signed [66:0] round_q48_to_q24;
        input signed [89:0] q48_value_i;
        reg negative;
        reg [89:0] magnitude;
        reg [65:0] base;
        reg [23:0] remainder;
        reg increment;
        reg [66:0] rounded;
        begin
            negative = q48_value_i[89];
            magnitude = negative ? (~q48_value_i + 90'd1) : q48_value_i;
            base = magnitude[89:24];
            remainder = magnitude[23:0];
            increment =
                (remainder > 24'h800000) ||
                ((remainder == 24'h800000) && base[0]);
            rounded = {1'b0, base} + {{66{1'b0}}, increment};
            round_q48_to_q24 =
                negative ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    assign start_ready_o =
        rst_ni && !clear_i && !active_q && !out_valid_q &&
        start_metadata_known_w && config_valid_w;
    assign term_ready_o =
        rst_ni && !clear_i && active_q && !out_valid_q && term_known_w;
    assign out_valid_o = out_valid_q;
    assign value_f16_o = out_value_f16_q;
    assign query_head_o = out_query_head_q;
    assign value_head_o = out_value_head_q;
    assign query_position_o = {8'd0, out_query_position_q};
    assign dimension_o = out_dimension_q;
    assign row_error_o = out_row_error_q;
    assign cache_miss_o = out_cache_miss_q;
    assign invalid_operand_o = out_invalid_q;
    assign saturation_o = out_saturation_q;
    assign busy_o = active_q || out_valid_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_q <= 1'b0;
            context_count_q <= {CONTEXT_COUNT_WIDTH{1'b0}};
            term_index_q <= {CONTEXT_INDEX_WIDTH{1'b0}};
            accumulator_q <= 90'sd0;
            row_error_q <= 1'b0;
            cache_miss_q <= 1'b0;
            invalid_q <= 1'b0;
            query_head_q <= 4'd0;
            value_head_q <= 4'd0;
            query_position_q <= 7'd0;
            dimension_q <= 6'd0;
            out_valid_q <= 1'b0;
            out_value_f16_q <= 16'd0;
            out_query_head_q <= 4'd0;
            out_value_head_q <= 4'd0;
            out_query_position_q <= 7'd0;
            out_dimension_q <= 6'd0;
            out_row_error_q <= 1'b0;
            out_cache_miss_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else if (clear_i) begin
            active_q <= 1'b0;
            context_count_q <= {CONTEXT_COUNT_WIDTH{1'b0}};
            term_index_q <= {CONTEXT_INDEX_WIDTH{1'b0}};
            accumulator_q <= 90'sd0;
            row_error_q <= 1'b0;
            cache_miss_q <= 1'b0;
            invalid_q <= 1'b0;
            query_head_q <= 4'd0;
            value_head_q <= 4'd0;
            query_position_q <= 7'd0;
            dimension_q <= 6'd0;
            out_valid_q <= 1'b0;
            out_value_f16_q <= 16'd0;
            out_query_head_q <= 4'd0;
            out_value_head_q <= 4'd0;
            out_query_position_q <= 7'd0;
            out_dimension_q <= 6'd0;
            out_row_error_q <= 1'b0;
            out_cache_miss_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;

            if (start_valid_i && start_ready_o) begin
                active_q <= 1'b1;
                context_count_q <=
                    context_count_i[CONTEXT_COUNT_WIDTH-1:0];
                term_index_q <= {CONTEXT_INDEX_WIDTH{1'b0}};
                accumulator_q <= 90'sd0;
                row_error_q <= 1'b0;
                cache_miss_q <= 1'b0;
                invalid_q <= 1'b0;
                query_head_q <= query_head_i;
                value_head_q <= value_head_i;
                query_position_q <= query_position_i[6:0];
                dimension_q <= dimension_i;
            end

            if (term_valid_i && term_ready_o) begin
                accumulator_q <= accumulator_next_w;
                row_error_q <= row_error_next_w;
                cache_miss_q <= cache_miss_next_w;
                invalid_q <= invalid_next_w;
                if (term_count_index_w ==
                    context_count_q -
                    {{(CONTEXT_COUNT_WIDTH-1){1'b0}}, 1'b1}) begin
                    active_q <= 1'b0;
                    term_index_q <= {CONTEXT_INDEX_WIDTH{1'b0}};
                    out_valid_q <= 1'b1;
                    out_value_f16_q <=
                        (row_error_next_w || cache_miss_next_w ||
                         invalid_next_w)
                        ? 16'h0000 : rounded_value_f16_w;
                    out_query_head_q <= query_head_q;
                    out_value_head_q <= value_head_q;
                    out_query_position_q <= query_position_q;
                    out_dimension_q <= dimension_q;
                    out_row_error_q <= row_error_next_w ||
                        cache_miss_next_w || invalid_next_w;
                    out_cache_miss_q <= cache_miss_next_w;
                    out_invalid_q <= invalid_next_w;
                    out_saturation_q <=
                        !row_error_next_w && !cache_miss_next_w &&
                        !invalid_next_w && rounded_saturation_w;
                end else begin
                    term_index_q <= term_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
