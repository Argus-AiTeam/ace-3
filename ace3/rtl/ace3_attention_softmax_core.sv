`timescale 1ns/1ps
`default_nettype none

module ace3_attention_softmax_core #(
    parameter integer CONTEXT_MAX = 128
) (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        start_valid_i,
    output wire        start_ready_o,
    input  wire [3:0]  query_head_i,
    input  wire [14:0] query_position_i,
    input  wire [15:0] context_count_i,

    input  wire        score_valid_i,
    output wire        score_ready_o,
    input  wire [15:0] score_f16_i,
    input  wire [14:0] key_position_i,
    input  wire        causal_i,
    input  wire        cache_miss_i,
    input  wire        invalid_operand_i,

    output wire        out_valid_o,
    input  wire        out_ready_i,
    output wire [15:0] probability_f16_o,
    output wire [3:0]  query_head_o,
    output wire [14:0] query_position_o,
    output wire [14:0] key_position_o,
    output wire [15:0] out_index_o,
    output wire        out_last_o,
    output wire        row_error_o,
    output wire        cache_miss_o,
    output wire        invalid_operand_o,
    output wire        busy_o
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_COLLECT = 2'd1;
    localparam [1:0] ST_EXP = 2'd2;
    localparam [1:0] ST_OUTPUT = 2'd3;

    reg [1:0] state_q;
    reg [3:0] query_head_q;
    reg [14:0] query_position_q;
    reg [15:0] context_count_q;
    reg [15:0] index_q;
    reg signed [40:0] max_score_q;
    reg eligible_seen_q;
    reg row_cache_miss_q;
    reg row_invalid_q;
    reg [32:0] exp_sum_q;

    reg signed [40:0] score_mem [0:CONTEXT_MAX-1];
    reg [14:0] key_position_mem [0:CONTEXT_MAX-1];
    reg causal_mem [0:CONTEXT_MAX-1];
    reg [24:0] exp_mem [0:CONTEXT_MAX-1];

    wire signed [40:0] score_q24_w;
    wire score_finite_w;
    wire unused_score_sign_w;
    wire expected_causal_w = key_position_i <= query_position_q;
    wire causal_mismatch_w = causal_i != expected_causal_w;
    wire score_controls_known_w =
        ((causal_i === 1'b0) || (causal_i === 1'b1)) &&
        ((cache_miss_i === 1'b0) || (cache_miss_i === 1'b1)) &&
        ((invalid_operand_i === 1'b0) ||
         (invalid_operand_i === 1'b1));
    wire score_payload_known_w =
        known16(score_f16_i) && known15(key_position_i);
    wire start_metadata_known_w =
        known4(query_head_i) && known15(query_position_i) &&
        known16(context_count_i);
    wire config_valid_w =
        (query_head_i < 4'd14) &&
        (context_count_i != 16'd0) &&
        (context_count_i <= CONTEXT_MAX);
    wire row_error_w =
        row_cache_miss_q || row_invalid_q || !eligible_seen_q;

    wire signed [41:0] current_delta_signed_w =
        $signed({max_score_q[40], max_score_q}) -
        $signed({score_mem[index_q][40], score_mem[index_q]});
    wire [41:0] current_delta_w =
        current_delta_signed_w[41] ? 42'd0 :
        current_delta_signed_w[41:0];
    wire [24:0] current_exp_w =
        (!row_error_w && causal_mem[index_q])
        ? exp_approx_q24(current_delta_w) : 25'd0;
    wire [32:0] exp_sum_next_w =
        exp_sum_q + {{8{1'b0}}, current_exp_w};

    reg [49:0] probability_numerator_w;
    reg [49:0] probability_quotient_w;
    reg [32:0] probability_remainder_w;
    reg probability_increment_w;
    reg [24:0] probability_q24_w;
    wire signed [25:0] probability_q24_signed_w =
        $signed({1'b0, probability_q24_w});
    wire [15:0] rounded_probability_f16_w;
    wire unused_probability_saturation_w;

    ace3_fp16_to_q24 decode_score (
        .f16_i(score_f16_i),
        .q24_o(score_q24_w),
        .finite_o(score_finite_w),
        .sign_o(unused_score_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(26)
    ) round_probability (
        .q24_i(probability_q24_signed_w),
        .zero_sign_i(1'b0),
        .f16_o(rounded_probability_f16_w),
        .saturation_o(unused_probability_saturation_w)
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

    function automatic [24:0] exp_lut_q24;
        input [4:0] table_index;
        begin
            case (table_index)
                5'd0:  exp_lut_q24 = 25'd16777216;
                5'd1:  exp_lut_q24 = 25'd16065917;
                5'd2:  exp_lut_q24 = 25'd15384775;
                5'd3:  exp_lut_q24 = 25'd14732511;
                5'd4:  exp_lut_q24 = 25'd14107901;
                5'd5:  exp_lut_q24 = 25'd13509772;
                5'd6:  exp_lut_q24 = 25'd12937002;
                5'd7:  exp_lut_q24 = 25'd12388516;
                5'd8:  exp_lut_q24 = 25'd11863283;
                5'd9:  exp_lut_q24 = 25'd11360319;
                5'd10: exp_lut_q24 = 25'd10878679;
                5'd11: exp_lut_q24 = 25'd10417458;
                5'd12: exp_lut_q24 = 25'd9975792;
                5'd13: exp_lut_q24 = 25'd9552851;
                5'd14: exp_lut_q24 = 25'd9147842;
                5'd15: exp_lut_q24 = 25'd8760003;
                default: exp_lut_q24 = 25'd8388608;
            endcase
        end
    endfunction

    function automatic [24:0] exp_approx_q24;
        input [41:0] delta_q24;
        reg [58:0] log_product;
        reg [34:0] y_q16;
        reg [4:0] integer_part;
        reg [3:0] table_index;
        reg [11:0] fraction;
        reg [24:0] upper_value;
        reg [24:0] lower_value;
        reg [24:0] table_difference;
        reg [36:0] interpolation_product;
        reg [24:0] interpolation_base;
        reg interpolation_increment;
        reg [24:0] interpolation_drop;
        reg [24:0] interpolated_value;
        reg [24:0] shift_mask;
        reg [24:0] shift_remainder;
        reg [24:0] shift_half;
        reg [24:0] shift_base;
        reg shift_increment;
        begin
            log_product = delta_q24 * 17'd94548;
            y_q16 = log_product >> 24;
            integer_part = y_q16[20:16];
            table_index = y_q16[15:12];
            fraction = y_q16[11:0];
            upper_value = exp_lut_q24({1'b0, table_index});
            lower_value = exp_lut_q24(
                {1'b0, table_index} + 5'd1
            );
            table_difference = upper_value - lower_value;
            interpolation_product = table_difference * fraction;
            interpolation_base = interpolation_product >> 12;
            interpolation_increment =
                (interpolation_product[11:0] > 12'h800) ||
                ((interpolation_product[11:0] == 12'h800) &&
                 interpolation_base[0]);
            interpolation_drop = interpolation_base +
                {{24{1'b0}}, interpolation_increment};
            interpolated_value = upper_value - interpolation_drop;
            shift_mask = 25'd0;
            shift_remainder = 25'd0;
            shift_half = 25'd0;
            shift_base = 25'd0;
            shift_increment = 1'b0;
            if ((y_q16 >> 16) >= 25) begin
                exp_approx_q24 = 25'd0;
            end else if (integer_part == 5'd0) begin
                exp_approx_q24 = interpolated_value;
            end else begin
                shift_mask = (25'd1 << integer_part) - 25'd1;
                shift_remainder = interpolated_value & shift_mask;
                shift_half = 25'd1 << (integer_part - 1'b1);
                shift_base = interpolated_value >> integer_part;
                shift_increment =
                    (shift_remainder > shift_half) ||
                    ((shift_remainder == shift_half) &&
                     shift_base[0]);
                exp_approx_q24 =
                    shift_base + {{24{1'b0}}, shift_increment};
            end
        end
    endfunction

    always @* begin
        probability_numerator_w = 50'd0;
        probability_quotient_w = 50'd0;
        probability_remainder_w = 33'd0;
        probability_increment_w = 1'b0;
        probability_q24_w = 25'd0;
        if (!row_error_w && (exp_sum_q != 33'd0)) begin
            probability_numerator_w =
                {exp_mem[index_q], 24'd0};
            probability_quotient_w =
                probability_numerator_w / exp_sum_q;
            probability_remainder_w =
                probability_numerator_w % exp_sum_q;
            probability_increment_w =
                ({probability_remainder_w, 1'b0} >
                 {1'b0, exp_sum_q}) ||
                (({probability_remainder_w, 1'b0} ==
                  {1'b0, exp_sum_q}) &&
                 probability_quotient_w[0]);
            probability_q24_w =
                probability_quotient_w[24:0] +
                {{24{1'b0}}, probability_increment_w};
        end
    end

    assign start_ready_o =
        rst_ni && !clear_i && (state_q == ST_IDLE) &&
        start_metadata_known_w && config_valid_w;
    assign score_ready_o =
        rst_ni && !clear_i && (state_q == ST_COLLECT) &&
        score_controls_known_w && score_payload_known_w;
    assign out_valid_o = state_q == ST_OUTPUT;
    assign probability_f16_o =
        row_error_w ? 16'h0000 : rounded_probability_f16_w;
    assign query_head_o = query_head_q;
    assign query_position_o = query_position_q;
    assign key_position_o = key_position_mem[index_q];
    assign out_index_o = index_q;
    assign out_last_o =
        (state_q == ST_OUTPUT) && (index_q == context_count_q - 1'b1);
    assign row_error_o = row_error_w;
    assign cache_miss_o = row_cache_miss_q;
    assign invalid_operand_o = row_invalid_q || !eligible_seen_q;
    assign busy_o = state_q != ST_IDLE;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            query_head_q <= 4'd0;
            query_position_q <= 15'd0;
            context_count_q <= 16'd0;
            index_q <= 16'd0;
            max_score_q <= 41'sd0;
            eligible_seen_q <= 1'b0;
            row_cache_miss_q <= 1'b0;
            row_invalid_q <= 1'b0;
            exp_sum_q <= 33'd0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            query_head_q <= 4'd0;
            query_position_q <= 15'd0;
            context_count_q <= 16'd0;
            index_q <= 16'd0;
            max_score_q <= 41'sd0;
            eligible_seen_q <= 1'b0;
            row_cache_miss_q <= 1'b0;
            row_invalid_q <= 1'b0;
            exp_sum_q <= 33'd0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= ST_COLLECT;
                        query_head_q <= query_head_i;
                        query_position_q <= query_position_i;
                        context_count_q <= context_count_i;
                        index_q <= 16'd0;
                        max_score_q <= 41'sd0;
                        eligible_seen_q <= 1'b0;
                        row_cache_miss_q <= 1'b0;
                        row_invalid_q <= 1'b0;
                        exp_sum_q <= 33'd0;
                    end
                end

                ST_COLLECT: begin
                    if (score_valid_i && score_ready_o) begin
                        score_mem[index_q] <= score_q24_w;
                        key_position_mem[index_q] <= key_position_i;
                        causal_mem[index_q] <= expected_causal_w;
                        if (expected_causal_w) begin
                            if (!eligible_seen_q ||
                                (score_q24_w > max_score_q))
                                max_score_q <= score_q24_w;
                            eligible_seen_q <= 1'b1;
                        end
                        if (expected_causal_w && cache_miss_i)
                            row_cache_miss_q <= 1'b1;
                        if (causal_mismatch_w ||
                            invalid_operand_i || !score_finite_w)
                            row_invalid_q <= 1'b1;
                        if (index_q == context_count_q - 1'b1) begin
                            state_q <= ST_EXP;
                            index_q <= 16'd0;
                            exp_sum_q <= 33'd0;
                        end else begin
                            index_q <= index_q + 1'b1;
                        end
                    end
                end

                ST_EXP: begin
                    exp_mem[index_q] <= current_exp_w;
                    exp_sum_q <= exp_sum_next_w;
                    if (index_q == context_count_q - 1'b1) begin
                        state_q <= ST_OUTPUT;
                        index_q <= 16'd0;
                    end else begin
                        index_q <= index_q + 1'b1;
                    end
                end

                ST_OUTPUT: begin
                    if (out_valid_o && out_ready_i) begin
                        if (index_q == context_count_q - 1'b1) begin
                            state_q <= ST_IDLE;
                            index_q <= 16'd0;
                        end else begin
                            index_q <= index_q + 1'b1;
                        end
                    end
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
