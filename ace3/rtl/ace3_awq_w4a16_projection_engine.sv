`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_projection_engine #(
    parameter integer IN_FEATURES = 896,
    parameter integer OUT_FEATURES = 896,
    parameter integer BIAS_ENABLE = 0
) (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire [12:0]          first_output_channel_i,
    input  wire [12:0]          output_count_i,

    input  wire                 meta_valid_i,
    output wire                 meta_ready_o,
    output wire [12:0]          meta_output_channel_o,
    output wire [5:0]           meta_group_index_o,
    output wire [9:0]           meta_output_word_o,
    output wire [2:0]           meta_logical_lane_o,
    input  wire [31:0]          qzeros_i,
    input  wire [15:0]          scale_f16_i,

    input  wire                 pair_valid_i,
    output wire                 pair_ready_o,
    output wire [12:0]          pair_input_index_o,
    output wire [12:0]          pair_output_channel_o,
    output wire [5:0]           pair_group_index_o,
    output wire [9:0]           pair_output_word_o,
    output wire [2:0]           pair_logical_lane_o,
    input  wire [15:0]          activation_f16_i,
    input  wire [31:0]          qweight_i,

    input  wire                 bias_valid_i,
    output wire                 bias_ready_o,
    output wire [12:0]          bias_output_channel_o,
    input  wire [15:0]          bias_f16_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire [12:0]          out_channel_o,
    output wire [15:0]          out_f16_o,
    output wire signed [101:0]  acc_q53_48_o,
    output wire                 invalid_operand_o,
    output wire                 saturation_o,
    output wire                 busy_o
);
    localparam integer GROUP_COUNT = IN_FEATURES / 128;
    localparam [5:0] LAST_GROUP =
        GROUP_COUNT[5:0] - 6'd1;
    localparam [13:0] OUT_FEATURES_LIMIT =
        OUT_FEATURES[13:0];

    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_META       = 3'd1;
    localparam [2:0] ST_PAIRS      = 3'd2;
    localparam [2:0] ST_GROUP_WAIT = 3'd3;
    localparam [2:0] ST_BIAS       = 3'd4;
    localparam [2:0] ST_OUTPUT     = 3'd5;

    reg [2:0] state_q;
    reg [12:0] current_output_q;
    reg [12:0] outputs_remaining_q;
    reg [5:0] group_index_q;
    reg [6:0] pair_index_q;
    reg signed [101:0] cross_accumulator_q;
    reg cross_invalid_q;
    reg out_valid_q;
    reg [12:0] out_channel_q;
    reg [15:0] out_f16_q;
    reg signed [101:0] out_accumulator_q;
    reg out_invalid_q;
    reg out_saturation_q;
    reg [15:0] rounded_projection_q;
    reg rounded_projection_saturation_q;

    wire lane_start_valid_w;
    wire lane_start_ready_w;
    wire lane_pair_valid_w;
    wire lane_pair_ready_w;
    wire lane_out_valid_w;
    wire lane_out_ready_w;
    wire [15:0] lane_out_f16_unused_w;
    wire signed [95:0] lane_accumulator_w;
    wire lane_invalid_w;
    wire lane_saturation_unused_w;

    wire signed [101:0] lane_accumulator_extended_w;
    wire signed [101:0] cross_sum_w;
    wire [15:0] final_f16_w;
    wire final_saturation_w;
    wire final_invalid_w;
    wire [15:0] biased_f16_w;
    wire bias_invalid_w;
    wire bias_saturation_w;
    wire [13:0] requested_output_limit_w;
    wire start_config_valid_w;

    assign requested_output_limit_w =
        {1'b0, first_output_channel_i} + {1'b0, output_count_i};
    assign start_config_valid_w =
        (output_count_i != 13'd0) &&
        ({1'b0, first_output_channel_i} < OUT_FEATURES_LIMIT) &&
        (requested_output_limit_w <= OUT_FEATURES_LIMIT);

    assign start_ready_o =
        rst_ni && !clear_i && (state_q == ST_IDLE) && start_config_valid_w;
    assign busy_o = state_q != ST_IDLE;

    assign meta_output_channel_o = current_output_q;
    assign meta_group_index_o = group_index_q;
    assign meta_output_word_o = current_output_q[12:3];
    assign meta_logical_lane_o = current_output_q[2:0];
    assign pair_input_index_o = {group_index_q, pair_index_q};
    assign pair_output_channel_o = current_output_q;
    assign pair_group_index_o = group_index_q;
    assign pair_output_word_o = current_output_q[12:3];
    assign pair_logical_lane_o = current_output_q[2:0];
    assign bias_output_channel_o = current_output_q;
    assign bias_ready_o =
        rst_ni && !clear_i && (state_q == ST_BIAS) &&
        known16(bias_f16_i);

    assign lane_start_valid_w =
        rst_ni && !clear_i && (state_q == ST_META) && meta_valid_i;
    assign meta_ready_o =
        rst_ni && !clear_i && (state_q == ST_META) && lane_start_ready_w;
    assign lane_pair_valid_w =
        rst_ni && !clear_i && (state_q == ST_PAIRS) && pair_valid_i;
    assign pair_ready_o =
        rst_ni && !clear_i && (state_q == ST_PAIRS) && lane_pair_ready_w;
    assign lane_out_ready_w =
        rst_ni && !clear_i && (state_q == ST_GROUP_WAIT);

    assign lane_accumulator_extended_w = {
        {6{lane_accumulator_w[95]}},
        lane_accumulator_w
    };
    assign cross_sum_w =
        cross_accumulator_q + lane_accumulator_extended_w;
    assign final_invalid_w = cross_invalid_q || lane_invalid_w;

    assign out_valid_o = out_valid_q;
    assign out_channel_o = out_channel_q;
    assign out_f16_o = out_f16_q;
    assign acc_q53_48_o = out_accumulator_q;
    assign invalid_operand_o = out_invalid_q;
    assign saturation_o = out_saturation_q;

    function automatic known16;
        input [15:0] value;
        begin
            known16 = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    ace3_awq_w4a16_g128_dot_lane group_lane (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(lane_start_valid_w),
        .start_ready_o(lane_start_ready_w),
        .logical_lane_i(current_output_q[2:0]),
        .qzeros_i(qzeros_i),
        .scale_f16_i(scale_f16_i),
        .pair_valid_i(lane_pair_valid_w),
        .pair_ready_o(lane_pair_ready_w),
        .activation_f16_i(activation_f16_i),
        .qweight_i(qweight_i),
        .out_valid_o(lane_out_valid_w),
        .out_ready_i(lane_out_ready_w),
        .out_f16_o(lane_out_f16_unused_w),
        .acc_q47_48_o(lane_accumulator_w),
        .invalid_operand_o(lane_invalid_w),
        .saturation_o(lane_saturation_unused_w)
    );

    ace3_q47_48_to_f16_rne #(
        .ACC_WIDTH(102)
    ) final_rounder (
        .fixed_i(cross_sum_w),
        .f16_o(final_f16_w),
        .saturation_o(final_saturation_w)
    );

    ace3_fp16_add add_projection_bias (
        .a_f16_i(rounded_projection_q),
        .b_f16_i(bias_f16_i),
        .sum_f16_o(biased_f16_w),
        .invalid_operand_o(bias_invalid_w),
        .saturation_o(bias_saturation_w)
    );

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            current_output_q <= 13'd0;
            outputs_remaining_q <= 13'd0;
            group_index_q <= 6'd0;
            pair_index_q <= 7'd0;
            cross_accumulator_q <= 102'sd0;
            cross_invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_channel_q <= 13'd0;
            out_f16_q <= 16'd0;
            out_accumulator_q <= 102'sd0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
            rounded_projection_q <= 16'd0;
            rounded_projection_saturation_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            current_output_q <= 13'd0;
            outputs_remaining_q <= 13'd0;
            group_index_q <= 6'd0;
            pair_index_q <= 7'd0;
            cross_accumulator_q <= 102'sd0;
            cross_invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_channel_q <= 13'd0;
            out_f16_q <= 16'd0;
            out_accumulator_q <= 102'sd0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
            rounded_projection_q <= 16'd0;
            rounded_projection_saturation_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= ST_META;
                        current_output_q <= first_output_channel_i;
                        outputs_remaining_q <= output_count_i;
                        group_index_q <= 6'd0;
                        pair_index_q <= 7'd0;
                        cross_accumulator_q <= 102'sd0;
                        cross_invalid_q <= 1'b0;
                        out_invalid_q <= 1'b0;
                        out_saturation_q <= 1'b0;
                    end
                end
                ST_META: begin
                    if (meta_valid_i && meta_ready_o) begin
                        state_q <= ST_PAIRS;
                        pair_index_q <= 7'd0;
                    end
                end
                ST_PAIRS: begin
                    if (pair_valid_i && pair_ready_o) begin
                        if (pair_index_q == 7'd127)
                            state_q <= ST_GROUP_WAIT;
                        else
                            pair_index_q <= pair_index_q + 7'd1;
                    end
                end
                ST_GROUP_WAIT: begin
                    if (lane_out_valid_w && lane_out_ready_w) begin
                        cross_accumulator_q <= cross_sum_w;
                        cross_invalid_q <= final_invalid_w;
                        if (group_index_q == LAST_GROUP) begin
                            out_channel_q <= current_output_q;
                            out_accumulator_q <= cross_sum_w;
                            out_invalid_q <= final_invalid_w;
                            rounded_projection_q <=
                                final_invalid_w ? 16'h0000 : final_f16_w;
                            rounded_projection_saturation_q <=
                                final_invalid_w ? 1'b0 : final_saturation_w;
                            if (BIAS_ENABLE != 0) begin
                                state_q <= ST_BIAS;
                            end else begin
                                state_q <= ST_OUTPUT;
                                out_valid_q <= 1'b1;
                                out_f16_q <=
                                    final_invalid_w ? 16'h0000 : final_f16_w;
                                out_saturation_q <=
                                    final_invalid_w ? 1'b0 : final_saturation_w;
                            end
                        end else begin
                            state_q <= ST_META;
                            group_index_q <= group_index_q + 6'd1;
                        end
                    end
                end
                ST_BIAS: begin
                    if (bias_valid_i && bias_ready_o) begin
                        state_q <= ST_OUTPUT;
                        out_valid_q <= 1'b1;
                        out_invalid_q <= out_invalid_q || bias_invalid_w;
                        if (out_invalid_q || bias_invalid_w) begin
                            out_f16_q <= 16'h0000;
                            out_saturation_q <= 1'b0;
                        end else begin
                            out_f16_q <= biased_f16_w;
                            out_saturation_q <=
                                rounded_projection_saturation_q ||
                                bias_saturation_w;
                        end
                    end
                end
                ST_OUTPUT: begin
                    if (out_valid_q && out_ready_i && !clear_i) begin
                        out_valid_q <= 1'b0;
                        if (outputs_remaining_q == 13'd1) begin
                            state_q <= ST_IDLE;
                        end else begin
                            state_q <= ST_META;
                            current_output_q <= current_output_q + 13'd1;
                            outputs_remaining_q <=
                                outputs_remaining_q - 13'd1;
                            group_index_q <= 6'd0;
                            pair_index_q <= 7'd0;
                            cross_accumulator_q <= 102'sd0;
                            cross_invalid_q <= 1'b0;
                            out_invalid_q <= 1'b0;
                            out_saturation_q <= 1'b0;
                        end
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
