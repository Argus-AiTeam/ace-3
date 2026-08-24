`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_adaptation_verilator_top (
    input  wire clk_i,
    input  wire rst_ni,
    input  wire clear_i,
    input  wire rr_start_valid_i,
    output wire rr_start_ready_o,
    input  wire [12:0] rr_element_count_i,
    input  wire rr_in_valid_i,
    output wire rr_in_ready_o,
    input  wire [15:0] rr_projection_f16_i,
    input  wire [15:0] rr_residual_f16_i,
    output wire rr_out_valid_o,
    input  wire rr_out_ready_i,
    output wire [15:0] rr_out_f16_o,
    output wire [12:0] rr_out_index_o,
    output wire rr_out_last_o,
    output wire rr_invalid_o,
    output wire rr_saturation_o,
    output wire rr_busy_o,
    input  wire sg_start_valid_i,
    output wire sg_start_ready_o,
    input  wire [12:0] sg_element_count_i,
    input  wire sg_in_valid_i,
    output wire sg_in_ready_o,
    input  wire [15:0] sg_gate_f16_i,
    input  wire [15:0] sg_up_f16_i,
    output wire sg_out_valid_o,
    input  wire sg_out_ready_i,
    output wire [15:0] sg_out_f16_o,
    output wire [12:0] sg_out_index_o,
    output wire sg_out_last_o,
    output wire sg_invalid_o,
    output wire sg_saturation_o,
    output wire sg_busy_o,
    input  wire rn_start_valid_i,
    output wire rn_start_ready_o,
    input  wire [12:0] rn_element_count_i,
    input  wire rn_in_valid_i,
    output wire rn_in_ready_o,
    input  wire [15:0] rn_activation_f16_i,
    input  wire [15:0] rn_weight_f16_i,
    output wire rn_out_valid_o,
    input  wire rn_out_ready_i,
    output wire [15:0] rn_out_f16_o,
    output wire [12:0] rn_out_index_o,
    output wire rn_out_last_o,
    output wire rn_invalid_o,
    output wire rn_saturation_o,
    output wire rn_busy_o,
    output wire [45:0] rn_rms_q24_o
);
    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) residual (
        .clk_i(clk_i), .rst_ni(rst_ni), .clear_i(clear_i),
        .start_valid_i(rr_start_valid_i), .start_ready_o(rr_start_ready_o),
        .element_count_i(rr_element_count_i),
        .in_valid_i(rr_in_valid_i), .in_ready_o(rr_in_ready_o),
        .projection_f16_i(rr_projection_f16_i),
        .residual_f16_i(rr_residual_f16_i),
        .out_valid_o(rr_out_valid_o), .out_ready_i(rr_out_ready_i),
        .out_f16_o(rr_out_f16_o), .out_index_o(rr_out_index_o),
        .out_last_o(rr_out_last_o), .invalid_operand_o(rr_invalid_o),
        .saturation_o(rr_saturation_o), .busy_o(rr_busy_o)
    );

    ace3_fp16_silu_gate_core #(.INTERMEDIATE_SIZE(4864)) silu (
        .clk_i(clk_i), .rst_ni(rst_ni), .clear_i(clear_i),
        .start_valid_i(sg_start_valid_i), .start_ready_o(sg_start_ready_o),
        .element_count_i(sg_element_count_i),
        .in_valid_i(sg_in_valid_i), .in_ready_o(sg_in_ready_o),
        .gate_f16_i(sg_gate_f16_i), .up_f16_i(sg_up_f16_i),
        .out_valid_o(sg_out_valid_o), .out_ready_i(sg_out_ready_i),
        .out_f16_o(sg_out_f16_o), .out_index_o(sg_out_index_o),
        .out_last_o(sg_out_last_o), .invalid_operand_o(sg_invalid_o),
        .saturation_o(sg_saturation_o), .busy_o(sg_busy_o)
    );

    ace3_fp16_rmsnorm_core #(.HIDDEN_SIZE(8)) rmsnorm (
        .clk_i(clk_i), .rst_ni(rst_ni), .clear_i(clear_i),
        .start_valid_i(rn_start_valid_i), .start_ready_o(rn_start_ready_o),
        .element_count_i(rn_element_count_i),
        .in_valid_i(rn_in_valid_i), .in_ready_o(rn_in_ready_o),
        .activation_f16_i(rn_activation_f16_i),
        .weight_f16_i(rn_weight_f16_i),
        .out_valid_o(rn_out_valid_o), .out_ready_i(rn_out_ready_i),
        .out_f16_o(rn_out_f16_o), .out_index_o(rn_out_index_o),
        .out_last_o(rn_out_last_o), .invalid_operand_o(rn_invalid_o),
        .saturation_o(rn_saturation_o), .busy_o(rn_busy_o),
        .rms_q24_o(rn_rms_q24_o)
    );
endmodule

`default_nettype wire
