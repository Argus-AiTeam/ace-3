`timescale 1ns/1ps
`default_nettype none

module ace3_final_rmsnorm (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         clear_i,
    input  wire         start_valid_i,
    output wire         start_ready_o,
    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [15:0]  activation_f16_i,
    input  wire [15:0]  weight_f16_i,
    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [15:0]  out_f16_o,
    output wire [12:0]  out_index_o,
    output wire         out_last_o,
    output wire         invalid_operand_o,
    output wire         saturation_o,
    output wire         busy_o,
    output wire [45:0]  rms_q24_o
);
    ace3_fp16_rmsnorm_core #(
        .HIDDEN_SIZE(896),
        .EPSILON_Q48(64'd281474977)
    ) final_norm_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(start_valid_i),
        .start_ready_o(start_ready_o),
        .element_count_i(13'd896),
        .in_valid_i(in_valid_i),
        .in_ready_o(in_ready_o),
        .activation_f16_i(activation_f16_i),
        .weight_f16_i(weight_f16_i),
        .out_valid_o(out_valid_o),
        .out_ready_i(out_ready_i),
        .out_f16_o(out_f16_o),
        .out_index_o(out_index_o),
        .out_last_o(out_last_o),
        .invalid_operand_o(invalid_operand_o),
        .saturation_o(saturation_o),
        .busy_o(busy_o),
        .rms_q24_o(rms_q24_o)
    );
endmodule

`default_nettype wire
