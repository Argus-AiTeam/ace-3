`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_single_round_residual_core #(
    parameter integer VECTOR_SIZE = 896
) (
    input wire clk_i, input wire rst_ni, input wire clear_i,
    input wire start_valid_i, output wire start_ready_o,
    input wire [12:0] element_count_i,
    input wire in_valid_i, output wire in_ready_o,
    input wire [15:0] projection_f16_i,
    input wire [15:0] residual_f16_i,
    input wire [15:0] attention_f16_i,
    output wire out_valid_o, input wire out_ready_i,
    output wire [15:0] out_f16_o, output wire [12:0] out_index_o,
    output wire out_last_o, output wire invalid_operand_o,
    output wire saturation_o, output wire busy_o
);
    reg active_q, out_valid_q, out_last_q, out_invalid_q, out_saturation_q;
    reg [12:0] remaining_q, index_q, out_index_q;
    reg [15:0] out_f16_q;
    wire signed [40:0] hidden_w, attention_w, down_w;
    wire hidden_finite_w, attention_finite_w, down_finite_w;
    wire hidden_sign_w, attention_sign_w, down_sign_w;
    // Three finite binary16 values fit exactly in signed 43-bit Q24.
    wire signed [42:0] sum_w =
        $signed({{2{hidden_w[40]}}, hidden_w}) +
        $signed({{2{attention_w[40]}}, attention_w}) +
        $signed({{2{down_w[40]}}, down_w});
    wire zero_sign_w = (residual_f16_i == 16'h8000) &&
                       (attention_f16_i == 16'h8000) &&
                       (projection_f16_i == 16'h8000);
    wire [15:0] rounded_w;
    wire overflow_w;
    wire operands_invalid_w = !hidden_finite_w || !attention_finite_w ||
                              !down_finite_w;
    wire invalid_w = operands_invalid_w || overflow_w;
    wire config_valid_w = (element_count_i != 13'd0) &&
                          (element_count_i <= VECTOR_SIZE[12:0]);

    ace3_fp16_to_q24 hidden_decode (
        .f16_i(residual_f16_i), .q24_o(hidden_w),
        .finite_o(hidden_finite_w), .sign_o(hidden_sign_w));
    ace3_fp16_to_q24 attention_decode (
        .f16_i(attention_f16_i), .q24_o(attention_w),
        .finite_o(attention_finite_w), .sign_o(attention_sign_w));
    ace3_fp16_to_q24 down_decode (
        .f16_i(projection_f16_i), .q24_o(down_w),
        .finite_o(down_finite_w), .sign_o(down_sign_w));
    ace3_q24_to_fp16_rne #(.WIDTH(43)) round_sum (
        .q24_i(sum_w), .zero_sign_i(zero_sign_w),
        .f16_o(rounded_w), .saturation_o(overflow_w));

    assign start_ready_o = rst_ni && !clear_i && !active_q &&
                           !out_valid_q && config_valid_w;
    assign in_ready_o = rst_ni && !clear_i && active_q &&
                        (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign out_f16_o = out_f16_q;
    assign out_index_o = out_index_q;
    assign out_last_o = out_last_q;
    assign invalid_operand_o = out_invalid_q;
    assign saturation_o = out_saturation_q;
    assign busy_o = active_q || out_valid_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_q <= 1'b0;
            remaining_q <= 13'd0;
            index_q <= 13'd0;
            out_valid_q <= 1'b0;
            out_f16_q <= 16'd0;
            out_index_q <= 13'd0;
            out_last_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else if (clear_i) begin
            active_q <= 1'b0;
            remaining_q <= 13'd0;
            index_q <= 13'd0;
            out_valid_q <= 1'b0;
            out_f16_q <= 16'd0;
            out_index_q <= 13'd0;
            out_last_q <= 1'b0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            if (start_valid_i && start_ready_o) begin
                active_q <= 1'b1;
                remaining_q <= element_count_i;
                index_q <= 13'd0;
            end
            if (in_valid_i && in_ready_o) begin
                out_valid_q <= 1'b1;
                // Reject overflow rather than returning the helper's saturated value.
                out_f16_q <= invalid_w ? 16'h0000 : rounded_w;
                out_index_q <= index_q;
                out_last_q <= remaining_q == 13'd1;
                out_invalid_q <= invalid_w;
                out_saturation_q <= !operands_invalid_w && overflow_w;
                if (remaining_q == 13'd1) begin
                    active_q <= 1'b0;
                    remaining_q <= 13'd0;
                end else begin
                    remaining_q <= remaining_q - 13'd1;
                    index_q <= index_q + 13'd1;
                end
            end
        end
    end
endmodule

`default_nettype wire
