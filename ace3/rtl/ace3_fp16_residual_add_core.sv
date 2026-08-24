`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_residual_add_core #(
    parameter integer VECTOR_SIZE = 896
) (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         clear_i,

    input  wire         start_valid_i,
    output wire         start_ready_o,
    input  wire [12:0]  element_count_i,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [15:0]  projection_f16_i,
    input  wire [15:0]  residual_f16_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [15:0]  out_f16_o,
    output wire [12:0]  out_index_o,
    output wire         out_last_o,
    output wire         invalid_operand_o,
    output wire         saturation_o,
    output wire         busy_o
);
    reg active_q;
    reg [12:0] remaining_q;
    reg [12:0] index_q;
    reg out_valid_q;
    reg [15:0] out_f16_q;
    reg [12:0] out_index_q;
    reg out_last_q;
    reg out_invalid_q;
    reg out_saturation_q;

    wire signed [40:0] projection_q24_w;
    wire signed [40:0] residual_q24_w;
    wire projection_finite_w;
    wire residual_finite_w;
    wire projection_zero_w = projection_f16_i[14:0] == 15'd0;
    wire residual_zero_w = residual_f16_i[14:0] == 15'd0;
    wire projection_sign_w;
    wire residual_sign_w;
    wire signed [41:0] sum_q24_w =
        $signed({projection_q24_w[40], projection_q24_w}) +
        $signed({residual_q24_w[40], residual_q24_w});
    wire invalid_w = !projection_finite_w || !residual_finite_w;
    wire zero_sign_w = projection_zero_w && residual_zero_w &&
                       projection_sign_w && residual_sign_w;
    wire [15:0] rounded_f16_w;
    wire rounded_saturation_w;
    wire config_valid_w =
        (element_count_i != 13'd0) &&
        (element_count_i <= VECTOR_SIZE[12:0]);

    ace3_fp16_to_q24 decode_projection (
        .f16_i(projection_f16_i),
        .q24_o(projection_q24_w),
        .finite_o(projection_finite_w),
        .sign_o(projection_sign_w)
    );

    ace3_fp16_to_q24 decode_residual (
        .f16_i(residual_f16_i),
        .q24_o(residual_q24_w),
        .finite_o(residual_finite_w),
        .sign_o(residual_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(42)
    ) round_sum (
        .q24_i(sum_q24_w),
        .zero_sign_i(zero_sign_w),
        .f16_o(rounded_f16_w),
        .saturation_o(rounded_saturation_w)
    );

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
                out_f16_q <= invalid_w ? 16'h0000 : rounded_f16_w;
                out_index_q <= index_q;
                out_last_q <= remaining_q == 13'd1;
                out_invalid_q <= invalid_w;
                out_saturation_q <= invalid_w
                    ? 1'b0 : rounded_saturation_w;
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
