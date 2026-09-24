`timescale 1ns/1ps
`default_nettype none

// First rounding boundary only; the existing residual core performs the second.
module ace3_fp16_o_down_sum (
    input wire [15:0] attention_o_i,
    input wire [15:0] mlp_down_i,
    output wire [15:0] sum_o,
    output wire invalid_o,
    output wire saturation_o
);
    wire signed [40:0] o_q24, down_q24;
    wire o_finite, down_finite, o_sign, down_sign;
    wire signed [41:0] total =
        $signed({o_q24[40], o_q24}) + $signed({down_q24[40], down_q24});
    wire negative_zero = o_sign && down_sign &&
        (attention_o_i[14:0] == 15'd0) && (mlp_down_i[14:0] == 15'd0);

    ace3_fp16_to_q24 decode_o (
        .f16_i(attention_o_i), .q24_o(o_q24),
        .finite_o(o_finite), .sign_o(o_sign)
    );
    ace3_fp16_to_q24 decode_down (
        .f16_i(mlp_down_i), .q24_o(down_q24),
        .finite_o(down_finite), .sign_o(down_sign)
    );
    ace3_q24_to_fp16_rne #(.WIDTH(42)) round_inner (
        .q24_i(total), .zero_sign_i(negative_zero),
        .f16_o(sum_o), .saturation_o(saturation_o)
    );
    assign invalid_o = !o_finite || !down_finite;
endmodule
`default_nettype wire
