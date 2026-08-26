`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

module ace3_fp16_to_q24 (
    input  wire [15:0]        f16_i,
    output reg  signed [40:0] q24_o,
    output wire               finite_o,
    output wire               sign_o
);
    reg [40:0] magnitude;

    assign finite_o = f16_i[14:10] != 5'h1f;
    assign sign_o = f16_i[15];

    always @* begin
        magnitude = 41'd0;
        if (f16_i[14:10] == 5'd0)
            magnitude = {31'd0, f16_i[9:0]};
        else if (finite_o)
            magnitude = {30'd0, 1'b1, f16_i[9:0]}
                        << (f16_i[14:10] - 5'd1);

        if (!finite_o)
            q24_o = 41'sd0;
        else if (f16_i[15])
            q24_o = -$signed(magnitude);
        else
            q24_o = $signed(magnitude);
    end
endmodule

module ace3_q24_to_fp16_rne #(
    parameter integer WIDTH = 84
) (
    input  wire signed [WIDTH-1:0] q24_i,
    input  wire                    zero_sign_i,
    output wire [15:0]             f16_o,
    output wire                    saturation_o
);
    function automatic [16:0] convert_q24;
        input signed [WIDTH-1:0] value;
        input                    zero_sign;
        reg                      value_sign;
        reg [WIDTH-1:0]          magnitude;
        reg [10:0]               retained;
        reg [11:0]               rounded;
        reg                      guard_bit;
        reg                      sticky_bit;
        reg [15:0]               result;
        reg                      saturated;
        integer                  bit_index;
        integer                  most_significant_bit;
        integer                  right_shift;
        integer                  unbiased_exponent;
        reg [4:0]                encoded_exponent;
        begin
            value_sign = value[WIDTH-1];
            magnitude = value_sign
                ? (~value + {{(WIDTH-1){1'b0}}, 1'b1})
                : value;
            retained = 11'd0;
            rounded = 12'd0;
            guard_bit = 1'b0;
            sticky_bit = 1'b0;
            result = {zero_sign, 15'd0};
            saturated = 1'b0;
            most_significant_bit = -1;
            encoded_exponent = 5'd0;

            for (bit_index = 0; bit_index < WIDTH;
                 bit_index = bit_index + 1)
                if (magnitude[bit_index])
                    most_significant_bit = bit_index;

            if (most_significant_bit >= 10) begin
                right_shift = most_significant_bit - 10;
                for (bit_index = 0; bit_index < 11;
                     bit_index = bit_index + 1)
                    if ((right_shift + bit_index) < WIDTH)
                        retained[bit_index] =
                            magnitude[right_shift + bit_index];
                if (right_shift > 0) begin
                    guard_bit = magnitude[right_shift - 1];
                    for (bit_index = 0; bit_index < WIDTH;
                         bit_index = bit_index + 1)
                        if (bit_index < (right_shift - 1))
                            sticky_bit = sticky_bit | magnitude[bit_index];
                end
                rounded = {1'b0, retained} +
                    {{11{1'b0}},
                     (guard_bit && (sticky_bit || retained[0]))};
                unbiased_exponent = most_significant_bit - 24;
                if (rounded[11]) begin
                    rounded = rounded >> 1;
                    unbiased_exponent = unbiased_exponent + 1;
                end
                if (unbiased_exponent > 15) begin
                    result = {value_sign, 5'h1e, 10'h3ff};
                    saturated = 1'b1;
                end else begin
                    encoded_exponent = 5'(unbiased_exponent + 15);
                    result = {
                        value_sign,
                        encoded_exponent,
                        rounded[9:0]
                    };
                end
            end else if (most_significant_bit >= 0) begin
                result = {
                    value_sign,
                    5'd0,
                    magnitude[9:0]
                };
            end

            convert_q24 = {saturated, result};
        end
    endfunction

    wire [16:0] converted_w = convert_q24(q24_i, zero_sign_i);
    assign f16_o = converted_w[15:0];
    assign saturation_o = converted_w[16];
endmodule

module ace3_fp16_add (
    input  wire [15:0] a_f16_i,
    input  wire [15:0] b_f16_i,
    output wire [15:0] sum_f16_o,
    output wire        invalid_operand_o,
    output wire        saturation_o
);
    wire signed [40:0] a_q24_w;
    wire signed [40:0] b_q24_w;
    wire a_finite_w;
    wire b_finite_w;
    wire a_sign_w;
    wire b_sign_w;
    wire a_zero_w = a_f16_i[14:0] == 15'd0;
    wire b_zero_w = b_f16_i[14:0] == 15'd0;
    wire signed [41:0] sum_q24_w =
        $signed({a_q24_w[40], a_q24_w}) +
        $signed({b_q24_w[40], b_q24_w});
    wire zero_sign_w = a_zero_w && b_zero_w && a_sign_w && b_sign_w;
    wire [15:0] rounded_sum_w;
    wire rounded_saturation_w;

    ace3_fp16_to_q24 decode_a (
        .f16_i(a_f16_i),
        .q24_o(a_q24_w),
        .finite_o(a_finite_w),
        .sign_o(a_sign_w)
    );
    ace3_fp16_to_q24 decode_b (
        .f16_i(b_f16_i),
        .q24_o(b_q24_w),
        .finite_o(b_finite_w),
        .sign_o(b_sign_w)
    );
    ace3_q24_to_fp16_rne #(
        .WIDTH(42)
    ) round_sum (
        .q24_i(sum_q24_w),
        .zero_sign_i(zero_sign_w),
        .f16_o(rounded_sum_w),
        .saturation_o(rounded_saturation_w)
    );

    assign invalid_operand_o = !a_finite_w || !b_finite_w;
    assign sum_f16_o = invalid_operand_o ? 16'h0000 : rounded_sum_w;
    assign saturation_o = invalid_operand_o ? 1'b0 : rounded_saturation_w;
endmodule

/* verilator lint_on DECLFILENAME */
`default_nettype wire
