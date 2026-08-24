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

/* verilator lint_on DECLFILENAME */
`default_nettype wire
