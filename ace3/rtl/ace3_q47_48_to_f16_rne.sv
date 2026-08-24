`timescale 1ns/1ps
`default_nettype none

module ace3_q47_48_to_f16_rne #(
    parameter integer ACC_WIDTH = 102
) (
    input  wire signed [ACC_WIDTH-1:0] fixed_i,
    output wire [15:0]                f16_o,
    output wire                       saturation_o
);
    wire [16:0] rounded_w;

    function automatic [16:0] round_q48_to_f16;
        input signed [ACC_WIDTH-1:0] fixed_value;
        reg value_sign;
        reg [ACC_WIDTH-1:0] magnitude;
        reg [11:0] retained;
        reg [12:0] rounded;
        reg guard_bit;
        reg sticky_bit;
        reg [15:0] result_bits;
        reg saturated;
        integer bit_index;
        integer most_significant_bit;
        integer right_shift;
        integer unbiased_exponent;
        reg [4:0] encoded_exponent;
        begin
            value_sign = fixed_value[ACC_WIDTH-1];
            magnitude = value_sign
                ? (~fixed_value + {{(ACC_WIDTH-1){1'b0}}, 1'b1})
                : fixed_value;
            retained = 12'd0;
            rounded = 13'd0;
            guard_bit = 1'b0;
            sticky_bit = 1'b0;
            result_bits = 16'h0000;
            saturated = 1'b0;
            most_significant_bit = -1;
            encoded_exponent = 5'd0;

            for (bit_index = 0; bit_index < ACC_WIDTH;
                 bit_index = bit_index + 1)
                if (magnitude[bit_index])
                    most_significant_bit = bit_index;

            if (most_significant_bit >= 34) begin
                right_shift = most_significant_bit - 10;
                retained = 12'd0;
                for (bit_index = 0; bit_index < 12;
                     bit_index = bit_index + 1)
                    if ((right_shift + bit_index) < ACC_WIDTH)
                        retained[bit_index] =
                            magnitude[right_shift + bit_index];
                guard_bit = magnitude[right_shift - 1];
                sticky_bit = 1'b0;
                for (bit_index = 0; bit_index < ACC_WIDTH;
                     bit_index = bit_index + 1)
                    if (bit_index < (right_shift - 1))
                        sticky_bit = sticky_bit | magnitude[bit_index];
                rounded = {1'b0, retained} +
                    {{12{1'b0}}, (guard_bit &&
                    (sticky_bit || retained[0]))};
                unbiased_exponent = most_significant_bit - 48;
                encoded_exponent =
                    most_significant_bit[4:0] - 5'd1;
                if (rounded[11]) begin
                    rounded = rounded >> 1;
                    unbiased_exponent = unbiased_exponent + 1;
                    encoded_exponent = encoded_exponent + 5'd1;
                end
                if (unbiased_exponent > 15) begin
                    result_bits = {value_sign, 5'h1e, 10'h3ff};
                    saturated = 1'b1;
                end else begin
                    result_bits = {
                        value_sign,
                        encoded_exponent,
                        rounded[9:0]
                    };
                end
            end else if (most_significant_bit >= 0) begin
                retained = magnitude[35:24];
                guard_bit = magnitude[23];
                sticky_bit = |magnitude[22:0];
                rounded = {1'b0, retained} +
                    {{12{1'b0}}, (guard_bit &&
                    (sticky_bit || retained[0]))};
                if (rounded >= 13'd1024)
                    result_bits = {value_sign, 5'd1, 10'd0};
                else if (rounded == 13'd0)
                    result_bits = 16'h0000;
                else
                    result_bits = {value_sign, 5'd0, rounded[9:0]};
            end
            round_q48_to_f16 = {saturated, result_bits};
        end
    endfunction

    assign rounded_w = round_q48_to_f16(fixed_i);
    assign f16_o = rounded_w[15:0];
    assign saturation_o = rounded_w[16];
endmodule

`default_nettype wire
