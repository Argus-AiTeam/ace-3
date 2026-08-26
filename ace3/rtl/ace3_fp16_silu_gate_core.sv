`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_silu_gate_core #(
    parameter integer INTERMEDIATE_SIZE = 4864,
    parameter integer ACCURATE_SIGMOID = 0
) (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         clear_i,

    input  wire         start_valid_i,
    output wire         start_ready_o,
    input  wire [12:0]  element_count_i,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [15:0]  gate_f16_i,
    input  wire [15:0]  up_f16_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [15:0]  out_f16_o,
    output wire [12:0]  out_index_o,
    output wire         out_last_o,
    output wire         invalid_operand_o,
    output wire         saturation_o,
    output wire         busy_o
);
    localparam [40:0] ONE_Q24 = 41'd16777216;
    localparam [24:0] HALF_Q24 = 25'd8388608;

    reg active_q;
    reg [12:0] remaining_q;
    reg [12:0] index_q;
    reg out_valid_q;
    reg [15:0] out_f16_q;
    reg [12:0] out_index_q;
    reg out_last_q;
    reg out_invalid_q;
    reg out_saturation_q;

    wire signed [40:0] gate_q24_w;
    wire signed [40:0] up_q24_w;
    wire gate_finite_w;
    wire up_finite_w;
    wire gate_sign_w;
    wire up_sign_w;
    wire [40:0] gate_magnitude_w = gate_q24_w[40]
        ? (~gate_q24_w + 41'd1) : gate_q24_w;
    wire [65:0] sigmoid_numerator_w =
        {25'd0, gate_magnitude_w} << 24;
    wire [41:0] sigmoid_denominator_w =
        ({1'b0, ONE_Q24} + {1'b0, gate_magnitude_w}) << 1;
    wire [65:0] sigmoid_denominator_extended_w =
        {24'd0, sigmoid_denominator_w};
    wire [65:0] sigmoid_quotient_w =
        sigmoid_numerator_w / sigmoid_denominator_extended_w;
    wire [65:0] sigmoid_remainder_w =
        sigmoid_numerator_w % sigmoid_denominator_extended_w;
    wire sigmoid_increment_w =
        ({1'b0, sigmoid_remainder_w} << 1) >
            {1'b0, sigmoid_denominator_extended_w} ||
        ((({1'b0, sigmoid_remainder_w} << 1) ==
            {1'b0, sigmoid_denominator_extended_w}) &&
         sigmoid_quotient_w[0]);
    wire [25:0] sigmoid_term_extended_w =
        {1'b0, sigmoid_quotient_w[24:0]} +
        {{25{1'b0}}, sigmoid_increment_w};
    wire [24:0] sigmoid_term_w =
        (|sigmoid_quotient_w[65:25]) || sigmoid_term_extended_w[25]
            ? HALF_Q24 : sigmoid_term_extended_w[24:0];
    wire [24:0] sigmoid_q24_w = gate_q24_w[40]
        ? HALF_Q24 - sigmoid_term_w
        : HALF_Q24 + sigmoid_term_w;
    wire [24:0] accurate_sigmoid_q24_w;
    wire signed [81:0] gate_up_product_w = gate_q24_w * up_q24_w;
    wire signed [107:0] gated_product_w =
        gate_up_product_w * $signed(
            {1'b0, (ACCURATE_SIGMOID != 0)
                ? accurate_sigmoid_q24_w : sigmoid_q24_w}
        );
    wire signed [63:0] gated_q24_w;
    wire invalid_w = !gate_finite_w || !up_finite_w;
    wire [15:0] rounded_f16_w;
    wire rounded_saturation_w;
    wire zero_sign_w = gate_sign_w ^ up_sign_w;
    wire config_valid_w =
        (element_count_i != 13'd0) &&
        (element_count_i <= INTERMEDIATE_SIZE[12:0]);

    function automatic signed [63:0] round_q72_to_q24;
        input signed [107:0] value;
        reg value_sign;
        reg [107:0] magnitude;
        reg [59:0] retained;
        reg [60:0] rounded;
        reg guard_bit;
        reg sticky_bit;
        begin
            value_sign = value[107];
            magnitude = value_sign ? (~value + 108'd1) : value;
            retained = magnitude[107:48];
            guard_bit = magnitude[47];
            sticky_bit = |magnitude[46:0];
            rounded = {1'b0, retained} +
                {{60{1'b0}},
                 (guard_bit && (sticky_bit || retained[0]))};
            round_q72_to_q24 = value_sign
                ? -$signed({3'd0, rounded})
                : $signed({3'd0, rounded});
        end
    endfunction

    function automatic signed [63:0] round_shift_q24;
        input signed [127:0] value;
        reg value_sign;
        reg [127:0] magnitude;
        reg [103:0] retained;
        reg guard_bit;
        reg sticky_bit;
        begin
            value_sign = value[127];
            magnitude = value_sign ? (~value + 128'd1) : value;
            retained = magnitude[127:24];
            guard_bit = magnitude[23];
            sticky_bit = |magnitude[22:0];
            if (guard_bit && (sticky_bit || retained[0]))
                retained = retained + 104'd1;
            round_shift_q24 = value_sign
                ? -$signed(retained[63:0]) : $signed(retained[63:0]);
        end
    endfunction

    function automatic [24:0] exp_sigmoid_q24;
        input signed [40:0] gate;
        reg [40:0] magnitude;
        reg [40:0] remainder;
        reg [40:0] exponent;
        reg signed [63:0] polynomial;
        reg signed [127:0] product;
        reg signed [63:0] exponential;
        reg signed [63:0] shifted_exponential;
        reg signed [63:0] discarded_exponential;
        reg [88:0] division_numerator;
        reg [88:0] division_denominator;
        reg [88:0] quotient;
        reg [88:0] division_remainder;
        reg [24:0] negative_sigmoid;
        begin
            magnitude = gate[40] ? (~gate + 41'd1) : gate;
            exponent = magnitude / 41'd11629080;
            remainder = magnitude % 41'd11629080;
            polynomial = -64'sd3329;
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = 64'sd23302 +
                round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = -64'sd139810 + round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = 64'sd699051 + round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = -64'sd2796203 + round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = 64'sd8388608 + round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            polynomial = -64'sd16777216 + round_shift_q24(product);
            product = $signed({1'b0, remainder}) * polynomial;
            exponential = 64'sd16777216 + round_shift_q24(product);
            if (exponent >= 41'd63)
                exponential = 64'sd0;
            else if (exponent != 0) begin
                shifted_exponential = exponential >>> exponent;
                discarded_exponential = exponential -
                    (shifted_exponential <<< exponent);
                if ((discarded_exponential <<< 1) >
                        (64'sd1 <<< exponent) ||
                    (((discarded_exponential <<< 1) ==
                        (64'sd1 <<< exponent)) &&
                     shifted_exponential[0]))
                    shifted_exponential = shifted_exponential + 64'sd1;
                exponential = shifted_exponential;
            end
            division_numerator =
                {25'd0, exponential[63:0]} << 24;
            division_denominator =
                89'd16777216 + {25'd0, exponential[63:0]};
            quotient = division_numerator / division_denominator;
            division_remainder =
                division_numerator % division_denominator;
            if ((division_remainder << 1) >
                   division_denominator ||
                (((division_remainder << 1) ==
                   division_denominator) &&
                 quotient[0]))
                quotient = quotient + 89'd1;
            negative_sigmoid = quotient[24:0];
            exp_sigmoid_q24 = gate[40]
                ? negative_sigmoid : 25'd16777216 - negative_sigmoid;
        end
    endfunction

    assign accurate_sigmoid_q24_w = exp_sigmoid_q24(gate_q24_w);

    assign gated_q24_w = round_q72_to_q24(gated_product_w);

    ace3_fp16_to_q24 decode_gate (
        .f16_i(gate_f16_i),
        .q24_o(gate_q24_w),
        .finite_o(gate_finite_w),
        .sign_o(gate_sign_w)
    );

    ace3_fp16_to_q24 decode_up (
        .f16_i(up_f16_i),
        .q24_o(up_q24_w),
        .finite_o(up_finite_w),
        .sign_o(up_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(64)
    ) round_output (
        .q24_i(gated_q24_w),
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
