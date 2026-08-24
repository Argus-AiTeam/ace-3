`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_g128_dot_lane (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire [2:0]           logical_lane_i,
    input  wire [31:0]          qzeros_i,
    input  wire [15:0]          scale_f16_i,

    input  wire                 pair_valid_i,
    output wire                 pair_ready_o,
    input  wire [15:0]          activation_f16_i,
    input  wire [31:0]          qweight_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire [15:0]          out_f16_o,
    output wire signed [95:0]   acc_q47_48_o,
    output wire                 invalid_operand_o,
    output wire                 saturation_o
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;
    localparam [1:0] ST_DONE = 2'd2;

    reg [1:0] state_q;
    reg [6:0] pair_count_q;
    reg [2:0] logical_lane_q;
    reg [3:0] qzero_nibble_q;
    reg [15:0] scale_f16_q;
    reg signed [95:0] accumulator_q;
    reg invalid_q;
    reg out_valid_q;
    reg [15:0] out_f16_q;
    reg out_invalid_q;
    reg out_saturation_q;

    reg [3:0] qweight_nibble_w;
    reg signed [5:0] quantized_delta_w;
    reg [4:0] delta_magnitude_w;
    reg [10:0] activation_significand_w;
    reg [10:0] scale_significand_w;
    reg [21:0] significand_product_w;
    reg [26:0] product_magnitude_w;
    reg [95:0] shifted_magnitude_w;
    reg signed [95:0] addend_w;
    reg product_negative_w;
    reg operand_invalid_w;
    integer activation_exponent_w;
    integer scale_exponent_w;
    integer accumulator_shift_w;

    wire signed [95:0] accumulated_w;
    wire [16:0] rounded_w;
    wire completion_invalid_w;

    function automatic [3:0] select_awq_nibble;
        input [31:0] packed_word;
        input [2:0] logical_lane;
        begin
            case (logical_lane)
                3'd0: select_awq_nibble = packed_word[3:0];
                3'd1: select_awq_nibble = packed_word[19:16];
                3'd2: select_awq_nibble = packed_word[7:4];
                3'd3: select_awq_nibble = packed_word[23:20];
                3'd4: select_awq_nibble = packed_word[11:8];
                3'd5: select_awq_nibble = packed_word[27:24];
                3'd6: select_awq_nibble = packed_word[15:12];
                default: select_awq_nibble = packed_word[31:28];
            endcase
        end
    endfunction

    function automatic [16:0] round_q47_48_to_f16;
        input signed [95:0] fixed_value;
        reg value_sign;
        reg [95:0] magnitude;
        reg [11:0] retained;
        reg [12:0] rounded;
        reg guard_bit;
        reg sticky_bit;
        reg [15:0] result_bits;
        reg [4:0] result_exponent;
        reg saturated;
        integer bit_index;
        integer most_significant_bit;
        integer right_shift;
        integer unbiased_exponent;
        begin
            value_sign = fixed_value[95];
            magnitude = value_sign ? (~fixed_value + 96'd1) : fixed_value;
            retained = 12'd0;
            rounded = 13'd0;
            guard_bit = 1'b0;
            sticky_bit = 1'b0;
            result_bits = 16'h0000;
            result_exponent = 5'd0;
            saturated = 1'b0;
            most_significant_bit = -1;

            for (bit_index = 0; bit_index < 96; bit_index = bit_index + 1)
                if (magnitude[bit_index])
                    most_significant_bit = bit_index;

            if (most_significant_bit >= 34) begin
                right_shift = most_significant_bit - 10;
                retained = magnitude[right_shift +: 12];
                guard_bit = magnitude[right_shift - 1];
                sticky_bit = 1'b0;
                for (bit_index = 0; bit_index < 96; bit_index = bit_index + 1)
                    if (bit_index < (right_shift - 1))
                        sticky_bit = sticky_bit | magnitude[bit_index];
                rounded = {1'b0, retained} +
                    {{12{1'b0}}, (guard_bit &&
                    (sticky_bit || retained[0]))};
                unbiased_exponent = most_significant_bit - 48;
                result_exponent =
                    most_significant_bit[4:0] - 5'd1;
                if (rounded[11]) begin
                    rounded = rounded >> 1;
                    unbiased_exponent = unbiased_exponent + 1;
                    result_exponent = result_exponent + 5'd1;
                end
                if (unbiased_exponent > 15) begin
                    result_bits = {value_sign, 5'h1e, 10'h3ff};
                    saturated = 1'b1;
                end else begin
                    result_bits = {
                        value_sign,
                        result_exponent,
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
            round_q47_48_to_f16 = {saturated, result_bits};
        end
    endfunction

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign pair_ready_o = (state_q == ST_RUN);
    assign out_valid_o = out_valid_q;
    assign out_f16_o = out_f16_q;
    assign acc_q47_48_o = accumulator_q;
    assign invalid_operand_o = out_invalid_q;
    assign saturation_o = out_saturation_q;
    assign accumulated_w = accumulator_q + addend_w;
    assign rounded_w = round_q47_48_to_f16(accumulated_w);
    assign completion_invalid_w = invalid_q || operand_invalid_w;

    always @* begin
        qweight_nibble_w = select_awq_nibble(qweight_i, logical_lane_q);
        quantized_delta_w =
            $signed({1'b0, qweight_nibble_w}) -
            $signed({1'b0, qzero_nibble_q});
        delta_magnitude_w = quantized_delta_w[5]
            ? (~quantized_delta_w[4:0] + 5'd1)
            : quantized_delta_w[4:0];

        if (activation_f16_i[14:10] == 5'd0) begin
            activation_significand_w = {1'b0, activation_f16_i[9:0]};
            activation_exponent_w = -24;
        end else begin
            activation_significand_w = {1'b1, activation_f16_i[9:0]};
            activation_exponent_w =
                {27'd0, activation_f16_i[14:10]} - 32'sd25;
        end
        if (scale_f16_q[14:10] == 5'd0) begin
            scale_significand_w = {1'b0, scale_f16_q[9:0]};
            scale_exponent_w = -24;
        end else begin
            scale_significand_w = {1'b1, scale_f16_q[9:0]};
            scale_exponent_w =
                {27'd0, scale_f16_q[14:10]} - 32'sd25;
        end

        operand_invalid_w =
            (activation_f16_i[14:10] == 5'h1f) ||
            (scale_f16_q[14:10] == 5'h1f);
        product_negative_w =
            activation_f16_i[15] ^ scale_f16_q[15] ^ quantized_delta_w[5];
        significand_product_w =
            activation_significand_w * scale_significand_w;
        product_magnitude_w = significand_product_w * delta_magnitude_w;
        accumulator_shift_w =
            activation_exponent_w + scale_exponent_w + 48;
        shifted_magnitude_w = 96'd0;
        addend_w = 96'sd0;
        if (!operand_invalid_w && (product_magnitude_w != 27'd0)) begin
            shifted_magnitude_w =
                {{69{1'b0}}, product_magnitude_w} << accumulator_shift_w;
            addend_w = product_negative_w
                ? -$signed(shifted_magnitude_w)
                : $signed(shifted_magnitude_w);
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            pair_count_q <= 7'd0;
            logical_lane_q <= 3'd0;
            qzero_nibble_q <= 4'd0;
            scale_f16_q <= 16'd0;
            accumulator_q <= 96'sd0;
            invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_f16_q <= 16'd0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            pair_count_q <= 7'd0;
            logical_lane_q <= 3'd0;
            qzero_nibble_q <= 4'd0;
            scale_f16_q <= 16'd0;
            accumulator_q <= 96'sd0;
            invalid_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_f16_q <= 16'd0;
            out_invalid_q <= 1'b0;
            out_saturation_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= ST_RUN;
                        pair_count_q <= 7'd0;
                        logical_lane_q <= logical_lane_i;
                        qzero_nibble_q <=
                            select_awq_nibble(qzeros_i, logical_lane_i);
                        scale_f16_q <= scale_f16_i;
                        accumulator_q <= 96'sd0;
                        invalid_q <= (scale_f16_i[14:10] == 5'h1f);
                        out_invalid_q <= 1'b0;
                        out_saturation_q <= 1'b0;
                    end
                end
                ST_RUN: begin
                    if (pair_valid_i && pair_ready_o) begin
                        accumulator_q <= accumulated_w;
                        invalid_q <= completion_invalid_w;
                        if (pair_count_q == 7'd127) begin
                            state_q <= ST_DONE;
                            out_valid_q <= 1'b1;
                            out_invalid_q <= completion_invalid_w;
                            if (completion_invalid_w) begin
                                out_f16_q <= 16'h0000;
                                out_saturation_q <= 1'b0;
                            end else begin
                                out_f16_q <= rounded_w[15:0];
                                out_saturation_q <= rounded_w[16];
                            end
                        end else begin
                            pair_count_q <= pair_count_q + 7'd1;
                        end
                    end
                end
                default: begin
                    if (out_valid_q && out_ready_i) begin
                        out_valid_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end
                end
            endcase
        end
    end
endmodule

`default_nettype wire
