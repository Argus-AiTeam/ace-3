`timescale 1ns/1ps
`default_nettype none

module ace3_qwen2_rope_pair (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        in_valid_i,
    output wire        in_ready_o,
    input  wire        is_key_i,
    input  wire [3:0]  head_index_i,
    input  wire [4:0]  pair_index_i,
    input  wire [14:0] position_i,
    input  wire [15:0] low_f16_i,
    input  wire [15:0] high_f16_i,
    input  wire [15:0] cos_f16_i,
    input  wire [15:0] sin_f16_i,

    output wire        out_valid_o,
    input  wire        out_ready_i,
    output wire        is_key_o,
    output wire [3:0]  head_index_o,
    output wire [4:0]  pair_index_o,
    output wire [14:0] position_o,
    output wire [15:0] low_f16_o,
    output wire [15:0] high_f16_o,
    output wire        invalid_operand_o,
    output wire        saturation_o
);
    wire signed [40:0] low_q24_w;
    wire signed [40:0] high_q24_w;
    wire signed [40:0] cos_q24_w;
    wire signed [40:0] sin_q24_w;
    wire low_finite_w;
    wire high_finite_w;
    wire cos_finite_w;
    wire sin_finite_w;
    wire low_sign_w;
    wire high_sign_w;
    wire cos_sign_w;
    wire sin_sign_w;

    ace3_fp16_to_q24 decode_low (
        .f16_i(low_f16_i),
        .q24_o(low_q24_w),
        .finite_o(low_finite_w),
        .sign_o(low_sign_w)
    );
    ace3_fp16_to_q24 decode_high (
        .f16_i(high_f16_i),
        .q24_o(high_q24_w),
        .finite_o(high_finite_w),
        .sign_o(high_sign_w)
    );
    ace3_fp16_to_q24 decode_cos (
        .f16_i(cos_f16_i),
        .q24_o(cos_q24_w),
        .finite_o(cos_finite_w),
        .sign_o(cos_sign_w)
    );
    ace3_fp16_to_q24 decode_sin (
        .f16_i(sin_f16_i),
        .q24_o(sin_q24_w),
        .finite_o(sin_finite_w),
        .sign_o(sin_sign_w)
    );

    wire signed [81:0] low_cos_product_w =
        $signed(low_q24_w) * $signed(cos_q24_w);
    wire signed [81:0] neg_high_sin_product_w =
        -($signed(high_q24_w) * $signed(sin_q24_w));
    wire signed [81:0] high_cos_product_w =
        $signed(high_q24_w) * $signed(cos_q24_w);
    wire signed [81:0] low_sin_product_w =
        $signed(low_q24_w) * $signed(sin_q24_w);

    wire [15:0] low_cos_raw_w;
    wire [15:0] neg_high_sin_raw_w;
    wire [15:0] high_cos_raw_w;
    wire [15:0] low_sin_raw_w;
    wire low_cos_saturation_w;
    wire neg_high_sin_saturation_w;
    wire high_cos_saturation_w;
    wire low_sin_saturation_w;

    ace3_q47_48_to_f16_rne #(.ACC_WIDTH(82)) round_low_cos (
        .fixed_i(low_cos_product_w),
        .f16_o(low_cos_raw_w),
        .saturation_o(low_cos_saturation_w)
    );
    ace3_q47_48_to_f16_rne #(.ACC_WIDTH(82)) round_neg_high_sin (
        .fixed_i(neg_high_sin_product_w),
        .f16_o(neg_high_sin_raw_w),
        .saturation_o(neg_high_sin_saturation_w)
    );
    ace3_q47_48_to_f16_rne #(.ACC_WIDTH(82)) round_high_cos (
        .fixed_i(high_cos_product_w),
        .f16_o(high_cos_raw_w),
        .saturation_o(high_cos_saturation_w)
    );
    ace3_q47_48_to_f16_rne #(.ACC_WIDTH(82)) round_low_sin (
        .fixed_i(low_sin_product_w),
        .f16_o(low_sin_raw_w),
        .saturation_o(low_sin_saturation_w)
    );

    wire [15:0] low_cos_f16_w = (low_cos_product_w == 82'sd0)
        ? {low_sign_w ^ cos_sign_w, 15'd0} : low_cos_raw_w;
    wire [15:0] neg_high_sin_f16_w =
        (neg_high_sin_product_w == 82'sd0)
        ? {high_sign_w ^ sin_sign_w ^ 1'b1, 15'd0}
        : neg_high_sin_raw_w;
    wire [15:0] high_cos_f16_w = (high_cos_product_w == 82'sd0)
        ? {high_sign_w ^ cos_sign_w, 15'd0} : high_cos_raw_w;
    wire [15:0] low_sin_f16_w = (low_sin_product_w == 82'sd0)
        ? {low_sign_w ^ sin_sign_w, 15'd0} : low_sin_raw_w;

    wire signed [40:0] low_cos_q24_w;
    wire signed [40:0] neg_high_sin_q24_w;
    wire signed [40:0] high_cos_q24_w;
    wire signed [40:0] low_sin_q24_w;
    wire unused_finite_0;
    wire unused_finite_1;
    wire unused_finite_2;
    wire unused_finite_3;
    wire unused_sign_0;
    wire unused_sign_1;
    wire unused_sign_2;
    wire unused_sign_3;

    ace3_fp16_to_q24 decode_low_cos (
        .f16_i(low_cos_f16_w),
        .q24_o(low_cos_q24_w),
        .finite_o(unused_finite_0),
        .sign_o(unused_sign_0)
    );
    ace3_fp16_to_q24 decode_neg_high_sin (
        .f16_i(neg_high_sin_f16_w),
        .q24_o(neg_high_sin_q24_w),
        .finite_o(unused_finite_1),
        .sign_o(unused_sign_1)
    );
    ace3_fp16_to_q24 decode_high_cos (
        .f16_i(high_cos_f16_w),
        .q24_o(high_cos_q24_w),
        .finite_o(unused_finite_2),
        .sign_o(unused_sign_2)
    );
    ace3_fp16_to_q24 decode_low_sin (
        .f16_i(low_sin_f16_w),
        .q24_o(low_sin_q24_w),
        .finite_o(unused_finite_3),
        .sign_o(unused_sign_3)
    );

    wire signed [41:0] low_sum_q24_w =
        $signed(low_cos_q24_w) + $signed(neg_high_sin_q24_w);
    wire signed [41:0] high_sum_q24_w =
        $signed(high_cos_q24_w) + $signed(low_sin_q24_w);
    wire low_zero_sign_w =
        (low_cos_f16_w[14:0] == 15'd0) &&
        (neg_high_sin_f16_w[14:0] == 15'd0) &&
        low_cos_f16_w[15] && neg_high_sin_f16_w[15];
    wire high_zero_sign_w =
        (high_cos_f16_w[14:0] == 15'd0) &&
        (low_sin_f16_w[14:0] == 15'd0) &&
        high_cos_f16_w[15] && low_sin_f16_w[15];
    wire [15:0] rotated_low_w;
    wire [15:0] rotated_high_w;
    wire low_final_saturation_w;
    wire high_final_saturation_w;

    ace3_q24_to_fp16_rne #(.WIDTH(42)) round_low_sum (
        .q24_i(low_sum_q24_w),
        .zero_sign_i(low_zero_sign_w),
        .f16_o(rotated_low_w),
        .saturation_o(low_final_saturation_w)
    );
    ace3_q24_to_fp16_rne #(.WIDTH(42)) round_high_sum (
        .q24_i(high_sum_q24_w),
        .zero_sign_i(high_zero_sign_w),
        .f16_o(rotated_high_w),
        .saturation_o(high_final_saturation_w)
    );

    wire operands_finite_w =
        low_finite_w && high_finite_w && cos_finite_w && sin_finite_w;
    wire geometry_valid_w =
        ((!is_key_i && (head_index_i < 4'd14)) ||
         (is_key_i && (head_index_i < 4'd2)));
    wire result_saturation_w =
        low_cos_saturation_w || neg_high_sin_saturation_w ||
        high_cos_saturation_w || low_sin_saturation_w ||
        low_final_saturation_w || high_final_saturation_w;

    reg out_valid_q;
    reg is_key_q;
    reg [3:0] head_index_q;
    reg [4:0] pair_index_q;
    reg [14:0] position_q;
    reg [15:0] low_f16_q;
    reg [15:0] high_f16_q;
    reg invalid_q;
    reg saturation_q;

    assign in_ready_o = rst_ni && !clear_i && geometry_valid_w &&
        (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign is_key_o = is_key_q;
    assign head_index_o = head_index_q;
    assign pair_index_o = pair_index_q;
    assign position_o = position_q;
    assign low_f16_o = low_f16_q;
    assign high_f16_o = high_f16_q;
    assign invalid_operand_o = invalid_q;
    assign saturation_o = saturation_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_valid_q <= 1'b0;
            is_key_q <= 1'b0;
            head_index_q <= 4'd0;
            pair_index_q <= 5'd0;
            position_q <= 15'd0;
            low_f16_q <= 16'd0;
            high_f16_q <= 16'd0;
            invalid_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (clear_i) begin
            out_valid_q <= 1'b0;
            is_key_q <= 1'b0;
            head_index_q <= 4'd0;
            pair_index_q <= 5'd0;
            position_q <= 15'd0;
            low_f16_q <= 16'd0;
            high_f16_q <= 16'd0;
            invalid_q <= 1'b0;
            saturation_q <= 1'b0;
        end else begin
            if (in_valid_i && in_ready_o) begin
                out_valid_q <= 1'b1;
                is_key_q <= is_key_i;
                head_index_q <= head_index_i;
                pair_index_q <= pair_index_i;
                position_q <= position_i;
                invalid_q <= !operands_finite_w;
                if (!operands_finite_w) begin
                    low_f16_q <= 16'd0;
                    high_f16_q <= 16'd0;
                    saturation_q <= 1'b0;
                end else begin
                    low_f16_q <= rotated_low_w;
                    high_f16_q <= rotated_high_w;
                    saturation_q <= result_saturation_w;
                end
            end else if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
