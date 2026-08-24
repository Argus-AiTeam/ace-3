`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_rmsnorm_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter [63:0] EPSILON_Q48 = 64'd281474977
) (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         clear_i,

    input  wire         start_valid_i,
    output wire         start_ready_o,
    input  wire [12:0]  element_count_i,

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
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_LOAD = 2'd1;
    localparam [1:0] ST_SQRT = 2'd2;
    localparam [1:0] ST_OUTPUT = 2'd3;
    localparam integer INDEX_WIDTH =
        (HIDDEN_SIZE <= 1) ? 1 : $clog2(HIDDEN_SIZE);
    localparam [INDEX_WIDTH-1:0] LAST_INDEX =
        INDEX_WIDTH'(HIDDEN_SIZE - 1);
    localparam [91:0] EPSILON_TOTAL =
        EPSILON_Q48 * HIDDEN_SIZE;
    localparam [91:0] HIDDEN_SIZE_92 = 92'(HIDDEN_SIZE);

    reg [1:0] state_q;
    reg [INDEX_WIDTH-1:0] input_index_q;
    reg [INDEX_WIDTH-1:0] output_index_q;
    reg [91:0] sumsq_q;
    reg [91:0] mean_q48_q;
    reg [45:0] sqrt_root_q;
    reg [5:0] sqrt_bit_q;
    reg invalid_seen_q;
    reg signed [40:0] activation_mem [0:HIDDEN_SIZE-1];
    reg signed [40:0] weight_mem [0:HIDDEN_SIZE-1];
    reg zero_sign_mem [0:HIDDEN_SIZE-1];

    wire signed [40:0] activation_q24_w;
    wire signed [40:0] weight_q24_w;
    wire activation_finite_w;
    wire weight_finite_w;
    wire activation_sign_w;
    wire weight_sign_w;
    wire [40:0] activation_magnitude_w = activation_q24_w[40]
        ? (~activation_q24_w + 41'd1) : activation_q24_w;
    wire [81:0] activation_square_w =
        activation_magnitude_w * activation_magnitude_w;
    wire [91:0] sumsq_next_w =
        sumsq_q + {{10{1'b0}}, activation_square_w};
    wire [91:0] mean_numerator_w = sumsq_next_w + EPSILON_TOTAL;
    wire [91:0] mean_quotient_w = mean_numerator_w / HIDDEN_SIZE_92;
    wire [91:0] mean_remainder_w = mean_numerator_w % HIDDEN_SIZE_92;
    wire mean_increment_w =
        ({1'b0, mean_remainder_w} << 1) > {1'b0, HIDDEN_SIZE_92} ||
        ((({1'b0, mean_remainder_w} << 1) ==
           {1'b0, HIDDEN_SIZE_92}) &&
         mean_quotient_w[0]);
    wire [91:0] mean_rounded_w =
        mean_quotient_w + {{91{1'b0}}, mean_increment_w};

    wire [45:0] sqrt_mask_w = 46'd1 << sqrt_bit_q;
    wire [45:0] sqrt_candidate_w = sqrt_root_q | sqrt_mask_w;
    wire [91:0] sqrt_candidate_square_w =
        sqrt_candidate_w * sqrt_candidate_w;
    wire sqrt_candidate_fits_w =
        sqrt_candidate_square_w <= mean_q48_q;

    wire signed [81:0] output_product_w =
        activation_mem[output_index_q] * weight_mem[output_index_q];
    wire output_product_sign_w = output_product_w[81];
    wire [81:0] output_product_magnitude_w = output_product_sign_w
        ? (~output_product_w + 82'd1) : output_product_w;
    wire [81:0] sqrt_root_extended_w = {{36{1'b0}}, sqrt_root_q};
    wire [81:0] output_quotient_w =
        output_product_magnitude_w / sqrt_root_extended_w;
    wire [81:0] output_remainder_w =
        output_product_magnitude_w % sqrt_root_extended_w;
    wire output_increment_w =
        ({1'b0, output_remainder_w} << 1) >
            {1'b0, sqrt_root_extended_w} ||
        ((({1'b0, output_remainder_w} << 1) ==
          {1'b0, sqrt_root_extended_w}) && output_quotient_w[0]);
    wire [82:0] output_rounded_magnitude_w =
        {1'b0, output_quotient_w} +
        {{82{1'b0}}, output_increment_w};
    wire signed [83:0] output_q24_w = output_product_sign_w
        ? -$signed({1'b0, output_rounded_magnitude_w})
        : $signed({1'b0, output_rounded_magnitude_w});
    wire [15:0] rounded_output_w;
    wire rounded_saturation_w;

    ace3_fp16_to_q24 decode_activation (
        .f16_i(activation_f16_i),
        .q24_o(activation_q24_w),
        .finite_o(activation_finite_w),
        .sign_o(activation_sign_w)
    );

    ace3_fp16_to_q24 decode_weight (
        .f16_i(weight_f16_i),
        .q24_o(weight_q24_w),
        .finite_o(weight_finite_w),
        .sign_o(weight_sign_w)
    );

    ace3_q24_to_fp16_rne #(
        .WIDTH(84)
    ) round_output (
        .q24_i(output_q24_w),
        .zero_sign_i(zero_sign_mem[output_index_q]),
        .f16_o(rounded_output_w),
        .saturation_o(rounded_saturation_w)
    );

    assign start_ready_o = rst_ni && !clear_i && (state_q == ST_IDLE) &&
                           (element_count_i == HIDDEN_SIZE[12:0]);
    assign in_ready_o = rst_ni && !clear_i && (state_q == ST_LOAD);
    assign out_valid_o = rst_ni && !clear_i && (state_q == ST_OUTPUT);
    assign out_f16_o = invalid_seen_q ? 16'h0000 : rounded_output_w;
    assign out_index_o =
        {{(13-INDEX_WIDTH){1'b0}}, output_index_q};
    assign out_last_o = output_index_q == LAST_INDEX;
    assign invalid_operand_o = invalid_seen_q;
    assign saturation_o = invalid_seen_q ? 1'b0 : rounded_saturation_w;
    assign busy_o = state_q != ST_IDLE;
    assign rms_q24_o = sqrt_root_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            input_index_q <= {INDEX_WIDTH{1'b0}};
            output_index_q <= {INDEX_WIDTH{1'b0}};
            sumsq_q <= 92'd0;
            mean_q48_q <= 92'd0;
            sqrt_root_q <= 46'd1;
            sqrt_bit_q <= 6'd45;
            invalid_seen_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            input_index_q <= {INDEX_WIDTH{1'b0}};
            output_index_q <= {INDEX_WIDTH{1'b0}};
            sumsq_q <= 92'd0;
            mean_q48_q <= 92'd0;
            sqrt_root_q <= 46'd1;
            sqrt_bit_q <= 6'd45;
            invalid_seen_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= ST_LOAD;
                        input_index_q <= {INDEX_WIDTH{1'b0}};
                        output_index_q <= {INDEX_WIDTH{1'b0}};
                        sumsq_q <= 92'd0;
                        mean_q48_q <= 92'd0;
                        sqrt_root_q <= 46'd0;
                        sqrt_bit_q <= 6'd45;
                        invalid_seen_q <= 1'b0;
                    end
                end

                ST_LOAD: begin
                    if (in_valid_i && in_ready_o) begin
                        activation_mem[input_index_q] <= activation_q24_w;
                        weight_mem[input_index_q] <= weight_q24_w;
                        zero_sign_mem[input_index_q] <=
                            activation_sign_w ^ weight_sign_w;
                        sumsq_q <= sumsq_next_w;
                        invalid_seen_q <= invalid_seen_q ||
                                          !activation_finite_w ||
                                          !weight_finite_w;
                        if (input_index_q == LAST_INDEX) begin
                            input_index_q <= {INDEX_WIDTH{1'b0}};
                            output_index_q <= {INDEX_WIDTH{1'b0}};
                            mean_q48_q <= mean_rounded_w;
                            sqrt_root_q <= 46'd0;
                            sqrt_bit_q <= 6'd45;
                            state_q <= ST_SQRT;
                        end else begin
                            input_index_q <= input_index_q +
                                {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end

                ST_SQRT: begin
                    if (sqrt_candidate_fits_w)
                        sqrt_root_q <= sqrt_candidate_w;
                    if (sqrt_bit_q == 6'd0) begin
                        output_index_q <= {INDEX_WIDTH{1'b0}};
                        state_q <= ST_OUTPUT;
                    end else begin
                        sqrt_bit_q <= sqrt_bit_q - 6'd1;
                    end
                end

                default: begin
                    if (out_valid_o && out_ready_i) begin
                        if (output_index_q == LAST_INDEX) begin
                            state_q <= ST_IDLE;
                            output_index_q <= {INDEX_WIDTH{1'b0}};
                        end else begin
                            output_index_q <= output_index_q +
                                {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end
            endcase
        end
    end
endmodule

`default_nettype wire
