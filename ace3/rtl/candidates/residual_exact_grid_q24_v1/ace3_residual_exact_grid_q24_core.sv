`include "ace3_residual_exact_grid_q24_public.svh"

    localparam [12:0] MAX_ELEMENTS = 13'(VECTOR_SIZE);
    reg [12:0] count_q;
    reg [12:0] accepted_q;
    reg faulted_q;
    reg signed [40:0] decoded;
    reg signed [64:0] sum;
    reg [63:0] magnitude;
    reg [63:0] significand;
    reg [63:0] remainder_bits;
    reg [63:0] halfway;
    reg sign_bit;
    reg result_negzero;
    reg [15:0] result_f16;
    reg [3:0] fault_code;
    integer leading_bit;
    integer shift_bits;
    integer exponent_bits;
    integer bit_index;

    generate
        if (STATE_WIDTH != 64 || FRAC_BITS != 24 ||
            VECTOR_SIZE < 1 || VECTOR_SIZE > 8191) begin : unsupported_profile
            initial $fatal(1, "Q24 v1 requires width=64, frac=24, vector size 1..8191");
        end
    endgenerate

    assign start_ready_o = rst_ni && !clear_i && !busy_o && !faulted_q &&
                           ((^element_count_i) !== 1'bx) &&
                           element_count_i != 13'd0 && element_count_i <= MAX_ELEMENTS;
    assign in_ready_o = rst_ni && !clear_i && busy_o && !faulted_q &&
                        accepted_q < count_q &&
                        (!out_valid_o || (out_ready_i === 1'b1));

    always_comb begin
        decoded = 41'd0;
        if (in_addend_f16_i[14:10] == 5'd0)
            decoded = $signed({31'd0, in_addend_f16_i[9:0]});
        else
            decoded = $signed({30'd0, 1'b1, in_addend_f16_i[9:0]}) <<
                      (in_addend_f16_i[14:10] - 5'd1);
        if (in_addend_f16_i[15])
            decoded = -decoded;
        sum = $signed({in_state_q24_i[STATE_WIDTH-1], in_state_q24_i}) +
              $signed({{24{decoded[40]}}, decoded});
        result_negzero = (sum == 65'sd0) && (in_state_q24_i == 0) &&
                         in_state_negzero_i && (in_addend_f16_i == 16'h8000);
        sign_bit = sum[63];
        magnitude = sign_bit ? (~sum[63:0] + 64'd1) : sum[63:0];
        leading_bit = 0;
        shift_bits = 0;
        exponent_bits = 0;
        significand = 64'd0;
        remainder_bits = 64'd0;
        halfway = 64'd0;
        result_f16 = 16'd0;
        fault_code = 4'd0;

        if (magnitude == 64'd0)
            result_f16 = {result_negzero, 15'd0};
        else if (magnitude < 64'd1024)
            result_f16 = {sign_bit, 5'd0, magnitude[9:0]};
        else begin
            for (bit_index = 0; bit_index < 64; bit_index = bit_index + 1)
                if (magnitude[bit_index])
                    leading_bit = bit_index;
            shift_bits = leading_bit - 10;
            exponent_bits = leading_bit - 9;
            significand = magnitude >> shift_bits;
            if (shift_bits > 0) begin
                remainder_bits = magnitude & ((64'd1 << shift_bits) - 64'd1);
                halfway = 64'd1 << (shift_bits - 1);
                if (remainder_bits > halfway ||
                    (remainder_bits == halfway && significand[0]))
                    significand = significand + 64'd1;
            end
            if (significand == 64'd2048) begin
                significand = 64'd1024;
                exponent_bits = exponent_bits + 1;
            end
            if (exponent_bits >= 31)
                result_f16 = {sign_bit, 5'h1f, 10'd0};
            else
                result_f16 = {sign_bit, exponent_bits[4:0], significand[9:0]};
        end

        if ((^{in_state_q24_i, in_state_negzero_i, in_addend_f16_i}) === 1'bx ||
            in_addend_f16_i[14:10] == 5'h1f)
            fault_code = 4'd1;
        else if (in_state_q24_i != 0 && in_state_negzero_i)
            fault_code = 4'd2;
        else if (sum[64] != sum[63])
            fault_code = 4'd3;
        else if (result_f16[14:10] == 5'h1f)
            fault_code = 4'd4;
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (rst_ni === 1'b0) begin
            count_q <= 13'd0;
            accepted_q <= 13'd0;
            faulted_q <= 1'b0;
            busy_o <= 1'b0;
            out_valid_o <= 1'b0;
            out_state_q24_o <= {STATE_WIDTH{1'b0}};
            out_state_negzero_o <= 1'b0;
            out_f16_o <= 16'd0;
            out_index_o <= 13'd0;
            out_last_o <= 1'b0;
            out_fault_code_o <= 4'd0;
        end else if (rst_ni === 1'b1 && clear_i === 1'b1) begin
            count_q <= 13'd0;
            accepted_q <= 13'd0;
            faulted_q <= 1'b0;
            busy_o <= 1'b0;
            out_valid_o <= 1'b0;
            out_state_q24_o <= {STATE_WIDTH{1'b0}};
            out_state_negzero_o <= 1'b0;
            out_f16_o <= 16'd0;
            out_index_o <= 13'd0;
            out_last_o <= 1'b0;
            out_fault_code_o <= 4'd0;
        end else if (rst_ni === 1'b1 && clear_i === 1'b0) begin
            if (start_valid_i === 1'b1 && start_ready_o === 1'b1) begin
                count_q <= element_count_i;
                accepted_q <= 13'd0;
                busy_o <= 1'b1;
            end
            if (out_valid_o && out_ready_i === 1'b1) begin
                out_valid_o <= 1'b0;
                if (out_last_o)
                    busy_o <= 1'b0;
            end
            if (in_valid_i === 1'b1 && in_ready_o === 1'b1) begin
                out_valid_o <= 1'b1;
                out_index_o <= accepted_q;
                out_last_o <= (accepted_q == count_q - 13'd1) || fault_code != 4'd0;
                out_fault_code_o <= fault_code;
                accepted_q <= accepted_q + 13'd1;
                if (fault_code != 4'd0) begin
                    faulted_q <= 1'b1;
                    out_state_q24_o <= {STATE_WIDTH{1'b0}};
                    out_state_negzero_o <= 1'b0;
                    out_f16_o <= 16'd0;
                end else begin
                    out_state_q24_o <= sum[63:0];
                    out_state_negzero_o <= result_negzero;
                    out_f16_o <= result_f16;
                end
            end
        end
    end
endmodule
