// Compile-only port binding. There is no clock generator or vector execution.
module ace3_residual_exact_grid_q24_elab;
    wire start_ready;
    wire in_ready;
    wire out_valid;
    wire signed [63:0] state_q24;
    wire negzero;
    wire [15:0] f16;
    wire [12:0] index_value;
    wire last_value;
    wire [3:0] fault;
    wire busy;

    ace3_residual_exact_grid_q24_core #(
        .VECTOR_SIZE(896), .STATE_WIDTH(64), .FRAC_BITS(24)
    ) dut (
        .clk_i(1'b0), .rst_ni(1'b0), .clear_i(1'b0),
        .start_valid_i(1'b0), .start_ready_o(start_ready), .element_count_i(13'd896),
        .in_valid_i(1'b0), .in_ready_o(in_ready),
        .in_state_q24_i(64'sd0), .in_state_negzero_i(1'b0), .in_addend_f16_i(16'd0),
        .out_valid_o(out_valid), .out_ready_i(1'b0),
        .out_state_q24_o(state_q24), .out_state_negzero_o(negzero), .out_f16_o(f16),
        .out_index_o(index_value), .out_last_o(last_value), .out_fault_code_o(fault),
        .busy_o(busy)
    );
endmodule
