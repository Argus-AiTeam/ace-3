module ace3_residual_exact_grid_q24_core #(
    parameter integer VECTOR_SIZE = 896,
    parameter integer STATE_WIDTH = 64,
    parameter integer FRAC_BITS = 24
) (
    input  wire clk_i,
    input  wire rst_ni,
    input  wire clear_i,
    input  wire start_valid_i,
    output wire start_ready_o,
    input  wire [12:0] element_count_i,
    input  wire in_valid_i,
    output wire in_ready_o,
    input  wire signed [STATE_WIDTH-1:0] in_state_q24_i,
    input  wire in_state_negzero_i,
    input  wire [15:0] in_addend_f16_i,
    output reg out_valid_o,
    input  wire out_ready_i,
    output reg signed [STATE_WIDTH-1:0] out_state_q24_o,
    output reg out_state_negzero_o,
    output reg [15:0] out_f16_o,
    output reg [12:0] out_index_o,
    output reg out_last_o,
    output reg [3:0] out_fault_code_o,
    output reg busy_o
);
