`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_kv_cache #(
    parameter [2:0] CACHE_SLOTS = 3'd2,
    parameter [15:0] MAX_TOKENS = 16'd128,
    parameter [4:0] KV_HEADS = 5'd2,
    parameter [6:0] HEAD_DIM = 7'd64
) (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        write_valid_i,
    output wire        write_ready_o,
    input  wire [1:0]  write_cache_slot_i,
    input  wire [14:0] write_position_i,
    input  wire [3:0]  write_head_i,
    input  wire [5:0]  write_dimension_i,
    input  wire [15:0] write_k_f16_i,
    input  wire [15:0] write_v_f16_i,

    input  wire        read_valid_i,
    output wire        read_ready_o,
    input  wire [1:0]  read_cache_slot_i,
    input  wire [14:0] read_position_i,
    input  wire [3:0]  read_head_i,
    input  wire [5:0]  read_dimension_i,

    output wire        out_valid_o,
    input  wire        out_ready_i,
    output wire        out_hit_o,
    output wire [1:0]  out_cache_slot_o,
    output wire [14:0] out_position_o,
    output wire [3:0]  out_head_o,
    output wire [5:0]  out_dimension_o,
    output wire [15:0] out_k_f16_o,
    output wire [15:0] out_v_f16_o
);
    localparam integer ENTRY_COUNT =
        CACHE_SLOTS * MAX_TOKENS * KV_HEADS * HEAD_DIM;
    reg [15:0] k_mem [0:ENTRY_COUNT-1];
    reg [15:0] v_mem [0:ENTRY_COUNT-1];
    reg valid_mem [0:ENTRY_COUNT-1];

    wire write_config_valid_w =
        ({1'b0, write_cache_slot_i} < CACHE_SLOTS) &&
        ({1'b0, write_position_i} < MAX_TOKENS) &&
        ({1'b0, write_head_i} < KV_HEADS) &&
        ({1'b0, write_dimension_i} < HEAD_DIM);
    wire read_config_valid_w =
        ({1'b0, read_cache_slot_i} < CACHE_SLOTS) &&
        ({1'b0, read_position_i} < MAX_TOKENS) &&
        ({1'b0, read_head_i} < KV_HEADS) &&
        ({1'b0, read_dimension_i} < HEAD_DIM);
    wire [31:0] write_cache_slot_ext_w = {30'd0, write_cache_slot_i};
    wire [31:0] write_position_ext_w = {17'd0, write_position_i};
    wire [31:0] write_head_ext_w = {28'd0, write_head_i};
    wire [31:0] write_dimension_ext_w = {26'd0, write_dimension_i};
    wire [31:0] read_cache_slot_ext_w = {30'd0, read_cache_slot_i};
    wire [31:0] read_position_ext_w = {17'd0, read_position_i};
    wire [31:0] read_head_ext_w = {28'd0, read_head_i};
    wire [31:0] read_dimension_ext_w = {26'd0, read_dimension_i};
    wire [31:0] write_address_w =
        (((write_cache_slot_ext_w * MAX_TOKENS + write_position_ext_w) *
          KV_HEADS + write_head_ext_w) * HEAD_DIM) + write_dimension_ext_w;
    wire [31:0] read_address_w =
        (((read_cache_slot_ext_w * MAX_TOKENS + read_position_ext_w) *
          KV_HEADS + read_head_ext_w) * HEAD_DIM) + read_dimension_ext_w;
    wire simultaneous_same_address_w =
        write_valid_i && write_ready_o &&
        (write_address_w == read_address_w);

    reg out_valid_q;
    reg out_hit_q;
    reg [1:0] out_cache_slot_q;
    reg [14:0] out_position_q;
    reg [3:0] out_head_q;
    reg [5:0] out_dimension_q;
    reg [15:0] out_k_f16_q;
    reg [15:0] out_v_f16_q;
    integer entry_index;

    assign write_ready_o = rst_ni && !clear_i && write_config_valid_w;
    assign read_ready_o = rst_ni && !clear_i && read_config_valid_w &&
        (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign out_hit_o = out_hit_q;
    assign out_cache_slot_o = out_cache_slot_q;
    assign out_position_o = out_position_q;
    assign out_head_o = out_head_q;
    assign out_dimension_o = out_dimension_q;
    assign out_k_f16_o = out_k_f16_q;
    assign out_v_f16_o = out_v_f16_q;

    /* verilator lint_off BLKSEQ */
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_valid_q <= 1'b0;
            out_hit_q <= 1'b0;
            out_cache_slot_q <= 2'd0;
            out_position_q <= 15'd0;
            out_head_q <= 4'd0;
            out_dimension_q <= 6'd0;
            out_k_f16_q <= 16'd0;
            out_v_f16_q <= 16'd0;
            for (entry_index = 0; entry_index < ENTRY_COUNT;
                 entry_index = entry_index + 1)
                valid_mem[entry_index] = 1'b0;
        end else if (clear_i) begin
            out_valid_q <= 1'b0;
            out_hit_q <= 1'b0;
            out_cache_slot_q <= 2'd0;
            out_position_q <= 15'd0;
            out_head_q <= 4'd0;
            out_dimension_q <= 6'd0;
            out_k_f16_q <= 16'd0;
            out_v_f16_q <= 16'd0;
            for (entry_index = 0; entry_index < ENTRY_COUNT;
                 entry_index = entry_index + 1)
                valid_mem[entry_index] = 1'b0;
        end else begin
            if (read_valid_i && read_ready_o) begin
                out_valid_q <= 1'b1;
                out_cache_slot_q <= read_cache_slot_i;
                out_position_q <= read_position_i;
                out_head_q <= read_head_i;
                out_dimension_q <= read_dimension_i;
                if (simultaneous_same_address_w) begin
                    out_hit_q <= 1'b1;
                    out_k_f16_q <= write_k_f16_i;
                    out_v_f16_q <= write_v_f16_i;
                end else if (valid_mem[read_address_w]) begin
                    out_hit_q <= 1'b1;
                    out_k_f16_q <= k_mem[read_address_w];
                    out_v_f16_q <= v_mem[read_address_w];
                end else begin
                    out_hit_q <= 1'b0;
                    out_k_f16_q <= 16'd0;
                    out_v_f16_q <= 16'd0;
                end
            end else if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
            end

            if (write_valid_i && write_ready_o) begin
                k_mem[write_address_w] <= write_k_f16_i;
                v_mem[write_address_w] <= write_v_f16_i;
                valid_mem[write_address_w] <= 1'b1;
            end
        end
    end
    /* verilator lint_on BLKSEQ */
endmodule

`default_nettype wire
