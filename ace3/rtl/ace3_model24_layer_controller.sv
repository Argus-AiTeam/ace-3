`timescale 1ns/1ps
`default_nettype none

module ace3_model24_layer_controller (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        clear_i,

    input  wire        start_valid_i,
    output wire        start_ready_o,
    input  wire [1:0]  start_cache_slot_i,
    input  wire [14:0] start_position_i,

    output wire        layer_start_valid_o,
    input  wire        layer_start_ready_i,
    output wire [4:0]  layer_start_index_o,
    output wire [1:0]  layer_start_cache_slot_o,
    output wire [14:0] layer_start_position_o,

    input  wire        layer_done_valid_i,
    output wire        layer_done_ready_o,
    input  wire [4:0]  layer_done_index_i,
    input  wire        layer_done_fault_i,

    output wire        checkpoint_valid_o,
    input  wire        checkpoint_ready_i,
    output wire [4:0]  checkpoint_completed_layer_o,
    output wire [4:0]  checkpoint_next_layer_o,
    output wire        checkpoint_terminal_o,

    output wire        done_valid_o,
    input  wire        done_ready_i,
    output wire [1:0]  done_cache_slot_o,
    output wire [14:0] done_position_o,

    output wire        busy_o,
    output wire        fault_o,
    output wire [4:0]  active_layer_o
);
    localparam [2:0] S_IDLE       = 3'd0;
    localparam [2:0] S_LAYER_START = 3'd1;
    localparam [2:0] S_LAYER_RUN   = 3'd2;
    localparam [2:0] S_CHECKPOINT  = 3'd3;
    localparam [2:0] S_DONE        = 3'd4;

    reg [2:0] state_q;
    reg [4:0] active_layer_q;
    reg [1:0] cache_slot_q;
    reg [14:0] position_q;
    reg fault_q;

    function automatic known1;
        input value;
        begin
            known1 = (value === 1'b0) || (value === 1'b1);
        end
    endfunction

    function automatic known2;
        input [1:0] value;
        begin
            known2 = (^value === 1'b0) || (^value === 1'b1);
        end
    endfunction

    function automatic known5;
        input [4:0] value;
        begin
            known5 = (^value === 1'b0) || (^value === 1'b1);
        end
    endfunction

    function automatic known15;
        input [14:0] value;
        begin
            known15 = (^value === 1'b0) || (^value === 1'b1);
        end
    endfunction

    assign start_ready_o = !fault_q && (state_q == S_IDLE);
    assign layer_start_valid_o = !fault_q && (state_q == S_LAYER_START);
    assign layer_start_index_o = active_layer_q;
    assign layer_start_cache_slot_o = cache_slot_q;
    assign layer_start_position_o = position_q;
    assign layer_done_ready_o = !fault_q && (state_q == S_LAYER_RUN);
    assign checkpoint_valid_o = !fault_q && (state_q == S_CHECKPOINT);
    assign checkpoint_completed_layer_o = active_layer_q;
    assign checkpoint_next_layer_o = active_layer_q + 5'd1;
    assign checkpoint_terminal_o = active_layer_q == 5'd23;
    assign done_valid_o = !fault_q && (state_q == S_DONE);
    assign done_cache_slot_o = cache_slot_q;
    assign done_position_o = position_q;
    assign busy_o = !fault_q && (state_q != S_IDLE);
    assign fault_o = fault_q;
    assign active_layer_o = active_layer_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= S_IDLE;
            active_layer_q <= 5'd0;
            cache_slot_q <= 2'd0;
            position_q <= 15'd0;
            fault_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= S_IDLE;
            active_layer_q <= 5'd0;
            cache_slot_q <= 2'd0;
            position_q <= 15'd0;
            fault_q <= 1'b0;
        end else if (!fault_q) begin
            case (state_q)
                S_IDLE: begin
                    if (layer_done_valid_i === 1'b1) begin
                        fault_q <= 1'b1;
                    end else if ((start_valid_i === 1'b1) && start_ready_o) begin
                        if (!known2(start_cache_slot_i) ||
                            !known15(start_position_i) ||
                            (start_cache_slot_i >= 2) ||
                            (start_position_i >= 128)) begin
                            fault_q <= 1'b1;
                        end else begin
                            active_layer_q <= 5'd0;
                            cache_slot_q <= start_cache_slot_i;
                            position_q <= start_position_i;
                            state_q <= S_LAYER_START;
                        end
                    end
                end

                S_LAYER_START: begin
                    if (layer_done_valid_i === 1'b1) begin
                        fault_q <= 1'b1;
                    end else if (layer_start_ready_i === 1'b1) begin
                        state_q <= S_LAYER_RUN;
                    end
                end

                S_LAYER_RUN: begin
                    if (layer_done_valid_i === 1'b1) begin
                        if (!known5(layer_done_index_i) ||
                            !known1(layer_done_fault_i) ||
                            (layer_done_fault_i !== 1'b0) ||
                            (layer_done_index_i != active_layer_q)) begin
                            fault_q <= 1'b1;
                        end else begin
                            state_q <= S_CHECKPOINT;
                        end
                    end
                end

                S_CHECKPOINT: begin
                    if (layer_done_valid_i === 1'b1) begin
                        fault_q <= 1'b1;
                    end else if (checkpoint_ready_i === 1'b1) begin
                        if (active_layer_q == 5'd23) begin
                            state_q <= S_DONE;
                        end else begin
                            active_layer_q <= active_layer_q + 5'd1;
                            state_q <= S_LAYER_START;
                        end
                    end
                end

                S_DONE: begin
                    if (layer_done_valid_i === 1'b1) begin
                        fault_q <= 1'b1;
                    end else if (done_ready_i === 1'b1) begin
                        state_q <= S_IDLE;
                        active_layer_q <= 5'd0;
                    end
                end

                default: fault_q <= 1'b1;
            endcase
        end
    end
endmodule

`default_nettype wire
