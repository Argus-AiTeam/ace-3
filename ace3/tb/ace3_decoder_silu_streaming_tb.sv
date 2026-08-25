`timescale 1ns/1ps
`default_nettype none

module ace3_decoder_silu_streaming_tb (
    input wire clk_i
);
    wire clk = clk_i;
    reg rst_n = 1'b0;
    integer cycles = 0;
    integer phase32_cycles = 0;
    integer trace_count = 0;
    integer backpressure_cycles = 0;
    integer index;
    reg saw_phase33 = 1'b0;
    reg saw_successor_phase = 1'b0;

    wire [2:0] projection_kind;
    wire trace_valid;
    reg trace_ready = 1'b1;
    wire [4:0] trace_stage;
    wire [12:0] trace_index;
    wire [15:0] trace_f16;
    wire [5:0] phase;

    ace3_decoder_layer0_token_engine dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(1'b0),
        .load_valid_i(1'b0), .load_ready_o(), .load_kind_i(2'd0),
        .load_index_i(13'd0), .load_f16_i(16'd0),
        .start_valid_i(1'b0), .start_ready_o(),
        .start_cache_slot_i(2'd0), .start_position_i(15'd0),
        .projection_kind_o(projection_kind),
        .projection_meta_valid_i(1'b0), .projection_meta_ready_o(),
        .projection_meta_output_channel_o(), .projection_meta_group_o(),
        .projection_meta_word_o(), .projection_meta_lane_o(),
        .projection_qzeros_i(32'd0), .projection_scale_f16_i(16'd0),
        .projection_pair_valid_i(1'b0), .projection_pair_ready_o(),
        .projection_pair_input_o(), .projection_pair_output_o(),
        .projection_pair_group_o(), .projection_pair_word_o(),
        .projection_pair_lane_o(), .projection_qweight_i(32'd0),
        .projection_bias_valid_i(1'b0), .projection_bias_ready_o(),
        .projection_bias_output_channel_o(), .projection_bias_f16_i(16'd0),
        .rope_valid_i(1'b0), .rope_ready_o(), .rope_position_o(),
        .rope_pair_o(), .rope_cos_f16_i(16'd0), .rope_sin_f16_i(16'd0),
        .trace_valid_o(trace_valid), .trace_ready_i(trace_ready),
        .trace_stage_o(trace_stage), .trace_index_o(trace_index),
        .trace_f16_o(trace_f16), .trace_position_o(),
        .final_valid_o(), .final_ready_i(1'b0), .final_index_o(),
        .final_f16_o(), .final_last_o(), .done_valid_o(),
        .done_ready_i(1'b0), .done_cache_slot_o(), .done_position_o(),
        .done_cycles_o(), .done_stall_cycles_o(), .busy_o(),
        .phase_o(phase)
    );

    always @(negedge clk) begin
        if (cycles < 3)
            rst_n = 1'b0;
        else
            rst_n = 1'b1;
        trace_ready = (cycles % 7) != 0;
        if (cycles == 3) begin
            dut.state_q <= 6'd31;
            dut.psel_q <= 3'd5;
        end
        if (phase == 6'd4 && projection_kind == 3'd6)
            saw_successor_phase = 1'b1;
        if (cycles >= 3 && saw_successor_phase && trace_count == 4864) begin
            if (projection_kind !== 3'd6 || !saw_phase33 ||
                dut.si_output_count_q != 13'd4864 ||
                dut.intermediate_index_q != 13'd4863 ||
                backpressure_cycles == 0)
                $fatal(1,
                    "DECODER_SILU_STREAMING_PROGRESS_FAIL phase33=%0d trace=%0d outputs=%0d input_last=%0d backpressure=%0d",
                    saw_phase33, trace_count, dut.si_output_count_q,
                    dut.intermediate_index_q, backpressure_cycles);
            $display(
                "DECODER_SILU_STREAMING_PASS projection_kind=5 phase=32 inputs=4864 outputs=4864 traces=4864 phase33=observed successor_projection_kind=6 successor_phase=4 backpressure_cycles=%0d phase32_cycles=%0d value=3e00",
                backpressure_cycles, phase32_cycles);
            $finish;
        end
        if (cycles >= 20000)
            $fatal(1,
                "DECODER_SILU_STREAMING_TIMEOUT cycles=%0d phase=%0d projection_kind=%0d trace=%0d",
                cycles, phase, projection_kind, trace_count);
    end

    always @(posedge clk) begin
        if (phase == 6'd32)
            phase32_cycles = phase32_cycles + 1;
        if (phase == 6'd32 && !trace_ready)
            backpressure_cycles = backpressure_cycles + 1;
        if (phase == 6'd33)
            saw_phase33 = 1'b1;
        if (trace_valid && trace_ready) begin
            if (trace_stage !== 5'd16 || trace_index !== trace_count[12:0] ||
                trace_f16 !== 16'h3e00)
                $fatal(1,
                    "DECODER_SILU_STREAMING_TRACE_FAIL count=%0d stage=%0d index=%0d value=%04x",
                    trace_count, trace_stage, trace_index, trace_f16);
            trace_count = trace_count + 1;
        end
        if (phase == 6'd63)
            $fatal(1, "DECODER_SILU_STREAMING_FAULT cycle=%0d", cycles);
        cycles = cycles + 1;
    end

    initial begin
        for (index = 0; index < 4864; index = index + 1) begin
            dut.gate_mem[index] = 16'h3c00;
            dut.up_mem[index] = 16'h4000;
        end
    end
endmodule

`ifndef VERILATOR
module ace3_decoder_silu_streaming_iverilog_tb;
    reg clk = 1'b0;

    always #5 clk = ~clk;

    ace3_decoder_silu_streaming_tb test (
        .clk_i(clk)
    );
endmodule
`endif

`default_nettype wire
