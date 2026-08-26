`timescale 1ns/1ps
`default_nettype none

module ace3_decoder_width_boundary_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg load_valid;
    wire load_ready;
    reg [1:0] load_kind;
    reg [12:0] load_index;
    reg [15:0] load_f16;
    reg start_valid;
    wire start_ready;
    reg [1:0] start_slot;
    reg [14:0] start_position;
    reg projection_meta_valid;
    wire projection_meta_ready;
    reg [31:0] projection_qzeros;
    reg [15:0] projection_scale;
    reg projection_pair_valid;
    wire projection_pair_ready;
    reg [31:0] projection_qweight;
    reg projection_bias_valid;
    wire projection_bias_ready;
    reg [15:0] projection_bias;
    reg rope_valid;
    wire rope_ready;
    reg [15:0] rope_cos;
    reg [15:0] rope_sin;
    reg trace_ready;
    reg final_ready;
    reg done_ready;
    wire trace_valid;
    wire final_valid;
    wire done_valid;
    wire busy;
    integer failures;

    ace3_decoder_layer0_token_engine dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .load_valid_i(load_valid), .load_ready_o(load_ready),
        .load_kind_i(load_kind), .load_index_i(load_index),
        .load_f16_i(load_f16),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .start_cache_slot_i(start_slot), .start_position_i(start_position),
        .projection_kind_o(),
        .projection_meta_valid_i(projection_meta_valid),
        .projection_meta_ready_o(projection_meta_ready),
        .projection_meta_output_channel_o(),
        .projection_meta_group_o(), .projection_meta_word_o(),
        .projection_meta_lane_o(), .projection_qzeros_i(projection_qzeros),
        .projection_scale_f16_i(projection_scale),
        .projection_pair_valid_i(projection_pair_valid),
        .projection_pair_ready_o(projection_pair_ready),
        .projection_pair_input_o(), .projection_pair_output_o(),
        .projection_pair_group_o(), .projection_pair_word_o(),
        .projection_pair_lane_o(), .projection_qweight_i(projection_qweight),
        .projection_bias_valid_i(projection_bias_valid),
        .projection_bias_ready_o(projection_bias_ready),
        .projection_bias_output_channel_o(),
        .projection_bias_f16_i(projection_bias),
        .rope_valid_i(rope_valid), .rope_ready_o(rope_ready),
        .rope_position_o(), .rope_pair_o(), .rope_cos_f16_i(rope_cos),
        .rope_sin_f16_i(rope_sin),
        .trace_valid_o(trace_valid), .trace_ready_i(trace_ready),
        .trace_stage_o(), .trace_index_o(), .trace_f16_o(),
        .trace_position_o(), .final_valid_o(final_valid),
        .final_ready_i(final_ready), .final_index_o(), .final_f16_o(),
        .final_last_o(), .done_valid_o(done_valid),
        .done_ready_i(done_ready), .done_cache_slot_o(),
        .done_position_o(), .done_cycles_o(), .done_stall_cycles_o(),
        .busy_o(busy), .phase_o()
    );

    always #5 clk = ~clk;

    task expect_false;
        input value;
        input [8*80-1:0] message;
        begin
            if (value !== 1'b0) begin
                $display("DECODER_WIDTH_BOUNDARY_FAIL %0s", message);
                failures = failures + 1;
            end
        end
    endtask

    task expect_true;
        input value;
        input [8*80-1:0] message;
        begin
            if (value !== 1'b1) begin
                $display("DECODER_WIDTH_BOUNDARY_FAIL %0s", message);
                failures = failures + 1;
            end
        end
    endtask

    task expect_exp;
        input [41:0] delta_q24;
        input [24:0] expected_q24;
        input [8*80-1:0] message;
        reg [24:0] actual_q24;
        begin
            actual_q24 = dut.softmax.exp_approx_q24(delta_q24);
            if (actual_q24 !== expected_q24) begin
                $display("DECODER_WIDTH_BOUNDARY_FAIL %0s actual=%0d expected=%0d",
                         message, actual_q24, expected_q24);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        clear = 0;
        load_valid = 0;
        load_kind = 0;
        load_index = 0;
        load_f16 = 0;
        start_valid = 0;
        start_slot = 0;
        start_position = 0;
        projection_meta_valid = 0;
        projection_qzeros = 0;
        projection_scale = 0;
        projection_pair_valid = 0;
        projection_qweight = 0;
        projection_bias_valid = 0;
        projection_bias = 0;
        rope_valid = 0;
        rope_cos = 0;
        rope_sin = 0;
        trace_ready = 0;
        final_ready = 0;
        done_ready = 0;
        failures = 0;

        #1;
        expect_false(load_ready, "load ready during reset");
        expect_false(start_ready, "start ready during reset");
        expect_false(busy, "busy during reset");
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1;
        dut.activation_loaded_q = 1'b1;
        dut.n1_loaded_q = 1'b1;
        dut.n2_loaded_q = 1'b1;
        #1;
        expect_true(dut.disabled_bias_contract_ok_w,
                    "disabled bias contract rejected reset metadata");
        expect_false(dut.o_bias_ready_w, "O bias unexpectedly ready");
        expect_false(dut.f_bias_ready_w, "FFN bias unexpectedly ready");
        expect_false(dut.d_bias_ready_w, "down bias unexpectedly ready");

        expect_exp(42'd0, 25'd16777216, "softmax zero delta");
        expect_exp(42'd1, 25'd16777216,
                   "softmax nonzero discarded residue");
        expect_exp(42'd177, 25'd16777216,
                   "softmax pre-quotient transition");
        expect_exp(42'd178, 25'd16777042,
                   "softmax quotient transition");
        expect_exp(42'd3030787, 25'd14006412,
                   "softmax maximal attainable discarded residue");

        dut.context_index_q = 7'd0;
        dut.score_causal_mem[0] = 1'b1;
        dut.score_cache_miss_mem[0] = 1'b1;
        dut.score_invalid_mem[0] = 1'b1;
        dut.softmax_row_error_q = 1'b1;
        #1;
        expect_true(dut.softmax.causal_i,
                    "score causal state not propagated to softmax");
        expect_true(dut.softmax.cache_miss_i,
                    "score cache miss not propagated to softmax");
        expect_true(dut.softmax.invalid_operand_i,
                    "score invalid state not propagated to softmax");
        expect_true(dut.value.row_error_i,
                    "softmax row error not propagated to value");
        dut.score_causal_mem[0] = 1'b0;
        dut.score_cache_miss_mem[0] = 1'b0;
        dut.score_invalid_mem[0] = 1'b0;
        dut.softmax_row_error_q = 1'b0;

        dut.context_len_q[0] = 8'd0;
        dut.context_len_q[1] = 8'd0;
        start_slot = 0;
        start_position = 0;
        #1 expect_true(start_ready, "slot 0 position 0 rejected");
        start_slot = 1;
        #1 expect_true(start_ready, "slot 1 position 0 rejected");
        dut.context_len_q[0] = 8'd1;
        start_slot = 0;
        start_position = 1;
        #1 expect_true(start_ready, "position 1 rejected");
        dut.context_len_q[1] = 8'd127;
        start_slot = 1;
        start_position = 127;
        #1 expect_true(start_ready, "position 127 rejected");

        start_slot = 2;
        #1 expect_false(start_ready, "slot 2 accepted");
        start_slot = 3;
        #1 expect_false(start_ready, "slot 3 accepted");
        start_slot = 0;
        start_position = 128;
        #1 expect_false(start_ready, "position 128 accepted");
        start_position = 15'h7fff;
        #1 expect_false(start_ready, "maximum invalid position accepted");
        start_position = 1;
        start_slot = 2'bx;
        #1 expect_false(start_ready, "X slot accepted");
        start_slot = 2'bz;
        #1 expect_false(start_ready, "Z slot accepted");
        start_slot = 0;
        start_position = 15'bx;
        #1 expect_false(start_ready, "X position accepted");
        start_position = 15'bz;
        #1 expect_false(start_ready, "Z position accepted");

        load_valid = 1;
        load_kind = 2'bx;
        load_index = 0;
        load_f16 = 0;
        #1 expect_false(load_ready, "X load kind accepted");
        load_kind = 0;
        load_index = 13'bz;
        #1 expect_false(load_ready, "Z load index accepted");
        load_index = 0;
        load_f16 = 16'bx;
        #1 expect_false(load_ready, "X load payload accepted");
        load_valid = 0;

        dut.head_q = 4'd13;
        dut.dim_q = 6'd63;
        #1;
        if ((dut.q_flat_index_w !== 10'd895) ||
            (dut.q_flat_valid_w !== 1'b1)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL Q index 895");
            failures = failures + 1;
        end
        dut.dim_q = 6'd31;
        #1;
        if ((dut.q_flat_index_w !== 10'd863) ||
            (dut.q_flat_high_index_w !== 10'd895) ||
            (dut.q_pair_valid_w !== 1'b1)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL Q half-split pair 863/895");
            failures = failures + 1;
        end
        dut.head_q = 4'd1;
        dut.dim_q = 6'd31;
        #1;
        if ((dut.kv_flat_index_w !== 7'd95) ||
            (dut.kv_flat_high_index_w !== 7'd127) ||
            (dut.kv_pair_valid_w !== 1'b1)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL K/V half-split pair 95/127");
            failures = failures + 1;
        end
        dut.head_q = 4'd1;
        dut.dim_q = 6'd63;
        #1;
        if ((dut.kv_flat_index_w !== 7'd127) ||
            (dut.kv_flat_valid_w !== 1'b1)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL K/V index 127");
            failures = failures + 1;
        end
        dut.hidden_index_q = 10'd895;
        dut.intermediate_index_q = 13'd4863;
        dut.context_index_q = 7'd127;
        #1;
        if ((dut.hidden_index_q !== 10'd895) ||
            (dut.intermediate_index_q !== 13'd4863) ||
            (dut.context_index_q !== 7'd127)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL final counter indices");
            failures = failures + 1;
        end

        force dut.o_bias_ready_w = 1'b1;
        @(posedge clk);
        #1;
        release dut.o_bias_ready_w;
        expect_true(dut.fault_q, "disabled bias violation did not latch fault");
        expect_false(trace_valid, "disabled bias fault emitted trace");
        clear = 1;
        @(posedge clk);
        #1;
        clear = 0;

        trace_ready = 1;
        dut.state_q = 6'd5;
        dut.psel_q = 3'd0;
        dut.projection_output_index_q = 13'd0;
        force dut.b_busy_w = 1'b1;
        force dut.b_out_valid_w = 1'b1;
        force dut.b_invalid_w = 1'b1;
        @(posedge clk);
        #1;
        release dut.b_busy_w;
        release dut.b_out_valid_w;
        release dut.b_invalid_w;
        expect_true(dut.fault_q,
                    "projection invalid operand did not latch fault");
        expect_false(trace_valid, "invalid projection emitted trace");
        clear = 1;
        @(posedge clk);
        #1;
        clear = 0;

        dut.state_q = 6'd24;
        dut.head_q = 4'd0;
        dut.dim_q = 6'd0;
        dut.token_position_q = 7'd0;
        force dut.av_busy_w = 1'b1;
        force dut.av_out_valid_w = 1'b1;
        force dut.av_row_error_w = 1'b1;
        @(posedge clk);
        #1;
        release dut.av_busy_w;
        release dut.av_out_valid_w;
        release dut.av_row_error_w;
        expect_true(dut.fault_q, "attention row error did not latch fault");
        expect_false(trace_valid, "attention row error emitted trace");

        clear = 1;
        @(posedge clk);
        #1;
        expect_false(busy, "busy after clear");
        expect_false(trace_valid, "trace valid after clear");
        expect_false(final_valid, "final valid after clear");
        expect_false(done_valid, "done valid after clear");
        expect_false(dut.fault_q, "fault remained latched after clear");
        expect_true(dut.disabled_bias_contract_ok_w,
                    "disabled bias contract failed after clear");
        if ((dut.hidden_index_q !== 10'd0) ||
            (dut.intermediate_index_q !== 13'd0) ||
            (dut.context_index_q !== 7'd0) ||
            (dut.context_len_q[0] !== 8'd0) ||
            (dut.context_len_q[1] !== 8'd0)) begin
            $display("DECODER_WIDTH_BOUNDARY_FAIL clear state");
            failures = failures + 1;
        end

        if (failures == 0) begin
            $display("DECODER_WIDTH_BOUNDARY_4STATE_PASS");
            $finish;
        end
        $fatal(1, "DECODER_WIDTH_BOUNDARY_4STATE_FAIL failures=%0d",
               failures);
    end
endmodule

`default_nettype wire
