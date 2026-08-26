`timescale 1ns/1ps
`default_nettype none

module ace3_attention_block_tb;
    `include "attention_params.svh"

    reg clk;
    reg rst_n;
    reg clear;

    reg score_start_valid;
    wire score_start_ready;
    reg [3:0] score_query_head;
    reg [3:0] score_key_head;
    reg [14:0] score_query_position;
    reg [14:0] score_key_position;
    reg score_pair_valid;
    wire score_pair_ready;
    reg [15:0] score_q;
    reg [15:0] score_k;
    reg score_cache_hit;
    wire score_out_valid;
    reg score_out_ready;
    wire [15:0] score_out;
    wire [3:0] score_out_query_head;
    wire [3:0] score_out_key_head;
    wire [14:0] score_out_query_position;
    wire [14:0] score_out_key_position;
    wire score_out_causal;
    wire score_out_cache_miss;
    wire score_out_invalid;
    wire score_out_saturation;
    wire score_busy;

    reg softmax_start_valid;
    wire softmax_start_ready;
    reg [3:0] softmax_query_head;
    reg [14:0] softmax_query_position;
    reg [15:0] softmax_context_count;
    reg softmax_score_valid;
    wire softmax_score_ready;
    reg [15:0] softmax_score;
    reg [14:0] softmax_key_position;
    reg softmax_causal;
    reg softmax_cache_miss;
    reg softmax_invalid;
    wire softmax_out_valid;
    reg softmax_out_ready;
    wire [15:0] softmax_probability;
    wire [3:0] softmax_out_query_head;
    wire [14:0] softmax_out_query_position;
    wire [14:0] softmax_out_key_position;
    wire [15:0] softmax_out_index;
    wire softmax_out_last;
    wire softmax_row_error;
    wire softmax_out_cache_miss;
    wire softmax_out_invalid;
    wire softmax_busy;

    reg value_start_valid;
    wire value_start_ready;
    reg [3:0] value_query_head;
    reg [3:0] value_head;
    reg [14:0] value_query_position;
    reg [5:0] value_dimension;
    reg [15:0] value_context_count;
    reg value_term_valid;
    wire value_term_ready;
    reg [15:0] value_probability;
    reg [15:0] value_cached;
    reg value_hit;
    reg value_row_error;
    wire value_out_valid;
    reg value_out_ready;
    wire [15:0] value_out;
    wire [3:0] value_out_query_head;
    wire [3:0] value_out_head;
    wire [14:0] value_out_query_position;
    wire [5:0] value_out_dimension;
    wire value_out_row_error;
    wire value_out_cache_miss;
    wire value_out_invalid;
    wire value_out_saturation;
    wire value_busy;

    reg [63:0] score_terms [0:`ATTENTION_SCORE_TERMS-1];
    reg [127:0] score_expected [0:`ATTENTION_SCORE_CASES-1];
    reg [63:0] softmax_rows [0:`ATTENTION_SOFTMAX_ROWS-1];
    reg [63:0] softmax_terms [0:`ATTENTION_SOFTMAX_TERMS-1];
    reg [127:0] value_cases [0:`ATTENTION_VALUE_CASES-1];
    reg [63:0] value_terms [0:`ATTENTION_VALUE_TERMS-1];

    integer failures;
    integer score_outputs;
    integer softmax_outputs;
    integer value_outputs;
    integer stalls;
    integer reset_checks;
    integer clear_checks;
    integer gqa_rejection_checks;
    integer xz_checks;
    integer case_index;
    integer softmax_term_base;
    integer value_term_base;
    string vector_dir;
    string vector_path;

    ace3_attention_score_core score_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(score_start_valid),
        .start_ready_o(score_start_ready),
        .query_head_i(score_query_head),
        .key_head_i(score_key_head),
        .query_position_i(score_query_position),
        .key_position_i(score_key_position),
        .pair_valid_i(score_pair_valid),
        .pair_ready_o(score_pair_ready),
        .q_f16_i(score_q),
        .k_f16_i(score_k),
        .cache_hit_i(score_cache_hit),
        .out_valid_o(score_out_valid),
        .out_ready_i(score_out_ready),
        .score_f16_o(score_out),
        .query_head_o(score_out_query_head),
        .key_head_o(score_out_key_head),
        .query_position_o(score_out_query_position),
        .key_position_o(score_out_key_position),
        .causal_o(score_out_causal),
        .cache_miss_o(score_out_cache_miss),
        .invalid_operand_o(score_out_invalid),
        .saturation_o(score_out_saturation),
        .busy_o(score_busy)
    );

    ace3_attention_softmax_core #(
        .CONTEXT_MAX(16)
    ) softmax_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(softmax_start_valid),
        .start_ready_o(softmax_start_ready),
        .query_head_i(softmax_query_head),
        .query_position_i(softmax_query_position),
        .context_count_i(softmax_context_count),
        .score_valid_i(softmax_score_valid),
        .score_ready_o(softmax_score_ready),
        .score_f16_i(softmax_score),
        .key_position_i(softmax_key_position),
        .causal_i(softmax_causal),
        .cache_miss_i(softmax_cache_miss),
        .invalid_operand_i(softmax_invalid),
        .out_valid_o(softmax_out_valid),
        .out_ready_i(softmax_out_ready),
        .probability_f16_o(softmax_probability),
        .query_head_o(softmax_out_query_head),
        .query_position_o(softmax_out_query_position),
        .key_position_o(softmax_out_key_position),
        .out_index_o(softmax_out_index),
        .out_last_o(softmax_out_last),
        .row_error_o(softmax_row_error),
        .cache_miss_o(softmax_out_cache_miss),
        .invalid_operand_o(softmax_out_invalid),
        .busy_o(softmax_busy)
    );

    ace3_attention_value_core #(
        .CONTEXT_MAX(16)
    ) value_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(value_start_valid),
        .start_ready_o(value_start_ready),
        .query_head_i(value_query_head),
        .value_head_i(value_head),
        .query_position_i(value_query_position),
        .dimension_i(value_dimension),
        .context_count_i(value_context_count),
        .term_valid_i(value_term_valid),
        .term_ready_o(value_term_ready),
        .probability_f16_i(value_probability),
        .value_f16_i(value_cached),
        .value_hit_i(value_hit),
        .row_error_i(value_row_error),
        .out_valid_o(value_out_valid),
        .out_ready_i(value_out_ready),
        .value_f16_o(value_out),
        .query_head_o(value_out_query_head),
        .value_head_o(value_out_head),
        .query_position_o(value_out_query_position),
        .dimension_o(value_out_dimension),
        .row_error_o(value_out_row_error),
        .cache_miss_o(value_out_cache_miss),
        .invalid_operand_o(value_out_invalid),
        .saturation_o(value_out_saturation),
        .busy_o(value_busy)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task drive_idle;
        begin
            clear = 1'b0;
            score_start_valid = 1'b0;
            score_query_head = 4'd0;
            score_key_head = 4'd0;
            score_query_position = 15'd0;
            score_key_position = 15'd0;
            score_pair_valid = 1'b0;
            score_q = 16'd0;
            score_k = 16'd0;
            score_cache_hit = 1'b1;
            score_out_ready = 1'b0;
            softmax_start_valid = 1'b0;
            softmax_query_head = 4'd0;
            softmax_query_position = 15'd0;
            softmax_context_count = 16'd1;
            softmax_score_valid = 1'b0;
            softmax_score = 16'd0;
            softmax_key_position = 15'd0;
            softmax_causal = 1'b1;
            softmax_cache_miss = 1'b0;
            softmax_invalid = 1'b0;
            softmax_out_ready = 1'b0;
            value_start_valid = 1'b0;
            value_query_head = 4'd0;
            value_head = 4'd0;
            value_query_position = 15'd0;
            value_dimension = 6'd0;
            value_context_count = 16'd1;
            value_term_valid = 1'b0;
            value_probability = 16'd0;
            value_cached = 16'd0;
            value_hit = 1'b1;
            value_row_error = 1'b0;
            value_out_ready = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            drive_idle();
            rst_n = 1'b0;
            #2;
            if (score_start_ready || score_out_valid || score_busy ||
                softmax_start_ready || softmax_out_valid || softmax_busy ||
                value_start_ready || value_out_valid || value_busy) begin
                $display("ATTENTION_ASYNC_RESET_FAIL");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            #1;
            if ((score_start_ready !== 1'b1) ||
                (softmax_start_ready !== 1'b1) ||
                (value_start_ready !== 1'b1)) begin
                $display("ATTENTION_RESET_RELEASE_FAIL");
                failures = failures + 1;
            end
            reset_checks = reset_checks + 1;
        end
    endtask

    task run_score_case;
        input integer selected_case;
        reg [127:0] expected;
        reg [63:0] term;
        reg [72:0] held;
        integer dimension;
        begin
            expected = score_expected[selected_case];
            @(negedge clk);
            score_query_head = expected[19:16];
            score_key_head = expected[23:20];
            score_query_position = expected[38:24];
            score_key_position = expected[53:39];
            score_start_valid = 1'b1;
            #1;
            if (score_start_ready !== 1'b1) begin
                $display("SCORE_START_NOT_READY case=%0d", selected_case);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            score_start_valid = 1'b0;
            for (dimension = 0; dimension < 64; dimension = dimension + 1) begin
                term = score_terms[selected_case * 64 + dimension];
                score_q = term[15:0];
                score_k = term[31:16];
                score_cache_hit = term[32];
                score_pair_valid = 1'b1;
                #1;
                if (score_pair_ready !== 1'b1) begin
                    $display("SCORE_PAIR_NOT_READY case=%0d dim=%0d",
                             selected_case, dimension);
                    failures = failures + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
            score_pair_valid = 1'b0;
            #1;
            if ((score_out_valid !== 1'b1) ||
                (score_out !== expected[15:0]) ||
                (score_out_query_head !== expected[19:16]) ||
                (score_out_key_head !== expected[23:20]) ||
                (score_out_query_position !== expected[38:24]) ||
                (score_out_key_position !== expected[53:39]) ||
                (score_out_causal !== expected[54]) ||
                (score_out_cache_miss !== expected[55]) ||
                (score_out_invalid !== expected[56]) ||
                (score_out_saturation !== expected[57])) begin
                $display("SCORE_OUTPUT_MISMATCH case=%0d expected=%h got=%h flags=%b%b%b%b",
                         selected_case, expected[15:0], score_out,
                         score_out_causal, score_out_cache_miss,
                         score_out_invalid, score_out_saturation);
                failures = failures + 1;
            end
            if (selected_case == 0) begin
                held = {
                    score_out, score_out_query_head, score_out_key_head,
                    score_out_query_position, score_out_key_position,
                    score_out_causal, score_out_cache_miss,
                    score_out_invalid, score_out_saturation
                };
                repeat (2) begin
                    @(posedge clk);
                    @(negedge clk);
                    if ((score_out_valid !== 1'b1) ||
                        (held !== {
                            score_out, score_out_query_head,
                            score_out_key_head, score_out_query_position,
                            score_out_key_position, score_out_causal,
                            score_out_cache_miss, score_out_invalid,
                            score_out_saturation
                        })) begin
                        $display("SCORE_BACKPRESSURE_FAIL");
                        failures = failures + 1;
                    end
                    stalls = stalls + 1;
                end
            end
            score_out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            score_out_ready = 1'b0;
            if (score_out_valid) begin
                $display("SCORE_OUTPUT_NOT_RELEASED case=%0d", selected_case);
                failures = failures + 1;
            end
            score_outputs = score_outputs + 1;
        end
    endtask

    task run_softmax_row;
        input integer selected_row;
        input integer term_base;
        reg [63:0] header;
        reg [63:0] term;
        reg [69:0] held;
        integer count;
        integer item;
        integer timeout;
        begin
            header = softmax_rows[selected_row];
            count = header[34:19];
            @(negedge clk);
            softmax_query_head = header[3:0];
            softmax_query_position = header[18:4];
            softmax_context_count = count;
            softmax_start_valid = 1'b1;
            #1;
            if (softmax_start_ready !== 1'b1) begin
                $display("SOFTMAX_START_NOT_READY row=%0d", selected_row);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            softmax_start_valid = 1'b0;
            for (item = 0; item < count; item = item + 1) begin
                term = softmax_terms[term_base + item];
                softmax_score = term[15:0];
                softmax_key_position = term[30:16];
                softmax_causal = term[31];
                softmax_cache_miss = term[32];
                softmax_invalid = term[33];
                softmax_score_valid = 1'b1;
                #1;
                if (softmax_score_ready !== 1'b1) begin
                    $display("SOFTMAX_SCORE_NOT_READY row=%0d item=%0d",
                             selected_row, item);
                    failures = failures + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
            softmax_score_valid = 1'b0;
            timeout = 0;
            while ((softmax_out_valid !== 1'b1) && (timeout < 64)) begin
                @(posedge clk);
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (softmax_out_valid !== 1'b1) begin
                $display("SOFTMAX_TIMEOUT row=%0d", selected_row);
                failures = failures + 1;
            end
            for (item = 0; item < count; item = item + 1) begin
                term = softmax_terms[term_base + item];
                #1;
                if ((softmax_out_valid !== 1'b1) ||
                    (softmax_probability !== term[49:34]) ||
                    (softmax_out_query_head !== header[3:0]) ||
                    (softmax_out_query_position !== header[18:4]) ||
                    (softmax_out_key_position !== term[30:16]) ||
                    (softmax_out_index !== item) ||
                    (softmax_out_last !== (item == count - 1)) ||
                    (softmax_row_error !== header[35]) ||
                    (softmax_out_cache_miss !== header[36]) ||
                    (softmax_out_invalid !== header[37])) begin
                    $display("SOFTMAX_OUTPUT_MISMATCH row=%0d item=%0d expected=%h got=%h",
                             selected_row, item, term[49:34],
                             softmax_probability);
                    failures = failures + 1;
                end
                if ((selected_row == 0) && (item == 0)) begin
                    held = {
                        softmax_probability, softmax_out_query_head,
                        softmax_out_query_position,
                        softmax_out_key_position, softmax_out_index,
                        softmax_out_last, softmax_row_error,
                        softmax_out_cache_miss, softmax_out_invalid
                    };
                    repeat (2) begin
                        @(posedge clk);
                        @(negedge clk);
                        if ((softmax_out_valid !== 1'b1) ||
                            (held !== {
                                softmax_probability,
                                softmax_out_query_head,
                                softmax_out_query_position,
                                softmax_out_key_position,
                                softmax_out_index, softmax_out_last,
                                softmax_row_error,
                                softmax_out_cache_miss,
                                softmax_out_invalid
                            })) begin
                            $display("SOFTMAX_BACKPRESSURE_FAIL");
                            failures = failures + 1;
                        end
                        stalls = stalls + 1;
                    end
                end
                softmax_out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                softmax_out_ready = 1'b0;
                softmax_outputs = softmax_outputs + 1;
            end
            if (softmax_out_valid) begin
                $display("SOFTMAX_OUTPUT_NOT_RELEASED row=%0d", selected_row);
                failures = failures + 1;
            end
        end
    endtask

    task run_value_case;
        input integer selected_case;
        input integer term_base;
        reg [127:0] header;
        reg [63:0] term;
        reg [67:0] held;
        integer count;
        integer item;
        begin
            header = value_cases[selected_case];
            count = header[44:29];
            @(negedge clk);
            value_query_head = header[3:0];
            value_head = header[7:4];
            value_query_position = header[22:8];
            value_dimension = header[28:23];
            value_context_count = count;
            value_start_valid = 1'b1;
            #1;
            if (value_start_ready !== 1'b1) begin
                $display("VALUE_START_NOT_READY case=%0d", selected_case);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            value_start_valid = 1'b0;
            for (item = 0; item < count; item = item + 1) begin
                term = value_terms[term_base + item];
                value_probability = term[15:0];
                value_cached = term[31:16];
                value_hit = term[32];
                value_row_error = term[33];
                value_term_valid = 1'b1;
                #1;
                if (value_term_ready !== 1'b1) begin
                    $display("VALUE_TERM_NOT_READY case=%0d item=%0d",
                             selected_case, item);
                    failures = failures + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
            value_term_valid = 1'b0;
            #1;
            if ((value_out_valid !== 1'b1) ||
                (value_out !== header[60:45]) ||
                (value_out_query_head !== header[3:0]) ||
                (value_out_head !== header[7:4]) ||
                (value_out_query_position !== header[22:8]) ||
                (value_out_dimension !== header[28:23]) ||
                (value_out_row_error !== header[61]) ||
                (value_out_cache_miss !== header[62]) ||
                (value_out_invalid !== header[63]) ||
                (value_out_saturation !== header[64])) begin
                $display("VALUE_OUTPUT_MISMATCH case=%0d expected=%h got=%h flags=%b%b%b%b",
                         selected_case, header[60:45], value_out,
                         value_out_row_error, value_out_cache_miss,
                         value_out_invalid, value_out_saturation);
                failures = failures + 1;
            end
            if (selected_case == 0) begin
                held = {
                    value_out, value_out_query_head, value_out_head,
                    value_out_query_position, value_out_dimension,
                    value_out_row_error, value_out_cache_miss,
                    value_out_invalid, value_out_saturation
                };
                repeat (2) begin
                    @(posedge clk);
                    @(negedge clk);
                    if ((value_out_valid !== 1'b1) ||
                        (held !== {
                            value_out, value_out_query_head,
                            value_out_head, value_out_query_position,
                            value_out_dimension, value_out_row_error,
                            value_out_cache_miss, value_out_invalid,
                            value_out_saturation
                        })) begin
                        $display("VALUE_BACKPRESSURE_FAIL");
                        failures = failures + 1;
                    end
                    stalls = stalls + 1;
                end
            end
            value_out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            value_out_ready = 1'b0;
            if (value_out_valid) begin
                $display("VALUE_OUTPUT_NOT_RELEASED case=%0d", selected_case);
                failures = failures + 1;
            end
            value_outputs = value_outputs + 1;
        end
    endtask

    task check_invalid_gqa_rejection;
        begin
            @(negedge clk);
            score_query_head = 4'd6;
            score_key_head = 4'd1;
            score_start_valid = 1'b1;
            #1;
            if (score_start_ready !== 1'b0) begin
                $display("SCORE_INVALID_GQA_ACCEPTED");
                failures = failures + 1;
            end
            gqa_rejection_checks = gqa_rejection_checks + 1;
            score_start_valid = 1'b0;
            score_query_head = 4'd0;
            score_key_head = 4'd0;

            value_query_head = 4'd7;
            value_head = 4'd0;
            value_start_valid = 1'b1;
            #1;
            if (value_start_ready !== 1'b0) begin
                $display("VALUE_INVALID_GQA_ACCEPTED");
                failures = failures + 1;
            end
            gqa_rejection_checks = gqa_rejection_checks + 1;
            value_start_valid = 1'b0;
            value_query_head = 4'd0;
            value_head = 4'd0;
        end
    endtask

    task check_clear_abort;
        begin
            @(negedge clk);
            score_start_valid = 1'b1;
            softmax_start_valid = 1'b1;
            value_start_valid = 1'b1;
            #1;
            if (!score_start_ready || !softmax_start_ready ||
                !value_start_ready) begin
                $display("ATTENTION_CLEAR_SETUP_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            score_start_valid = 1'b0;
            softmax_start_valid = 1'b0;
            value_start_valid = 1'b0;
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            #1;
            if (score_busy || score_out_valid ||
                softmax_busy || softmax_out_valid ||
                value_busy || value_out_valid) begin
                $display("ATTENTION_CLEAR_ABORT_FAIL");
                failures = failures + 1;
            end
            clear_checks = clear_checks + 1;
        end
    endtask

    task check_reset_abort;
        begin
            @(negedge clk);
            score_start_valid = 1'b1;
            softmax_start_valid = 1'b1;
            value_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            score_start_valid = 1'b0;
            softmax_start_valid = 1'b0;
            value_start_valid = 1'b0;
            rst_n = 1'b0;
            #2;
            if (score_busy || score_out_valid ||
                softmax_busy || softmax_out_valid ||
                value_busy || value_out_valid) begin
                $display("ATTENTION_RESET_ABORT_FAIL");
                failures = failures + 1;
            end
            @(negedge clk);
            rst_n = 1'b1;
            reset_checks = reset_checks + 1;
        end
    endtask

`ifndef VERILATOR
    task check_xz_rejection;
        begin
            @(negedge clk);
            score_query_head = 4'bx;
            score_start_valid = 1'b1;
            #1;
            if (score_start_ready === 1'b1) failures = failures + 1;
            score_start_valid = 1'b0;
            score_query_head = 4'd0;
            xz_checks = xz_checks + 1;

            softmax_context_count = 16'bz;
            softmax_start_valid = 1'b1;
            #1;
            if (softmax_start_ready === 1'b1) failures = failures + 1;
            softmax_start_valid = 1'b0;
            softmax_context_count = 16'd1;
            xz_checks = xz_checks + 1;

            value_head = 4'bx;
            value_start_valid = 1'b1;
            #1;
            if (value_start_ready === 1'b1) failures = failures + 1;
            value_start_valid = 1'b0;
            value_head = 4'd0;
            xz_checks = xz_checks + 1;

            score_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            score_start_valid = 1'b0;
            score_pair_valid = 1'b1;
            score_cache_hit = 1'bx;
            #1;
            if (score_pair_ready === 1'b1) failures = failures + 1;
            score_pair_valid = 1'b0;
            score_cache_hit = 1'b1;
            xz_checks = xz_checks + 1;

            softmax_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            softmax_start_valid = 1'b0;
            softmax_score_valid = 1'b1;
            softmax_causal = 1'bx;
            #1;
            if (softmax_score_ready === 1'b1) failures = failures + 1;
            softmax_score_valid = 1'b0;
            softmax_causal = 1'b1;
            xz_checks = xz_checks + 1;

            value_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            value_start_valid = 1'b0;
            value_term_valid = 1'b1;
            value_hit = 1'bz;
            #1;
            if (value_term_ready === 1'b1) failures = failures + 1;
            value_term_valid = 1'b0;
            value_hit = 1'b1;
            xz_checks = xz_checks + 1;

            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
        end
    endtask
`endif

    initial begin
        failures = 0;
        score_outputs = 0;
        softmax_outputs = 0;
        value_outputs = 0;
        stalls = 0;
        reset_checks = 0;
        clear_checks = 0;
        gqa_rejection_checks = 0;
        xz_checks = 0;
        softmax_term_base = 0;
        value_term_base = 0;
        rst_n = 1'b1;
        drive_idle();
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "build/attention_vectors";
        vector_path = {vector_dir, "/attention_score_terms.hex"};
        $readmemh(vector_path, score_terms);
        vector_path = {vector_dir, "/attention_score_expected.hex"};
        $readmemh(vector_path, score_expected);
        vector_path = {vector_dir, "/attention_softmax_rows.hex"};
        $readmemh(vector_path, softmax_rows);
        vector_path = {vector_dir, "/attention_softmax_terms.hex"};
        $readmemh(vector_path, softmax_terms);
        vector_path = {vector_dir, "/attention_value_cases.hex"};
        $readmemh(vector_path, value_cases);
        vector_path = {vector_dir, "/attention_value_terms.hex"};
        $readmemh(vector_path, value_terms);

        apply_reset();
        for (case_index = 0;
             case_index < `ATTENTION_SCORE_CASES;
             case_index = case_index + 1)
            run_score_case(case_index);
        for (case_index = 0;
             case_index < `ATTENTION_SOFTMAX_ROWS;
             case_index = case_index + 1) begin
            run_softmax_row(case_index, softmax_term_base);
            softmax_term_base = softmax_term_base +
                softmax_rows[case_index][34:19];
        end
        for (case_index = 0;
             case_index < `ATTENTION_VALUE_CASES;
             case_index = case_index + 1) begin
            run_value_case(case_index, value_term_base);
            value_term_base = value_term_base +
                value_cases[case_index][44:29];
        end
        check_invalid_gqa_rejection();
        check_clear_abort();
`ifndef VERILATOR
        check_xz_rejection();
`endif
        check_reset_abort();

        if (softmax_term_base != `ATTENTION_SOFTMAX_TERMS ||
            value_term_base != `ATTENTION_VALUE_TERMS) begin
            $display("ATTENTION_VECTOR_FRAMING_FAIL");
            failures = failures + 1;
        end
        if (failures == 0) begin
`ifdef VERILATOR
            $display("ACE3_ATTENTION_VERILATOR_PASS score_cases=%0d softmax_outputs=%0d value_cases=%0d stalls=%0d reset=%0d clear=%0d gqa_rejections=%0d cache_miss=pass causal=pass gqa=14_to_2",
`else
            $display("ACE3_ATTENTION_IVERILOG_PASS score_cases=%0d softmax_outputs=%0d value_cases=%0d stalls=%0d reset=%0d clear=%0d gqa_rejections=%0d xz=%0d cache_miss=pass causal=pass gqa=14_to_2",
`endif
                     score_outputs, softmax_outputs, value_outputs,
                     stalls, reset_checks, clear_checks,
                     gqa_rejection_checks
`ifndef VERILATOR
                     , xz_checks
`endif
            );
        end else begin
            $fatal(1, "ACE3_ATTENTION_FAIL failures=%0d", failures);
        end
        $finish;
    end
endmodule

`default_nettype wire
