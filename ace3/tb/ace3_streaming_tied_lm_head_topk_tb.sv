`timescale 1ns/1ps
`default_nettype none

module ace3_streaming_tied_lm_head_topk_tb;
    localparam integer HIDDEN_SIZE = 4;
    localparam integer VOCAB_SIZE = 5;
    localparam integer TOP_K = 3;

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg hidden_valid;
    wire hidden_ready;
    reg [1:0] hidden_index;
    reg [15:0] hidden_f16;
    reg hidden_last;
    reg hidden_end;
    reg weight_valid;
    wire weight_ready;
    reg [2:0] weight_token;
    reg [1:0] weight_feature;
    reg [15:0] weight_f16;
    reg weight_last_feature;
    reg weight_last_token;
    reg weight_end;
    wire logit_valid;
    reg logit_ready;
    wire [2:0] logit_token;
    wire [15:0] logit_f16;
    wire signed [95:0] accumulator;
    wire logit_saturation;
    wire top_valid;
    reg top_ready;
    wire [1:0] top_rank;
    wire [2:0] top_token;
    wire [15:0] top_logit;
    wire done_valid;
    reg done_ready;
    wire error_valid;
    wire [3:0] error_code;
    wire invalid_operand;
    wire saturation;
    wire busy;

    integer failures;
    integer four_state_probes;
    integer negative_underflow_probes;
    integer logit_count;
    integer top_count;
    reg previous_logit_stall;
    reg [2:0] held_logit_token;
    reg [15:0] held_logit;
    reg signed [95:0] held_accumulator;
    reg previous_top_stall;
    reg [1:0] held_top_rank;
    reg [2:0] held_top_token;
    reg [15:0] held_top_logit;

    ace3_streaming_tied_lm_head_topk #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .VOCAB_SIZE(VOCAB_SIZE),
        .TOP_K(TOP_K),
        .TOKEN_INDEX_WIDTH(3),
        .FEATURE_INDEX_WIDTH(2),
        .TOP_RANK_WIDTH(2)
    ) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .hidden_valid_i(hidden_valid), .hidden_ready_o(hidden_ready),
        .hidden_index_i(hidden_index), .hidden_f16_i(hidden_f16),
        .hidden_last_i(hidden_last), .hidden_end_i(hidden_end),
        .weight_valid_i(weight_valid), .weight_ready_o(weight_ready),
        .weight_token_index_i(weight_token), .weight_feature_index_i(weight_feature),
        .weight_f16_i(weight_f16), .weight_last_feature_i(weight_last_feature),
        .weight_last_token_i(weight_last_token), .weight_end_i(weight_end),
        .logit_valid_o(logit_valid), .logit_ready_i(logit_ready),
        .logit_token_index_o(logit_token), .logit_f16_o(logit_f16),
        .acc_q47_48_o(accumulator), .logit_saturation_o(logit_saturation),
        .top_valid_o(top_valid), .top_ready_i(top_ready), .top_rank_o(top_rank),
        .top_token_index_o(top_token), .top_logit_f16_o(top_logit),
        .done_valid_o(done_valid), .done_ready_i(done_ready),
        .error_valid_o(error_valid), .error_code_o(error_code),
        .invalid_operand_o(invalid_operand), .saturation_o(saturation), .busy_o(busy)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task idle_inputs;
        begin
            clear = 1'b0;
            start_valid = 1'b0;
            hidden_valid = 1'b0;
            hidden_index = 2'd0;
            hidden_f16 = 16'd0;
            hidden_last = 1'b0;
            hidden_end = 1'b0;
            weight_valid = 1'b0;
            weight_token = 3'd0;
            weight_feature = 2'd0;
            weight_f16 = 16'd0;
            weight_last_feature = 1'b0;
            weight_last_token = 1'b0;
            weight_end = 1'b0;
            logit_ready = 1'b0;
            top_ready = 1'b0;
            done_ready = 1'b0;
        end
    endtask

    task pulse_clear;
        begin
            @(negedge clk); clear = 1'b1;
            @(posedge clk);
            @(negedge clk); clear = 1'b0;
        end
    endtask

    task accept_start;
        begin
            @(negedge clk); start_valid = 1'b1;
            if (start_ready !== 1'b1) begin
                $display("START_READY_FAIL"); failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk); start_valid = 1'b0;
        end
    endtask

    task send_hidden;
        input [1:0] index_value;
        input [15:0] bits;
        input last_value;
        begin
            @(negedge clk);
            hidden_index = index_value;
            hidden_f16 = bits;
            hidden_last = last_value;
            hidden_end = last_value;
            hidden_valid = 1'b1;
            if (hidden_ready !== 1'b1) begin
                $display("HIDDEN_READY_FAIL index=%0d", index_value); failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk); hidden_valid = 1'b0; hidden_end = 1'b0;
        end
    endtask

    task send_weight;
        input [2:0] token_value;
        input [1:0] feature_value;
        input [15:0] bits;
        begin
            @(negedge clk);
            weight_token = token_value;
            weight_feature = feature_value;
            weight_f16 = bits;
            weight_last_feature = feature_value == HIDDEN_SIZE - 1;
            weight_last_token = (token_value == VOCAB_SIZE - 1) && weight_last_feature;
            weight_end = weight_last_token;
            weight_valid = 1'b1;
            if (weight_ready !== 1'b1) begin
                $display("WEIGHT_READY_FAIL token=%0d feature=%0d", token_value, feature_value);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk); weight_valid = 1'b0; weight_end = 1'b0;
        end
    endtask

    task accept_logit;
        input [2:0] expected_token;
        input [15:0] expected_bits;
        begin
            while (logit_valid !== 1'b1) @(negedge clk);
            logit_ready = 1'b0;
            held_logit_token = logit_token;
            held_logit = logit_f16;
            held_accumulator = accumulator;
            @(posedge clk);
            @(negedge clk);
            if ((logit_token !== held_logit_token) || (logit_f16 !== held_logit) ||
                (accumulator !== held_accumulator)) begin
                $display("LOGIT_BACKPRESSURE_FAIL"); failures = failures + 1;
            end
            if ((logit_token !== expected_token) || (logit_f16 !== expected_bits) ||
                logit_saturation || invalid_operand) begin
                $display("LOGIT_VALUE_FAIL token=%0d got=%h", logit_token, logit_f16);
                failures = failures + 1;
            end
            logit_ready = 1'b1;
            @(posedge clk);
            @(negedge clk); logit_ready = 1'b0;
            logit_count = logit_count + 1;
        end
    endtask

    task send_row;
        input [2:0] token_value;
        input [15:0] w0;
        input [15:0] w1;
        input [15:0] w2;
        input [15:0] w3;
        input [15:0] expected_bits;
        begin
            send_weight(token_value, 0, w0);
            send_weight(token_value, 1, w1);
            send_weight(token_value, 2, w2);
            send_weight(token_value, 3, w3);
            accept_logit(token_value, expected_bits);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            previous_logit_stall <= 1'b0;
            previous_top_stall <= 1'b0;
        end else begin
            if (previous_logit_stall &&
                ((logit_valid !== 1'b1) || (logit_token !== held_logit_token) ||
                 (logit_f16 !== held_logit) || (accumulator !== held_accumulator))) begin
                $display("PROPERTY_LOGIT_STABILITY_FAIL"); failures = failures + 1;
            end
            if (previous_top_stall &&
                ((top_valid !== 1'b1) || (top_rank !== held_top_rank) ||
                 (top_token !== held_top_token) || (top_logit !== held_top_logit))) begin
                $display("PROPERTY_TOP_STABILITY_FAIL"); failures = failures + 1;
            end
            previous_logit_stall <= logit_valid && !logit_ready;
            previous_top_stall <= top_valid && !top_ready;
            if (logit_valid && !logit_ready) begin
                held_logit_token <= logit_token;
                held_logit <= logit_f16;
                held_accumulator <= accumulator;
            end
            if (top_valid && !top_ready) begin
                held_top_rank <= top_rank;
                held_top_token <= top_token;
                held_top_logit <= top_logit;
            end
        end
    end

    initial begin
        failures = 0;
        four_state_probes = 0;
        negative_underflow_probes = 0;
        logit_count = 0;
        top_count = 0;
        previous_logit_stall = 1'b0;
        previous_top_stall = 1'b0;
        rst_n = 1'b0;
        idle_inputs();
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0001, 0);
        send_hidden(3, 16'h0000, 1);

        send_row(0, 16'h3c00, 16'h0000, 16'h0000, 16'h0000, 16'h3c00);
        send_row(1, 16'h0000, 16'h3c00, 16'h0000, 16'h0000, 16'h3c00);
        send_row(2, 16'h3c00, 16'h3c00, 16'h0000, 16'h0000, 16'h4000);
        send_row(3, 16'hbc00, 16'h0000, 16'h0000, 16'h0000, 16'hbc00);
        send_row(4, 16'h0000, 16'h0000, 16'h8001, 16'h0000, 16'h8000);
        negative_underflow_probes = negative_underflow_probes + 1;
        if (held_accumulator !== -96'sd1) begin
            $display("NEGATIVE_UNDERFLOW_ACCUMULATOR_FAIL got=%h", held_accumulator);
            failures = failures + 1;
        end

        while (top_valid !== 1'b1) @(negedge clk);
        repeat (TOP_K) begin
            top_ready = 1'b0;
            held_top_rank = top_rank; held_top_token = top_token; held_top_logit = top_logit;
            @(posedge clk); @(negedge clk);
            if ((top_rank !== held_top_rank) || (top_token !== held_top_token) ||
                (top_logit !== held_top_logit)) begin
                $display("TOP_BACKPRESSURE_FAIL"); failures = failures + 1;
            end
            case (top_count)
                0: if ((top_token !== 2) || (top_logit !== 16'h4000)) failures = failures + 1;
                1: if ((top_token !== 0) || (top_logit !== 16'h3c00)) failures = failures + 1;
                2: if ((top_token !== 1) || (top_logit !== 16'h3c00)) failures = failures + 1;
            endcase
            top_ready = 1'b1;
            @(posedge clk); @(negedge clk);
            top_ready = 1'b0;
            top_count = top_count + 1;
        end
        if (done_valid !== 1'b1) begin $display("DONE_VALID_FAIL"); failures = failures + 1; end
        done_ready = 1'b1; @(posedge clk); @(negedge clk); done_ready = 1'b0;
        if ((start_ready !== 1'b1) || busy || saturation || error_valid) failures = failures + 1;

        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0000, 0);
        send_hidden(3, 16'h0000, 1);
        @(negedge clk);
        weight_token = 0; weight_feature = 0; weight_f16 = 16'h3c00;
        weight_last_feature = 0; weight_last_token = 0; weight_end = 1; weight_valid = 1;
        @(posedge clk); @(negedge clk); weight_valid = 0; weight_end = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd4)) begin
            $display("TRUNCATED_REJECTION_FAIL code=%0d", error_code); failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        @(negedge clk);
        hidden_valid = 1'b1; hidden_index = 0; hidden_f16 = 16'h7c00;
        hidden_last = 0; hidden_end = 0;
        @(posedge clk); @(negedge clk); hidden_valid = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd2) || !invalid_operand) begin
            $display("NONFINITE_REJECTION_FAIL"); failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        @(negedge clk);
        hidden_valid = 1'b1; hidden_index = 2'bx; hidden_f16 = 16'h3c00;
        hidden_last = 0; hidden_end = 0; four_state_probes = four_state_probes + 1;
        @(posedge clk); @(negedge clk); hidden_valid = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd1)) begin
            $display("FOUR_STATE_REJECTION_FAIL"); failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        @(negedge clk);
        hidden_valid = 1'b1; hidden_index = 0; hidden_f16 = 16'h3c00;
        hidden_last = 1'b0; hidden_end = 1'b1;
        @(posedge clk); @(negedge clk); hidden_valid = 1'b0; hidden_end = 1'b0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd4)) begin
            $display("HIDDEN_TRUNCATED_REJECTION_FAIL code=%0d", error_code);
            failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0000, 0);
        send_hidden(3, 16'h0000, 1);
        @(negedge clk);
        weight_token = 0; weight_feature = 0; weight_f16 = 16'h7c00;
        weight_last_feature = 0; weight_last_token = 0; weight_end = 0; weight_valid = 1;
        @(posedge clk); @(negedge clk); weight_valid = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd2) || !invalid_operand) begin
            $display("WEIGHT_NONFINITE_REJECTION_FAIL"); failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0000, 0);
        send_hidden(3, 16'h0000, 1);
        @(negedge clk);
        weight_token = 0; weight_feature = 1; weight_f16 = 16'h3c00;
        weight_last_feature = 0; weight_last_token = 0; weight_end = 0; weight_valid = 1;
        @(posedge clk); @(negedge clk); weight_valid = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd3)) begin
            $display("WEIGHT_ORDER_REJECTION_FAIL code=%0d", error_code);
            failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0000, 0);
        send_hidden(3, 16'h0000, 1);
        @(negedge clk);
        weight_token = 0; weight_feature = 0; weight_f16 = 16'h3c00;
        weight_last_feature = 1; weight_last_token = 0; weight_end = 0; weight_valid = 1;
        @(posedge clk); @(negedge clk); weight_valid = 0; weight_last_feature = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd4)) begin
            $display("WEIGHT_FRAMING_REJECTION_FAIL code=%0d", error_code);
            failures = failures + 1;
        end

        pulse_clear();
        accept_start();
        send_hidden(0, 16'h3c00, 0);
        send_hidden(1, 16'h3c00, 0);
        send_hidden(2, 16'h0000, 0);
        send_hidden(3, 16'h0000, 1);
        @(negedge clk);
        weight_token = 0; weight_feature = 0; weight_f16 = 16'h3c00;
        weight_last_feature = 0; weight_last_token = 0; weight_end = 1'bx;
        weight_valid = 1; four_state_probes = four_state_probes + 1;
        @(posedge clk); @(negedge clk); weight_valid = 0; weight_end = 0;
        if ((error_valid !== 1'b1) || (error_code !== 4'd1)) begin
            $display("WEIGHT_FOUR_STATE_REJECTION_FAIL code=%0d", error_code);
            failures = failures + 1;
        end

        @(negedge clk); rst_n = 1'b0;
        @(posedge clk); @(negedge clk); rst_n = 1'b1;
        @(posedge clk); @(negedge clk);
        if ((start_ready !== 1'b1) || busy || error_valid || invalid_operand || saturation) begin
            $display("RESET_RECOVERY_FAIL"); failures = failures + 1;
        end

        if ((failures == 0) && (logit_count == VOCAB_SIZE) &&
            (top_count == TOP_K) && (four_state_probes == 2) &&
            (negative_underflow_probes == 1)) begin
            $display("STREAMING_LM_HEAD_PROTOCOL_PASS logits=%0d top_k=%0d four_state=%0d negative_underflow=%0d",
                     logit_count, top_count, four_state_probes,
                     negative_underflow_probes);
            $finish;
        end
        $fatal(1, "STREAMING_LM_HEAD_PROTOCOL_FAIL failures=%0d logits=%0d top=%0d",
               failures, logit_count, top_count);
    end
endmodule

`default_nettype wire
