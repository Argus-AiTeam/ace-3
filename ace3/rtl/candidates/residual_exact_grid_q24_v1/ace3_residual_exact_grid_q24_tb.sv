`timescale 1ns/1ps

module ace3_residual_exact_grid_q24_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 0, clear = 0, start_valid = 0, in_valid = 0, out_ready = 0;
    reg [12:0] count = 0;
    reg signed [63:0] state_in = 0;
    reg tag_in = 0;
    reg [15:0] addend = 0;
    wire start_ready, in_ready, out_valid, tag_out, last, busy;
    wire signed [63:0] state_out;
    wire [15:0] view_out;
    wire [12:0] index_out;
    wire [3:0] fault;
    wire [100:0] tuple_out = {busy, out_valid, state_out, tag_out,
                              view_out, index_out, last, fault};
    reg [100:0] held;
    reg [31:0] random_state = 32'h00ace324;
    reg signed [63:0] expected_state, saved_state;
    reg expected_tag, saved_tag;
    reg [15:0] expected_view;
    reg [3:0] expected_fault;
    integer file_handle, rows, scanned, row, mode, j, delay_cycles, step, diagnosed = 0;
    string vectors_path, wave_path;

    ace3_residual_exact_grid_q24_core dut (
        .clk_i(clk), .rst_ni(rst), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .element_count_i(count), .in_valid_i(in_valid), .in_ready_o(in_ready),
        .in_state_q24_i(state_in), .in_state_negzero_i(tag_in),
        .in_addend_f16_i(addend), .out_valid_o(out_valid), .out_ready_i(out_ready),
        .out_state_q24_o(state_out), .out_state_negzero_o(tag_out),
        .out_f16_o(view_out), .out_index_o(index_out), .out_last_o(last),
        .out_fault_code_o(fault), .busy_o(busy)
    );

    task tick;
        begin @(posedge clk); #1; end
    endtask

    task cancel(input bit asynchronous);
        begin
            @(negedge clk);
            if (asynchronous) rst = 0; else clear = 1;
            // Cancellation wins even when both sides request a transfer.
            in_valid = 1; out_ready = 1; start_valid = 1; count = 1;
            if (asynchronous) begin
                #1;
                if (tuple_out !== 101'd0) $fatal(1, "asynchronous reset cancellation");
            end
            tick;
            if (tuple_out !== 101'd0) $fatal(1, "reset/clear cancellation");
            @(negedge clk);
            rst = 1; clear = 0; in_valid = 0; out_ready = 0; start_valid = 0;
            #1;
            if (start_ready !== 1'b1 || in_ready !== 1'b0)
                $fatal(1, "cancel did not restore idle");
        end
    endtask

    task start_vector(input [12:0] length);
        begin
            @(negedge clk);
            count = length; start_valid = 1; out_ready = 0; in_valid = 0;
            #1;
            if (start_ready !== 1'b1 || in_ready !== 1'b0)
                $fatal(1, "start not idle or same-cycle input allowed");
            tick;
            if (busy !== 1'b1 || out_valid !== 1'b0) $fatal(1, "start handshake");
            @(negedge clk); start_valid = 0;
        end
    endtask

    task feed(input signed [63:0] integer_value, input zero_tag, input [15:0] word);
        begin
            @(negedge clk);
            state_in = integer_value; tag_in = zero_tag; addend = word; in_valid = 1;
            #1;
            if (in_ready !== 1'b1) $fatal(1, "input unexpectedly blocked");
            tick;
            @(negedge clk); in_valid = 0;
        end
    endtask

    task expect_pair(input signed [63:0] integer_value, input zero_tag,
                     input [15:0] word, input [12:0] index_value,
                     input last_value, input [3:0] code);
        begin
            if (tuple_out !== {1'b1, 1'b1, integer_value, zero_tag,
                              word, index_value, last_value, code})
                $fatal(1, "row=%0d expected I=%h Z=%b H=%h index=%0d last=%b fault=%0d; actual=%h",
                       row, integer_value, zero_tag, word, index_value, last_value, code, tuple_out);
        end
    endtask

    task stall(input integer cycles);
        integer k;
        begin
            held = tuple_out;
            for (k = 0; k < cycles; k = k + 1) begin
                @(negedge clk); out_ready = 0; in_valid = 1;
                #1;
                if (in_ready !== 1'b0) $fatal(1, "input accepted while full");
                tick;
                if (tuple_out !== held) $fatal(1, "stalled tuple changed");
            end
            @(negedge clk); in_valid = 0;
        end
    endtask

    task drain_last;
        begin
            @(negedge clk); out_ready = 1; in_valid = 0;
            tick;
            if (busy !== 1'b0 || out_valid !== 1'b0) $fatal(1, "last did not complete");
            @(negedge clk); out_ready = 0;
        end
    endtask

    task check_fault_lock;
        begin
            @(negedge clk); count = 1; start_valid = 1; in_valid = 1;
            #1;
            if (start_ready !== 1'b0 || in_ready !== 1'b0) $fatal(1, "fault not locked");
            tick;
            if (out_valid !== 1'b0 || busy !== 1'b0) $fatal(1, "fault restarted");
            cancel(0);
        end
    endtask

    initial begin
        #20000000;
        $fatal(1, "watchdog: incomplete Q24 simulation");
    end

    initial begin
        if (!$value$plusargs("vectors=%s", vectors_path) ||
            !$value$plusargs("wave=%s", wave_path)) $fatal(1, "missing vectors/wave");
        $dumpfile(wave_path);
        $dumpvars(1, ace3_residual_exact_grid_q24_tb);
        cancel(1);
        // Keep a bounded waveform of protocol/fault tests; arithmetic failures
        // report the exact preserved vector row instead of a very large trace.
        file_handle = $fopen(vectors_path, "r");
        if (!file_handle) $fatal(1, "cannot open vectors");
        scanned = $fscanf(file_handle, "%d\n", rows);
        if (scanned != 1 || rows < 63488) $fatal(1, "invalid vector count");
        for (row = 0; row < rows; row = row + 1) begin
            if (row == 16) $dumpoff;
            scanned = $fscanf(file_handle, "%h %h %h %h %h %h %h\n",
                              state_in, tag_in, addend, expected_state,
                              expected_tag, expected_view, expected_fault);
            if (scanned != 7) $fatal(1, "truncated vector row %0d", row);
            start_vector(1);
            feed(state_in, tag_in, addend);
            expect_pair(expected_state, expected_tag, expected_view, 0, 1, expected_fault);
            if (row % 257 == 0 || expected_fault != 0) stall(3);
            drain_last;
            if (expected_fault != 0) check_fault_lock;
        end
        scanned = $fscanf(file_handle, "%h", state_in);
        if (scanned != -1) $fatal(1, "trailing vector data");
        $fclose(file_handle);
        $dumpon;

        // The same 896 transactions, including elastic replacement, in both
        // no-stall and deterministic randomly stalled schedules.
        for (mode = 0; mode < 2; mode = mode + 1) begin
            start_vector(896);
            for (j = 0; j < 896; j = j + 1) begin
                if (mode == 1 && j != 0) begin
                    random_state = random_state * 32'd1664525 + 32'd1013904223;
                    delay_cycles = random_state[31:30];
                    stall(delay_cycles);
                end
                @(negedge clk);
                out_ready = 1; in_valid = 1; state_in = j; tag_in = 0; addend = 0;
                #1;
                if (in_ready !== 1'b1) $fatal(1, "elastic replacement blocked");
                tick;
                expect_pair(64'(j), 0, 16'(j), 13'(j), j == 895, 0);
            end
            drain_last;
        end

        saved_state = 64'd16777216; saved_tag = 0;
        for (step = 1; step <= 8; step = step + 1) begin
            start_vector(1);
            feed(saved_state, saved_tag, 16'h1000);
            expected_view = 16'h3c00 + 16'((step / 2) + ((step % 2) && ((step / 2) % 2)));
            expect_pair(64'd16777216 + 64'(step) * 64'd8192, 0, expected_view, 0, 1, 0);
            saved_state = state_out; saved_tag = tag_out;
            stall(2);
            drain_last;
        end

        // Invalid counts and unknown controls are explicitly diagnosed; they
        // cannot be counted as accepted work or as a successful zero-work run.
        for (j = 0; j < 5; j = j + 1) begin
            @(negedge clk);
            case (j)
                0: count = 0;
                1: count = 897;
                2: count = 8191;
                3: count = 13'bx;
                4: count = 13'bz;
            endcase
            start_valid = 1; #1;
            if (start_ready !== 1'b0) $fatal(1, "invalid count ready");
            tick;
            if (busy !== 1'b0 || out_valid !== 1'b0) $fatal(1, "invalid count accepted");
            diagnosed = diagnosed + 1;
        end
        @(negedge clk); start_valid = 0; count = 1;
        for (mode = 0; mode < 2; mode = mode + 1) begin
            @(negedge clk); start_valid = mode ? 1'bz : 1'bx;
            tick;
            if (busy !== 1'b0) $fatal(1, "unknown start accepted");
            diagnosed = diagnosed + 1;
        end
        start_vector(1);
        for (mode = 0; mode < 2; mode = mode + 1) begin
            @(negedge clk); in_valid = mode ? 1'bz : 1'bx;
            tick;
            if (out_valid !== 1'b0) $fatal(1, "unknown input valid accepted");
            diagnosed = diagnosed + 1;
        end
        feed(0, 0, 0);
        held = tuple_out;
        for (j = 0; j < 6; j = j + 1) begin
            @(negedge clk);
            out_ready = 1; clear = 0; rst = 1; in_valid = 1;
            case (j)
                0: out_ready = 1'bx;
                1: out_ready = 1'bz;
                2: clear = 1'bx;
                3: clear = 1'bz;
                4: rst = 1'bx;
                5: rst = 1'bz;
            endcase
            tick;
            if (tuple_out !== held) $fatal(1, "unknown control changed pending output");
            diagnosed = diagnosed + 1;
        end
        @(negedge clk); rst = 1; clear = 0; in_valid = 0;
        drain_last;
        for (j = 0; j < 6; j = j + 1) begin
            start_vector(2);
            case (j)
                0: feed(64'bx, 0, 0);
                1: feed(64'bz, 0, 0);
                2: feed(0, 1'bx, 0);
                3: feed(0, 1'bz, 0);
                4: feed(0, 0, 16'bx);
                5: feed(0, 0, 16'bz);
            endcase
            expect_pair(0, 0, 0, 0, 1, 1);
            stall(3);
            drain_last;
            check_fault_lock;
        end
        for (mode = 0; mode < 2; mode = mode + 1) begin
            start_vector(2); cancel(mode != 0);
            start_vector(1); feed(1, 0, 0); cancel(mode != 0);
            start_vector(2); feed(1, 0, 0);
            @(negedge clk); out_ready = 1; tick;
            if (busy !== 1'b1 || out_valid !== 1'b0) $fatal(1, "partial vector completion");
            cancel(mode != 0);
            start_vector(2); feed(0, 0, 16'h7c00); cancel(mode != 0);
            start_vector(1); feed(0, 0, 0); expect_pair(0, 0, 0, 0, 1, 0); drain_last;
        end
        start_vector(3); feed(1, 0, 0);
        @(negedge clk); out_ready = 1; tick;
        @(negedge clk); out_ready = 0;
        feed(0, 0, 16'h7c00);
        expect_pair(0, 0, 0, 1, 1, 1);
        stall(3); drain_last; check_fault_lock;
        if (diagnosed != 15) $fatal(1, "missing control diagnostics");
        $display("Q24_PRIMITIVE_PASS rows=%0d pipeline_pairs=1792 recurrence_steps=8 xz_payloads=6 control_diagnostics=15", rows);
        $finish;
    end
endmodule
