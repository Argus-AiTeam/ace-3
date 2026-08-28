`timescale 1ns/1ps
`default_nettype none

module ace3_model24_layer_controller_tb;
    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;
    reg clear_i = 1'b0;
    reg start_valid_i = 1'b0;
    wire start_ready_o;
    reg [1:0] start_cache_slot_i = 2'd0;
    reg [14:0] start_position_i = 15'd0;
    wire layer_start_valid_o;
    reg layer_start_ready_i = 1'b0;
    wire [4:0] layer_start_index_o;
    wire [1:0] layer_start_cache_slot_o;
    wire [14:0] layer_start_position_o;
    reg layer_done_valid_i = 1'b0;
    wire layer_done_ready_o;
    reg [4:0] layer_done_index_i = 5'd0;
    reg layer_done_fault_i = 1'b0;
    wire checkpoint_valid_o;
    reg checkpoint_ready_i = 1'b0;
    wire [4:0] checkpoint_completed_layer_o;
    wire [4:0] checkpoint_next_layer_o;
    wire checkpoint_terminal_o;
    wire done_valid_o;
    reg done_ready_i = 1'b0;
    wire [1:0] done_cache_slot_o;
    wire [14:0] done_position_o;
    wire busy_o;
    wire fault_o;
    wire [4:0] active_layer_o;

    reg [15:0] cascade_events [0:23];
    reg [1023:0] vector_dir;
    reg [1023:0] raw_dir;
    reg record_events = 1'b0;
    reg terminal_written = 1'b0;
    integer raw_events;
    integer raw_terminal;
    integer inject_failure_after_launch = 0;
    integer launch_count = 0;
    integer checkpoint_count = 0;
    integer done_count = 0;
    integer layer;
    integer cycles = 0;

    ace3_model24_layer_controller dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(start_valid_i),
        .start_ready_o(start_ready_o),
        .start_cache_slot_i(start_cache_slot_i),
        .start_position_i(start_position_i),
        .layer_start_valid_o(layer_start_valid_o),
        .layer_start_ready_i(layer_start_ready_i),
        .layer_start_index_o(layer_start_index_o),
        .layer_start_cache_slot_o(layer_start_cache_slot_o),
        .layer_start_position_o(layer_start_position_o),
        .layer_done_valid_i(layer_done_valid_i),
        .layer_done_ready_o(layer_done_ready_o),
        .layer_done_index_i(layer_done_index_i),
        .layer_done_fault_i(layer_done_fault_i),
        .checkpoint_valid_o(checkpoint_valid_o),
        .checkpoint_ready_i(checkpoint_ready_i),
        .checkpoint_completed_layer_o(checkpoint_completed_layer_o),
        .checkpoint_next_layer_o(checkpoint_next_layer_o),
        .checkpoint_terminal_o(checkpoint_terminal_o),
        .done_valid_o(done_valid_o),
        .done_ready_i(done_ready_i),
        .done_cache_slot_o(done_cache_slot_o),
        .done_position_o(done_position_o),
        .busy_o(busy_o),
        .fault_o(fault_o),
        .active_layer_o(active_layer_o)
    );

    always #5 clk_i = ~clk_i;

    always @(posedge clk_i) begin
        cycles <= cycles + 1;
        if (cycles > 1000)
            $fatal(1, "MODEL24_LAYER_CONTROLLER_TIMEOUT");
        if (record_events && layer_start_valid_o && layer_start_ready_i) begin
            $fwrite(raw_events, "%08x\n", 32'h10000000 | layer_start_index_o);
            $fflush(raw_events);
            launch_count <= launch_count + 1;
            if (inject_failure_after_launch == (launch_count + 1)) begin
                $fwrite(raw_terminal,
                        "schema=ace3_model24_controller_raw_v1 natural_terminal=0 exit_code=2 launches=%0d checkpoints=%0d done=%0d terminal_layer=none\n",
                        launch_count + 1, checkpoint_count, done_count);
                $fflush(raw_terminal);
                terminal_written <= 1'b1;
                $fatal(1, "MODEL24_LAYER_CONTROLLER_INJECTED_FAILURE");
            end
        end
        if (record_events && checkpoint_valid_o && checkpoint_ready_i) begin
            $fwrite(raw_events, "%08x\n",
                    32'h20000000 |
                    (checkpoint_terminal_o << 10) |
                    (checkpoint_next_layer_o << 5) |
                    checkpoint_completed_layer_o);
            $fflush(raw_events);
            checkpoint_count <= checkpoint_count + 1;
        end
        if (record_events && done_valid_o && done_ready_i) begin
            $fwrite(raw_events, "%08x\n", 32'h30000000 | active_layer_o);
            $fflush(raw_events);
            done_count <= done_count + 1;
        end
    end

    task reset_dut;
        begin
            rst_ni = 1'b0;
            repeat (3) @(posedge clk_i);
            @(negedge clk_i);
            rst_ni = 1'b1;
            @(posedge clk_i);
            if ((start_ready_o !== 1'b1) || (fault_o !== 1'b0))
                $fatal(1, "reset did not restore idle");
        end
    endtask

    task clear_dut;
        begin
            @(negedge clk_i);
            clear_i = 1'b1;
            @(posedge clk_i);
            @(negedge clk_i);
            clear_i = 1'b0;
            if ((fault_o !== 1'b0) || (start_ready_o !== 1'b1))
                $fatal(1, "clear did not restore idle");
        end
    endtask

    task start_token;
        input [1:0] cache_slot;
        input [14:0] position;
        begin
            @(negedge clk_i);
            start_cache_slot_i = cache_slot;
            start_position_i = position;
            start_valid_i = 1'b1;
            if (start_ready_o !== 1'b1)
                $fatal(1, "start was not accepted from idle");
            @(posedge clk_i);
            @(negedge clk_i);
            start_valid_i = 1'b0;
        end
    endtask

    task accept_layer_start;
        input integer expected_layer;
        begin
            if ((layer_start_valid_o !== 1'b1) ||
                (layer_start_index_o !== expected_layer[4:0]) ||
                (layer_start_cache_slot_o !== 2'd1) ||
                (layer_start_position_o !== 15'd127))
                $fatal(1, "layer launch metadata mismatch at %0d", expected_layer);
            repeat (2) begin
                @(posedge clk_i);
                if ((layer_start_valid_o !== 1'b1) ||
                    (layer_start_index_o !== expected_layer[4:0]))
                    $fatal(1, "layer launch was not retained at %0d", expected_layer);
            end
            @(negedge clk_i);
            layer_start_ready_i = 1'b1;
            @(posedge clk_i);
            @(negedge clk_i);
            layer_start_ready_i = 1'b0;
            if (layer_done_ready_o !== 1'b1)
                $fatal(1, "layer completion was not enabled at %0d", expected_layer);
        end
    endtask

    task complete_layer;
        input integer expected_layer;
        begin
            layer_done_index_i = expected_layer[4:0];
            layer_done_fault_i = 1'b0;
            layer_done_valid_i = 1'b1;
            @(posedge clk_i);
            @(negedge clk_i);
            layer_done_valid_i = 1'b0;
            if (checkpoint_valid_o !== 1'b1)
                $fatal(1, "checkpoint missing after layer %0d", expected_layer);
        end
    endtask

    task accept_checkpoint;
        input integer expected_layer;
        reg [15:0] expected;
        begin
            expected = cascade_events[expected_layer];
            if ((checkpoint_completed_layer_o !== expected[4:0]) ||
                (checkpoint_next_layer_o !== expected[9:5]) ||
                (checkpoint_terminal_o !== expected[10]))
                $fatal(1, "checkpoint payload mismatch at layer %0d", expected_layer);
            repeat (2) begin
                @(posedge clk_i);
                if ((checkpoint_valid_o !== 1'b1) ||
                    (checkpoint_completed_layer_o !== expected[4:0]) ||
                    (checkpoint_next_layer_o !== expected[9:5]) ||
                    (checkpoint_terminal_o !== expected[10]) ||
                    (layer_start_valid_o !== 1'b0))
                    $fatal(1, "checkpoint backpressure failure at layer %0d",
                           expected_layer);
            end
            @(negedge clk_i);
            checkpoint_ready_i = 1'b1;
            @(posedge clk_i);
            @(negedge clk_i);
            checkpoint_ready_i = 1'b0;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "VECTOR_DIR is required");
        if (!$value$plusargs("RAW_DIR=%s", raw_dir))
            $fatal(1, "RAW_DIR is required");
        if (!$value$plusargs("INJECT_FAILURE_AFTER_LAUNCH=%d",
                             inject_failure_after_launch))
            inject_failure_after_launch = 0;
        raw_events = $fopen({raw_dir, "/controller_events.hex"}, "w");
        raw_terminal = $fopen({raw_dir, "/terminal.txt"}, "w");
        if ((raw_events == 0) || (raw_terminal == 0))
            $fatal(1, "unable to open controller raw evidence");
        $readmemh({vector_dir, "/cascade_events.hex"}, cascade_events);
        reset_dut();

        record_events = 1'b1;
        start_token(2'd1, 15'd127);
        for (layer = 0; layer < 24; layer = layer + 1) begin
            accept_layer_start(layer);
            complete_layer(layer);
            accept_checkpoint(layer);
        end
        if ((done_valid_o !== 1'b1) || (done_cache_slot_o !== 2'd1) ||
            (done_position_o !== 15'd127) || (active_layer_o !== 5'd23))
            $fatal(1, "terminal metadata mismatch");
        repeat (2) begin
            @(posedge clk_i);
            if (done_valid_o !== 1'b1)
                $fatal(1, "terminal completion was not retained");
        end
        @(negedge clk_i);
        done_ready_i = 1'b1;
        @(posedge clk_i);
        @(negedge clk_i);
        done_ready_i = 1'b0;
        record_events = 1'b0;
        if ((start_ready_o !== 1'b1) || (busy_o !== 1'b0))
            $fatal(1, "controller did not return to idle");
        if ((launch_count != 24) || (checkpoint_count != 24) ||
            (done_count != 1))
            $fatal(1, "raw controller event counts are incomplete");

        start_token(2'd2, 15'd128);
        if ((fault_o !== 1'b1) || (start_ready_o !== 1'b0) ||
            (layer_start_valid_o !== 1'b0))
            $fatal(1, "out-of-range start did not fail closed");
        clear_dut();

        @(negedge clk_i);
        layer_done_index_i = 5'd0;
        layer_done_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        if (fault_o !== 1'b1)
            $fatal(1, "unsolicited completion did not fault");
        @(negedge clk_i);
        layer_done_valid_i = 1'b0;
        clear_dut();

        start_token(2'd1, 15'd127);
        accept_layer_start(0);
        layer_done_index_i = 5'd1;
        layer_done_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        if ((fault_o !== 1'b1) || (checkpoint_valid_o !== 1'b0) ||
            (done_valid_o !== 1'b0))
            $fatal(1, "mismatched completion did not fail closed");
        @(negedge clk_i);
        layer_done_valid_i = 1'b0;
        clear_dut();

        start_token(2'd1, 15'd127);
        accept_layer_start(0);
        reset_dut();
        if ((busy_o !== 1'b0) || (active_layer_o !== 5'd0))
            $fatal(1, "reset during a layer did not restore idle");

        start_token(2'd1, 15'd127);
        accept_layer_start(0);
        layer_done_index_i = 5'd0;
        layer_done_fault_i = 1'b1;
        layer_done_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        if ((fault_o !== 1'b1) || (checkpoint_valid_o !== 1'b0))
            $fatal(1, "faulted layer completion did not fail closed");
        @(negedge clk_i);
        layer_done_valid_i = 1'b0;
        layer_done_fault_i = 1'b0;
        clear_dut();

        $fwrite(raw_terminal,
                "schema=ace3_model24_controller_raw_v1 natural_terminal=1 exit_code=0 launches=%0d checkpoints=%0d done=%0d terminal_layer=23\n",
                launch_count, checkpoint_count, done_count);
        $fflush(raw_terminal);
        terminal_written = 1'b1;
        $fclose(raw_events);
        $fclose(raw_terminal);
        $display("MODEL24_LAYER_CONTROLLER_IVERILOG_PASS layers=24 checkpoints=24 retained_backpressure=pass terminal=layer23 fail_closed=pass clear_recovery=pass numerical_rtl=not_claimed");
        $finish;
    end

    final begin
        if (!terminal_written) begin
            if (raw_events != 0)
                $fclose(raw_events);
            if (raw_terminal != 0)
                $fclose(raw_terminal);
        end
    end
endmodule

`default_nettype wire
