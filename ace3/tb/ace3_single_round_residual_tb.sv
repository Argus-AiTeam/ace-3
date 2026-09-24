`timescale 1ns/1ps
`default_nettype none

module ace3_single_round_residual_tb;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst = 0, clear = 0, start = 0, valid = 0, ready = 0;
    reg [12:0] count = 0;
    reg [15:0] h, o, d;
    wire start_ready, in_ready, out_valid, last, invalid, saturation, busy;
    wire [15:0] result;
    wire [12:0] index;
    ace3_fp16_single_round_residual_core dut (
        .clk_i(clk), .rst_ni(rst), .clear_i(clear),
        .start_valid_i(start), .start_ready_o(start_ready), .element_count_i(count),
        .in_valid_i(valid), .in_ready_o(in_ready),
        .projection_f16_i(d), .residual_f16_i(h), .attention_f16_i(o),
        .out_valid_o(out_valid), .out_ready_i(ready), .out_f16_o(result),
        .out_index_o(index), .out_last_o(last), .invalid_operand_o(invalid),
        .saturation_o(saturation), .busy_o(busy));
    integer fd, raw, total, row, local_index, scanned, expected_invalid, expected_sat;
    reg [15:0] expected;
    string input_path, raw_path, wave_path;
    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask
    task compare_output;
        begin
            if (out_valid !== 1'b1 || result !== expected ||
                invalid !== (expected_invalid != 0) ||
                saturation !== (expected_sat != 0) ||
                index !== 13'(local_index) || last !== (local_index == count-1) ||
                busy !== 1'b1)
                $fatal(1, "row=%0d actual=%h expected=%h index=%0d invalid=%b sat=%b",
                       row, result, expected, index, invalid, saturation);
        end
    endtask
    initial begin
        if (!$value$plusargs("INPUT=%s", input_path) ||
            !$value$plusargs("RAW=%s", raw_path) ||
            !$value$plusargs("WAVE=%s", wave_path))
            $fatal(1, "missing paths");
        fd = $fopen(input_path, "r");
        raw = $fopen(raw_path, "w");
        if (!fd || !raw) $fatal(1, "input/output open failed");
        $dumpfile(wave_path); $dumpvars(0, ace3_single_round_residual_tb);
        scanned = $fscanf(fd, "%d\n", total);
        if (scanned != 1 || total < 1) $fatal(1, "invalid vector count");
        tick(); rst = 1; tick();
        count = 0; #1;
        if (start_ready !== 0) $fatal(1, "zero count admitted");
        count = 897; #1;
        if (start_ready !== 0) $fatal(1, "oversize count admitted");
        row = 0;
        while (row < total) begin
            count = 13'((total-row > 896) ? 896 : total-row);
            #1;
            if (start_ready !== 1) $fatal(1, "start not ready");
            start = 1; tick(); start = 0;
            for (local_index = 0; local_index < count; local_index = local_index+1) begin
                scanned = $fscanf(fd, "%h %h %h %h %d %d\n",
                                  h, o, d, expected, expected_invalid, expected_sat);
                if (scanned != 6) $fatal(1, "truncated input");
                if (in_ready !== 1) $fatal(1, "input not ready");
                valid = 1; tick(); valid = 0; #1;
                compare_output();
                if (in_ready !== 0) $fatal(1, "backpressure not propagated");
                tick(); compare_output();
                $fdisplay(raw, "%0d %04h %0d %0d", row, result, invalid, saturation);
                ready = 1; tick(); ready = 0; #1;
                if (out_valid !== 0) $fatal(1, "output not consumed");
                row = row+1;
            end
        end
        // Clear cancels a pending output and resets the next transaction index.
        count = 1; start = 1; tick(); start = 0;
        h = 16'h3c00; o = 0; d = 0; valid = 1; tick(); valid = 0;
        clear = 1; #1;
        if (in_ready !== 0 || start_ready !== 0) $fatal(1, "clear handshake");
        tick(); clear = 0; #1;
        if (out_valid !== 0 || busy !== 0 || start_ready !== 1)
            $fatal(1, "clear did not cancel transaction");
        rst = 0; #1;
        if (out_valid !== 0 || busy !== 0) $fatal(1, "asynchronous reset");
        $fclose(fd); $fclose(raw);
        $display("SINGLE_ROUND_RESIDUAL_PASS cases=%0d", total);
        $finish;
    end
    initial begin #2000000; $fatal(1, "watchdog"); end
endmodule

`default_nettype wire
