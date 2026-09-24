`timescale 1ns/1ps
`default_nettype none

module ace3_q_only_fused_rope_candidate_tb;
    reg clk = 0, rst_n = 0, clear = 0, valid = 0, ready = 0, key = 0;
    reg [3:0] head = 0;
    reg [4:0] pair_index = 31;
    reg [14:0] position = 32767;
    reg [15:0] low_value, high_value, cosine, sine;
    wire in_ready, out_valid, out_key, invalid, saturation;
    wire [3:0] out_head;
    wire [4:0] out_pair;
    wire [14:0] out_position;
    wire [15:0] out_low, out_high;
    wire base_ready, base_valid, base_key, base_invalid, base_saturation;
    wire [3:0] base_head;
    wire [4:0] base_pair;
    wire [14:0] base_position;
    wire [15:0] base_low, base_high;
    reg [59:0] held;
    integer fd, fields, count = 0;
    integer key_in, low_in, high_in, cos_in, sin_in;
    integer expected_low, expected_high, expected_invalid, expected_saturation;
    string cases_path;

    ace3_qwen2_rope_pair dut (
        clk, rst_n, clear, valid, in_ready, key, head, pair_index, position,
        low_value, high_value, cosine, sine, out_valid, ready, out_key,
        out_head, out_pair, out_position, out_low, out_high, invalid, saturation
    );
    ace3_qwen2_rope_pair_baseline baseline (
        clk, rst_n, clear, valid, base_ready, key, head, pair_index, position,
        low_value, high_value, cosine, sine, base_valid, ready, base_key,
        base_head, base_pair, base_position, base_low, base_high,
        base_invalid, base_saturation
    );

    task tick;
        begin #5; clk = 1; #5; clk = 0; #1; end
    endtask

    initial begin
        if (!$value$plusargs("CASES=%s", cases_path)) $fatal(1, "missing cases");
        fd = $fopen(cases_path, "r");
        if (!fd) $fatal(1, "cannot open cases");
        tick;
        if (out_valid || in_ready) $fatal(1, "reset protocol");
        rst_n = 1;
        while (!$feof(fd)) begin
            fields = $fscanf(fd, "%h %h %h %h %h %h %h %h %h\n",
                key_in, low_in, high_in, cos_in, sin_in,
                expected_low, expected_high, expected_invalid, expected_saturation);
            if (fields != 9) $fatal(1, "malformed case %0d", count);
            key = key_in[0]; head = key ? 1 : 13;
            low_value = low_in[15:0]; high_value = high_in[15:0];
            cosine = cos_in[15:0]; sine = sin_in[15:0];
            valid = 1; ready = 0;
            #1;
            if (!in_ready || !base_ready) $fatal(1, "input handshake");
            tick;
            valid = 0;
            if (!out_valid || !base_valid || out_key !== key ||
                out_head !== head || out_pair !== pair_index ||
                out_position !== position) $fatal(1, "output metadata");
            if (!key && {out_low, out_high, invalid, saturation} !==
                {expected_low[15:0], expected_high[15:0],
                 expected_invalid[0], expected_saturation[0]})
                $fatal(1, "Q case %0d actual=%h/%h/%b/%b expected=%h/%h/%b/%b",
                    count, out_low, out_high, invalid, saturation,
                    expected_low[15:0], expected_high[15:0],
                    expected_invalid[0], expected_saturation[0]);
            if (key && {out_low, out_high, invalid, saturation} !==
                       {base_low, base_high, base_invalid, base_saturation})
                $fatal(1, "K baseline regression %0d", count);
            held = {out_valid, out_key, out_head, out_pair, out_position,
                    out_low, out_high, invalid, saturation};
            repeat (2) begin
                tick;
                if (in_ready || held !==
                    {out_valid, out_key, out_head, out_pair, out_position,
                     out_low, out_high, invalid, saturation})
                    $fatal(1, "backpressure instability");
            end
            ready = 1;
            tick;
            if (out_valid) $fatal(1, "output not consumed");
            count = count + 1;
        end
        key = 1; head = 2; valid = 1;
        #1;
        if (in_ready) $fatal(1, "invalid key geometry accepted");
        head = 1;
        tick;
        clear = 1;
        #1;
        if (in_ready) $fatal(1, "clear handshake");
        tick;
        if (out_valid) $fatal(1, "clear did not invalidate");
        $fclose(fd);
        $display("PASS Q-only boundary and K-preservation cases=%0d", count);
        $finish;
    end
endmodule

`default_nettype wire
