`timescale 1ns/1ps
`default_nettype none

module ace3_layer08_stage08_diagnostic_tb;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0;
    reg start_valid = 0, pair_valid = 0, out_ready = 0;
    wire start_ready, pair_ready, out_valid;
    reg [3:0] query_head = 0, key_head = 0;
    reg [14:0] key_position = 0;
    reg [15:0] q = 0, k = 0;
    wire [15:0] score;
    wire [3:0] out_query_head, out_key_head;
    wire [14:0] out_query_position, out_key_position;
    wire causal, cache_miss, invalid_operand, saturation, busy;
    reg [15:0] queries [0:895];
    reg [15:0] keys [0:383];
    integer head, position, dimension, raw, terminal, count = 0;

    ace3_attention_score_core #(.HEAD_DIM(64)) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(1'b0),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .query_head_i(query_head), .key_head_i(key_head),
        .query_position_i(15'd2), .key_position_i(key_position),
        .pair_valid_i(pair_valid), .pair_ready_o(pair_ready),
        .q_f16_i(q), .k_f16_i(k), .cache_hit_i(1'b1),
        .out_valid_o(out_valid), .out_ready_i(out_ready),
        .score_f16_o(score), .query_head_o(out_query_head),
        .key_head_o(out_key_head), .query_position_o(out_query_position),
        .key_position_o(out_key_position), .causal_o(causal),
        .cache_miss_o(cache_miss), .invalid_operand_o(invalid_operand),
        .saturation_o(saturation), .busy_o(busy)
    );

    initial begin
        #1000000;
        $fatal(1, "score replay timeout");
    end

    initial begin
        $readmemh("q.hex", queries);
        $readmemh("k.hex", keys);
        $dumpfile("score.vcd");
        $dumpvars(0, dut);
        raw = $fopen("scores.hex", "w");
        if (!raw) $fatal(1, "cannot open score output");
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (head = 0; head < 14; head = head + 1) begin
            for (position = 0; position < 3; position = position + 1) begin
                @(negedge clk);
                query_head = head;
                key_head = head / 7;
                key_position = position;
                #1;
                if (start_ready !== 1'b1) $fatal(1, "start not ready");
                start_valid = 1;
                @(negedge clk);
                start_valid = 0;
                for (dimension = 0; dimension < 64; dimension = dimension + 1) begin
                    q = queries[head * 64 + dimension];
                    k = keys[position * 128 + (head / 7) * 64 + dimension];
                    pair_valid = 1;
                    #1;
                    if (pair_ready !== 1'b1) $fatal(1, "pair not ready");
                    @(negedge clk);
                end
                pair_valid = 0;
                if (out_valid !== 1'b1 || out_query_head !== query_head ||
                    out_key_head !== key_head || out_query_position !== 15'd2 ||
                    out_key_position !== key_position ||
                    {causal, cache_miss, invalid_operand, saturation} !== 4'b1000 ||
                    (^score === 1'bx))
                    $fatal(1, "bad score result metadata/flags/value");
                $fdisplay(raw, "%04x", score);
                $fflush(raw);
                count = count + 1;
                out_ready = 1;
                @(negedge clk);
                out_ready = 0;
            end
        end
        if (count != 42) $fatal(1, "wrong score count");
        $fclose(raw);
        terminal = $fopen("terminal.txt", "w");
        if (!terminal) $fatal(1, "cannot open terminal");
        $fdisplay(terminal, "natural_terminal=1 exit_code=0 scores=42");
        $fclose(terminal);
        $finish;
    end
endmodule

`default_nettype wire
