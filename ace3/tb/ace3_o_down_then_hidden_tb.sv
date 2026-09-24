`timescale 1ns/1ps
`default_nettype none

module ace3_o_down_then_hidden_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, clear = 0, start = 0, valid = 0, ready = 0;
    reg [15:0] hidden = 0, attention_o = 0, down = 0;
    wire [15:0] inner, result;
    wire inner_invalid, inner_saturation, start_ready, in_ready, out_valid;
    wire [12:0] index;
    wire last, invalid, saturation, busy;
    reg [15:0] expected_inner, expected_final;
    integer cases, outputs, scan, count = 0, j;
    reg [4095:0] case_path, output_path;

    ace3_fp16_o_down_sum inner_add (
        .attention_o_i(attention_o), .mlp_down_i(down), .sum_o(inner),
        .invalid_o(inner_invalid), .saturation_o(inner_saturation)
    );
    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) final_add (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start), .start_ready_o(start_ready), .element_count_i(13'd1),
        .in_valid_i(valid), .in_ready_o(in_ready),
        .projection_f16_i((inner_invalid || inner_saturation) ? 16'h7e00 : inner),
        .residual_f16_i(hidden), .out_valid_o(out_valid), .out_ready_i(ready),
        .out_f16_o(result), .out_index_o(index), .out_last_o(last),
        .invalid_operand_o(invalid), .saturation_o(saturation), .busy_o(busy)
    );

    initial begin
        if (!$value$plusargs("CASES=%s", case_path) ||
            !$value$plusargs("OUTPUT=%s", output_path)) $fatal(1, "missing paths");
        cases = $fopen(case_path, "r");
        outputs = $fopen(output_path, "w");
        if (!cases || !outputs) $fatal(1, "cannot open operands/output");
        repeat (2) @(negedge clk);
        if (out_valid !== 0 || busy !== 0) $fatal(1, "reset");
        rst_n = 1;
        while (!$feof(cases)) begin
            scan = $fscanf(cases, "%h %h %h %h %h\n",
                          hidden, attention_o, down, expected_inner, expected_final);
            if (scan != 5) $fatal(1, "operand record");
            #1;
            if (inner !== expected_inner || inner_invalid !== 0 ||
                inner_saturation !== 0) $fatal(1, "inner case %0d", count);
            if (start_ready !== 1) $fatal(1, "start not ready");
            start = 1;
            @(negedge clk); start = 0;
            if (in_ready !== 1) $fatal(1, "input not ready");
            valid = 1;
            @(negedge clk); valid = 0;
            for (j = 0; j < 3; j = j + 1) begin
                if (out_valid !== 1 || result !== expected_final || index !== 0 ||
                    last !== 1 || invalid !== 0 || saturation !== 0 || busy !== 1)
                    $fatal(1, "result/backpressure case %0d", count);
                @(negedge clk);
            end
            $fdisplay(outputs, "%04h %04h", inner, result);
            ready = 1;
            @(negedge clk); ready = 0;
            if (out_valid !== 0 || busy !== 0) $fatal(1, "completion");
            count = count + 1;
        end
        attention_o = 16'h7c00; down = 0; #1;
        if (inner_invalid !== 1) $fatal(1, "nonfinite not reported");
        attention_o = 16'h7bff; down = 16'h7bff; #1;
        if (inner_saturation !== 1) $fatal(1, "overflow not reported");
        attention_o = 16'hxxxx; down = 0; #1;
        if (inner_invalid === 0) $fatal(1, "unknown input hidden");
        clear = 1;
        @(negedge clk);
        if (busy !== 0 || out_valid !== 0 || start_ready !== 0)
            $fatal(1, "clear");
        $fclose(cases); $fclose(outputs);
        $display("O_DOWN_THEN_HIDDEN_PASS cases=%0d", count);
        $finish;
    end
    initial begin
        #100000000;
        $fatal(1, "timeout");
    end
endmodule
`default_nettype wire
