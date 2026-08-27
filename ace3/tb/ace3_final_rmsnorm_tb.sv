`timescale 1ns/1ps
`default_nettype none

module ace3_final_rmsnorm_tb;
    localparam integer HIDDEN_SIZE = 896;
    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;
    reg clear_i = 1'b0;
    reg start_valid_i = 1'b0;
    wire start_ready_o;
    reg in_valid_i = 1'b0;
    wire in_ready_o;
    reg [15:0] activation_f16_i = 16'd0;
    reg [15:0] weight_f16_i = 16'd0;
    wire out_valid_o;
    reg out_ready_i = 1'b1;
    wire [15:0] out_f16_o;
    wire [12:0] out_index_o;
    wire out_last_o;
    wire invalid_operand_o;
    wire saturation_o;
    wire busy_o;
    wire [45:0] rms_q24_o;

    reg [15:0] activations [0:4*HIDDEN_SIZE-1];
    reg [15:0] weights [0:HIDDEN_SIZE-1];
    string activations_path;
    string weights_path;
    string raw_path;
    string terminal_path;
    integer case_index;
    integer fail_after;
    integer raw_fd;
    integer terminal_fd;
    integer input_index;
    integer output_count;
    integer cycle_count;
    integer plusarg_status;

    always #5 clk_i = ~clk_i;

    ace3_final_rmsnorm dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(start_valid_i),
        .start_ready_o(start_ready_o),
        .in_valid_i(in_valid_i),
        .in_ready_o(in_ready_o),
        .activation_f16_i(activation_f16_i),
        .weight_f16_i(weight_f16_i),
        .out_valid_o(out_valid_o),
        .out_ready_i(out_ready_i),
        .out_f16_o(out_f16_o),
        .out_index_o(out_index_o),
        .out_last_o(out_last_o),
        .invalid_operand_o(invalid_operand_o),
        .saturation_o(saturation_o),
        .busy_o(busy_o),
        .rms_q24_o(rms_q24_o)
    );

    task automatic write_terminal;
        input integer natural_terminal;
        input integer recorded_exit_code;
        begin
            terminal_fd = $fopen(terminal_path, "w");
            if (terminal_fd == 0)
                $fatal(1, "cannot open terminal output");
            $fdisplay(terminal_fd, "schema=ace3-final-rmsnorm-terminal-v1");
            $fdisplay(terminal_fd, "natural_terminal=%0d", natural_terminal);
            $fdisplay(terminal_fd, "recorded_exit_code=%0d", recorded_exit_code);
            $fdisplay(terminal_fd, "output_count=%0d", output_count);
            $fdisplay(terminal_fd, "case_index=%0d", case_index);
            $fflush(terminal_fd);
            $fclose(terminal_fd);
        end
    endtask

    always @(posedge clk_i) begin
        if (rst_ni) begin
            cycle_count = cycle_count + 1;
            if (cycle_count > 20000) begin
                $display("timeout state=%0d input_index=%0d output_index=%0d start_ready=%b in_ready=%b out_valid=%b busy=%b",
                         dut.final_norm_core.state_q,
                         dut.final_norm_core.input_index_q,
                         dut.final_norm_core.output_index_q,
                         start_ready_o, in_ready_o, out_valid_o, busy_o);
                write_terminal(0, 1);
                $fatal(1, "timeout");
            end
            if (out_valid_o && out_ready_i) begin
                if ((^out_f16_o) === 1'bx || (^out_index_o) === 1'bx ||
                    out_last_o === 1'bx || invalid_operand_o === 1'bx ||
                    saturation_o === 1'bx)
                    $fatal(1, "four-state unknown on output beat");
                if (out_index_o !== output_count[12:0])
                    $fatal(1, "output index mismatch");
                $fdisplay(raw_fd, "%0d %04h", out_index_o, out_f16_o);
                $fflush(raw_fd);
                output_count = output_count + 1;
                if (fail_after > 0 && output_count == fail_after) begin
                    write_terminal(0, 1);
                    $fclose(raw_fd);
                    $fatal(1, "injected simulator failure");
                end
                if (out_last_o) begin
                    if (output_count != HIDDEN_SIZE)
                        $fatal(1, "wrong output count");
                    write_terminal(1, 0);
                    $fclose(raw_fd);
                    $finish;
                end
            end
        end
    end

    initial begin
        case_index = 0;
        fail_after = 0;
        output_count = 0;
        cycle_count = 0;
        if (!$value$plusargs("ACTIVATIONS=%s", activations_path) ||
            !$value$plusargs("WEIGHTS=%s", weights_path) ||
            !$value$plusargs("RAW=%s", raw_path) ||
            !$value$plusargs("TERMINAL=%s", terminal_path))
            $fatal(1, "missing required path plusarg");
        plusarg_status = $value$plusargs("CASE=%d", case_index);
        plusarg_status = $value$plusargs("FAIL_AFTER=%d", fail_after);
        if (case_index < 0 || case_index > 3)
            $fatal(1, "case index out of range");
        $readmemh(activations_path, activations);
        $readmemh(weights_path, weights);
        raw_fd = $fopen(raw_path, "w");
        if (raw_fd == 0)
            $fatal(1, "cannot open raw output");

        repeat (4) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;
        start_valid_i = 1'b1;
        #1;
        if (start_ready_o !== 1'b1)
            $fatal(1, "start was not ready after reset release");
        @(posedge clk_i);
        @(negedge clk_i);
        start_valid_i = 1'b0;

        for (input_index = 0; input_index < HIDDEN_SIZE; input_index = input_index + 1) begin
            activation_f16_i = activations[case_index * HIDDEN_SIZE + input_index];
            weight_f16_i = weights[input_index];
            in_valid_i = 1'b1;
            while (!in_ready_o) @(negedge clk_i);
            @(posedge clk_i);
            @(negedge clk_i);
        end
        in_valid_i = 1'b0;
    end
endmodule

`default_nettype wire
