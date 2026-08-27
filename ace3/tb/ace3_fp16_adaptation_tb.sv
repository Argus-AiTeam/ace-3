`timescale 1ns/1ps
`default_nettype none

module ace3_fp16_adaptation_tb;
    `include "fp16_adaptation_params.svh"

    reg clk = 1'b0;
    reg rst_ni = 1'b0;
    reg clear_i = 1'b0;
    always #5 clk = ~clk;

    reg rr_start_valid;
    wire rr_start_ready;
    reg [12:0] rr_count;
    reg rr_in_valid;
    wire rr_in_ready;
    reg [15:0] rr_projection;
    reg [15:0] rr_residual;
    wire rr_out_valid;
    reg rr_out_ready;
    wire [15:0] rr_out;
    wire [12:0] rr_out_index;
    wire rr_out_last;
    wire rr_invalid;
    wire rr_saturation;
    wire rr_busy;

    reg sg_start_valid;
    wire sg_start_ready;
    reg [12:0] sg_count;
    reg sg_in_valid;
    wire sg_in_ready;
    reg [15:0] sg_gate;
    reg [15:0] sg_up;
    wire sg_out_valid;
    reg sg_out_ready;
    wire [15:0] sg_out;
    wire [12:0] sg_out_index;
    wire sg_out_last;
    wire sg_invalid;
    wire sg_saturation;
    wire sg_busy;

    reg rn_start_valid;
    wire rn_start_ready;
    reg [12:0] rn_count;
    reg rn_in_valid;
    wire rn_in_ready;
    reg [15:0] rn_activation;
    reg [15:0] rn_weight;
    wire rn_out_valid;
    reg rn_out_ready;
    wire [15:0] rn_out;
    wire [12:0] rn_out_index;
    wire rn_out_last;
    wire rn_invalid;
    wire rn_saturation;
    wire rn_busy;
    wire [45:0] rn_rms_q24;

    reg [49:0] residual_cases [0:FP16_RESIDUAL_CASES-1];
    reg [49:0] silu_cases [0:FP16_SILU_CASES-1];
    reg [31:0] rms_inputs [0:FP16_RMS_CASES*FP16_RMS_SIZE-1];
    reg [17:0] rms_expected [0:FP16_RMS_CASES*FP16_RMS_SIZE-1];
    reg [46:0] rms_meta [0:FP16_RMS_CASES-1];
    reg [1023:0] vector_dir;
    reg [1023:0] vector_path;

    integer case_index;
    integer element_index;
    integer stream_index;
    integer sqrt_cycles;
    integer accepted_inputs;
    integer checked_outputs;
    integer backpressure_checks;
    integer reset_checks;
    integer clear_checks;
    integer invalid_start_checks;
    integer xz_checks;
    integer invalid_outputs;
    integer saturation_outputs;
    reg [15:0] held_output;
    reg [12:0] held_index;
    reg held_last;
    reg held_invalid;
    reg held_saturation;

    ace3_fp16_residual_add_core #(
        .VECTOR_SIZE(896)
    ) residual_dut (
        .clk_i(clk),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(rr_start_valid),
        .start_ready_o(rr_start_ready),
        .element_count_i(rr_count),
        .in_valid_i(rr_in_valid),
        .in_ready_o(rr_in_ready),
        .projection_f16_i(rr_projection),
        .residual_f16_i(rr_residual),
        .out_valid_o(rr_out_valid),
        .out_ready_i(rr_out_ready),
        .out_f16_o(rr_out),
        .out_index_o(rr_out_index),
        .out_last_o(rr_out_last),
        .invalid_operand_o(rr_invalid),
        .saturation_o(rr_saturation),
        .busy_o(rr_busy)
    );

    ace3_fp16_silu_gate_core #(
        .INTERMEDIATE_SIZE(4864)
    ) silu_dut (
        .clk_i(clk),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(sg_start_valid),
        .start_ready_o(sg_start_ready),
        .element_count_i(sg_count),
        .in_valid_i(sg_in_valid),
        .in_ready_o(sg_in_ready),
        .gate_f16_i(sg_gate),
        .up_f16_i(sg_up),
        .out_valid_o(sg_out_valid),
        .out_ready_i(sg_out_ready),
        .out_f16_o(sg_out),
        .out_index_o(sg_out_index),
        .out_last_o(sg_out_last),
        .invalid_operand_o(sg_invalid),
        .saturation_o(sg_saturation),
        .busy_o(sg_busy)
    );

    ace3_fp16_rmsnorm_core #(
        .HIDDEN_SIZE(FP16_RMS_SIZE)
    ) rmsnorm_dut (
        .clk_i(clk),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(rn_start_valid),
        .start_ready_o(rn_start_ready),
        .element_count_i(rn_count),
        .in_valid_i(rn_in_valid),
        .in_ready_o(rn_in_ready),
        .activation_f16_i(rn_activation),
        .weight_f16_i(rn_weight),
        .out_valid_o(rn_out_valid),
        .out_ready_i(rn_out_ready),
        .out_f16_o(rn_out),
        .out_index_o(rn_out_index),
        .out_last_o(rn_out_last),
        .invalid_operand_o(rn_invalid),
        .saturation_o(rn_saturation),
        .busy_o(rn_busy),
        .rms_q24_o(rn_rms_q24)
    );

    task automatic drive_idle;
        begin
            rr_start_valid = 1'b0;
            rr_count = 13'd1;
            rr_in_valid = 1'b0;
            rr_projection = 16'd0;
            rr_residual = 16'd0;
            rr_out_ready = 1'b0;
            sg_start_valid = 1'b0;
            sg_count = 13'd1;
            sg_in_valid = 1'b0;
            sg_gate = 16'd0;
            sg_up = 16'd0;
            sg_out_ready = 1'b0;
            rn_start_valid = 1'b0;
            rn_count = FP16_RMS_SIZE;
            rn_in_valid = 1'b0;
            rn_activation = 16'd0;
            rn_weight = 16'd0;
            rn_out_ready = 1'b0;
        end
    endtask

    task automatic pulse_start_residual;
        begin
            @(negedge clk);
            rr_count = 13'd1;
            rr_start_valid = 1'b1;
            #1;
            if (!rr_start_ready)
                $fatal(1, "residual legal start was not ready");
            @(posedge clk);
            #1;
            @(negedge clk);
            rr_start_valid = 1'b0;
        end
    endtask

    task automatic pulse_start_silu;
        begin
            @(negedge clk);
            sg_count = 13'd1;
            sg_start_valid = 1'b1;
            #1;
            if (!sg_start_ready)
                $fatal(1, "SiLU legal start was not ready");
            @(posedge clk);
            #1;
            @(negedge clk);
            sg_start_valid = 1'b0;
        end
    endtask

    task automatic pulse_start_rms;
        begin
            @(negedge clk);
            rn_count = FP16_RMS_SIZE;
            rn_start_valid = 1'b1;
            #1;
            if (!rn_start_ready)
                $fatal(1, "RMSNorm legal start was not ready");
            @(posedge clk);
            #1;
            @(negedge clk);
            rn_start_valid = 1'b0;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "VECTOR_DIR plusarg is required");
        vector_path = {vector_dir, "/residual_cases.hex"};
        $readmemh(vector_path, residual_cases);
        vector_path = {vector_dir, "/silu_cases.hex"};
        $readmemh(vector_path, silu_cases);
        vector_path = {vector_dir, "/rms_inputs.hex"};
        $readmemh(vector_path, rms_inputs);
        vector_path = {vector_dir, "/rms_expected.hex"};
        $readmemh(vector_path, rms_expected);
        vector_path = {vector_dir, "/rms_meta.hex"};
        $readmemh(vector_path, rms_meta);
        $dumpfile("build/iverilog/ace3_fp16_adaptation.vcd");
        $dumpvars(0, ace3_fp16_adaptation_tb);

        accepted_inputs = 0;
        checked_outputs = 0;
        backpressure_checks = 0;
        reset_checks = 0;
        clear_checks = 0;
        invalid_start_checks = 0;
        xz_checks = 0;
        invalid_outputs = 0;
        saturation_outputs = 0;
        drive_idle();
        repeat (3) @(posedge clk);
        rst_ni = 1'b1;
        @(posedge clk);
        #1;

        @(negedge clk);
        rr_count = 13'd0;
        sg_count = 13'd4865;
        rn_count = FP16_RMS_SIZE - 1;
        rr_start_valid = 1'b1;
        sg_start_valid = 1'b1;
        rn_start_valid = 1'b1;
        #1;
        if (rr_start_ready || sg_start_ready || rn_start_ready)
            $fatal(1, "invalid start configuration was accepted");
        invalid_start_checks = 3;
        @(posedge clk);
        #1;
        drive_idle();

        @(negedge clk);
        rr_projection = 16'hxxxx;
        rr_residual = 16'hzzzz;
        sg_gate = 16'hxxxx;
        sg_up = 16'hzzzz;
        rn_activation = 16'hxxxx;
        rn_weight = 16'hzzzz;
        @(posedge clk);
        #1;
        if ({rr_start_ready, rr_in_ready, rr_out_valid, rr_busy,
             sg_start_ready, sg_in_ready, sg_out_valid, sg_busy,
             rn_start_ready, rn_in_ready, rn_out_valid, rn_busy}
            !== 12'b1000_1000_1000) begin
            $display(
                "XZ_CONTROL_STATE rr=%b%b%b%b sg=%b%b%b%b rn=%b%b%b%b",
                rr_start_ready, rr_in_ready, rr_out_valid, rr_busy,
                sg_start_ready, sg_in_ready, sg_out_valid, sg_busy,
                rn_start_ready, rn_in_ready, rn_out_valid, rn_busy
            );
            $fatal(1, "inactive X/Z data contaminated protocol controls");
        end
        if (rr_out_valid || sg_out_valid || rn_out_valid)
            $fatal(1, "inactive X/Z data created an output");
        xz_checks = 6;
        drive_idle();

        @(negedge clk);
        rr_count = 13'd2;
        rr_start_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        rr_start_valid = 1'b0;
        rr_projection = 16'h3c00;
        rr_residual = 16'h3c00;
        rr_in_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        rr_in_valid = 1'b0;
        clear_i = 1'b1;
        @(posedge clk);
        #1;
        if (rr_out_valid || rr_busy)
            $fatal(1, "clear did not abort residual transaction");
        clear_checks = clear_checks + 1;
        @(negedge clk);
        clear_i = 1'b0;

        @(negedge clk);
        sg_count = 13'd2;
        sg_start_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        sg_start_valid = 1'b0;
        sg_gate = 16'h3c00;
        sg_up = 16'h3c00;
        sg_in_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        sg_in_valid = 1'b0;
        rst_ni = 1'b0;
        #1;
        if (sg_out_valid || sg_busy || sg_start_ready)
            $fatal(1, "asynchronous reset did not abort SiLU transaction");
        reset_checks = reset_checks + 1;
        @(posedge clk);
        @(negedge clk);
        rst_ni = 1'b1;
        drive_idle();
        @(posedge clk);
        #1;

        pulse_start_rms();
        for (element_index = 0; element_index < 2;
             element_index = element_index + 1) begin
            rn_activation = 16'h3c00;
            rn_weight = 16'h3c00;
            rn_in_valid = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
        end
        rn_in_valid = 1'b0;
        clear_i = 1'b1;
        @(posedge clk);
        #1;
        if (rn_out_valid || rn_busy)
            $fatal(1, "clear did not abort RMSNorm transaction");
        clear_checks = clear_checks + 1;
        @(negedge clk);
        clear_i = 1'b0;

        for (case_index = 0; case_index < FP16_RESIDUAL_CASES;
             case_index = case_index + 1) begin
            pulse_start_residual();
            rr_projection = residual_cases[case_index][15:0];
            rr_residual = residual_cases[case_index][31:16];
            rr_in_valid = 1'b1;
            rr_out_ready = 1'b0;
            if (!rr_in_ready)
                $fatal(1, "residual input not ready");
            @(posedge clk);
            #1;
            rr_in_valid = 1'b0;
            accepted_inputs = accepted_inputs + 1;
            if (!rr_out_valid || rr_out_index !== 13'd0 || !rr_out_last ||
                rr_out !== residual_cases[case_index][47:32] ||
                rr_invalid !== residual_cases[case_index][48] ||
                rr_saturation !== residual_cases[case_index][49])
                $fatal(1, "residual mismatch case=%0d", case_index);
            invalid_outputs = invalid_outputs + rr_invalid;
            saturation_outputs = saturation_outputs + rr_saturation;
            held_output = rr_out;
            if (case_index == 0) begin
                @(posedge clk);
                #1;
                if (!rr_out_valid || rr_out !== held_output)
                    $fatal(1, "residual output changed under backpressure");
                backpressure_checks = backpressure_checks + 1;
            end
            @(negedge clk);
            rr_out_ready = 1'b1;
            @(posedge clk);
            #1;
            checked_outputs = checked_outputs + 1;
            @(negedge clk);
            rr_out_ready = 1'b0;
        end

        for (case_index = 0; case_index < FP16_SILU_CASES;
             case_index = case_index + 1) begin
            pulse_start_silu();
            sg_gate = silu_cases[case_index][15:0];
            sg_up = silu_cases[case_index][31:16];
            sg_in_valid = 1'b1;
            sg_out_ready = 1'b0;
            if (!sg_in_ready)
                $fatal(1, "SiLU input not ready");
            @(posedge clk);
            #1;
            sg_in_valid = 1'b0;
            accepted_inputs = accepted_inputs + 1;
            if (!sg_out_valid || sg_out_index !== 13'd0 || !sg_out_last ||
                sg_out !== silu_cases[case_index][47:32] ||
                sg_invalid !== silu_cases[case_index][48] ||
                sg_saturation !== silu_cases[case_index][49])
                $fatal(1, "SiLU mismatch case=%0d", case_index);
            invalid_outputs = invalid_outputs + sg_invalid;
            saturation_outputs = saturation_outputs + sg_saturation;
            held_output = sg_out;
            if (case_index == 0) begin
                @(posedge clk);
                #1;
                if (!sg_out_valid || sg_out !== held_output)
                    $fatal(1, "SiLU output changed under backpressure");
                backpressure_checks = backpressure_checks + 1;
            end
            @(negedge clk);
            sg_out_ready = 1'b1;
            @(posedge clk);
            #1;
            checked_outputs = checked_outputs + 1;
            @(negedge clk);
            sg_out_ready = 1'b0;
        end

        stream_index = 0;
        for (case_index = 0; case_index < FP16_RMS_CASES;
             case_index = case_index + 1) begin
            pulse_start_rms();
            for (element_index = 0; element_index < FP16_RMS_SIZE;
                 element_index = element_index + 1) begin
                rn_activation = rms_inputs[stream_index][15:0];
                rn_weight = rms_inputs[stream_index][31:16];
                rn_in_valid = 1'b1;
                if (!rn_in_ready)
                    $fatal(1, "RMSNorm input not ready");
                @(posedge clk);
                #1;
                accepted_inputs = accepted_inputs + 1;
                stream_index = stream_index + 1;
                @(negedge clk);
            end
            rn_in_valid = 1'b0;
            sqrt_cycles = 0;
            while (!rn_out_valid) begin
                @(posedge clk);
                #1;
                sqrt_cycles = sqrt_cycles + 1;
                if (sqrt_cycles > 48)
                    $fatal(1, "RMSNorm sqrt timeout case=%0d", case_index);
            end
            if (sqrt_cycles != 48)
                $fatal(1, "RMSNorm sqrt cycles=%0d expected=48", sqrt_cycles);
            if (rn_rms_q24 !== rms_meta[case_index][45:0])
                $fatal(1, "RMSNorm root mismatch case=%0d", case_index);

            for (element_index = 0; element_index < FP16_RMS_SIZE;
                 element_index = element_index + 1) begin
                if (!rn_out_valid ||
                    rn_out !== rms_expected[
                        case_index*FP16_RMS_SIZE + element_index][15:0] ||
                    rn_invalid !== rms_expected[
                        case_index*FP16_RMS_SIZE + element_index][16] ||
                    rn_saturation !== rms_expected[
                        case_index*FP16_RMS_SIZE + element_index][17] ||
                    rn_out_index !== element_index ||
                    rn_out_last !== (element_index == FP16_RMS_SIZE-1))
                    $fatal(1, "RMSNorm mismatch case=%0d element=%0d",
                           case_index, element_index);
                invalid_outputs = invalid_outputs + rn_invalid;
                saturation_outputs = saturation_outputs + rn_saturation;
                if ((case_index == 0) && (element_index == 0)) begin
                    held_output = rn_out;
                    held_index = rn_out_index;
                    held_last = rn_out_last;
                    held_invalid = rn_invalid;
                    held_saturation = rn_saturation;
                    @(posedge clk);
                    #1;
                    if (!rn_out_valid || rn_out !== held_output ||
                        rn_out_index !== held_index ||
                        rn_out_last !== held_last ||
                        rn_invalid !== held_invalid ||
                        rn_saturation !== held_saturation)
                        $fatal(1, "RMSNorm output changed under backpressure");
                    backpressure_checks = backpressure_checks + 1;
                end
                @(negedge clk);
                rn_out_ready = 1'b1;
                @(posedge clk);
                #1;
                checked_outputs = checked_outputs + 1;
                @(negedge clk);
                rn_out_ready = 1'b0;
            end
        end

        if (accepted_inputs == 0 || checked_outputs == 0 ||
            backpressure_checks != 3 || reset_checks != 1 ||
            clear_checks != 2 || invalid_start_checks != 3 ||
            xz_checks != 6 || invalid_outputs == 0 ||
            saturation_outputs == 0)
            $fatal(1, "non-vacuous coverage counters failed");
        $display(
            "ACE3_FP16_ADAPTATION_IVERILOG_PASS accepted_inputs=%0d outputs=%0d backpressure=%0d reset=%0d clear=%0d invalid_starts=%0d xz=%0d invalid_outputs=%0d saturation_outputs=%0d residual_latency=1 silu_latency=1 rms_sqrt_cycles=48",
            accepted_inputs, checked_outputs, backpressure_checks,
            reset_checks, clear_checks, invalid_start_checks, xz_checks,
            invalid_outputs, saturation_outputs
        );
        $finish;
    end
endmodule

`default_nettype wire
