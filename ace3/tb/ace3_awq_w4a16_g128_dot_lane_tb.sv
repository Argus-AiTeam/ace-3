`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_g128_dot_lane_tb;
    `include "vector_params.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg [2:0] logical_lane;
    reg [31:0] qzeros;
    reg [15:0] scale_f16;
    reg pair_valid;
    wire pair_ready;
    reg [15:0] activation_f16;
    reg [31:0] qweight;
    wire out_valid;
    reg out_ready;
    wire [15:0] out_f16;
    wire signed [95:0] accumulator;
    wire invalid_operand;
    wire saturation;

    reg [164:0] meta_mem [0:VECTOR_CASES-1];
    reg [47:0] pair_mem [0:VECTOR_PAIRS-1];
    reg [15:0] held_out;
    reg signed [95:0] held_acc;
    reg held_invalid;
    reg held_saturation;
    integer failures;
    integer case_index;
    integer pair_index;
    integer stall_index;

    ace3_awq_w4a16_g128_dot_lane dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .logical_lane_i(logical_lane),
        .qzeros_i(qzeros),
        .scale_f16_i(scale_f16),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .activation_f16_i(activation_f16),
        .qweight_i(qweight),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_f16_o(out_f16),
        .acc_q47_48_o(accumulator),
        .invalid_operand_o(invalid_operand),
        .saturation_o(saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task drive_idle;
        begin
            clear = 1'b0;
            start_valid = 1'b0;
            logical_lane = 3'd0;
            qzeros = 32'd0;
            scale_f16 = 16'd0;
            pair_valid = 1'b0;
            activation_f16 = 16'd0;
            qweight = 32'd0;
            out_ready = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            drive_idle();
            rst_n = 1'b0;
            #2;
            if (out_valid || pair_ready || (out_f16 !== 16'd0) ||
                invalid_operand || saturation) begin
                $display("RESET_ASYNC_FAIL");
                failures = failures + 1;
            end
            repeat (3) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            @(posedge clk);
            if (!start_ready) begin
                $display("RESET_RELEASE_FAIL");
                failures = failures + 1;
            end
        end
    endtask

    task start_case;
        input integer selected_case;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            qzeros = meta_mem[selected_case][31:0];
            scale_f16 = meta_mem[selected_case][47:32];
            logical_lane = meta_mem[selected_case][50:48];
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task test_abort_paths;
        begin
            start_case(0);
            pair_index = 0;
            repeat (3) begin
                @(negedge clk);
                qweight = pair_mem[pair_index][31:0];
                activation_f16 = pair_mem[pair_index][47:32];
                pair_valid = 1'b1;
                @(posedge clk);
                pair_index = pair_index + 1;
            end
            @(negedge clk);
            pair_valid = 1'b0;
            rst_n = 1'b0;
            #2;
            if (out_valid || pair_ready || invalid_operand || saturation) begin
                $display("RESET_MID_ACTIVITY_FAIL");
                failures = failures + 1;
            end
            @(negedge clk);
            rst_n = 1'b1;
            @(posedge clk);
            if (!start_ready) begin
                $display("RESET_MID_ACTIVITY_RELEASE_FAIL");
                failures = failures + 1;
            end

            start_case(0);
            @(negedge clk);
            qweight = pair_mem[0][31:0];
            activation_f16 = pair_mem[0][47:32];
            pair_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            pair_valid = 1'b0;
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            if (!start_ready || out_valid || pair_ready ||
                (accumulator !== 96'sd0)) begin
                $display("CLEAR_MID_ACTIVITY_FAIL");
                failures = failures + 1;
            end
        end
    endtask

    task run_case;
        input integer selected_case;
        integer selected_pair;
        begin
            start_case(selected_case);
            for (selected_pair = 0; selected_pair < 128;
                 selected_pair = selected_pair + 1) begin
                if ((selected_pair % 11) == 3) begin
                    pair_valid = 1'b0;
                    repeat (2) @(posedge clk);
                    if (!pair_ready) begin
                        $display("INPUT_STALL_READY_FAIL case=%0d pair=%0d",
                                 selected_case, selected_pair);
                        failures = failures + 1;
                    end
                end
                @(negedge clk);
                qweight = pair_mem[selected_case*128 + selected_pair][31:0];
                activation_f16 =
                    pair_mem[selected_case*128 + selected_pair][47:32];
                pair_valid = 1'b1;
                @(posedge clk);
                if (!pair_ready) begin
                    $display("PAIR_HANDSHAKE_FAIL case=%0d pair=%0d",
                             selected_case, selected_pair);
                    failures = failures + 1;
                end
                @(negedge clk);
                pair_valid = 1'b0;
            end

            if (!out_valid) begin
                @(posedge clk);
                if (!out_valid) begin
                    $display("OUTPUT_MISSING case=%0d", selected_case);
                    failures = failures + 1;
                end
            end
            held_out = out_f16;
            held_acc = accumulator;
            held_invalid = invalid_operand;
            held_saturation = saturation;
            for (stall_index = 0; stall_index < 4;
                 stall_index = stall_index + 1) begin
                @(posedge clk);
                if (!out_valid || start_ready || (out_f16 !== held_out) ||
                    (accumulator !== held_acc) ||
                    (invalid_operand !== held_invalid) ||
                    (saturation !== held_saturation)) begin
                    $display("OUTPUT_BACKPRESSURE_STABILITY_FAIL case=%0d",
                             selected_case);
                    failures = failures + 1;
                end
            end
            if ((out_f16 !== meta_mem[selected_case][162:147]) ||
                (accumulator !== meta_mem[selected_case][146:51]) ||
                (invalid_operand !== meta_mem[selected_case][163]) ||
                (saturation !== meta_mem[selected_case][164])) begin
                $display(
                    "NUMERIC_MISMATCH case=%0d got_fp=%04x exp_fp=%04x got_acc=%024x exp_acc=%024x got_invalid=%0d exp_invalid=%0d got_sat=%0d exp_sat=%0d",
                    selected_case,
                    out_f16,
                    meta_mem[selected_case][162:147],
                    accumulator,
                    meta_mem[selected_case][146:51],
                    invalid_operand,
                    meta_mem[selected_case][163],
                    saturation,
                    meta_mem[selected_case][164]
                );
                failures = failures + 1;
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    initial begin
        failures = 0;
        rst_n = 1'b1;
        drive_idle();
        $readmemh("generated/meta.hex", meta_mem);
        $readmemh("generated/pairs.hex", pair_mem);
        apply_reset();
        test_abort_paths();
        for (case_index = 0; case_index < VECTOR_CASES;
             case_index = case_index + 1)
            run_case(case_index);
        if (failures == 0) begin
            $display(
                "AWQ_W4A16_G128_PASS cases=%0d pairs=%0d ulp_bound=0 reset=pass clear=pass backpressure=pass",
                VECTOR_CASES,
                VECTOR_PAIRS
            );
            $finish;
        end
        $fatal(1, "AWQ_W4A16_G128_FAIL failures=%0d", failures);
    end
endmodule

`default_nettype wire
