`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_g128_dot_lane_protocol_tb;
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

    integer failures;
    integer property_failures;
    integer four_state_control_probes;
    integer four_state_data_probes;
    integer completed_starts;
    integer completed_pairs;
    integer completed_outputs;
    integer pair_number;
    integer monitor_pair_count;
    reg monitor_active;
    reg previous_backpressured;
    reg [15:0] previous_out_f16;
    reg signed [95:0] previous_accumulator;
    reg previous_invalid;
    reg previous_saturation;
    reg [15:0] held_out_f16;
    reg signed [95:0] held_accumulator;
    reg held_invalid;
    reg held_saturation;

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
            scale_f16 = 16'h3c00;
            pair_valid = 1'b0;
            activation_f16 = 16'd0;
            qweight = 32'd0;
            out_ready = 1'b0;
        end
    endtask

    task check_accepting_idle;
        begin
            #1;
            if ((start_ready !== 1'b1) || (pair_ready !== 1'b0) ||
                (out_valid !== 1'b0)) begin
                $display(
                    "PROTOCOL_ACCEPTING_IDLE_FAIL time=%0t start_ready=%b pair_ready=%b out_valid=%b",
                    $time,
                    start_ready,
                    pair_ready,
                    out_valid
                );
                failures = failures + 1;
            end
        end
    endtask

    task check_idle_state;
        begin
            #1;
            if ((start_ready !== 1'b1) || (pair_ready !== 1'b0) ||
                (out_valid !== 1'b0) || (out_f16 !== 16'd0) ||
                (accumulator !== 96'sd0) ||
                (invalid_operand !== 1'b0) ||
                (saturation !== 1'b0)) begin
                $display(
                    "PROTOCOL_IDLE_STATE_FAIL time=%0t start_ready=%b pair_ready=%b out_valid=%b out_f16=%h accumulator=%h invalid=%b saturation=%b",
                    $time,
                    start_ready,
                    pair_ready,
                    out_valid,
                    out_f16,
                    accumulator,
                    invalid_operand,
                    saturation
                );
                failures = failures + 1;
            end
        end
    endtask

    task accept_start;
        begin
            @(negedge clk);
            logical_lane = 3'd0;
            qzeros = 32'd0;
            scale_f16 = 16'h3c00;
            start_valid = 1'b1;
            if (start_ready !== 1'b1) begin
                $display("PROTOCOL_START_READY_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            completed_starts = completed_starts + 1;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task accept_one_pair;
        begin
            @(negedge clk);
            activation_f16 = 16'h3c00;
            qweight = 32'h00000001;
            pair_valid = 1'b1;
            if (pair_ready !== 1'b1) begin
                $display("PROTOCOL_PAIR_READY_FAIL pair=%0d", pair_number);
                failures = failures + 1;
            end
            @(posedge clk);
            completed_pairs = completed_pairs + 1;
            @(negedge clk);
            pair_valid = 1'b0;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            monitor_active <= 1'b0;
            monitor_pair_count <= 0;
            previous_backpressured <= 1'b0;
            previous_out_f16 <= 16'd0;
            previous_accumulator <= 96'sd0;
            previous_invalid <= 1'b0;
            previous_saturation <= 1'b0;
        end else begin
            if ((clear === 1'b0) &&
                (start_valid === 1'b1) &&
                (start_ready === 1'b1)) begin
                if (monitor_active) begin
                    $display("PROPERTY_OVERLAPPING_START_FAIL");
                    property_failures = property_failures + 1;
                end
                if ((^logical_lane === 1'bx) ||
                    (^qzeros === 1'bx) ||
                    (^scale_f16 === 1'bx)) begin
                    $display("PROPERTY_UNKNOWN_ACCEPTED_START_DATA_FAIL");
                    property_failures = property_failures + 1;
                end
                monitor_active <= 1'b1;
                monitor_pair_count <= 0;
            end
            if ((clear === 1'b0) &&
                (pair_valid === 1'b1) &&
                (pair_ready === 1'b1)) begin
                if (!monitor_active || (monitor_pair_count >= 128)) begin
                    $display("PROPERTY_ILLEGAL_PAIR_COUNT_FAIL count=%0d",
                             monitor_pair_count);
                    property_failures = property_failures + 1;
                end
                if ((^activation_f16 === 1'bx) ||
                    (^qweight === 1'bx)) begin
                    $display("PROPERTY_UNKNOWN_ACCEPTED_PAIR_DATA_FAIL");
                    property_failures = property_failures + 1;
                end
                monitor_pair_count <= monitor_pair_count + 1;
            end
            if (out_valid === 1'b1) begin
                if (!monitor_active || (monitor_pair_count != 128)) begin
                    $display("PROPERTY_OUTPUT_PAIR_COUNT_FAIL count=%0d",
                             monitor_pair_count);
                    property_failures = property_failures + 1;
                end
                if ((^out_f16 === 1'bx) ||
                    (^accumulator === 1'bx) ||
                    (invalid_operand !== 1'b0 &&
                     invalid_operand !== 1'b1) ||
                    (saturation !== 1'b0 && saturation !== 1'b1)) begin
                    $display("PROPERTY_UNKNOWN_VALID_OUTPUT_FAIL");
                    property_failures = property_failures + 1;
                end
            end else if ((out_valid !== 1'b0) &&
                         (out_valid !== 1'b1)) begin
                $display("PROPERTY_UNKNOWN_OUT_VALID_FAIL");
                property_failures = property_failures + 1;
            end
            if (previous_backpressured &&
                ((out_valid !== 1'b1) ||
                 (out_f16 !== previous_out_f16) ||
                 (accumulator !== previous_accumulator) ||
                 (invalid_operand !== previous_invalid) ||
                 (saturation !== previous_saturation))) begin
                $display("PROPERTY_BACKPRESSURE_STABILITY_FAIL");
                property_failures = property_failures + 1;
            end
            if ((clear === 1'b0) &&
                (out_valid === 1'b1) &&
                (out_ready === 1'b1))
                monitor_active <= 1'b0;
            if (clear === 1'b1) begin
                monitor_active <= 1'b0;
                monitor_pair_count <= 0;
            end
            previous_backpressured <=
                (out_valid === 1'b1) &&
                !((clear === 1'b0) && (out_ready === 1'b1));
            previous_out_f16 <= out_f16;
            previous_accumulator <= accumulator;
            previous_invalid <= invalid_operand;
            previous_saturation <= saturation;
        end
    end

    initial begin
        failures = 0;
        property_failures = 0;
        four_state_control_probes = 0;
        four_state_data_probes = 0;
        completed_starts = 0;
        completed_pairs = 0;
        completed_outputs = 0;
        pair_number = 0;
        monitor_active = 1'b0;
        monitor_pair_count = 0;
        previous_backpressured = 1'b0;
        rst_n = 1'b0;
        drive_idle();

        #1;
        if ((start_ready !== 1'b0) || (pair_ready !== 1'b0) ||
            (out_valid !== 1'b0) || (out_f16 !== 16'd0) ||
            (accumulator !== 96'sd0) ||
            (invalid_operand !== 1'b0) || (saturation !== 1'b0)) begin
            $display("PROTOCOL_ASYNC_RESET_FAIL");
            failures = failures + 1;
        end
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        #1;
        check_idle_state();

        @(negedge clk);
        logical_lane = 3'bxxx;
        qzeros = 32'hxxxxxxxx;
        scale_f16 = 16'hxxxx;
        start_valid = 1'bx;
        four_state_control_probes = four_state_control_probes + 1;
        four_state_data_probes = four_state_data_probes + 1;
        @(posedge clk);
        @(negedge clk);
        start_valid = 1'b0;
        logical_lane = 3'd0;
        qzeros = 32'd0;
        scale_f16 = 16'h3c00;
        check_idle_state();

        @(negedge clk);
        start_valid = 1'bz;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        start_valid = 1'b0;
        check_idle_state();

        @(negedge clk);
        clear = 1'bx;
        start_valid = 1'b1;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        start_valid = 1'b0;
        check_idle_state();

        @(negedge clk);
        clear = 1'bz;
        start_valid = 1'b1;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        start_valid = 1'b0;
        check_idle_state();

        accept_start();

        @(negedge clk);
        activation_f16 = 16'h3c00;
        qweight = 32'h00000001;
        pair_valid = 1'b1;
        clear = 1'bx;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        pair_valid = 1'b0;
        if (accumulator !== 96'sd0) begin
            $display("PROTOCOL_X_CLEAR_ACCEPTED_PAIR_FAIL");
            failures = failures + 1;
        end

        clear = 1'bz;
        pair_valid = 1'b1;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        pair_valid = 1'b0;
        if (accumulator !== 96'sd0) begin
            $display("PROTOCOL_Z_CLEAR_ACCEPTED_PAIR_FAIL");
            failures = failures + 1;
        end

        @(negedge clk);
        activation_f16 = 16'hxxxx;
        qweight = 32'hzzzzzzzz;
        pair_valid = 1'b0;
        four_state_data_probes = four_state_data_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if (accumulator !== 96'sd0) begin
            $display("PROTOCOL_INVALID_DATA_CHANGED_STATE_FAIL");
            failures = failures + 1;
        end

        pair_valid = 1'bx;
        four_state_control_probes = four_state_control_probes + 1;
        four_state_data_probes = four_state_data_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if (accumulator !== 96'sd0) begin
            $display("PROTOCOL_X_PAIR_VALID_ACCEPTED_FAIL");
            failures = failures + 1;
        end

        pair_valid = 1'bz;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        pair_valid = 1'b0;
        activation_f16 = 16'd0;
        qweight = 32'd0;
        if (accumulator !== 96'sd0) begin
            $display("PROTOCOL_Z_PAIR_VALID_ACCEPTED_FAIL");
            failures = failures + 1;
        end

        for (pair_number = 0; pair_number < 128;
             pair_number = pair_number + 1)
            accept_one_pair();
        if (out_valid !== 1'b1) begin
            $display("PROTOCOL_OUTPUT_MISSING");
            failures = failures + 1;
        end
        held_out_f16 = out_f16;
        held_accumulator = accumulator;
        held_invalid = invalid_operand;
        held_saturation = saturation;

        @(negedge clk);
        clear = 1'bx;
        out_ready = 1'b1;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if ((out_valid !== 1'b1) || (out_f16 !== held_out_f16) ||
            (accumulator !== held_accumulator) ||
            (invalid_operand !== held_invalid) ||
            (saturation !== held_saturation)) begin
            $display("PROTOCOL_X_CLEAR_ACCEPTED_OUTPUT_FAIL");
            failures = failures + 1;
        end

        clear = 1'bz;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if ((out_valid !== 1'b1) || (out_f16 !== held_out_f16) ||
            (accumulator !== held_accumulator) ||
            (invalid_operand !== held_invalid) ||
            (saturation !== held_saturation)) begin
            $display("PROTOCOL_Z_CLEAR_ACCEPTED_OUTPUT_FAIL");
            failures = failures + 1;
        end
        clear = 1'b0;
        out_ready = 1'b0;

        @(negedge clk);
        out_ready = 1'bx;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if ((out_valid !== 1'b1) || (out_f16 !== held_out_f16) ||
            (accumulator !== held_accumulator) ||
            (invalid_operand !== held_invalid) ||
            (saturation !== held_saturation)) begin
            $display("PROTOCOL_X_OUT_READY_ACCEPTED_FAIL");
            failures = failures + 1;
        end

        out_ready = 1'bz;
        four_state_control_probes = four_state_control_probes + 1;
        @(posedge clk);
        @(negedge clk);
        if ((out_valid !== 1'b1) || (out_f16 !== held_out_f16) ||
            (accumulator !== held_accumulator) ||
            (invalid_operand !== held_invalid) ||
            (saturation !== held_saturation)) begin
            $display("PROTOCOL_Z_OUT_READY_ACCEPTED_FAIL");
            failures = failures + 1;
        end

        out_ready = 1'b0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        if ((out_valid !== 1'b1) || (out_f16 !== held_out_f16) ||
            (accumulator !== held_accumulator) ||
            (invalid_operand !== held_invalid) ||
            (saturation !== held_saturation)) begin
            $display("PROTOCOL_BACKPRESSURE_FAIL");
            failures = failures + 1;
        end

        out_ready = 1'b1;
        @(posedge clk);
        completed_outputs = completed_outputs + 1;
        @(negedge clk);
        out_ready = 1'b0;
        check_accepting_idle();

        accept_start();
        pair_number = 0;
        accept_one_pair();
        @(negedge clk);
        clear = 1'b1;
        #1;
        if ((start_ready !== 1'b0) || (pair_ready !== 1'b0)) begin
            $display("PROTOCOL_CLEAR_READY_FAIL");
            failures = failures + 1;
        end
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        check_idle_state();

        if ((completed_starts != 2) || (completed_pairs != 129) ||
            (completed_outputs != 1)) begin
            $display(
                "PROTOCOL_TRANSACTION_COUNT_FAIL starts=%0d pairs=%0d outputs=%0d",
                completed_starts,
                completed_pairs,
                completed_outputs
            );
            failures = failures + 1;
        end
        if ((four_state_control_probes != 12) ||
            (four_state_data_probes != 3)) begin
            $display("PROTOCOL_XZ_PROBE_COUNT_FAIL controls=%0d data=%0d",
                     four_state_control_probes, four_state_data_probes);
            failures = failures + 1;
        end

        if ((failures == 0) && (property_failures == 0)) begin
            $display(
                "AWQ_W4A16_G128_PROTOCOL_PASS four_state=icarus controls=%0d data=%0d completed_transactions=%0d completed_pairs=128 aborted_transactions=1 reset=pass clear=pass invariants=pass",
                four_state_control_probes,
                four_state_data_probes,
                completed_outputs
            );
            $finish;
        end
        $fatal(
            1,
            "AWQ_W4A16_G128_PROTOCOL_FAIL failures=%0d property_failures=%0d",
            failures,
            property_failures
        );
    end
endmodule

`default_nettype wire
