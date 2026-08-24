`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_projection_engine_tb;
    `include "projection_params.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg [12:0] first_output_channel;
    reg [12:0] output_count;
    reg meta_valid;
    wire meta_ready;
    wire [12:0] meta_output_channel;
    wire [5:0] meta_group_index;
    wire [9:0] meta_output_word;
    wire [2:0] meta_logical_lane;
    reg [31:0] qzeros;
    reg [15:0] scale_f16;
    reg pair_valid;
    wire pair_ready;
    wire [12:0] pair_input_index;
    wire [12:0] pair_output_channel;
    wire [5:0] pair_group_index;
    wire [9:0] pair_output_word;
    wire [2:0] pair_logical_lane;
    reg [15:0] activation_f16;
    reg [31:0] qweight;
    wire out_valid;
    reg out_ready;
    wire [12:0] out_channel;
    wire [15:0] out_f16;
    wire signed [101:0] accumulator;
    wire invalid_operand;
    wire saturation;
    wire busy;

    reg [27:0] transaction_mem [0:PROJECTION_TRANSACTIONS-1];
    reg [135:0] expected_mem [0:PROJECTION_OUTPUTS-1];
    reg [47:0] meta_mem [0:PROJECTION_GROUPS-1];
    reg [47:0] pair_mem [0:PROJECTION_PAIRS-1];

    integer failures;
    integer property_failures;
    integer cycle_count;
    integer transaction_index;
    integer expected_index;
    integer meta_index;
    integer pair_index;
    integer local_output_index;
    integer group_index;
    integer element_index;
    integer vector_start_count;
    integer vector_meta_count;
    integer vector_pair_count;
    integer vector_output_count;
    integer input_stall_cycles;
    integer output_backpressure_cycles;
    integer four_state_control_probes;
    integer four_state_data_probes;
    integer transaction_start_cycle;
    integer previous_output_accept_cycle;
    integer first_output_compute_cycles;
    integer next_output_compute_cycles;
    integer visibility_cycle;
    integer expected_channel;
    reg previous_backpressured;
    reg [12:0] previous_out_channel;
    reg [15:0] previous_out_f16;
    reg signed [101:0] previous_accumulator;
    reg previous_invalid;
    reg previous_saturation;
    reg [12:0] held_channel;
    reg [15:0] held_f16;
    reg signed [101:0] held_accumulator;
    reg held_invalid;
    reg held_saturation;
    string vector_dir;
    string transactions_path;
    string expected_path;
    string meta_path;
    string pairs_path;

    ace3_awq_w4a16_projection_engine #(
        .IN_FEATURES(896),
        .OUT_FEATURES(896)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .first_output_channel_i(first_output_channel),
        .output_count_i(output_count),
        .meta_valid_i(meta_valid),
        .meta_ready_o(meta_ready),
        .meta_output_channel_o(meta_output_channel),
        .meta_group_index_o(meta_group_index),
        .meta_output_word_o(meta_output_word),
        .meta_logical_lane_o(meta_logical_lane),
        .qzeros_i(qzeros),
        .scale_f16_i(scale_f16),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .pair_input_index_o(pair_input_index),
        .pair_output_channel_o(pair_output_channel),
        .pair_group_index_o(pair_group_index),
        .pair_output_word_o(pair_output_word),
        .pair_logical_lane_o(pair_logical_lane),
        .activation_f16_i(activation_f16),
        .qweight_i(qweight),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_channel_o(out_channel),
        .out_f16_o(out_f16),
        .acc_q53_48_o(accumulator),
        .invalid_operand_o(invalid_operand),
        .saturation_o(saturation),
        .busy_o(busy)
    );

    initial begin
        clk = 1'b0;
        cycle_count = 0;
        forever begin
            #5;
            clk = 1'b1;
            cycle_count = cycle_count + 1;
            #5;
            clk = 1'b0;
        end
    end

    task drive_idle;
        begin
            clear = 1'b0;
            start_valid = 1'b0;
            first_output_channel = 13'd0;
            output_count = 13'd1;
            meta_valid = 1'b0;
            qzeros = 32'd0;
            scale_f16 = 16'h3c00;
            pair_valid = 1'b0;
            activation_f16 = 16'd0;
            qweight = 32'd0;
            out_ready = 1'b0;
        end
    endtask

    task check_cleared;
        begin
            #1;
            if ((start_ready !== 1'b1) || busy || meta_ready || pair_ready ||
                out_valid || (out_channel !== 13'd0) ||
                (out_f16 !== 16'd0) || (accumulator !== 102'sd0) ||
                invalid_operand || saturation) begin
                $display("PROJECTION_CLEARED_STATE_FAIL time=%0t", $time);
                failures = failures + 1;
            end
        end
    endtask

    task apply_reset;
        begin
            drive_idle();
            rst_n = 1'b0;
            #2;
            if (start_ready || busy || meta_ready || pair_ready || out_valid ||
                (out_channel !== 13'd0) || (out_f16 !== 16'd0) ||
                (accumulator !== 102'sd0) ||
                invalid_operand || saturation) begin
                $display("PROJECTION_ASYNC_RESET_FAIL");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            check_cleared();
        end
    endtask

    task start_raw;
        input [12:0] first_output;
        input [12:0] count;
        begin
            @(negedge clk);
            first_output_channel = first_output;
            output_count = count;
            start_valid = 1'b1;
            #1;
            if (start_ready !== 1'b1) begin
                $display("PROJECTION_START_READY_FAIL first=%0d count=%0d",
                         first_output, count);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task accept_raw_meta;
        input [31:0] selected_qzero;
        input [15:0] selected_scale;
        begin
            while (meta_ready !== 1'b1) @(negedge clk);
            qzeros = selected_qzero;
            scale_f16 = selected_scale;
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;
        end
    endtask

    task accept_raw_pair;
        input [31:0] selected_qweight;
        input [15:0] selected_activation;
        begin
            while (pair_ready !== 1'b1) @(negedge clk);
            qweight = selected_qweight;
            activation_f16 = selected_activation;
            pair_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            pair_valid = 1'b0;
        end
    endtask

    task test_abort_and_unknown_paths;
        begin
            @(negedge clk);
            first_output_channel = 13'd0;
            output_count = 13'd0;
            start_valid = 1'b1;
            #1;
            if ((start_ready !== 1'b0) || busy) begin
                $display("PROJECTION_ZERO_COUNT_ACCEPTED_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            first_output_channel = 13'd895;
            output_count = 13'd2;
            #1;
            if ((start_ready !== 1'b0) || busy) begin
                $display("PROJECTION_RANGE_OVERFLOW_ACCEPTED_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            first_output_channel = 13'd896;
            output_count = 13'd1;
            #1;
            if ((start_ready !== 1'b0) || busy) begin
                $display("PROJECTION_OUT_OF_RANGE_ACCEPTED_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            @(negedge clk);
            first_output_channel = 13'hxxxx;
            output_count = 13'hxxxx;
            start_valid = 1'bx;
            four_state_control_probes = four_state_control_probes + 1;
            four_state_data_probes = four_state_data_probes + 1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            first_output_channel = 13'd0;
            output_count = 13'd1;
            #1;
            if (busy) begin
                $display("PROJECTION_X_START_ACCEPTED_FAIL");
                failures = failures + 1;
            end

            start_valid = 1'bz;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            if (busy) begin
                $display("PROJECTION_Z_START_ACCEPTED_FAIL");
                failures = failures + 1;
            end

            clear = 1'bx;
            start_valid = 1'b1;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            start_valid = 1'b0;
            if (busy) begin
                $display("PROJECTION_X_CLEAR_ACCEPTED_START_FAIL");
                failures = failures + 1;
            end

            clear = 1'bz;
            start_valid = 1'b1;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            start_valid = 1'b0;
            if (busy) begin
                $display("PROJECTION_Z_CLEAR_ACCEPTED_START_FAIL");
                failures = failures + 1;
            end

            start_raw(13'd0, 13'd1);
            qzeros = 32'hxxxxxxxx;
            scale_f16 = 16'hxxxx;
            meta_valid = 1'bx;
            four_state_control_probes = four_state_control_probes + 1;
            four_state_data_probes = four_state_data_probes + 1;
            @(posedge clk);
            @(negedge clk);
            if ((meta_group_index !== 6'd0) ||
                (meta_output_channel !== 13'd0)) begin
                $display("PROJECTION_X_META_ACCEPTED_FAIL");
                failures = failures + 1;
            end
            meta_valid = 1'bz;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;
            accept_raw_meta(32'd0, 16'h3c00);

            activation_f16 = 16'hxxxx;
            qweight = 32'hzzzzzzzz;
            pair_valid = 1'bx;
            four_state_control_probes = four_state_control_probes + 1;
            four_state_data_probes = four_state_data_probes + 1;
            @(posedge clk);
            @(negedge clk);
            if (pair_input_index !== 13'd0) begin
                $display("PROJECTION_X_PAIR_ACCEPTED_FAIL");
                failures = failures + 1;
            end
            pair_valid = 1'bz;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            @(negedge clk);
            pair_valid = 1'b0;
            accept_raw_pair(32'd0, 16'd0);
            accept_raw_pair(32'd0, 16'd0);
            rst_n = 1'b0;
            #2;
            if (busy || meta_ready || pair_ready || out_valid) begin
                $display("PROJECTION_RESET_MID_ACTIVITY_FAIL");
                failures = failures + 1;
            end
            @(negedge clk);
            rst_n = 1'b1;
            check_cleared();

            start_raw(13'd0, 13'd1);
            accept_raw_meta(32'd0, 16'h3c00);
            accept_raw_pair(32'd0, 16'd0);
            clear = 1'b1;
            #1;
            if (meta_ready || pair_ready || start_ready) begin
                $display("PROJECTION_CLEAR_READY_FAIL");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            check_cleared();
        end
    endtask

    task start_vector_transaction;
        input integer selected_transaction;
        begin
            @(negedge clk);
            first_output_channel =
                transaction_mem[selected_transaction][12:0];
            output_count =
                transaction_mem[selected_transaction][25:13];
            start_valid = 1'b1;
            #1;
            if (start_ready !== 1'b1) begin
                $display("PROJECTION_VECTOR_START_READY_FAIL txn=%0d",
                         selected_transaction);
                failures = failures + 1;
            end
            @(posedge clk);
            vector_start_count = vector_start_count + 1;
            transaction_start_cycle = cycle_count;
            previous_output_accept_cycle = transaction_start_cycle;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task send_vector_meta;
        input integer inject_stall;
        begin
            while (meta_ready !== 1'b1) @(negedge clk);
            if (inject_stall != 0) begin
                meta_valid = 1'b0;
                @(posedge clk);
                input_stall_cycles = input_stall_cycles + 1;
                @(negedge clk);
            end
            qzeros = meta_mem[meta_index][31:0];
            scale_f16 = meta_mem[meta_index][47:32];
            meta_valid = 1'b1;
            #1;
            expected_channel = expected_mem[expected_index][12:0];
            if ((meta_output_channel !== expected_channel[12:0]) ||
                (meta_group_index !== group_index[5:0]) ||
                (meta_output_word !== expected_channel[12:3]) ||
                (meta_logical_lane !== expected_channel[2:0])) begin
                $display(
                    "PROJECTION_META_INDEX_FAIL output=%0d group=%0d got_output=%0d got_group=%0d got_word=%0d got_lane=%0d",
                    expected_channel,
                    group_index,
                    meta_output_channel,
                    meta_group_index,
                    meta_output_word,
                    meta_logical_lane
                );
                failures = failures + 1;
            end
            @(posedge clk);
            vector_meta_count = vector_meta_count + 1;
            meta_index = meta_index + 1;
            @(negedge clk);
            meta_valid = 1'b0;
        end
    endtask

    task send_vector_pair;
        input integer inject_stall;
        begin
            while (pair_ready !== 1'b1) @(negedge clk);
            if (inject_stall != 0) begin
                pair_valid = 1'b0;
                @(posedge clk);
                input_stall_cycles = input_stall_cycles + 1;
                @(negedge clk);
            end
            qweight = pair_mem[pair_index][31:0];
            activation_f16 = pair_mem[pair_index][47:32];
            pair_valid = 1'b1;
            #1;
            expected_channel = expected_mem[expected_index][12:0];
            if ((pair_input_index !==
                 (group_index * 128 + element_index)) ||
                (pair_output_channel !== expected_channel[12:0]) ||
                (pair_group_index !== group_index[5:0]) ||
                (pair_output_word !== expected_channel[12:3]) ||
                (pair_logical_lane !== expected_channel[2:0])) begin
                $display(
                    "PROJECTION_PAIR_INDEX_FAIL output=%0d group=%0d element=%0d got_input=%0d got_output=%0d",
                    expected_channel,
                    group_index,
                    element_index,
                    pair_input_index,
                    pair_output_channel
                );
                failures = failures + 1;
            end
            @(posedge clk);
            vector_pair_count = vector_pair_count + 1;
            pair_index = pair_index + 1;
            @(negedge clk);
            pair_valid = 1'b0;
        end
    endtask

    task check_vector_output;
        begin
            while (out_valid !== 1'b1) @(negedge clk);
            visibility_cycle = cycle_count;
            expected_channel = expected_mem[expected_index][12:0];
            if (transaction_index == 0) begin
                if (local_output_index == 0) begin
                    first_output_compute_cycles =
                        visibility_cycle - transaction_start_cycle;
                    if (first_output_compute_cycles != 910) begin
                        $display(
                            "PROJECTION_FIRST_OUTPUT_CYCLE_FAIL got=%0d expected=910",
                            first_output_compute_cycles
                        );
                        failures = failures + 1;
                    end
                end else begin
                    next_output_compute_cycles =
                        visibility_cycle - previous_output_accept_cycle;
                    if (next_output_compute_cycles != 910) begin
                        $display(
                            "PROJECTION_NEXT_OUTPUT_CYCLE_FAIL output=%0d got=%0d expected=910",
                            local_output_index,
                            next_output_compute_cycles
                        );
                        failures = failures + 1;
                    end
                end
            end
            if ((out_channel !== expected_channel[12:0]) ||
                (accumulator !== expected_mem[expected_index][114:13]) ||
                (out_f16 !== expected_mem[expected_index][130:115]) ||
                (invalid_operand !== expected_mem[expected_index][131]) ||
                (saturation !== expected_mem[expected_index][132])) begin
                $display(
                    "PROJECTION_NUMERIC_MISMATCH index=%0d channel=%0d got_channel=%0d got_acc=%026x exp_acc=%026x got_f16=%04x exp_f16=%04x got_invalid=%0d exp_invalid=%0d got_sat=%0d exp_sat=%0d",
                    expected_index,
                    expected_channel,
                    out_channel,
                    accumulator,
                    expected_mem[expected_index][114:13],
                    out_f16,
                    expected_mem[expected_index][130:115],
                    invalid_operand,
                    expected_mem[expected_index][131],
                    saturation,
                    expected_mem[expected_index][132]
                );
                failures = failures + 1;
            end
            if ((^out_channel === 1'bx) || (^out_f16 === 1'bx) ||
                (^accumulator === 1'bx) ||
                (invalid_operand !== 1'b0 && invalid_operand !== 1'b1) ||
                (saturation !== 1'b0 && saturation !== 1'b1)) begin
                $display("PROJECTION_UNKNOWN_VALID_OUTPUT_FAIL");
                failures = failures + 1;
            end
            held_channel = out_channel;
            held_f16 = out_f16;
            held_accumulator = accumulator;
            held_invalid = invalid_operand;
            held_saturation = saturation;

            clear = 1'bx;
            out_ready = 1'b1;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            output_backpressure_cycles = output_backpressure_cycles + 1;
            @(negedge clk);
            clear = 1'bz;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            output_backpressure_cycles = output_backpressure_cycles + 1;
            @(negedge clk);
            clear = 1'b0;
            out_ready = 1'b0;
            if ((out_valid !== 1'b1) ||
                (out_channel !== held_channel) ||
                (out_f16 !== held_f16) ||
                (accumulator !== held_accumulator) ||
                (invalid_operand !== held_invalid) ||
                (saturation !== held_saturation)) begin
                $display(
                    "PROJECTION_UNKNOWN_CLEAR_BACKPRESSURE_FAIL index=%0d",
                    expected_index
                );
                failures = failures + 1;
            end

            out_ready = 1'bx;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            output_backpressure_cycles = output_backpressure_cycles + 1;
            @(negedge clk);
            out_ready = 1'bz;
            four_state_control_probes = four_state_control_probes + 1;
            @(posedge clk);
            output_backpressure_cycles = output_backpressure_cycles + 1;
            @(negedge clk);
            out_ready = 1'b0;
            repeat (3) begin
                @(posedge clk);
                output_backpressure_cycles =
                    output_backpressure_cycles + 1;
                @(negedge clk);
                if ((out_valid !== 1'b1) ||
                    (out_channel !== held_channel) ||
                    (out_f16 !== held_f16) ||
                    (accumulator !== held_accumulator) ||
                    (invalid_operand !== held_invalid) ||
                    (saturation !== held_saturation)) begin
                    $display(
                        "PROJECTION_OUTPUT_BACKPRESSURE_FAIL index=%0d",
                        expected_index
                    );
                    failures = failures + 1;
                end
            end
            out_ready = 1'b1;
            @(posedge clk);
            vector_output_count = vector_output_count + 1;
            previous_output_accept_cycle = cycle_count;
            @(negedge clk);
            out_ready = 1'b0;
            expected_index = expected_index + 1;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            previous_backpressured <= 1'b0;
            previous_out_channel <= 13'd0;
            previous_out_f16 <= 16'd0;
            previous_accumulator <= 102'sd0;
            previous_invalid <= 1'b0;
            previous_saturation <= 1'b0;
        end else begin
            if (previous_backpressured &&
                ((out_valid !== 1'b1) ||
                 (out_channel !== previous_out_channel) ||
                 (out_f16 !== previous_out_f16) ||
                 (accumulator !== previous_accumulator) ||
                 (invalid_operand !== previous_invalid) ||
                 (saturation !== previous_saturation))) begin
                $display("PROJECTION_PROPERTY_BACKPRESSURE_FAIL");
                property_failures = property_failures + 1;
            end
            if ((out_valid === 1'b1) &&
                ((^out_channel === 1'bx) || (^out_f16 === 1'bx) ||
                 (^accumulator === 1'bx) ||
                 (invalid_operand !== 1'b0 && invalid_operand !== 1'b1) ||
                 (saturation !== 1'b0 && saturation !== 1'b1))) begin
                $display("PROJECTION_PROPERTY_UNKNOWN_OUTPUT_FAIL");
                property_failures = property_failures + 1;
            end
            previous_backpressured <=
                (out_valid === 1'b1) &&
                !((clear === 1'b0) && (out_ready === 1'b1));
            previous_out_channel <= out_channel;
            previous_out_f16 <= out_f16;
            previous_accumulator <= accumulator;
            previous_invalid <= invalid_operand;
            previous_saturation <= saturation;
        end
    end

    initial begin
        failures = 0;
        property_failures = 0;
        vector_start_count = 0;
        vector_meta_count = 0;
        vector_pair_count = 0;
        vector_output_count = 0;
        input_stall_cycles = 0;
        output_backpressure_cycles = 0;
        four_state_control_probes = 0;
        four_state_data_probes = 0;
        first_output_compute_cycles = 0;
        next_output_compute_cycles = 0;
        expected_index = 0;
        meta_index = 0;
        pair_index = 0;
        previous_backpressured = 1'b0;
        rst_n = 1'b1;
        drive_idle();

        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "build/projection_vectors";
        transactions_path = {vector_dir, "/transactions.hex"};
        expected_path = {vector_dir, "/expected.hex"};
        meta_path = {vector_dir, "/meta.hex"};
        pairs_path = {vector_dir, "/pairs.hex"};
        $readmemh(transactions_path, transaction_mem);
        $readmemh(expected_path, expected_mem);
        $readmemh(meta_path, meta_mem);
        $readmemh(pairs_path, pair_mem);

        apply_reset();
        test_abort_and_unknown_paths();

        for (transaction_index = 0;
             transaction_index < PROJECTION_TRANSACTIONS;
             transaction_index = transaction_index + 1) begin
            start_vector_transaction(transaction_index);
            for (local_output_index = 0;
                 local_output_index <
                    transaction_mem[transaction_index][25:13];
                 local_output_index = local_output_index + 1) begin
                for (group_index = 0; group_index < 7;
                     group_index = group_index + 1) begin
                    send_vector_meta(
                        (transaction_index != 0) &&
                        ((group_index % 3) == 1)
                    );
                    for (element_index = 0; element_index < 128;
                         element_index = element_index + 1)
                        send_vector_pair(
                            (transaction_index != 0) &&
                            ((element_index % 113) == 17)
                        );
                end
                check_vector_output();
            end
        end

        if ((vector_start_count != PROJECTION_TRANSACTIONS) ||
            (vector_meta_count != PROJECTION_GROUPS) ||
            (vector_pair_count != PROJECTION_PAIRS) ||
            (vector_output_count != PROJECTION_OUTPUTS) ||
            (expected_index != PROJECTION_OUTPUTS) ||
            (meta_index != PROJECTION_GROUPS) ||
            (pair_index != PROJECTION_PAIRS)) begin
            $display(
                "PROJECTION_TRANSACTION_COUNT_FAIL starts=%0d meta=%0d pairs=%0d outputs=%0d",
                vector_start_count,
                vector_meta_count,
                vector_pair_count,
                vector_output_count
            );
            failures = failures + 1;
        end
        if ((four_state_control_probes != 64) ||
            (four_state_data_probes != 3)) begin
            $display("PROJECTION_XZ_COUNT_FAIL controls=%0d data=%0d",
                     four_state_control_probes, four_state_data_probes);
            failures = failures + 1;
        end
        if ((failures == 0) && (property_failures == 0)) begin
            $display(
                "AWQ_W4A16_PROJECTION_PASS transactions=%0d outputs=%0d official_outputs=8 groups=%0d pairs=%0d ulp_bound=0 first_output_compute_cycles=%0d next_output_compute_cycles=%0d input_stall_cycles=%0d output_backpressure_cycles=%0d reset=pass clear=pass protocol=pass four_state_controls=%0d four_state_data=%0d",
                PROJECTION_TRANSACTIONS,
                PROJECTION_OUTPUTS,
                PROJECTION_GROUPS,
                PROJECTION_PAIRS,
                first_output_compute_cycles,
                next_output_compute_cycles,
                input_stall_cycles,
                output_backpressure_cycles,
                four_state_control_probes,
                four_state_data_probes
            );
            $finish;
        end
        $fatal(
            1,
            "AWQ_W4A16_PROJECTION_FAIL failures=%0d property_failures=%0d",
            failures,
            property_failures
        );
    end
endmodule

`default_nettype wire
