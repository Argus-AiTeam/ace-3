`timescale 1ns/1ps
`default_nettype none

module ace3_awq_w4a16_projection_4864_cycle_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg [12:0] first_output;
    reg [12:0] output_count;
    reg meta_valid;
    wire meta_ready;
    wire [5:0] meta_group;
    reg [31:0] qzeros;
    reg [15:0] scale;
    reg pair_valid;
    wire pair_ready;
    wire [12:0] pair_input_index;
    reg [15:0] activation;
    reg [31:0] qweight;
    wire out_valid;
    reg out_ready;
    wire [12:0] out_channel;
    wire [15:0] out_f16;
    wire signed [101:0] accumulator;
    wire invalid_operand;
    wire saturation;
    wire busy;
    integer cycle_count;
    integer start_cycle;
    integer compute_cycles;
    integer group_index;
    integer element_index;
    integer failures;

    ace3_awq_w4a16_projection_engine #(
        .IN_FEATURES(4864),
        .OUT_FEATURES(896)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .first_output_channel_i(first_output),
        .output_count_i(output_count),
        .meta_valid_i(meta_valid),
        .meta_ready_o(meta_ready),
        .meta_output_channel_o(),
        .meta_group_index_o(meta_group),
        .meta_output_word_o(),
        .meta_logical_lane_o(),
        .qzeros_i(qzeros),
        .scale_f16_i(scale),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .pair_input_index_o(pair_input_index),
        .pair_output_channel_o(),
        .pair_group_index_o(),
        .pair_output_word_o(),
        .pair_logical_lane_o(),
        .activation_f16_i(activation),
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

    initial begin
        failures = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        start_valid = 1'b0;
        first_output = 13'd0;
        output_count = 13'd1;
        meta_valid = 1'b0;
        qzeros = 32'd0;
        scale = 16'h3c00;
        pair_valid = 1'b0;
        activation = 16'd0;
        qweight = 32'd0;
        out_ready = 1'b0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        #1;
        if (!start_ready || busy)
            failures = failures + 1;

        @(negedge clk);
        start_valid = 1'b1;
        @(posedge clk);
        start_cycle = cycle_count;
        @(negedge clk);
        start_valid = 1'b0;

        for (group_index = 0; group_index < 38;
             group_index = group_index + 1) begin
            while (!meta_ready) @(negedge clk);
            if (meta_group !== group_index[5:0])
                failures = failures + 1;
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;
            for (element_index = 0; element_index < 128;
                 element_index = element_index + 1) begin
                while (!pair_ready) @(negedge clk);
                if (pair_input_index !==
                    (group_index * 128 + element_index))
                    failures = failures + 1;
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end
        end
        while (!out_valid) @(negedge clk);
        compute_cycles = cycle_count - start_cycle;
        if ((compute_cycles != 4940) || (out_channel !== 13'd0) ||
            (out_f16 !== 16'd0) || (accumulator !== 102'sd0) ||
            invalid_operand || saturation)
            failures = failures + 1;
        repeat (2) begin
            @(posedge clk);
            @(negedge clk);
            if (!out_valid || (out_f16 !== 16'd0) ||
                (accumulator !== 102'sd0))
                failures = failures + 1;
        end
        out_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        out_ready = 1'b0;
        if (out_valid || busy)
            failures = failures + 1;

        if (failures == 0) begin
            $display(
                "AWQ_W4A16_PROJECTION_4864_CYCLE_PASS in_features=4864 groups=38 pairs=4864 compute_cycles=%0d result=zero backpressure=pass",
                compute_cycles
            );
            $finish;
        end
        $fatal(
            1,
            "AWQ_W4A16_PROJECTION_4864_CYCLE_FAIL failures=%0d",
            failures
        );
    end
endmodule

`default_nettype wire
