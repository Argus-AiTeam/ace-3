`timescale 1ns/1ps
`default_nettype none

module ace3_qkv_projection_geometry_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg [2:0] start_valid;
    wire [2:0] start_ready;
    reg [38:0] first_output_channel;
    reg [38:0] output_count;
    reg [2:0] meta_valid;
    wire [2:0] meta_ready;
    wire [38:0] meta_output_channel;
    wire [17:0] meta_group_index;
    wire [29:0] meta_output_word;
    wire [8:0] meta_logical_lane;
    reg [95:0] qzeros;
    reg [47:0] scales;
    reg [2:0] pair_valid;
    wire [2:0] pair_ready;
    wire [38:0] pair_input_index;
    wire [38:0] pair_output_channel;
    wire [17:0] pair_group_index;
    wire [29:0] pair_output_word;
    wire [8:0] pair_logical_lane;
    reg [47:0] activations;
    reg [95:0] qweights;
    wire [2:0] out_valid;
    reg [2:0] out_ready;
    wire [38:0] out_channel;
    wire [47:0] out_f16;
    wire [305:0] accumulators;
    wire [2:0] invalid;
    wire [2:0] saturation;
    wire [2:0] busy;
    wire all_idle;
    integer failures;

    ace3_qkv_projection_cluster dut (
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
        .scale_f16_i(scales),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .pair_input_index_o(pair_input_index),
        .pair_output_channel_o(pair_output_channel),
        .pair_group_index_o(pair_group_index),
        .pair_output_word_o(pair_output_word),
        .pair_logical_lane_o(pair_logical_lane),
        .activation_f16_i(activations),
        .qweight_i(qweights),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_channel_o(out_channel),
        .out_f16_o(out_f16),
        .acc_q53_48_o(accumulators),
        .invalid_operand_o(invalid),
        .saturation_o(saturation),
        .busy_o(busy),
        .all_idle_o(all_idle)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        failures = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        start_valid = 3'b000;
        first_output_channel = 39'd0;
        output_count = {13'd128, 13'd128, 13'd896};
        meta_valid = 3'b000;
        qzeros = 96'd0;
        scales = {3{16'h3c00}};
        pair_valid = 3'b000;
        activations = 48'd0;
        qweights = 96'd0;
        out_ready = 3'b000;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        #1;
        if ((start_ready !== 3'b111) || !all_idle)
            failures = failures + 1;

        start_valid = 3'b111;
        @(posedge clk);
        @(negedge clk);
        start_valid = 3'b000;
        #1;
        if ((busy !== 3'b111) || (meta_ready !== 3'b111) ||
            (meta_output_channel !== 39'd0) ||
            (meta_group_index !== 18'd0) ||
            (meta_output_word !== 30'd0) ||
            (meta_logical_lane !== 9'd0))
            failures = failures + 1;

        meta_valid = 3'b111;
        @(posedge clk);
        @(negedge clk);
        meta_valid = 3'b000;
        #1;
        if ((pair_ready !== 3'b111) ||
            (pair_input_index !== 39'd0) ||
            (pair_output_channel !== 39'd0))
            failures = failures + 1;

        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        #1;
        if ((busy !== 3'b000) || !all_idle || (out_valid !== 3'b000))
            failures = failures + 1;

        output_count = {13'd128, 13'd129, 13'd896};
        start_valid = 3'b010;
        #1;
        if (start_ready[1] !== 1'b0)
            failures = failures + 1;
        output_count = {13'd129, 13'd128, 13'd897};
        start_valid = 3'b101;
        #1;
        if ((start_ready[2] !== 1'b0) || (start_ready[0] !== 1'b0))
            failures = failures + 1;
        start_valid = 3'b000;

        if (failures == 0)
            $display("QKV_PROJECTION_GEOMETRY_PASS q=896x896 k=896x128 v=896x128 groups=7");
        else
            $display("QKV_PROJECTION_GEOMETRY_FAIL failures=%0d", failures);
        $finish(failures != 0);
    end
endmodule

`default_nettype wire
