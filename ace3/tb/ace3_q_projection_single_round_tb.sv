`timescale 1ns/1ps
`default_nettype none

module ace3_q_projection_single_round_tb;
    parameter integer CASES = 1;
    parameter integer SELECT_Q = 1;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, clear = 0;
    reg start_valid = 0, meta_valid = 0, pair_valid = 0, bias_valid = 0, out_ready = 0;
    reg [12:0] channel = 0;
    reg [31:0] qzeros = 0, qweight = 0;
    reg [15:0] scale = 0, activation = 0, bias = 0;
    wire [2:0] sr, mr, pr, br, ov, invalid, saturation, busy;
    wire [38:0] mc, pi, pc, bc, oc;
    wire [17:0] mg, pg;
    wire [29:0] mw, pw;
    wire [8:0] ml, pl;
    wire [47:0] value;
    wire [305:0] acc;
    generate if (SELECT_Q != 0) begin : selected
        ace3_qkv_projection_cluster dut (
            .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
            .start_valid_i({2'd0,start_valid}), .start_ready_o(sr),
            .first_output_channel_i({26'd0,channel}), .output_count_i(39'd1),
            .meta_valid_i({2'd0,meta_valid}), .meta_ready_o(mr),
            .meta_output_channel_o(mc), .meta_group_index_o(mg),
            .meta_output_word_o(mw), .meta_logical_lane_o(ml),
            .qzeros_i({64'd0,qzeros}), .scale_f16_i({32'd0,scale}),
            .pair_valid_i({2'd0,pair_valid}), .pair_ready_o(pr),
            .pair_input_index_o(pi), .pair_output_channel_o(pc),
            .pair_group_index_o(pg), .pair_output_word_o(pw), .pair_logical_lane_o(pl),
            .activation_f16_i({32'd0,activation}), .qweight_i({64'd0,qweight}),
            .bias_valid_i({2'd0,bias_valid}), .bias_ready_o(br),
            .bias_output_channel_o(bc), .bias_f16_i({32'd0,bias}),
            .out_valid_o(ov), .out_ready_i({2'd0,out_ready}), .out_channel_o(oc),
            .out_f16_o(value), .acc_q53_48_o(acc), .invalid_operand_o(invalid),
            .saturation_o(saturation), .busy_o(busy), .all_idle_o()
        );
    end else begin : legacy
        ace3_awq_w4a16_projection_engine #(
            .IN_FEATURES(896), .OUT_FEATURES(896), .BIAS_ENABLE(1)
        ) dut (
            .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
            .start_valid_i(start_valid), .start_ready_o(sr[0]),
            .first_output_channel_i(channel), .output_count_i(13'd1),
            .meta_valid_i(meta_valid), .meta_ready_o(mr[0]),
            .meta_output_channel_o(mc[12:0]), .meta_group_index_o(mg[5:0]),
            .meta_output_word_o(mw[9:0]), .meta_logical_lane_o(ml[2:0]),
            .qzeros_i(qzeros), .scale_f16_i(scale),
            .pair_valid_i(pair_valid), .pair_ready_o(pr[0]),
            .pair_input_index_o(pi[12:0]), .pair_output_channel_o(pc[12:0]),
            .pair_group_index_o(pg[5:0]), .pair_output_word_o(pw[9:0]),
            .pair_logical_lane_o(pl[2:0]),
            .activation_f16_i(activation), .qweight_i(qweight),
            .bias_valid_i(bias_valid), .bias_ready_o(br[0]),
            .bias_output_channel_o(bc[12:0]), .bias_f16_i(bias),
            .out_valid_o(ov[0]), .out_ready_i(out_ready), .out_channel_o(oc[12:0]),
            .out_f16_o(value[15:0]), .acc_q53_48_o(acc[101:0]),
            .invalid_operand_o(invalid[0]), .saturation_o(saturation[0]), .busy_o(busy[0])
        );
    end endgenerate

    reg [31:0] headers [0:CASES-1];
    reg [47:0] metadata [0:CASES*7-1];
    reg [47:0] pairs [0:CASES*896-1];
    reg [132:0] held;
    string dir, output_path, wave;
    integer file, c, g, p, stall;
    task tick;
        begin @(posedge clk); #1; end
    endtask
    task begin_case;
        begin
            @(negedge clk);
            start_valid = 1;
            #1;
            if (sr[0] !== 1) $fatal(1, "start handshake");
            tick();
            @(negedge clk); start_valid = 0;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s",dir) ||
            !$value$plusargs("OUTPUT=%s",output_path)) $fatal(1, "missing paths");
        $readmemh({dir,"/cases.hex"},headers);
        $readmemh({dir,"/meta.hex"},metadata);
        $readmemh({dir,"/pairs.hex"},pairs);
        file = $fopen(output_path,"w");
        if (!file) $fatal(1, "output open");
        if ($value$plusargs("WAVE=%s",wave)) begin
            $dumpfile(wave); $dumpvars(1,ace3_q_projection_single_round_tb);
        end
        repeat (3) tick();
        @(negedge clk); rst_n = 1;
        channel = headers[0][28:16];
        begin_case();
        // Abort a transaction before metadata, then restart from clean state.
        clear = 1; tick();
        if (ov[0] !== 0 || busy[0] !== 0) $fatal(1, "clear abort");
        @(negedge clk); clear = 0;
        for (c = 0; c < CASES; c = c + 1) begin
            channel = headers[c][28:16];
            begin_case();
            for (g = 0; g < 7; g = g + 1) begin
                tick();
                @(negedge clk);
                {scale,qzeros} = metadata[c*7+g]; meta_valid = 1;
                #1;
                if (mr[0] !== 1 || mc[12:0] !== channel ||
                    mg[5:0] !== g[5:0] || mw[9:0] !== channel[12:3] ||
                    ml[2:0] !== channel[2:0]) $fatal(1, "metadata address");
                tick();
                @(negedge clk); meta_valid = 0;
                for (p = 0; p < 128; p = p + 1) begin
                    if (p % 31 == 0) begin
                        pair_valid = 0; tick(); @(negedge clk);
                    end
                    {activation,qweight} = pairs[c*896+g*128+p]; pair_valid = 1;
                    #1;
                    if (pr[0] !== 1 || pi[12:0] !== (g*128+p) ||
                        pc[12:0] !== channel || pg[5:0] !== g[5:0] ||
                        pw[9:0] !== channel[12:3] || pl[2:0] !== channel[2:0])
                        $fatal(1, "pair address");
                    tick();
                    @(negedge clk); pair_valid = 0;
                end
                tick();
                @(negedge clk);
            end
            bias_valid = 1; bias = 16'hxxxx; #1;
            if (br[0] !== 0) $fatal(1, "X bias accepted");
            tick();
            @(negedge clk); bias = 16'hzzzz; #1;
            if (br[0] !== 0) $fatal(1, "Z bias accepted");
            tick();
            @(negedge clk); bias = headers[c][15:0]; #1;
            if (br[0] !== 1 || bc[12:0] !== channel) $fatal(1, "bias handshake");
            tick();
            @(negedge clk); bias_valid = 0; bias = 16'h7e00;
            if (ov[0] !== 1) $fatal(1, "output absent");
            held = {oc[12:0],value[15:0],acc[101:0],invalid[0],saturation[0]};
            for (stall = 0; stall < 3; stall = stall + 1) begin
                tick();
                if (ov[0] !== 1 ||
                    {oc[12:0],value[15:0],acc[101:0],invalid[0],saturation[0]} !== held)
                    $fatal(1, "output changed under backpressure");
            end
            $fdisplay(file,"%04x %04x %026x %01x %01x",
                      oc[12:0],value[15:0],acc[101:0],invalid[0],saturation[0]);
            @(negedge clk); out_ready = 1;
            tick();
            @(negedge clk); out_ready = 0;
        end
        $fclose(file);
        $display("Q_PROJECTION_STREAM_PASS cases=%0d select_q=%0d",CASES,SELECT_Q);
        $finish;
    end
    initial begin #10000000; $fatal(1, "watchdog"); end
endmodule
`default_nettype wire
