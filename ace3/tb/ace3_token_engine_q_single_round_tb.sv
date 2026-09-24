`timescale 1ns/1ps
`default_nettype none

module ace3_token_engine_q_single_round_tb;
    parameter integer JOBS = 1;
    parameter integer OUTPUTS = 1;
    parameter integer LAYER_INDEX = 0;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, clear = 0;
    reg meta_valid = 0, pair_valid = 0, bias_valid = 0, trace_ready = 0;
    reg [31:0] qzeros = 0, qweight = 0;
    reg [15:0] scale = 0, bias = 0;
    wire mr, pr, br, tv;
    wire [2:0] kind, ml, pl;
    wire [12:0] mc, pi, pc, bc, ti;
    wire [5:0] mg, pg, phase;
    wire [9:0] mw, pw;
    wire [4:0] ts, layer;
    wire [15:0] value;
    wire [14:0] position;
    ace3_decoder_layer0_token_engine #(.LAYER_INDEX(LAYER_INDEX)) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .load_valid_i(1'b0), .load_ready_o(), .load_kind_i(2'd0),
        .load_index_i(13'd0), .load_f16_i(16'd0),
        .start_valid_i(1'b0), .start_ready_o(),
        .start_cache_slot_i(2'd0), .start_position_i(15'd0),
        .projection_kind_o(kind),
        .projection_meta_valid_i(meta_valid), .projection_meta_ready_o(mr),
        .projection_meta_output_channel_o(mc), .projection_meta_group_o(mg),
        .projection_meta_word_o(mw), .projection_meta_lane_o(ml),
        .projection_qzeros_i(qzeros), .projection_scale_f16_i(scale),
        .projection_pair_valid_i(pair_valid), .projection_pair_ready_o(pr),
        .projection_pair_input_o(pi), .projection_pair_output_o(pc),
        .projection_pair_group_o(pg), .projection_pair_word_o(pw),
        .projection_pair_lane_o(pl), .projection_qweight_i(qweight),
        .projection_bias_valid_i(bias_valid), .projection_bias_ready_o(br),
        .projection_bias_output_channel_o(bc), .projection_bias_f16_i(bias),
        .rope_valid_i(1'b0), .rope_ready_o(), .rope_position_o(), .rope_pair_o(),
        .rope_cos_f16_i(16'h3c00), .rope_sin_f16_i(16'd0),
        .trace_valid_o(tv), .trace_ready_i(trace_ready), .trace_stage_o(ts),
        .trace_index_o(ti), .trace_f16_o(value), .trace_position_o(position),
        .final_valid_o(), .final_ready_i(1'b0), .final_index_o(),
        .final_f16_o(), .final_last_o(), .done_valid_o(), .done_ready_i(1'b0),
        .done_cache_slot_o(), .done_position_o(), .done_cycles_o(),
        .done_stall_cycles_o(), .busy_o(), .phase_o(phase), .layer_index_o(layer)
    );
    reg [15:0] jobs [0:JOBS-1];
    reg [15:0] biases [0:OUTPUTS-1];
    reg [47:0] metadata [0:OUTPUTS*7-1];
    reg [47:0] pairs [0:OUTPUTS*896-1];
    reg [132:0] held;
    reg [48:0] held_trace;
    string dir, output_path, wave;
    integer file, j, c, g, p, stall, row = 0;
    task tick;
        begin @(posedge clk); #1; end
    endtask
    task abort_job;
        begin
            @(negedge clk); clear = 1; meta_valid = 0; pair_valid = 0; bias_valid = 0;
            tick();
            if (tv !== 0 || dut.pb_busy_w !== 0 || phase !== dut.S_IDLE)
                $fatal(1, "clear failed");
            @(negedge clk); clear = 0;
        end
    endtask
    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s",dir) ||
            !$value$plusargs("OUTPUT=%s",output_path)) $fatal(1, "missing paths");
        $readmemh({dir,"/jobs.hex"},jobs);
        $readmemh({dir,"/biases.hex"},biases);
        $readmemh({dir,"/token_meta.hex"},metadata);
        $readmemh({dir,"/token_pairs.hex"},pairs);
        file = $fopen(output_path,"w");
        if (!file) $fatal(1, "output open");
        if ($value$plusargs("WAVE=%s",wave)) begin
            $dumpfile(wave); $dumpvars(1,ace3_token_engine_q_single_round_tb);
            $dumpvars(0,dut.b_acc_w,dut.b_out_f16_w,dut.b_invalid_w,
                      dut.b_saturation_w,dut.pb_busy_w);
        end
        if (dut.bias_policy[1].p_bias.SINGLE_ROUND_BIAS != 1 ||
            dut.bias_policy[0].p_bias.SINGLE_ROUND_BIAS != 0 ||
            dut.p_out.BIAS_ENABLE != 0 || dut.p_out.SINGLE_ROUND_BIAS != 0 ||
            dut.p_ffn.BIAS_ENABLE != 0 || dut.p_ffn.SINGLE_ROUND_BIAS != 0 ||
            dut.p_down.BIAS_ENABLE != 0 || dut.p_down.SINGLE_ROUND_BIAS != 0 ||
            dut.ACCURATE_SILU != (LAYER_INDEX >= 3)) $fatal(1, "policy parameters");
        repeat (3) tick();
        @(negedge clk); rst_n = 1;
        for (j = 0; j < JOBS; j = j + 1) begin
            abort_job();
            // Seed only the projection input boundary, never results or accumulators.
            for (p = 0; p < 896; p = p + 1)
                dut.norm1_mem[p] = pairs[row*896+p][47:32];
            dut.psel_q = jobs[j][15:13];
            dut.token_position_q = 7'd2;
            dut.state_q = dut.S_P_START;
            #1;
            if (dut.b_start_ready_w !== 1 || layer !== LAYER_INDEX[4:0])
                $fatal(1, "start or layer parameter");
            tick();
            if (phase !== dut.S_P_RUN) $fatal(1, "controller start");
            for (c = 0; c < jobs[j][12:0]; c = c + 1) begin
                for (g = 0; g < 7; g = g + 1) begin
                    @(negedge clk);
                    meta_valid = 1; scale = 16'hxxxx; qzeros = 32'hzzzz; #1;
                    if (mr !== 0) $fatal(1, "unknown metadata accepted");
                    tick();
                    @(negedge clk); {scale,qzeros} = metadata[row*7+g]; #1;
                    if (mr !== 1 || mc !== c[12:0] || mg !== g[5:0] ||
                        mw !== c[12:3] || ml !== c[2:0]) $fatal(1, "metadata address");
                    tick();
                    @(negedge clk); meta_valid = 0;
                    for (p = 0; p < 128; p = p + 1) begin
                        if (p % 31 == 0) begin
                            pair_valid = 1; qweight = 32'hxxxxxxxx; #1;
                            if (pr !== 0) $fatal(1, "unknown pair accepted");
                            tick();
                            @(negedge clk); pair_valid = 0;
                            tick(); @(negedge clk);
                        end
                        qweight = pairs[row*896+g*128+p][31:0]; pair_valid = 1; #1;
                        if (pr !== 1 || pi !== (g*128+p) || pc !== c[12:0] ||
                            pg !== g[5:0] || pw !== c[12:3] || pl !== c[2:0])
                            $fatal(1, "pair address");
                        tick();
                        @(negedge clk); pair_valid = 0;
                    end
                    tick();
                end
                @(negedge clk); bias_valid = 1; bias = 16'hxxxx; #1;
                if (br !== 0) $fatal(1, "X bias accepted");
                tick();
                @(negedge clk); bias = 16'hzzzz; #1;
                if (br !== 0) $fatal(1, "Z bias accepted");
                tick();
                @(negedge clk); bias = biases[row]; #1;
                if (br !== 1 || bc !== c[12:0]) $fatal(1, "bias address");
                tick();
                @(negedge clk); bias_valid = 0;
                if (dut.b_out_valid_w !== 1) $fatal(1, "missing projection output");
                held = {dut.b_out_ch_w,dut.b_out_f16_w,dut.b_acc_w,
                        dut.b_invalid_w,dut.b_saturation_w};
                $fdisplay(file,"%04x %01x %04x %04x %026x %01x %01x",
                    row,kind,dut.b_out_ch_w,dut.b_out_f16_w,dut.b_acc_w,
                    dut.b_invalid_w,dut.b_saturation_w);
                tick();
                if (held[1:0] != 0) begin
                    if (phase !== dut.S_FAULT || tv !== 0 || mr || pr || br)
                        $fatal(1, "numeric fault not fail-closed");
                    repeat (3) tick();
                    if (phase !== dut.S_FAULT) $fatal(1, "fault not sticky");
                end else begin
                    if (tv !== 1 || ts !== (jobs[j][15:13]+1) ||
                        ti !== c[12:0] || value !== held[119:104] ||
                        position !== 15'd2) $fatal(1, "trace identity or value");
                    held_trace = {ts,ti,value,position};
                    repeat (3) begin
                        tick();
                        if (tv !== 1 || {ts,ti,value,position} !== held_trace)
                            $fatal(1, "trace changed under backpressure");
                    end
                    @(negedge clk); trace_ready = 1;
                    tick();
                    @(negedge clk); trace_ready = 0;
                end
                row = row + 1;
            end
        end
        abort_job();
        if (row != OUTPUTS) $fatal(1, "output count");
        $fclose(file);
        $display("TOKEN_ENGINE_Q_POLICY_STREAM_PASS outputs=%0d jobs=%0d layer=%0d",
                 row,JOBS,LAYER_INDEX);
        $finish;
    end
    initial begin #100000000; $fatal(1, "watchdog"); end
endmodule
`default_nettype wire
