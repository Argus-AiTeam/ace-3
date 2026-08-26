`timescale 1ns/1ps
`default_nettype none

module ace3_projection_bias_tb;
    localparam integer CASES = 5;
    localparam integer PAIRS_PER_CASE = 128;
    reg clk, rst_n, clear, start_valid, meta_valid, pair_valid, bias_valid, out_ready;
    reg [12:0] first_channel, output_count;
    reg [31:0] qzeros, qweight;
    reg [15:0] scale_f16, activation_f16, bias_f16;
    wire start_ready, meta_ready, pair_ready, bias_ready, out_valid, busy;
    wire [12:0] meta_channel, pair_input, pair_channel, bias_channel, out_channel;
    wire [5:0] meta_group, pair_group;
    wire [9:0] meta_word, pair_word;
    wire [2:0] meta_lane, pair_lane;
    wire [15:0] out_f16;
    wire signed [101:0] accumulator;
    wire invalid_operand, saturation;
    reg [31:0] cases_mem [0:CASES-1];
    reg [47:0] meta_mem [0:CASES-1];
    reg [47:0] pairs_mem [0:CASES*PAIRS_PER_CASE-1];
    reg [127:0] expected_mem [0:CASES-1];
    integer failures, cycles, stalls, xz_probes, case_no, pair_no;
    integer expected_lane;
    string vector_dir, path;
    reg [12:0] held_channel;
    reg [15:0] held_f16;
    reg signed [101:0] held_acc;
    reg held_invalid, held_saturation;

    ace3_awq_w4a16_projection_engine #(
        .IN_FEATURES(128), .OUT_FEATURES(8), .BIAS_ENABLE(1)
    ) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .first_output_channel_i(first_channel), .output_count_i(output_count),
        .meta_valid_i(meta_valid), .meta_ready_o(meta_ready),
        .meta_output_channel_o(meta_channel), .meta_group_index_o(meta_group),
        .meta_output_word_o(meta_word), .meta_logical_lane_o(meta_lane),
        .qzeros_i(qzeros), .scale_f16_i(scale_f16),
        .pair_valid_i(pair_valid), .pair_ready_o(pair_ready),
        .pair_input_index_o(pair_input), .pair_output_channel_o(pair_channel),
        .pair_group_index_o(pair_group), .pair_output_word_o(pair_word),
        .pair_logical_lane_o(pair_lane), .activation_f16_i(activation_f16),
        .qweight_i(qweight), .bias_valid_i(bias_valid),
        .bias_ready_o(bias_ready), .bias_output_channel_o(bias_channel),
        .bias_f16_i(bias_f16), .out_valid_o(out_valid), .out_ready_i(out_ready),
        .out_channel_o(out_channel), .out_f16_o(out_f16),
        .acc_q53_48_o(accumulator), .invalid_operand_o(invalid_operand),
        .saturation_o(saturation), .busy_o(busy)
    );

    always #5 clk = ~clk;
    always @(posedge clk) cycles <= cycles + 1;

    task idle;
        begin
            clear=0; start_valid=0; first_channel=0; output_count=1;
            meta_valid=0; qzeros=0; scale_f16=16'h3c00;
            pair_valid=0; activation_f16=0; qweight=0;
            bias_valid=0; bias_f16=0; out_ready=0;
        end
    endtask

    task reset_dut;
        begin
            idle; rst_n=0; #2;
            if (start_ready || meta_ready || pair_ready || bias_ready || out_valid || busy) begin
                $display("PROJECTION_BIAS_ASYNC_RESET_FAIL"); failures=failures+1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk); rst_n=1; #1;
            if (!start_ready || busy || out_valid) begin
                $display("PROJECTION_BIAS_RESET_RELEASE_FAIL"); failures=failures+1;
            end
        end
    endtask

    task begin_case;
        input integer n;
        begin
            @(negedge clk);
            first_channel=cases_mem[n][23:16]; output_count=1; start_valid=1;
            #1;
            if (start_ready !== 1'b1) begin
                $display("PROJECTION_BIAS_START_FAIL case=%0d",n); failures=failures+1;
            end
            @(posedge clk); @(negedge clk); start_valid=0;
        end
    endtask

    task send_meta;
        input integer n;
        input integer inject_stall;
        begin
            while (meta_ready !== 1'b1) @(negedge clk);
            if (inject_stall != 0) begin @(posedge clk); stalls=stalls+1; @(negedge clk); end
            qzeros=meta_mem[n][31:0]; scale_f16=meta_mem[n][47:32]; meta_valid=1;
            #1;
            if (meta_channel!==cases_mem[n][23:16] || meta_group!==0 ||
                meta_word!==0 || meta_lane!==cases_mem[n][18:16]) begin
                $display("PROJECTION_BIAS_META_ADDRESS_FAIL case=%0d",n); failures=failures+1;
            end
            @(posedge clk); @(negedge clk); meta_valid=0;
        end
    endtask

    task send_pair;
        input integer n;
        input integer p;
        input integer inject_stall;
        integer at;
        begin
            at=n*PAIRS_PER_CASE+p;
            while (pair_ready !== 1'b1) @(negedge clk);
            if (inject_stall != 0) begin @(posedge clk); stalls=stalls+1; @(negedge clk); end
            qweight=pairs_mem[at][31:0]; activation_f16=pairs_mem[at][47:32]; pair_valid=1;
            #1;
            if (pair_input!==p || pair_channel!==cases_mem[n][23:16] ||
                pair_group!==0 || pair_word!==0 || pair_lane!==cases_mem[n][18:16]) begin
                $display("PROJECTION_BIAS_PAIR_ADDRESS_FAIL case=%0d pair=%0d",n,p); failures=failures+1;
            end
            @(posedge clk); @(negedge clk); pair_valid=0;
        end
    endtask

    task send_bias;
        input integer n;
        input integer inject_stall;
        begin
            while (bias_ready !== 1'b1) @(negedge clk);
            if (inject_stall != 0) begin @(posedge clk); stalls=stalls+1; @(negedge clk); end
            bias_f16=cases_mem[n][15:0]; bias_valid=1;
            #1;
            if (bias_channel!==cases_mem[n][23:16]) begin
                $display("PROJECTION_BIAS_ADDRESS_FAIL case=%0d",n); failures=failures+1;
            end
            @(posedge clk); @(negedge clk); bias_valid=0;
        end
    endtask

    task check_output;
        input integer n;
        begin
            while (out_valid !== 1'b1) @(negedge clk);
            if (out_channel!==cases_mem[n][23:16] ||
                accumulator!==expected_mem[n][109:8] ||
                out_f16!==expected_mem[n][125:110] ||
                invalid_operand!==expected_mem[n][126] ||
                saturation!==expected_mem[n][127]) begin
                $display("PROJECTION_BIAS_NUMERIC_FAIL case=%0d channel=%0d acc=%026x f16=%04x invalid=%0d sat=%0d",
                    n,out_channel,accumulator,out_f16,invalid_operand,saturation);
                failures=failures+1;
            end
            held_channel=out_channel; held_f16=out_f16; held_acc=accumulator;
            held_invalid=invalid_operand; held_saturation=saturation;
            repeat (3) begin
                @(posedge clk); stalls=stalls+1; @(negedge clk);
                if (out_valid!==1'b1 || out_channel!==held_channel || out_f16!==held_f16 ||
                    accumulator!==held_acc || invalid_operand!==held_invalid ||
                    saturation!==held_saturation) begin
                    $display("PROJECTION_BIAS_OUTPUT_STABILITY_FAIL case=%0d",n); failures=failures+1;
                end
            end
            out_ready=1; @(posedge clk); @(negedge clk); out_ready=0;
        end
    endtask

    task protocol_probes;
        begin
            @(negedge clk); first_channel=0; output_count=0; start_valid=1; #1;
            if (start_ready!==0) begin $display("PROJECTION_BIAS_ZERO_CONFIG_FAIL"); failures=failures+1; end
            output_count=1; first_channel=8; #1;
            if (start_ready!==0) begin $display("PROJECTION_BIAS_RANGE_CONFIG_FAIL"); failures=failures+1; end
            start_valid=0;
`ifndef VERILATOR
            first_channel=13'hxxx; start_valid=1'bx; xz_probes=xz_probes+1;
            @(posedge clk); @(negedge clk); start_valid=0; first_channel=0;
            if (busy) begin $display("PROJECTION_BIAS_X_START_FAIL"); failures=failures+1; end
            begin_case(0);
            qzeros=32'hx; scale_f16=16'hx; meta_valid=1'bx; xz_probes=xz_probes+1;
            @(posedge clk); @(negedge clk); meta_valid=0;
            if (meta_channel!==0) begin $display("PROJECTION_BIAS_X_META_FAIL"); failures=failures+1; end
            send_meta(0,0);
            qweight=32'hz; activation_f16=16'hx; pair_valid=1'bx; xz_probes=xz_probes+1;
            @(posedge clk); @(negedge clk); pair_valid=0;
            if (pair_input!==0) begin $display("PROJECTION_BIAS_X_PAIR_FAIL"); failures=failures+1; end
            send_pair(0,0,0);
            clear=1; @(posedge clk); @(negedge clk); clear=0;
`endif
        end
    endtask

    initial begin
        failures=0; cycles=0; stalls=0; xz_probes=0; clk=0; rst_n=1; idle;
        if (!$value$plusargs("VECTOR_DIR=%s",vector_dir))
            vector_dir="build/projection_bias_vectors";
        path={vector_dir,"/cases.hex"}; $readmemh(path,cases_mem);
        path={vector_dir,"/meta.hex"}; $readmemh(path,meta_mem);
        path={vector_dir,"/pairs.hex"}; $readmemh(path,pairs_mem);
        path={vector_dir,"/expected.hex"}; $readmemh(path,expected_mem);
        reset_dut;
        protocol_probes;

        /* Clear while a real transaction is waiting for its bias, then restart it. */
        begin_case(0); send_meta(0,1);
        for (pair_no=0; pair_no<PAIRS_PER_CASE; pair_no=pair_no+1) send_pair(0,pair_no,(pair_no==17));
        while (bias_ready!==1) @(negedge clk);
        clear=1; @(posedge clk); @(negedge clk); clear=0;
        #1;
        if (busy || out_valid) begin
            $display("PROJECTION_BIAS_CLEAR_PENDING_FAIL"); failures=failures+1;
        end

        for (case_no=0; case_no<CASES; case_no=case_no+1) begin
            begin_case(case_no); send_meta(case_no,(case_no%2));
            for (pair_no=0; pair_no<PAIRS_PER_CASE; pair_no=pair_no+1)
                send_pair(case_no,pair_no,((pair_no==11)||(pair_no==93)));
            send_bias(case_no,(case_no%2));
            check_output(case_no);
        end
        if (expected_mem[0][125:110]===16'h0001) begin end
        else begin $display("PROJECTION_BIAS_POST_ROUND_RESULT_FAIL"); failures=failures+1; end
        if (16'h0000===expected_mem[0][125:110]) begin
            $display("PROJECTION_BIAS_PRE_ROUND_HYPOTHETICAL_NOT_DISTINCT_FAIL"); failures=failures+1;
        end
        if (failures==0) begin
            $display("PROJECTION_BIAS_PASS cases=5 pairs=640 cycles=%0d stalls=%0d xz_probes=%0d post_round=pass",
                cycles,stalls,xz_probes);
            $finish;
        end
        $fatal(1,"PROJECTION_BIAS_FAIL failures=%0d",failures);
    end
endmodule

`default_nettype wire
