`timescale 1ns/1ps
`default_nettype none

module ace3_decoder_preload_tb;
    localparam integer HIDDEN = 896;
    localparam integer MAX_WAIT = 8;
    reg clk=0, rst_n=1, clear=0;
    reg load_valid=0, start_valid=0;
    reg [1:0] load_kind=0, start_slot=0;
    reg [12:0] load_index=0;
    reg [14:0] start_position=0;
    reg [15:0] load_f16=0;
    wire load_ready, start_ready, busy;
    wire [5:0] phase;
    wire trace_valid, final_valid, done_valid;
    integer accepted [0:2];
    integer attempted_kind, attempted_index, accepted_kind, accepted_index;
    integer wait_cycles, i;

    always #5 clk=~clk;

    ace3_decoder_layer0_token_engine dut (
        .clk_i(clk),.rst_ni(rst_n),.clear_i(clear),
        .load_valid_i(load_valid),.load_ready_o(load_ready),.load_kind_i(load_kind),
        .load_index_i(load_index),.load_f16_i(load_f16),
        .start_valid_i(start_valid),.start_ready_o(start_ready),
        .start_cache_slot_i(start_slot),.start_position_i(start_position),
        .projection_meta_valid_i(1'b0),.projection_qzeros_i(32'd0),
        .projection_scale_f16_i(16'd0),.projection_pair_valid_i(1'b0),
        .projection_qweight_i(32'd0),.projection_bias_valid_i(1'b0),
        .projection_bias_f16_i(16'd0),.rope_valid_i(1'b0),
        .rope_cos_f16_i(16'd0),.rope_sin_f16_i(16'd0),
        .trace_valid_o(trace_valid),.trace_ready_i(1'b1),
        .final_valid_o(final_valid),.final_ready_i(1'b1),
        .done_valid_o(done_valid),.done_ready_i(1'b1),
        .busy_o(busy),.phase_o(phase)
    );

    task automatic reset_dut;
        begin
            load_valid=0; start_valid=0; clear=0; rst_n=0; #1;
            if (load_ready!==0 || start_ready!==0 || busy!==0 || phase!==0)
                $fatal(1,"DECODER_PRELOAD_RESET_FAIL ready=%b start=%b busy=%b phase=%0d",
                    load_ready,start_ready,busy,phase);
            repeat(2) @(posedge clk);
            @(negedge clk); rst_n=1; #1;
            if (phase!==0 || start_ready!==0 || dut.load_n1_q!==0 ||
                dut.load_n2_q!==0 || dut.load_act_q!==0)
                $fatal(1,"DECODER_PRELOAD_RESET_STATE_FAIL");
        end
    endtask

    task automatic reject_load(input [1:0] kind, input [12:0] index);
        begin
            @(negedge clk); load_kind=kind; load_index=index;
            load_f16=16'h1234; load_valid=1; #1;
            if (load_ready!==0)
                $fatal(1,"DECODER_PRELOAD_REJECT_FAIL kind=%0d index=%0d ready=%b",
                    kind,index,load_ready);
            @(posedge clk); #1; load_valid=0;
        end
    endtask

    task automatic accept_load(input [1:0] kind, input [12:0] index);
        begin
            @(negedge clk); load_kind=kind; load_index=index;
            load_f16={kind,1'b0,index}; load_valid=1;
            attempted_kind=kind; attempted_index=index; #1; wait_cycles=0;
            while (load_ready!==1) begin
                if (load_ready!==0)
                    $fatal(1,"DECODER_PRELOAD_READY_XZ kind=%0d index=%0d ready=%b",
                        kind,index,load_ready);
                if (wait_cycles>=MAX_WAIT)
                    $fatal(1,"DECODER_PRELOAD_TIMEOUT kind=%0d index=%0d ready=%b phase=%0d accepts=%0d,%0d,%0d",
                        kind,index,load_ready,phase,accepted[0],accepted[1],accepted[2]);
                wait_cycles=wait_cycles+1; @(negedge clk); #1;
            end
            if (phase!==0 || busy!==0)
                $fatal(1,"DECODER_PRELOAD_NONIDLE_ACCEPT kind=%0d index=%0d phase=%0d",
                    kind,index,phase);
            @(posedge clk); #1;
            accepted[kind]=accepted[kind]+1;
            accepted_kind=kind; accepted_index=index; load_valid=0;
        end
    endtask

    task automatic load_kind_all(input [1:0] kind);
        begin
            for (i=0;i<HIDDEN;i=i+1) begin
                if (i==895) begin
                    if (kind==0 && dut.load_act_q!==895) $fatal(1,"DECODER_PRELOAD_FINAL_INDEX_ACT_FAIL");
                    if (kind==1 && dut.load_n1_q!==895) $fatal(1,"DECODER_PRELOAD_FINAL_INDEX_N1_FAIL");
                    if (kind==2 && dut.load_n2_q!==895) $fatal(1,"DECODER_PRELOAD_FINAL_INDEX_N2_FAIL");
                end
                accept_load(kind,i[12:0]);
            end
        end
    endtask

    task automatic accept_start;
        begin
            @(negedge clk); start_valid=1; #1;
            if (start_ready!==1 || phase!==0 || busy!==0)
                $fatal(1,"DECODER_PRELOAD_START_PREREQ_FAIL ready=%b phase=%0d busy=%b",
                    start_ready,phase,busy);
            @(posedge clk); #1; start_valid=0;
            if (phase!==1 || busy!==1)
                $fatal(1,"DECODER_PRELOAD_VACUOUS_START phase=%0d busy=%b",phase,busy);
        end
    endtask

    task automatic clear_dut;
        begin
            @(negedge clk); clear=1; @(posedge clk); #1; clear=0;
            if (phase!==0 || busy!==0 || start_ready!==0 ||
                dut.activation_loaded_q!==0 || dut.n1_loaded_q!==0 ||
                dut.n2_loaded_q!==0 || dut.load_act_q!==0 ||
                dut.load_n1_q!==0 || dut.load_n2_q!==0)
                $fatal(1,"DECODER_PRELOAD_CLEAR_FAIL phase=%0d busy=%b start_ready=%b flags=%b%b%b",
                    phase,busy,start_ready,dut.activation_loaded_q,dut.n1_loaded_q,dut.n2_loaded_q);
        end
    endtask

    initial begin
        accepted[0]=0; accepted[1]=0; accepted[2]=0;
        attempted_kind=0; attempted_index=0; accepted_kind=0; accepted_index=0;
        reset_dut;
        if ($test$plusargs("EXPECT_TIMEOUT")) begin
            @(negedge clk); load_kind=1; load_index=1; load_f16=16'h1234; load_valid=1; #1;
            for (wait_cycles=0;wait_cycles<MAX_WAIT;wait_cycles=wait_cycles+1) begin
                if (load_ready!==0)
                    $fatal(1,"DECODER_PRELOAD_EXPECTED_TIMEOUT_READY ready=%b",load_ready);
                @(negedge clk); #1;
            end
            $fatal(1,"DECODER_PRELOAD_TIMEOUT kind=1 index=1 ready=%b phase=%0d accepts=0,0,0",
                load_ready,phase);
        end

        reject_load(1,1);
        reject_load(3,0);
`ifndef VERILATOR
        @(negedge clk); load_valid=1'bx; load_kind=2'bx; load_index=13'hx;
        load_f16=16'hx; #1;
        if (load_ready!==0) $fatal(1,"DECODER_PRELOAD_XZ_ACCEPTED ready=%b",load_ready);
        @(posedge clk); #1; load_valid=0; load_kind=0; load_index=0; load_f16=0;
`endif
        load_kind_all(1);
        if (dut.n1_loaded_q!==1 || start_ready!==0 || accepted[1]!=896)
            $fatal(1,"DECODER_PRELOAD_N1_COMPLETION_FAIL");
        load_kind_all(2);
        if (dut.n2_loaded_q!==1 || start_ready!==0 || accepted[2]!=896)
            $fatal(1,"DECODER_PRELOAD_N2_COMPLETION_FAIL");
        load_kind_all(0);
        if (dut.activation_loaded_q!==1 || start_ready!==1 || accepted[0]!=896)
            $fatal(1,"DECODER_PRELOAD_ACT_COMPLETION_FAIL");
        accept_start;
        clear_dut;
        load_kind_all(1); load_kind_all(2); load_kind_all(0);
        if (start_ready!==1 || accepted[0]!=1792 ||
            accepted[1]!=1792 || accepted[2]!=1792)
            $fatal(1,"DECODER_PRELOAD_RELOAD_PREREQ_FAIL ready=%b accepts=%0d,%0d,%0d",
                start_ready,accepted[0],accepted[1],accepted[2]);
        accept_start;
        $display("DECODER_PRELOAD_IVERILOG_PASS accepts=%0d,%0d,%0d epoch_handshakes=2688 total_handshakes=5376 epochs=2 last_kind=%0d last_index=%0d reset=pass sequence=1,2,0 flags=pass ready_known=pass final_index=895 start_transition=0_to_1 clear_reload=pass",
            accepted[0],accepted[1],accepted[2],accepted_kind,accepted_index);
        $finish;
    end
endmodule

`default_nettype wire
