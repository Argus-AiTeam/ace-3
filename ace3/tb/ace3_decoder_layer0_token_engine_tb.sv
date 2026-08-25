`timescale 1ns/1ps
`default_nettype none

module ace3_decoder_layer0_token_engine_tb;
    localparam integer HIDDEN=896, INTERMEDIATE=4864, TRACE_RECORDS=46676, FINAL_RECORDS=1792;
    reg clk, rst_n, clear, load_valid, start_valid;
    reg [1:0] load_kind, start_slot;
    reg [12:0] load_index;
    reg [15:0] load_f16;
    wire load_ready, start_ready;
    wire [2:0] projection_kind;
    reg projection_meta_valid, projection_pair_valid, projection_bias_valid, rope_valid;
    reg [31:0] projection_qzeros, projection_qweight;
    reg [15:0] projection_scale, projection_bias, rope_cos, rope_sin;
    wire projection_meta_ready, projection_pair_ready, projection_bias_ready, rope_ready;
    wire [12:0] projection_meta_output, projection_pair_input, projection_pair_output, projection_bias_output;
    wire [5:0] projection_meta_group, projection_pair_group;
    wire [9:0] projection_meta_word, projection_pair_word;
    wire [2:0] projection_meta_lane, projection_pair_lane;
    wire [14:0] rope_position;
    wire [4:0] rope_pair;
    wire trace_valid, final_valid, final_last, done_valid, busy;
    reg trace_ready, final_ready, done_ready;
    wire [4:0] trace_stage;
    wire [12:0] trace_index, final_index;
    wire [15:0] trace_f16, final_f16;
    wire [14:0] trace_position, done_position;
    wire [1:0] done_slot;
    wire [31:0] done_cycles, done_stalls;
    wire [5:0] phase;

    reg [39:0] inputs [0:1791], finals [0:FINAL_RECORDS-1];
    reg [63:0] trace_expected [0:TRACE_RECORDS-1];
    reg [47:0] rope_mem [0:63];
    reg [15:0] norm1 [0:HIDDEN-1], norm2 [0:HIDDEN-1];
    reg [31:0] q_qweight [0:100351], q_qzeros [0:783], o_qweight [0:100351], o_qzeros [0:783];
    reg [31:0] k_qweight [0:14335], k_qzeros [0:111], v_qweight [0:14335], v_qzeros [0:111];
    reg [31:0] gate_qweight [0:544767], gate_qzeros [0:4255], up_qweight [0:544767], up_qzeros [0:4255];
    reg [31:0] down_qweight [0:544767], down_qzeros [0:4255];
    reg [15:0] q_scales [0:6271], k_scales [0:895], v_scales [0:895], o_scales [0:6271];
    reg [15:0] gate_scales [0:34047], up_scales [0:34047], down_scales [0:34047];
    reg [15:0] q_bias [0:895], k_bias [0:127], v_bias [0:127];
    integer cycles, external_stalls, failures, trace_count, final_count, done_count, load_count;
    integer preload_accepts [0:2], preload_wait_cycles, progress_interval;
    integer attempted_load_kind, attempted_load_index, accepted_load_kind, accepted_load_index;
    integer start_attempts, start_accepts, start_progress_deadline;
    integer phase_cycles [0:36];
    integer token_start_cycle [0:1], token_done_cycle [0:1], active_token;
    reg checking, xz_payload_mode, data_loaded;
    integer i, linear, qwords;
    string vector_dir, path;
    reg trace_hold, final_hold, done_hold;
    reg [4:0] trace_stage_hold;
    reg [12:0] trace_index_hold, final_index_hold;
    reg [15:0] trace_f16_hold, final_f16_hold;
    reg [14:0] trace_pos_hold, done_pos_hold;
    reg final_last_hold;
    reg [1:0] done_slot_hold;
    reg [31:0] done_cycles_hold, done_stalls_hold;
    wire qzeros_address_valid;
    wire [15:0] qzeros_address;

    ace3_decoder_qzeros_address qzeros_address_check (
        .meta_live_i(projection_meta_valid && projection_meta_ready),
        .projection_kind_i(projection_kind),
        .group_i(projection_meta_group),
        .word_i(projection_meta_word),
        .address_valid_o(qzeros_address_valid),
        .address_o(qzeros_address)
    );

    ace3_decoder_layer0_token_engine dut (
        .clk_i(clk),.rst_ni(rst_n),.clear_i(clear),
        .load_valid_i(load_valid),.load_ready_o(load_ready),.load_kind_i(load_kind),
        .load_index_i(load_index),.load_f16_i(load_f16),
        .start_valid_i(start_valid),.start_ready_o(start_ready),
        .start_cache_slot_i(start_slot),.start_position_i(start_position),
        .projection_kind_o(projection_kind),.projection_meta_valid_i(projection_meta_valid),
        .projection_meta_ready_o(projection_meta_ready),.projection_meta_output_channel_o(projection_meta_output),
        .projection_meta_group_o(projection_meta_group),.projection_meta_word_o(projection_meta_word),
        .projection_meta_lane_o(projection_meta_lane),.projection_qzeros_i(projection_qzeros),
        .projection_scale_f16_i(projection_scale),.projection_pair_valid_i(projection_pair_valid),
        .projection_pair_ready_o(projection_pair_ready),.projection_pair_input_o(projection_pair_input),
        .projection_pair_output_o(projection_pair_output),.projection_pair_group_o(projection_pair_group),
        .projection_pair_word_o(projection_pair_word),.projection_pair_lane_o(projection_pair_lane),
        .projection_qweight_i(projection_qweight),.projection_bias_valid_i(projection_bias_valid),
        .projection_bias_ready_o(projection_bias_ready),.projection_bias_output_channel_o(projection_bias_output),
        .projection_bias_f16_i(projection_bias),.rope_valid_i(rope_valid),.rope_ready_o(rope_ready),
        .rope_position_o(rope_position),.rope_pair_o(rope_pair),.rope_cos_f16_i(rope_cos),
        .rope_sin_f16_i(rope_sin),.trace_valid_o(trace_valid),.trace_ready_i(trace_ready),
        .trace_stage_o(trace_stage),.trace_index_o(trace_index),.trace_f16_o(trace_f16),
        .trace_position_o(trace_position),.final_valid_o(final_valid),.final_ready_i(final_ready),
        .final_index_o(final_index),.final_f16_o(final_f16),.final_last_o(final_last),
        .done_valid_o(done_valid),.done_ready_i(done_ready),.done_cache_slot_o(done_slot),
        .done_position_o(done_position),.done_cycles_o(done_cycles),.done_stall_cycles_o(done_stalls),
        .busy_o(busy),.phase_o(phase)
    );
    reg [14:0] start_position;
    always #5 clk=~clk;

    /*
     * Tensor memories are immutable after $readmemh.  A manual sensitivity
     * list avoids making Icarus rescan its 1.6M-word tensor arrays each cycle.
     */
    always @(projection_kind or projection_meta_group or projection_meta_word or
             projection_meta_output or projection_pair_input or projection_pair_word or
             projection_bias_output or rope_position or rope_pair or cycles or
             projection_meta_ready or projection_pair_ready or projection_bias_ready or
             rope_ready or qzeros_address_valid or qzeros_address or
             xz_payload_mode or data_loaded) begin
        projection_meta_valid = (cycles % 23) != 0;
        projection_pair_valid = (cycles % 71) != 0;
        projection_bias_valid = (cycles % 37) != 0;
        rope_valid = (cycles % 29) != 0;
        trace_ready = (cycles % 17) != 0;
        final_ready = (cycles % 19) != 0;
        done_ready = (cycles % 13) != 0;
        projection_qzeros=0; projection_scale=0; projection_qweight=0; projection_bias=0;
        rope_cos=0; rope_sin=0;
        if (projection_meta_valid && projection_meta_ready) begin
            if (!qzeros_address_valid)
                $fatal(1,"DECODER_LIVE_QZEROS_ADDRESS_FAIL kind=%0d group=%0d word=%0d",
                       projection_kind,projection_meta_group,projection_meta_word);
            case (projection_kind)
              0: begin projection_qzeros=q_qzeros[qzeros_address];
                       if (projection_meta_output>=896) $fatal(1,"DECODER_Q_SCALE_ADDRESS_FAIL");
                       projection_scale=q_scales[projection_meta_group*896+projection_meta_output]; end
              1: begin projection_qzeros=k_qzeros[qzeros_address];
                       if (projection_meta_output>=128) $fatal(1,"DECODER_K_SCALE_ADDRESS_FAIL");
                       projection_scale=k_scales[projection_meta_group*128+projection_meta_output]; end
              2: begin projection_qzeros=v_qzeros[qzeros_address];
                       if (projection_meta_output>=128) $fatal(1,"DECODER_V_SCALE_ADDRESS_FAIL");
                       projection_scale=v_scales[projection_meta_group*128+projection_meta_output]; end
              3: begin projection_qzeros=o_qzeros[qzeros_address];
                       if (projection_meta_output>=896) $fatal(1,"DECODER_O_SCALE_ADDRESS_FAIL");
                       projection_scale=o_scales[projection_meta_group*896+projection_meta_output]; end
              4: begin projection_qzeros=gate_qzeros[qzeros_address];
                       if (projection_meta_output>=4864) $fatal(1,"DECODER_GATE_SCALE_ADDRESS_FAIL");
                       projection_scale=gate_scales[projection_meta_group*4864+projection_meta_output]; end
              5: begin projection_qzeros=up_qzeros[qzeros_address];
                       if (projection_meta_output>=4864) $fatal(1,"DECODER_UP_SCALE_ADDRESS_FAIL");
                       projection_scale=up_scales[projection_meta_group*4864+projection_meta_output]; end
              6: begin projection_qzeros=down_qzeros[qzeros_address];
                       if (projection_meta_output>=896) $fatal(1,"DECODER_DOWN_SCALE_ADDRESS_FAIL");
                       projection_scale=down_scales[projection_meta_group*896+projection_meta_output]; end
              default: $fatal(1,"DECODER_META_KIND_FAIL kind=%0d",projection_kind);
            endcase
        end
        if (projection_pair_valid && projection_pair_ready) begin
            case (projection_kind)
              0,3: begin
                  if (projection_pair_input>=896 || projection_pair_word>=112)
                      $fatal(1,"DECODER_QO_QWEIGHT_ADDRESS_FAIL");
                  if (projection_kind==0) projection_qweight=q_qweight[projection_pair_input*112+projection_pair_word];
                  else projection_qweight=o_qweight[projection_pair_input*112+projection_pair_word];
              end
              1,2: begin
                  if (projection_pair_input>=896 || projection_pair_word>=16)
                      $fatal(1,"DECODER_KV_QWEIGHT_ADDRESS_FAIL");
                  if (projection_kind==1) projection_qweight=k_qweight[projection_pair_input*16+projection_pair_word];
                  else projection_qweight=v_qweight[projection_pair_input*16+projection_pair_word];
              end
              4,5: begin
                  if (projection_pair_input>=896 || projection_pair_word>=608)
                      $fatal(1,"DECODER_FFN_QWEIGHT_ADDRESS_FAIL");
                  if (projection_kind==4) projection_qweight=gate_qweight[projection_pair_input*608+projection_pair_word];
                  else projection_qweight=up_qweight[projection_pair_input*608+projection_pair_word];
              end
              6: begin
                  if (projection_pair_input>=4864 || projection_pair_word>=112)
                      $fatal(1,"DECODER_DOWN_QWEIGHT_ADDRESS_FAIL");
                  projection_qweight=down_qweight[projection_pair_input*112+projection_pair_word];
              end
              default: $fatal(1,"DECODER_PAIR_KIND_FAIL kind=%0d",projection_kind);
            endcase
        end
        if (projection_bias_valid && projection_bias_ready) begin
            case (projection_kind)
              0: begin if (projection_bias_output>=896) $fatal(1,"DECODER_Q_BIAS_ADDRESS_FAIL");
                       projection_bias=q_bias[projection_bias_output]; end
              1: begin if (projection_bias_output>=128) $fatal(1,"DECODER_K_BIAS_ADDRESS_FAIL");
                       projection_bias=k_bias[projection_bias_output]; end
              2: begin if (projection_bias_output>=128) $fatal(1,"DECODER_V_BIAS_ADDRESS_FAIL");
                       projection_bias=v_bias[projection_bias_output]; end
              default: $fatal(1,"DECODER_BIAS_KIND_FAIL kind=%0d",projection_kind);
            endcase
        end
        if (rope_valid && rope_ready) begin
            linear=rope_position*32+rope_pair;
            if (linear>=64) $fatal(1,"DECODER_ROPE_ADDRESS_FAIL address=%0d",linear);
            rope_cos=rope_mem[linear][31:16];
            rope_sin=rope_mem[linear][15:0];
        end
`ifndef VERILATOR
        if (xz_payload_mode) begin
            projection_meta_valid=1'bx; projection_pair_valid=1'bz; projection_bias_valid=1'bx;
            rope_valid=1'bz; projection_qzeros=32'hx; projection_scale=16'hz;
            projection_qweight=32'hx; projection_bias=16'hz; rope_cos=16'hx; rope_sin=16'hz;
        end
`endif
    end

    always @(posedge clk) begin
        cycles<=cycles+1;
        if (progress_interval>0 && cycles>0 && (cycles%progress_interval)==0)
            $display("DECODER_PROGRESS cycles=%0d phase=%0d projection_kind=%0d load_ready=%b attempted_load_kind=%0d attempted_load_index=%0d accepted_load_kind=%0d accepted_load_index=%0d load_accepts=%0d,%0d,%0d start_attempts=%0d start_accepts=%0d trace=%0d final=%0d done=%0d",
                cycles,phase,projection_kind,load_ready,attempted_load_kind,attempted_load_index,
                accepted_load_kind,accepted_load_index,preload_accepts[0],preload_accepts[1],
                preload_accepts[2],start_attempts,start_accepts,trace_count,final_count,done_count);
        if (busy && phase<=36) phase_cycles[phase]<=phase_cycles[phase]+1;
        if ((projection_meta_ready&&!projection_meta_valid) || (projection_pair_ready&&!projection_pair_valid) ||
            (projection_bias_ready&&!projection_bias_valid) || (rope_ready&&!rope_valid) ||
            (trace_valid&&!trace_ready) || (final_valid&&(!final_ready||!trace_ready)) ||
            (done_valid&&!done_ready)) external_stalls<=external_stalls+1;
        if (checking && trace_valid && trace_ready &&
            (!final_valid || final_ready)) begin
            if (trace_count>=TRACE_RECORDS ||
                trace_expected[trace_count] !== {active_token[7:0],1'b0,trace_position,
                                                  3'b0,trace_stage,3'b0,trace_index,trace_f16}) begin
                $display("DECODER_TRACE_MISMATCH count=%0d token=%0d pos=%0d stage=%0d index=%0d f16=%04x",
                    trace_count,active_token,trace_position,trace_stage,trace_index,trace_f16); failures<=failures+1;
            end
            trace_count<=trace_count+1;
        end
        if (checking && final_valid && final_ready && trace_ready) begin
            if (final_count>=FINAL_RECORDS ||
                finals[final_count] !== {8'(active_token),3'b0,final_index,final_f16}) begin
                $display("DECODER_FINAL_MISMATCH count=%0d token=%0d index=%0d f16=%04x",
                    final_count,active_token,final_index,final_f16); failures<=failures+1;
            end
            if (final_last !== (final_index==895)) begin $display("DECODER_FINAL_LAST_FAIL"); failures<=failures+1; end
            final_count<=final_count+1;
        end
        if (done_valid && done_ready && checking) begin
            done_count<=done_count+1; token_done_cycle[active_token]<=cycles;
            if (done_slot!==0 || done_position!==active_token || done_cycles==0 ||
                done_cycles<=done_stalls) begin
                $display("DECODER_DONE_METADATA_FAIL slot=%0d pos=%0d cycles=%0d stalls=%0d",
                    done_slot,done_position,done_cycles,done_stalls); failures<=failures+1;
            end
        end
        if (trace_hold && (!trace_valid || trace_stage!==trace_stage_hold || trace_index!==trace_index_hold ||
            trace_f16!==trace_f16_hold || trace_position!==trace_pos_hold)) begin
            $display("DECODER_TRACE_STABILITY_FAIL"); failures<=failures+1;
        end
        if (final_hold && (!final_valid || final_index!==final_index_hold || final_f16!==final_f16_hold ||
            final_last!==final_last_hold)) begin $display("DECODER_FINAL_STABILITY_FAIL"); failures<=failures+1; end
        if (done_hold && (!done_valid || done_slot!==done_slot_hold || done_position!==done_pos_hold ||
            done_cycles!==done_cycles_hold || done_stalls!==done_stalls_hold)) begin
            $display("DECODER_DONE_STABILITY_FAIL"); failures<=failures+1;
        end
        trace_hold<=trace_valid&&!trace_ready; trace_stage_hold<=trace_stage; trace_index_hold<=trace_index;
        trace_f16_hold<=trace_f16; trace_pos_hold<=trace_position;
        final_hold<=final_valid&&(!final_ready||!trace_ready); final_index_hold<=final_index; final_f16_hold<=final_f16; final_last_hold<=final_last;
        done_hold<=done_valid&&!done_ready; done_slot_hold<=done_slot; done_pos_hold<=done_position;
        done_cycles_hold<=done_cycles; done_stalls_hold<=done_stalls;
    end

    task drive_idle;
        begin clear=0; load_valid=0; load_kind=0; load_index=0; load_f16=0; start_valid=0; start_slot=0; start_position=0; end
    endtask
    task reset_dut;
        begin
            drive_idle; rst_n=0; #2;
            if (load_ready||start_ready||busy||done_valid) begin $display("DECODER_ASYNC_RESET_FAIL"); failures=failures+1; end
            repeat(2) @(posedge clk); @(negedge clk); rst_n=1;
        end
    endtask
    task load_vector;
        input [1:0] kind;
        input integer token;
        begin
            for (i=0;i<HIDDEN;i=i+1) begin
                @(negedge clk); load_kind=kind; load_index=i;
                if (kind==0) load_f16=inputs[token*HIDDEN+i][15:0];
                else if (kind==1) load_f16=norm1[i]; else load_f16=norm2[i];
                attempted_load_kind=kind; attempted_load_index=i;
                load_valid=1; #1; preload_wait_cycles=0;
                while(load_ready!==1) begin
                    if (load_ready!==0)
                        $fatal(1,"DECODER_PRELOAD_READY_XZ kind=%0d index=%0d ready=%b phase=%0d",kind,i,load_ready,phase);
                    if (preload_wait_cycles>=32)
                        $fatal(1,"DECODER_PRELOAD_TIMEOUT kind=%0d index=%0d ready=%b phase=%0d accepts=%0d,%0d,%0d",
                            kind,i,load_ready,phase,preload_accepts[0],preload_accepts[1],preload_accepts[2]);
                    preload_wait_cycles=preload_wait_cycles+1; @(negedge clk); #1;
                end
                @(posedge clk); load_count=load_count+1; preload_accepts[kind]=preload_accepts[kind]+1;
                accepted_load_kind=kind; accepted_load_index=i;
                @(negedge clk); load_valid=0;
            end
        end
    endtask
    task start_token;
        input [1:0] slot;
        input [14:0] pos;
        input integer token;
        begin
            start_attempts=start_attempts+1;
            @(negedge clk); start_slot=slot; start_position=pos; start_valid=1;
            #1;
            if (start_ready!==1)
                $fatal(1,"DECODER_START_REJECTED slot=%0d pos=%0d phase=%0d load_accepts=%0d,%0d,%0d",
                    slot,pos,phase,preload_accepts[0],preload_accepts[1],preload_accepts[2]);
            @(posedge clk); token_start_cycle[token]=cycles; active_token=token; start_accepts=start_accepts+1;
            #1;
            if (!busy || phase==0) $fatal(1,"DECODER_VACUOUS_START phase=%0d busy=%b",phase,busy);
            @(negedge clk); start_valid=0;
        end
    endtask
    task wait_real_work_then_clear;
        begin
            start_progress_deadline=cycles+100000;
            while (!(busy && (projection_meta_ready||projection_pair_ready||rope_ready))) begin
                if (cycles>=start_progress_deadline)
                    $fatal(1,"DECODER_START_PROGRESS_TIMEOUT phase=%0d start_accepts=%0d trace=%0d final=%0d done=%0d",
                        phase,start_accepts,trace_count,final_count,done_count);
                @(negedge clk);
            end
            clear=1; @(posedge clk); @(negedge clk); clear=0;
            if (busy||done_valid||start_ready) begin
                $display("DECODER_CLEAR_ABORT_FAIL"); failures=failures+1;
            end
        end
    endtask

    initial begin
        cycles=0; external_stalls=0; failures=0; trace_count=0; final_count=0; done_count=0; load_count=0;
        preload_accepts[0]=0; preload_accepts[1]=0; preload_accepts[2]=0;
        attempted_load_kind=0; attempted_load_index=0; accepted_load_kind=0; accepted_load_index=0;
        preload_wait_cycles=0; progress_interval=0; start_attempts=0; start_accepts=0;
        trace_hold=0; final_hold=0; done_hold=0; active_token=0; checking=0; xz_payload_mode=0; data_loaded=0; clk=0; rst_n=1; drive_idle;
        for(i=0;i<=36;i=i+1) phase_cycles[i]=0;
        if (!$value$plusargs("PROGRESS_INTERVAL=%d",progress_interval)) progress_interval=1000000;
        if (!$value$plusargs("VECTOR_DIR=%s",vector_dir)) vector_dir="build/decoder_layer0_vectors";
        path={vector_dir,"/inputs.hex"}; $readmemh(path,inputs);
        path={vector_dir,"/trace.hex"}; $readmemh(path,trace_expected);
        path={vector_dir,"/final.hex"}; $readmemh(path,finals);
        path={vector_dir,"/rope_coefficients.hex"}; $readmemh(path,rope_mem);
        path={vector_dir,"/tensors/layer0_input_layernorm_weight.fp16le.bin.hex"}; $readmemh(path,norm1);
        path={vector_dir,"/tensors/layer0_post_attention_layernorm_weight.fp16le.bin.hex"}; $readmemh(path,norm2);
        path={vector_dir,"/tensors/layer0_self_attn_q_proj_qweight.i32le.bin.hex"}; $readmemh(path,q_qweight);
        path={vector_dir,"/tensors/layer0_self_attn_q_proj_qzeros.i32le.bin.hex"}; $readmemh(path,q_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_q_proj_scales.fp16le.bin.hex"}; $readmemh(path,q_scales);
        path={vector_dir,"/tensors/layer0_self_attn_q_proj_bias.fp16le.bin.hex"}; $readmemh(path,q_bias);
        path={vector_dir,"/tensors/layer0_self_attn_k_proj_qweight.i32le.bin.hex"}; $readmemh(path,k_qweight);
        path={vector_dir,"/tensors/layer0_self_attn_k_proj_qzeros.i32le.bin.hex"}; $readmemh(path,k_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_k_proj_scales.fp16le.bin.hex"}; $readmemh(path,k_scales);
        path={vector_dir,"/tensors/layer0_self_attn_k_proj_bias.fp16le.bin.hex"}; $readmemh(path,k_bias);
        path={vector_dir,"/tensors/layer0_self_attn_v_proj_qweight.i32le.bin.hex"}; $readmemh(path,v_qweight);
        path={vector_dir,"/tensors/layer0_self_attn_v_proj_qzeros.i32le.bin.hex"}; $readmemh(path,v_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_v_proj_scales.fp16le.bin.hex"}; $readmemh(path,v_scales);
        path={vector_dir,"/tensors/layer0_self_attn_v_proj_bias.fp16le.bin.hex"}; $readmemh(path,v_bias);
        path={vector_dir,"/tensors/layer0_self_attn_o_proj_qweight.i32le.bin.hex"}; $readmemh(path,o_qweight);
        path={vector_dir,"/tensors/layer0_self_attn_o_proj_qzeros.i32le.bin.hex"}; $readmemh(path,o_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_o_proj_scales.fp16le.bin.hex"}; $readmemh(path,o_scales);
        path={vector_dir,"/tensors/layer0_mlp_gate_proj_qweight.i32le.bin.hex"}; $readmemh(path,gate_qweight);
        path={vector_dir,"/tensors/layer0_mlp_gate_proj_qzeros.i32le.bin.hex"}; $readmemh(path,gate_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_gate_proj_scales.fp16le.bin.hex"}; $readmemh(path,gate_scales);
        path={vector_dir,"/tensors/layer0_mlp_up_proj_qweight.i32le.bin.hex"}; $readmemh(path,up_qweight);
        path={vector_dir,"/tensors/layer0_mlp_up_proj_qzeros.i32le.bin.hex"}; $readmemh(path,up_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_up_proj_scales.fp16le.bin.hex"}; $readmemh(path,up_scales);
        path={vector_dir,"/tensors/layer0_mlp_down_proj_qweight.i32le.bin.hex"}; $readmemh(path,down_qweight);
        path={vector_dir,"/tensors/layer0_mlp_down_proj_qzeros.i32le.bin.hex"}; $readmemh(path,down_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_down_proj_scales.fp16le.bin.hex"}; $readmemh(path,down_scales);
        data_loaded=1;
        reset_dut;
        @(negedge clk); start_valid=1; #1; if(start_ready!==0) begin $display("DECODER_INCOMPLETE_LOAD_ACCEPTED"); failures=failures+1; end start_valid=0;
`ifndef VERILATOR
        @(negedge clk); load_valid=1'bx; load_kind=2'bx; load_index=13'hx; load_f16=16'hx;
        @(posedge clk); @(negedge clk); load_valid=0;
        if(load_ready!==0) begin $display("DECODER_X_LOAD_ACCEPTED"); failures=failures+1; end
        start_valid=1'bx; start_slot=2'bx; start_position=15'hx; @(posedge clk); @(negedge clk); start_valid=0;
`endif
        /* Abort after actual projection work and prove synchronous clear revokes loaded state. */
        load_vector(1,0); load_vector(2,0); load_vector(0,0);
        @(negedge clk); start_slot=2; start_position=0; start_valid=1; #1;
        if(start_ready!==0) begin $display("DECODER_INVALID_SLOT_ACCEPTED"); failures=failures+1; end
        start_slot=0; start_position=1; #1;
        if(start_ready!==0) begin $display("DECODER_NONSEQUENTIAL_START_ACCEPTED"); failures=failures+1; end
        start_valid=0;
        start_token(0,0,0);
`ifndef VERILATOR
        while (!projection_meta_ready) @(negedge clk);
        xz_payload_mode=1; @(posedge clk); @(negedge clk); xz_payload_mode=0;
`endif
        wait_real_work_then_clear;
        /*
         * Populate slot zero, then prove a position-zero transaction in slot one
         * is independently accepted.  Clear aborts that slot-one transaction, so
         * the authenticated comparison below begins from a clean cache.
         */
        load_vector(1,0); load_vector(2,0); load_vector(0,0); start_token(0,0,0);
        while(busy) begin @(negedge clk); if(cycles>100000000) $fatal(1,"DECODER_TIMEOUT isolation-token0"); end
        load_vector(0,1); start_token(1,0,1); wait_real_work_then_clear;
        /* A full clean reload is required after the cache-isolation abort. */
        load_vector(1,0); load_vector(2,0); load_vector(0,0); checking=1; start_token(0,0,0);
        while(done_count<1) begin @(negedge clk); if(cycles>100000000) $fatal(1,"DECODER_TIMEOUT token0"); end
        load_vector(0,1); start_token(0,1,1);
        while(done_count<2) begin @(negedge clk); if(cycles>100000000) $fatal(1,"DECODER_TIMEOUT token1"); end
        if (trace_count!=TRACE_RECORDS || final_count!=FINAL_RECORDS || done_count!=2 ||
            token_done_cycle[0]<=token_start_cycle[0] || token_done_cycle[1]<=token_start_cycle[1]) begin
            $display("DECODER_COUNT_OR_CYCLE_FAIL trace=%0d final=%0d done=%0d",trace_count,final_count,done_count); failures=failures+1;
        end
        if (failures==0) begin
            $display("DECODER_PHASE_CYCLES_0_12 %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3],phase_cycles[4],
                phase_cycles[5],phase_cycles[6],phase_cycles[7],phase_cycles[8],phase_cycles[9],
                phase_cycles[10],phase_cycles[11],phase_cycles[12]);
            $display("DECODER_PHASE_CYCLES_13_24 %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                phase_cycles[13],phase_cycles[14],phase_cycles[15],phase_cycles[16],phase_cycles[17],
                phase_cycles[18],phase_cycles[19],phase_cycles[20],phase_cycles[21],phase_cycles[22],
                phase_cycles[23],phase_cycles[24]);
            $display("DECODER_PHASE_CYCLES_25_36 %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                phase_cycles[25],phase_cycles[26],phase_cycles[27],phase_cycles[28],phase_cycles[29],
                phase_cycles[30],phase_cycles[31],phase_cycles[32],phase_cycles[33],phase_cycles[34],
                phase_cycles[35],phase_cycles[36]);
            $display("DECODER_LAYER0_TOKEN_ENGINE_PASS trace_count=46676 final_count=1792 cycles=%0d stalls=%0d token0_cycles=%0d token1_cycles=%0d phase_p_run=%0d phase_final=%0d",
                cycles,external_stalls,token_done_cycle[0]-token_start_cycle[0],token_done_cycle[1]-token_start_cycle[1],phase_cycles[5],phase_cycles[36]);
            $finish;
        end
        $fatal(1,"DECODER_LAYER0_TOKEN_ENGINE_FAIL failures=%0d",failures);
    end
endmodule

`default_nettype wire
