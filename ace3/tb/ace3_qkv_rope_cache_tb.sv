`timescale 1ns/1ps
`default_nettype none

module ace3_qkv_rope_cache_tb;
    `include "qkv_params.svh"

    reg clk;
    reg rst_n;
    reg clear;

    reg rope_in_valid;
    wire rope_in_ready;
    reg rope_is_key;
    reg [3:0] rope_head;
    reg [4:0] rope_pair;
    reg [14:0] rope_position;
    reg [15:0] rope_low;
    reg [15:0] rope_high;
    reg [15:0] rope_cos;
    reg [15:0] rope_sin;
    wire rope_out_valid;
    reg rope_out_ready;
    wire rope_out_is_key;
    wire [3:0] rope_out_head;
    wire [4:0] rope_out_pair;
    wire [14:0] rope_out_position;
    wire [15:0] rope_out_low;
    wire [15:0] rope_out_high;
    wire rope_invalid;
    wire rope_saturation;

    reg cache_write_valid;
    wire cache_write_ready;
    reg [1:0] cache_write_slot;
    reg [14:0] cache_write_position;
    reg [3:0] cache_write_head;
    reg [5:0] cache_write_dimension;
    reg [15:0] cache_write_k;
    reg [15:0] cache_write_v;
    reg cache_read_valid;
    wire cache_read_ready;
    reg [1:0] cache_read_slot;
    reg [14:0] cache_read_position;
    reg [3:0] cache_read_head;
    reg [5:0] cache_read_dimension;
    wire cache_out_valid;
    reg cache_out_ready;
    wire cache_out_hit;
    wire [1:0] cache_out_slot;
    wire [14:0] cache_out_position;
    wire [3:0] cache_out_head;
    wire [5:0] cache_out_dimension;
    wire [15:0] cache_out_k;
    wire [15:0] cache_out_v;

    reg [127:0] rope_mem [0:`QKV_ROPE_CASES-1];
    reg [63:0] cache_mem [0:`QKV_CACHE_CASES-1];
    integer query_head_seen [0:13];
    integer key_head_seen [0:1];
    integer value_head_seen [0:1];
    integer failures;
    integer rope_outputs;
    integer rope_stalls;
    integer cache_writes;
    integer cache_reads;
    integer cache_stalls;
    integer reset_checks;
    integer clear_checks;
    integer xz_checks;
    integer index;
    string vector_dir;
    string rope_path;
    string cache_path;

    ace3_qwen2_rope_pair rope_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(rope_in_valid),
        .in_ready_o(rope_in_ready),
        .is_key_i(rope_is_key),
        .head_index_i(rope_head),
        .pair_index_i(rope_pair),
        .position_i(rope_position),
        .low_f16_i(rope_low),
        .high_f16_i(rope_high),
        .cos_f16_i(rope_cos),
        .sin_f16_i(rope_sin),
        .out_valid_o(rope_out_valid),
        .out_ready_i(rope_out_ready),
        .is_key_o(rope_out_is_key),
        .head_index_o(rope_out_head),
        .pair_index_o(rope_out_pair),
        .position_o(rope_out_position),
        .low_f16_o(rope_out_low),
        .high_f16_o(rope_out_high),
        .invalid_operand_o(rope_invalid),
        .saturation_o(rope_saturation)
    );

    ace3_fp16_kv_cache #(
        .CACHE_SLOTS(2),
        .MAX_TOKENS(8),
        .KV_HEADS(2),
        .HEAD_DIM(64)
    ) cache_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .write_valid_i(cache_write_valid),
        .write_ready_o(cache_write_ready),
        .write_cache_slot_i(cache_write_slot),
        .write_position_i(cache_write_position),
        .write_head_i(cache_write_head),
        .write_dimension_i(cache_write_dimension),
        .write_k_f16_i(cache_write_k),
        .write_v_f16_i(cache_write_v),
        .read_valid_i(cache_read_valid),
        .read_ready_o(cache_read_ready),
        .read_cache_slot_i(cache_read_slot),
        .read_position_i(cache_read_position),
        .read_head_i(cache_read_head),
        .read_dimension_i(cache_read_dimension),
        .out_valid_o(cache_out_valid),
        .out_ready_i(cache_out_ready),
        .out_hit_o(cache_out_hit),
        .out_cache_slot_o(cache_out_slot),
        .out_position_o(cache_out_position),
        .out_head_o(cache_out_head),
        .out_dimension_o(cache_out_dimension),
        .out_k_f16_o(cache_out_k),
        .out_v_f16_o(cache_out_v)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task drive_idle;
        begin
            clear = 1'b0;
            rope_in_valid = 1'b0;
            rope_is_key = 1'b0;
            rope_head = 4'd0;
            rope_pair = 5'd0;
            rope_position = 15'd0;
            rope_low = 16'd0;
            rope_high = 16'd0;
            rope_cos = 16'h3c00;
            rope_sin = 16'd0;
            rope_out_ready = 1'b0;
            cache_write_valid = 1'b0;
            cache_write_slot = 2'd0;
            cache_write_position = 15'd0;
            cache_write_head = 4'd0;
            cache_write_dimension = 6'd0;
            cache_write_k = 16'd0;
            cache_write_v = 16'd0;
            cache_read_valid = 1'b0;
            cache_read_slot = 2'd0;
            cache_read_position = 15'd0;
            cache_read_head = 4'd0;
            cache_read_dimension = 6'd0;
            cache_out_ready = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            drive_idle();
            rst_n = 1'b0;
            #2;
            if (rope_in_ready || rope_out_valid || cache_write_ready ||
                cache_read_ready || cache_out_valid) begin
                $display("QKV_ASYNC_RESET_FAIL");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            #1;
            if (!rope_in_ready || !cache_write_ready || !cache_read_ready ||
                rope_out_valid || cache_out_valid) begin
                $display("QKV_RESET_RELEASE_FAIL");
                failures = failures + 1;
            end
            reset_checks = reset_checks + 1;
        end
    endtask

    task run_rope_case;
        input [127:0] record;
        input integer case_index;
        reg [47:0] held_data;
        begin
            @(negedge clk);
            rope_is_key = record[98];
            rope_head = record[102:99];
            rope_pair = record[107:103];
            rope_position = record[122:108];
            rope_low = record[15:0];
            rope_high = record[31:16];
            rope_cos = record[47:32];
            rope_sin = record[63:48];
            rope_in_valid = 1'b1;
            #1;
            if (rope_in_ready !== 1'b1) begin
                $display("ROPE_INPUT_NOT_READY case=%0d", case_index);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            rope_in_valid = 1'b0;
            #1;
            if (!rope_out_valid ||
                (rope_out_low !== record[79:64]) ||
                (rope_out_high !== record[95:80]) ||
                (rope_invalid !== record[96]) ||
                (rope_saturation !== record[97]) ||
                (rope_out_is_key !== record[98]) ||
                (rope_out_head !== record[102:99]) ||
                (rope_out_pair !== record[107:103]) ||
                (rope_out_position !== record[122:108])) begin
                $display("ROPE_OUTPUT_MISMATCH case=%0d", case_index);
                failures = failures + 1;
            end
            if ((case_index % 29) == 0) begin
                held_data = {rope_out_low, rope_out_high, rope_invalid,
                             rope_saturation, rope_out_head, rope_out_pair};
                repeat (2) begin
                    @(posedge clk);
                    @(negedge clk);
                    if (!rope_out_valid ||
                        held_data !== {rope_out_low, rope_out_high, rope_invalid,
                                      rope_saturation, rope_out_head,
                                      rope_out_pair}) begin
                        $display("ROPE_BACKPRESSURE_FAIL case=%0d", case_index);
                        failures = failures + 1;
                    end
                    rope_stalls = rope_stalls + 1;
                end
            end
            rope_out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rope_out_ready = 1'b0;
            if (rope_out_valid)
                failures = failures + 1;
            rope_outputs = rope_outputs + 1;
            if (record[98])
                key_head_seen[record[99]] = key_head_seen[record[99]] + 1;
            else
                query_head_seen[record[102:99]] =
                    query_head_seen[record[102:99]] + 1;
        end
    endtask

    task write_cache;
        input [1:0] slot;
        input [14:0] position;
        input [3:0] head;
        input [5:0] dimension;
        input [15:0] k_value;
        input [15:0] v_value;
        begin
            @(negedge clk);
            cache_write_slot = slot;
            cache_write_position = position;
            cache_write_head = head;
            cache_write_dimension = dimension;
            cache_write_k = k_value;
            cache_write_v = v_value;
            cache_write_valid = 1'b1;
            #1;
            if (cache_write_ready !== 1'b1) begin
                $display("CACHE_WRITE_NOT_READY slot=%0d pos=%0d head=%0d dim=%0d",
                         slot, position, head, dimension);
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            cache_write_valid = 1'b0;
            cache_writes = cache_writes + 1;
        end
    endtask

    task read_cache;
        input [1:0] slot;
        input [14:0] position;
        input [3:0] head;
        input [5:0] dimension;
        input expected_hit;
        input [15:0] expected_k;
        input [15:0] expected_v;
        input stall;
        reg [59:0] held_data;
        begin
            @(negedge clk);
            cache_read_slot = slot;
            cache_read_position = position;
            cache_read_head = head;
            cache_read_dimension = dimension;
            cache_read_valid = 1'b1;
            #1;
            if (cache_read_ready !== 1'b1) begin
                $display("CACHE_READ_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            @(negedge clk);
            cache_read_valid = 1'b0;
            #1;
            if (!cache_out_valid ||
                (cache_out_hit !== expected_hit) ||
                (cache_out_slot !== slot) ||
                (cache_out_position !== position) ||
                (cache_out_head !== head) ||
                (cache_out_dimension !== dimension) ||
                (cache_out_k !== expected_k) ||
                (cache_out_v !== expected_v)) begin
                $display("CACHE_READ_MISMATCH slot=%0d pos=%0d head=%0d dim=%0d",
                         slot, position, head, dimension);
                failures = failures + 1;
            end
            if (stall) begin
                held_data = {cache_out_hit, cache_out_slot, cache_out_position,
                             cache_out_head, cache_out_dimension,
                             cache_out_k, cache_out_v};
                repeat (2) begin
                    @(posedge clk);
                    @(negedge clk);
                    if (!cache_out_valid ||
                        held_data !== {cache_out_hit, cache_out_slot,
                                      cache_out_position, cache_out_head,
                                      cache_out_dimension, cache_out_k,
                                      cache_out_v}) begin
                        $display("CACHE_BACKPRESSURE_FAIL");
                        failures = failures + 1;
                    end
                    cache_stalls = cache_stalls + 1;
                end
            end
            cache_out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cache_out_ready = 1'b0;
            if (cache_out_valid)
                failures = failures + 1;
            cache_reads = cache_reads + 1;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "build/qkv_rope_cache_vectors";
        rope_path = {vector_dir, "/rope_cases.hex"};
        cache_path = {vector_dir, "/cache_cases.hex"};
        $readmemh(rope_path, rope_mem);
        $readmemh(cache_path, cache_mem);
        $dumpfile("build/iverilog/ace3_qkv_rope_cache.vcd");
        $dumpvars(0, ace3_qkv_rope_cache_tb);

        failures = 0;
        rope_outputs = 0;
        rope_stalls = 0;
        cache_writes = 0;
        cache_reads = 0;
        cache_stalls = 0;
        reset_checks = 0;
        clear_checks = 0;
        xz_checks = 0;
        for (index = 0; index < 14; index = index + 1)
            query_head_seen[index] = 0;
        for (index = 0; index < 2; index = index + 1) begin
            key_head_seen[index] = 0;
            value_head_seen[index] = 0;
        end
        rst_n = 1'b1;
        apply_reset();

        for (index = 0; index < `QKV_ROPE_CASES; index = index + 1)
            run_rope_case(rope_mem[index], index);
        for (index = 0; index < 14; index = index + 1)
            if (query_head_seen[index] != 32)
                failures = failures + 1;
        for (index = 0; index < 2; index = index + 1)
            if (key_head_seen[index] != 32)
                failures = failures + 1;

        @(negedge clk);
        rope_is_key = 1'b1;
        rope_head = 4'd2;
        rope_in_valid = 1'b1;
        #1;
        if (rope_in_ready !== 1'b0)
            failures = failures + 1;
        @(posedge clk);
        @(negedge clk);
        rope_in_valid = 1'b0;
        if (rope_out_valid)
            failures = failures + 1;

        rope_is_key = 1'b0;
        rope_head = 4'd0;
        rope_pair = 5'd0;
        rope_position = 15'd0;
        rope_low = 16'h7c00;
        rope_high = 16'h3c00;
        rope_cos = 16'h3c00;
        rope_sin = 16'd0;
        rope_in_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        rope_in_valid = 1'b0;
        #1;
        if (!rope_out_valid || !rope_invalid ||
            (rope_out_low !== 16'd0) || (rope_out_high !== 16'd0))
            failures = failures + 1;
        rope_out_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        rope_out_ready = 1'b0;

        rope_low = 16'h3c00;
        rope_in_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        rope_in_valid = 1'b0;
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        if (rope_out_valid)
            failures = failures + 1;
        clear_checks = clear_checks + 1;

        rope_low = 16'hxxxx;
        rope_high = 16'hzzzz;
        rope_cos = 16'hxxxx;
        rope_sin = 16'hzzzz;
        rope_in_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (rope_out_valid)
            failures = failures + 1;
        xz_checks = xz_checks + 1;
        rope_low = 16'd0;
        rope_high = 16'd0;
        rope_cos = 16'h3c00;
        rope_sin = 16'd0;

        for (index = 0; index < `QKV_CACHE_CASES; index = index + 1) begin
            write_cache(
                cache_mem[index][55:54],
                cache_mem[index][53:39],
                {3'd0, cache_mem[index][38]},
                cache_mem[index][37:32],
                cache_mem[index][15:0],
                cache_mem[index][31:16]
            );
            value_head_seen[cache_mem[index][38]] =
                value_head_seen[cache_mem[index][38]] + 1;
        end
        for (index = 0; index < `QKV_CACHE_CASES; index = index + 1)
            read_cache(
                cache_mem[index][55:54],
                cache_mem[index][53:39],
                {3'd0, cache_mem[index][38]},
                cache_mem[index][37:32],
                1'b1,
                cache_mem[index][15:0],
                cache_mem[index][31:16],
                (index % 31) == 0
            );
        if ((value_head_seen[0] != 64) || (value_head_seen[1] != 64))
            failures = failures + 1;

        write_cache(0, 3, 0, 0, 16'haaaa, 16'hbbbb);
        write_cache(0, 3, 0, 0, 16'hcccc, 16'hdddd);
        read_cache(0, 3, 0, 0, 1'b1, 16'hcccc, 16'hdddd, 1'b1);
        read_cache(0, 3, 0, 1, 1'b1,
                   cache_mem[1][15:0], cache_mem[1][31:16], 1'b0);

        write_cache(1, 3, 0, 0, 16'h1111, 16'h2222);
        write_cache(0, 4, 0, 0, 16'h3333, 16'h4444);
        read_cache(1, 3, 0, 0, 1'b1, 16'h1111, 16'h2222, 1'b0);
        read_cache(0, 4, 0, 0, 1'b1, 16'h3333, 16'h4444, 1'b0);
        read_cache(0, 3, 0, 0, 1'b1, 16'hcccc, 16'hdddd, 1'b0);
        read_cache(1, 4, 1, 63, 1'b0, 16'd0, 16'd0, 1'b0);

        @(negedge clk);
        cache_write_slot = 1;
        cache_write_position = 5;
        cache_write_head = 1;
        cache_write_dimension = 7;
        cache_write_k = 16'h5555;
        cache_write_v = 16'h6666;
        cache_read_slot = 1;
        cache_read_position = 5;
        cache_read_head = 1;
        cache_read_dimension = 7;
        cache_write_valid = 1'b1;
        cache_read_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cache_write_valid = 1'b0;
        cache_read_valid = 1'b0;
        #1;
        if (!cache_out_valid || !cache_out_hit ||
            (cache_out_k !== 16'h5555) || (cache_out_v !== 16'h6666))
            failures = failures + 1;
        cache_out_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cache_out_ready = 1'b0;

        write_cache(0, 2, 1, 9, 16'h7777, 16'h8888);
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        clear_checks = clear_checks + 1;
        read_cache(0, 2, 1, 9, 1'b0, 16'd0, 16'd0, 1'b0);

        write_cache(0, 1, 1, 8, 16'h9999, 16'haaaa);
        apply_reset();
        read_cache(0, 1, 1, 8, 1'b0, 16'd0, 16'd0, 1'b0);

        @(negedge clk);
        cache_write_valid = 1'b0;
        cache_read_valid = 1'b0;
        cache_write_slot = 2'bxx;
        cache_write_position = 15'hxxxx;
        cache_write_k = 16'hzzzz;
        cache_write_v = 16'hxxxx;
        cache_read_slot = 2'bzz;
        cache_read_position = 15'hxxxx;
        @(posedge clk);
        @(negedge clk);
        if (cache_out_valid)
            failures = failures + 1;
        xz_checks = xz_checks + 1;

        cache_write_slot = 0;
        cache_write_position = 8;
        cache_write_head = 0;
        cache_write_dimension = 0;
        cache_write_valid = 1'b1;
        #1;
        if (cache_write_ready !== 1'b0)
            failures = failures + 1;
        cache_write_valid = 1'b0;
        cache_read_slot = 0;
        cache_read_position = 8;
        cache_read_head = 0;
        cache_read_dimension = 0;
        cache_read_valid = 1'b1;
        #1;
        if (cache_read_ready !== 1'b0)
            failures = failures + 1;
        cache_read_valid = 1'b0;

        if (failures == 0) begin
            $display(
                "ACE3_QKV_ROPE_CACHE_IVERILOG_PASS rope_outputs=%0d rope_stalls=%0d query_heads=14 key_heads=2 value_heads=2 cache_writes=%0d cache_reads=%0d cache_stalls=%0d overwrite=pass isolation=pass write_through=pass reset=%0d clear=%0d xz=%0d",
                rope_outputs, rope_stalls, cache_writes, cache_reads,
                cache_stalls, reset_checks, clear_checks, xz_checks
            );
        end else begin
            $display("ACE3_QKV_ROPE_CACHE_IVERILOG_FAIL failures=%0d", failures);
        end
        $finish(failures != 0);
    end
endmodule

`default_nettype wire
