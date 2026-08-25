`timescale 1ns/1ps
`default_nettype none

module ace3_decoder_qzeros_boundary_tb;
    reg meta_live;
    reg [2:0] projection_kind;
    reg [5:0] group_index;
    reg [9:0] word_index;
    wire address_valid;
    wire [15:0] address;
    reg [31:0] selected_word;
    reg [31:0] q_qzeros [0:783], k_qzeros [0:111], v_qzeros [0:111];
    reg [31:0] o_qzeros [0:783], gate_qzeros [0:4255];
    reg [31:0] up_qzeros [0:4255], down_qzeros [0:4255];
    string vector_dir, path;

    ace3_decoder_qzeros_address dut (
        .meta_live_i(meta_live),
        .projection_kind_i(projection_kind),
        .group_i(group_index),
        .word_i(word_index),
        .address_valid_o(address_valid),
        .address_o(address)
    );

    task check_word;
        input [2:0] kind;
        input [5:0] group_value;
        input [9:0] word_value;
        input [15:0] expected_address;
        input [31:0] expected_word;
        begin
            projection_kind=kind; group_index=group_value; word_index=word_value;
            meta_live=1; #1;
            if (!address_valid || address !== expected_address)
                $fatal(1, "DECODER_QZEROS_ADDRESS_FAIL kind=%0d group=%0d word=%0d address=%0d valid=%b",
                       kind, group_value, word_value, address, address_valid);
            case (kind)
              0: selected_word=q_qzeros[address];
              1: selected_word=k_qzeros[address];
              2: selected_word=v_qzeros[address];
              3: selected_word=o_qzeros[address];
              4: selected_word=gate_qzeros[address];
              5: selected_word=up_qzeros[address];
              6: selected_word=down_qzeros[address];
              default: $fatal(1, "DECODER_QZEROS_KIND_FAIL kind=%0d", kind);
            endcase
            if (selected_word !== expected_word)
                $fatal(1, "DECODER_QZEROS_SERIALIZATION_FAIL kind=%0d address=%0d expected=%08x actual=%08x",
                       kind, address, expected_word, selected_word);
            meta_live=0; #1;
            if (address_valid || address !== 0)
                $fatal(1, "DECODER_QZEROS_IDLE_DEREFERENCE_FAIL kind=%0d address=%0d valid=%b",
                       kind, address, address_valid);
        end
    endtask

    initial begin
        meta_live=0; projection_kind=0; group_index=0; word_index=0; selected_word=0;
        if (!$value$plusargs("VECTOR_DIR=%s",vector_dir))
            vector_dir="build/decoder_layer0_vectors";
        path={vector_dir,"/tensors/layer0_self_attn_q_proj_qzeros.i32le.bin.hex"}; $readmemh(path,q_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_k_proj_qzeros.i32le.bin.hex"}; $readmemh(path,k_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_v_proj_qzeros.i32le.bin.hex"}; $readmemh(path,v_qzeros);
        path={vector_dir,"/tensors/layer0_self_attn_o_proj_qzeros.i32le.bin.hex"}; $readmemh(path,o_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_gate_proj_qzeros.i32le.bin.hex"}; $readmemh(path,gate_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_up_proj_qzeros.i32le.bin.hex"}; $readmemh(path,up_qzeros);
        path={vector_dir,"/tensors/layer0_mlp_down_proj_qzeros.i32le.bin.hex"}; $readmemh(path,down_qzeros);

        if ($bits(group_index) != 6 || $bits(word_index) != 10 || $bits(address) != 16)
            $fatal(1, "DECODER_QZEROS_WIDTH_FAIL");
        check_word(0,0,0,0,32'hb6674377); check_word(0,6,111,783,32'h67975798);
        check_word(1,0,0,0,32'h57749977);

        projection_kind=1; group_index=6; word_index=111; meta_live=0; #1;
        if (address_valid || address !== 0)
            $fatal(1, "DECODER_QZEROS_Q_TO_K_IDLE_FAIL address=%0d valid=%b",address,address_valid);
        meta_live=1; #1;
        if (address_valid)
            $fatal(1, "DECODER_QZEROS_LIVE_RANGE_REJECTION_FAIL address=%0d",address);
        meta_live=0;

        check_word(1,6,15,111,32'h38984897);
        check_word(2,0,0,0,32'ha868c896); check_word(2,6,15,111,32'h88996979);
        check_word(3,0,0,0,32'h68877776); check_word(3,6,111,783,32'h77688778);
        check_word(4,0,0,0,32'h78898787); check_word(4,6,607,4255,32'h49777889);
        check_word(5,0,0,0,32'h88897888); check_word(5,6,607,4255,32'h787a8686);
        check_word(6,0,0,0,32'h89877778); check_word(6,37,111,4255,32'h98888877);
        $display("DECODER_QZEROS_BOUNDARY_PASS domains=q:0..783,k:0..111,v:0..111,o:0..783,gate:0..4255,up:0..4255,down:0..4255 q_to_k_stale=qualified live_oob=rejected serialized_edges=exact");
        $finish;
    end
endmodule

`default_nettype wire
