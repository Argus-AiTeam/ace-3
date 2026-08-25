#include "Vace3_decoder_qzeros_address.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> hex_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::vector<uint32_t> values;
    std::string line;
    while (std::getline(file, line))
        values.push_back(static_cast<uint32_t>(std::stoull(line, nullptr, 16)));
    return values;
}

static uint32_t checked_at(const std::vector<uint32_t>& values, size_t address) {
    if (address >= values.size())
        throw std::runtime_error("qzeros boundary address out of range");
    return values[address];
}

int main(int argc, char** argv) {
    try {
        Verilated::commandArgs(argc, argv);
        std::string vector_dir;
        for (int index=1; index<argc; ++index) {
            const std::string argument=argv[index];
            if (argument=="--vector-dir" && index+1<argc) vector_dir=argv[++index];
        }
        if (vector_dir.empty()) throw std::runtime_error("usage: --vector-dir PATH");
        const std::string tensors=vector_dir+"/tensors/";
        const std::array<std::string,7> names{{
            "layer0_self_attn_q_proj_qzeros.i32le.bin.hex",
            "layer0_self_attn_k_proj_qzeros.i32le.bin.hex",
            "layer0_self_attn_v_proj_qzeros.i32le.bin.hex",
            "layer0_self_attn_o_proj_qzeros.i32le.bin.hex",
            "layer0_mlp_gate_proj_qzeros.i32le.bin.hex",
            "layer0_mlp_up_proj_qzeros.i32le.bin.hex",
            "layer0_mlp_down_proj_qzeros.i32le.bin.hex"}};
        std::array<std::vector<uint32_t>,7> qzeros;
        for (size_t kind=0; kind<qzeros.size(); ++kind)
            qzeros[kind]=hex_file(tensors+names[kind]);
        const std::array<size_t,7> sizes{{784,112,112,784,4256,4256,4256}};
        for (size_t kind=0; kind<qzeros.size(); ++kind)
            if (qzeros[kind].size()!=sizes[kind])
                throw std::runtime_error("qzeros serialized geometry mismatch");

        Vace3_decoder_qzeros_address top;
        const auto check=[&](unsigned kind, unsigned group, unsigned word,
                             unsigned expected_address, uint32_t expected_word) {
            top.meta_live_i=1; top.projection_kind_i=kind;
            top.group_i=group; top.word_i=word; top.eval();
            if (!top.address_valid_o || top.address_o!=expected_address)
                throw std::runtime_error("qzeros mapped address mismatch");
            if (checked_at(qzeros[kind],top.address_o)!=expected_word)
                throw std::runtime_error("qzeros serialized edge mismatch");
            top.meta_live_i=0; top.eval();
            if (top.address_valid_o || top.address_o)
                throw std::runtime_error("idle qzeros address was live");
        };
        check(0,0,0,0,0xb6674377); check(0,6,111,783,0x67975798);
        check(1,0,0,0,0x57749977);

        top.meta_live_i=0; top.projection_kind_i=1; top.group_i=6; top.word_i=111; top.eval();
        if (top.address_valid_o || top.address_o)
            throw std::runtime_error("stale Q-to-K metadata was live");
        top.meta_live_i=1; top.eval();
        if (top.address_valid_o)
            throw std::runtime_error("live out-of-range K address was accepted");

        check(1,6,15,111,0x38984897);
        check(2,0,0,0,0xa868c896); check(2,6,15,111,0x88996979);
        check(3,0,0,0,0x68877776); check(3,6,111,783,0x77688778);
        check(4,0,0,0,0x78898787); check(4,6,607,4255,0x49777889);
        check(5,0,0,0,0x88897888); check(5,6,607,4255,0x787a8686);
        check(6,0,0,0,0x89877778); check(6,37,111,4255,0x98888877);
        top.final();
        std::cout << "DECODER_QZEROS_BOUNDARY_PASS domains=q:0..783,k:0..111,v:0..111,o:0..783,gate:0..4255,up:0..4255,down:0..4255 q_to_k_stale=qualified live_oob=rejected serialized_edges=exact\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DECODER_QZEROS_BOUNDARY_FAIL " << error.what() << "\n";
        return 1;
    }
}
