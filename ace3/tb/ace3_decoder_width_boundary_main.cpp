#include "Vace3_decoder_layer0_token_engine.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>

namespace {
uint64_t cycles = 0;
unsigned failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "DECODER_WIDTH_BOUNDARY_FAIL cycle=" << cycles
                  << " " << message << '\n';
        ++failures;
    }
}

void idle(Vace3_decoder_layer0_token_engine& top) {
    top.clear_i = 0;
    top.load_valid_i = 0;
    top.load_kind_i = 0;
    top.load_index_i = 0;
    top.load_f16_i = 0;
    top.start_valid_i = 0;
    top.start_cache_slot_i = 0;
    top.start_position_i = 0;
    top.projection_meta_valid_i = 0;
    top.projection_qzeros_i = 0;
    top.projection_scale_f16_i = 0;
    top.projection_pair_valid_i = 0;
    top.projection_qweight_i = 0;
    top.projection_bias_valid_i = 0;
    top.projection_bias_f16_i = 0;
    top.rope_valid_i = 0;
    top.rope_cos_f16_i = 0;
    top.rope_sin_f16_i = 0;
    top.trace_ready_i = 0;
    top.final_ready_i = 0;
    top.done_ready_i = 0;
}

void settle(Vace3_decoder_layer0_token_engine& top) {
    top.clk_i = 0;
    top.eval();
}

void tick(Vace3_decoder_layer0_token_engine& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    ++cycles;
}

void load_vector(Vace3_decoder_layer0_token_engine& top, unsigned kind) {
    top.load_valid_i = 1;
    top.load_kind_i = kind;
    top.load_f16_i = 0x3c00;
    for (unsigned index = 0; index < 896; ++index) {
        top.load_index_i = index;
        settle(top);
        expect(top.load_ready_o, "legal sequential load was rejected");
        tick(top);
    }
    top.load_valid_i = 0;
}
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vace3_decoder_layer0_token_engine top;
    idle(top);

    top.rst_ni = 0;
    settle(top);
    expect(!top.load_ready_o && !top.start_ready_o && !top.busy_o,
           "asynchronous reset did not fail closed");
    tick(top);
    tick(top);
    top.rst_ni = 1;
    settle(top);
    expect(!top.projection_bias_ready_o,
           "projection bias channel was ready while idle");

    top.load_valid_i = 1;
    top.load_kind_i = 0;
    top.load_index_i = 1;
    top.load_f16_i = 0x3c00;
    settle(top);
    expect(!top.load_ready_o, "out-of-sequence load index was accepted");
    top.load_valid_i = 0;

    load_vector(top, 0);
    load_vector(top, 1);
    load_vector(top, 2);

    top.start_cache_slot_i = 2;
    top.start_position_i = 0;
    settle(top);
    expect(!top.start_ready_o, "cache slot 2 was accepted");
    top.start_cache_slot_i = 3;
    settle(top);
    expect(!top.start_ready_o, "cache slot 3 was accepted");
    top.start_cache_slot_i = 0;
    top.start_position_i = 128;
    settle(top);
    expect(!top.start_ready_o, "position 128 was accepted");
    top.start_position_i = 0x7fff;
    settle(top);
    expect(!top.start_ready_o, "maximum invalid position was accepted");
    top.start_cache_slot_i = 0;
    top.start_position_i = 0;
    settle(top);
    expect(top.start_ready_o, "slot 0 position 0 was rejected");
    top.start_cache_slot_i = 1;
    settle(top);
    expect(top.start_ready_o, "slot 1 position 0 was rejected");

    top.start_cache_slot_i = 0;
    top.start_valid_i = 1;
    tick(top);
    top.start_valid_i = 0;
    settle(top);
    expect(top.busy_o, "legal start did not enter the controller");
    top.clear_i = 1;
    tick(top);
    top.clear_i = 0;
    settle(top);
    expect(!top.busy_o && !top.trace_valid_o && !top.final_valid_o &&
               !top.done_valid_o && !top.start_ready_o,
           "clear did not return to an unloaded fail-closed state");
    expect(!top.projection_bias_ready_o,
           "projection bias channel was ready after clear");

    top.final();
    if (failures != 0)
        return 1;
    std::cout << "DECODER_WIDTH_BOUNDARY_VERILATOR_PASS cycles="
              << cycles << '\n';
    return 0;
}
