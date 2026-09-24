#include "Vace3_streaming_tied_lm_head_topk.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static void fail(const char* message) {
    std::cerr << message << "\n";
    std::exit(1);
}

static void tick(Vace3_streaming_tied_lm_head_topk& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
}

static void idle(Vace3_streaming_tied_lm_head_topk& top) {
    top.clear_i = 0;
    top.start_valid_i = 0;
    top.hidden_valid_i = 0;
    top.hidden_index_i = 0;
    top.hidden_f16_i = 0;
    top.hidden_last_i = 0;
    top.hidden_end_i = 0;
    top.weight_valid_i = 0;
    top.weight_token_index_i = 0;
    top.weight_feature_index_i = 0;
    top.weight_f16_i = 0;
    top.weight_last_feature_i = 0;
    top.weight_last_token_i = 0;
    top.weight_end_i = 0;
    top.logit_ready_i = 0;
    top.top_ready_i = 0;
    top.done_ready_i = 0;
}

static void send_hidden(
    Vace3_streaming_tied_lm_head_topk& top,
    uint32_t index,
    uint16_t bits,
    bool last
) {
    if (!top.hidden_ready_o) fail("hidden channel stalled");
    top.hidden_index_i = index;
    top.hidden_f16_i = bits;
    top.hidden_last_i = last;
    top.hidden_end_i = last;
    top.hidden_valid_i = 1;
    tick(top);
    top.hidden_valid_i = 0;
    top.hidden_last_i = 0;
    top.hidden_end_i = 0;
}

static void send_weight(
    Vace3_streaming_tied_lm_head_topk& top,
    uint32_t token,
    uint32_t feature,
    uint16_t bits,
    bool last_token
) {
    if (!top.weight_ready_o) fail("weight channel stalled");
    top.weight_token_index_i = token;
    top.weight_feature_index_i = feature;
    top.weight_f16_i = bits;
    top.weight_last_feature_i = feature == 1;
    top.weight_last_token_i = last_token && feature == 1;
    top.weight_end_i = top.weight_last_token_i;
    top.weight_valid_i = 1;
    tick(top);
    top.weight_valid_i = 0;
    top.weight_last_feature_i = 0;
    top.weight_last_token_i = 0;
    top.weight_end_i = 0;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vace3_streaming_tied_lm_head_topk top;
    idle(top);
    top.rst_ni = 0;
    tick(top);
    tick(top);
    top.rst_ni = 1;
    top.eval();
    if (!top.start_ready_o) fail("DUT not ready after reset");
    top.start_valid_i = 1;
    tick(top);
    top.start_valid_i = 0;
    send_hidden(top, 0, 0x3c00, false);
    send_hidden(top, 1, 0x3c00, true);

    const std::array<std::array<uint16_t, 2>, 4> weights = {{
        {{0x3c00, 0x0000}},
        {{0x3c00, 0x3c00}},
        {{0x3c00, 0x3c00}},
        {{0xbc00, 0x0000}},
    }};
    const std::array<uint16_t, 4> expected = {{0x3c00, 0x4000, 0x4000, 0xbc00}};
    for (uint32_t token = 0; token < weights.size(); ++token) {
        send_weight(top, token, 0, weights[token][0], false);
        send_weight(top, token, 1, weights[token][1], token + 1 == weights.size());
        if (!top.logit_valid_o || top.logit_token_index_o != token ||
            top.logit_f16_o != expected[token] || top.logit_saturation_o ||
            top.invalid_operand_o || top.error_valid_o)
            fail("Verilator numerical logit mismatch");
        const uint16_t held = top.logit_f16_o;
        tick(top);
        if (!top.logit_valid_o || top.logit_f16_o != held)
            fail("Verilator logit backpressure mismatch");
        top.logit_ready_i = 1;
        tick(top);
        top.logit_ready_i = 0;
    }

    const std::array<uint32_t, 3> expected_tokens = {{1, 2, 0}};
    const std::array<uint16_t, 3> expected_top = {{0x4000, 0x4000, 0x3c00}};
    for (uint32_t rank = 0; rank < expected_tokens.size(); ++rank) {
        if (!top.top_valid_o || top.top_rank_o != rank ||
            top.top_token_index_o != expected_tokens[rank] ||
            top.top_logit_f16_o != expected_top[rank])
            fail("Verilator Top-K or tie-break mismatch");
        top.top_ready_i = 1;
        tick(top);
        top.top_ready_i = 0;
    }
    if (!top.done_valid_o) fail("Verilator completion missing");
    top.done_ready_i = 1;
    tick(top);
    top.done_ready_i = 0;
    if (!top.start_ready_o || top.busy_o || top.error_valid_o)
        fail("Verilator DUT did not return idle");
    std::cout << "STREAMING_LM_HEAD_VERILATOR_PREFLIGHT_PASS logits=4 top_k=3 ties=ascending\n";
    return 0;
}
