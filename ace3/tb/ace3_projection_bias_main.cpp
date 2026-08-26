#include "Vace3_awq_w4a16_projection_engine.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Case {
    unsigned number, lane, bias;
    std::array<uint32_t, 4> accumulator;
    unsigned result;
    bool invalid, saturation;
};

static uint64_t cycles;

static std::vector<std::string> lines(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(input, line))
        result.push_back(line);
    return result;
}

static std::array<uint32_t, 4> low_102(const std::string& hex) {
    std::array<uint32_t, 4> value{};
    unsigned bit = 0;
    for (auto it = hex.rbegin(); it != hex.rend(); ++it) {
        const unsigned nibble = *it <= '9' ? *it - '0' : *it - 'a' + 10;
        if (bit < 102)
            value[bit / 32] |= nibble << (bit % 32);
        bit += 4;
    }
    return value;
}

static std::vector<Case> load_cases(
    const std::string& directory, std::vector<uint64_t>& meta,
    std::vector<uint64_t>& pairs
) {
    const auto case_lines = lines(directory + "/cases.hex");
    const auto meta_lines = lines(directory + "/meta.hex");
    const auto pair_lines = lines(directory + "/pairs.hex");
    const auto expected_lines = lines(directory + "/expected.hex");
    const auto manifest = lines(directory + "/manifest.json");
    if (case_lines.size() != 5 || meta_lines.size() != 5 ||
        pair_lines.size() != 640 || expected_lines.size() != 5 ||
        manifest.empty())
        throw std::runtime_error("focused vector inventory mismatch");
    std::string manifest_text;
    for (const auto& line : manifest)
        manifest_text += line;
    if (manifest_text.find("\"in_features\": 128") == std::string::npos ||
        manifest_text.find("\"bias_enable\": 1") == std::string::npos ||
        manifest_text.find("post-round-bias-distinguishes-preaccumulator") ==
            std::string::npos)
        throw std::runtime_error("focused vector manifest mismatch");
    meta.reserve(5);
    pairs.reserve(640);
    for (const auto& line : meta_lines)
        meta.push_back(std::stoull(line, nullptr, 16));
    for (const auto& line : pair_lines)
        pairs.push_back(std::stoull(line, nullptr, 16));
    std::vector<Case> result;
    for (size_t index = 0; index < 5; ++index) {
        if (case_lines[index].size() != 8 || expected_lines[index].size() != 32)
            throw std::runtime_error("malformed focused vector");
        const uint32_t header = std::stoul(case_lines[index], nullptr, 16);
        const std::array<uint32_t, 4> packed = low_102(expected_lines[index]);
        const uint64_t high = std::stoull(expected_lines[index].substr(0, 16), nullptr, 16);
        Case item{
            header >> 24, (header >> 16) & 0xff, header & 0xffff, {},
            static_cast<unsigned>((high >> 46) & 0xffff),
            (high >> 62) & 1, (high >> 63) & 1
        };
        for (unsigned word = 0; word < 4; ++word) {
            uint64_t joined = packed[word];
            if (word != 3)
                joined |= static_cast<uint64_t>(packed[word + 1]) << 32;
            item.accumulator[word] = (joined >> (word == 0 ? 8 : 0)) & 0xffffffffu;
        }
        // The packed 102-bit accumulator starts at bit 8; decode it directly.
        item.accumulator[0] = (packed[0] >> 8) | (packed[1] << 24);
        item.accumulator[1] = (packed[1] >> 8) | (packed[2] << 24);
        item.accumulator[2] = (packed[2] >> 8) | (packed[3] << 24);
        item.accumulator[3] = (packed[3] >> 8) & 0x3fu;
        if (item.number != index)
            throw std::runtime_error("focused case numbering mismatch");
        result.push_back(item);
    }
    return result;
}

static void tick(Vace3_awq_w4a16_projection_engine& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    ++cycles;
}

static void idle(Vace3_awq_w4a16_projection_engine& top) {
    top.clear_i = 0; top.start_valid_i = 0; top.first_output_channel_i = 0;
    top.output_count_i = 1; top.meta_valid_i = 0; top.qzeros_i = 0;
    top.scale_f16_i = 0; top.pair_valid_i = 0; top.activation_f16_i = 0;
    top.qweight_i = 0; top.bias_valid_i = 0; top.bias_f16_i = 0;
    top.out_ready_i = 0;
}

static bool is_clear(const Vace3_awq_w4a16_projection_engine& top) {
    return top.start_ready_o && !top.busy_o && !top.meta_ready_o &&
           !top.pair_ready_o && !top.bias_ready_o && !top.out_valid_o;
}

static void start(Vace3_awq_w4a16_projection_engine& top, unsigned channel,
                  unsigned failures) {
    top.first_output_channel_i = channel; top.output_count_i = 1;
    top.start_valid_i = 1; top.eval();
    if (!top.start_ready_o)
        throw std::runtime_error("legal start rejected");
    tick(top);
    top.start_valid_i = 0;
}

static void wait_for(bool value, Vace3_awq_w4a16_projection_engine& top,
                     const char* what) {
    for (unsigned timeout = 0; timeout != 10000; ++timeout) {
        top.eval();
        if (value)
            return;
        tick(top);
        if (std::string(what) == "meta") value = top.meta_ready_o;
        if (std::string(what) == "pair") value = top.pair_ready_o;
        if (std::string(what) == "bias") value = top.bias_ready_o;
        if (std::string(what) == "output") value = top.out_valid_o;
    }
    throw std::runtime_error(std::string("timeout waiting for ") + what);
}

int main(int argc, char** argv) {
    try {
        Verilated::commandArgs(argc, argv);
        std::string vector_dir;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--vector-dir" && i + 1 < argc)
                vector_dir = argv[++i];
        }
        if (vector_dir.empty())
            throw std::runtime_error("usage: --vector-dir PATH");
        std::vector<uint64_t> meta, pairs;
        const auto cases = load_cases(vector_dir, meta, pairs);
        Vace3_awq_w4a16_projection_engine top;
        unsigned failures = 0, stalls = 0, outputs = 0;
        cycles = 0;
        idle(top);
        top.rst_ni = 0; top.eval();
        if (top.start_ready_o || top.busy_o || top.out_valid_o) ++failures;
        tick(top); tick(top);
        top.rst_ni = 1; top.eval();
        if (!is_clear(top)) ++failures;

        // Reject both invalid configuration classes before any focused work.
        top.start_valid_i = 1; top.output_count_i = 0; top.eval();
        if (top.start_ready_o) ++failures;
        top.output_count_i = 1; top.first_output_channel_i = 8; top.eval();
        if (top.start_ready_o) ++failures;
        top.start_valid_i = 0; top.first_output_channel_i = 0;

        // An asynchronous reset in a live pair stream must erase its transaction.
        start(top, 0, failures);
        top.qzeros_i = meta[0]; top.scale_f16_i = meta[0] >> 32; top.meta_valid_i = 1;
        tick(top); top.meta_valid_i = 0;
        top.qweight_i = pairs[0]; top.activation_f16_i = pairs[0] >> 32; top.pair_valid_i = 1;
        tick(top); top.pair_valid_i = 0; top.rst_ni = 0; top.eval();
        if (top.busy_o || top.meta_ready_o || top.pair_ready_o || top.out_valid_o) ++failures;
        top.rst_ni = 1; top.eval();
        if (!is_clear(top)) ++failures;

        // Clear precisely at the bias boundary and prove a clean synchronous restart.
        start(top, 0, failures);
        top.qzeros_i = meta[0]; top.scale_f16_i = meta[0] >> 32; top.meta_valid_i = 1;
        tick(top); top.meta_valid_i = 0;
        for (unsigned p = 0; p < 128; ++p) {
            top.qweight_i = pairs[p]; top.activation_f16_i = pairs[p] >> 32;
            top.pair_valid_i = 1; tick(top); top.pair_valid_i = 0;
        }
        while (!top.bias_ready_o) tick(top);
        top.clear_i = 1; tick(top); top.clear_i = 0; top.eval();
        if (!is_clear(top)) ++failures;

        for (const Case& item : cases) {
            start(top, item.lane, failures);
            while (!top.meta_ready_o) tick(top);
            if (top.meta_output_channel_o != item.lane || top.meta_group_index_o != 0 ||
                top.meta_output_word_o != 0 || top.meta_logical_lane_o != item.lane) ++failures;
            if (item.number & 1) { tick(top); ++stalls; }
            top.qzeros_i = meta[item.number]; top.scale_f16_i = meta[item.number] >> 32;
            top.meta_valid_i = 1; tick(top); top.meta_valid_i = 0;
            for (unsigned p = 0; p < 128; ++p) {
                while (!top.pair_ready_o) tick(top);
                if (top.pair_input_index_o != p || top.pair_output_channel_o != item.lane ||
                    top.pair_group_index_o != 0 || top.pair_output_word_o != 0 ||
                    top.pair_logical_lane_o != item.lane) ++failures;
                if (p == 11 || p == 93) { tick(top); ++stalls; }
                const uint64_t word = pairs[item.number * 128 + p];
                top.qweight_i = word; top.activation_f16_i = word >> 32; top.pair_valid_i = 1;
                tick(top); top.pair_valid_i = 0;
            }
            while (!top.bias_ready_o) tick(top);
            if (top.bias_output_channel_o != item.lane) ++failures;
            if (item.number & 1) { tick(top); ++stalls; }
            top.bias_f16_i = item.bias; top.bias_valid_i = 1; tick(top); top.bias_valid_i = 0;
            while (!top.out_valid_o) tick(top);
            if (top.out_channel_o != item.lane || top.out_f16_o != item.result ||
                top.acc_q53_48_o[0] != item.accumulator[0] ||
                top.acc_q53_48_o[1] != item.accumulator[1] ||
                top.acc_q53_48_o[2] != item.accumulator[2] ||
                top.acc_q53_48_o[3] != item.accumulator[3] ||
                top.invalid_operand_o != item.invalid || top.saturation_o != item.saturation) {
                std::cerr << "PROJECTION_BIAS_MISMATCH case=" << item.number
                          << " cycle=" << cycles << " expected_channel=" << item.lane
                          << " actual_channel=" << top.out_channel_o << " expected_f16="
                          << item.result << " actual_f16=" << top.out_f16_o << "\n";
                std::cerr << "  expected_acc=" << std::hex << item.accumulator[3] << ":"
                          << item.accumulator[2] << ":" << item.accumulator[1] << ":"
                          << item.accumulator[0] << " actual_acc=" << top.acc_q53_48_o[3]
                          << ":" << top.acc_q53_48_o[2] << ":" << top.acc_q53_48_o[1]
                          << ":" << top.acc_q53_48_o[0] << std::dec << "\n";
                ++failures;
            }
            const auto channel = top.out_channel_o, f16 = top.out_f16_o;
            const std::array<uint32_t, 4> acc = {top.acc_q53_48_o[0], top.acc_q53_48_o[1],
                                                  top.acc_q53_48_o[2], top.acc_q53_48_o[3]};
            const bool invalid = top.invalid_operand_o, saturation = top.saturation_o;
            for (unsigned held = 0; held != 3; ++held) {
                tick(top); ++stalls;
                if (!top.out_valid_o || top.out_channel_o != channel || top.out_f16_o != f16 ||
                    top.acc_q53_48_o[0] != acc[0] || top.acc_q53_48_o[1] != acc[1] ||
                    top.acc_q53_48_o[2] != acc[2] || top.acc_q53_48_o[3] != acc[3] ||
                    top.invalid_operand_o != invalid || top.saturation_o != saturation) ++failures;
            }
            top.out_ready_i = 1; tick(top); top.out_ready_i = 0; ++outputs;
        }
        if (cases[0].result != 1 || cases[0].result == 2 || outputs != 5 || stalls != 29)
            ++failures;
        top.final();
        if (failures) {
            std::cerr << "PROJECTION_BIAS_VERILATOR_FAIL failures=" << failures << "\n";
            return 1;
        }
        std::cout << "PROJECTION_BIAS_VERILATOR_PASS cases=5 pairs=640 outputs=" << outputs
                  << " cycles=" << cycles << " stalls=" << stalls
                  << " post_round=pass reset=pass clear=pass invalid_config=pass\n";
    } catch (const std::exception& error) {
        std::cerr << "PROJECTION_BIAS_VERILATOR_FAIL " << error.what() << "\n";
        return 2;
    }
}
