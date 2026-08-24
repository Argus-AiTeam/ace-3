#include "Vace3_awq_w4a16_g128_dot_lane.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Case {
    std::string name;
    uint32_t qzero;
    uint16_t scale;
    uint8_t lane;
    uint32_t accumulator[3];
    uint16_t result;
    bool invalid;
    bool saturation;
};

static void tick(Vace3_awq_w4a16_g128_dot_lane* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
}

static std::vector<Case> read_cases(const char* path) {
    std::ifstream input(path);
    std::vector<Case> cases;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        Case item{};
        std::string qzero;
        std::string scale;
        std::string accumulator;
        std::string result;
        unsigned lane;
        unsigned invalid;
        unsigned saturation;
        fields >> item.name >> qzero >> scale >> lane >> accumulator >> result
               >> invalid >> saturation;
        if (!fields || accumulator.size() != 24) {
            std::cerr << "invalid case record: " << line << "\n";
            std::exit(2);
        }
        item.qzero = std::stoul(qzero, nullptr, 16);
        item.scale = std::stoul(scale, nullptr, 16);
        item.lane = lane;
        item.accumulator[2] =
            std::stoul(accumulator.substr(0, 8), nullptr, 16);
        item.accumulator[1] =
            std::stoul(accumulator.substr(8, 8), nullptr, 16);
        item.accumulator[0] =
            std::stoul(accumulator.substr(16, 8), nullptr, 16);
        item.result = std::stoul(result, nullptr, 16);
        item.invalid = invalid;
        item.saturation = saturation;
        cases.push_back(item);
    }
    return cases;
}

static std::vector<uint64_t> read_pairs(const char* path) {
    std::ifstream input(path);
    std::vector<uint64_t> pairs;
    std::string line;
    while (std::getline(input, line))
        pairs.push_back(std::stoull(line, nullptr, 16));
    return pairs;
}

static void drive_idle(Vace3_awq_w4a16_g128_dot_lane* top) {
    top->clear_i = 0;
    top->start_valid_i = 0;
    top->logical_lane_i = 0;
    top->qzeros_i = 0;
    top->scale_f16_i = 0;
    top->pair_valid_i = 0;
    top->activation_f16_i = 0;
    top->qweight_i = 0;
    top->out_ready_i = 0;
}

static void start_case(
    Vace3_awq_w4a16_g128_dot_lane* top, const Case& item
) {
    if (!top->start_ready_o) {
        std::cerr << "start not ready for " << item.name << "\n";
        std::exit(2);
    }
    top->qzeros_i = item.qzero;
    top->scale_f16_i = item.scale;
    top->logical_lane_i = item.lane;
    top->start_valid_i = 1;
    tick(top);
    top->start_valid_i = 0;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* top = new Vace3_awq_w4a16_g128_dot_lane;
    const auto cases = read_cases("generated/cases.txt");
    const auto pairs = read_pairs("generated/pairs.hex");
    unsigned failures = 0;

    drive_idle(top);
    top->rst_ni = 0;
    top->eval();
    if (top->out_valid_o || top->pair_ready_o || top->invalid_operand_o ||
        top->saturation_o)
        ++failures;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    tick(top);
    if (!top->start_ready_o)
        ++failures;

    start_case(top, cases.at(0));
    for (unsigned index = 0; index < 3; ++index) {
        const uint64_t pair = pairs.at(index);
        top->qweight_i = pair & 0xffffffffu;
        top->activation_f16_i = pair >> 32;
        top->pair_valid_i = 1;
        tick(top);
    }
    top->pair_valid_i = 0;
    top->rst_ni = 0;
    top->eval();
    if (top->out_valid_o || top->pair_ready_o || top->invalid_operand_o ||
        top->saturation_o)
        ++failures;
    top->rst_ni = 1;
    tick(top);
    if (!top->start_ready_o)
        ++failures;

    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const Case& item = cases.at(case_index);
        start_case(top, item);
        for (unsigned pair_index = 0; pair_index < 128; ++pair_index) {
            if (pair_index % 11 == 3) {
                top->pair_valid_i = 0;
                tick(top);
                tick(top);
                if (!top->pair_ready_o)
                    ++failures;
            }
            const uint64_t pair = pairs.at(case_index * 128 + pair_index);
            top->qweight_i = pair & 0xffffffffu;
            top->activation_f16_i = pair >> 32;
            top->pair_valid_i = 1;
            if (!top->pair_ready_o)
                ++failures;
            tick(top);
            top->pair_valid_i = 0;
        }
        if (!top->out_valid_o) {
            std::cerr << "missing output for " << item.name << "\n";
            ++failures;
            continue;
        }
        const uint16_t held_result = top->out_f16_o;
        const uint32_t held_accumulator[3] = {
            top->acc_q47_48_o[0],
            top->acc_q47_48_o[1],
            top->acc_q47_48_o[2],
        };
        const bool held_invalid = top->invalid_operand_o;
        const bool held_saturation = top->saturation_o;
        for (unsigned stall = 0; stall < 4; ++stall) {
            tick(top);
            if (!top->out_valid_o || top->start_ready_o ||
                top->out_f16_o != held_result ||
                top->acc_q47_48_o[0] != held_accumulator[0] ||
                top->acc_q47_48_o[1] != held_accumulator[1] ||
                top->acc_q47_48_o[2] != held_accumulator[2] ||
                top->invalid_operand_o != held_invalid ||
                top->saturation_o != held_saturation)
                ++failures;
        }
        if (top->out_f16_o != item.result ||
            top->acc_q47_48_o[0] != item.accumulator[0] ||
            top->acc_q47_48_o[1] != item.accumulator[1] ||
            top->acc_q47_48_o[2] != item.accumulator[2] ||
            top->invalid_operand_o != item.invalid ||
            top->saturation_o != item.saturation) {
            std::cerr << "numeric mismatch for " << item.name << "\n";
            ++failures;
        }
        top->out_ready_i = 1;
        tick(top);
        top->out_ready_i = 0;
    }

    top->clear_i = 1;
    tick(top);
    top->clear_i = 0;
    if (!top->start_ready_o || top->out_valid_o || top->pair_ready_o)
        ++failures;

    if (failures == 0) {
        std::cout
            << "AWQ_W4A16_G128_VERILATOR_PASS cases=" << cases.size()
            << " pairs=" << pairs.size()
            << " ulp_bound=0 reset=pass backpressure=pass\n";
    } else {
        std::cerr << "AWQ_W4A16_G128_VERILATOR_FAIL failures=" << failures
                  << "\n";
    }
    top->final();
    delete top;
    return failures == 0 ? 0 : 1;
}
