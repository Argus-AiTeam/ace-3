#include "Vace3_awq_w4a16_g128_dot_lane.h"
#include "verilated.h"

#include <cstdlib>
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

struct InputPaths {
    std::string cases;
    std::string pairs;
};

static InputPaths parse_input_paths(int argc, char** argv) {
    InputPaths paths;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if ((argument == "--cases" || argument == "--pairs") &&
            index + 1 >= argc) {
            std::cerr << argument << " requires a path\n";
            std::exit(2);
        }
        if (argument == "--cases")
            paths.cases = argv[++index];
        else if (argument == "--pairs")
            paths.pairs = argv[++index];
    }
    if (paths.cases.empty() || paths.pairs.empty()) {
        std::cerr << "usage: verilator-sim --cases PATH --pairs PATH\n";
        std::exit(2);
    }
    return paths;
}

static void tick(Vace3_awq_w4a16_g128_dot_lane* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
}

static std::vector<Case> read_cases(const char* path) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open case records: " << path << "\n";
        std::exit(2);
    }
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
    if (!input) {
        std::cerr << "cannot open pair records: " << path << "\n";
        std::exit(2);
    }
    std::vector<uint64_t> pairs;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() != 12) {
            std::cerr << "invalid pair record: " << line << "\n";
            std::exit(2);
        }
        pairs.push_back(std::stoull(line, nullptr, 16));
    }
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

static bool state_is_cleared(
    const Vace3_awq_w4a16_g128_dot_lane* top
) {
    return !top->out_valid_o && !top->pair_ready_o &&
           top->start_ready_o && top->out_f16_o == 0 &&
           top->acc_q47_48_o[0] == 0 && top->acc_q47_48_o[1] == 0 &&
           top->acc_q47_48_o[2] == 0 && !top->invalid_operand_o &&
           !top->saturation_o;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const InputPaths paths = parse_input_paths(argc, argv);
    auto* top = new Vace3_awq_w4a16_g128_dot_lane;
    const auto cases = read_cases(paths.cases.c_str());
    const auto pairs = read_pairs(paths.pairs.c_str());
    unsigned failures = 0;
    size_t vector_starts = 0;
    size_t vector_pairs = 0;
    size_t vector_outputs = 0;

    if (cases.size() != 30 || pairs.size() != 3840) {
        std::cerr << "vector geometry mismatch: cases=" << cases.size()
                  << " pairs=" << pairs.size() << "\n";
        delete top;
        return 2;
    }

    drive_idle(top);
    top->rst_ni = 0;
    top->eval();
    if (top->out_valid_o || top->pair_ready_o || top->invalid_operand_o ||
        top->saturation_o || top->out_f16_o ||
        top->acc_q47_48_o[0] || top->acc_q47_48_o[1] ||
        top->acc_q47_48_o[2])
        ++failures;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    tick(top);
    if (!state_is_cleared(top))
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
    if (!state_is_cleared(top))
        ++failures;

    start_case(top, cases.at(0));
    top->qweight_i = pairs.at(0) & 0xffffffffu;
    top->activation_f16_i = pairs.at(0) >> 32;
    top->pair_valid_i = 1;
    tick(top);
    top->pair_valid_i = 0;
    top->clear_i = 1;
    tick(top);
    top->clear_i = 0;
    top->eval();
    if (!state_is_cleared(top))
        ++failures;

    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const Case& item = cases.at(case_index);
        start_case(top, item);
        ++vector_starts;
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
            if (!top->pair_ready_o) {
                ++failures;
            } else {
                ++vector_pairs;
            }
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
        if (!top->out_valid_o)
            ++failures;
        else
            ++vector_outputs;
        tick(top);
        top->out_ready_i = 0;
        if (top->out_valid_o || !top->start_ready_o)
            ++failures;
    }

    top->clear_i = 1;
    tick(top);
    top->clear_i = 0;
    top->eval();
    if (!state_is_cleared(top))
        ++failures;
    if (vector_starts != 30 || vector_pairs != 3840 ||
        vector_outputs != 30)
        ++failures;

    if (failures == 0) {
        std::cout
            << "AWQ_W4A16_G128_VERILATOR_PASS cases=" << cases.size()
            << " pairs=" << pairs.size()
            << " ulp_bound=0 reset=pass clear=pass backpressure=pass"
            << " protocol=pass transactions=" << vector_outputs
            << " four_state=unsupported\n";
    } else {
        std::cerr << "AWQ_W4A16_G128_VERILATOR_FAIL failures=" << failures
                  << "\n";
    }
    top->final();
    delete top;
    return failures == 0 ? 0 : 1;
}
