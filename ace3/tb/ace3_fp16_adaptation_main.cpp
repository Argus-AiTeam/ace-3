#include "Vace3_fp16_adaptation_verilator_top.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static uint64_t cycles = 0;

static std::vector<uint64_t> read_hex(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(2);
    }
    std::vector<uint64_t> records;
    std::string line;
    while (std::getline(input, line))
        records.push_back(std::stoull(line, nullptr, 16));
    return records;
}

static void tick(Vace3_fp16_adaptation_verilator_top* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
    ++cycles;
}

static void idle(Vace3_fp16_adaptation_verilator_top* top) {
    top->clear_i = 0;
    top->rr_start_valid_i = 0;
    top->rr_element_count_i = 1;
    top->rr_in_valid_i = 0;
    top->rr_projection_f16_i = 0;
    top->rr_residual_f16_i = 0;
    top->rr_out_ready_i = 0;
    top->sg_start_valid_i = 0;
    top->sg_element_count_i = 1;
    top->sg_in_valid_i = 0;
    top->sg_gate_f16_i = 0;
    top->sg_up_f16_i = 0;
    top->sg_out_ready_i = 0;
    top->rn_start_valid_i = 0;
    top->rn_element_count_i = 8;
    top->rn_in_valid_i = 0;
    top->rn_activation_f16_i = 0;
    top->rn_weight_f16_i = 0;
    top->rn_out_ready_i = 0;
    top->eval();
}

static void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string vector_dir;
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--vector-dir" && index + 1 < argc)
            vector_dir = argv[++index];
    }
    if (vector_dir.empty()) {
        std::cerr << "usage: fp16-adaptation --vector-dir PATH\n";
        return 2;
    }
    const auto residual = read_hex(vector_dir + "/residual_cases.hex");
    const auto silu = read_hex(vector_dir + "/silu_cases.hex");
    const auto rms_inputs = read_hex(vector_dir + "/rms_inputs.hex");
    const auto rms_expected = read_hex(vector_dir + "/rms_expected.hex");
    const auto rms_meta = read_hex(vector_dir + "/rms_meta.hex");
    require(!residual.empty() && !silu.empty() && rms_meta.size() == 5,
            "vector streams are vacuous");
    require(rms_inputs.size() == 40 && rms_expected.size() == 40,
            "RMS vector stream count mismatch");

    auto* top = new Vace3_fp16_adaptation_verilator_top;
    idle(top);
    top->rst_ni = 0;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    tick(top);

    unsigned accepted = 0;
    unsigned outputs = 0;
    unsigned stalls = 0;
    unsigned invalid_outputs = 0;
    unsigned saturation_outputs = 0;
    unsigned invalid_starts = 0;
    unsigned clear_checks = 0;
    unsigned reset_checks = 0;

    top->rr_element_count_i = 0;
    top->sg_element_count_i = 4865;
    top->rn_element_count_i = 7;
    top->eval();
    require(!top->rr_start_ready_o && !top->sg_start_ready_o &&
                !top->rn_start_ready_o,
            "Verilator invalid-start rejection failed");
    invalid_starts = 3;
    idle(top);

    for (uint64_t record : residual) {
        top->rr_element_count_i = 1;
        top->rr_start_valid_i = 1;
        top->eval();
        require(top->rr_start_ready_o, "residual start not ready");
        tick(top);
        top->rr_start_valid_i = 0;
        top->rr_projection_f16_i = record & 0xffffu;
        top->rr_residual_f16_i = (record >> 16) & 0xffffu;
        top->rr_in_valid_i = 1;
        top->eval();
        require(top->rr_in_ready_o, "residual input not ready");
        const uint64_t accepted_cycle = cycles;
        tick(top);
        ++accepted;
        top->rr_in_valid_i = 0;
        require(top->rr_out_valid_o && cycles - accepted_cycle == 1,
                "residual latency mismatch");
        require(top->rr_out_f16_o == ((record >> 32) & 0xffffu) &&
                    top->rr_invalid_o == ((record >> 48) & 1u) &&
                    top->rr_saturation_o == ((record >> 49) & 1u) &&
                    top->rr_out_last_o && top->rr_out_index_o == 0,
                "residual result mismatch");
        const uint16_t held = top->rr_out_f16_o;
        tick(top);
        require(top->rr_out_valid_o && top->rr_out_f16_o == held,
                "residual output changed under stall");
        ++stalls;
        invalid_outputs += top->rr_invalid_o;
        saturation_outputs += top->rr_saturation_o;
        top->rr_out_ready_i = 1;
        tick(top);
        top->rr_out_ready_i = 0;
        ++outputs;
    }

    top->rr_element_count_i = 2;
    top->rr_start_valid_i = 1;
    tick(top);
    top->rr_start_valid_i = 0;
    top->rr_in_valid_i = 1;
    top->rr_projection_f16_i = 0x3c00;
    top->rr_residual_f16_i = 0x3c00;
    tick(top);
    top->rr_in_valid_i = 0;
    top->clear_i = 1;
    tick(top);
    require(!top->rr_out_valid_o && !top->rr_busy_o,
            "residual clear abort failed");
    ++clear_checks;
    top->clear_i = 0;

    for (uint64_t record : silu) {
        top->sg_element_count_i = 1;
        top->sg_start_valid_i = 1;
        top->eval();
        require(top->sg_start_ready_o, "SiLU start not ready");
        tick(top);
        top->sg_start_valid_i = 0;
        top->sg_gate_f16_i = record & 0xffffu;
        top->sg_up_f16_i = (record >> 16) & 0xffffu;
        top->sg_in_valid_i = 1;
        top->eval();
        require(top->sg_in_ready_o, "SiLU input not ready");
        const uint64_t accepted_cycle = cycles;
        tick(top);
        ++accepted;
        top->sg_in_valid_i = 0;
        require(top->sg_out_valid_o && cycles - accepted_cycle == 1,
                "SiLU latency mismatch");
        require(top->sg_out_f16_o == ((record >> 32) & 0xffffu) &&
                    top->sg_invalid_o == ((record >> 48) & 1u) &&
                    top->sg_saturation_o == ((record >> 49) & 1u),
                "SiLU result mismatch");
        const uint16_t held = top->sg_out_f16_o;
        tick(top);
        require(top->sg_out_valid_o && top->sg_out_f16_o == held,
                "SiLU output changed under stall");
        ++stalls;
        invalid_outputs += top->sg_invalid_o;
        saturation_outputs += top->sg_saturation_o;
        top->sg_out_ready_i = 1;
        tick(top);
        top->sg_out_ready_i = 0;
        ++outputs;
    }

    top->sg_element_count_i = 2;
    top->sg_start_valid_i = 1;
    tick(top);
    top->sg_start_valid_i = 0;
    top->sg_in_valid_i = 1;
    top->sg_gate_f16_i = 0x3c00;
    top->sg_up_f16_i = 0x3c00;
    tick(top);
    top->sg_in_valid_i = 0;
    top->rst_ni = 0;
    top->eval();
    require(!top->sg_out_valid_o && !top->sg_busy_o &&
                !top->sg_start_ready_o,
            "SiLU asynchronous reset abort failed");
    ++reset_checks;
    tick(top);
    top->rst_ni = 1;
    tick(top);
    idle(top);

    size_t rms_stream_index = 0;
    for (size_t transaction = 0; transaction < rms_meta.size(); ++transaction) {
        top->rn_element_count_i = 8;
        top->rn_start_valid_i = 1;
        top->eval();
        require(top->rn_start_ready_o, "RMSNorm start not ready");
        tick(top);
        top->rn_start_valid_i = 0;
        for (unsigned element = 0; element < 8; ++element) {
            const uint64_t record = rms_inputs.at(rms_stream_index++);
            top->rn_activation_f16_i = record & 0xffffu;
            top->rn_weight_f16_i = (record >> 16) & 0xffffu;
            top->rn_in_valid_i = 1;
            top->eval();
            require(top->rn_in_ready_o, "RMSNorm input not ready");
            tick(top);
            ++accepted;
        }
        top->rn_in_valid_i = 0;
        unsigned sqrt_cycles = 0;
        while (!top->rn_out_valid_o && sqrt_cycles <= 48) {
            tick(top);
            ++sqrt_cycles;
        }
        require(sqrt_cycles == 48, "RMSNorm square-root cycle mismatch");
        require(top->rn_rms_q24_o == (rms_meta.at(transaction) & ((1ull << 46) - 1)),
                "RMSNorm root mismatch");
        for (unsigned element = 0; element < 8; ++element) {
            const uint64_t expected = rms_expected.at(transaction * 8 + element);
            require(top->rn_out_valid_o &&
                        top->rn_out_f16_o == (expected & 0xffffu) &&
                        top->rn_invalid_o == ((expected >> 16) & 1u) &&
                        top->rn_saturation_o == ((expected >> 17) & 1u) &&
                        top->rn_out_index_o == element &&
                        top->rn_out_last_o == (element == 7),
                    "RMSNorm result mismatch");
            if (transaction == 0 && element == 0) {
                const uint16_t held = top->rn_out_f16_o;
                tick(top);
                require(top->rn_out_valid_o && top->rn_out_f16_o == held &&
                            top->rn_out_index_o == 0,
                        "RMSNorm output changed under stall");
                ++stalls;
            }
            invalid_outputs += top->rn_invalid_o;
            saturation_outputs += top->rn_saturation_o;
            top->rn_out_ready_i = 1;
            tick(top);
            top->rn_out_ready_i = 0;
            ++outputs;
        }
    }

    top->rn_element_count_i = 8;
    top->rn_start_valid_i = 1;
    tick(top);
    top->rn_start_valid_i = 0;
    top->rn_in_valid_i = 1;
    tick(top);
    top->rn_in_valid_i = 0;
    top->clear_i = 1;
    tick(top);
    require(!top->rn_out_valid_o && !top->rn_busy_o,
            "RMSNorm clear abort failed");
    ++clear_checks;

    require(accepted > 0 && outputs > 0 && stalls > 0 &&
                invalid_outputs > 0 && saturation_outputs > 0 &&
                invalid_starts == 3 && clear_checks == 2 && reset_checks == 1,
            "Verilator non-vacuous counters failed");
    std::cout
        << "ACE3_FP16_ADAPTATION_VERILATOR_PASS accepted_inputs=" << accepted
        << " outputs=" << outputs << " stalls=" << stalls
        << " invalid_outputs=" << invalid_outputs
        << " saturation_outputs=" << saturation_outputs
        << " invalid_starts=" << invalid_starts
        << " clear=" << clear_checks << " reset=" << reset_checks
        << " residual_latency=1 silu_latency=1 rms_sqrt_cycles=48\n";
    top->final();
    delete top;
    return 0;
}
