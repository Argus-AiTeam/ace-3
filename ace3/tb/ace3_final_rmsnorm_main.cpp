#include "Vace3_final_rmsnorm.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kHiddenSize = 896;

struct Arguments {
    std::string activations;
    std::string weights;
    std::string raw;
    std::string terminal;
    int case_index = -1;
};

Arguments parse_arguments(int argc, char **argv) {
    Arguments args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc)
            throw std::runtime_error("missing argument value");
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--activations") args.activations = value;
        else if (key == "--weights") args.weights = value;
        else if (key == "--raw") args.raw = value;
        else if (key == "--terminal") args.terminal = value;
        else if (key == "--case") args.case_index = std::stoi(value);
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (args.activations.empty() || args.weights.empty() || args.raw.empty() ||
        args.terminal.empty() || args.case_index < 0 || args.case_index > 3)
        throw std::runtime_error("incomplete arguments");
    return args;
}

std::vector<std::uint16_t> read_memh(const std::string &path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open input: " + path);
    std::vector<std::uint16_t> values;
    std::string token;
    while (stream >> token)
        values.push_back(static_cast<std::uint16_t>(std::stoul(token, nullptr, 16)));
    return values;
}

void write_terminal(
    const std::string &path,
    bool natural_terminal,
    int recorded_exit_code,
    std::size_t output_count,
    int case_index
) {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot open terminal output");
    stream << "schema=ace3-final-rmsnorm-terminal-v1\n"
           << "natural_terminal=" << (natural_terminal ? 1 : 0) << "\n"
           << "recorded_exit_code=" << recorded_exit_code << "\n"
           << "output_count=" << output_count << "\n"
           << "case_index=" << case_index << "\n";
}

class Simulation {
  public:
    Vace3_final_rmsnorm top;
    vluint64_t time = 0;

    void cycle() {
        top.clk_i = 0;
        top.eval();
        ++time;
        top.clk_i = 1;
        top.eval();
        ++time;
    }
};
}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    std::size_t output_count = 0;
    int case_index = -1;
    std::string terminal_path;
    try {
        const Arguments args = parse_arguments(argc, argv);
        case_index = args.case_index;
        terminal_path = args.terminal;
        const auto activations = read_memh(args.activations);
        const auto weights = read_memh(args.weights);
        if (activations.size() != 4 * kHiddenSize || weights.size() != kHiddenSize)
            throw std::runtime_error("input vector geometry mismatch");
        std::ofstream raw(args.raw, std::ios::trunc);
        if (!raw)
            throw std::runtime_error("cannot open raw output");

        Simulation simulation;
        simulation.top.rst_ni = 0;
        simulation.top.clear_i = 0;
        simulation.top.start_valid_i = 0;
        simulation.top.in_valid_i = 0;
        simulation.top.out_ready_i = 1;
        for (int cycle = 0; cycle < 4; ++cycle) simulation.cycle();
        simulation.top.rst_ni = 1;
        while (!simulation.top.start_ready_o) simulation.cycle();
        simulation.top.start_valid_i = 1;
        simulation.cycle();
        simulation.top.start_valid_i = 0;

        for (std::size_t index = 0; index < kHiddenSize; ++index) {
            while (!simulation.top.in_ready_o) simulation.cycle();
            simulation.top.activation_f16_i =
                activations[static_cast<std::size_t>(case_index) * kHiddenSize + index];
            simulation.top.weight_f16_i = weights[index];
            simulation.top.in_valid_i = 1;
            simulation.cycle();
            simulation.top.in_valid_i = 0;
        }

        std::size_t guard = 0;
        while (output_count < kHiddenSize) {
            simulation.top.clk_i = 0;
            simulation.top.eval();
            if (simulation.top.out_valid_o && simulation.top.out_ready_i) {
                if (simulation.top.out_index_o != output_count)
                    throw std::runtime_error("output index mismatch");
                raw << output_count << ' ' << std::hex << std::setw(4)
                    << std::setfill('0') << simulation.top.out_f16_o << std::dec << '\n';
                ++output_count;
            }
            simulation.top.clk_i = 1;
            simulation.top.eval();
            if (++guard > 20000)
                throw std::runtime_error("simulation timeout");
        }
        raw.flush();
        write_terminal(args.terminal, true, 0, output_count, case_index);
        simulation.top.final();
        return 0;
    } catch (const std::exception &error) {
        if (!terminal_path.empty()) {
            try {
                write_terminal(terminal_path, false, 1, output_count, case_index);
            } catch (...) {
            }
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
