#include "Vace3_model24_layer_controller.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint64_t cycles = 0;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void settle(Vace3_model24_layer_controller& top) {
    top.clk_i = 0;
    top.eval();
}

void tick(Vace3_model24_layer_controller& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    top.clk_i = 0;
    top.eval();
    ++cycles;
}

void idle_inputs(Vace3_model24_layer_controller& top) {
    top.clear_i = 0;
    top.start_valid_i = 0;
    top.start_cache_slot_i = 0;
    top.start_position_i = 0;
    top.layer_start_ready_i = 0;
    top.layer_done_valid_i = 0;
    top.layer_done_index_i = 0;
    top.layer_done_fault_i = 0;
    top.checkpoint_ready_i = 0;
    top.done_ready_i = 0;
}

void reset(Vace3_model24_layer_controller& top) {
    top.rst_ni = 0;
    tick(top);
    tick(top);
    top.rst_ni = 1;
    settle(top);
    require(top.start_ready_o && !top.busy_o && !top.fault_o,
            "reset did not restore idle");
}

void clear(Vace3_model24_layer_controller& top) {
    top.clear_i = 1;
    tick(top);
    top.clear_i = 0;
    settle(top);
    require(top.start_ready_o && !top.busy_o && !top.fault_o,
            "clear did not restore idle");
}

void start(Vace3_model24_layer_controller& top) {
    top.start_cache_slot_i = 1;
    top.start_position_i = 127;
    top.start_valid_i = 1;
    settle(top);
    require(top.start_ready_o, "legal start was not ready");
    tick(top);
    top.start_valid_i = 0;
    settle(top);
}

std::string argument_value(
    int argc,
    char** argv,
    const std::string& prefix
) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument.compare(0, prefix.size(), prefix) == 0)
            return argument.substr(prefix.size());
    }
    throw std::runtime_error("missing argument " + prefix);
}

std::vector<uint16_t> load_events(const std::string& path) {
    std::ifstream stream(path);
    require(stream.good(), "unable to open cascade event vectors");
    std::vector<uint16_t> events;
    std::string line;
    while (std::getline(stream, line)) {
        require(line.size() == 4, "malformed cascade event vector");
        events.push_back(static_cast<uint16_t>(std::stoul(line, nullptr, 16)));
    }
    require(events.size() == 24, "cascade event vector count mismatch");
    return events;
}

void write_event(std::ofstream& stream, uint32_t word) {
    stream << std::hex << std::setfill('0') << std::setw(8) << word << '\n';
    stream.flush();
    require(stream.good(), "unable to persist controller event");
}
}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const std::string vector_dir = argument_value(argc, argv, "+VECTOR_DIR=");
    const std::string raw_dir = argument_value(argc, argv, "+RAW_DIR=");
    const std::vector<uint16_t> expected =
        load_events(vector_dir + "/cascade_events.hex");
    std::ofstream events(raw_dir + "/controller_events.hex");
    std::ofstream terminal(raw_dir + "/terminal.txt");
    require(events.good() && terminal.good(), "unable to open raw evidence");

    Vace3_model24_layer_controller top;
    idle_inputs(top);
    reset(top);
    start(top);

    for (unsigned layer = 0; layer < 24; ++layer) {
        require(
            top.layer_start_valid_o
                && top.layer_start_index_o == layer
                && top.layer_start_cache_slot_o == 1
                && top.layer_start_position_o == 127,
            "layer launch metadata mismatch"
        );
        tick(top);
        tick(top);
        require(top.layer_start_valid_o && top.layer_start_index_o == layer,
                "layer launch was not retained under backpressure");
        top.layer_start_ready_i = 1;
        settle(top);
        write_event(events, 0x10000000U | layer);
        tick(top);
        top.layer_start_ready_i = 0;
        settle(top);
        require(top.layer_done_ready_o, "layer completion was not enabled");

        top.layer_done_index_i = layer;
        top.layer_done_fault_i = 0;
        top.layer_done_valid_i = 1;
        tick(top);
        top.layer_done_valid_i = 0;
        settle(top);
        require(
            top.checkpoint_valid_o
                && top.checkpoint_completed_layer_o == layer
                && top.checkpoint_next_layer_o == layer + 1
                && top.checkpoint_terminal_o == (layer == 23),
            "checkpoint metadata mismatch"
        );
        const uint32_t checkpoint_word =
            0x20000000U
            | (static_cast<uint32_t>(top.checkpoint_terminal_o) << 10)
            | (static_cast<uint32_t>(top.checkpoint_next_layer_o) << 5)
            | static_cast<uint32_t>(top.checkpoint_completed_layer_o);
        require(
            (checkpoint_word & 0xffffU) == expected[layer],
            "checkpoint differs from independent vector"
        );
        tick(top);
        tick(top);
        require(top.checkpoint_valid_o && !top.layer_start_valid_o,
                "checkpoint gating failed under backpressure");
        top.checkpoint_ready_i = 1;
        settle(top);
        write_event(events, checkpoint_word);
        tick(top);
        top.checkpoint_ready_i = 0;
        settle(top);
    }

    require(top.done_valid_o && top.active_layer_o == 23,
            "terminal layer completion is missing");
    tick(top);
    tick(top);
    require(top.done_valid_o, "terminal completion was not retained");
    top.done_ready_i = 1;
    settle(top);
    write_event(events, 0x30000000U | top.active_layer_o);
    tick(top);
    top.done_ready_i = 0;
    settle(top);
    require(top.start_ready_o && !top.busy_o,
            "controller did not return to idle");

    top.start_cache_slot_i = 2;
    top.start_position_i = 128;
    top.start_valid_i = 1;
    tick(top);
    top.start_valid_i = 0;
    settle(top);
    require(top.fault_o && !top.start_ready_o,
            "out-of-range start did not fail closed");
    clear(top);

    top.layer_done_valid_i = 1;
    tick(top);
    top.layer_done_valid_i = 0;
    settle(top);
    require(top.fault_o, "unsolicited completion did not fault");
    clear(top);

    start(top);
    top.layer_start_ready_i = 1;
    tick(top);
    top.layer_start_ready_i = 0;
    top.layer_done_index_i = 1;
    top.layer_done_valid_i = 1;
    tick(top);
    top.layer_done_valid_i = 0;
    settle(top);
    require(top.fault_o && !top.checkpoint_valid_o,
            "mismatched layer completion did not fault");
    clear(top);

    start(top);
    top.layer_start_ready_i = 1;
    tick(top);
    top.layer_start_ready_i = 0;
    reset(top);
    require(top.active_layer_o == 0, "active reset retained a layer index");

    start(top);
    top.layer_start_ready_i = 1;
    tick(top);
    top.layer_start_ready_i = 0;
    top.layer_done_index_i = 0;
    top.layer_done_fault_i = 1;
    top.layer_done_valid_i = 1;
    tick(top);
    top.layer_done_valid_i = 0;
    top.layer_done_fault_i = 0;
    settle(top);
    require(top.fault_o && !top.checkpoint_valid_o,
            "faulted completion did not fail closed");
    clear(top);

    terminal
        << "schema=ace3_model24_controller_raw_v1 natural_terminal=1 "
        << "exit_code=0 launches=24 checkpoints=24 done=1 terminal_layer=23\n";
    terminal.flush();
    require(terminal.good(), "unable to persist controller terminal");
    top.final();
    std::cout
        << "MODEL24_LAYER_CONTROLLER_VERILATOR_PASS layers=24 checkpoints=24 "
        << "retained_backpressure=pass terminal=layer23 fail_closed=pass "
        << "clear_recovery=pass cycles=" << cycles
        << " numerical_rtl=not_claimed\n";
    return 0;
}
