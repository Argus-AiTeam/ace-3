#include "Vace3_decoder_layer0_token_engine.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

struct Probe {
    Vace3_decoder_layer0_token_engine top;
    uint64_t cycles = 0;
    std::array<unsigned, 3> accepts{};
    std::array<bool, 3> loaded{};
    unsigned attempted_kind = 0, attempted_index = 0;
    unsigned accepted_kind = 0, accepted_index = 0;

    Probe() {
        top.clk_i=0; top.rst_ni=1; top.clear_i=0;
        top.load_valid_i=0; top.load_kind_i=0; top.load_index_i=0; top.load_f16_i=0;
        top.start_valid_i=0; top.start_cache_slot_i=0; top.start_position_i=0;
        top.projection_meta_valid_i=0; top.projection_qzeros_i=0; top.projection_scale_f16_i=0;
        top.projection_pair_valid_i=0; top.projection_qweight_i=0;
        top.projection_bias_valid_i=0; top.projection_bias_f16_i=0;
        top.rope_valid_i=0; top.rope_cos_f16_i=0; top.rope_sin_f16_i=0;
        top.trace_ready_i=1; top.final_ready_i=1; top.done_ready_i=1;
    }

    std::string progress() const {
        return "kind=" + std::to_string(attempted_kind) +
               " index=" + std::to_string(attempted_index) +
               " load_ready=" + std::to_string(unsigned(top.load_ready_o)) +
               " phase=" + std::to_string(unsigned(top.phase_o)) +
               " accepts=" + std::to_string(accepts[0]) + "," +
               std::to_string(accepts[1]) + "," + std::to_string(accepts[2]);
    }

    void eval_low() {
        top.clk_i=0;
        top.eval();
    }

    bool tick() {
        eval_low();
        const bool load_accept=top.load_valid_i && top.load_ready_o;
        const bool start_accept=top.start_valid_i && top.start_ready_o;
        const unsigned kind=top.load_kind_i;
        const unsigned index=top.load_index_i;
        top.clk_i=1;
        top.eval();
        ++cycles;
        if (load_accept) {
            ++accepts[kind];
            accepted_kind=kind;
            accepted_index=index;
            if (index==895) loaded[kind]=true;
        }
        if (start_accept) loaded[0]=false;
        return load_accept;
    }

    void reset() {
        top.rst_ni=0;
        top.eval();
        if (top.load_ready_o || top.start_ready_o || top.busy_o || top.phase_o)
            throw std::runtime_error("reset outputs not fail-closed");
        tick();
        tick();
        top.rst_ni=1;
        eval_low();
        if (top.start_ready_o || top.phase_o)
            throw std::runtime_error("reset state invalid");
    }

    void clear() {
        top.clear_i=1;
        tick();
        top.clear_i=0;
        loaded.fill(false);
        eval_low();
        if (top.start_ready_o || top.busy_o || top.phase_o)
            throw std::runtime_error("clear did not revoke preload state");
    }

    void reject(unsigned kind, unsigned index) {
        attempted_kind=kind; attempted_index=index;
        top.load_kind_i=kind; top.load_index_i=index; top.load_f16_i=0x1234;
        top.load_valid_i=1; eval_low();
        if (top.load_ready_o)
            throw std::runtime_error("out-of-sequence load accepted " + progress());
        tick();
        top.load_valid_i=0;
    }

    void load(unsigned kind) {
        const uint64_t deadline=cycles+4096;
        for (unsigned index=0; index<896; ++index) {
            attempted_kind=kind; attempted_index=index;
            top.load_kind_i=kind; top.load_index_i=index;
            top.load_f16_i=uint16_t((kind<<13)^index);
            top.load_valid_i=1;
            while (true) {
                if (cycles>=deadline)
                    throw std::runtime_error("DECODER_PRELOAD_TIMEOUT " + progress());
                if (tick()) break;
            }
            top.load_valid_i=0;
        }
        eval_low();
    }

    void start() {
        top.start_valid_i=1;
        eval_low();
        if (!top.start_ready_o)
            throw std::runtime_error("start prerequisites rejected " + progress());
        tick();
        top.start_valid_i=0;
        if (!top.busy_o || top.phase_o!=1)
            throw std::runtime_error("vacuous start transition " + progress());
    }

    void legacy_edge_probe() {
        load(1);
        clear();
        top.load_valid_i=1; top.load_kind_i=1; top.load_f16_i=0x1234;
        top.load_index_i=0; tick();
        top.load_index_i=1; eval_low();
        if (!top.load_ready_o) throw std::runtime_error("legacy probe pre-edge ready missing");
        tick();
        if (top.load_ready_o)
            throw std::runtime_error("legacy probe post-edge ready unexpectedly retained");
        std::cout << "DECODER_PRELOAD_LEGACY_EDGE kind=1 index=1 accepted=1"
                  << " post_edge_load_ready=0 next_expected_index=2"
                  << " legacy_wait_condition=post_accept_recheck\n";
        top.load_valid_i=0;
        clear();
        accepts.fill(0);
    }
};

int main(int argc, char** argv) {
    try {
        Verilated::commandArgs(argc,argv);
        bool expect_timeout=false;
        for (int i=1;i<argc;++i)
            if (std::string(argv[i])=="--expect-timeout") expect_timeout=true;
        Probe probe;
        probe.reset();
        if (expect_timeout) {
            probe.attempted_kind=1; probe.attempted_index=1;
            probe.top.load_kind_i=1; probe.top.load_index_i=1;
            probe.top.load_f16_i=0x1234; probe.top.load_valid_i=1;
            for (unsigned wait=0;wait<8;++wait) {
                probe.eval_low();
                if (probe.top.load_ready_o)
                    throw std::runtime_error("expected timeout load became ready");
                probe.tick();
            }
            throw std::runtime_error("DECODER_PRELOAD_TIMEOUT " + probe.progress());
        }
        probe.legacy_edge_probe();
        probe.reject(1,1);
        probe.reject(3,0);
        probe.load(1);
        if (probe.top.start_ready_o || !probe.loaded[1])
            throw std::runtime_error("norm1 completion prerequisite failure");
        probe.load(2);
        if (probe.top.start_ready_o || !probe.loaded[2])
            throw std::runtime_error("norm2 completion prerequisite failure");
        probe.load(0);
        if (!probe.top.start_ready_o || !probe.loaded[0] ||
            probe.accepts != std::array<unsigned, 3>{896, 896, 896})
            throw std::runtime_error("activation completion prerequisite failure");
        probe.start();
        probe.clear();
        probe.load(1); probe.load(2); probe.load(0);
        if (!probe.top.start_ready_o ||
            probe.accepts != std::array<unsigned, 3>{1792, 1792, 1792})
            throw std::runtime_error("clear/reload acceptance count failure " + probe.progress());
        probe.start();
        probe.top.final();
        std::cout << "DECODER_PRELOAD_VERILATOR_PASS accepts="
                  << probe.accepts[0] << "," << probe.accepts[1] << "," << probe.accepts[2]
                  << " epoch_handshakes=2688 total_handshakes=5376 epochs=2"
                  << " last_kind=" << probe.accepted_kind
                  << " last_index=" << probe.accepted_index
                  << " reset=pass sequence=1,2,0 flags=pass ready_known=two_state"
                  << " final_index=895 start_transition=0_to_1 clear_reload=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DECODER_PRELOAD_VERILATOR_FAIL " << error.what() << "\n";
        return 1;
    }
}
