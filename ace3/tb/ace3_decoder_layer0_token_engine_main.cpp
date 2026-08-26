#include "Vace3_decoder_layer0_token_engine.h"
#include "ace3_layer0_trace_capture_policy.h"
#include "verilated.h"
#include "verilated_save.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Projection {
    std::vector<uint32_t> qweight, qzeros;
    std::vector<uint16_t> scales, bias;
};

static constexpr std::array<unsigned, 7> projection_inputs{
    896, 896, 896, 896, 896, 896, 4864};
static constexpr std::array<unsigned, 7> projection_outputs{
    896, 128, 128, 896, 4864, 4864, 896};
static constexpr std::array<unsigned, 7> projection_groups{
    7, 7, 7, 7, 7, 7, 38};
static constexpr std::array<unsigned, 7> projection_words{
    112, 16, 16, 112, 608, 608, 112};

static uint64_t cycles, stalls, failures, trace_count, final_count, done_count;
static unsigned active_layer_index;
static std::array<uint64_t, 37> phase_cycles{};
static bool checking;
static std::array<uint64_t, 2> token_start{}, token_done{};
static std::array<uint64_t, 3> load_accepts{};
static uint64_t progress_interval, preload_timeout_cycles = 4096;
static uint64_t start_timeout_cycles = 100000;
static uint64_t start_attempts, start_accepts;
static unsigned attempted_load_kind, attempted_load_index;
static unsigned accepted_load_kind, accepted_load_index;

static std::vector<std::string> lines(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> result; std::string line;
    while (std::getline(file, line)) result.push_back(line);
    return result;
}
template <typename T> static std::vector<T> hex_file(const std::string& path) {
    std::vector<T> result;
    for (const auto& line : lines(path)) result.push_back(static_cast<T>(std::stoull(line, nullptr, 16)));
    return result;
}
static std::string tensor(const std::string& dir, const std::string& name) {
    return dir + "/tensors/" + name + ".hex";
}
static Projection projection(const std::string& dir, const std::string& prefix, bool bias) {
    Projection p;
    p.qweight = hex_file<uint32_t>(tensor(dir, prefix + "_qweight.i32le.bin"));
    p.qzeros = hex_file<uint32_t>(tensor(dir, prefix + "_qzeros.i32le.bin"));
    p.scales = hex_file<uint16_t>(tensor(dir, prefix + "_scales.fp16le.bin"));
    if (bias) p.bias = hex_file<uint16_t>(tensor(dir, prefix + "_bias.fp16le.bin"));
    return p;
}
static uint16_t at(const std::vector<uint16_t>& values, size_t index, const char* name) {
    if (index >= values.size()) throw std::runtime_error(std::string("tensor address out of range: ") + name);
    return values[index];
}
static uint32_t at(const std::vector<uint32_t>& values, size_t index, const char* name) {
    if (index >= values.size()) throw std::runtime_error(std::string("tensor address out of range: ") + name);
    return values[index];
}
template <typename T, size_t N>
static T at(const std::array<T, N>& values, size_t index, const char* name) {
    if (index >= values.size()) throw std::runtime_error(std::string("tensor address out of range: ") + name);
    return values[index];
}

struct Harness {
    Vace3_decoder_layer0_token_engine top;
    std::array<std::vector<uint16_t>, 2> inputs;
    std::vector<uint16_t> norm1, norm2;
    std::array<Projection, 7> p;
    std::array<uint16_t, 128 * 32> rope_cos{}, rope_sin{};
    std::ofstream raw_trace, raw_final;
    bool fail_after_raw;
    bool transaction;
    unsigned transaction_position;
    bool trace_hold = false, final_hold = false, done_hold = false;
    std::array<uint64_t, 4> trace_held{}, final_held{}, done_held{};
    std::array<bool, 3> loaded{};

    std::string progress() const {
        return "phase=" + std::to_string(unsigned(top.phase_o)) +
               " load_ready=" + std::to_string(unsigned(top.load_ready_o)) +
               " attempted_load_kind=" + std::to_string(attempted_load_kind) +
               " attempted_load_index=" + std::to_string(attempted_load_index) +
               " accepted_load_kind=" + std::to_string(accepted_load_kind) +
               " accepted_load_index=" + std::to_string(accepted_load_index) +
               " load_accepts=" + std::to_string(load_accepts[0]) + "," +
               std::to_string(load_accepts[1]) + "," +
               std::to_string(load_accepts[2]) +
               " loaded=" + std::to_string(loaded[0]) + "," +
               std::to_string(loaded[1]) + "," + std::to_string(loaded[2]) +
               " start_ready=" + std::to_string(unsigned(top.start_ready_o)) +
               " start_attempts=" + std::to_string(start_attempts) +
               " start_accepts=" + std::to_string(start_accepts) +
               " trace=" + std::to_string(trace_count) +
               " final=" + std::to_string(final_count) +
               " done=" + std::to_string(done_count);
    }

    Harness(const std::string& dir, const std::string& raw_dir, bool inject_failure,
            unsigned layer_index, bool transaction_mode = false,
            unsigned position = 0, const std::string& input_path = "",
            const std::string& rope_path = "")
        : raw_trace(raw_dir + "/trace.hex", std::ios::trunc),
          raw_final(raw_dir + "/final.hex", std::ios::trunc),
          fail_after_raw(inject_failure), transaction(transaction_mode),
          transaction_position(position) {
        if (!raw_trace || !raw_final)
            throw std::runtime_error("cannot open raw output files");
        const auto input_lines = lines(
            input_path.empty() ? dir + "/inputs.hex" : input_path);
        const auto rope_lines = lines(
            rope_path.empty() ? dir + "/rope_coefficients.hex" : rope_path);
        const size_t expected_inputs = transaction ? 896 : 1792;
        const size_t expected_rope = transaction ? 32 : 64;
        if (input_lines.size() != expected_inputs ||
            rope_lines.size() != expected_rope)
            throw std::runtime_error("decoder input vector count mismatch");
        inputs[0].resize(896); inputs[1].resize(896);
        std::array<std::array<bool, 896>, 2> input_seen{};
        for (const auto& line : input_lines) {
            const unsigned token = std::stoul(line.substr(0, 2), nullptr, 16);
            const unsigned index = std::stoul(line.substr(2, 4), nullptr, 16);
            if (token > (transaction ? 0u : 1u) || index >= 896 ||
                input_seen[token][index])
                throw std::runtime_error("bad input record");
            input_seen[token][index] = true;
            inputs[token][index] = std::stoul(line.substr(6, 4), nullptr, 16);
        }
        for (unsigned token=0; token<(transaction ? 1u : 2u); ++token)
            for (unsigned index=0; index<896; ++index)
                if (!input_seen[token][index])
                    throw std::runtime_error("missing input record");
        std::array<std::array<bool, 32>, 128> rope_seen{};
        for (const auto& line : rope_lines) {
            const unsigned pos = std::stoul(line.substr(0,4),nullptr,16);
            const unsigned pair = std::stoul(line.substr(4,2),nullptr,16);
            if (pos >= 128 || pair >= 32 ||
                (transaction && pos != transaction_position) ||
                rope_seen[pos][pair])
                throw std::runtime_error("bad rope record");
            rope_seen[pos][pair] = true;
            rope_cos[pos*32+pair] = std::stoul(line.substr(6,4),nullptr,16);
            rope_sin[pos*32+pair] = std::stoul(line.substr(10,4),nullptr,16);
        }
        const unsigned first_position = transaction ? transaction_position : 0;
        const unsigned last_position = transaction ? transaction_position : 1;
        for (unsigned pos=first_position; pos<=last_position; ++pos)
            for (unsigned pair=0; pair<32; ++pair)
                if (!rope_seen[pos][pair])
                    throw std::runtime_error("missing rope record");
        const std::string layer = "layer" + std::to_string(layer_index) + "_";
        norm1 = hex_file<uint16_t>(tensor(dir, layer + "input_layernorm_weight.fp16le.bin"));
        norm2 = hex_file<uint16_t>(tensor(dir, layer + "post_attention_layernorm_weight.fp16le.bin"));
        p[0]=projection(dir,layer+"self_attn_q_proj",true);
        p[1]=projection(dir,layer+"self_attn_k_proj",true);
        p[2]=projection(dir,layer+"self_attn_v_proj",true);
        p[3]=projection(dir,layer+"self_attn_o_proj",false);
        p[4]=projection(dir,layer+"mlp_gate_proj",false);
        p[5]=projection(dir,layer+"mlp_up_proj",false);
        p[6]=projection(dir,layer+"mlp_down_proj",false);
        if (norm1.size()!=896 || norm2.size()!=896)
            throw std::runtime_error("decoder normalization tensor geometry mismatch");
        for (size_t kind=0; kind<p.size(); ++kind) {
            const size_t qweight_size=size_t(projection_inputs[kind])*projection_words[kind];
            const size_t qzeros_size=size_t(projection_groups[kind])*projection_words[kind];
            const size_t scales_size=size_t(projection_groups[kind])*projection_outputs[kind];
            const size_t bias_size=kind<=2 ? projection_outputs[kind] : 0;
            if (p[kind].qweight.size()!=qweight_size || p[kind].qzeros.size()!=qzeros_size ||
                p[kind].scales.size()!=scales_size || p[kind].bias.size()!=bias_size)
                throw std::runtime_error("decoder projection tensor geometry mismatch kind=" +
                                         std::to_string(kind));
            if (projection_groups[kind]-1>63 || projection_words[kind]-1>1023 ||
                qzeros_size-1>65535)
                throw std::runtime_error("decoder qzeros address width mismatch kind=" +
                                         std::to_string(kind));
        }
    }
    void idle() {
        top.clear_i=0; top.load_valid_i=0; top.load_kind_i=0; top.load_index_i=0; top.load_f16_i=0;
        top.start_valid_i=0; top.start_cache_slot_i=0; top.start_position_i=0;
        top.projection_meta_valid_i=0; top.projection_pair_valid_i=0; top.projection_bias_valid_i=0;
        top.rope_valid_i=0; top.trace_ready_i=0; top.final_ready_i=0; top.done_ready_i=0;
        top.projection_qzeros_i=0; top.projection_scale_f16_i=0; top.projection_qweight_i=0;
        top.projection_bias_f16_i=0; top.rope_cos_f16_i=0; top.rope_sin_f16_i=0;
    }
    void drive() {
        top.projection_meta_valid_i=(cycles%23)!=0; top.projection_pair_valid_i=(cycles%71)!=0;
        top.projection_bias_valid_i=(cycles%37)!=0; top.rope_valid_i=(cycles%29)!=0;
        top.trace_ready_i=(cycles%17)!=0; top.final_ready_i=(cycles%19)!=0; top.done_ready_i=(cycles%13)!=0;
        top.projection_qzeros_i=0; top.projection_scale_f16_i=0; top.projection_qweight_i=0;
        top.projection_bias_f16_i=0; top.rope_cos_f16_i=0; top.rope_sin_f16_i=0;
        top.eval();

        const unsigned kind=top.projection_kind_o;
        if ((top.projection_meta_valid_i&&top.projection_meta_ready_o) ||
            (top.projection_pair_valid_i&&top.projection_pair_ready_o) ||
            (top.projection_bias_valid_i&&top.projection_bias_ready_o)) {
            if (kind>6) throw std::runtime_error("projection kind out of range");
        }
        if (top.projection_meta_valid_i&&top.projection_meta_ready_o) {
            const size_t group=top.projection_meta_group_o;
            const size_t word=top.projection_meta_word_o;
            const size_t output=top.projection_meta_output_channel_o;
            if (group>=projection_groups[kind] || word>=projection_words[kind] ||
                output>=projection_outputs[kind])
                throw std::runtime_error("live projection metadata address out of range");
            top.projection_qzeros_i=at(
                p[kind].qzeros,group*projection_words[kind]+word,"qzeros");
            top.projection_scale_f16_i=at(
                p[kind].scales,group*projection_outputs[kind]+output,"scales");
        }
        if (top.projection_pair_valid_i&&top.projection_pair_ready_o) {
            const size_t input=top.projection_pair_input_o;
            const size_t word=top.projection_pair_word_o;
            if (input>=projection_inputs[kind] || word>=projection_words[kind])
                throw std::runtime_error("live projection pair address out of range");
            top.projection_qweight_i=at(
                p[kind].qweight,input*projection_words[kind]+word,"qweight");
        }
        if (top.projection_bias_valid_i&&top.projection_bias_ready_o) {
            const size_t output=top.projection_bias_output_channel_o;
            if (output>=p[kind].bias.size())
                throw std::runtime_error("live projection bias address out of range");
            top.projection_bias_f16_i=at(p[kind].bias,output,"bias");
        }
        if (top.rope_valid_i&&top.rope_ready_o) {
            const size_t rope_address=size_t(top.rope_position_o)*32+top.rope_pair_o;
            top.rope_cos_f16_i=at(rope_cos,rope_address,"rope cosine");
            top.rope_sin_f16_i=at(rope_sin,rope_address,"rope sine");
        }
        top.eval();
    }
    void mismatch(const std::string& message) {
        if (failures++ < 10) std::cerr << "DECODER_CONTROLLER_MISMATCH cycle=" << cycles << " " << message << "\n";
    }
    void observe() {
        if (trace_hold && (!top.trace_valid_o || top.trace_stage_o!=trace_held[0] ||
            top.trace_index_o!=trace_held[1] || top.trace_f16_o!=trace_held[2] ||
            top.trace_position_o!=trace_held[3]))
            mismatch("trace retained output changed");
        if (final_hold && (!top.final_valid_o || top.final_index_o!=final_held[0] ||
            top.final_f16_o!=final_held[1] || top.final_last_o!=final_held[2]))
            mismatch("final retained output changed");
        if (done_hold && (!top.done_valid_o || top.done_cache_slot_o!=done_held[0] ||
            top.done_position_o!=done_held[1] || top.done_cycles_o!=done_held[2] ||
            top.done_stall_cycles_o!=done_held[3]))
            mismatch("done retained output changed");
        if (top.busy_o && top.phase_o<=36) ++phase_cycles[top.phase_o];
        if ((top.projection_meta_ready_o&&!top.projection_meta_valid_i) ||
            (top.projection_pair_ready_o&&!top.projection_pair_valid_i) ||
            (top.projection_bias_ready_o&&!top.projection_bias_valid_i) ||
            (top.rope_ready_o&&!top.rope_valid_i) || (top.trace_valid_o&&!top.trace_ready_i) ||
            (top.final_valid_o&&(!top.final_ready_i||!top.trace_ready_i)) ||
            (top.done_valid_o&&!top.done_ready_i)) ++stalls;
        if (layer0_trace_capture_accept(
                checking, top.trace_valid_o, top.trace_ready_i,
                top.final_valid_o, top.final_ready_i)) {
            raw_trace << std::hex << std::setfill('0')
                      << std::setw(2) << done_count
                      << std::setw(4) << unsigned(top.trace_position_o)
                      << std::setw(2) << unsigned(top.trace_stage_o)
                      << std::setw(4) << unsigned(top.trace_index_o)
                      << std::setw(4) << unsigned(top.trace_f16_o) << '\n';
            if (!raw_trace) throw std::runtime_error("raw trace write failed");
            ++trace_count;
            if (fail_after_raw && trace_count==1) {
                raw_trace.flush();
                throw std::runtime_error("injected failure after raw trace");
            }
        }
        if (layer0_final_capture_accept(
                checking, top.final_valid_o, top.final_ready_i,
                top.trace_ready_i)) {
            raw_final << std::hex << std::setfill('0')
                      << std::setw(2) << done_count
                      << std::setw(4) << unsigned(top.final_index_o)
                      << std::setw(4) << unsigned(top.final_f16_o) << '\n';
            if (!raw_final) throw std::runtime_error("raw final write failed");
            if (top.final_last_o != (top.final_index_o==895))
                mismatch("final record=" + std::to_string(final_count));
            ++final_count;
        }
        if (checking && top.done_valid_o && top.done_ready_i) {
            const uint64_t expected_position =
                transaction ? transaction_position : done_count;
            if (top.done_cache_slot_o!=0 ||
                top.done_position_o!=expected_position ||
                top.done_cycles_o==0 || top.done_cycles_o<=top.done_stall_cycles_o)
                mismatch("done metadata");
            if (done_count<2) token_done[done_count]=cycles;
            ++done_count;
        }
        trace_hold=top.trace_valid_o&&!top.trace_ready_i; final_hold=top.final_valid_o&&(!top.final_ready_i||!top.trace_ready_i);
        done_hold=top.done_valid_o&&!top.done_ready_i;
        if (trace_hold)
            trace_held={top.trace_stage_o,top.trace_index_o,top.trace_f16_o,top.trace_position_o};
        if (final_hold)
            final_held={top.final_index_o,top.final_f16_o,top.final_last_o,0};
        if (done_hold)
            done_held={top.done_cache_slot_o,top.done_position_o,top.done_cycles_o,
                       top.done_stall_cycles_o};
    }
    bool tick() {
        top.clk_i=0; drive(); top.eval();
        const bool load_accept = top.load_valid_i && top.load_ready_o;
        const bool start_accept = top.start_valid_i && top.start_ready_o;
        const unsigned load_kind = top.load_kind_i;
        const unsigned load_index = top.load_index_i;
        observe(); top.clk_i=1; top.eval(); ++cycles;
        if (top.phase_o==63)
            throw std::runtime_error("controller fault " + progress());
        if (load_accept) {
            accepted_load_kind=load_kind; accepted_load_index=load_index;
            ++load_accepts[load_kind];
            if (load_index==895) loaded[load_kind]=true;
        }
        if (start_accept) {
            ++start_accepts;
            loaded[0]=false;
        }
        if (progress_interval && cycles % progress_interval == 0)
            std::cerr << "DECODER_PROGRESS cycles=" << cycles
                      << " projection_kind=" << unsigned(top.projection_kind_o)
                      << " " << progress() << "\n";
        return load_accept;
    }
    void reset() {
        idle(); top.rst_ni=0; top.eval();
        if (top.load_ready_o||top.start_ready_o||top.busy_o||top.done_valid_o) mismatch("async reset");
        tick(); tick(); top.rst_ni=1; drive(); top.eval();
    }
    void load(unsigned kind, unsigned token) {
        const uint64_t deadline=cycles+preload_timeout_cycles;
        for (unsigned index=0; index<896; ++index) {
            attempted_load_kind=kind; attempted_load_index=index;
            top.load_kind_i=kind; top.load_index_i=index;
            top.load_f16_i=(kind==0?inputs[token][index]:(kind==1?norm1[index]:norm2[index]));
            top.load_valid_i=1;
            while (true) {
                if (cycles>=deadline)
                    throw std::runtime_error("preload timeout " + progress());
                if (tick()) break;
            }
            top.load_valid_i=0;
        }
    }
    void start(unsigned slot, unsigned position, unsigned token, bool must_accept=true) {
        ++start_attempts;
        top.start_cache_slot_i=slot; top.start_position_i=position; top.start_valid_i=1; drive(); top.eval();
        if (bool(top.start_ready_o)!=must_accept) mismatch("start acceptance slot="+std::to_string(slot)+" position="+std::to_string(position));
        if (must_accept && !top.start_ready_o)
            throw std::runtime_error("required start rejected " + progress());
        if (top.start_ready_o) {
            if (checking) token_start[token]=cycles;
            tick();
            if (!top.busy_o || top.phase_o==0)
                throw std::runtime_error("vacuous start transition " + progress());
        }
        top.start_valid_i=0;
    }
    void clear_after_work() {
        const uint64_t deadline=cycles+start_timeout_cycles;
        while (!(top.busy_o && (top.projection_meta_ready_o||top.projection_pair_ready_o||top.rope_ready_o))) {
            if (cycles>=deadline)
                throw std::runtime_error("start progress timeout " + progress());
            tick();
        }
        top.clear_i=1; tick(); top.clear_i=0; drive(); top.eval();
        loaded.fill(false);
        if (top.busy_o||top.done_valid_o||top.start_ready_o) mismatch("clear abort");
    }
    void finish_unchecked() {
        uint64_t limit=cycles+40000000;
        while (top.busy_o||top.done_valid_o) { if (cycles>limit) throw std::runtime_error("timeout unchecked token"); tick(); }
    }
    void finish_checked(unsigned expected_done) {
        uint64_t limit=cycles+40000000;
        while (done_count<expected_done) { if (cycles>limit) throw std::runtime_error("timeout authenticated token"); tick(); }
    }
    void close_raw() {
        raw_trace.close();
        raw_final.close();
        if (!raw_trace || !raw_final)
            throw std::runtime_error("raw output close failed");
    }
    void restore_state(const std::string& path) {
        VerilatedRestore stream;
        stream.open(path.c_str());
        stream >> top;
        stream.close();
        idle();
        top.eval();
        if (top.busy_o || top.done_valid_o || top.phase_o != 0)
            throw std::runtime_error("restored transaction state is not idle");
    }
    void save_state(const std::string& path) {
        if (top.busy_o || top.done_valid_o || top.phase_o != 0)
            throw std::runtime_error("refusing to save nonterminal transaction state");
        const std::string partial = path + ".partial";
        std::remove(partial.c_str());
        VerilatedSave stream;
        stream.open(partial.c_str());
        stream << top;
        stream.close();
        std::ifstream saved(partial, std::ios::binary | std::ios::ate);
        if (!saved || saved.tellg() <= 0)
            throw std::runtime_error("transaction state save is empty");
        saved.close();
        std::remove(path.c_str());
        if (std::rename(partial.c_str(), path.c_str()) != 0)
            throw std::runtime_error("transaction state commit failed");
    }
};

static void savable_idle(Vace3_decoder_layer0_token_engine& top) {
    top.clear_i=0; top.load_valid_i=0; top.load_kind_i=0; top.load_index_i=0;
    top.load_f16_i=0; top.start_valid_i=0; top.start_cache_slot_i=0;
    top.start_position_i=0; top.projection_meta_valid_i=1;
    top.projection_pair_valid_i=1; top.projection_bias_valid_i=1;
    top.rope_valid_i=1; top.trace_ready_i=1; top.final_ready_i=1;
    top.done_ready_i=1; top.projection_qzeros_i=0;
    top.projection_scale_f16_i=0x3c00; top.projection_qweight_i=0;
    top.projection_bias_f16_i=0; top.rope_cos_f16_i=0x3c00;
    top.rope_sin_f16_i=0;
}

static void savable_tick(Vace3_decoder_layer0_token_engine& top) {
    top.clk_i=0; savable_idle(top); top.eval();
    top.clk_i=1; top.eval();
}

static std::array<uint64_t, 16> savable_observation(
        const Vace3_decoder_layer0_token_engine& top) {
    return {
        top.load_ready_o, top.start_ready_o, top.busy_o, top.phase_o,
        top.projection_kind_o, top.projection_meta_ready_o,
        top.projection_meta_output_channel_o, top.projection_pair_ready_o,
        top.projection_pair_input_o, top.rope_ready_o, top.rope_position_o,
        top.trace_valid_o, top.trace_stage_o, top.trace_index_o,
        top.trace_f16_o, top.done_valid_o,
    };
}

static void run_savable_self_test(const std::string& state_path) {
    Vace3_decoder_layer0_token_engine original;
    savable_idle(original);
    original.rst_ni=0; original.clk_i=0; original.eval();
    savable_tick(original); savable_tick(original);
    original.rst_ni=1;
    for (unsigned kind=0; kind<3; ++kind) {
        for (unsigned index=0; index<896; ++index) {
            original.load_kind_i=kind;
            original.load_index_i=index;
            original.load_f16_i=(kind == 0) ? (0x3000u + (index & 0xffu))
                                             : 0x3c00u;
            original.load_valid_i=1;
            original.clk_i=0; original.eval();
            if (!original.load_ready_o)
                throw std::runtime_error("savable self-test preload rejected");
            original.clk_i=1; original.eval();
            original.load_valid_i=0;
        }
    }
    original.start_cache_slot_i=0; original.start_position_i=0;
    original.start_valid_i=1; original.clk_i=0; original.eval();
    if (!original.start_ready_o)
        throw std::runtime_error("savable self-test start rejected");
    original.clk_i=1; original.eval(); original.start_valid_i=0;
    for (unsigned step=0; step<257; ++step) savable_tick(original);

    const std::string partial = state_path + ".partial";
    std::remove(partial.c_str());
    VerilatedSave save;
    save.open(partial.c_str());
    save << original;
    save.close();
    std::remove(state_path.c_str());
    if (std::rename(partial.c_str(), state_path.c_str()) != 0)
        throw std::runtime_error("savable self-test state commit failed");

    std::vector<std::array<uint64_t, 16>> expected;
    for (unsigned step=0; step<1024; ++step) {
        savable_tick(original);
        expected.push_back(savable_observation(original));
    }

    Vace3_decoder_layer0_token_engine restored;
    VerilatedRestore restore;
    restore.open(state_path.c_str());
    restore >> restored;
    restore.close();
    for (unsigned step=0; step<expected.size(); ++step) {
        savable_tick(restored);
        if (savable_observation(restored) != expected[step])
            throw std::runtime_error(
                "savable self-test restored trajectory diverged at step " +
                std::to_string(step));
    }
    original.final();
    restored.final();
    std::ifstream state(state_path, std::ios::binary | std::ios::ate);
    if (!state || state.tellg() <= 0)
        throw std::runtime_error("savable self-test produced no state");
    std::cout << "DECODER_LAYER_TOKEN_ENGINE_SAVABLE_PASS "
              << "preloaded_words=2688 compared_cycles=" << expected.size()
              << " state_bytes=" << state.tellg() << "\n";
}

static void write_terminal(const std::string& raw_dir, bool natural_terminal,
                           unsigned exit_code) {
    std::ofstream terminal(raw_dir + "/terminal.txt", std::ios::trunc);
    if (!terminal) return;
    if (active_layer_index == 0) {
        terminal << "schema=ace3_decoder_layer0_raw_v1 natural_terminal="
                 << (natural_terminal ? 1 : 0);
    } else {
        terminal << "schema=ace3_decoder_layer_raw_v1 layer_index="
                 << active_layer_index << " natural_terminal="
                 << (natural_terminal ? 1 : 0);
    }
    terminal << " exit_code=" << exit_code
             << " trace_count=" << trace_count
             << " final_count=" << final_count
             << " done_count=" << done_count << '\n';
}

int main(int argc, char** argv) {
    std::string raw_dir;
    try {
        Verilated::commandArgs(argc,argv); std::string dir;
        active_layer_index=0;
        bool fail_after_raw=false;
        bool transaction=false;
        unsigned transaction_position=0;
        std::string transaction_input, transaction_rope, state_in, state_out;
        std::string transaction_metadata, savable_self_test;
        for(int i=1;i<argc;++i) {
            const std::string argument=argv[i];
            if(argument=="--vector-dir" && i+1<argc) dir=argv[++i];
            else if(argument=="--raw-dir" && i+1<argc) raw_dir=argv[++i];
            else if(argument=="--layer-index" && i+1<argc)
                active_layer_index=std::stoul(argv[++i]);
            else if(argument=="--fail-after-raw") fail_after_raw=true;
            else if(argument=="--progress-interval" && i+1<argc)
                progress_interval=std::stoull(argv[++i]);
            else if(argument=="--preload-timeout-cycles" && i+1<argc)
                preload_timeout_cycles=std::stoull(argv[++i]);
            else if(argument=="--start-timeout-cycles" && i+1<argc)
                start_timeout_cycles=std::stoull(argv[++i]);
            else if(argument=="--transaction-position" && i+1<argc) {
                transaction=true;
                transaction_position=std::stoul(argv[++i]);
            } else if(argument=="--transaction-input" && i+1<argc)
                transaction_input=argv[++i];
            else if(argument=="--transaction-rope" && i+1<argc)
                transaction_rope=argv[++i];
            else if(argument=="--state-in" && i+1<argc)
                state_in=argv[++i];
            else if(argument=="--state-out" && i+1<argc)
                state_out=argv[++i];
            else if(argument=="--transaction-metadata" && i+1<argc)
                transaction_metadata=argv[++i];
            else if(argument=="--savable-self-test" && i+1<argc)
                savable_self_test=argv[++i];
        }
        if(!savable_self_test.empty()) {
            run_savable_self_test(savable_self_test);
            return 0;
        }
        if(dir.empty() || raw_dir.empty())
            throw std::runtime_error("usage: --vector-dir PATH --raw-dir PATH");
        if(active_layer_index>23)
            throw std::runtime_error("layer index must be in [0,23]");
        if(!preload_timeout_cycles || !start_timeout_cycles)
            throw std::runtime_error("timeout cycles must be nonzero");
        if(transaction &&
           (transaction_position>=128 || transaction_input.empty() ||
            transaction_rope.empty() || state_out.empty() ||
            transaction_metadata.empty() ||
            (transaction_position==0 && !state_in.empty()) ||
            (transaction_position>0 && state_in.empty()) ||
            state_in==state_out))
            throw std::runtime_error("invalid transaction arguments");
        cycles=stalls=failures=trace_count=final_count=done_count=0; checking=false;
        Harness h(
            dir, raw_dir, fail_after_raw, active_layer_index, transaction,
            transaction_position, transaction_input, transaction_rope);
        h.top.eval();
        if (unsigned(h.top.layer_index_o)!=active_layer_index)
            throw std::runtime_error("RTL layer parameter does not match vector layer");
        if (transaction) {
            if (transaction_position == 0) {
                h.idle(); h.reset();
                h.load(1,0); h.load(2,0);
            } else {
                h.restore_state(state_in);
            }
            h.load(0,0);
            checking=true;
            h.start(0,transaction_position,0);
            h.finish_checked(1);
            checking=false;
            if (failures || final_count!=896 || done_count!=1 || trace_count==0)
                throw std::runtime_error("transaction completion mismatch");
            h.idle(); h.top.eval();
            h.save_state(state_out);
            h.close_raw();
            std::ofstream metadata(transaction_metadata, std::ios::trunc);
            if (!metadata)
                throw std::runtime_error("cannot open transaction metadata");
            metadata
                << "{\"schema_version\":1,"
                << "\"kind\":\"ace3_decoder_verilator_transaction\","
                << "\"layer_index\":" << active_layer_index << ","
                << "\"position\":" << transaction_position << ","
                << "\"next_position\":" << transaction_position + 1 << ","
                << "\"cache_slot\":0,"
                << "\"trace_records\":" << trace_count << ","
                << "\"final_records\":" << final_count << ","
                << "\"done_records\":" << done_count << ","
                << "\"natural_terminal\":true}\n";
            metadata.close();
            if (!metadata)
                throw std::runtime_error("transaction metadata close failed");
            h.top.final();
            std::ofstream terminal(raw_dir + "/terminal.txt", std::ios::trunc);
            terminal
                << "schema=ace3_decoder_transaction_raw_v1 natural_terminal=1 "
                << "exit_code=0 layer_index=" << active_layer_index
                << " position=" << transaction_position
                << " trace_count=" << trace_count
                << " final_count=" << final_count
                << " done_count=" << done_count << '\n';
            std::cout << "DECODER_LAYER_TOKEN_ENGINE_TRANSACTION_PASS layer="
                      << active_layer_index << " position="
                      << transaction_position << " trace_count=" << trace_count
                      << " final_count=" << final_count << "\n";
            return 0;
        }
        h.idle(); h.reset();
        h.top.start_valid_i=1; h.drive(); h.top.eval(); if(h.top.start_ready_o) h.mismatch("incomplete start accepted"); h.top.start_valid_i=0;
        h.load(1,0); h.load(2,0); h.load(0,0);
        h.start(2,0,0,false); h.start(0,1,0,false);
        h.start(0,0,0); h.clear_after_work();
        h.load(1,0); h.load(2,0); h.load(0,0); checking=true; h.start(0,0,0); h.finish_checked(1);
        h.load(0,1); h.start(0,1,1); h.finish_checked(2);
        checking=false; h.load(0,0); h.start(1,0,0); h.clear_after_work();
        if(trace_count!=46676 || final_count!=1792 || done_count!=2 || stalls==0 ||
           token_done[0]<=token_start[0] || token_done[1]<=token_start[1]) h.mismatch("counts or cycles");
        h.top.final();
        h.close_raw();
        if(failures) {
            write_terminal(raw_dir,false,1);
            std::cerr<<"DECODER_LAYER_TOKEN_ENGINE_VERILATOR_FAIL layer="
                     <<active_layer_index<<" failures="<<failures<<"\n";
            return 1;
        }
        write_terminal(raw_dir,true,0);
        std::cout<<"DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS layer="<<active_layer_index
                 <<" trace_count="<<trace_count
                 <<" final_count="<<final_count<<" cycles="<<cycles<<" stalls="<<stalls
                 <<" token0_cycles="<<token_done[0]-token_start[0]<<" token1_cycles="<<token_done[1]-token_start[1]
                 <<" phase_p_run="<<phase_cycles[5]<<" phase_final="<<phase_cycles[36]
                 <<" phase_cycles=";
        for (unsigned phase=0; phase<=36; ++phase)
            std::cout << (phase ? "," : "") << phase_cycles[phase];
        std::cout<<" reset=pass clear=pass slot_isolation=pass\n";
    } catch(const std::exception& e) {
        if(!raw_dir.empty()) write_terminal(raw_dir,false,2);
        std::cerr<<"DECODER_LAYER_TOKEN_ENGINE_VERILATOR_FAIL layer="
                 <<active_layer_index<<" "<<e.what()<<"\n";
        return 2;
    }
}
