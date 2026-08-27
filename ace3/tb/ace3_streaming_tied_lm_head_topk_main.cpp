#include "Vace3_streaming_tied_lm_head_topk.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

struct ExpectedLogit {
    uint16_t bits;
    std::array<uint32_t, 3> accumulator;
};

struct ExpectedTop {
    uint32_t rank;
    uint32_t token;
    uint16_t bits;
};

static uint64_t cycles = 0;
static constexpr uint64_t kOfficialHiddenSize = 896;
static constexpr uint64_t kOfficialVocabSize = 151936;
static constexpr uint64_t kOfficialTopK = 10;
static constexpr uint64_t kOfficialCheckCount = 12;
static constexpr uint64_t kOfficialWeightValues =
    kOfficialHiddenSize * kOfficialVocabSize;

static void fail(const std::string& message) {
    std::cerr << message << "\n";
    std::exit(1);
}

static std::map<std::string, uint64_t> read_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open config: " + path);
    std::map<std::string, uint64_t> result;
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find('=');
        if (separator == std::string::npos) fail("invalid config line: " + line);
        result[line.substr(0, separator)] = std::stoull(line.substr(separator + 1));
    }
    return result;
}

static std::vector<uint16_t> read_hidden(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open hidden vectors: " + path);
    std::vector<uint16_t> result;
    std::string line;
    while (std::getline(input, line)) result.push_back(std::stoul(line, nullptr, 16));
    return result;
}

static std::array<uint32_t, 3> parse_accumulator(const std::string& text) {
    if (text.size() != 24) fail("invalid 96-bit accumulator: " + text);
    std::array<uint32_t, 3> words{};
    words[2] = std::stoul(text.substr(0, 8), nullptr, 16);
    words[1] = std::stoul(text.substr(8, 8), nullptr, 16);
    words[0] = std::stoul(text.substr(16, 8), nullptr, 16);
    return words;
}

static std::map<uint32_t, ExpectedLogit> read_checks(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open checks: " + path);
    std::map<uint32_t, ExpectedLogit> result;
    uint32_t token;
    std::string bits;
    std::string accumulator;
    while (input >> token >> bits >> accumulator)
        result[token] = ExpectedLogit{static_cast<uint16_t>(std::stoul(bits, nullptr, 16)), parse_accumulator(accumulator)};
    return result;
}

static std::vector<ExpectedTop> read_top(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open top-k: " + path);
    std::vector<ExpectedTop> result;
    uint32_t rank, token;
    std::string bits;
    int64_t ignored_value;
    while (input >> rank >> token >> bits >> ignored_value)
        result.push_back(ExpectedTop{rank, token, static_cast<uint16_t>(std::stoul(bits, nullptr, 16))});
    return result;
}

static void tick(Vace3_streaming_tied_lm_head_topk* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
    ++cycles;
}

static void idle(Vace3_streaming_tied_lm_head_topk* top) {
    top->clear_i = 0;
    top->start_valid_i = 0;
    top->hidden_valid_i = 0;
    top->hidden_index_i = 0;
    top->hidden_f16_i = 0;
    top->hidden_last_i = 0;
    top->hidden_end_i = 0;
    top->weight_valid_i = 0;
    top->weight_token_index_i = 0;
    top->weight_feature_index_i = 0;
    top->weight_f16_i = 0;
    top->weight_last_feature_i = 0;
    top->weight_last_token_i = 0;
    top->weight_end_i = 0;
    top->logit_ready_i = 0;
    top->top_ready_i = 0;
    top->done_ready_i = 0;
    top->eval();
}

static bool equal_accumulator(
    const Vace3_streaming_tied_lm_head_topk* top,
    const std::array<uint32_t, 3>& expected
) {
    return top->acc_q47_48_o[0] == expected[0] &&
           top->acc_q47_48_o[1] == expected[1] &&
           top->acc_q47_48_o[2] == expected[2];
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::map<std::string, std::string> paths;
    for (int index = 1; index + 1 < argc; index += 2)
        paths[argv[index]] = argv[index + 1];
    for (const char* name : {"--checkpoint", "--config", "--hidden", "--checks", "--topk"})
        if (!paths.count(name)) fail(std::string("missing argument ") + name);

    const auto config = read_config(paths["--config"]);
    const auto hidden = read_hidden(paths["--hidden"]);
    const auto checks = read_checks(paths["--checks"]);
    const auto expected_top = read_top(paths["--topk"]);
    if (config.at("hidden_size") != kOfficialHiddenSize ||
        config.at("vocab_size") != kOfficialVocabSize ||
        config.at("top_k") != kOfficialTopK ||
        config.at("check_count") != kOfficialCheckCount)
        fail("official geometry mismatch");
    if (config.at("weight_bytes") != kOfficialWeightValues * sizeof(uint16_t))
        fail("official weight byte count mismatch");
    if (hidden.size() != config.at("hidden_size")) fail("hidden vector count mismatch");
    if (checks.size() != config.at("check_count")) fail("selected check count mismatch");
    if (expected_top.size() != config.at("top_k")) fail("top-k count mismatch");
    for (size_t rank = 0; rank < expected_top.size(); ++rank) {
        if (expected_top[rank].rank != rank || expected_top[rank].token >= kOfficialVocabSize)
            fail("invalid official top-k framing");
    }
    for (const auto& check : checks) {
        if (check.first >= kOfficialVocabSize) fail("selected check token outside vocabulary");
    }

    const int descriptor = open(paths["--checkpoint"].c_str(), O_RDONLY);
    if (descriptor < 0) fail("cannot open checkpoint");
    struct stat status{};
    if (fstat(descriptor, &status) != 0 || static_cast<uint64_t>(status.st_size) != config.at("checkpoint_bytes"))
        fail("checkpoint byte count mismatch");
    const void* mapping = mmap(nullptr, status.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (mapping == MAP_FAILED) fail("cannot mmap checkpoint");
    const auto* bytes = static_cast<const uint8_t*>(mapping);
    const uint64_t weight_offset = config.at("weight_offset");
    const uint64_t weight_bytes = config.at("weight_bytes");
    if (weight_offset + weight_bytes > static_cast<uint64_t>(status.st_size)) fail("weight range outside checkpoint");
    const auto* weights = bytes + weight_offset;

    auto* top = new Vace3_streaming_tied_lm_head_topk;
    idle(top);
    top->rst_ni = 0;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    top->eval();
    if (!top->start_ready_o) fail("DUT not ready after reset");

    top->start_valid_i = 1;
    tick(top);
    top->start_valid_i = 0;
    for (uint32_t index = 0; index < hidden.size(); ++index) {
        if (!top->hidden_ready_o) fail("hidden channel unexpectedly stalled");
        top->hidden_index_i = index;
        top->hidden_f16_i = hidden[index];
        top->hidden_last_i = index + 1 == hidden.size();
        top->hidden_end_i = top->hidden_last_i;
        top->hidden_valid_i = 1;
        tick(top);
    }
    top->hidden_valid_i = 0;
    top->hidden_last_i = 0;
    top->hidden_end_i = 0;

    uint64_t accepted_weights = 0;
    uint32_t accepted_logits = 0;
    uint32_t matched_checks = 0;
    const uint32_t hidden_size = config.at("hidden_size");
    const uint32_t vocab_size = config.at("vocab_size");
    for (uint32_t token = 0; token < vocab_size; ++token) {
        for (uint32_t feature = 0; feature < hidden_size; ++feature) {
            if (!top->weight_ready_o) fail("weight channel unexpectedly stalled");
            const uint64_t offset = (static_cast<uint64_t>(token) * hidden_size + feature) * 2;
            const uint16_t bits = static_cast<uint16_t>(weights[offset]) |
                                  (static_cast<uint16_t>(weights[offset + 1]) << 8);
            top->weight_token_index_i = token;
            top->weight_feature_index_i = feature;
            top->weight_f16_i = bits;
            top->weight_last_feature_i = feature + 1 == hidden_size;
            top->weight_last_token_i = (token + 1 == vocab_size) && top->weight_last_feature_i;
            top->weight_end_i = top->weight_last_token_i;
            top->weight_valid_i = 1;
            tick(top);
            ++accepted_weights;
        }
        top->weight_valid_i = 0;
        top->weight_last_feature_i = 0;
        top->weight_last_token_i = 0;
        top->weight_end_i = 0;
        if (!top->logit_valid_o || top->logit_token_index_o != token)
            fail("missing or misindexed logit at token " + std::to_string(token));
        const auto check = checks.find(token);
        if (check != checks.end()) {
            if (top->logit_f16_o != check->second.bits || !equal_accumulator(top, check->second.accumulator))
                fail("selected logit mismatch at token " + std::to_string(token));
            ++matched_checks;
        }
        if (top->logit_saturation_o || top->invalid_operand_o || top->error_valid_o)
            fail("invalid or saturated official logit at token " + std::to_string(token));
        if ((token % 4093) == 0) {
            const uint16_t held_bits = top->logit_f16_o;
            const std::array<uint32_t, 3> held_acc = {
                top->acc_q47_48_o[0], top->acc_q47_48_o[1], top->acc_q47_48_o[2]
            };
            tick(top);
            if (!top->logit_valid_o || top->logit_token_index_o != token ||
                top->logit_f16_o != held_bits || !equal_accumulator(top, held_acc))
                fail("logit changed under backpressure");
        }
        top->logit_ready_i = 1;
        tick(top);
        top->logit_ready_i = 0;
        ++accepted_logits;
    }

    for (const ExpectedTop& expected : expected_top) {
        if (!top->top_valid_o) fail("missing top-k output");
        if (top->top_rank_o != expected.rank || top->top_token_index_o != expected.token ||
            top->top_logit_f16_o != expected.bits)
            fail("top-k mismatch at rank " + std::to_string(expected.rank));
        const uint32_t held_rank = top->top_rank_o;
        const uint32_t held_token = top->top_token_index_o;
        const uint16_t held_bits = top->top_logit_f16_o;
        tick(top);
        if (!top->top_valid_o || top->top_rank_o != held_rank ||
            top->top_token_index_o != held_token || top->top_logit_f16_o != held_bits)
            fail("top-k changed under backpressure");
        top->top_ready_i = 1;
        tick(top);
        top->top_ready_i = 0;
    }
    if (!top->done_valid_o) fail("missing completion output");
    top->done_ready_i = 1;
    tick(top);
    top->done_ready_i = 0;
    if (!top->start_ready_o || top->busy_o || top->error_valid_o) fail("DUT did not return idle");

    std::cout << "STREAMING_LM_HEAD_OFFICIAL_PASS hidden=" << hidden.size()
              << " vocab=" << accepted_logits
              << " weights=" << accepted_weights
              << " top_token=" << expected_top.front().token
              << " checks=" << matched_checks
              << " cycles=" << cycles
              << " integrated_dialogue=not_run synthesis=not_run ppa=not_measured fpga=not_run latency=not_claimed\n";

    munmap(const_cast<void*>(mapping), status.st_size);
    close(descriptor);
    delete top;
    return 0;
}
