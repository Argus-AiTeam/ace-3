#include "Vace3_awq_w4a16_projection_engine.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct InputPaths {
    std::string transactions;
    std::string expected;
    std::string meta;
    std::string pairs;
};

struct Transaction {
    uint16_t first_output;
    uint16_t output_count;
};

struct Expected {
    uint16_t channel;
    std::array<uint32_t, 4> accumulator;
    uint16_t result;
    bool invalid;
    bool saturation;
};

static uint64_t cycle_count = 0;

static InputPaths parse_paths(int argc, char** argv) {
    InputPaths paths;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if ((argument == "--transactions" || argument == "--expected" ||
             argument == "--meta" || argument == "--pairs") &&
            index + 1 >= argc) {
            std::cerr << argument << " requires a path\n";
            std::exit(2);
        }
        if (argument == "--transactions")
            paths.transactions = argv[++index];
        else if (argument == "--expected")
            paths.expected = argv[++index];
        else if (argument == "--meta")
            paths.meta = argv[++index];
        else if (argument == "--pairs")
            paths.pairs = argv[++index];
    }
    if (paths.transactions.empty() || paths.expected.empty() ||
        paths.meta.empty() || paths.pairs.empty()) {
        std::cerr
            << "usage: projection-sim --transactions PATH --expected PATH"
            << " --meta PATH --pairs PATH\n";
        std::exit(2);
    }
    return paths;
}

static std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(2);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
        lines.push_back(line);
    return lines;
}

static std::array<uint32_t, 5> parse_hex_words(const std::string& text) {
    std::array<uint32_t, 5> words{};
    unsigned bit_offset = 0;
    for (auto iterator = text.rbegin(); iterator != text.rend(); ++iterator) {
        unsigned nibble;
        if (*iterator >= '0' && *iterator <= '9')
            nibble = *iterator - '0';
        else if (*iterator >= 'a' && *iterator <= 'f')
            nibble = *iterator - 'a' + 10;
        else {
            std::cerr << "invalid hex record: " << text << "\n";
            std::exit(2);
        }
        words.at(bit_offset / 32) |= nibble << (bit_offset % 32);
        bit_offset += 4;
    }
    return words;
}

static uint32_t extract_bits(
    const std::array<uint32_t, 5>& words, unsigned offset, unsigned width
) {
    const unsigned word = offset / 32;
    const unsigned shift = offset % 32;
    uint64_t joined = words.at(word);
    if (word + 1 < words.size())
        joined |= static_cast<uint64_t>(words.at(word + 1)) << 32;
    const uint64_t mask =
        width == 32 ? 0xffffffffull : ((1ull << width) - 1);
    return (joined >> shift) & mask;
}

static std::vector<Transaction> read_transactions(const std::string& path) {
    std::vector<Transaction> transactions;
    for (const std::string& line : read_lines(path)) {
        if (line.size() != 7) {
            std::cerr << "invalid transaction record: " << line << "\n";
            std::exit(2);
        }
        const uint32_t packed = std::stoul(line, nullptr, 16);
        transactions.push_back(
            Transaction{
                static_cast<uint16_t>(packed & 0x1fffu),
                static_cast<uint16_t>((packed >> 13) & 0x1fffu),
            }
        );
    }
    return transactions;
}

static std::vector<Expected> read_expected(const std::string& path) {
    std::vector<Expected> expected;
    for (const std::string& line : read_lines(path)) {
        if (line.size() != 34) {
            std::cerr << "invalid expected record: " << line << "\n";
            std::exit(2);
        }
        const auto words = parse_hex_words(line);
        Expected item{};
        item.channel = extract_bits(words, 0, 13);
        item.accumulator[0] = extract_bits(words, 13, 32);
        item.accumulator[1] = extract_bits(words, 45, 32);
        item.accumulator[2] = extract_bits(words, 77, 32);
        item.accumulator[3] = extract_bits(words, 109, 6);
        item.result = extract_bits(words, 115, 16);
        item.invalid = extract_bits(words, 131, 1);
        item.saturation = extract_bits(words, 132, 1);
        expected.push_back(item);
    }
    return expected;
}

static std::vector<uint64_t> read_hex48(
    const std::string& path, size_t expected_count
) {
    std::vector<uint64_t> records;
    for (const std::string& line : read_lines(path)) {
        if (line.size() != 12) {
            std::cerr << "invalid 48-bit record: " << line << "\n";
            std::exit(2);
        }
        records.push_back(std::stoull(line, nullptr, 16));
    }
    if (records.size() != expected_count) {
        std::cerr << path << " count mismatch: " << records.size() << "\n";
        std::exit(2);
    }
    return records;
}

static void tick(Vace3_awq_w4a16_projection_engine* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
    ++cycle_count;
}

static void drive_idle(Vace3_awq_w4a16_projection_engine* top) {
    top->clear_i = 0;
    top->start_valid_i = 0;
    top->first_output_channel_i = 0;
    top->output_count_i = 1;
    top->meta_valid_i = 0;
    top->qzeros_i = 0;
    top->scale_f16_i = 0x3c00;
    top->pair_valid_i = 0;
    top->activation_f16_i = 0;
    top->qweight_i = 0;
    top->out_ready_i = 0;
    top->eval();
}

static bool cleared(const Vace3_awq_w4a16_projection_engine* top) {
    return top->start_ready_o && !top->busy_o && !top->meta_ready_o &&
           !top->pair_ready_o && !top->out_valid_o &&
           top->out_channel_o == 0 && top->out_f16_o == 0 &&
           top->acc_q53_48_o[0] == 0 && top->acc_q53_48_o[1] == 0 &&
           top->acc_q53_48_o[2] == 0 && top->acc_q53_48_o[3] == 0 &&
           !top->invalid_operand_o && !top->saturation_o;
}

static void start(
    Vace3_awq_w4a16_projection_engine* top,
    uint16_t first,
    uint16_t count,
    unsigned& failures
) {
    top->first_output_channel_i = first;
    top->output_count_i = count;
    top->start_valid_i = 1;
    top->eval();
    if (!top->start_ready_o)
        ++failures;
    tick(top);
    top->start_valid_i = 0;
}

static void send_meta(
    Vace3_awq_w4a16_projection_engine* top,
    uint64_t record,
    bool stall,
    uint64_t& stall_cycles,
    unsigned& failures
) {
    while (!top->meta_ready_o)
        tick(top);
    if (stall) {
        top->meta_valid_i = 0;
        tick(top);
        ++stall_cycles;
    }
    top->qzeros_i = record & 0xffffffffu;
    top->scale_f16_i = record >> 32;
    top->meta_valid_i = 1;
    top->eval();
    if (!top->meta_ready_o)
        ++failures;
    tick(top);
    top->meta_valid_i = 0;
}

static void send_pair(
    Vace3_awq_w4a16_projection_engine* top,
    uint64_t record,
    bool stall,
    uint64_t& stall_cycles,
    unsigned& failures
) {
    while (!top->pair_ready_o)
        tick(top);
    if (stall) {
        top->pair_valid_i = 0;
        tick(top);
        ++stall_cycles;
    }
    top->qweight_i = record & 0xffffffffu;
    top->activation_f16_i = record >> 32;
    top->pair_valid_i = 1;
    top->eval();
    if (!top->pair_ready_o)
        ++failures;
    tick(top);
    top->pair_valid_i = 0;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const InputPaths paths = parse_paths(argc, argv);
    const auto transactions = read_transactions(paths.transactions);
    const auto expected = read_expected(paths.expected);
    const auto meta = read_hex48(paths.meta, 98);
    const auto pairs = read_hex48(paths.pairs, 12544);
    if (transactions.size() != 7 || expected.size() != 14) {
        std::cerr << "projection vector geometry mismatch\n";
        return 2;
    }

    auto* top = new Vace3_awq_w4a16_projection_engine;
    unsigned failures = 0;
    size_t expected_index = 0;
    size_t meta_index = 0;
    size_t pair_index = 0;
    size_t starts = 0;
    size_t outputs = 0;
    uint64_t input_stall_cycles = 0;
    uint64_t output_backpressure_cycles = 0;
    uint64_t first_compute_cycles = 0;
    uint64_t next_compute_cycles = 0;

    drive_idle(top);
    top->rst_ni = 0;
    top->eval();
    if (top->start_ready_o || top->busy_o || top->out_valid_o)
        ++failures;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    top->eval();
    if (!cleared(top))
        ++failures;

    start(top, 0, 1, failures);
    send_meta(top, 0x3c0000000000ull, false, input_stall_cycles, failures);
    send_pair(top, 0, false, input_stall_cycles, failures);
    send_pair(top, 0, false, input_stall_cycles, failures);
    top->rst_ni = 0;
    top->eval();
    if (top->busy_o || top->meta_ready_o || top->pair_ready_o ||
        top->out_valid_o)
        ++failures;
    top->rst_ni = 1;
    top->eval();
    if (!cleared(top))
        ++failures;

    start(top, 0, 1, failures);
    send_meta(top, 0x3c0000000000ull, false, input_stall_cycles, failures);
    send_pair(top, 0, false, input_stall_cycles, failures);
    top->clear_i = 1;
    tick(top);
    top->clear_i = 0;
    top->eval();
    if (!cleared(top))
        ++failures;

    for (size_t transaction_index = 0;
         transaction_index < transactions.size();
         ++transaction_index) {
        const Transaction& transaction = transactions.at(transaction_index);
        start(
            top,
            transaction.first_output,
            transaction.output_count,
            failures
        );
        ++starts;
        const uint64_t start_cycle = cycle_count;
        uint64_t previous_accept_cycle = start_cycle;
        for (unsigned local_output = 0;
             local_output < transaction.output_count;
             ++local_output) {
            const Expected& item = expected.at(expected_index);
            for (unsigned group = 0; group < 7; ++group) {
                while (!top->meta_ready_o)
                    tick(top);
                if (top->meta_output_channel_o != item.channel ||
                    top->meta_group_index_o != group ||
                    top->meta_output_word_o != item.channel / 8 ||
                    top->meta_logical_lane_o != item.channel % 8)
                    ++failures;
                send_meta(
                    top,
                    meta.at(meta_index++),
                    transaction_index != 0 && group % 3 == 1,
                    input_stall_cycles,
                    failures
                );
                for (unsigned element = 0; element < 128; ++element) {
                    if (top->pair_input_index_o != group * 128 + element ||
                        top->pair_output_channel_o != item.channel ||
                        top->pair_group_index_o != group ||
                        top->pair_output_word_o != item.channel / 8 ||
                        top->pair_logical_lane_o != item.channel % 8)
                        ++failures;
                    send_pair(
                        top,
                        pairs.at(pair_index++),
                        transaction_index != 0 && element % 113 == 17,
                        input_stall_cycles,
                        failures
                    );
                }
            }
            while (!top->out_valid_o)
                tick(top);
            const uint64_t compute_cycles =
                cycle_count -
                (local_output == 0 ? start_cycle : previous_accept_cycle);
            if (transaction_index == 0 && local_output == 0)
                first_compute_cycles = compute_cycles;
            if (transaction_index == 0 && local_output != 0)
                next_compute_cycles = compute_cycles;
            if (transaction_index == 0 && compute_cycles != 910)
                ++failures;

            if (top->out_channel_o != item.channel ||
                top->out_f16_o != item.result ||
                top->acc_q53_48_o[0] != item.accumulator[0] ||
                top->acc_q53_48_o[1] != item.accumulator[1] ||
                top->acc_q53_48_o[2] != item.accumulator[2] ||
                top->acc_q53_48_o[3] != item.accumulator[3] ||
                top->invalid_operand_o != item.invalid ||
                top->saturation_o != item.saturation)
                ++failures;
            const uint16_t held_channel = top->out_channel_o;
            const uint16_t held_result = top->out_f16_o;
            const std::array<uint32_t, 4> held_accumulator = {
                top->acc_q53_48_o[0],
                top->acc_q53_48_o[1],
                top->acc_q53_48_o[2],
                top->acc_q53_48_o[3],
            };
            const bool held_invalid = top->invalid_operand_o;
            const bool held_saturation = top->saturation_o;
            for (unsigned stall = 0; stall < 4; ++stall) {
                tick(top);
                ++output_backpressure_cycles;
                if (!top->out_valid_o ||
                    top->out_channel_o != held_channel ||
                    top->out_f16_o != held_result ||
                    top->acc_q53_48_o[0] != held_accumulator[0] ||
                    top->acc_q53_48_o[1] != held_accumulator[1] ||
                    top->acc_q53_48_o[2] != held_accumulator[2] ||
                    top->acc_q53_48_o[3] != held_accumulator[3] ||
                    top->invalid_operand_o != held_invalid ||
                    top->saturation_o != held_saturation)
                    ++failures;
            }
            top->out_ready_i = 1;
            tick(top);
            top->out_ready_i = 0;
            previous_accept_cycle = cycle_count;
            ++outputs;
            ++expected_index;
        }
    }

    if (starts != 7 || outputs != 14 || expected_index != 14 ||
        meta_index != 98 || pair_index != 12544 ||
        input_stall_cycles != 54 || output_backpressure_cycles != 56)
        ++failures;

    if (failures == 0) {
        std::cout
            << "AWQ_W4A16_PROJECTION_VERILATOR_PASS transactions=" << starts
            << " outputs=" << outputs
            << " official_outputs=8 groups=" << meta_index
            << " pairs=" << pair_index
            << " ulp_bound=0 first_output_compute_cycles="
            << first_compute_cycles
            << " next_output_compute_cycles=" << next_compute_cycles
            << " input_stall_cycles=" << input_stall_cycles
            << " output_backpressure_cycles=" << output_backpressure_cycles
            << " reset=pass clear=pass protocol=pass"
            << " four_state=unsupported\n";
    } else {
        std::cerr << "AWQ_W4A16_PROJECTION_VERILATOR_FAIL failures="
                  << failures << "\n";
    }
    top->final();
    delete top;
    return failures == 0 ? 0 : 1;
}
