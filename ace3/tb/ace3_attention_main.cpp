#include "Vace3_attention_verilator_top.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

uint64_t bits(const std::string& hex, unsigned low, unsigned width) {
    if (width > 64) {
        throw std::runtime_error("slice width exceeds 64 bits");
    }
    uint64_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned source_bit = low + bit;
        const unsigned nibble_from_right = source_bit / 4;
        if (nibble_from_right >= hex.size()) {
            continue;
        }
        const char character = hex[hex.size() - 1 - nibble_from_right];
        unsigned nibble = 0;
        if (character >= '0' && character <= '9') {
            nibble = static_cast<unsigned>(character - '0');
        } else if (character >= 'a' && character <= 'f') {
            nibble = static_cast<unsigned>(character - 'a' + 10);
        } else {
            throw std::runtime_error("malformed hexadecimal record");
        }
        value |= static_cast<uint64_t>((nibble >> (source_bit % 4)) & 1U)
                 << bit;
    }
    return value;
}

class AttentionTest {
  public:
    explicit AttentionTest(const std::string& vector_dir)
        : score_terms_(read_lines(vector_dir + "/attention_score_terms.hex")),
          score_expected_(
              read_lines(vector_dir + "/attention_score_expected.hex")),
          softmax_rows_(
              read_lines(vector_dir + "/attention_softmax_rows.hex")),
          softmax_terms_(
              read_lines(vector_dir + "/attention_softmax_terms.hex")),
          value_cases_(
              read_lines(vector_dir + "/attention_value_cases.hex")),
          value_terms_(
              read_lines(vector_dir + "/attention_value_terms.hex")) {
        initialize();
    }

    int run() {
        reset();
        for (size_t index = 0; index < score_expected_.size(); ++index) {
            run_score(index);
        }
        size_t softmax_base = 0;
        for (size_t index = 0; index < softmax_rows_.size(); ++index) {
            run_softmax(index, softmax_base);
            softmax_base += bits(softmax_rows_[index], 19, 16);
        }
        size_t value_base = 0;
        for (size_t index = 0; index < value_cases_.size(); ++index) {
            run_value(index, value_base);
            value_base += bits(value_cases_[index], 29, 16);
        }
        check(softmax_base == softmax_terms_.size(),
              "softmax framing mismatch");
        check(value_base == value_terms_.size(), "value framing mismatch");
        check_invalid_configuration();
        clear_abort();
        reset_abort();
        if (failures_ != 0) {
            std::cerr << "ACE3_ATTENTION_VERILATOR_FAIL failures="
                      << failures_ << '\n';
            return 1;
        }
        std::cout
            << "ACE3_ATTENTION_VERILATOR_PASS score_cases=" << score_outputs_
            << " softmax_outputs=" << softmax_outputs_
            << " value_cases=" << value_outputs_ << " stalls=" << stalls_
            << " reset=" << reset_checks_ << " clear=" << clear_checks_
            << " cache_miss=pass causal=pass gqa=14_to_2\n";
        return 0;
    }

  private:
    Vace3_attention_verilator_top top_;
    std::vector<std::string> score_terms_;
    std::vector<std::string> score_expected_;
    std::vector<std::string> softmax_rows_;
    std::vector<std::string> softmax_terms_;
    std::vector<std::string> value_cases_;
    std::vector<std::string> value_terms_;
    unsigned failures_ = 0;
    unsigned score_outputs_ = 0;
    unsigned softmax_outputs_ = 0;
    unsigned value_outputs_ = 0;
    unsigned stalls_ = 0;
    unsigned reset_checks_ = 0;
    unsigned clear_checks_ = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures_;
        }
    }

    void evaluate() {
        top_.eval();
    }

    void tick() {
        top_.clk_i = 0;
        evaluate();
        top_.clk_i = 1;
        evaluate();
        top_.clk_i = 0;
        evaluate();
    }

    void initialize() {
        top_.clk_i = 0;
        top_.rst_ni = 1;
        top_.clear_i = 0;
        top_.score_start_valid_i = 0;
        top_.score_query_head_i = 0;
        top_.score_key_head_i = 0;
        top_.score_query_position_i = 0;
        top_.score_key_position_i = 0;
        top_.score_pair_valid_i = 0;
        top_.score_q_f16_i = 0;
        top_.score_k_f16_i = 0;
        top_.score_cache_hit_i = 1;
        top_.score_out_ready_i = 0;
        top_.softmax_start_valid_i = 0;
        top_.softmax_query_head_i = 0;
        top_.softmax_query_position_i = 0;
        top_.softmax_context_count_i = 1;
        top_.softmax_score_valid_i = 0;
        top_.softmax_score_f16_i = 0;
        top_.softmax_key_position_i = 0;
        top_.softmax_causal_i = 1;
        top_.softmax_cache_miss_i = 0;
        top_.softmax_invalid_i = 0;
        top_.softmax_out_ready_i = 0;
        top_.value_start_valid_i = 0;
        top_.value_query_head_i = 0;
        top_.value_head_i = 0;
        top_.value_query_position_i = 0;
        top_.value_dimension_i = 0;
        top_.value_context_count_i = 1;
        top_.value_term_valid_i = 0;
        top_.value_probability_f16_i = 0;
        top_.value_cached_f16_i = 0;
        top_.value_hit_i = 1;
        top_.value_row_error_i = 0;
        top_.value_out_ready_i = 0;
        evaluate();
    }

    void reset() {
        top_.rst_ni = 0;
        evaluate();
        check(!top_.score_start_ready_o && !top_.score_busy_o &&
                  !top_.score_out_valid_o && !top_.softmax_start_ready_o &&
                  !top_.softmax_busy_o && !top_.softmax_out_valid_o &&
                  !top_.value_start_ready_o && !top_.value_busy_o &&
                  !top_.value_out_valid_o,
              "Verilator asynchronous reset assertion failed");
        tick();
        tick();
        top_.rst_ni = 1;
        evaluate();
        check(top_.score_start_ready_o && top_.softmax_start_ready_o &&
                  top_.value_start_ready_o,
              "Verilator reset release failed");
        ++reset_checks_;
    }

    void run_score(size_t case_index) {
        const std::string& expected = score_expected_.at(case_index);
        top_.score_query_head_i = bits(expected, 16, 4);
        top_.score_key_head_i = bits(expected, 20, 4);
        top_.score_query_position_i = bits(expected, 24, 15);
        top_.score_key_position_i = bits(expected, 39, 15);
        top_.score_start_valid_i = 1;
        evaluate();
        check(top_.score_start_ready_o,
              "Verilator score start not ready case " +
                  std::to_string(case_index));
        tick();
        top_.score_start_valid_i = 0;
        for (unsigned dimension = 0; dimension < 64; ++dimension) {
            const std::string& term =
                score_terms_.at(case_index * 64 + dimension);
            top_.score_q_f16_i = bits(term, 0, 16);
            top_.score_k_f16_i = bits(term, 16, 16);
            top_.score_cache_hit_i = bits(term, 32, 1);
            top_.score_pair_valid_i = 1;
            evaluate();
            check(top_.score_pair_ready_o,
                  "Verilator score pair not ready");
            tick();
        }
        top_.score_pair_valid_i = 0;
        evaluate();
        check_score_output(expected, case_index);
        if (case_index == 0) {
            for (unsigned stall = 0; stall < 2; ++stall) {
                tick();
                check_score_output(expected, case_index);
                ++stalls_;
            }
        }
        top_.score_out_ready_i = 1;
        tick();
        top_.score_out_ready_i = 0;
        evaluate();
        check(!top_.score_out_valid_o,
              "Verilator score output did not release");
        ++score_outputs_;
    }

    void check_score_output(const std::string& expected, size_t case_index) {
        check(top_.score_out_valid_o &&
                  top_.score_f16_o == bits(expected, 0, 16) &&
                  top_.score_query_head_o == bits(expected, 16, 4) &&
                  top_.score_key_head_o == bits(expected, 20, 4) &&
                  top_.score_query_position_o == bits(expected, 24, 15) &&
                  top_.score_key_position_o == bits(expected, 39, 15) &&
                  top_.score_causal_o == bits(expected, 54, 1) &&
                  top_.score_cache_miss_o == bits(expected, 55, 1) &&
                  top_.score_invalid_o == bits(expected, 56, 1) &&
                  top_.score_saturation_o == bits(expected, 57, 1),
              "Verilator score mismatch case " +
                  std::to_string(case_index));
    }

    void run_softmax(size_t row_index, size_t term_base) {
        const std::string& header = softmax_rows_.at(row_index);
        const unsigned count = bits(header, 19, 16);
        top_.softmax_query_head_i = bits(header, 0, 4);
        top_.softmax_query_position_i = bits(header, 4, 15);
        top_.softmax_context_count_i = count;
        top_.softmax_start_valid_i = 1;
        evaluate();
        check(top_.softmax_start_ready_o,
              "Verilator softmax start not ready");
        tick();
        top_.softmax_start_valid_i = 0;
        for (unsigned item = 0; item < count; ++item) {
            const std::string& term = softmax_terms_.at(term_base + item);
            top_.softmax_score_f16_i = bits(term, 0, 16);
            top_.softmax_key_position_i = bits(term, 16, 15);
            top_.softmax_causal_i = bits(term, 31, 1);
            top_.softmax_cache_miss_i = bits(term, 32, 1);
            top_.softmax_invalid_i = bits(term, 33, 1);
            top_.softmax_score_valid_i = 1;
            evaluate();
            check(top_.softmax_score_ready_o,
                  "Verilator softmax score not ready");
            tick();
        }
        top_.softmax_score_valid_i = 0;
        unsigned timeout = 0;
        while (!top_.softmax_out_valid_o && timeout < 64) {
            tick();
            ++timeout;
        }
        check(top_.softmax_out_valid_o, "Verilator softmax timeout");
        for (unsigned item = 0; item < count; ++item) {
            const std::string& term = softmax_terms_.at(term_base + item);
            check_softmax_output(header, term, item, count, row_index);
            if (row_index == 0 && item == 0) {
                for (unsigned stall = 0; stall < 2; ++stall) {
                    tick();
                    check_softmax_output(
                        header, term, item, count, row_index
                    );
                    ++stalls_;
                }
            }
            top_.softmax_out_ready_i = 1;
            tick();
            top_.softmax_out_ready_i = 0;
            evaluate();
            ++softmax_outputs_;
        }
        check(!top_.softmax_out_valid_o,
              "Verilator softmax output did not release");
    }

    void check_softmax_output(
        const std::string& header,
        const std::string& term,
        unsigned item,
        unsigned count,
        size_t row_index
    ) {
        check(top_.softmax_out_valid_o &&
                  top_.softmax_probability_f16_o == bits(term, 34, 16) &&
                  top_.softmax_query_head_o == bits(header, 0, 4) &&
                  top_.softmax_query_position_o == bits(header, 4, 15) &&
                  top_.softmax_key_position_o == bits(term, 16, 15) &&
                  top_.softmax_index_o == item &&
                  top_.softmax_last_o == (item == count - 1) &&
                  top_.softmax_row_error_o == bits(header, 35, 1) &&
                  top_.softmax_out_cache_miss_o == bits(header, 36, 1) &&
                  top_.softmax_out_invalid_o == bits(header, 37, 1),
              "Verilator softmax mismatch row " +
                  std::to_string(row_index) + " item " +
                  std::to_string(item));
    }

    void run_value(size_t case_index, size_t term_base) {
        const std::string& header = value_cases_.at(case_index);
        const unsigned count = bits(header, 29, 16);
        top_.value_query_head_i = bits(header, 0, 4);
        top_.value_head_i = bits(header, 4, 4);
        top_.value_query_position_i = bits(header, 8, 15);
        top_.value_dimension_i = bits(header, 23, 6);
        top_.value_context_count_i = count;
        top_.value_start_valid_i = 1;
        evaluate();
        check(top_.value_start_ready_o,
              "Verilator value start not ready");
        tick();
        top_.value_start_valid_i = 0;
        for (unsigned item = 0; item < count; ++item) {
            const std::string& term = value_terms_.at(term_base + item);
            top_.value_probability_f16_i = bits(term, 0, 16);
            top_.value_cached_f16_i = bits(term, 16, 16);
            top_.value_hit_i = bits(term, 32, 1);
            top_.value_row_error_i = bits(term, 33, 1);
            top_.value_term_valid_i = 1;
            evaluate();
            check(top_.value_term_ready_o,
                  "Verilator value term not ready");
            tick();
        }
        top_.value_term_valid_i = 0;
        evaluate();
        check_value_output(header, case_index);
        if (case_index == 0) {
            for (unsigned stall = 0; stall < 2; ++stall) {
                tick();
                check_value_output(header, case_index);
                ++stalls_;
            }
        }
        top_.value_out_ready_i = 1;
        tick();
        top_.value_out_ready_i = 0;
        evaluate();
        check(!top_.value_out_valid_o,
              "Verilator value output did not release");
        ++value_outputs_;
    }

    void check_value_output(
        const std::string& header, size_t case_index
    ) {
        check(top_.value_out_valid_o &&
                  top_.value_f16_o == bits(header, 45, 16) &&
                  top_.value_query_head_o == bits(header, 0, 4) &&
                  top_.value_head_o == bits(header, 4, 4) &&
                  top_.value_query_position_o == bits(header, 8, 15) &&
                  top_.value_dimension_o == bits(header, 23, 6) &&
                  top_.value_row_error_o == bits(header, 61, 1) &&
                  top_.value_cache_miss_o == bits(header, 62, 1) &&
                  top_.value_invalid_o == bits(header, 63, 1) &&
                  top_.value_saturation_o == bits(header, 64, 1),
              "Verilator value mismatch case " +
                  std::to_string(case_index));
    }

    void check_invalid_configuration() {
        top_.score_query_head_i = 6;
        top_.score_key_head_i = 1;
        top_.score_start_valid_i = 1;
        evaluate();
        check(!top_.score_start_ready_o,
              "Verilator accepted invalid score GQA mapping");
        top_.score_start_valid_i = 0;
        top_.value_query_head_i = 7;
        top_.value_head_i = 0;
        top_.value_start_valid_i = 1;
        evaluate();
        check(!top_.value_start_ready_o,
              "Verilator accepted invalid value GQA mapping");
        top_.value_start_valid_i = 0;
        top_.score_query_head_i = 0;
        top_.score_key_head_i = 0;
        top_.value_query_head_i = 0;
        top_.value_head_i = 0;
        evaluate();
    }

    void start_all() {
        top_.score_start_valid_i = 1;
        top_.softmax_start_valid_i = 1;
        top_.value_start_valid_i = 1;
        evaluate();
        check(top_.score_start_ready_o && top_.softmax_start_ready_o &&
                  top_.value_start_ready_o,
              "Verilator abort setup failed");
        tick();
        top_.score_start_valid_i = 0;
        top_.softmax_start_valid_i = 0;
        top_.value_start_valid_i = 0;
    }

    void clear_abort() {
        start_all();
        top_.clear_i = 1;
        tick();
        top_.clear_i = 0;
        evaluate();
        check(!top_.score_busy_o && !top_.score_out_valid_o &&
                  !top_.softmax_busy_o && !top_.softmax_out_valid_o &&
                  !top_.value_busy_o && !top_.value_out_valid_o,
              "Verilator clear abort failed");
        ++clear_checks_;
    }

    void reset_abort() {
        start_all();
        top_.rst_ni = 0;
        evaluate();
        check(!top_.score_busy_o && !top_.score_out_valid_o &&
                  !top_.softmax_busy_o && !top_.softmax_out_valid_o &&
                  !top_.value_busy_o && !top_.value_out_valid_o,
              "Verilator reset abort failed");
        top_.rst_ni = 1;
        evaluate();
        ++reset_checks_;
    }
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string vector_dir = "build/attention_vectors";
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        const std::string prefix = "--vector-dir=";
        if (argument.rfind(prefix, 0) == 0) {
            vector_dir = argument.substr(prefix.size());
        }
    }
    try {
        AttentionTest test(vector_dir);
        return test.run();
    } catch (const std::exception& error) {
        std::cerr << "ACE3_ATTENTION_VERILATOR_ERROR " << error.what()
                  << '\n';
        return 1;
    }
}
