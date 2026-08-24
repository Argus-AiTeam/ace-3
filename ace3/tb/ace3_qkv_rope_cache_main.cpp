#include "Vace3_qkv_rope_cache_verilator_top.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct WideRecord {
    uint64_t low;
    uint64_t high;
};

static void cycle(Vace3_qkv_rope_cache_verilator_top &top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    top.clk_i = 0;
    top.eval();
}

static std::vector<WideRecord> load_wide(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open " + path);
    std::vector<WideRecord> records;
    std::string line;
    while (input >> line) {
        if (line.size() != 32)
            throw std::runtime_error("malformed 128-bit record");
        records.push_back(
            {std::stoull(line.substr(16), nullptr, 16),
             std::stoull(line.substr(0, 16), nullptr, 16)});
    }
    return records;
}

static std::vector<uint64_t> load_narrow(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open " + path);
    std::vector<uint64_t> records;
    std::string line;
    while (input >> line)
        records.push_back(std::stoull(line, nullptr, 16));
    return records;
}

static uint64_t bits(uint64_t value, unsigned shift, uint64_t mask) {
    return (value >> shift) & mask;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    std::string vector_dir = "build/qkv_rope_cache_vectors";
    for (int index = 1; index + 1 < argc; ++index)
        if (std::string(argv[index]) == "--vector-dir")
            vector_dir = argv[index + 1];

    const auto rope_records = load_wide(vector_dir + "/rope_cases.hex");
    const auto cache_records = load_narrow(vector_dir + "/cache_cases.hex");
    if (rope_records.size() != 512 || cache_records.size() != 128)
        throw std::runtime_error("vector count mismatch");

    Vace3_qkv_rope_cache_verilator_top top;
    top.clear_i = 0;
    top.rope_in_valid_i = 0;
    top.rope_out_ready_i = 0;
    top.cache_write_valid_i = 0;
    top.cache_read_valid_i = 0;
    top.cache_out_ready_i = 0;
    top.rst_ni = 0;
    cycle(top);
    cycle(top);
    top.rst_ni = 1;
    top.eval();
    if (!top.rope_in_ready_o || !top.cache_write_ready_o ||
        !top.cache_read_ready_o)
        throw std::runtime_error("reset release readiness failure");

    unsigned query_counts[14] = {};
    unsigned key_counts[2] = {};
    unsigned rope_stalls = 0;
    for (size_t index = 0; index < rope_records.size(); ++index) {
        const auto &record = rope_records[index];
        const bool is_key = bits(record.high, 34, 1);
        const unsigned head = bits(record.high, 35, 0xf);
        const unsigned pair = bits(record.high, 39, 0x1f);
        const unsigned position = bits(record.high, 44, 0x7fff);
        top.rope_is_key_i = is_key;
        top.rope_head_i = head;
        top.rope_pair_i = pair;
        top.rope_position_i = position;
        top.rope_low_i = bits(record.low, 0, 0xffff);
        top.rope_high_i = bits(record.low, 16, 0xffff);
        top.rope_cos_i = bits(record.low, 32, 0xffff);
        top.rope_sin_i = bits(record.low, 48, 0xffff);
        top.rope_in_valid_i = 1;
        top.eval();
        if (!top.rope_in_ready_o)
            throw std::runtime_error("RoPE input not ready");
        cycle(top);
        top.rope_in_valid_i = 0;
        top.eval();
        if (!top.rope_out_valid_o ||
            top.rope_out_low_o != bits(record.high, 0, 0xffff) ||
            top.rope_out_high_o != bits(record.high, 16, 0xffff) ||
            top.rope_invalid_o != bits(record.high, 32, 1) ||
            top.rope_saturation_o != bits(record.high, 33, 1) ||
            top.rope_out_is_key_o != is_key ||
            top.rope_out_head_o != head ||
            top.rope_out_pair_o != pair ||
            top.rope_out_position_o != position)
            throw std::runtime_error("RoPE result mismatch");
        if (index % 29 == 0) {
            const uint64_t held =
                static_cast<uint64_t>(top.rope_out_low_o) |
                (static_cast<uint64_t>(top.rope_out_high_o) << 16) |
                (static_cast<uint64_t>(top.rope_out_head_o) << 32);
            cycle(top);
            cycle(top);
            const uint64_t observed =
                static_cast<uint64_t>(top.rope_out_low_o) |
                (static_cast<uint64_t>(top.rope_out_high_o) << 16) |
                (static_cast<uint64_t>(top.rope_out_head_o) << 32);
            if (!top.rope_out_valid_o || observed != held)
                throw std::runtime_error("RoPE backpressure stability failure");
            rope_stalls += 2;
        }
        top.rope_out_ready_i = 1;
        cycle(top);
        top.rope_out_ready_i = 0;
        if (is_key)
            ++key_counts[head];
        else
            ++query_counts[head];
    }
    for (unsigned count : query_counts)
        if (count != 32)
            throw std::runtime_error("query-head coverage failure");
    for (unsigned count : key_counts)
        if (count != 32)
            throw std::runtime_error("key-head coverage failure");

    unsigned cache_writes = 0;
    auto write_cache = [&](unsigned slot, unsigned position, unsigned head,
                           unsigned dimension, unsigned k_value,
                           unsigned v_value) {
        top.cache_write_slot_i = slot;
        top.cache_write_position_i = position;
        top.cache_write_head_i = head;
        top.cache_write_dimension_i = dimension;
        top.cache_write_k_i = k_value;
        top.cache_write_v_i = v_value;
        top.cache_write_valid_i = 1;
        top.eval();
        if (!top.cache_write_ready_o)
            throw std::runtime_error("cache write not ready");
        cycle(top);
        top.cache_write_valid_i = 0;
        ++cache_writes;
    };
    unsigned cache_reads = 0;
    unsigned cache_stalls = 0;
    auto read_cache = [&](unsigned slot, unsigned position, unsigned head,
                          unsigned dimension, bool hit, unsigned k_value,
                          unsigned v_value, bool stall) {
        top.cache_read_slot_i = slot;
        top.cache_read_position_i = position;
        top.cache_read_head_i = head;
        top.cache_read_dimension_i = dimension;
        top.cache_read_valid_i = 1;
        top.eval();
        if (!top.cache_read_ready_o)
            throw std::runtime_error("cache read not ready");
        cycle(top);
        top.cache_read_valid_i = 0;
        top.eval();
        if (!top.cache_out_valid_o || top.cache_out_hit_o != hit ||
            top.cache_out_slot_o != slot ||
            top.cache_out_position_o != position ||
            top.cache_out_head_o != head ||
            top.cache_out_dimension_o != dimension ||
            top.cache_out_k_o != k_value || top.cache_out_v_o != v_value)
            throw std::runtime_error("cache read mismatch");
        if (stall) {
            const uint64_t held =
                static_cast<uint64_t>(top.cache_out_k_o) |
                (static_cast<uint64_t>(top.cache_out_v_o) << 16);
            cycle(top);
            cycle(top);
            const uint64_t observed =
                static_cast<uint64_t>(top.cache_out_k_o) |
                (static_cast<uint64_t>(top.cache_out_v_o) << 16);
            if (!top.cache_out_valid_o || observed != held)
                throw std::runtime_error("cache backpressure stability failure");
            cache_stalls += 2;
        }
        top.cache_out_ready_i = 1;
        cycle(top);
        top.cache_out_ready_i = 0;
        ++cache_reads;
    };

    for (uint64_t record : cache_records)
        write_cache(bits(record, 54, 3), bits(record, 39, 0x7fff),
                    bits(record, 38, 0x1), bits(record, 32, 0x3f),
                    bits(record, 0, 0xffff), bits(record, 16, 0xffff));
    for (size_t index = 0; index < cache_records.size(); ++index) {
        const uint64_t record = cache_records[index];
        read_cache(bits(record, 54, 3), bits(record, 39, 0x7fff),
                   bits(record, 38, 0x1), bits(record, 32, 0x3f), true,
                   bits(record, 0, 0xffff), bits(record, 16, 0xffff),
                   index % 31 == 0);
    }

    write_cache(0, 3, 0, 0, 0xaaaa, 0xbbbb);
    write_cache(0, 3, 0, 0, 0xcccc, 0xdddd);
    read_cache(0, 3, 0, 0, true, 0xcccc, 0xdddd, true);
    write_cache(1, 3, 0, 0, 0x1111, 0x2222);
    write_cache(0, 4, 0, 0, 0x3333, 0x4444);
    read_cache(1, 3, 0, 0, true, 0x1111, 0x2222, false);
    read_cache(0, 4, 0, 0, true, 0x3333, 0x4444, false);
    read_cache(0, 3, 0, 0, true, 0xcccc, 0xdddd, false);
    read_cache(1, 4, 1, 63, false, 0, 0, false);

    write_cache(0, 2, 1, 9, 0x7777, 0x8888);
    top.clear_i = 1;
    cycle(top);
    top.clear_i = 0;
    read_cache(0, 2, 1, 9, false, 0, 0, false);
    write_cache(0, 1, 1, 8, 0x9999, 0xaaaa);
    top.rst_ni = 0;
    cycle(top);
    top.rst_ni = 1;
    top.eval();
    read_cache(0, 1, 1, 8, false, 0, 0, false);

    std::cout
        << "ACE3_QKV_ROPE_CACHE_VERILATOR_PASS rope_outputs=512 rope_stalls="
        << rope_stalls
        << " query_heads=14 key_heads=2 value_heads=2 cache_writes="
        << cache_writes << " cache_reads=" << cache_reads
        << " cache_stalls=" << cache_stalls
        << " overwrite=pass isolation=pass reset=1 clear=1\n";
    top.final();
    return EXIT_SUCCESS;
}
