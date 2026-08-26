#include "ace3_layer0_trace_capture_policy.h"

#include <cstdlib>
#include <iostream>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "LAYER0_TRACE_CAPTURE_POLICY_FAIL " << message << "\n";
        std::exit(1);
    }
}

int main() {
    require(!layer0_trace_capture_accept(false, true, true, false, false),
            "precheck trace was captured");
    require(layer0_trace_capture_accept(true, true, true, false, false),
            "authenticated non-final trace was rejected");
    require(!layer0_trace_capture_accept(true, true, true, true, false),
            "stalled final trace was duplicated");
    require(layer0_trace_capture_accept(true, true, true, true, true),
            "accepted final trace was rejected");
    require(layer0_final_capture_accept(true, true, true, true),
            "accepted final row was rejected");

    std::vector<unsigned> trace;
    std::vector<unsigned> final;
    const bool checking[] = {false, true, true, true};
    const bool final_valid[] = {false, false, true, true};
    const bool final_ready[] = {false, false, false, true};
    const unsigned value[] = {0xaa49, 0x207a, 0x2d79, 0x2d79};
    for (unsigned index = 0; index < 4; ++index) {
        if (layer0_trace_capture_accept(checking[index], true, true,
                                        final_valid[index],
                                        final_ready[index]))
            trace.push_back(value[index]);
        if (layer0_final_capture_accept(checking[index], final_valid[index],
                                        final_ready[index], true))
            final.push_back(value[index]);
    }
    require(trace.size() == 2 && trace[0] == 0x207a &&
                trace[1] == 0x2d79,
            "captured trace sequence is not canonical");
    require(final.size() == 1 && final[0] == 0x2d79,
            "final output changed or was duplicated");

    std::cout
        << "LAYER0_TRACE_CAPTURE_POLICY_PASS precheck_aa49=suppressed "
           "first_authenticated=207a stalled_final=suppressed "
           "accepted_final=2d79 trace_rows=2 final_rows=1\n";
    return 0;
}
