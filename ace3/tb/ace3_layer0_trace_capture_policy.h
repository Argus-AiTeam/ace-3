#pragma once

inline bool layer0_trace_capture_accept(bool checking, bool trace_valid,
                                        bool trace_ready, bool final_valid,
                                        bool final_ready) {
    return checking && trace_valid && trace_ready &&
           (!final_valid || final_ready);
}

inline bool layer0_final_capture_accept(bool checking, bool final_valid,
                                        bool final_ready, bool trace_ready) {
    return checking && final_valid && final_ready && trace_ready;
}
