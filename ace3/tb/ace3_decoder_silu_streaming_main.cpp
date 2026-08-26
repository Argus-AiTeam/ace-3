#include "Vace3_decoder_silu_streaming_tb.h"
#include "verilated.h"

double main_time = 0;

double sc_time_stamp() {
    return main_time;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vace3_decoder_silu_streaming_tb top;

    for (unsigned cycle = 0;
         cycle < 20000 && !Verilated::gotFinish();
         ++cycle) {
        top.clk_i = 0;
        top.eval();
        ++main_time;
        top.clk_i = 1;
        top.eval();
        ++main_time;
    }

    top.final();
    return Verilated::gotFinish() ? 0 : 1;
}
