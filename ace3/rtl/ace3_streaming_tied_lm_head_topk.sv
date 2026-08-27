`timescale 1ns/1ps
`default_nettype none

module ace3_streaming_tied_lm_head_topk #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer VOCAB_SIZE = 151936,
    parameter integer TOP_K = 10,
    parameter integer TOKEN_INDEX_WIDTH = 18,
    parameter integer FEATURE_INDEX_WIDTH = 10,
    parameter integer TOP_RANK_WIDTH = 4
) (
    input  wire                             clk_i,
    input  wire                             rst_ni,
    input  wire                             clear_i,

    input  wire                             start_valid_i,
    output wire                             start_ready_o,

    input  wire                             hidden_valid_i,
    output wire                             hidden_ready_o,
    input  wire [FEATURE_INDEX_WIDTH-1:0]   hidden_index_i,
    input  wire [15:0]                      hidden_f16_i,
    input  wire                             hidden_last_i,
    input  wire                             hidden_end_i,

    input  wire                             weight_valid_i,
    output wire                             weight_ready_o,
    input  wire [TOKEN_INDEX_WIDTH-1:0]     weight_token_index_i,
    input  wire [FEATURE_INDEX_WIDTH-1:0]   weight_feature_index_i,
    input  wire [15:0]                      weight_f16_i,
    input  wire                             weight_last_feature_i,
    input  wire                             weight_last_token_i,
    input  wire                             weight_end_i,

    output wire                             logit_valid_o,
    input  wire                             logit_ready_i,
    output wire [TOKEN_INDEX_WIDTH-1:0]     logit_token_index_o,
    output wire [15:0]                      logit_f16_o,
    output wire signed [95:0]               acc_q47_48_o,
    output wire                             logit_saturation_o,

    output wire                             top_valid_o,
    input  wire                             top_ready_i,
    output wire [TOP_RANK_WIDTH-1:0]        top_rank_o,
    output wire [TOKEN_INDEX_WIDTH-1:0]     top_token_index_o,
    output wire [15:0]                      top_logit_f16_o,

    output wire                             done_valid_o,
    input  wire                             done_ready_i,
    output wire                             error_valid_o,
    output wire [3:0]                       error_code_o,
    output wire                             invalid_operand_o,
    output wire                             saturation_o,
    output wire                             busy_o
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_HIDDEN = 3'd1;
    localparam [2:0] ST_WEIGHT = 3'd2;
    localparam [2:0] ST_LOGIT  = 3'd3;
    localparam [2:0] ST_TOP    = 3'd4;
    localparam [2:0] ST_DONE   = 3'd5;
    localparam [2:0] ST_ERROR  = 3'd6;

    localparam [3:0] ERROR_UNKNOWN   = 4'd1;
    localparam [3:0] ERROR_NONFINITE = 4'd2;
    localparam [3:0] ERROR_ORDER     = 4'd3;
    localparam [3:0] ERROR_FRAMING   = 4'd4;

    localparam [FEATURE_INDEX_WIDTH-1:0] LAST_FEATURE =
        FEATURE_INDEX_WIDTH'(HIDDEN_SIZE - 1);
    localparam [TOKEN_INDEX_WIDTH-1:0] LAST_TOKEN =
        TOKEN_INDEX_WIDTH'(VOCAB_SIZE - 1);
    localparam [TOP_RANK_WIDTH-1:0] LAST_RANK =
        TOP_RANK_WIDTH'(TOP_K - 1);

    reg [2:0] state_q;
    reg [FEATURE_INDEX_WIDTH-1:0] hidden_index_q;
    reg [FEATURE_INDEX_WIDTH-1:0] feature_index_q;
    reg [TOKEN_INDEX_WIDTH-1:0] token_index_q;
    reg signed [95:0] accumulator_q;

    reg signed [40:0] hidden_q24_q [0:HIDDEN_SIZE-1];
    reg top_slot_valid_q [0:TOP_K-1];
    reg [TOKEN_INDEX_WIDTH-1:0] top_token_q [0:TOP_K-1];
    reg [15:0] top_logit_q [0:TOP_K-1];
    reg signed [40:0] top_value_q [0:TOP_K-1];

    reg top_slot_valid_d [0:TOP_K-1];
    reg [TOKEN_INDEX_WIDTH-1:0] top_token_d [0:TOP_K-1];
    reg [15:0] top_logit_d [0:TOP_K-1];
    reg signed [40:0] top_value_d [0:TOP_K-1];
    reg insert_valid;
    reg [TOKEN_INDEX_WIDTH-1:0] insert_token;
    reg [15:0] insert_logit;
    reg signed [40:0] insert_value;
    reg swap_valid;
    reg [TOKEN_INDEX_WIDTH-1:0] swap_token;
    reg [15:0] swap_logit;
    reg signed [40:0] swap_value;

    reg logit_valid_q;
    reg [TOKEN_INDEX_WIDTH-1:0] logit_token_q;
    reg [15:0] logit_f16_q;
    reg signed [95:0] logit_accumulator_q;
    reg logit_saturation_q;
    reg [TOP_RANK_WIDTH-1:0] top_rank_q;
    reg [3:0] error_code_q;
    reg invalid_operand_q;
    reg saturation_q;

    wire signed [40:0] hidden_input_q24_w;
    wire hidden_input_finite_w;
    wire hidden_input_sign_unused_w;
    wire signed [40:0] weight_input_q24_w;
    wire weight_input_finite_w;
    wire weight_input_sign_unused_w;
    wire signed [81:0] product_w;
    wire signed [95:0] product_extended_w;
    wire signed [95:0] accumulated_w;
    wire [15:0] rounded_logit_raw_w;
    wire [15:0] rounded_logit_w;
    wire rounded_saturation_w;
    wire signed [40:0] held_logit_q24_w;
    wire held_logit_finite_unused_w;
    wire held_logit_sign_unused_w;
    wire parameters_valid_w;

    integer index;

    function automatic known1;
        input value;
        begin
            known1 = (value === 1'b0) || (value === 1'b1);
        end
    endfunction

    function automatic known16;
        input [15:0] value;
        begin
            known16 = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic known_feature_index;
        input [FEATURE_INDEX_WIDTH-1:0] value;
        begin
            known_feature_index = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic known_token_index;
        input [TOKEN_INDEX_WIDTH-1:0] value;
        begin
            known_token_index = ((^value === 1'b0) || (^value === 1'b1));
        end
    endfunction

    function automatic candidate_better;
        input signed [40:0] candidate_value;
        input [TOKEN_INDEX_WIDTH-1:0] candidate_token;
        input signed [40:0] resident_value;
        input [TOKEN_INDEX_WIDTH-1:0] resident_token;
        begin
            candidate_better =
                (candidate_value > resident_value) ||
                ((candidate_value == resident_value) &&
                 (candidate_token < resident_token));
        end
    endfunction

    assign parameters_valid_w =
        (HIDDEN_SIZE > 0) && (VOCAB_SIZE >= TOP_K) && (TOP_K > 0) &&
        (HIDDEN_SIZE <= (1 << FEATURE_INDEX_WIDTH)) &&
        (VOCAB_SIZE <= (1 << TOKEN_INDEX_WIDTH)) &&
        (TOP_K <= (1 << TOP_RANK_WIDTH));

    assign start_ready_o = rst_ni && !clear_i && parameters_valid_w &&
                           (state_q == ST_IDLE);
    assign hidden_ready_o = rst_ni && !clear_i && (state_q == ST_HIDDEN);
    assign weight_ready_o = rst_ni && !clear_i && (state_q == ST_WEIGHT);
    assign logit_valid_o = logit_valid_q;
    assign logit_token_index_o = logit_token_q;
    assign logit_f16_o = logit_f16_q;
    assign acc_q47_48_o = logit_accumulator_q;
    assign logit_saturation_o = logit_saturation_q;
    assign top_valid_o = (state_q == ST_TOP) && top_slot_valid_q[top_rank_q];
    assign top_rank_o = top_rank_q;
    assign top_token_index_o = top_token_q[top_rank_q];
    assign top_logit_f16_o = top_logit_q[top_rank_q];
    assign done_valid_o = state_q == ST_DONE;
    assign error_valid_o = state_q == ST_ERROR;
    assign error_code_o = error_code_q;
    assign invalid_operand_o = invalid_operand_q;
    assign saturation_o = saturation_q;
    assign busy_o = state_q != ST_IDLE;

    ace3_fp16_to_q24 decode_hidden_input (
        .f16_i(hidden_f16_i),
        .q24_o(hidden_input_q24_w),
        .finite_o(hidden_input_finite_w),
        .sign_o(hidden_input_sign_unused_w)
    );

    ace3_fp16_to_q24 decode_weight_input (
        .f16_i(weight_f16_i),
        .q24_o(weight_input_q24_w),
        .finite_o(weight_input_finite_w),
        .sign_o(weight_input_sign_unused_w)
    );

    assign product_w =
        $signed(hidden_q24_q[feature_index_q]) * $signed(weight_input_q24_w);
    assign product_extended_w = {{14{product_w[81]}}, product_w};
    assign accumulated_w = accumulator_q + product_extended_w;

    ace3_q47_48_to_f16_rne #(
        .ACC_WIDTH(96)
    ) round_logit (
        .fixed_i(accumulated_w),
        .f16_o(rounded_logit_raw_w),
        .saturation_o(rounded_saturation_w)
    );

    assign rounded_logit_w =
        ((rounded_logit_raw_w[14:0] == 15'd0) && accumulated_w[95])
            ? 16'h8000
            : rounded_logit_raw_w;

    ace3_fp16_to_q24 decode_held_logit (
        .f16_i(logit_f16_q),
        .q24_o(held_logit_q24_w),
        .finite_o(held_logit_finite_unused_w),
        .sign_o(held_logit_sign_unused_w)
    );

    always @* begin
        for (index = 0; index < TOP_K; index = index + 1) begin
            top_slot_valid_d[index] = top_slot_valid_q[index];
            top_token_d[index] = top_token_q[index];
            top_logit_d[index] = top_logit_q[index];
            top_value_d[index] = top_value_q[index];
        end
        insert_valid = logit_valid_q;
        insert_token = logit_token_q;
        insert_logit = logit_f16_q;
        insert_value = held_logit_q24_w;
        swap_valid = 1'b0;
        swap_token = {TOKEN_INDEX_WIDTH{1'b0}};
        swap_logit = 16'd0;
        swap_value = 41'sd0;
        for (index = 0; index < TOP_K; index = index + 1) begin
            if (insert_valid &&
                (!top_slot_valid_d[index] ||
                 candidate_better(
                     insert_value,
                     insert_token,
                     top_value_d[index],
                     top_token_d[index]
                 ))) begin
                swap_valid = top_slot_valid_d[index];
                swap_token = top_token_d[index];
                swap_logit = top_logit_d[index];
                swap_value = top_value_d[index];
                top_slot_valid_d[index] = 1'b1;
                top_token_d[index] = insert_token;
                top_logit_d[index] = insert_logit;
                top_value_d[index] = insert_value;
                insert_valid = swap_valid;
                insert_token = swap_token;
                insert_logit = swap_logit;
                insert_value = swap_value;
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            hidden_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
            feature_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
            token_index_q <= {TOKEN_INDEX_WIDTH{1'b0}};
            accumulator_q <= 96'sd0;
            logit_valid_q <= 1'b0;
            logit_token_q <= {TOKEN_INDEX_WIDTH{1'b0}};
            logit_f16_q <= 16'd0;
            logit_accumulator_q <= 96'sd0;
            logit_saturation_q <= 1'b0;
            top_rank_q <= {TOP_RANK_WIDTH{1'b0}};
            error_code_q <= 4'd0;
            invalid_operand_q <= 1'b0;
            saturation_q <= 1'b0;
            for (index = 0; index < TOP_K; index = index + 1) begin
                top_slot_valid_q[index] <= 1'b0;
                top_token_q[index] <= {TOKEN_INDEX_WIDTH{1'b0}};
                top_logit_q[index] <= 16'd0;
                top_value_q[index] <= 41'sd0;
            end
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            hidden_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
            feature_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
            token_index_q <= {TOKEN_INDEX_WIDTH{1'b0}};
            accumulator_q <= 96'sd0;
            logit_valid_q <= 1'b0;
            logit_token_q <= {TOKEN_INDEX_WIDTH{1'b0}};
            logit_f16_q <= 16'd0;
            logit_accumulator_q <= 96'sd0;
            logit_saturation_q <= 1'b0;
            top_rank_q <= {TOP_RANK_WIDTH{1'b0}};
            error_code_q <= 4'd0;
            invalid_operand_q <= 1'b0;
            saturation_q <= 1'b0;
            for (index = 0; index < TOP_K; index = index + 1) begin
                top_slot_valid_q[index] <= 1'b0;
                top_token_q[index] <= {TOKEN_INDEX_WIDTH{1'b0}};
                top_logit_q[index] <= 16'd0;
                top_value_q[index] <= 41'sd0;
            end
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= ST_HIDDEN;
                        hidden_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
                        feature_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
                        token_index_q <= {TOKEN_INDEX_WIDTH{1'b0}};
                        accumulator_q <= 96'sd0;
                        logit_valid_q <= 1'b0;
                        logit_token_q <= {TOKEN_INDEX_WIDTH{1'b0}};
                        logit_f16_q <= 16'd0;
                        logit_accumulator_q <= 96'sd0;
                        logit_saturation_q <= 1'b0;
                        top_rank_q <= {TOP_RANK_WIDTH{1'b0}};
                        error_code_q <= 4'd0;
                        invalid_operand_q <= 1'b0;
                        saturation_q <= 1'b0;
                        for (index = 0; index < TOP_K; index = index + 1)
                            top_slot_valid_q[index] <= 1'b0;
                    end
                end

                ST_HIDDEN: begin
                    if ((hidden_end_i === 1'b1) &&
                        !((hidden_valid_i === 1'b1) && hidden_ready_o)) begin
                        state_q <= ST_ERROR;
                        error_code_q <= ERROR_FRAMING;
                    end else if ((hidden_valid_i === 1'b1) && hidden_ready_o) begin
                        if (!known_feature_index(hidden_index_i) ||
                            !known16(hidden_f16_i) ||
                            !known1(hidden_last_i) ||
                            !known1(hidden_end_i)) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_UNKNOWN;
                        end else if (!hidden_input_finite_w) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_NONFINITE;
                            invalid_operand_q <= 1'b1;
                        end else if (hidden_index_i != hidden_index_q) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_ORDER;
                        end else if ((hidden_last_i !=
                                      (hidden_index_q == LAST_FEATURE)) ||
                                     (hidden_end_i !=
                                      (hidden_index_q == LAST_FEATURE))) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_FRAMING;
                        end else begin
                            hidden_q24_q[hidden_index_q] <= hidden_input_q24_w;
                            if (hidden_index_q == LAST_FEATURE) begin
                                state_q <= ST_WEIGHT;
                                feature_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
                                token_index_q <= {TOKEN_INDEX_WIDTH{1'b0}};
                                accumulator_q <= 96'sd0;
                            end else begin
                                hidden_index_q <= hidden_index_q + 1'b1;
                            end
                        end
                    end
                end

                ST_WEIGHT: begin
                    if ((weight_end_i === 1'b1) &&
                        !((weight_valid_i === 1'b1) && weight_ready_o)) begin
                        state_q <= ST_ERROR;
                        error_code_q <= ERROR_FRAMING;
                    end else if ((weight_valid_i === 1'b1) && weight_ready_o) begin
                        if (!known_token_index(weight_token_index_i) ||
                            !known_feature_index(weight_feature_index_i) ||
                            !known16(weight_f16_i) ||
                            !known1(weight_last_feature_i) ||
                            !known1(weight_last_token_i) ||
                            !known1(weight_end_i)) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_UNKNOWN;
                        end else if (!weight_input_finite_w) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_NONFINITE;
                            invalid_operand_q <= 1'b1;
                        end else if ((weight_token_index_i != token_index_q) ||
                                     (weight_feature_index_i != feature_index_q)) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_ORDER;
                        end else if ((weight_last_feature_i !=
                                      (feature_index_q == LAST_FEATURE)) ||
                                     (weight_last_token_i !=
                                      ((feature_index_q == LAST_FEATURE) &&
                                       (token_index_q == LAST_TOKEN))) ||
                                     (weight_end_i !=
                                      ((feature_index_q == LAST_FEATURE) &&
                                       (token_index_q == LAST_TOKEN)))) begin
                            state_q <= ST_ERROR;
                            error_code_q <= ERROR_FRAMING;
                        end else if (feature_index_q == LAST_FEATURE) begin
                            logit_valid_q <= 1'b1;
                            logit_token_q <= token_index_q;
                            logit_f16_q <= rounded_logit_w;
                            logit_accumulator_q <= accumulated_w;
                            logit_saturation_q <= rounded_saturation_w;
                            saturation_q <= saturation_q || rounded_saturation_w;
                            state_q <= ST_LOGIT;
                        end else begin
                            accumulator_q <= accumulated_w;
                            feature_index_q <= feature_index_q + 1'b1;
                        end
                    end
                end

                ST_LOGIT: begin
                    if (logit_valid_q && (logit_ready_i === 1'b1)) begin
                        for (index = 0; index < TOP_K; index = index + 1) begin
                            top_slot_valid_q[index] <= top_slot_valid_d[index];
                            top_token_q[index] <= top_token_d[index];
                            top_logit_q[index] <= top_logit_d[index];
                            top_value_q[index] <= top_value_d[index];
                        end
                        logit_valid_q <= 1'b0;
                        accumulator_q <= 96'sd0;
                        feature_index_q <= {FEATURE_INDEX_WIDTH{1'b0}};
                        if (token_index_q == LAST_TOKEN) begin
                            top_rank_q <= {TOP_RANK_WIDTH{1'b0}};
                            state_q <= ST_TOP;
                        end else begin
                            token_index_q <= token_index_q + 1'b1;
                            state_q <= ST_WEIGHT;
                        end
                    end
                end

                ST_TOP: begin
                    if (top_valid_o && (top_ready_i === 1'b1)) begin
                        if (top_rank_q == LAST_RANK)
                            state_q <= ST_DONE;
                        else
                            top_rank_q <= top_rank_q + 1'b1;
                    end
                end

                ST_DONE: begin
                    if (done_ready_i === 1'b1)
                        state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_ERROR;
                    if (error_code_q == 4'd0)
                        error_code_q <= ERROR_UNKNOWN;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
