`timescale 1ns/1ps
`default_nettype none

module ace3_generated_token_feedback #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer VOCAB_SIZE = 151936,
    parameter integer TOP_K = 10,
    parameter integer TOKEN_INDEX_WIDTH = 18,
    parameter integer FEATURE_INDEX_WIDTH = 10,
    parameter integer TOP_RANK_WIDTH = 4,
    parameter [255:0] EXPECTED_CHECKPOINT_SHA256 =
        256'hc50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b,
    parameter [255:0] EXPECTED_VOCABULARY_SHA256 =
        256'hc0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539,
    parameter [255:0] EXPECTED_TOKENIZER_CONFIG_SHA256 =
        256'h5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
) (
    input wire clk_i, input wire rst_ni, input wire clear_i,
    input wire start_valid_i, output wire start_ready_o,
    input wire [255:0] checkpoint_sha256_i,
    input wire [255:0] vocabulary_sha256_i,
    input wire [255:0] tokenizer_config_sha256_i,
    input wire [255:0] trusted_state_tip_i,
    input wire [255:0] presented_state_tip_i,
    input wire [31:0] prior_position_i, input wire [31:0] next_position_i,
    input wire top_valid_i, output wire top_ready_o,
    input wire [TOP_RANK_WIDTH-1:0] top_rank_i,
    input wire [TOKEN_INDEX_WIDTH-1:0] top_token_index_i,
    input wire [15:0] top_logit_f16_i, input wire top_last_i,
    output wire selected_valid_o, input wire selected_ready_i,
    output wire [TOKEN_INDEX_WIDTH-1:0] selected_token_index_o,
    output wire [15:0] selected_logit_f16_o,
    output wire embedding_request_valid_o, input wire embedding_request_ready_i,
    output wire [TOKEN_INDEX_WIDTH-1:0] embedding_request_token_index_o,
    input wire embedding_valid_i, output wire embedding_ready_o,
    input wire [TOKEN_INDEX_WIDTH-1:0] embedding_token_index_i,
    input wire [FEATURE_INDEX_WIDTH-1:0] embedding_feature_index_i,
    input wire [15:0] embedding_f16_i, input wire embedding_last_i,
    input wire embedding_end_i,
    output wire next_input_valid_o, input wire next_input_ready_i,
    output wire [TOKEN_INDEX_WIDTH-1:0] next_input_token_index_o,
    output wire [FEATURE_INDEX_WIDTH-1:0] next_input_feature_index_o,
    output wire [15:0] next_input_f16_o, output wire next_input_last_o,
    output wire [31:0] next_input_position_o,
    output wire commit_valid_o, input wire commit_ready_i,
    output wire [TOKEN_INDEX_WIDTH-1:0] commit_token_index_o,
    output wire [31:0] commit_position_o,
    output wire [255:0] commit_prior_state_tip_o,
    output wire error_valid_o, output wire [3:0] error_code_o,
    output wire busy_o
);
    localparam [3:0] ST_IDLE=4'd0, ST_TOP=4'd1, ST_SELECTED=4'd2,
        ST_REQUEST=4'd3, ST_EMBED=4'd4, ST_DRAIN=4'd5,
        ST_COMMIT=4'd6, ST_ERROR=4'd7;
    localparam [3:0] ERROR_UNKNOWN=4'd1, ERROR_BINDING=4'd2,
        ERROR_TOP_ORDER=4'd3, ERROR_TOKEN=4'd4, ERROR_EMBED_ORDER=4'd5,
        ERROR_FRAMING=4'd6, ERROR_NONFINITE=4'd7, ERROR_DUPLICATE=4'd8;
    localparam [FEATURE_INDEX_WIDTH-1:0] LAST_FEATURE = FEATURE_INDEX_WIDTH'(HIDDEN_SIZE-1);
    localparam [TOP_RANK_WIDTH-1:0] LAST_RANK = TOP_RANK_WIDTH'(TOP_K-1);
    reg [3:0] state_q, error_code_q;
    reg [TOP_RANK_WIDTH-1:0] expected_rank_q;
    reg [FEATURE_INDEX_WIDTH-1:0] expected_feature_q, next_feature_q;
    reg [TOKEN_INDEX_WIDTH-1:0] selected_token_q;
    reg [15:0] selected_logit_q, next_f16_q;
    reg selected_pending_q, next_valid_q, next_last_q;
    reg [255:0] prior_state_tip_q;
    reg [31:0] next_position_q;

    function automatic known1(input value); known1=(value===1'b0)||(value===1'b1); endfunction
    function automatic known16(input [15:0] value); known16=(^value===1'b0)||(^value===1'b1); endfunction
    function automatic known32(input [31:0] value); known32=(^value===1'b0)||(^value===1'b1); endfunction
    function automatic known256(input [255:0] value); known256=(^value===1'b0)||(^value===1'b1); endfunction
    function automatic known_token(input [TOKEN_INDEX_WIDTH-1:0] value); known_token=(^value===1'b0)||(^value===1'b1); endfunction
    function automatic known_feature(input [FEATURE_INDEX_WIDTH-1:0] value); known_feature=(^value===1'b0)||(^value===1'b1); endfunction
    function automatic known_rank(input [TOP_RANK_WIDTH-1:0] value); known_rank=(^value===1'b0)||(^value===1'b1); endfunction

    wire parameters_valid_w=(HIDDEN_SIZE>0)&&(VOCAB_SIZE>=TOP_K)&&(TOP_K>0)&&
        (HIDDEN_SIZE<=(1<<FEATURE_INDEX_WIDTH))&&(VOCAB_SIZE<=(1<<TOKEN_INDEX_WIDTH))&&
        (TOP_K<=(1<<TOP_RANK_WIDTH));
    wire metadata_known_w=known256(checkpoint_sha256_i)&&known256(vocabulary_sha256_i)&&
        known256(tokenizer_config_sha256_i)&&known256(trusted_state_tip_i)&&
        known256(presented_state_tip_i)&&known32(prior_position_i)&&known32(next_position_i);
    wire metadata_matches_w=(checkpoint_sha256_i==EXPECTED_CHECKPOINT_SHA256)&&
        (vocabulary_sha256_i==EXPECTED_VOCABULARY_SHA256)&&
        (tokenizer_config_sha256_i==EXPECTED_TOKENIZER_CONFIG_SHA256)&&
        (trusted_state_tip_i!=256'd0)&&(presented_state_tip_i==trusted_state_tip_i)&&
        (prior_position_i!=32'hffffffff)&&(next_position_i==prior_position_i+1'b1);
    assign start_ready_o=rst_ni&&!clear_i&&parameters_valid_w&&(state_q==ST_IDLE);
    assign top_ready_o=rst_ni&&!clear_i&&(state_q==ST_TOP);
    assign selected_valid_o=selected_pending_q;
    assign selected_token_index_o=selected_token_q;
    assign selected_logit_f16_o=selected_logit_q;
    assign embedding_request_valid_o=state_q==ST_REQUEST;
    assign embedding_request_token_index_o=selected_token_q;
    assign embedding_ready_o=rst_ni&&!clear_i&&(state_q==ST_EMBED)&&
        (!next_valid_q||(next_input_ready_i===1'b1));
    assign next_input_valid_o=next_valid_q;
    assign next_input_token_index_o=selected_token_q;
    assign next_input_feature_index_o=next_feature_q;
    assign next_input_f16_o=next_f16_q;
    assign next_input_last_o=next_last_q;
    assign next_input_position_o=next_position_q;
    assign commit_valid_o=state_q==ST_COMMIT;
    assign commit_token_index_o=selected_token_q;
    assign commit_position_o=next_position_q;
    assign commit_prior_state_tip_o=prior_state_tip_q;
    assign error_valid_o=state_q==ST_ERROR;
    assign error_code_o=error_code_q;
    assign busy_o=state_q!=ST_IDLE;

    always @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni) begin
        state_q<=ST_IDLE;error_code_q<=0;expected_rank_q<='0;expected_feature_q<='0;
        selected_token_q<='0;selected_logit_q<=0;selected_pending_q<=0;
        prior_state_tip_q<=0;next_position_q<=0;next_valid_q<=0;next_feature_q<='0;
        next_f16_q<=0;next_last_q<=0;
      end else if(clear_i) begin
        state_q<=ST_IDLE;error_code_q<=0;expected_rank_q<='0;expected_feature_q<='0;
        selected_token_q<='0;selected_logit_q<=0;selected_pending_q<=0;
        prior_state_tip_q<=0;next_position_q<=0;next_valid_q<=0;next_feature_q<='0;
        next_f16_q<=0;next_last_q<=0;
      end else begin
        if(next_valid_q&&(next_input_ready_i===1'b1)) next_valid_q<=0;
        case(state_q)
          ST_IDLE: if(start_valid_i&&start_ready_o) begin
            error_code_q<=0;expected_rank_q<='0;expected_feature_q<='0;
            selected_pending_q<=0;next_valid_q<=0;
            if(!metadata_known_w||!metadata_matches_w) begin state_q<=ST_ERROR;error_code_q<=ERROR_BINDING;end
            else begin prior_state_tip_q<=trusted_state_tip_i;next_position_q<=next_position_i;state_q<=ST_TOP;end
          end
          ST_TOP: begin
            if(selected_pending_q&&(selected_ready_i===1'b1)) selected_pending_q<=0;
            if(top_valid_i&&top_ready_o) begin
              if(!known_rank(top_rank_i)||!known_token(top_token_index_i)||!known16(top_logit_f16_i)||!known1(top_last_i)) begin state_q<=ST_ERROR;error_code_q<=ERROR_UNKNOWN;end
              else if((top_rank_i!=expected_rank_q)||(top_token_index_i>=VOCAB_SIZE)) begin state_q<=ST_ERROR;error_code_q<=ERROR_TOP_ORDER;end
              else if(top_logit_f16_i[14:10]==5'h1f) begin state_q<=ST_ERROR;error_code_q<=ERROR_NONFINITE;end
              else if(top_last_i!=(top_rank_i==LAST_RANK)) begin state_q<=ST_ERROR;error_code_q<=ERROR_FRAMING;end
              else begin
                if(top_rank_i==0) begin selected_token_q<=top_token_index_i;selected_logit_q<=top_logit_f16_i;selected_pending_q<=1;end
                if(top_rank_i==LAST_RANK) begin
                  if((top_rank_i==0)||(selected_pending_q&&!(selected_ready_i===1'b1))) state_q<=ST_SELECTED;
                  else state_q<=ST_REQUEST;
                end else expected_rank_q<=expected_rank_q+1'b1;
              end
            end
          end
          ST_SELECTED: begin
            if(top_valid_i) begin state_q<=ST_ERROR;error_code_q<=ERROR_DUPLICATE;end
            else if(selected_pending_q&&(selected_ready_i===1'b1)) begin selected_pending_q<=0;state_q<=ST_REQUEST;end
          end
          ST_REQUEST: begin
            if(top_valid_i||embedding_valid_i) begin state_q<=ST_ERROR;error_code_q<=ERROR_DUPLICATE;end
            else if(embedding_request_ready_i===1'b1) begin expected_feature_q<='0;state_q<=ST_EMBED;end
          end
          ST_EMBED: begin
            if((embedding_end_i===1'b1)&&!((embedding_valid_i===1'b1)&&embedding_ready_o)) begin state_q<=ST_ERROR;error_code_q<=ERROR_FRAMING;end
            else if((embedding_valid_i===1'b1)&&embedding_ready_o) begin
              if(!known_token(embedding_token_index_i)||!known_feature(embedding_feature_index_i)||!known16(embedding_f16_i)||!known1(embedding_last_i)||!known1(embedding_end_i)) begin state_q<=ST_ERROR;error_code_q<=ERROR_UNKNOWN;end
              else if(embedding_token_index_i!=selected_token_q) begin state_q<=ST_ERROR;error_code_q<=ERROR_TOKEN;end
              else if(embedding_feature_index_i!=expected_feature_q) begin state_q<=ST_ERROR;error_code_q<=ERROR_EMBED_ORDER;end
              else if(embedding_f16_i[14:10]==5'h1f) begin state_q<=ST_ERROR;error_code_q<=ERROR_NONFINITE;end
              else if((embedding_last_i!=(expected_feature_q==LAST_FEATURE))||(embedding_end_i!=(expected_feature_q==LAST_FEATURE))) begin state_q<=ST_ERROR;error_code_q<=ERROR_FRAMING;end
              else begin
                next_valid_q<=1;next_feature_q<=expected_feature_q;next_f16_q<=embedding_f16_i;next_last_q<=expected_feature_q==LAST_FEATURE;
                if(expected_feature_q==LAST_FEATURE) state_q<=ST_DRAIN;else expected_feature_q<=expected_feature_q+1'b1;
              end
            end
          end
          ST_DRAIN: begin
            if(embedding_valid_i) begin state_q<=ST_ERROR;error_code_q<=ERROR_DUPLICATE;end
            else if(next_valid_q&&(next_input_ready_i===1'b1)) state_q<=ST_COMMIT;
          end
          ST_COMMIT: begin
            if(embedding_valid_i||top_valid_i) begin state_q<=ST_ERROR;error_code_q<=ERROR_DUPLICATE;end
            else if(commit_ready_i===1'b1) state_q<=ST_IDLE;
          end
          default: begin state_q<=ST_ERROR;error_code_q<=ERROR_UNKNOWN;end
        endcase
      end
    end
endmodule
`default_nettype wire
