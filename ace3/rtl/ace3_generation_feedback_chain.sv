`timescale 1ns/1ps
`default_nettype none
module ace3_generation_feedback_chain #(
 parameter integer HIDDEN_SIZE=896,parameter integer VOCAB_SIZE=151936,
 parameter integer TOP_K=10,parameter integer TOKEN_INDEX_WIDTH=18,
 parameter integer FEATURE_INDEX_WIDTH=10,parameter integer TOP_RANK_WIDTH=4,
 parameter [255:0] EXPECTED_CHECKPOINT_SHA256=256'hc50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b,
 parameter [255:0] EXPECTED_VOCABULARY_SHA256=256'hc0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539,
 parameter [255:0] EXPECTED_TOKENIZER_CONFIG_SHA256=256'h5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
)(
 input wire clk_i,input wire rst_ni,input wire clear_i,input wire start_valid_i,output wire start_ready_o,
 input wire [255:0] checkpoint_sha256_i,input wire [255:0] vocabulary_sha256_i,input wire [255:0] tokenizer_config_sha256_i,
 input wire [255:0] trusted_state_tip_i,input wire [255:0] presented_state_tip_i,input wire [31:0] prior_position_i,input wire [31:0] next_position_i,
 input wire hidden_valid_i,output wire hidden_ready_o,input wire [FEATURE_INDEX_WIDTH-1:0] hidden_index_i,input wire [15:0] hidden_f16_i,input wire hidden_last_i,input wire hidden_end_i,
 input wire weight_valid_i,output wire weight_ready_o,input wire [TOKEN_INDEX_WIDTH-1:0] weight_token_index_i,input wire [FEATURE_INDEX_WIDTH-1:0] weight_feature_index_i,input wire [15:0] weight_f16_i,input wire weight_last_feature_i,input wire weight_last_token_i,input wire weight_end_i,
 output wire logit_valid_o,input wire logit_ready_i,output wire [TOKEN_INDEX_WIDTH-1:0] logit_token_index_o,output wire [15:0] logit_f16_o,
 output wire selected_valid_o,input wire selected_ready_i,output wire [TOKEN_INDEX_WIDTH-1:0] selected_token_index_o,output wire [15:0] selected_logit_f16_o,
 output wire embedding_request_valid_o,input wire embedding_request_ready_i,output wire [TOKEN_INDEX_WIDTH-1:0] embedding_request_token_index_o,
 input wire embedding_valid_i,output wire embedding_ready_o,input wire [TOKEN_INDEX_WIDTH-1:0] embedding_token_index_i,input wire [FEATURE_INDEX_WIDTH-1:0] embedding_feature_index_i,input wire [15:0] embedding_f16_i,input wire embedding_last_i,input wire embedding_end_i,
 output wire next_input_valid_o,input wire next_input_ready_i,output wire [TOKEN_INDEX_WIDTH-1:0] next_input_token_index_o,output wire [FEATURE_INDEX_WIDTH-1:0] next_input_feature_index_o,output wire [15:0] next_input_f16_o,output wire next_input_last_o,output wire [31:0] next_input_position_o,
 output wire commit_valid_o,input wire commit_ready_i,output wire [TOKEN_INDEX_WIDTH-1:0] commit_token_index_o,output wire [31:0] commit_position_o,output wire [255:0] commit_prior_state_tip_o,
 output wire error_valid_o,output wire [3:0] error_code_o,output wire invalid_operand_o,output wire saturation_o,output wire busy_o);
 wire lm_start_ready,fb_start_ready,lm_top_valid,fb_top_ready,lm_done,lm_error,fb_error,lm_invalid,lm_sat,lm_busy,fb_busy;
 wire [TOP_RANK_WIDTH-1:0] lm_rank;wire [TOKEN_INDEX_WIDTH-1:0] lm_token;wire [15:0] lm_top_logit;wire [3:0] lm_error_code,fb_error_code;
 assign start_ready_o=lm_start_ready&&fb_start_ready;assign error_valid_o=lm_error||fb_error;assign error_code_o=lm_error?lm_error_code:fb_error_code;
 assign invalid_operand_o=lm_invalid;assign saturation_o=lm_sat;assign busy_o=lm_busy||fb_busy;
 ace3_streaming_tied_lm_head_topk #(.HIDDEN_SIZE(HIDDEN_SIZE),.VOCAB_SIZE(VOCAB_SIZE),.TOP_K(TOP_K),.TOKEN_INDEX_WIDTH(TOKEN_INDEX_WIDTH),.FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),.TOP_RANK_WIDTH(TOP_RANK_WIDTH)) lm_head(
  .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(start_valid_i&&fb_start_ready),.start_ready_o(lm_start_ready),
  .hidden_valid_i(hidden_valid_i),.hidden_ready_o(hidden_ready_o),.hidden_index_i(hidden_index_i),.hidden_f16_i(hidden_f16_i),.hidden_last_i(hidden_last_i),.hidden_end_i(hidden_end_i),
  .weight_valid_i(weight_valid_i),.weight_ready_o(weight_ready_o),.weight_token_index_i(weight_token_index_i),.weight_feature_index_i(weight_feature_index_i),.weight_f16_i(weight_f16_i),.weight_last_feature_i(weight_last_feature_i),.weight_last_token_i(weight_last_token_i),.weight_end_i(weight_end_i),
  .logit_valid_o(logit_valid_o),.logit_ready_i(logit_ready_i),.logit_token_index_o(logit_token_index_o),.logit_f16_o(logit_f16_o),.acc_q47_48_o(),.logit_saturation_o(),
  .top_valid_o(lm_top_valid),.top_ready_i(fb_top_ready),.top_rank_o(lm_rank),.top_token_index_o(lm_token),.top_logit_f16_o(lm_top_logit),
  .done_valid_o(lm_done),.done_ready_i(1'b1),.error_valid_o(lm_error),.error_code_o(lm_error_code),.invalid_operand_o(lm_invalid),.saturation_o(lm_sat),.busy_o(lm_busy));
 ace3_generated_token_feedback #(.HIDDEN_SIZE(HIDDEN_SIZE),.VOCAB_SIZE(VOCAB_SIZE),.TOP_K(TOP_K),.TOKEN_INDEX_WIDTH(TOKEN_INDEX_WIDTH),.FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),.TOP_RANK_WIDTH(TOP_RANK_WIDTH),.EXPECTED_CHECKPOINT_SHA256(EXPECTED_CHECKPOINT_SHA256),.EXPECTED_VOCABULARY_SHA256(EXPECTED_VOCABULARY_SHA256),.EXPECTED_TOKENIZER_CONFIG_SHA256(EXPECTED_TOKENIZER_CONFIG_SHA256)) feedback(
  .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.start_valid_i(start_valid_i&&lm_start_ready),.start_ready_o(fb_start_ready),.checkpoint_sha256_i(checkpoint_sha256_i),.vocabulary_sha256_i(vocabulary_sha256_i),.tokenizer_config_sha256_i(tokenizer_config_sha256_i),.trusted_state_tip_i(trusted_state_tip_i),.presented_state_tip_i(presented_state_tip_i),.prior_position_i(prior_position_i),.next_position_i(next_position_i),
  .top_valid_i(lm_top_valid),.top_ready_o(fb_top_ready),.top_rank_i(lm_rank),.top_token_index_i(lm_token),.top_logit_f16_i(lm_top_logit),.top_last_i(lm_rank==TOP_K-1),
  .selected_valid_o(selected_valid_o),.selected_ready_i(selected_ready_i),.selected_token_index_o(selected_token_index_o),.selected_logit_f16_o(selected_logit_f16_o),
  .embedding_request_valid_o(embedding_request_valid_o),.embedding_request_ready_i(embedding_request_ready_i),.embedding_request_token_index_o(embedding_request_token_index_o),
  .embedding_valid_i(embedding_valid_i),.embedding_ready_o(embedding_ready_o),.embedding_token_index_i(embedding_token_index_i),.embedding_feature_index_i(embedding_feature_index_i),.embedding_f16_i(embedding_f16_i),.embedding_last_i(embedding_last_i),.embedding_end_i(embedding_end_i),
  .next_input_valid_o(next_input_valid_o),.next_input_ready_i(next_input_ready_i),.next_input_token_index_o(next_input_token_index_o),.next_input_feature_index_o(next_input_feature_index_o),.next_input_f16_o(next_input_f16_o),.next_input_last_o(next_input_last_o),.next_input_position_o(next_input_position_o),
  .commit_valid_o(commit_valid_o),.commit_ready_i(commit_ready_i),.commit_token_index_o(commit_token_index_o),.commit_position_o(commit_position_o),.commit_prior_state_tip_o(commit_prior_state_tip_o),.error_valid_o(fb_error),.error_code_o(fb_error_code),.busy_o(fb_busy));
endmodule
`default_nettype wire
