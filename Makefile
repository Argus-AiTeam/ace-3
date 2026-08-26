.DEFAULT_GOAL := test
.NOTPARALLEL:

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(ROOT)/build
VECTOR_DIR := $(BUILD_DIR)/vectors
PROJECTION_VECTOR_DIR := $(BUILD_DIR)/projection_vectors
LOG_DIR := $(BUILD_DIR)/logs
IVERILOG_DIR := $(BUILD_DIR)/iverilog
VERILATOR_DIR := $(BUILD_DIR)/verilator
VERILATOR_OBJ_DIR := $(VERILATOR_DIR)/obj_dir
PROJECTION_VERILATOR_DIR := $(BUILD_DIR)/projection_verilator
PROJECTION_VERILATOR_OBJ_DIR := $(PROJECTION_VERILATOR_DIR)/obj_dir
PROJECTION_GEOMETRY_DIR := $(BUILD_DIR)/projection_geometry

PYTHON ?= python3
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
STRACE ?= strace
OFFICIAL_TENSOR_DIR ?= $(ROOT)/ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj

RTL := $(ROOT)/ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv
PROJECTION_ROUNDER_RTL := $(ROOT)/ace3/rtl/ace3_q47_48_to_f16_rne.sv
PROJECTION_RTL := $(ROOT)/ace3/rtl/ace3_awq_w4a16_projection_engine.sv
SV_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_tb.sv
PROTOCOL_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_protocol_tb.sv
CPP_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_main.cpp
PROJECTION_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_projection_engine_tb.sv
PROJECTION_4864_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_projection_4864_cycle_tb.sv
PROJECTION_CPP_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_projection_engine_main.cpp
ORACLE := $(ROOT)/ace3/model/awq_bit_oracle.py
GENERATOR := $(ROOT)/ace3/model/generate_vectors.py
VALIDATOR := $(ROOT)/ace3/model/validate_vectors.py
PROJECTION_ORACLE := $(ROOT)/ace3/model/projection_oracle.py
PROJECTION_GENERATOR := $(ROOT)/ace3/model/generate_projection_vectors.py
PROJECTION_VALIDATOR := $(ROOT)/ace3/model/validate_projection_vectors.py
CONTRACT := $(ROOT)/ace3/contracts/awq_w4a16_g128_dot_lane.json
EVIDENCE_BINDINGS := $(ROOT)/ace3/contracts/awq_w4a16_g128_evidence_bindings.json
FROZEN_MANIFEST := $(ROOT)/ace3/contracts/awq_w4a16_g128_vectors_manifest.json
STANDALONE_BINDINGS := $(ROOT)/ace3/contracts/awq_w4a16_g128_standalone_vector_bindings.json
PROJECTION_CONTRACT := $(ROOT)/ace3/contracts/awq_w4a16_projection_engine.json
PROJECTION_BINDINGS := $(ROOT)/ace3/contracts/awq_w4a16_projection_vector_bindings.json
TAMPER_DIR := $(BUILD_DIR)/tamper-vectors
PROJECTION_TAMPER_DIR := $(BUILD_DIR)/tamper-projection-vectors
FP16_VECTOR_DIR := $(BUILD_DIR)/fp16_adaptation_vectors
FP16_TAMPER_DIR := $(BUILD_DIR)/tamper-fp16-adaptation-vectors
FP16_IVERILOG_DIR := $(BUILD_DIR)/fp16_iverilog
FP16_IVERILOG_BIN := $(FP16_IVERILOG_DIR)/ace3_fp16_adaptation.vvp
FP16_VERILATOR_DIR := $(BUILD_DIR)/fp16_verilator
FP16_VERILATOR_OBJ_DIR := $(FP16_VERILATOR_DIR)/obj_dir
FP16_VERILATOR_BIN := $(FP16_VERILATOR_OBJ_DIR)/Vace3_fp16_adaptation_verilator_top
FP16_GEOMETRY_DIR := $(BUILD_DIR)/fp16_geometry
FP16_FIXED_RTL := $(ROOT)/ace3/rtl/ace3_fp16_fixed.sv
FP16_RESIDUAL_RTL := $(ROOT)/ace3/rtl/ace3_fp16_residual_add_core.sv
FP16_RMS_RTL := $(ROOT)/ace3/rtl/ace3_fp16_rmsnorm_core.sv
FP16_SILU_RTL := $(ROOT)/ace3/rtl/ace3_fp16_silu_gate_core.sv
FP16_RTL := $(FP16_FIXED_RTL) $(FP16_RESIDUAL_RTL) $(FP16_RMS_RTL) $(FP16_SILU_RTL)
FP16_ORACLE := $(ROOT)/ace3/model/fp16_adaptation_oracle.py
FP16_GENERATOR := $(ROOT)/ace3/model/generate_fp16_adaptation_vectors.py
FP16_VALIDATOR := $(ROOT)/ace3/model/validate_fp16_adaptation_vectors.py
FP16_CONTRACT := $(ROOT)/ace3/contracts/fp16_adaptation_operators.json
FP16_BINDINGS := $(ROOT)/ace3/contracts/fp16_adaptation_vector_bindings.json
FP16_TB := $(ROOT)/ace3/tb/ace3_fp16_adaptation_tb.sv
FP16_VERILATOR_TOP := $(ROOT)/ace3/tb/ace3_fp16_adaptation_verilator_top.sv
FP16_CPP_TB := $(ROOT)/ace3/tb/ace3_fp16_adaptation_main.cpp
QKV_VECTOR_DIR := $(BUILD_DIR)/qkv_rope_cache_vectors
QKV_TAMPER_DIR := $(BUILD_DIR)/tamper-qkv-rope-cache-vectors
QKV_IVERILOG_DIR := $(BUILD_DIR)/qkv_iverilog
QKV_IVERILOG_BIN := $(QKV_IVERILOG_DIR)/ace3_qkv_rope_cache.vvp
QKV_GEOMETRY_BIN := $(QKV_IVERILOG_DIR)/ace3_qkv_projection_geometry.vvp
QKV_VERILATOR_DIR := $(BUILD_DIR)/qkv_verilator
QKV_VERILATOR_OBJ_DIR := $(QKV_VERILATOR_DIR)/obj_dir
QKV_VERILATOR_BIN := $(QKV_VERILATOR_OBJ_DIR)/Vace3_qkv_rope_cache_verilator_top
QKV_ROPE_RTL := $(ROOT)/ace3/rtl/ace3_qwen2_rope_pair.sv
QKV_CACHE_RTL := $(ROOT)/ace3/rtl/ace3_fp16_kv_cache.sv
QKV_CLUSTER_RTL := $(ROOT)/ace3/rtl/ace3_qkv_projection_cluster.sv
QKV_ORACLE := $(ROOT)/ace3/model/qwen2_rope_oracle.py
QKV_GENERATOR := $(ROOT)/ace3/model/generate_qkv_rope_cache_vectors.py
QKV_VALIDATOR := $(ROOT)/ace3/model/validate_qkv_rope_cache_vectors.py
QKV_CONTRACT := $(ROOT)/ace3/contracts/qkv_rope_kv_cache.json
QKV_BINDINGS := $(ROOT)/ace3/contracts/qkv_rope_kv_cache_vector_bindings.json
QKV_TB := $(ROOT)/ace3/tb/ace3_qkv_rope_cache_tb.sv
QKV_GEOMETRY_TB := $(ROOT)/ace3/tb/ace3_qkv_projection_geometry_tb.sv
QKV_VERILATOR_TOP := $(ROOT)/ace3/tb/ace3_qkv_rope_cache_verilator_top.sv
QKV_CPP_TB := $(ROOT)/ace3/tb/ace3_qkv_rope_cache_main.cpp
ATTENTION_VECTOR_DIR := $(BUILD_DIR)/attention_vectors
ATTENTION_TAMPER_DIR := $(BUILD_DIR)/tamper-attention-vectors
ATTENTION_IVERILOG_DIR := $(BUILD_DIR)/attention_iverilog
ATTENTION_IVERILOG_BIN := $(ATTENTION_IVERILOG_DIR)/ace3_attention_block.vvp
ATTENTION_VERILATOR_DIR := $(BUILD_DIR)/attention_verilator
ATTENTION_VERILATOR_OBJ_DIR := $(ATTENTION_VERILATOR_DIR)/obj_dir
ATTENTION_VERILATOR_BIN := $(ATTENTION_VERILATOR_OBJ_DIR)/Vace3_attention_verilator_top
ATTENTION_SCORE_RTL := $(ROOT)/ace3/rtl/ace3_attention_score_core.sv
ATTENTION_SOFTMAX_RTL := $(ROOT)/ace3/rtl/ace3_attention_softmax_core.sv
ATTENTION_VALUE_RTL := $(ROOT)/ace3/rtl/ace3_attention_value_core.sv
ATTENTION_RTL := $(ATTENTION_SCORE_RTL) $(ATTENTION_SOFTMAX_RTL) $(ATTENTION_VALUE_RTL)
ATTENTION_ORACLE := $(ROOT)/ace3/model/attention_oracle.py
ATTENTION_GENERATOR := $(ROOT)/ace3/model/generate_attention_vectors.py
ATTENTION_VALIDATOR := $(ROOT)/ace3/model/validate_attention_vectors.py
ATTENTION_CONTRACT := $(ROOT)/ace3/contracts/attention_block.json
ATTENTION_BINDINGS := $(ROOT)/ace3/contracts/attention_vector_bindings.json
ATTENTION_TB := $(ROOT)/ace3/tb/ace3_attention_block_tb.sv
ATTENTION_VERILATOR_TOP := $(ROOT)/ace3/tb/ace3_attention_verilator_top.sv
ATTENTION_CPP_TB := $(ROOT)/ace3/tb/ace3_attention_main.cpp
DECODER_VECTOR_DIR := $(BUILD_DIR)/decoder_layer0_vectors
DECODER_TAMPER_DIR := $(BUILD_DIR)/tamper-decoder-layer0-vectors
DECODER_IVERILOG_DIR := $(BUILD_DIR)/decoder_layer0_iverilog
DECODER_IVERILOG_BIN := $(DECODER_IVERILOG_DIR)/ace3_decoder_layer0_token_engine.vvp
DECODER_IVERILOG_RAW_DIR := $(DECODER_IVERILOG_DIR)/raw
DECODER_IVERILOG_FAIL_RAW_DIR := $(DECODER_IVERILOG_DIR)/raw-injected-failure
DECODER_VERILATOR_DIR := $(BUILD_DIR)/decoder_layer0_verilator
DECODER_VERILATOR_OBJ_DIR := $(DECODER_VERILATOR_DIR)/obj_dir
DECODER_VERILATOR_BIN := $(DECODER_VERILATOR_OBJ_DIR)/Vace3_decoder_layer0_token_engine
DECODER_VERILATOR_RAW_DIR := $(DECODER_VERILATOR_DIR)/raw
DECODER_VERILATOR_FAIL_RAW_DIR := $(DECODER_VERILATOR_DIR)/raw-injected-failure
DECODER_WIDTH_DIR := $(BUILD_DIR)/decoder_width_boundary
DECODER_WIDTH_IVERILOG_BIN := $(DECODER_WIDTH_DIR)/ace3_decoder_width_boundary.vvp
DECODER_WIDTH_VERILATOR_OBJ_DIR := $(DECODER_WIDTH_DIR)/obj_dir
DECODER_WIDTH_VERILATOR_BIN := $(DECODER_WIDTH_VERILATOR_OBJ_DIR)/Vace3_decoder_layer0_token_engine
DECODER_PRELOAD_DIR := $(BUILD_DIR)/decoder_preload_micro_cf01
DECODER_PRELOAD_IVERILOG_BIN := $(DECODER_PRELOAD_DIR)/ace3_decoder_preload.vvp
DECODER_PRELOAD_VERILATOR_OBJ_DIR := $(DECODER_PRELOAD_DIR)/verilator_obj
DECODER_PRELOAD_VERILATOR_BIN := $(DECODER_PRELOAD_VERILATOR_OBJ_DIR)/Vace3_decoder_layer0_token_engine
DECODER_SILU_DIR := $(BUILD_DIR)/decoder_silu_streaming
DECODER_SILU_IVERILOG_BIN := $(DECODER_SILU_DIR)/ace3_decoder_silu_streaming.vvp
DECODER_SILU_VERILATOR_DIR := $(DECODER_SILU_DIR)/verilator
DECODER_QZEROS_DIR := $(BUILD_DIR)/decoder_qzeros_boundary
DECODER_QZEROS_IVERILOG_BIN := $(DECODER_QZEROS_DIR)/ace3_decoder_qzeros_boundary.vvp
DECODER_QZEROS_VERILATOR_DIR := $(DECODER_QZEROS_DIR)/verilator_obj
DECODER_QZEROS_VERILATOR_BIN := $(DECODER_QZEROS_VERILATOR_DIR)/Vace3_decoder_qzeros_address
DECODER_QZEROS_ADDRESS_RTL := $(ROOT)/ace3/rtl/ace3_decoder_qzeros_address.sv
DECODER_QZEROS_TB := $(ROOT)/ace3/tb/ace3_decoder_qzeros_boundary_tb.sv
DECODER_QZEROS_CPP_TB := $(ROOT)/ace3/tb/ace3_decoder_qzeros_boundary_main.cpp
DECODER_ENGINE_RTL := $(ROOT)/ace3/rtl/ace3_decoder_layer0_token_engine.sv
DECODER_RTL := $(FP16_FIXED_RTL) $(PROJECTION_ROUNDER_RTL) $(RTL) \
	$(PROJECTION_RTL) $(FP16_RMS_RTL) $(FP16_RESIDUAL_RTL) \
	$(FP16_SILU_RTL) $(QKV_ROPE_RTL) $(QKV_CACHE_RTL) \
	$(ATTENTION_RTL) $(DECODER_QZEROS_ADDRESS_RTL) $(DECODER_ENGINE_RTL)
DECODER_GENERATOR := $(ROOT)/ace3/model/generate_decoder_layer0_vectors.py
DECODER_VALIDATOR := $(ROOT)/ace3/model/validate_decoder_layer0_vectors.py
DECODER_CONTRACT := $(ROOT)/ace3/contracts/decoder_layer0_token_engine.json
DECODER_BINDINGS := $(ROOT)/ace3/contracts/decoder_layer0_vector_bindings.json
DECODER_TB := $(ROOT)/ace3/tb/ace3_decoder_layer0_token_engine_tb.sv
DECODER_CPP_TB := $(ROOT)/ace3/tb/ace3_decoder_layer0_token_engine_main.cpp
DECODER_LAYER1_VECTOR_DIR := $(BUILD_DIR)/decoder_layer1_vectors
DECODER_LAYER1_TAMPER_DIR := $(BUILD_DIR)/tamper-decoder-layer1-handoff
DECODER_LAYER1_GENERATOR := $(ROOT)/ace3/model/generate_decoder_layer1_vectors.py
DECODER_LAYER1_IVERILOG_DIR := $(BUILD_DIR)/decoder_layer1_iverilog
DECODER_LAYER1_IVERILOG_BIN := $(DECODER_LAYER1_IVERILOG_DIR)/ace3_decoder_layer1_boundary.vvp
DECODER_LAYER1_VERILATOR_DIR := $(BUILD_DIR)/decoder_layer1_verilator
DECODER_LAYER1_VERILATOR_OBJ_DIR := $(DECODER_LAYER1_VERILATOR_DIR)/obj_dir
DECODER_LAYER1_VERILATOR_BIN := $(DECODER_LAYER1_VERILATOR_OBJ_DIR)/Vace3_decoder_layer0_token_engine
DECODER_LAYER01_CASCADE_DIR := $(BUILD_DIR)/decoder_layer01_cascade
DECODER_LAYER2_VECTOR_DIR := $(BUILD_DIR)/decoder_layer2_vectors
DECODER_LAYER2_TAMPER_DIR := $(BUILD_DIR)/tamper-decoder-layer2-handoff
DECODER_LAYER2_GENERATOR := $(ROOT)/ace3/model/generate_decoder_layer2_vectors.py
DECODER_LAYER2_IVERILOG_DIR := $(BUILD_DIR)/decoder_layer2_iverilog
DECODER_LAYER2_IVERILOG_BIN := $(DECODER_LAYER2_IVERILOG_DIR)/ace3_decoder_layer2_boundary.vvp
DECODER_LAYER2_VERILATOR_DIR := $(BUILD_DIR)/decoder_layer2_verilator
DECODER_LAYER2_VERILATOR_OBJ_DIR := $(DECODER_LAYER2_VERILATOR_DIR)/obj_dir
DECODER_LAYER2_VERILATOR_BIN := $(DECODER_LAYER2_VERILATOR_OBJ_DIR)/Vace3_decoder_layer0_token_engine
DECODER_LAYER012_CASCADE_DIR := $(BUILD_DIR)/decoder_layer012_cascade
DECODER_WIDTH_TB := $(ROOT)/ace3/tb/ace3_decoder_width_boundary_tb.sv
DECODER_WIDTH_CPP_TB := $(ROOT)/ace3/tb/ace3_decoder_width_boundary_main.cpp
DECODER_PRELOAD_TB := $(ROOT)/ace3/tb/ace3_decoder_preload_tb.sv
DECODER_PRELOAD_CPP_TB := $(ROOT)/ace3/tb/ace3_decoder_preload_main.cpp
DECODER_SILU_TB := $(ROOT)/ace3/tb/ace3_decoder_silu_streaming_tb.sv
DECODER_SILU_CPP_TB := $(ROOT)/ace3/tb/ace3_decoder_silu_streaming_main.cpp
MODEL24_VECTOR_DIR := $(BUILD_DIR)/model24_execution_vectors
MODEL24_GENERATOR := $(ROOT)/ace3/model/generate_model24_execution_vectors.py
MODEL24_VALIDATOR := $(ROOT)/ace3/model/validate_model24_execution_vectors.py
MODEL24_TEST := $(ROOT)/ace3/model/tests/test_model24_execution.py
MODEL24_LAYER_HANDOFF_TEST := $(ROOT)/ace3/tb/test_model24_layer_indexed_handoff.py
MODEL24_TENSOR_MAP := $(ROOT)/ace3/contracts/model24_tensor_map.json
MODEL24_LAYER_CONTROLLER_DIR := $(BUILD_DIR)/model24_layer_controller
MODEL24_LAYER_CONTROLLER_BIN := $(MODEL24_LAYER_CONTROLLER_DIR)/ace3_model24_layer_controller.vvp
MODEL24_LAYER_CONTROLLER_RTL := $(ROOT)/ace3/rtl/ace3_model24_layer_controller.sv
MODEL24_LAYER_CONTROLLER_TB := $(ROOT)/ace3/tb/ace3_model24_layer_controller_tb.sv
MODEL24_LAYER_CONTROLLER_CPP_TB := $(ROOT)/ace3/tb/ace3_model24_layer_controller_main.cpp
MODEL24_LAYER_CONTROLLER_CONTRACT := $(ROOT)/ace3/contracts/model24_layer_controller.json
MODEL24_LAYER_CONTROLLER_GENERATOR := $(ROOT)/ace3/model/generate_model24_layer_controller_vectors.py
MODEL24_LAYER_CONTROLLER_VALIDATOR := $(ROOT)/ace3/model/validate_model24_layer_controller_vectors.py
MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR := $(MODEL24_LAYER_CONTROLLER_DIR)/iverilog-raw
MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR := $(MODEL24_LAYER_CONTROLLER_DIR)/verilator-raw
MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR := $(MODEL24_LAYER_CONTROLLER_DIR)/failure-raw
MODEL24_LAYER_CONTROLLER_VERILATOR_DIR := $(MODEL24_LAYER_CONTROLLER_DIR)/verilator
MODEL24_LAYER_CONTROLLER_VERILATOR_BIN := $(MODEL24_LAYER_CONTROLLER_VERILATOR_DIR)/Vace3_model24_layer_controller
MODEL24_CONTROLLER_CASCADE_DIR := $(BUILD_DIR)/model24_controller_cascade
MODEL24_CONTROLLER_CASCADE_BINDINGS := $(MODEL24_CONTROLLER_CASCADE_DIR)/bindings.json
MODEL24_CONTROLLER_CASCADE_EXECUTOR := $(ROOT)/ace3/model/controller_model24_cascade.py
MODEL24_CONTROLLER_CASCADE_TEST := $(ROOT)/ace3/model/tests/test_controller_model24_cascade.py
MODEL24_RTL_CASCADE_DIR := $(BUILD_DIR)/model24_rtl_cascade
MODEL24_RTL_CASCADE_EXECUTOR := $(ROOT)/ace3/model/controller_model24_rtl_cascade.py
MODEL24_TOKEN0_DIAGNOSTIC := $(ROOT)/ace3/model/layer3_token0_diagnostic.py
MODEL24_TOKEN0_DIAGNOSTIC_TEST := $(ROOT)/ace3/model/tests/test_layer3_token0_diagnostic.py
MODEL24_RTL_LAYER_INDEX ?= 0
MODEL24_RTL_ACCURATE_SILU ?= 0
MODEL24_RTL_LAYER_DIR := $(MODEL24_RTL_CASCADE_DIR)/compiled/layer$(MODEL24_RTL_LAYER_INDEX)
MODEL24_RTL_LAYER_OBJ_DIR := $(MODEL24_RTL_LAYER_DIR)/obj_dir
MODEL24_RTL_LAYER_BIN := $(MODEL24_RTL_LAYER_OBJ_DIR)/Vace3_decoder_layer0_token_engine
FIRST_VOICE_DIR := $(BUILD_DIR)/model24_first_voice_hybrid
FIRST_VOICE_CONTRACT := $(ROOT)/ace3/contracts/model24_first_voice_hybrid.json
FIRST_VOICE_DRIVER := $(ROOT)/ace3/model/model24_first_voice_hybrid.py
FIRST_VOICE_TEST := $(ROOT)/ace3/model/tests/test_model24_first_voice_hybrid.py
FIRST_VOICE_MAX_NEW_TOKENS ?= 2
FIRST_VOICE_RTL_LAYER_INDEX ?= 0
FIRST_VOICE_RTL_LAYER_DIR := $(FIRST_VOICE_DIR)/compiled/layer$(FIRST_VOICE_RTL_LAYER_INDEX)
FIRST_VOICE_RTL_LAYER_OBJ_DIR := $(FIRST_VOICE_RTL_LAYER_DIR)/obj_dir
FIRST_VOICE_RTL_LAYER_BIN := $(FIRST_VOICE_RTL_LAYER_OBJ_DIR)/Vace3_decoder_layer0_token_engine
FIRST_VOICE_SAVABLE_DIR := $(FIRST_VOICE_DIR)/savable_self_test
FIRST_VOICE_SAVABLE_OBJ_DIR := $(FIRST_VOICE_SAVABLE_DIR)/obj_dir
FIRST_VOICE_SAVABLE_BIN := $(FIRST_VOICE_SAVABLE_OBJ_DIR)/Vace3_decoder_layer0_token_engine
VL15_LAYER0_HANDOFF ?= $(BUILD_DIR)/model24-prep-worktree/build/freshlayer0execute37-vl15/raw-final.rows
OFFICIAL_MODEL24_VECTOR_DIR := $(BUILD_DIR)/official_model24_next_token
OFFICIAL_MODEL24_EXECUTOR := $(ROOT)/ace3/model/official_model24_next_token.py
OFFICIAL_MODEL24_TEST := $(ROOT)/ace3/model/tests/test_official_model24_next_token.py
OFFICIAL_MODEL24_CHECKPOINT ?= $(ROOT)/model24_execution_vectors/model.safetensors
OFFICIAL_MODEL24_TOKENIZER_DIR ?= $(ROOT)/model24_execution_vectors/tokenizer
OFFICIAL_MODEL24_DIALOGUE_VECTOR_DIR := $(BUILD_DIR)/official_model24_dialogue
OFFICIAL_MODEL24_DIALOGUE_EXECUTOR := $(ROOT)/ace3/model/official_model24_dialogue.py
OFFICIAL_MODEL24_DIALOGUE_TEST := $(ROOT)/ace3/model/tests/test_official_model24_dialogue.py
OFFICIAL_MODEL24_SHOWCASE_VECTOR_DIR := $(BUILD_DIR)/official_model24_showcase
OFFICIAL_MODEL24_SHOWCASE_EXECUTOR := $(ROOT)/ace3/model/official_model24_showcase.py
OFFICIAL_MODEL24_SHOWCASE_TEST := $(ROOT)/ace3/model/tests/test_official_model24_showcase.py
OFFICIAL_MODEL24_SYSTEMATIC_VECTOR_DIR := $(BUILD_DIR)/official_model24_systematic_continuations
OFFICIAL_MODEL24_SYSTEMATIC_EXECUTOR := $(ROOT)/ace3/model/official_model24_systematic_continuations.py
OFFICIAL_MODEL24_SYSTEMATIC_TEST := $(ROOT)/ace3/model/tests/test_official_model24_systematic_continuations.py

IVERILOG_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane.vvp
PROTOCOL_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane_protocol.vvp
PROJECTION_IVERILOG_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_projection_engine.vvp
PROJECTION_4864_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_projection_4864_cycle.vvp
VERILATOR_BIN := $(VERILATOR_OBJ_DIR)/Vace3_awq_w4a16_g128_dot_lane
PROJECTION_VERILATOR_BIN := $(PROJECTION_VERILATOR_OBJ_DIR)/Vace3_awq_w4a16_projection_engine

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: \
	test tracked-source-path-regression _validate oracle vectors json-validation tamper-rejection \
	iverilog iverilog-compile iverilog-simulation \
	iverilog-protocol-compile iverilog-protocol-simulation \
	verilator verilator-compile verilator-simulation \
	projection projection-oracle projection-vectors \
	projection-json-validation projection-tamper-rejection \
	projection-geometry projection-iverilog-compile \
	projection-iverilog-simulation projection-4864-compile \
	projection-4864-simulation projection-verilator-compile \
	projection-verilator-simulation \
	fp16-adaptation fp16-oracle fp16-vectors fp16-json-validation \
	fp16-tamper-rejection fp16-geometry fp16-iverilog-compile \
	fp16-iverilog-simulation fp16-verilator-compile \
	fp16-verilator-simulation \
	qkv-rope-cache qkv-oracle qkv-vectors qkv-json-validation \
	qkv-tamper-rejection qkv-geometry qkv-iverilog-compile \
	qkv-iverilog-simulation qkv-verilator-compile \
	qkv-verilator-simulation \
	attention attention-oracle attention-vectors \
	attention-json-validation attention-tamper-rejection \
	attention-iverilog-compile attention-iverilog-simulation \
	attention-verilator-compile attention-verilator-simulation \
	decoder-layer0 decoder-layer0-vectors decoder-layer0-json-validation \
	decoder-layer0-tamper-rejection decoder-layer0-width-boundary \
	decoder-preload-micro decoder-preload-micro-iverilog decoder-preload-micro-verilator \
	decoder-qzeros-boundary decoder-qzeros-boundary-iverilog decoder-qzeros-boundary-verilator \
	decoder-layer0-width-iverilog \
	decoder-layer0-path-alias-regression \
	decoder-layer0-iverilog-path-safety decoder-layer0-verilator-path-safety \
	decoder-layer0-iverilog-compile decoder-layer0-iverilog-fatal-terminal \
	decoder-layer0-iverilog-documented-limit \
	decoder-layer0-iverilog-simulation \
	decoder-layer0-verilator-compile decoder-layer0-verilator-simulation \
	decoder-layer1-vectors decoder-layer1-iverilog-boundary \
	decoder-layer1-verilator-compile decoder-layer01-verilator-cascade \
	decoder-layer2-vectors decoder-layer2-iverilog-boundary \
	decoder-layer2-verilator-compile decoder-layer012-verilator-cascade \
	model24-execution model24-execution-vectors \
	model24-execution-validation model24-execution-tests \
	model24-layer-indexed-handoff model24-publication-tests \
	model24-layer-controller model24-layer-controller-vectors \
	model24-layer-controller-validation model24-layer-controller-iverilog \
	model24-layer-controller-verilator model24-layer-controller-simulations \
	model24-layer-controller-failure-gate model24-controller-cascade \
	model24-controller-cascade-bindings model24-controller-cascade-execution \
	model24-controller-cascade-comparison model24-controller-cascade-validation \
	model24-controller-cascade-tests model24-controller-rtl-cascade \
	model24-rtl-layer-compile model24-token0-diagnostic-tests \
	model24-first-voice-hybrid model24-first-voice-hybrid-tests \
	model24-first-voice-savable-compile model24-first-voice-savable-test \
	model24-first-voice-layer-compile model24-first-voice-compile-all \
	official-model24-next-token official-model24-next-token-vectors \
	official-model24-next-token-validation official-model24-next-token-tests \
	official-model24-dialogue official-model24-dialogue-vectors \
	official-model24-dialogue-validation official-model24-dialogue-tests \
	official-model24-showcase official-model24-showcase-vectors \
	official-model24-showcase-validation official-model24-showcase-tests \
	official-model24-systematic-continuations \
	official-model24-systematic-continuations-vectors \
	official-model24-systematic-continuations-validation \
	official-model24-systematic-continuations-tests clean

tracked-source-path-regression:
	@cd "$(ROOT)" && "$(PYTHON)" ace3/model/tests/test_tracked_source_paths.py

test: tracked-source-path-regression
	@mkdir -p "$(LOG_DIR)"
	@{ \
	  "$(PYTHON)" --version; \
	  "$(IVERILOG)" -V 2>&1 | head -n 1; \
	  "$(VERILATOR)" --version; \
	  "$(MAKE)" --version | head -n 1; \
	} > "$(LOG_DIR)/tool-versions.log"
	@git -C "$(ROOT)" status --short --untracked-files=all > "$(LOG_DIR)/git-status-before.log"
	@$(MAKE) --no-print-directory _validate
	@git -C "$(ROOT)" status --short --untracked-files=all > "$(LOG_DIR)/git-status-after.log"
	@set -eu; log="$(LOG_DIR)/repository-hygiene.log"; \
	{ \
	  printf '%s\n' '$ git diff --check'; \
	  git -C "$(ROOT)" diff --check; \
	  printf '%s\n' '$ cmp build/logs/git-status-before.log build/logs/git-status-after.log'; \
	  cmp "$(LOG_DIR)/git-status-before.log" "$(LOG_DIR)/git-status-after.log"; \
	  printf '%s\n' '$ git check-ignore -q build/vectors/manifest.json'; \
	  git -C "$(ROOT)" check-ignore -q build/vectors/manifest.json; \
	  git -C "$(ROOT)" check-ignore -q build/projection_vectors/manifest.json; \
	  git -C "$(ROOT)" check-ignore -q build/qkv_rope_cache_vectors/manifest.json; \
	  git -C "$(ROOT)" check-ignore -q build/attention_vectors/manifest.json; \
	  git -C "$(ROOT)" check-ignore -q build/decoder_layer0_vectors/boundary_manifest.json; \
	  git -C "$(ROOT)" check-ignore -q build/model24_execution_vectors/manifest.json; \
	  test ! -e "$(ROOT)/ace3/generated"; \
	  printf '%s\n' 'REPOSITORY_HYGIENE_PASS status_unchanged=yes build_ignored=yes legacy_generated_absent=yes diff_check=pass'; \
	} > "$$log" 2>&1 || { status=$$?; cat "$$log"; exit $$status; }; \
	cat "$$log"
	@printf '%s\n' 'STANDALONE_VALIDATION_PASS semantic_checks=fresh primitive=pass projection=pass fp16_adaptation=pass qkv_rope_cache=pass attention=pass serialized_sha256=pass tamper_rejection=pass iverilog=pass protocol_4state=pass verilator=pass geometry_parameters=pass hygiene=pass'

_validate: oracle tamper-rejection iverilog verilator projection fp16-adaptation qkv-rope-cache attention

oracle:
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/python-oracle.log"; \
	printf '%s\n' '$ python3 ace3/model/awq_bit_oracle.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" ace3/model/awq_bit_oracle.py >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

vectors:
	@rm -rf "$(VECTOR_DIR)"
	@mkdir -p "$(VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" ace3/model/generate_vectors.py \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

json-validation: vectors
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/json-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_vectors.py --generated-dir build/vectors --frozen-manifest ace3/contracts/awq_w4a16_g128_vectors_manifest.json --contract ace3/contracts/awq_w4a16_g128_dot_lane.json --evidence-bindings ace3/contracts/awq_w4a16_g128_evidence_bindings.json --standalone-bindings ace3/contracts/awq_w4a16_g128_standalone_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" ace3/model/validate_vectors.py \
	    --generated-dir "$(VECTOR_DIR)" \
	    --frozen-manifest "$(FROZEN_MANIFEST)" \
	    --contract "$(CONTRACT)" \
	    --evidence-bindings "$(EVIDENCE_BINDINGS)" \
	    --standalone-bindings "$(STANDALONE_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

tamper-rejection: json-validation
	@rm -rf "$(TAMPER_DIR)"
	@cp -a "$(VECTOR_DIR)" "$(TAMPER_DIR)"
	@printf '0' >> "$(TAMPER_DIR)/meta.hex"
	@set -eu; attempt="$(LOG_DIR)/tamper-rejection-attempt.log"; \
	log="$(LOG_DIR)/tamper-rejection.log"; \
	printf '%s\n' '$ printf 0 >> build/tamper-vectors/meta.hex' > "$$attempt"; \
	printf '%s\n' '$ python3 ace3/model/validate_vectors.py --generated-dir build/tamper-vectors ... (expected failure)' >> "$$attempt"; \
	if cd "$(ROOT)" && "$(PYTHON)" ace3/model/validate_vectors.py \
	    --generated-dir "$(TAMPER_DIR)" \
	    --frozen-manifest "$(FROZEN_MANIFEST)" \
	    --contract "$(CONTRACT)" \
	    --evidence-bindings "$(EVIDENCE_BINDINGS)" \
	    --standalone-bindings "$(STANDALONE_BINDINGS)" >> "$$attempt" 2>&1; then \
	  printf '%s\n' 'tampered artifact was incorrectly accepted' >> "$$attempt"; \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'meta.hex SHA256 mismatch' "$$attempt"; then \
	  printf '%s\n' 'validator failed for an unexpected reason' >> "$$attempt"; \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'TAMPER_REJECTION_PASS artifact=meta.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

iverilog: iverilog-simulation iverilog-protocol-simulation

iverilog-compile: json-validation
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@rm -f "$(IVERILOG_BIN)"
	@set -eu; log="$(LOG_DIR)/iverilog-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -I build/vectors -s ace3_awq_w4a16_g128_dot_lane_tb -o build/iverilog/ace3_awq_w4a16_g128_dot_lane.vvp ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv ace3/tb/ace3_awq_w4a16_g128_dot_lane_tb.sv' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -I "$(VECTOR_DIR)" \
	    -s ace3_awq_w4a16_g128_dot_lane_tb -o "$(IVERILOG_BIN)" \
	    "$(RTL)" "$(SV_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

iverilog-simulation: iverilog-compile
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/iverilog-simulation.log"; \
	printf '%s\n' '$ vvp build/iverilog/ace3_awq_w4a16_g128_dot_lane.vvp +VECTOR_DIR=build/vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(IVERILOG_BIN)" \
	    +VECTOR_DIR="$(VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

iverilog-protocol-compile: json-validation
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@rm -f "$(PROTOCOL_BIN)"
	@set -eu; log="$(LOG_DIR)/iverilog-protocol-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -s ace3_awq_w4a16_g128_dot_lane_protocol_tb -o build/iverilog/ace3_awq_w4a16_g128_dot_lane_protocol.vvp ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv ace3/tb/ace3_awq_w4a16_g128_dot_lane_protocol_tb.sv' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall \
	    -s ace3_awq_w4a16_g128_dot_lane_protocol_tb -o "$(PROTOCOL_BIN)" \
	    "$(RTL)" "$(PROTOCOL_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

iverilog-protocol-simulation: iverilog-protocol-compile
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/iverilog-protocol-simulation.log"; \
	printf '%s\n' '$ vvp build/iverilog/ace3_awq_w4a16_g128_dot_lane_protocol.vvp' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(PROTOCOL_BIN)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

verilator: verilator-simulation

verilator-compile: json-validation
	@mkdir -p "$(VERILATOR_DIR)" "$(LOG_DIR)"
	@rm -rf "$(VERILATOR_OBJ_DIR)"
	@set -eu; log="$(LOG_DIR)/verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build --Wall --top-module ace3_awq_w4a16_g128_dot_lane --Mdir build/verilator/obj_dir ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv ace3/tb/ace3_awq_w4a16_g128_dot_lane_main.cpp' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall \
	    --top-module ace3_awq_w4a16_g128_dot_lane \
	    --Mdir "$(VERILATOR_OBJ_DIR)" "$(RTL)" "$(CPP_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(VERILATOR_BIN)"; cat "$$log"

verilator-simulation: verilator-compile
	@mkdir -p "$(VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/verilator-simulation.log"; \
	printf '%s\n' '$ build/verilator/obj_dir/Vace3_awq_w4a16_g128_dot_lane --cases build/vectors/cases.txt --pairs build/vectors/pairs.hex' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR_BIN)" \
	    --cases "$(VECTOR_DIR)/cases.txt" \
	    --pairs "$(VECTOR_DIR)/pairs.hex" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection: projection-oracle projection-tamper-rejection \
	projection-geometry projection-iverilog-simulation \
	projection-4864-simulation projection-verilator-simulation

projection-oracle:
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-oracle.log"; \
	printf '%s\n' '$ python3 ace3/model/projection_oracle.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" ace3/model/projection_oracle.py \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-vectors:
	@rm -rf "$(PROJECTION_VECTOR_DIR)"
	@mkdir -p "$(PROJECTION_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_projection_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/projection_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" \
	    ace3/model/generate_projection_vectors.py \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(PROJECTION_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-json-validation: projection-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-json-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_projection_vectors.py --generated-dir build/projection_vectors --contract ace3/contracts/awq_w4a16_projection_engine.json --bindings ace3/contracts/awq_w4a16_projection_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" \
	    ace3/model/validate_projection_vectors.py \
	    --generated-dir "$(PROJECTION_VECTOR_DIR)" \
	    --contract "$(PROJECTION_CONTRACT)" \
	    --bindings "$(PROJECTION_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-tamper-rejection: projection-json-validation
	@rm -rf "$(PROJECTION_TAMPER_DIR)"
	@cp -a "$(PROJECTION_VECTOR_DIR)" "$(PROJECTION_TAMPER_DIR)"
	@printf '0' >> "$(PROJECTION_TAMPER_DIR)/pairs.hex"
	@set -eu; attempt="$(LOG_DIR)/projection-tamper-attempt.log"; \
	log="$(LOG_DIR)/projection-tamper-rejection.log"; \
	printf '%s\n' '$ printf 0 >> build/tamper-projection-vectors/pairs.hex' > "$$attempt"; \
	if cd "$(ROOT)" && "$(PYTHON)" \
	    ace3/model/validate_projection_vectors.py \
	    --generated-dir "$(PROJECTION_TAMPER_DIR)" \
	    --contract "$(PROJECTION_CONTRACT)" \
	    --bindings "$(PROJECTION_BINDINGS)" >> "$$attempt" 2>&1; then \
	  printf '%s\n' 'tampered projection artifact was accepted' >> "$$attempt"; \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'pairs.hex SHA256 mismatch' "$$attempt"; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'PROJECTION_TAMPER_REJECTION_PASS artifact=pairs.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

projection-geometry: projection-json-validation
	@rm -rf "$(PROJECTION_GEOMETRY_DIR)"
	@mkdir -p "$(PROJECTION_GEOMETRY_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-geometry.log"; \
	printf '%s\n' '$ elaborate/lint q_o=896x896 k_v=896x128 gate_up=896x4864 down=4864x896' > "$$log"; \
	for spec in '896 896 q_o' '896 128 k_v' \
	            '896 4864 gate_up' '4864 896 down'; do \
	  set -- $$spec; \
	  "$(IVERILOG)" -g2012 -Wall \
	    -s ace3_awq_w4a16_projection_engine \
	    -Pace3_awq_w4a16_projection_engine.IN_FEATURES=$$1 \
	    -Pace3_awq_w4a16_projection_engine.OUT_FEATURES=$$2 \
	    -o "$(PROJECTION_GEOMETRY_DIR)/$$3.vvp" \
	    "$(FP16_FIXED_RTL)" "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    >> "$$log" 2>&1 || { cat "$$log"; exit 1; }; \
	  "$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_awq_w4a16_projection_engine \
	    -GIN_FEATURES=$$1 -GOUT_FEATURES=$$2 \
	    "$(FP16_FIXED_RTL)" "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    >> "$$log" 2>&1 || { cat "$$log"; exit 1; }; \
	done; \
	printf '%s\n' 'PROJECTION_GEOMETRY_PASS simulators=iverilog,verilator geometries=4 q_o=896x896 k_v=896x128 gate_up=896x4864 down=4864x896' >> "$$log"; \
	cat "$$log"

projection-iverilog-compile: projection-json-validation
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@rm -f "$(PROJECTION_IVERILOG_BIN)"
	@set -eu; log="$(LOG_DIR)/projection-iverilog-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -I build/projection_vectors -s ace3_awq_w4a16_projection_engine_tb -o build/iverilog/ace3_awq_w4a16_projection_engine.vvp ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -I "$(PROJECTION_VECTOR_DIR)" \
	    -s ace3_awq_w4a16_projection_engine_tb \
	    -o "$(PROJECTION_IVERILOG_BIN)" \
	    "$(FP16_FIXED_RTL)" "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    "$(PROJECTION_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-iverilog-simulation: projection-iverilog-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-iverilog-simulation.log"; \
	printf '%s\n' '$ vvp build/iverilog/ace3_awq_w4a16_projection_engine.vvp +VECTOR_DIR=build/projection_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(PROJECTION_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(PROJECTION_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-4864-compile: projection-json-validation
	@mkdir -p "$(IVERILOG_DIR)" "$(LOG_DIR)"
	@rm -f "$(PROJECTION_4864_BIN)"
	@set -eu; log="$(LOG_DIR)/projection-4864-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -s ace3_awq_w4a16_projection_4864_cycle_tb -o build/iverilog/ace3_awq_w4a16_projection_4864_cycle.vvp ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall \
	    -s ace3_awq_w4a16_projection_4864_cycle_tb \
	    -o "$(PROJECTION_4864_BIN)" \
	    "$(FP16_FIXED_RTL)" "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    "$(PROJECTION_4864_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-4864-simulation: projection-4864-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-4864-simulation.log"; \
	printf '%s\n' '$ vvp build/iverilog/ace3_awq_w4a16_projection_4864_cycle.vvp' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(PROJECTION_4864_BIN)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

projection-verilator-compile: projection-json-validation
	@rm -rf "$(PROJECTION_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(PROJECTION_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build --Wall --top-module ace3_awq_w4a16_projection_engine --Mdir build/projection_verilator/obj_dir ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall \
	    --top-module ace3_awq_w4a16_projection_engine \
	    --Mdir "$(PROJECTION_VERILATOR_OBJ_DIR)" \
	    "$(FP16_FIXED_RTL)" "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    "$(PROJECTION_CPP_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(PROJECTION_VERILATOR_BIN)"; cat "$$log"

projection-verilator-simulation: projection-verilator-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/projection-verilator-simulation.log"; \
	printf '%s\n' '$ build/projection_verilator/obj_dir/Vace3_awq_w4a16_projection_engine --transactions build/projection_vectors/transactions.hex --expected build/projection_vectors/expected.hex --meta build/projection_vectors/meta.hex --pairs build/projection_vectors/pairs.hex' > "$$log"; \
	if cd "$(ROOT)" && "$(PROJECTION_VERILATOR_BIN)" \
	    --transactions "$(PROJECTION_VECTOR_DIR)/transactions.hex" \
	    --expected "$(PROJECTION_VECTOR_DIR)/expected.hex" \
	    --meta "$(PROJECTION_VECTOR_DIR)/meta.hex" \
	    --pairs "$(PROJECTION_VECTOR_DIR)/pairs.hex" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-adaptation: fp16-oracle fp16-tamper-rejection fp16-geometry \
	fp16-iverilog-simulation fp16-verilator-simulation

fp16-oracle:
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-oracle.log"; \
	printf '%s\n' '$ python3 ace3/model/fp16_adaptation_oracle.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(FP16_ORACLE)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-vectors:
	@rm -rf "$(FP16_VECTOR_DIR)"
	@mkdir -p "$(FP16_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_fp16_adaptation_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/fp16_adaptation_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(FP16_GENERATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(FP16_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-json-validation: fp16-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-json-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_fp16_adaptation_vectors.py --generated-dir build/fp16_adaptation_vectors --contract ace3/contracts/fp16_adaptation_operators.json --bindings ace3/contracts/fp16_adaptation_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(FP16_VALIDATOR)" \
	    --generated-dir "$(FP16_VECTOR_DIR)" \
	    --contract "$(FP16_CONTRACT)" \
	    --bindings "$(FP16_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-tamper-rejection: fp16-json-validation
	@rm -rf "$(FP16_TAMPER_DIR)"
	@cp -a "$(FP16_VECTOR_DIR)" "$(FP16_TAMPER_DIR)"
	@printf '0' >> "$(FP16_TAMPER_DIR)/residual_cases.hex"
	@set -eu; attempt="$(LOG_DIR)/fp16-adaptation-tamper-attempt.log"; \
	log="$(LOG_DIR)/fp16-adaptation-tamper-rejection.log"; \
	printf '%s\n' '$ printf 0 >> build/tamper-fp16-adaptation-vectors/residual_cases.hex' > "$$attempt"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(FP16_VALIDATOR)" \
	    --generated-dir "$(FP16_TAMPER_DIR)" \
	    --contract "$(FP16_CONTRACT)" \
	    --bindings "$(FP16_BINDINGS)" >> "$$attempt" 2>&1; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'residual_cases.hex SHA256 mismatch' "$$attempt"; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'FP16_ADAPTATION_TAMPER_REJECTION_PASS artifact=residual_cases.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

fp16-geometry: fp16-json-validation
	@rm -rf "$(FP16_GEOMETRY_DIR)"
	@mkdir -p "$(FP16_GEOMETRY_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-geometry.log"; \
	printf '%s\n' '$ elaborate/lint residual=896 rmsnorm=896 silu_gate=4864' > "$$log"; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_fp16_residual_add_core \
	    -Pace3_fp16_residual_add_core.VECTOR_SIZE=896 \
	    -o "$(FP16_GEOMETRY_DIR)/residual.vvp" \
	    "$(FP16_FIXED_RTL)" "$(FP16_RESIDUAL_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_fp16_rmsnorm_core \
	    -Pace3_fp16_rmsnorm_core.HIDDEN_SIZE=896 \
	    -o "$(FP16_GEOMETRY_DIR)/rmsnorm.vvp" \
	    "$(FP16_FIXED_RTL)" "$(FP16_RMS_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_fp16_silu_gate_core \
	    -Pace3_fp16_silu_gate_core.INTERMEDIATE_SIZE=4864 \
	    -o "$(FP16_GEOMETRY_DIR)/silu.vvp" \
	    "$(FP16_FIXED_RTL)" "$(FP16_SILU_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_fp16_residual_add_core -GVECTOR_SIZE=896 \
	    "$(FP16_FIXED_RTL)" "$(FP16_RESIDUAL_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_fp16_rmsnorm_core -GHIDDEN_SIZE=896 \
	    "$(FP16_FIXED_RTL)" "$(FP16_RMS_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_fp16_silu_gate_core -GINTERMEDIATE_SIZE=4864 \
	    "$(FP16_FIXED_RTL)" "$(FP16_SILU_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	printf '%s\n' 'FP16_ADAPTATION_GEOMETRY_PASS simulators=iverilog,verilator residual=896 rmsnorm=896 silu_gate=4864' >> "$$log"; \
	cat "$$log"

fp16-iverilog-compile: fp16-json-validation
	@rm -rf "$(FP16_IVERILOG_DIR)"
	@mkdir -p "$(FP16_IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-iverilog-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -I build/fp16_adaptation_vectors -s ace3_fp16_adaptation_tb -o build/fp16_iverilog/ace3_fp16_adaptation.vvp ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -I "$(FP16_VECTOR_DIR)" \
	    -s ace3_fp16_adaptation_tb -o "$(FP16_IVERILOG_BIN)" \
	    $(FP16_RTL) "$(FP16_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-iverilog-simulation: fp16-iverilog-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-iverilog-simulation.log"; \
	printf '%s\n' '$ vvp build/fp16_iverilog/ace3_fp16_adaptation.vvp +VECTOR_DIR=build/fp16_adaptation_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(FP16_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(FP16_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

fp16-verilator-compile: fp16-json-validation
	@rm -rf "$(FP16_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(FP16_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build --Wall --top-module ace3_fp16_adaptation_verilator_top --Mdir build/fp16_verilator/obj_dir ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall \
	    --top-module ace3_fp16_adaptation_verilator_top \
	    --Mdir "$(FP16_VERILATOR_OBJ_DIR)" \
	    $(FP16_RTL) "$(FP16_VERILATOR_TOP)" "$(FP16_CPP_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(FP16_VERILATOR_BIN)"; cat "$$log"

fp16-verilator-simulation: fp16-verilator-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/fp16-adaptation-verilator-simulation.log"; \
	printf '%s\n' '$ build/fp16_verilator/obj_dir/Vace3_fp16_adaptation_verilator_top --vector-dir build/fp16_adaptation_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(FP16_VERILATOR_BIN)" \
	    --vector-dir "$(FP16_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-rope-cache: qkv-oracle qkv-tamper-rejection qkv-geometry \
	qkv-iverilog-simulation qkv-verilator-simulation

qkv-oracle:
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-oracle.log"; \
	printf '%s\n' '$ python3 ace3/model/qwen2_rope_oracle.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(QKV_ORACLE)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-vectors:
	@rm -rf "$(QKV_VECTOR_DIR)"
	@mkdir -p "$(QKV_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_qkv_rope_cache_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/qkv_rope_cache_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(QKV_GENERATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(QKV_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-json-validation: qkv-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-json-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_qkv_rope_cache_vectors.py --generated-dir build/qkv_rope_cache_vectors --contract ace3/contracts/qkv_rope_kv_cache.json --bindings ace3/contracts/qkv_rope_kv_cache_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(QKV_VALIDATOR)" \
	    --generated-dir "$(QKV_VECTOR_DIR)" \
	    --contract "$(QKV_CONTRACT)" \
	    --bindings "$(QKV_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-tamper-rejection: qkv-json-validation
	@rm -rf "$(QKV_TAMPER_DIR)"
	@cp -a "$(QKV_VECTOR_DIR)" "$(QKV_TAMPER_DIR)"
	@printf '0' >> "$(QKV_TAMPER_DIR)/rope_cases.hex"
	@set -eu; attempt="$(LOG_DIR)/qkv-rope-cache-tamper-attempt.log"; \
	log="$(LOG_DIR)/qkv-rope-cache-tamper-rejection.log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(QKV_VALIDATOR)" \
	    --generated-dir "$(QKV_TAMPER_DIR)" \
	    --contract "$(QKV_CONTRACT)" \
	    --bindings "$(QKV_BINDINGS)" > "$$attempt" 2>&1; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'rope_cases.hex SHA256 mismatch' "$$attempt"; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'QKV_ROPE_CACHE_TAMPER_REJECTION_PASS artifact=rope_cases.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

qkv-geometry: qkv-json-validation
	@rm -rf "$(QKV_IVERILOG_DIR)"
	@mkdir -p "$(QKV_IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-projection-geometry.log"; \
	: > "$$log"; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_qkv_projection_geometry_tb \
	    -o "$(QKV_GEOMETRY_BIN)" \
	    "$(FP16_FIXED_RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(RTL)" "$(PROJECTION_RTL)" \
	    "$(QKV_CLUSTER_RTL)" "$(QKV_GEOMETRY_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VVP)" "$(QKV_GEOMETRY_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_qkv_projection_cluster \
	    "$(FP16_FIXED_RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(RTL)" "$(PROJECTION_RTL)" \
	    "$(QKV_CLUSTER_RTL)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	printf '%s\n' 'QKV_GEOMETRY_PASS simulators=iverilog,verilator q=896x896 k=896x128 v=896x128 query_heads=14 kv_heads=2 head_dim=64' >> "$$log"; \
	cat "$$log"

qkv-iverilog-compile: qkv-json-validation
	@mkdir -p "$(QKV_IVERILOG_DIR)" "$(LOG_DIR)"
	@rm -f "$(QKV_IVERILOG_BIN)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-iverilog-compile.log"; \
	: > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -I "$(QKV_VECTOR_DIR)" \
	    -s ace3_qkv_rope_cache_tb -o "$(QKV_IVERILOG_BIN)" \
	    "$(FP16_FIXED_RTL)" "$(PROJECTION_ROUNDER_RTL)" \
	    "$(QKV_ROPE_RTL)" "$(QKV_CACHE_RTL)" "$(QKV_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-iverilog-simulation: qkv-iverilog-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-iverilog-simulation.log"; \
	: > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(QKV_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(QKV_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

qkv-verilator-compile: qkv-json-validation
	@rm -rf "$(QKV_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(QKV_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-verilator-compile.log"; \
	: > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall \
	    --top-module ace3_qkv_rope_cache_verilator_top \
	    --Mdir "$(QKV_VERILATOR_OBJ_DIR)" \
	    "$(FP16_FIXED_RTL)" "$(PROJECTION_ROUNDER_RTL)" \
	    "$(QKV_ROPE_RTL)" "$(QKV_CACHE_RTL)" "$(QKV_VERILATOR_TOP)" \
	    "$(QKV_CPP_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(QKV_VERILATOR_BIN)"; cat "$$log"

qkv-verilator-simulation: qkv-verilator-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/qkv-rope-cache-verilator-simulation.log"; \
	: > "$$log"; \
	if cd "$(ROOT)" && "$(QKV_VERILATOR_BIN)" \
	    --vector-dir "$(QKV_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention: attention-oracle attention-tamper-rejection \
	attention-iverilog-simulation attention-verilator-simulation

attention-oracle:
	@mkdir -p "$(BUILD_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-oracle.log"; \
	printf '%s\n' '$ python3 ace3/model/attention_oracle.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(ATTENTION_ORACLE)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention-vectors:
	@rm -rf "$(ATTENTION_VECTOR_DIR)"
	@mkdir -p "$(ATTENTION_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_attention_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/attention_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(ATTENTION_GENERATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(ATTENTION_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention-json-validation: attention-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-vector-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_attention_vectors.py --generated-dir build/attention_vectors --contract ace3/contracts/attention_block.json --bindings ace3/contracts/attention_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(ATTENTION_VALIDATOR)" \
	    --generated-dir "$(ATTENTION_VECTOR_DIR)" \
	    --contract "$(ATTENTION_CONTRACT)" \
	    --bindings "$(ATTENTION_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention-tamper-rejection: attention-json-validation
	@rm -rf "$(ATTENTION_TAMPER_DIR)"
	@cp -a "$(ATTENTION_VECTOR_DIR)" "$(ATTENTION_TAMPER_DIR)"
	@printf '0' >> "$(ATTENTION_TAMPER_DIR)/attention_score_terms.hex"
	@set -eu; attempt="$(LOG_DIR)/attention-tamper-attempt.log"; \
	log="$(LOG_DIR)/attention-tamper-rejection.log"; \
	printf '%s\n' '$ printf 0 >> build/tamper-attention-vectors/attention_score_terms.hex' > "$$attempt"; \
	printf '%s\n' '$ python3 ace3/model/validate_attention_vectors.py ... (expected failure)' >> "$$attempt"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(ATTENTION_VALIDATOR)" \
	    --generated-dir "$(ATTENTION_TAMPER_DIR)" \
	    --contract "$(ATTENTION_CONTRACT)" \
	    --bindings "$(ATTENTION_BINDINGS)" >> "$$attempt" 2>&1; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'attention_score_terms.hex SHA256 mismatch' "$$attempt"; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'ACE3_ATTENTION_TAMPER_REJECTION_PASS artifact=attention_score_terms.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

attention-iverilog-compile: attention-json-validation
	@rm -rf "$(ATTENTION_IVERILOG_DIR)"
	@mkdir -p "$(ATTENTION_IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-iverilog-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -I build/attention_vectors -s ace3_attention_block_tb -o build/attention_iverilog/ace3_attention_block.vvp ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -I "$(ATTENTION_VECTOR_DIR)" \
	    -s ace3_attention_block_tb -o "$(ATTENTION_IVERILOG_BIN)" \
	    "$(FP16_FIXED_RTL)" $(ATTENTION_RTL) "$(ATTENTION_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention-iverilog-simulation: attention-iverilog-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-iverilog-simulation.log"; \
	printf '%s\n' '$ vvp build/attention_iverilog/ace3_attention_block.vvp +VECTOR_DIR=build/attention_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(ATTENTION_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(ATTENTION_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

attention-verilator-compile: attention-json-validation
	@rm -rf "$(ATTENTION_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(ATTENTION_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build --Wall -Wno-fatal --top-module ace3_attention_verilator_top --Mdir build/attention_verilator/obj_dir ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build \
	    --Wall -Wno-fatal --top-module ace3_attention_verilator_top \
	    --Mdir "$(ATTENTION_VERILATOR_OBJ_DIR)" \
	    "$(FP16_FIXED_RTL)" $(ATTENTION_RTL) \
	    "$(ATTENTION_VERILATOR_TOP)" "$(ATTENTION_CPP_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(ATTENTION_VERILATOR_BIN)"; cat "$$log"

attention-verilator-simulation: attention-verilator-compile
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/attention-verilator-simulation.log"; \
	printf '%s\n' '$ build/attention_verilator/obj_dir/Vace3_attention_verilator_top --vector-dir=build/attention_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(ATTENTION_VERILATOR_BIN)" \
	    --vector-dir="$(ATTENTION_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

decoder-layer0: decoder-layer0-tamper-rejection decoder-layer0-width-boundary \
	decoder-qzeros-boundary-iverilog decoder-layer0-iverilog-fatal-terminal \
	decoder-layer0-iverilog-documented-limit \
	decoder-layer0-path-alias-regression \
	decoder-layer0-verilator-simulation
	@printf '%s\n' 'DECODER_LAYER0_PASS scope=two_authenticated_tokens trace=46676 final=1792 cache_reuse=pass reset=pass clear=pass fault_injection=pass natural_terminal_gate=pass independent_oracle_compare=pass injected_failure_gate=pass raw_vector_alias_rejection=pass iverilog_abnormal_terminal=pass iverilog_focused=pass iverilog_full=runtime_limited verilator=pass'

decoder-silu-streaming: decoder-silu-streaming-iverilog decoder-silu-streaming-verilator
	@printf '%s\n' 'DECODER_SILU_STREAMING_DUAL_PASS simulators=iverilog,verilator projection_kind=5 phase=32 inputs=4864 outputs=4864 successor_phase=4 full_layer=not_run'

decoder-silu-streaming-iverilog:
	@rm -rf "$(DECODER_SILU_DIR)/iverilog" "$(DECODER_SILU_IVERILOG_BIN)"
	@mkdir -p "$(DECODER_SILU_DIR)/iverilog"
	@"$(IVERILOG)" -g2012 -Wall -s ace3_decoder_silu_streaming_iverilog_tb \
	    -o "$(DECODER_SILU_IVERILOG_BIN)" $(DECODER_RTL) "$(DECODER_SILU_TB)" \
	    >"$(DECODER_SILU_DIR)/iverilog/compile.log" 2>&1
	@"$(VVP)" "$(DECODER_SILU_IVERILOG_BIN)" \
	    >"$(DECODER_SILU_DIR)/iverilog/run.log" 2>&1
	@cat "$(DECODER_SILU_DIR)/iverilog/run.log"

decoder-silu-streaming-verilator:
	@rm -rf "$(DECODER_SILU_VERILATOR_DIR)"
	@mkdir -p "$(DECODER_SILU_VERILATOR_DIR)"
	@cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall -Wno-fatal \
	    --top-module ace3_decoder_silu_streaming_tb \
	    --Mdir "$(DECODER_SILU_VERILATOR_DIR)" $(DECODER_RTL) \
	    "$(DECODER_SILU_TB)" "$(DECODER_SILU_CPP_TB)" \
	    >"$(DECODER_SILU_DIR)/verilator-compile.log" 2>&1
	@"$(DECODER_SILU_VERILATOR_DIR)/Vace3_decoder_silu_streaming_tb" \
	    >"$(DECODER_SILU_DIR)/verilator-run.log" 2>&1
	@cat "$(DECODER_SILU_DIR)/verilator-run.log"

decoder-layer0-vectors:
	@rm -rf "$(DECODER_VECTOR_DIR)"
	@mkdir -p "$(DECODER_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_decoder_layer0_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --output-dir build/decoder_layer0_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_GENERATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --output-dir "$(DECODER_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

decoder-layer0-json-validation: decoder-layer0-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-vector-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_decoder_layer0_vectors.py --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" --generated-dir build/decoder_layer0_vectors --bindings ace3/contracts/decoder_layer0_vector_bindings.json' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_VALIDATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --generated-dir "$(DECODER_VECTOR_DIR)" \
	    --bindings "$(DECODER_BINDINGS)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

decoder-layer0-tamper-rejection: decoder-layer0-json-validation
	@rm -rf "$(DECODER_TAMPER_DIR)"
	@cp -a "$(DECODER_VECTOR_DIR)" "$(DECODER_TAMPER_DIR)"
	@printf '0' >> "$(DECODER_TAMPER_DIR)/trace.hex"
	@set -eu; attempt="$(LOG_DIR)/decoder-layer0-tamper-attempt.log"; \
	log="$(LOG_DIR)/decoder-layer0-tamper-rejection.log"; \
	printf '%s\n' '$ printf 0 >> build/tamper-decoder-layer0-vectors/trace.hex' > "$$attempt"; \
	printf '%s\n' '$ python3 ace3/model/validate_decoder_layer0_vectors.py ... (expected failure)' >> "$$attempt"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_VALIDATOR)" \
	    --official-tensor-dir "$(OFFICIAL_TENSOR_DIR)" \
	    --generated-dir "$(DECODER_TAMPER_DIR)" \
	    --bindings "$(DECODER_BINDINGS)" >> "$$attempt" 2>&1; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	if ! grep -Fq 'trace.hex SHA256' "$$attempt"; then \
	  cat "$$attempt"; exit 1; \
	fi; \
	printf '%s\n' 'DECODER_LAYER0_TAMPER_REJECTION_PASS artifact=trace.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched' > "$$log"; \
	cat "$$log"

decoder-layer0-width-iverilog:
	@rm -rf "$(DECODER_WIDTH_DIR)"
	@mkdir -p "$(DECODER_WIDTH_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-width-iverilog.log"; \
	printf '%s\n' '$ iverilog/vvp decoder width, reset, clear, and fault boundary' > "$$log"; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_decoder_width_boundary_tb \
	    -o "$(DECODER_WIDTH_IVERILOG_BIN)" $(DECODER_RTL) \
	    "$(DECODER_WIDTH_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VVP)" "$(DECODER_WIDTH_IVERILOG_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	printf '%s\n' 'DECODER_WIDTH_IVERILOG_PASS q_index=895 kv_index=127 intermediate_index=4863 position=127 half_split_rope=pass reset=pass clear=pass fault_injection=pass' >> "$$log"; \
	cat "$$log"

decoder-layer0-width-boundary: decoder-layer0-width-iverilog
	@rm -rf "$(DECODER_WIDTH_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(DECODER_WIDTH_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-width-verilator.log"; \
	printf '%s\n' '$ verilator decoder width, reset, clear, and fault boundary' > "$$log"; \
	cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall -Wno-fatal \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(DECODER_WIDTH_VERILATOR_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_WIDTH_CPP_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	test -x "$(DECODER_WIDTH_VERILATOR_BIN)"; \
	"$(DECODER_WIDTH_VERILATOR_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	cat "$$log"
	@printf '%s\n' 'DECODER_WIDTH_BOUNDARY_PASS simulators=iverilog,verilator q_index=895 kv_index=127 intermediate_index=4863 position=127 half_split_rope=pass reset=pass clear=pass fault_injection=pass'

decoder-preload-micro: decoder-preload-micro-iverilog decoder-preload-micro-verilator
	@printf '%s\n' 'DECODER_PRELOAD_MICRO_PASS simulators=iverilog,verilator epoch_handshakes=2688 total_handshakes=5376 epochs=2 reset=pass sequence=1,2,0 flags=pass ready_known=pass final_index=895 start_transition=0_to_1 clear_reload=pass bounded_timeout=pass'

decoder-preload-micro-iverilog:
	@rm -rf "$(DECODER_PRELOAD_DIR)/iverilog" "$(DECODER_PRELOAD_IVERILOG_BIN)"
	@mkdir -p "$(DECODER_PRELOAD_DIR)/iverilog"
	@set -eu; log="$(DECODER_PRELOAD_DIR)/iverilog/run.log"; \
	timeout_log="$(DECODER_PRELOAD_DIR)/iverilog/timeout.log"; \
	printf '%s\n' '$ iverilog/vvp decoder preload-only lifecycle' > "$$log"; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_decoder_preload_tb \
	    -o "$(DECODER_PRELOAD_IVERILOG_BIN)" $(DECODER_RTL) "$(DECODER_PRELOAD_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VVP)" "$(DECODER_PRELOAD_IVERILOG_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	printf '%s\n' '$ vvp ace3_decoder_preload.vvp +EXPECT_TIMEOUT (expected failure)' > "$$timeout_log"; \
	if "$(VVP)" "$(DECODER_PRELOAD_IVERILOG_BIN)" +EXPECT_TIMEOUT >> "$$timeout_log" 2>&1; then \
	  cat "$$timeout_log"; exit 1; \
	fi; \
	grep -F 'DECODER_PRELOAD_TIMEOUT kind=1 index=1' "$$timeout_log" >/dev/null \
	    || { cat "$$timeout_log"; exit 1; }; \
	grep -F 'phase=0 accepts=0,0,0' "$$timeout_log" >/dev/null \
	    || { cat "$$timeout_log"; exit 1; }; \
	cat "$$log"; cat "$$timeout_log"

decoder-preload-micro-verilator:
	@rm -rf "$(DECODER_PRELOAD_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(DECODER_PRELOAD_VERILATOR_OBJ_DIR)"
	@set -eu; log="$(DECODER_PRELOAD_DIR)/verilator.log"; \
	timeout_log="$(DECODER_PRELOAD_DIR)/verilator-timeout.log"; \
	printf '%s\n' '$ verilator decoder preload-only lifecycle' > "$$log"; \
	cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall -Wno-fatal \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(DECODER_PRELOAD_VERILATOR_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_PRELOAD_CPP_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	test -x "$(DECODER_PRELOAD_VERILATOR_BIN)"; \
	"$(DECODER_PRELOAD_VERILATOR_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	printf '%s\n' '$ Vace3_decoder_layer0_token_engine --expect-timeout (expected failure)' > "$$timeout_log"; \
	if "$(DECODER_PRELOAD_VERILATOR_BIN)" --expect-timeout >> "$$timeout_log" 2>&1; then \
	  cat "$$timeout_log"; exit 1; \
	fi; \
	grep -F 'DECODER_PRELOAD_TIMEOUT kind=1 index=1' "$$timeout_log" >/dev/null \
	    || { cat "$$timeout_log"; exit 1; }; \
	grep -F 'phase=0 accepts=0,0,0' "$$timeout_log" >/dev/null \
	    || { cat "$$timeout_log"; exit 1; }; \
	cat "$$log"; cat "$$timeout_log"

decoder-qzeros-boundary: decoder-qzeros-boundary-iverilog decoder-qzeros-boundary-verilator
	@printf '%s\n' 'DECODER_QZEROS_BOUNDARY_PASS simulators=iverilog,verilator domains=q:0..783,k:0..111,v:0..111,o:0..783,gate:0..4255,up:0..4255,down:0..4255 q_to_k_stale=qualified live_oob=rejected serialized_edges=exact'

decoder-qzeros-boundary-iverilog: decoder-layer0-json-validation
	@rm -rf "$(DECODER_QZEROS_DIR)/iverilog" "$(DECODER_QZEROS_IVERILOG_BIN)"
	@mkdir -p "$(DECODER_QZEROS_DIR)/iverilog"
	@set -eu; log="$(DECODER_QZEROS_DIR)/iverilog/run.log"; \
	"$(IVERILOG)" -g2012 -Wall -s ace3_decoder_qzeros_boundary_tb \
	    -o "$(DECODER_QZEROS_IVERILOG_BIN)" "$(DECODER_QZEROS_ADDRESS_RTL)" "$(DECODER_QZEROS_TB)" \
	    > "$$log" 2>&1 || { cat "$$log"; exit 1; }; \
	cd "$(ROOT)" && "$(VVP)" "$(DECODER_QZEROS_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(DECODER_VECTOR_DIR)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	cat "$$log"

decoder-qzeros-boundary-verilator: decoder-layer0-json-validation
	@rm -rf "$(DECODER_QZEROS_VERILATOR_DIR)"
	@mkdir -p "$(DECODER_QZEROS_VERILATOR_DIR)"
	@set -eu; log="$(DECODER_QZEROS_DIR)/verilator.log"; \
	cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --Wall -Wno-fatal \
	    --top-module ace3_decoder_qzeros_address \
	    --Mdir "$(DECODER_QZEROS_VERILATOR_DIR)" \
	    "$(DECODER_QZEROS_ADDRESS_RTL)" "$(DECODER_QZEROS_CPP_TB)" > "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	cd "$(ROOT)" && "$(DECODER_QZEROS_VERILATOR_BIN)" \
	    --vector-dir "$(DECODER_VECTOR_DIR)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	cat "$$log"

decoder-layer0-iverilog-compile: decoder-layer0-json-validation
	@rm -rf "$(DECODER_IVERILOG_DIR)"
	@mkdir -p "$(DECODER_IVERILOG_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-iverilog-compile.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -s ace3_decoder_layer0_token_engine_tb -o build/decoder_layer0_iverilog/ace3_decoder_layer0_token_engine.vvp ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall -s ace3_decoder_layer0_token_engine_tb \
	    -o "$(DECODER_IVERILOG_BIN)" $(DECODER_RTL) "$(DECODER_TB)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

decoder-layer0-iverilog-documented-limit: decoder-layer0-iverilog-compile
	@printf '%s\n' 'DECODER_LAYER0_IVERILOG_LIMIT full_trace=not_run focused_compile=pass width_reset_clear_fault=pass qzeros_address=pass reason=documented_5400s_runtime_limit full_target=decoder-layer0-iverilog-simulation'

decoder-layer0-iverilog-fatal-terminal: decoder-layer0-iverilog-path-safety decoder-layer0-iverilog-compile
	@rm -rf "$(DECODER_IVERILOG_FAIL_RAW_DIR)"
	@mkdir -p "$(DECODER_IVERILOG_FAIL_RAW_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-iverilog-fatal-terminal.log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(DECODER_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(DECODER_VECTOR_DIR)" +RAW_DIR="$(DECODER_IVERILOG_FAIL_RAW_DIR)" \
	    +FAIL_AFTER_OPEN > "$$log" 2>&1; then cat "$$log"; exit 1; fi; \
	expected='schema=ace3_decoder_layer0_raw_v1 natural_terminal=0 exit_code=1 trace_count=0 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_IVERILOG_FAIL_RAW_DIR)/terminal.txt")" = "$$expected"; \
	test ! -s "$(DECODER_IVERILOG_FAIL_RAW_DIR)/trace.hex"; \
	test ! -s "$(DECODER_IVERILOG_FAIL_RAW_DIR)/final.hex"; \
	printf '%s\n' 'DECODER_LAYER0_IVERILOG_FATAL_TERMINAL_PASS natural_terminal=0 raw_files_closed=1 trace=0 final=0' >> "$$log"; \
	tail -n 2 "$$log"

decoder-layer0-iverilog-path-safety:
	@set -eu; vector="$$(realpath -m "$(DECODER_VECTOR_DIR)")"; \
	for raw_path in "$(DECODER_IVERILOG_RAW_DIR)" "$(DECODER_IVERILOG_FAIL_RAW_DIR)"; do \
	  raw="$$(realpath -m "$$raw_path")"; \
	  if test "$$vector" = "$$raw"; then \
	    printf '%s\n' "DECODER_LAYER0_PATH_ALIAS_REJECT simulator=iverilog vector=$$vector raw=$$raw" >&2; \
	    exit 2; \
	  fi; \
	done

decoder-layer0-verilator-path-safety:
	@set -eu; vector="$$(realpath -m "$(DECODER_VECTOR_DIR)")"; \
	for raw_path in "$(DECODER_VERILATOR_RAW_DIR)" "$(DECODER_VERILATOR_FAIL_RAW_DIR)"; do \
	  raw="$$(realpath -m "$$raw_path")"; \
	  if test "$$vector" = "$$raw"; then \
	    printf '%s\n' "DECODER_LAYER0_PATH_ALIAS_REJECT simulator=verilator vector=$$vector raw=$$raw" >&2; \
	    exit 2; \
	  fi; \
	done

decoder-layer0-path-alias-regression: decoder-layer0-json-validation
	@mkdir -p "$(BUILD_DIR)/decoder_layer0_path_alias"
	@set -eu; alias_dir="$(BUILD_DIR)/decoder_layer0_path_alias/vector-link"; \
	rm -f "$$alias_dir"; ln -s "$(DECODER_VECTOR_DIR)" "$$alias_dir"; \
	before="$$(sha256sum "$(DECODER_VECTOR_DIR)/trace.hex" "$(DECODER_VECTOR_DIR)/final.hex")"; \
	if "$(MAKE)" --no-print-directory decoder-layer0-iverilog-simulation \
	    DECODER_IVERILOG_RAW_DIR="$(DECODER_VECTOR_DIR)" \
	    > "$(BUILD_DIR)/decoder_layer0_path_alias/iverilog-direct.log" 2>&1; then exit 1; fi; \
	grep -q 'DECODER_LAYER0_PATH_ALIAS_REJECT simulator=iverilog' \
	    "$(BUILD_DIR)/decoder_layer0_path_alias/iverilog-direct.log"; \
	if "$(MAKE)" --no-print-directory decoder-layer0-verilator-simulation \
	    DECODER_VERILATOR_RAW_DIR="$(DECODER_VECTOR_DIR)" \
	    > "$(BUILD_DIR)/decoder_layer0_path_alias/verilator-direct.log" 2>&1; then exit 1; fi; \
	grep -q 'DECODER_LAYER0_PATH_ALIAS_REJECT simulator=verilator' \
	    "$(BUILD_DIR)/decoder_layer0_path_alias/verilator-direct.log"; \
	if "$(MAKE)" --no-print-directory decoder-layer0-iverilog-simulation \
	    DECODER_IVERILOG_RAW_DIR="$$alias_dir" \
	    > "$(BUILD_DIR)/decoder_layer0_path_alias/iverilog-symlink.log" 2>&1; then exit 1; fi; \
	grep -q 'DECODER_LAYER0_PATH_ALIAS_REJECT simulator=iverilog' \
	    "$(BUILD_DIR)/decoder_layer0_path_alias/iverilog-symlink.log"; \
	if "$(MAKE)" --no-print-directory decoder-layer0-verilator-simulation \
	    DECODER_VERILATOR_RAW_DIR="$$alias_dir" \
	    > "$(BUILD_DIR)/decoder_layer0_path_alias/verilator-symlink.log" 2>&1; then exit 1; fi; \
	grep -q 'DECODER_LAYER0_PATH_ALIAS_REJECT simulator=verilator' \
	    "$(BUILD_DIR)/decoder_layer0_path_alias/verilator-symlink.log"; \
	after="$$(sha256sum "$(DECODER_VECTOR_DIR)/trace.hex" "$(DECODER_VECTOR_DIR)/final.hex")"; \
	test "$$before" = "$$after"; \
	printf '%s\n' "DECODER_LAYER0_PATH_ALIAS_REGRESSION_PASS direct=iverilog,verilator symlink=iverilog,verilator oracle_hashes=unchanged"

decoder-layer0-iverilog-simulation: decoder-layer0-iverilog-path-safety decoder-layer0-iverilog-compile
	@rm -rf "$(DECODER_IVERILOG_RAW_DIR)" "$(DECODER_IVERILOG_FAIL_RAW_DIR)"
	@mkdir -p "$(LOG_DIR)" "$(DECODER_IVERILOG_RAW_DIR)" "$(DECODER_IVERILOG_FAIL_RAW_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-iverilog-simulation.log"; \
	compare="$(LOG_DIR)/decoder-layer0-iverilog-comparison.log"; \
	fail_log="$(LOG_DIR)/decoder-layer0-iverilog-injected-failure.log"; \
	opens="$(LOG_DIR)/decoder-layer0-iverilog-injected-failure.opens"; \
	printf '%s\n' '$ vvp build/decoder_layer0_iverilog/ace3_decoder_layer0_token_engine.vvp +VECTOR_DIR=build/decoder_layer0_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(VVP)" "$(DECODER_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(DECODER_VECTOR_DIR)" +RAW_DIR="$(DECODER_IVERILOG_RAW_DIR)" \
	    +PROGRESS_INTERVAL=1000000 >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	expected='schema=ace3_decoder_layer0_raw_v1 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_IVERILOG_RAW_DIR)/terminal.txt")" = "$$expected"; \
	test "$$(wc -l < "$(DECODER_IVERILOG_RAW_DIR)/trace.hex")" -eq 46676; \
	test "$$(wc -l < "$(DECODER_IVERILOG_RAW_DIR)/final.hex")" -eq 1792; \
	cmp "$(DECODER_IVERILOG_RAW_DIR)/trace.hex" "$(DECODER_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_IVERILOG_RAW_DIR)/final.hex" "$(DECODER_VECTOR_DIR)/final.hex"; \
	printf '%s\n' 'DECODER_LAYER0_IVERILOG_COMPARISON_PASS natural_terminal=1 trace=46676 final=1792 oracle=independent' > "$$compare"; \
	if cd "$(ROOT)" && "$(STRACE)" -qq -f -e trace=openat -o "$$opens" \
	    "$(VVP)" "$(DECODER_IVERILOG_BIN)" +VECTOR_DIR="$(DECODER_VECTOR_DIR)" \
	    +RAW_DIR="$(DECODER_IVERILOG_FAIL_RAW_DIR)" +FAIL_AFTER_RAW > "$$fail_log" 2>&1; then \
	  cat "$$fail_log"; exit 1; \
	fi; \
	fail_expected='schema=ace3_decoder_layer0_raw_v1 natural_terminal=0 exit_code=1 trace_count=1 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_IVERILOG_FAIL_RAW_DIR)/terminal.txt")" = "$$fail_expected"; \
	test "$$(wc -l < "$(DECODER_IVERILOG_FAIL_RAW_DIR)/trace.hex")" -eq 1; \
	test ! -e "$(DECODER_IVERILOG_FAIL_RAW_DIR)/comparison.txt"; \
	if grep -E '/decoder_layer0_vectors/(trace|final)\.hex' "$$opens"; then exit 1; fi; \
	printf '%s\n' 'DECODER_LAYER0_IVERILOG_FAILURE_GATE_PASS natural_terminal=0 raw_rows=1 oracle_opened=0 comparison_created=0' >> "$$fail_log"; \
	cat "$$log"; cat "$$compare"; tail -n 2 "$$fail_log"

decoder-layer0-verilator-compile: decoder-layer0-json-validation
	@rm -rf "$(DECODER_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(DECODER_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build --Wall -Wno-fatal --top-module ace3_decoder_layer0_token_engine --Mdir build/decoder_layer0_verilator/obj_dir ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(DECODER_VERILATOR_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(DECODER_VERILATOR_BIN)"; cat "$$log"

decoder-layer0-verilator-simulation: decoder-layer0-verilator-path-safety decoder-layer0-verilator-compile
	@rm -rf "$(DECODER_VERILATOR_RAW_DIR)" "$(DECODER_VERILATOR_FAIL_RAW_DIR)"
	@mkdir -p "$(LOG_DIR)" "$(DECODER_VERILATOR_RAW_DIR)" "$(DECODER_VERILATOR_FAIL_RAW_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer0-verilator-simulation.log"; \
	compare="$(LOG_DIR)/decoder-layer0-verilator-comparison.log"; \
	fail_log="$(LOG_DIR)/decoder-layer0-verilator-injected-failure.log"; \
	opens="$(LOG_DIR)/decoder-layer0-verilator-injected-failure.opens"; \
	printf '%s\n' '$ build/decoder_layer0_verilator/obj_dir/Vace3_decoder_layer0_token_engine --vector-dir build/decoder_layer0_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(DECODER_VERILATOR_BIN)" \
	    --vector-dir "$(DECODER_VECTOR_DIR)" --raw-dir "$(DECODER_VERILATOR_RAW_DIR)" \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	expected='schema=ace3_decoder_layer0_raw_v1 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_VERILATOR_RAW_DIR)/terminal.txt")" = "$$expected"; \
	test "$$(wc -l < "$(DECODER_VERILATOR_RAW_DIR)/trace.hex")" -eq 46676; \
	test "$$(wc -l < "$(DECODER_VERILATOR_RAW_DIR)/final.hex")" -eq 1792; \
	cmp "$(DECODER_VERILATOR_RAW_DIR)/trace.hex" "$(DECODER_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_VERILATOR_RAW_DIR)/final.hex" "$(DECODER_VECTOR_DIR)/final.hex"; \
	printf '%s\n' 'DECODER_LAYER0_VERILATOR_COMPARISON_PASS natural_terminal=1 trace=46676 final=1792 oracle=independent' > "$$compare"; \
	if cd "$(ROOT)" && "$(STRACE)" -qq -f -e trace=openat -o "$$opens" \
	    "$(DECODER_VERILATOR_BIN)" --vector-dir "$(DECODER_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_VERILATOR_FAIL_RAW_DIR)" --fail-after-raw > "$$fail_log" 2>&1; then \
	  cat "$$fail_log"; exit 1; \
	fi; \
	fail_expected='schema=ace3_decoder_layer0_raw_v1 natural_terminal=0 exit_code=2 trace_count=1 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_VERILATOR_FAIL_RAW_DIR)/terminal.txt")" = "$$fail_expected"; \
	test "$$(wc -l < "$(DECODER_VERILATOR_FAIL_RAW_DIR)/trace.hex")" -eq 1; \
	test ! -e "$(DECODER_VERILATOR_FAIL_RAW_DIR)/comparison.txt"; \
	if grep -E '/decoder_layer0_vectors/(trace|final)\.hex' "$$opens"; then exit 1; fi; \
	printf '%s\n' 'DECODER_LAYER0_VERILATOR_FAILURE_GATE_PASS natural_terminal=0 raw_rows=1 oracle_opened=0 comparison_created=0' >> "$$fail_log"; \
	cat "$$log"; cat "$$compare"; tail -n 2 "$$fail_log"

decoder-layer1-vectors: decoder-layer0-vectors
	@rm -rf "$(DECODER_LAYER1_VECTOR_DIR)"
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer1-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_decoder_layer1_vectors.py --checkpoint model24_execution_vectors/model.safetensors --tensor-map ace3/contracts/model24_tensor_map.json --layer0-handoff build/decoder_layer0_vectors/final.hex --output-dir build/decoder_layer1_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER1_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer0-handoff "$(DECODER_VECTOR_DIR)/final.hex" \
	    --output-dir "$(DECODER_LAYER1_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test "$$(wc -l < "$(DECODER_LAYER1_VECTOR_DIR)/trace.hex")" -eq 46676; \
	test "$$(wc -l < "$(DECODER_LAYER1_VECTOR_DIR)/final.hex")" -eq 1792; \
	test "$$(sha256sum "$(DECODER_LAYER1_VECTOR_DIR)/inputs.hex" | cut -d' ' -f1)" = \
	    22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446; \
	test "$$(sha256sum "$(DECODER_LAYER1_VECTOR_DIR)/final.hex" | cut -d' ' -f1)" = \
	    2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85; \
	cat "$$log"

decoder-layer1-iverilog-boundary: decoder-layer1-vectors
	@rm -rf "$(DECODER_LAYER1_IVERILOG_DIR)"
	@mkdir -p "$(DECODER_LAYER1_IVERILOG_DIR)/raw" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer1-iverilog-boundary.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -P ace3_decoder_layer0_token_engine_tb.LAYER_INDEX=1 ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall \
	    -P ace3_decoder_layer0_token_engine_tb.LAYER_INDEX=1 \
	    -s ace3_decoder_layer0_token_engine_tb \
	    -o "$(DECODER_LAYER1_IVERILOG_BIN)" $(DECODER_RTL) "$(DECODER_TB)" \
	    >> "$$log" 2>&1; then :; else status=$$?; cat "$$log"; exit $$status; fi; \
	if cd "$(ROOT)" && "$(VVP)" "$(DECODER_LAYER1_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(DECODER_LAYER1_VECTOR_DIR)" \
	    +RAW_DIR="$(DECODER_LAYER1_IVERILOG_DIR)/raw" +FAIL_AFTER_RESET \
	    >> "$$log" 2>&1; then cat "$$log"; exit 1; fi; \
	expected='schema=ace3_decoder_layer_raw_v1 layer_index=1 natural_terminal=0 exit_code=1 trace_count=0 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_LAYER1_IVERILOG_DIR)/raw/terminal.txt")" = "$$expected"; \
	test "$$(wc -l < "$(DECODER_LAYER1_IVERILOG_DIR)/raw/trace.hex")" -eq 0; \
	test "$$(wc -l < "$(DECODER_LAYER1_IVERILOG_DIR)/raw/final.hex")" -eq 0; \
	printf '%s\n' 'DECODER_LAYER1_IVERILOG_BOUNDARY_PASS parameter=1 vectors=loaded reset=pass abnormal_terminal=pass full_numerical_run=not_run' >> "$$log"; \
	tail -n 2 "$$log"

decoder-layer1-verilator-compile: decoder-layer1-vectors
	@rm -rf "$(DECODER_LAYER1_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(DECODER_LAYER1_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer1-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build -GLAYER_INDEX=1 --top-module ace3_decoder_layer0_token_engine ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    -GLAYER_INDEX=1 --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(DECODER_LAYER1_VERILATOR_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(DECODER_LAYER1_VERILATOR_BIN)"; cat "$$log"

decoder-layer01-verilator-cascade: decoder-layer0-verilator-compile decoder-layer1-verilator-compile
	@rm -rf "$(DECODER_LAYER01_CASCADE_DIR)" "$(DECODER_LAYER1_VECTOR_DIR)" \
	    "$(DECODER_LAYER1_TAMPER_DIR)"
	@mkdir -p "$(DECODER_LAYER01_CASCADE_DIR)/layer0" \
	    "$(DECODER_LAYER01_CASCADE_DIR)/layer1" \
	    "$(DECODER_LAYER01_CASCADE_DIR)/layer1-fault" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer01-verilator-cascade.log"; \
	compare="$(LOG_DIR)/decoder-layer01-verilator-comparison.log"; \
	fault="$(LOG_DIR)/decoder-layer1-verilator-injected-failure.log"; \
	opens="$(LOG_DIR)/decoder-layer1-verilator-injected-failure.opens"; \
	printf '%s\n' '$ layer0 RTL -> natural terminal -> authenticated layer1 materialization -> layer1 RTL -> post-layer1 oracle' > "$$log"; \
	cd "$(ROOT)" && "$(DECODER_VERILATOR_BIN)" --layer-index 0 \
	    --vector-dir "$(DECODER_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_LAYER01_CASCADE_DIR)/layer0" >> "$$log" 2>&1; \
	expected0='schema=ace3_decoder_layer0_raw_v1 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_LAYER01_CASCADE_DIR)/layer0/terminal.txt")" = "$$expected0"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer0/trace.hex" "$(DECODER_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer0/final.hex" "$(DECODER_VECTOR_DIR)/final.hex"; \
	mkdir -p "$(DECODER_LAYER1_TAMPER_DIR)"; \
	cp "$(DECODER_LAYER01_CASCADE_DIR)/layer0/final.hex" "$(DECODER_LAYER1_TAMPER_DIR)/final.hex"; \
	printf '0000000000\n' >> "$(DECODER_LAYER1_TAMPER_DIR)/final.hex"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER1_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer0-handoff "$(DECODER_LAYER1_TAMPER_DIR)/final.hex" \
	    --output-dir "$(DECODER_LAYER1_TAMPER_DIR)/vectors" > "$(DECODER_LAYER1_TAMPER_DIR)/rejection.log" 2>&1; then \
	  cat "$(DECODER_LAYER1_TAMPER_DIR)/rejection.log"; exit 1; \
	fi; \
	test ! -e "$(DECODER_LAYER1_TAMPER_DIR)/vectors"; \
	cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER1_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer0-handoff "$(DECODER_LAYER01_CASCADE_DIR)/layer0/final.hex" \
	    --output-dir "$(DECODER_LAYER1_VECTOR_DIR)" >> "$$log" 2>&1; \
	test "$$(sha256sum "$(DECODER_LAYER1_VECTOR_DIR)/inputs.hex" | cut -d' ' -f1)" = \
	    22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446; \
	cd "$(ROOT)" && "$(DECODER_LAYER1_VERILATOR_BIN)" --layer-index 1 \
	    --vector-dir "$(DECODER_LAYER1_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_LAYER01_CASCADE_DIR)/layer1" >> "$$log" 2>&1; \
	expected1='schema=ace3_decoder_layer_raw_v1 layer_index=1 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_LAYER01_CASCADE_DIR)/layer1/terminal.txt")" = "$$expected1"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer1/trace.hex" "$(DECODER_LAYER1_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer1/final.hex" "$(DECODER_LAYER1_VECTOR_DIR)/final.hex"; \
	test "$$(sha256sum "$(DECODER_LAYER01_CASCADE_DIR)/layer1/final.hex" | cut -d' ' -f1)" = \
	    2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85; \
	printf '%s\n' 'DECODER_LAYER01_VERILATOR_COMPARISON_PASS layer0_trace=46676 layer1_trace=46676 post_layer1_final=1792 oracle=independent' > "$$compare"; \
	if cd "$(ROOT)" && "$(STRACE)" -qq -f -e trace=openat -o "$$opens" \
	    "$(DECODER_LAYER1_VERILATOR_BIN)" --layer-index 1 \
	    --vector-dir "$(DECODER_LAYER1_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_LAYER01_CASCADE_DIR)/layer1-fault" \
	    --fail-after-raw > "$$fault" 2>&1; then cat "$$fault"; exit 1; fi; \
	fail_expected='schema=ace3_decoder_layer_raw_v1 layer_index=1 natural_terminal=0 exit_code=2 trace_count=1 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_LAYER01_CASCADE_DIR)/layer1-fault/terminal.txt")" = "$$fail_expected"; \
	test "$$(wc -l < "$(DECODER_LAYER01_CASCADE_DIR)/layer1-fault/trace.hex")" -eq 1; \
	test ! -e "$(DECODER_LAYER01_CASCADE_DIR)/layer1-fault/comparison.txt"; \
	if grep -E '/decoder_layer1_vectors/(trace|final)\.hex' "$$opens"; then exit 1; fi; \
	printf '%s\n' 'DECODER_LAYER1_FAILURE_GATE_PASS natural_terminal=0 raw_rows=1 oracle_opened=0 comparison_created=0' >> "$$fault"; \
	cat "$$log"; cat "$$compare"; tail -n 2 "$$fault"; \
	printf '%s\n' 'DECODER_LAYER01_SCOPE_PASS layers=0,1 tokens=2 full24=excluded model=excluded dialogue=excluded synthesis=excluded ppa=excluded fpga=excluded performance=excluded'

decoder-layer2-vectors: decoder-layer1-vectors
	@rm -rf "$(DECODER_LAYER2_VECTOR_DIR)"
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer2-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_decoder_layer2_vectors.py --checkpoint model24_execution_vectors/model.safetensors --tensor-map ace3/contracts/model24_tensor_map.json --layer1-handoff build/decoder_layer1_vectors/final.hex --output-dir build/decoder_layer2_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER2_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer1-handoff "$(DECODER_LAYER1_VECTOR_DIR)/final.hex" \
	    --output-dir "$(DECODER_LAYER2_VECTOR_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test "$$(wc -l < "$(DECODER_LAYER2_VECTOR_DIR)/trace.hex")" -eq 46676; \
	test "$$(wc -l < "$(DECODER_LAYER2_VECTOR_DIR)/final.hex")" -eq 1792; \
	test "$$(sha256sum "$(DECODER_LAYER2_VECTOR_DIR)/inputs.hex" | cut -d' ' -f1)" = \
	    2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85; \
	test "$$(sha256sum "$(DECODER_LAYER2_VECTOR_DIR)/final.hex" | cut -d' ' -f1)" = \
	    244c9d1d52923ecfff743c165da563468746f47557284865a4b22910a967c511; \
	cat "$$log"

decoder-layer2-iverilog-boundary: decoder-layer2-vectors
	@rm -rf "$(DECODER_LAYER2_IVERILOG_DIR)"
	@mkdir -p "$(DECODER_LAYER2_IVERILOG_DIR)/raw" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer2-iverilog-boundary.log"; \
	printf '%s\n' '$ iverilog -g2012 -Wall -P ace3_decoder_layer0_token_engine_tb.LAYER_INDEX=2 ...' > "$$log"; \
	if "$(IVERILOG)" -g2012 -Wall \
	    -P ace3_decoder_layer0_token_engine_tb.LAYER_INDEX=2 \
	    -s ace3_decoder_layer0_token_engine_tb \
	    -o "$(DECODER_LAYER2_IVERILOG_BIN)" $(DECODER_RTL) "$(DECODER_TB)" \
	    >> "$$log" 2>&1; then :; else status=$$?; cat "$$log"; exit $$status; fi; \
	if cd "$(ROOT)" && "$(VVP)" "$(DECODER_LAYER2_IVERILOG_BIN)" \
	    +VECTOR_DIR="$(DECODER_LAYER2_VECTOR_DIR)" \
	    +RAW_DIR="$(DECODER_LAYER2_IVERILOG_DIR)/raw" +FAIL_AFTER_RESET \
	    >> "$$log" 2>&1; then cat "$$log"; exit 1; fi; \
	expected='schema=ace3_decoder_layer_raw_v1 layer_index=2 natural_terminal=0 exit_code=1 trace_count=0 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_LAYER2_IVERILOG_DIR)/raw/terminal.txt")" = "$$expected"; \
	test "$$(wc -l < "$(DECODER_LAYER2_IVERILOG_DIR)/raw/trace.hex")" -eq 0; \
	test "$$(wc -l < "$(DECODER_LAYER2_IVERILOG_DIR)/raw/final.hex")" -eq 0; \
	printf '%s\n' 'DECODER_LAYER2_IVERILOG_BOUNDARY_PASS parameter=2 vectors=loaded reset=pass abnormal_terminal=pass full_numerical_run=not_run' >> "$$log"; \
	tail -n 2 "$$log"

decoder-layer2-verilator-compile: decoder-layer2-vectors
	@rm -rf "$(DECODER_LAYER2_VERILATOR_OBJ_DIR)"
	@mkdir -p "$(DECODER_LAYER2_VERILATOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer2-verilator-compile.log"; \
	printf '%s\n' '$ verilator --cc --exe --build -GLAYER_INDEX=2 --top-module ace3_decoder_layer0_token_engine ...' > "$$log"; \
	if cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    -GLAYER_INDEX=2 --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(DECODER_LAYER2_VERILATOR_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	test -x "$(DECODER_LAYER2_VERILATOR_BIN)"; cat "$$log"

decoder-layer012-verilator-cascade: decoder-layer01-verilator-cascade decoder-layer2-verilator-compile
	@rm -rf "$(DECODER_LAYER012_CASCADE_DIR)" "$(DECODER_LAYER2_VECTOR_DIR)" \
	    "$(DECODER_LAYER2_TAMPER_DIR)"
	@mkdir -p "$(DECODER_LAYER012_CASCADE_DIR)/layer2" \
	    "$(DECODER_LAYER012_CASCADE_DIR)/layer2-fault" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/decoder-layer012-verilator-cascade.log"; \
	compare="$(LOG_DIR)/decoder-layer012-verilator-comparison.log"; \
	fault="$(LOG_DIR)/decoder-layer2-verilator-injected-failure.log"; \
	opens="$(LOG_DIR)/decoder-layer2-verilator-injected-failure.opens"; \
	printf '%s\n' '$ accepted layer0-to-layer1 cascade -> authenticated layer2 materialization -> layer2 RTL -> post-layer2 oracle' > "$$log"; \
	expected1='schema=ace3_decoder_layer_raw_v1 layer_index=1 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_LAYER01_CASCADE_DIR)/layer1/terminal.txt")" = "$$expected1"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer1/trace.hex" "$(DECODER_LAYER1_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_LAYER01_CASCADE_DIR)/layer1/final.hex" "$(DECODER_LAYER1_VECTOR_DIR)/final.hex"; \
	mkdir -p "$(DECODER_LAYER2_TAMPER_DIR)"; \
	cp "$(DECODER_LAYER01_CASCADE_DIR)/layer1/final.hex" "$(DECODER_LAYER2_TAMPER_DIR)/final.hex"; \
	printf '0000000000\n' >> "$(DECODER_LAYER2_TAMPER_DIR)/final.hex"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER2_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer1-handoff "$(DECODER_LAYER2_TAMPER_DIR)/final.hex" \
	    --output-dir "$(DECODER_LAYER2_TAMPER_DIR)/vectors" > "$(DECODER_LAYER2_TAMPER_DIR)/rejection.log" 2>&1; then \
	  cat "$(DECODER_LAYER2_TAMPER_DIR)/rejection.log"; exit 1; \
	fi; \
	test ! -e "$(DECODER_LAYER2_TAMPER_DIR)/vectors"; \
	cd "$(ROOT)" && "$(PYTHON)" "$(DECODER_LAYER2_GENERATOR)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --layer1-handoff "$(DECODER_LAYER01_CASCADE_DIR)/layer1/final.hex" \
	    --output-dir "$(DECODER_LAYER2_VECTOR_DIR)" >> "$$log" 2>&1; \
	test "$$(sha256sum "$(DECODER_LAYER2_VECTOR_DIR)/inputs.hex" | cut -d' ' -f1)" = \
	    2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85; \
	cd "$(ROOT)" && "$(DECODER_LAYER2_VERILATOR_BIN)" --layer-index 2 \
	    --vector-dir "$(DECODER_LAYER2_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_LAYER012_CASCADE_DIR)/layer2" >> "$$log" 2>&1; \
	expected2='schema=ace3_decoder_layer_raw_v1 layer_index=2 natural_terminal=1 exit_code=0 trace_count=46676 final_count=1792 done_count=2'; \
	test "$$(cat "$(DECODER_LAYER012_CASCADE_DIR)/layer2/terminal.txt")" = "$$expected2"; \
	cmp "$(DECODER_LAYER012_CASCADE_DIR)/layer2/trace.hex" "$(DECODER_LAYER2_VECTOR_DIR)/trace.hex"; \
	cmp "$(DECODER_LAYER012_CASCADE_DIR)/layer2/final.hex" "$(DECODER_LAYER2_VECTOR_DIR)/final.hex"; \
	test "$$(sha256sum "$(DECODER_LAYER012_CASCADE_DIR)/layer2/final.hex" | cut -d' ' -f1)" = \
	    244c9d1d52923ecfff743c165da563468746f47557284865a4b22910a967c511; \
	printf '%s\n' 'DECODER_LAYER012_VERILATOR_COMPARISON_PASS layer0_trace=46676 layer1_trace=46676 layer2_trace=46676 post_layer2_final=1792 oracle=independent' > "$$compare"; \
	if cd "$(ROOT)" && "$(STRACE)" -qq -f -e trace=openat -o "$$opens" \
	    "$(DECODER_LAYER2_VERILATOR_BIN)" --layer-index 2 \
	    --vector-dir "$(DECODER_LAYER2_VECTOR_DIR)" \
	    --raw-dir "$(DECODER_LAYER012_CASCADE_DIR)/layer2-fault" \
	    --fail-after-raw > "$$fault" 2>&1; then cat "$$fault"; exit 1; fi; \
	fail_expected='schema=ace3_decoder_layer_raw_v1 layer_index=2 natural_terminal=0 exit_code=2 trace_count=1 final_count=0 done_count=0'; \
	test "$$(cat "$(DECODER_LAYER012_CASCADE_DIR)/layer2-fault/terminal.txt")" = "$$fail_expected"; \
	test "$$(wc -l < "$(DECODER_LAYER012_CASCADE_DIR)/layer2-fault/trace.hex")" -eq 1; \
	test ! -e "$(DECODER_LAYER012_CASCADE_DIR)/layer2-fault/comparison.txt"; \
	if grep -E '/decoder_layer2_vectors/(trace|final)\.hex' "$$opens"; then exit 1; fi; \
	printf '%s\n' 'DECODER_LAYER2_FAILURE_GATE_PASS natural_terminal=0 raw_rows=1 oracle_opened=0 comparison_created=0' >> "$$fault"; \
	cat "$$log"; cat "$$compare"; tail -n 2 "$$fault"; \
	printf '%s\n' 'DECODER_LAYER012_SCOPE_PASS layers=0,1,2 tokens=2 tamper=pass fault=pass reset=pass backpressure=pass full24=excluded model=excluded dialogue=excluded synthesis=excluded ppa=excluded fpga=excluded performance=excluded'

model24-execution: model24-execution-validation model24-execution-tests
	@printf '%s\n' 'MODEL24_EXECUTION_PASS structural_schedule=483 layer0_rtl_binding=authenticated layer0_iverilog=separate_boundary layers_1_through_23_rtl=deferred full_model_numerics=deferred dialogue=deferred'

model24-execution-vectors:
	@rm -rf "$(MODEL24_VECTOR_DIR)"
	@mkdir -p "$(MODEL24_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/model24-execution-vector-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/generate_model24_execution_vectors.py --output-dir build/model24_execution_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_GENERATOR)" \
	    --output-dir "$(MODEL24_VECTOR_DIR)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

model24-execution-validation: model24-execution-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/model24-execution-vector-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/validate_model24_execution_vectors.py --vector-dir build/model24_execution_vectors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_VALIDATOR)" \
	    --vector-dir "$(MODEL24_VECTOR_DIR)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

model24-execution-tests:
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/model24-execution-tests.log"; \
	export ACE3_OFFICIAL_MODEL24_CHECKPOINT="$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR="$(OFFICIAL_MODEL24_TOKENIZER_DIR)"; \
	printf '%s\n' '$ python3 -m unittest ace3/model/tests/test_model24_execution.py' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_model24_execution.py >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

model24-layer-indexed-handoff:
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_LAYER_HANDOFF_TEST)" \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --handoff "$(VL15_LAYER0_HANDOFF)"

model24-publication-tests: model24-layer-controller \
	model24-controller-cascade-tests model24-token0-diagnostic-tests
	@printf '%s\n' 'MODEL24_PUBLICATION_TESTS_PASS controller=pass cascade_units=pass token0_units=pass full24_rerun=not_run'

model24-layer-controller: model24-layer-controller-simulations
	@printf '%s\n' 'MODEL24_LAYER_CONTROLLER_PASS simulators=iverilog,verilator layers=24 checkpoints=24 strict_order=pass reset=pass fault=pass backpressure=pass numerical_rtl=not_claimed'

model24-layer-controller-vectors:
	@rm -rf "$(MODEL24_LAYER_CONTROLLER_DIR)"
	@mkdir -p "$(MODEL24_LAYER_CONTROLLER_DIR)"
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_LAYER_CONTROLLER_GENERATOR)" \
	    --contract "$(MODEL24_LAYER_CONTROLLER_CONTRACT)" \
	    --output-dir "$(MODEL24_LAYER_CONTROLLER_DIR)"

model24-layer-controller-validation: model24-layer-controller-vectors
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_LAYER_CONTROLLER_VALIDATOR)" \
	    --contract "$(MODEL24_LAYER_CONTROLLER_CONTRACT)" \
	    --generated-dir "$(MODEL24_LAYER_CONTROLLER_DIR)"

model24-layer-controller-iverilog: model24-layer-controller-validation
	@rm -rf "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)"
	@mkdir -p "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)"
	@"$(IVERILOG)" -g2012 -Wall -s ace3_model24_layer_controller_tb \
	    -o "$(MODEL24_LAYER_CONTROLLER_BIN)" \
	    "$(MODEL24_LAYER_CONTROLLER_RTL)" "$(MODEL24_LAYER_CONTROLLER_TB)"
	@cd "$(ROOT)" && "$(VVP)" "$(MODEL24_LAYER_CONTROLLER_BIN)" \
	    +VECTOR_DIR="$(MODEL24_LAYER_CONTROLLER_DIR)" \
	    +RAW_DIR="$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)"

model24-layer-controller-verilator: model24-layer-controller-validation
	@rm -rf "$(MODEL24_LAYER_CONTROLLER_VERILATOR_DIR)" \
	    "$(MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR)"
	@mkdir -p "$(MODEL24_LAYER_CONTROLLER_VERILATOR_DIR)" \
	    "$(MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR)"
	@"$(VERILATOR)" --cc --exe --build -Wall \
	    --top-module ace3_model24_layer_controller \
	    --Mdir "$(MODEL24_LAYER_CONTROLLER_VERILATOR_DIR)" \
	    "$(MODEL24_LAYER_CONTROLLER_RTL)" "$(MODEL24_LAYER_CONTROLLER_CPP_TB)"
	@cd "$(ROOT)" && "$(MODEL24_LAYER_CONTROLLER_VERILATOR_BIN)" \
	    +VECTOR_DIR="$(MODEL24_LAYER_CONTROLLER_DIR)" \
	    +RAW_DIR="$(MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR)"

model24-layer-controller-simulations: model24-layer-controller-iverilog \
	model24-layer-controller-verilator
	@cmp "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)/controller_events.hex" \
	    "$(MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR)/controller_events.hex"
	@cmp "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)/terminal.txt" \
	    "$(MODEL24_LAYER_CONTROLLER_VERILATOR_RAW_DIR)/terminal.txt"

model24-controller-cascade-bindings: model24-layer-controller-simulations
	@rm -rf "$(MODEL24_CONTROLLER_CASCADE_DIR)"
	@mkdir -p "$(MODEL24_CONTROLLER_CASCADE_DIR)"
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" bindings \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)"

model24-controller-cascade-execution: model24-controller-cascade-bindings
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" execute \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)" \
	    --simulation-dir "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)" \
	    --output-dir "$(MODEL24_CONTROLLER_CASCADE_DIR)"

model24-controller-cascade-comparison: model24-controller-cascade-execution
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" compare \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)" \
	    --simulation-dir "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)" \
	    --output-dir "$(MODEL24_CONTROLLER_CASCADE_DIR)"

model24-controller-cascade-validation: model24-controller-cascade-comparison
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" validate \
	    --repository-root "$(ROOT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)" \
	    --simulation-dir "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)" \
	    --output-dir "$(MODEL24_CONTROLLER_CASCADE_DIR)"

model24-controller-cascade-tests:
	@cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" \
	    "$(MODEL24_RTL_CASCADE_EXECUTOR)" \
	    "$(MODEL24_CONTROLLER_CASCADE_TEST)"
	@cd "$(ROOT)" && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_controller_model24_cascade.py

model24-token0-diagnostic-tests:
	@cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(MODEL24_TOKEN0_DIAGNOSTIC)" \
	    "$(MODEL24_TOKEN0_DIAGNOSTIC_TEST)"
	@cd "$(ROOT)" && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_layer3_token0_diagnostic.py

model24-layer-controller-failure-gate: model24-layer-controller-validation \
	model24-controller-cascade-bindings
	@rm -rf "$(MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR)" \
	    "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure"
	@mkdir -p "$(MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR)" \
	    "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure"
	@set -eu; \
	if cd "$(ROOT)" && "$(VVP)" "$(MODEL24_LAYER_CONTROLLER_BIN)" \
	    +VECTOR_DIR="$(MODEL24_LAYER_CONTROLLER_DIR)" \
	    +RAW_DIR="$(MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR)" \
	    +INJECT_FAILURE_AFTER_LAUNCH=1 >/dev/null 2>&1; then \
	    echo "injected controller failure unexpectedly passed"; exit 1; \
	fi; \
	test "$$(cat "$(MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR)/terminal.txt")" = \
	    'schema=ace3_model24_controller_raw_v1 natural_terminal=0 exit_code=2 launches=1 checkpoints=0 done=0 terminal_layer=none'; \
	if cd "$(ROOT)" && "$(STRACE)" -f -e trace=openat \
	    -o "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure/opens.log" \
	    "$(PYTHON)" "$(MODEL24_CONTROLLER_CASCADE_EXECUTOR)" execute \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)" \
	    --simulation-dir "$(MODEL24_LAYER_CONTROLLER_FAILURE_RAW_DIR)" \
	    --output-dir "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure" \
	    >/dev/null 2>&1; then \
	    echo "failed simulation reached cascade execution"; exit 1; \
	fi; \
	! grep -F 'model.safetensors' \
	    "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure/opens.log"; \
	test "$$(cat "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure/terminal.txt")" = \
	    'schema=ace3_controller_model24_execution_v1 natural_terminal=0 exit_code=2 completed_layers=0 final_records=0'; \
	test ! -e "$(MODEL24_CONTROLLER_CASCADE_DIR)/failure/comparison.json"; \
	printf '%s\n' 'MODEL24_CONTROLLER_FAILURE_GATE_PASS partial_raw=1 natural_terminal=0 checkpoint_opened=0 comparison_created=0'

model24-controller-cascade: model24-controller-cascade-validation \
	model24-controller-cascade-tests model24-layer-controller-failure-gate
	@cd "$(ROOT)" && "$(PYTHON)" -c 'import json; from pathlib import Path; s=json.loads(Path("build/model24_controller_cascade/manifest.json").read_text())["summary"]; print(f"MODEL24_CONTROLLER_CASCADE_PASS layers={s['\''layers'\'']} checkpoints={s['\''checkpoints'\'']} tensors={s['\''consumed_tensors'\'']} terminal_layer={s['\''terminal_layer'\'']} tolerance={s['\''absolute_tolerance'\'']} max_abs_error={s['\''max_abs_error'\'']} tokenizer_dialogue=not_produced tied_lm_head=not_executed synthesis=not_run ppa=not_measured fpga=not_run latency=not_measured throughput=not_measured")'

model24-rtl-layer-compile:
	@test "$(MODEL24_RTL_LAYER_INDEX)" -ge 0 -a "$(MODEL24_RTL_LAYER_INDEX)" -le 23
	@rm -rf "$(MODEL24_RTL_LAYER_OBJ_DIR)"
	@mkdir -p "$(MODEL24_RTL_LAYER_DIR)"
	@"$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    -GLAYER_INDEX="$(MODEL24_RTL_LAYER_INDEX)" \
	    -GACCURATE_SILU="$(MODEL24_RTL_ACCURATE_SILU)" \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(MODEL24_RTL_LAYER_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)"
	@test -x "$(MODEL24_RTL_LAYER_BIN)"

model24-first-voice-savable-compile:
	@rm -rf "$(FIRST_VOICE_SAVABLE_OBJ_DIR)"
	@mkdir -p "$(FIRST_VOICE_SAVABLE_DIR)"
	@cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(FIRST_VOICE_SAVABLE_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)"
	@test -x "$(FIRST_VOICE_SAVABLE_BIN)"

model24-first-voice-savable-test: model24-first-voice-savable-compile
	@rm -f "$(FIRST_VOICE_SAVABLE_DIR)/rtl.state" \
	    "$(FIRST_VOICE_SAVABLE_DIR)/rtl.state.partial"
	@"$(FIRST_VOICE_SAVABLE_BIN)" \
	    --savable-self-test "$(FIRST_VOICE_SAVABLE_DIR)/rtl.state"
	@test -s "$(FIRST_VOICE_SAVABLE_DIR)/rtl.state"

model24-first-voice-layer-compile:
	@test "$(FIRST_VOICE_RTL_LAYER_INDEX)" -ge 0 \
	    -a "$(FIRST_VOICE_RTL_LAYER_INDEX)" -le 23
	@rm -rf "$(FIRST_VOICE_RTL_LAYER_OBJ_DIR)"
	@mkdir -p "$(FIRST_VOICE_RTL_LAYER_DIR)"
	@cd "$(ROOT)" && "$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \
	    -GLAYER_INDEX="$(FIRST_VOICE_RTL_LAYER_INDEX)" \
	    -GACCURATE_SILU="$$(test "$(FIRST_VOICE_RTL_LAYER_INDEX)" -ge 3 && echo 1 || echo 0)" \
	    --top-module ace3_decoder_layer0_token_engine \
	    --Mdir "$(FIRST_VOICE_RTL_LAYER_OBJ_DIR)" $(DECODER_RTL) \
	    "$(DECODER_CPP_TB)"
	@test -x "$(FIRST_VOICE_RTL_LAYER_BIN)"

model24-first-voice-compile-all:
	@set -eu; for layer in $$(seq 0 23); do \
	    "$(MAKE)" --no-print-directory model24-first-voice-layer-compile \
	        FIRST_VOICE_RTL_LAYER_INDEX="$$layer"; \
	done
	@cd "$(ROOT)" && "$(PYTHON)" "$(FIRST_VOICE_DRIVER)" \
	    --repository-root "$(ROOT)" \
	    --compiled-dir "$(FIRST_VOICE_DIR)/compiled" \
	    --bind-compiled

model24-first-voice-hybrid-tests: model24-first-voice-savable-test
	@cd "$(ROOT)" && PYTHONPYCACHEPREFIX="$(FIRST_VOICE_DIR)/pycache" \
	    "$(PYTHON)" -m py_compile \
	    "$(FIRST_VOICE_DRIVER)" "$(FIRST_VOICE_TEST)"
	@cd "$(ROOT)" && PYTHONPYCACHEPREFIX="$(FIRST_VOICE_DIR)/pycache" \
	    "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_model24_first_voice_hybrid.py

model24-first-voice-hybrid: model24-first-voice-compile-all \
	model24-first-voice-hybrid-tests
	@rm -rf "$(FIRST_VOICE_DIR)/execution"
	@cd "$(ROOT)" && "$(PYTHON)" "$(FIRST_VOICE_DRIVER)" \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --compiled-dir "$(FIRST_VOICE_DIR)/compiled" \
	    --output-dir "$(FIRST_VOICE_DIR)/execution" \
	    --max-new-tokens "$(FIRST_VOICE_MAX_NEW_TOKENS)"

model24-controller-rtl-cascade: model24-controller-cascade-bindings \
	model24-controller-cascade-tests
	@cd "$(ROOT)" && "$(PYTHON)" "$(MODEL24_RTL_CASCADE_EXECUTOR)" \
	    --repository-root "$(ROOT)" \
	    --checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --tensor-map "$(MODEL24_TENSOR_MAP)" \
	    --bindings "$(MODEL24_CONTROLLER_CASCADE_BINDINGS)" \
	    --simulation-dir "$(MODEL24_LAYER_CONTROLLER_IVERILOG_RAW_DIR)" \
	    --output-dir "$(MODEL24_RTL_CASCADE_DIR)"

official-model24-next-token: official-model24-next-token-validation \
	official-model24-next-token-tests
	@printf '%s\n' 'OFFICIAL_MODEL24_NEXT_TOKEN_PASS layers=24 layer_tensors=624 intermediate_hashes=480 terminal_reference=pytorch logits_reference=pytorch argmax=matched dialogue=not_demonstrated rtl=not_demonstrated synthesis=not_run ppa=not_measured fpga=not_run latency=not_measured throughput=not_measured'

official-model24-next-token-vectors:
	@rm -rf "$(OFFICIAL_MODEL24_VECTOR_DIR)"
	@mkdir -p "$(OFFICIAL_MODEL24_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-next-token-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_next_token.py generate --output-dir build/official_model24_next_token --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_EXECUTOR)" generate \
	    --output-dir "$(OFFICIAL_MODEL24_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-next-token-validation: official-model24-next-token-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-next-token-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_next_token.py validate --vector-dir build/official_model24_next_token --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_EXECUTOR)" validate \
	    --vector-dir "$(OFFICIAL_MODEL24_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-next-token-tests: official-model24-next-token-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-next-token-tests.log"; \
	export ACE3_OFFICIAL_MODEL24_CHECKPOINT="$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR="$(OFFICIAL_MODEL24_TOKENIZER_DIR)"; \
	printf '%s\n' '$ python3 -m py_compile ace3/model/official_model24_next_token.py ace3/model/tests/test_official_model24_next_token.py' > "$$log"; \
	printf '%s\n' '$ python3 -m unittest ace3/model/tests/test_official_model24_next_token.py' >> "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(OFFICIAL_MODEL24_EXECUTOR)" "$(OFFICIAL_MODEL24_TEST)" >> "$$log" 2>&1 \
	    && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_official_model24_next_token.py >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-dialogue: official-model24-dialogue-validation \
	official-model24-dialogue-tests
	@cd "$(ROOT)" && "$(PYTHON)" -c 'import json; from pathlib import Path; g=json.loads(Path("build/official_model24_dialogue/official_model24_dialogue.json").read_text())["generation"]; print(f"OFFICIAL_MODEL24_DIALOGUE_PASS layers=24 generated_tokens={len(g['\''generated_token_ids'\''])} decoded_text={g['\''decoded_text'\'']!r} stop={g['\''stop_reason'\'']} cache_lineage=extended pytorch_argmax=matched rtl=not_demonstrated synthesis=not_run ppa=not_measured fpga=not_run latency=not_measured throughput=not_measured broader_quality=fixed_prompt_only")'

official-model24-dialogue-vectors:
	@rm -rf "$(OFFICIAL_MODEL24_DIALOGUE_VECTOR_DIR)"
	@mkdir -p "$(OFFICIAL_MODEL24_DIALOGUE_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-dialogue-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_dialogue.py generate --output-dir build/official_model24_dialogue --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_DIALOGUE_EXECUTOR)" generate \
	    --output-dir "$(OFFICIAL_MODEL24_DIALOGUE_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-dialogue-validation: official-model24-dialogue-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-dialogue-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_dialogue.py validate --vector-dir build/official_model24_dialogue --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_DIALOGUE_EXECUTOR)" validate \
	    --vector-dir "$(OFFICIAL_MODEL24_DIALOGUE_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-dialogue-tests: official-model24-dialogue-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-dialogue-tests.log"; \
	export ACE3_OFFICIAL_MODEL24_CHECKPOINT="$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR="$(OFFICIAL_MODEL24_TOKENIZER_DIR)"; \
	printf '%s\n' '$ python3 -m py_compile ace3/model/official_model24_dialogue.py ace3/model/tests/test_official_model24_dialogue.py' > "$$log"; \
	printf '%s\n' '$ python3 -m unittest ace3/model/tests/test_official_model24_dialogue.py' >> "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(OFFICIAL_MODEL24_DIALOGUE_EXECUTOR)" "$(OFFICIAL_MODEL24_DIALOGUE_TEST)" >> "$$log" 2>&1 \
	    && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_official_model24_dialogue.py >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-showcase: official-model24-showcase-validation \
	official-model24-showcase-tests
	@cd "$(ROOT)" && "$(PYTHON)" -c 'import json; from pathlib import Path; d=json.loads(Path("build/official_model24_showcase/official_model24_showcase.json").read_text()); print(f"OFFICIAL_MODEL24_SHOWCASE_PASS prompts={len(d['\''rows'\''])} failures={len(d['\''failures'\''])} outputs=preserved primary_pytorch=per_token rtl=not_demonstrated synthesis=not_run ppa=not_measured fpga=not_run latency=not_measured throughput=not_measured")'

official-model24-showcase-vectors:
	@rm -rf "$(OFFICIAL_MODEL24_SHOWCASE_VECTOR_DIR)"
	@mkdir -p "$(OFFICIAL_MODEL24_SHOWCASE_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-showcase-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_showcase.py generate --output-dir build/official_model24_showcase --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_SHOWCASE_EXECUTOR)" generate \
	    --output-dir "$(OFFICIAL_MODEL24_SHOWCASE_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-showcase-validation: official-model24-showcase-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-showcase-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_showcase.py validate --vector-dir build/official_model24_showcase --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_SHOWCASE_EXECUTOR)" validate \
	    --vector-dir "$(OFFICIAL_MODEL24_SHOWCASE_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-showcase-tests: official-model24-showcase-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-showcase-tests.log"; \
	export ACE3_OFFICIAL_MODEL24_CHECKPOINT="$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR="$(OFFICIAL_MODEL24_TOKENIZER_DIR)"; \
	printf '%s\n' '$ python3 -m py_compile ace3/model/official_model24_dialogue.py ace3/model/official_model24_showcase.py ace3/model/tests/test_official_model24_showcase.py' > "$$log"; \
	printf '%s\n' '$ python3 -m unittest ace3/model/tests/test_official_model24_showcase.py' >> "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(OFFICIAL_MODEL24_DIALOGUE_EXECUTOR)" "$(OFFICIAL_MODEL24_SHOWCASE_EXECUTOR)" \
	    "$(OFFICIAL_MODEL24_SHOWCASE_TEST)" >> "$$log" 2>&1 \
	    && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_official_model24_showcase.py >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-systematic-continuations: \
	official-model24-systematic-continuations-validation \
	official-model24-systematic-continuations-tests
	@cd "$(ROOT)" && "$(PYTHON)" -c 'import json; from pathlib import Path; s=json.loads(Path("build/official_model24_systematic_continuations/summary.json").read_text()); r=s["results"]; print(f"OFFICIAL_MODEL24_SYSTEMATIC_CONTINUATIONS_PASS cases={r['\''cases'\'']} completed={r['\''completed_cases'\'']} steps={r['\''steps'\'']} mismatches={r['\''mismatches'\'']} execution_failures={r['\''execution_failures'\'']} baseline=showcasecontinuations15c unreviewed_486e5d848245=excluded_claim_evidence rtl=not_demonstrated synthesis=not_run ppa=not_measured fpga=not_run latency=diagnostic_only throughput=not_measured broader_quality=bounded_suite_only")'

official-model24-systematic-continuations-vectors:
	@rm -rf "$(OFFICIAL_MODEL24_SYSTEMATIC_VECTOR_DIR)"
	@mkdir -p "$(OFFICIAL_MODEL24_SYSTEMATIC_VECTOR_DIR)" "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-systematic-continuations-generation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_systematic_continuations.py generate --output-dir build/official_model24_systematic_continuations --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_SYSTEMATIC_EXECUTOR)" generate \
	    --output-dir "$(OFFICIAL_MODEL24_SYSTEMATIC_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-systematic-continuations-validation: \
	official-model24-systematic-continuations-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-systematic-continuations-validation.log"; \
	printf '%s\n' '$ python3 ace3/model/official_model24_systematic_continuations.py validate --vector-dir build/official_model24_systematic_continuations --official-checkpoint model24_execution_vectors/model.safetensors' > "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" "$(OFFICIAL_MODEL24_SYSTEMATIC_EXECUTOR)" validate \
	    --vector-dir "$(OFFICIAL_MODEL24_SYSTEMATIC_VECTOR_DIR)" \
	    --official-checkpoint "$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    --official-tokenizer-dir "$(OFFICIAL_MODEL24_TOKENIZER_DIR)" >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

official-model24-systematic-continuations-tests: \
	official-model24-systematic-continuations-vectors
	@mkdir -p "$(LOG_DIR)"
	@set -eu; log="$(LOG_DIR)/official-model24-systematic-continuations-tests.log"; \
	export ACE3_OFFICIAL_MODEL24_CHECKPOINT="$(OFFICIAL_MODEL24_CHECKPOINT)" \
	    ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR="$(OFFICIAL_MODEL24_TOKENIZER_DIR)"; \
	printf '%s\n' '$ python3 -m py_compile ace3/model/official_model24_systematic_continuations.py ace3/model/tests/test_official_model24_systematic_continuations.py' > "$$log"; \
	printf '%s\n' '$ python3 -m unittest ace3/model/tests/test_official_model24_systematic_continuations.py' >> "$$log"; \
	if cd "$(ROOT)" && "$(PYTHON)" -m py_compile \
	    "$(OFFICIAL_MODEL24_SYSTEMATIC_EXECUTOR)" "$(OFFICIAL_MODEL24_SYSTEMATIC_TEST)" \
	    >> "$$log" 2>&1 \
	    && "$(PYTHON)" -m unittest \
	    ace3/model/tests/test_official_model24_systematic_continuations.py \
	    >> "$$log" 2>&1; then :; \
	else status=$$?; cat "$$log"; exit $$status; fi; \
	cat "$$log"

clean:
	rm -rf "$(BUILD_DIR)"
