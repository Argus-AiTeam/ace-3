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
OFFICIAL_TENSOR_DIR ?= /home/argustest/ace-2/build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official

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

IVERILOG_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane.vvp
PROTOCOL_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane_protocol.vvp
PROJECTION_IVERILOG_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_projection_engine.vvp
PROJECTION_4864_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_projection_4864_cycle.vvp
VERILATOR_BIN := $(VERILATOR_OBJ_DIR)/Vace3_awq_w4a16_g128_dot_lane
PROJECTION_VERILATOR_BIN := $(PROJECTION_VERILATOR_OBJ_DIR)/Vace3_awq_w4a16_projection_engine

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: \
	test _validate oracle vectors json-validation tamper-rejection \
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
	attention-verilator-compile attention-verilator-simulation clean

test:
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
	    "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
	    >> "$$log" 2>&1 || { cat "$$log"; exit 1; }; \
	  "$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_awq_w4a16_projection_engine \
	    -GIN_FEATURES=$$1 -GOUT_FEATURES=$$2 \
	    "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
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
	    "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
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
	    "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
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
	    "$(RTL)" "$(PROJECTION_ROUNDER_RTL)" "$(PROJECTION_RTL)" \
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
	    "$(PROJECTION_ROUNDER_RTL)" "$(RTL)" "$(PROJECTION_RTL)" \
	    "$(QKV_CLUSTER_RTL)" "$(QKV_GEOMETRY_TB)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VVP)" "$(QKV_GEOMETRY_BIN)" >> "$$log" 2>&1 \
	    || { cat "$$log"; exit 1; }; \
	"$(VERILATOR)" --lint-only --Wall \
	    --top-module ace3_qkv_projection_cluster \
	    "$(PROJECTION_ROUNDER_RTL)" "$(RTL)" "$(PROJECTION_RTL)" \
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

clean:
	rm -rf "$(BUILD_DIR)"
