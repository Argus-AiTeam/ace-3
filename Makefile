.DEFAULT_GOAL := test
.NOTPARALLEL:

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(ROOT)/build
VECTOR_DIR := $(BUILD_DIR)/vectors
LOG_DIR := $(BUILD_DIR)/logs
IVERILOG_DIR := $(BUILD_DIR)/iverilog
VERILATOR_DIR := $(BUILD_DIR)/verilator
VERILATOR_OBJ_DIR := $(VERILATOR_DIR)/obj_dir

PYTHON ?= python3
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
OFFICIAL_TENSOR_DIR ?= /home/argustest/ace-2/build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official

RTL := $(ROOT)/ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv
SV_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_tb.sv
PROTOCOL_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_protocol_tb.sv
CPP_TB := $(ROOT)/ace3/tb/ace3_awq_w4a16_g128_dot_lane_main.cpp
ORACLE := $(ROOT)/ace3/model/awq_bit_oracle.py
GENERATOR := $(ROOT)/ace3/model/generate_vectors.py
VALIDATOR := $(ROOT)/ace3/model/validate_vectors.py
CONTRACT := $(ROOT)/ace3/contracts/awq_w4a16_g128_dot_lane.json
EVIDENCE_BINDINGS := $(ROOT)/ace3/contracts/awq_w4a16_g128_evidence_bindings.json
FROZEN_MANIFEST := $(ROOT)/ace3/contracts/awq_w4a16_g128_vectors_manifest.json
STANDALONE_BINDINGS := $(ROOT)/ace3/contracts/awq_w4a16_g128_standalone_vector_bindings.json
TAMPER_DIR := $(BUILD_DIR)/tamper-vectors

IVERILOG_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane.vvp
PROTOCOL_BIN := $(IVERILOG_DIR)/ace3_awq_w4a16_g128_dot_lane_protocol.vvp
VERILATOR_BIN := $(VERILATOR_OBJ_DIR)/Vace3_awq_w4a16_g128_dot_lane

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: \
	test _validate oracle vectors json-validation tamper-rejection \
	iverilog iverilog-compile iverilog-simulation \
	iverilog-protocol-compile iverilog-protocol-simulation \
	verilator verilator-compile verilator-simulation clean

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
	  test ! -e "$(ROOT)/ace3/generated"; \
	  printf '%s\n' 'REPOSITORY_HYGIENE_PASS status_unchanged=yes build_ignored=yes legacy_generated_absent=yes diff_check=pass'; \
	} > "$$log" 2>&1 || { status=$$?; cat "$$log"; exit $$status; }; \
	cat "$$log"
	@printf '%s\n' 'STANDALONE_VALIDATION_PASS semantic_checks=fresh oracle=pass vectors=pass serialized_sha256=pass tamper_rejection=pass iverilog=pass protocol_4state=pass verilator=pass hygiene=pass'

_validate: oracle tamper-rejection iverilog verilator

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

clean:
	rm -rf "$(BUILD_DIR)"
