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
	projection-verilator-simulation clean

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
	  test ! -e "$(ROOT)/ace3/generated"; \
	  printf '%s\n' 'REPOSITORY_HYGIENE_PASS status_unchanged=yes build_ignored=yes legacy_generated_absent=yes diff_check=pass'; \
	} > "$$log" 2>&1 || { status=$$?; cat "$$log"; exit $$status; }; \
	cat "$$log"
	@printf '%s\n' 'STANDALONE_VALIDATION_PASS semantic_checks=fresh primitive=pass projection=pass serialized_sha256=pass tamper_rejection=pass iverilog=pass protocol_4state=pass verilator=pass geometry_parameters=pass hygiene=pass'

_validate: oracle tamper-rejection iverilog verilator projection

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

clean:
	rm -rf "$(BUILD_DIR)"
