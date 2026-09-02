#!/usr/bin/env bash
set -euo pipefail
cd /home/argustest/ace3-argus

git rm --cached -- \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/README.md \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/autoawq-v0.2.9-packing_utils.py \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/config.json \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/model-api.json \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/sample-model_layers_0_self_attn_q_proj-qweight.bin \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/sample-model_layers_0_self_attn_q_proj-qzeros.bin \
    ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj/sample-model_layers_0_self_attn_q_proj-scales.bin \
    ace3/model/tests/test_tracked_source_paths.py
git add -- docs/results/TRANSACTION009_INPUT_BOUNDARY_READINESS.correction.command
git diff --cached --check
git commit -m "chore: isolate transaction009 readiness commit" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git rev-parse HEAD
