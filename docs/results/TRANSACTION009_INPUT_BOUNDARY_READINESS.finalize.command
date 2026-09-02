#!/usr/bin/env bash
set -euo pipefail
cd /home/argustest/ace3-argus

base=docs/results/TRANSACTION009_INPUT_BOUNDARY_READINESS
set +e
bash "${base}.validation.command" >"${base}.validation.output" 2>"${base}.validation.stderr"
status=$?
set -e
printf '%s\n' "${status}" >"${base}.validation.status"
cat "${base}.validation.output"
if [[ -s "${base}.validation.stderr" ]]; then
    cat "${base}.validation.stderr" >&2
fi
if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

git diff --check -- \
    "${base}.md" \
    "${base}.validation.command" \
    "${base}.validation.output" \
    "${base}.validation.stderr" \
    "${base}.validation.status" \
    "${base}.finalize.command"
git status --short
git add -- \
    "${base}.md" \
    "${base}.validation.command" \
    "${base}.validation.output" \
    "${base}.validation.stderr" \
    "${base}.validation.status" \
    "${base}.finalize.command"
git diff --cached --check
git commit -m "docs: record transaction009 input readiness" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git rev-parse HEAD
