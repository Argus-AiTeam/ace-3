#!/usr/bin/env bash
set -euo pipefail
cd /home/argustest/ace3-argus
python3 - <<'PY'
import hashlib
import json
import re
import subprocess
from pathlib import Path

note_path = Path("docs/results/TRANSACTION009_MANAGER_TRANSITION_REQUEST.md")
gate_path = Path(
    "build/argus-audit/tx008-r5/authority-transition-gate-r1/"
    "authority-transition.json"
)
readiness_path = Path("build/cursor9_next_position_runtime_readiness/readiness.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


branch = subprocess.run(
    ["git", "branch", "--show-current"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert branch == "argus/full-projection", branch

gate = json.loads(gate_path.read_text())
readiness = json.loads(readiness_path.read_text())
note = note_path.read_text()

assert gate["status"] == "PASS"
assert gate["transaction008_prerequisite"] == {
    "manager_review_ready": True,
    "status": "PASS",
}
assert gate["manager_prohibition"]["status"] == "ACTIVE_UNCLEARED"
assert gate["manager_prohibition"]["clearance_record_present"] is False
assert gate["transaction009_transition"] == {
    "manager_clearance_required": True,
    "prohibition_preserved": True,
    "status": "NOT_READY",
}
assert gate["transaction009_025_artifacts"] == {
    "created_by_this_gate": False,
    "named_authority_execution_or_publication_paths_found": [],
    "status": "ABSENT",
}
assert readiness["status"] == "NOT_READY"
assert readiness["launchable"] is False
assert readiness["cursor"]["generation"] == 9
assert readiness["cursor"]["next_transaction_index"] == 9

future_name = re.compile(
    r"transaction-?(?:0?9|0?1[0-9]|0?2[0-5])(?:\D|$)",
    re.IGNORECASE,
)
offenders = sorted(
    str(path)
    for path in Path("build").rglob("*")
    if future_name.search(path.name)
)
assert offenders == [], offenders

exact_prohibition = (
    "No transaction008 workload until those gates PASS; no transaction009-025\n"
    "> authority or execution."
)
assert exact_prohibition in note
for required_text in (
    "**Status: NOT_READY (fail closed).**",
    "transaction008_prerequisite.status: PASS",
    "transaction009_025_artifacts.status: ABSENT",
    "explicitly replace or clear the quoted prohibition",
    "transaction009 transition remains **NOT_READY**",
    "artifact count must remain\nzero",
):
    assert required_text in note, required_text

print("PASS manager transition request")
print(f"branch={branch}")
print(f"gate_sha256={sha256(gate_path)}")
print(f"note_sha256={sha256(note_path)}")
print("transaction008_prerequisite=PASS")
print("transaction009_transition=NOT_READY")
print("transaction009_025_named_paths=0")
PY
