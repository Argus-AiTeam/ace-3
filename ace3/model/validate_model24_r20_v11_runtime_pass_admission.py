#!/usr/bin/env python3
"""Fail-closed validation for the r20/V11 runtime-pass admission candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
R20_NONCE = "18357ce16f61ddae"
R20_TASK_ID = f"ace3-model24-r20-42c895c-{R20_NONCE}"
RECEIPT_CHAIN_TASK_ID = "5f46912a531a"
REVIEWED_SOURCE_COMMIT = "72ce3acf6ae88806c17a39c9c68a1a32928cfcbf"
REVIEWED_SOURCE_TREE = "94130048e117496f3d0c60f687e902ef4b501c26"
CANONICAL_OUTPUT = (
    REPOSITORY
    / "build/model24_selected_token_position3_continuations"
    / "r20-v11-runtime-pass-admission"
)
MANAGER_DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "active_manager_directive.json"
)
PREFLIGHT = (
    REPOSITORY
    / "build/model24_selected_token_position3_continuations"
    / "r20-v11-directive-preflight/preflight.json"
)
PROJECT_AUTHORITY = REPOSITORY / "chip-execution-authority.json"
ADOPTION_RECEIPT = (
    REPOSITORY
    / "build/model24_selected_token_position3_continuations"
    / "r20-project-authority-adoption/adoption-receipt.json"
)
RECEIPT_CHAIN_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/5f46912a531a/round-0002.json"
)
RECEIPT_CHAIN_MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/5f46912a531a/mission.json"
)
R20_PACKAGE_ROOT = Path(
    f"/home/argustest/ace3-model24-r20-prep-20260829-{R20_NONCE}/package"
)
R20_PACKAGE_MANIFEST = R20_PACKAGE_ROOT / "package.json"
R20_PACKAGE_SEAL = R20_PACKAGE_ROOT / "seal.json"
R20_REVIEW = Path(
    f"/home/argustest/ace3-model24-r20-review-20260829-{R20_NONCE}/review.json"
)
R20_AUTHORITY = Path(
    f"/home/argustest/ace3-model24-r20-authority-20260829-{R20_NONCE}.json"
)
LAUNCH_CWD = (
    f"/home/argustest/ace3-model24-r20-prep-20260829-{R20_NONCE}/source"
)
LAUNCH_ARGV = [
    "/home/argustest/miniconda3/bin/python3",
    "-B",
    (
        f"/home/argustest/ace3-model24-r20-prep-20260829-{R20_NONCE}"
        "/package/lifecycle.py"
    ),
    "launch",
    "--authority",
    str(R20_AUTHORITY),
    "--review",
    str(R20_REVIEW),
]
SNAPSHOT_SOURCES = {
    "manager_v11_directive": MANAGER_DIRECTIVE,
    "v11_no_execution_preflight": PREFLIGHT,
    "adopted_project_authority_preimage": PROJECT_AUTHORITY,
    "v10_adoption_receipt": ADOPTION_RECEIPT,
    "receipt_chain_review": RECEIPT_CHAIN_REVIEW,
    "receipt_chain_mission": RECEIPT_CHAIN_MISSION,
}
SNAPSHOT_PATHS = {
    "manager_v11_directive": "snapshots/manager-v11-directive.json",
    "v11_no_execution_preflight": "snapshots/v11-no-execution-preflight.json",
    "adopted_project_authority_preimage": (
        "snapshots/adopted-project-authority-preimage.json"
    ),
    "v10_adoption_receipt": "snapshots/v10-adoption-receipt.json",
    "receipt_chain_review": "snapshots/receipt-chain-review.json",
    "receipt_chain_mission": "snapshots/receipt-chain-mission.json",
}
EXPECTED_HASHES = {
    "manager_v11_directive": (
        "081f9179323eb86435f6873c5dc25e99f3efa1bcd2682cefac87b27858e65c80"
    ),
    "v11_no_execution_preflight": (
        "27ef9d1251a9c3254b4aa9f76a5f0c5c32dda05aa2e7f7eb23de53ffa1927eea"
    ),
    "adopted_project_authority_preimage": (
        "f775dd8fb4e3d911b4943fdf04d5c0144e11888147927e84f4abcb9c1aa3cdff"
    ),
    "v10_adoption_receipt": (
        "311fb3d5be679b0e7c5d766ba636221864ec5fc41e4f4a378b391b7673b90e3c"
    ),
    "receipt_chain_review": (
        "1f44427bb5fca218b27b4881a23512cea1062c76a0d45b6cd57bd63363f1102b"
    ),
    "receipt_chain_mission": (
        "bf37f3af2cced055d0e9daf257dd9594aaa4313628772b679c79358592700c57"
    ),
    "package_manifest": (
        "0ba86b47a27d137fff9da03ceaeb102d54f6bcdc794892d0c792f6690824f69a"
    ),
    "package_seal": (
        "7c404f49d15c10b359931be3a850ca116e62284251840acca382ef2587dee852"
    ),
    "independent_review": (
        "0f5310d903dbcaf36f3f46c7e5c1c8bf09315dfd7c8f09cce742a75f9c7d2b78"
    ),
    "accepted_authority": (
        "4e8c5c94d573ac438c95342e795325a14ab938526f0e169b168ed6d7bf2ef829"
    ),
}
EXTERNAL_SOURCES = {
    "package_manifest": R20_PACKAGE_MANIFEST,
    "package_seal": R20_PACKAGE_SEAL,
    "independent_review": R20_REVIEW,
    "accepted_authority": R20_AUTHORITY,
}
PROJECT_COUNTERS = {
    "authority_consumption",
    "authority_issuance",
    "controller_invocation",
    "durable_submission",
    "lifecycle_invocation",
    "model_execution",
    "oracle_execution",
    "payload_execution",
    "rtl_compile",
    "rtl_simulation",
    "runtime_mutation",
    "subagent_submission",
    "transaction_execution",
    "transaction_replay",
    "transaction_resume",
    "transaction_retry",
    "vector_generation",
}
PACKAGE_COUNTERS = {
    "authority_consumptions",
    "authority_creations",
    "durable_submissions",
    "execution_invocations",
    "launcher_invocations",
    "model_invocations",
    "output_creations",
    "payload_output_directory_creations",
    "real_controller_invocations",
    "real_model24_invocations",
    "real_rtl_simulator_invocations",
    "receipt_creations",
    "simulation_directory_creations",
    "simulator_invocations",
    "terminal_manifest_creations",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular(path: Path) -> bytes:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink forbidden: {path}")
    return path.read_bytes()


def load_object(path: Path, payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def source_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "path": str(path),
        "sha256": sha256(payload),
    }


def snapshot_record(
    candidate: Path,
    relative_path: str,
    source_path: Path,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "path": relative_path,
        "sha256": sha256(payload),
        "source_path": str(source_path),
    }


def require_zero_map(
    value: object,
    expected_keys: set[str],
    label: str,
) -> dict[str, int]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == expected_keys, f"{label} fields differ")
    require(
        all(type(counter) is int and counter == 0 for counter in value.values()),
        f"{label} contains nonzero or non-integer values",
    )
    return value


def checked_source_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for label, path in {**SNAPSHOT_SOURCES, **EXTERNAL_SOURCES}.items():
        payload = read_regular(path)
        require(
            sha256(payload) == EXPECTED_HASHES[label],
            f"{label} SHA256 differs",
        )
        payloads[label] = payload
    return payloads


def validate_reviewed_source_commit() -> None:
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "rev-parse",
            f"{REVIEWED_SOURCE_COMMIT}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        commit.returncode == 0
        and commit.stdout.strip() == REVIEWED_SOURCE_COMMIT,
        "reviewed receipt-chain source commit unavailable",
    )
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "rev-parse",
            f"{REVIEWED_SOURCE_COMMIT}^{{tree}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        tree.returncode == 0 and tree.stdout.strip() == REVIEWED_SOURCE_TREE,
        "reviewed receipt-chain source tree differs",
    )


def _validate_snapshot_records(
    candidate: Path,
    admission: dict[str, Any],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    records = admission.get("snapshots")
    require(
        isinstance(records, dict) and set(records) == set(SNAPSHOT_PATHS),
        "admission snapshot set differs",
    )
    payloads: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for label, relative_path in SNAPSHOT_PATHS.items():
        require(
            records[label].get("path") == relative_path
            and records[label].get("source_path")
            == str(SNAPSHOT_SOURCES[label]),
            f"{label} snapshot path differs",
        )
        path = candidate / relative_path
        payload = read_regular(path)
        expected_record = snapshot_record(
            candidate,
            relative_path,
            SNAPSHOT_SOURCES[label],
            payload,
        )
        require(records[label] == expected_record, f"{label} record differs")
        require(
            expected_record["sha256"] == EXPECTED_HASHES[label],
            f"{label} SHA256 differs",
        )
        payloads[label] = payload
        documents[label] = load_object(path, payload)
    return payloads, documents


def _validate_directive(directive: dict[str, Any]) -> None:
    require(
        set(directive)
        == {
            "objective_sha256",
            "revision",
            "set_at",
            "source",
            "text",
            "version",
        }
        and directive["version"] == 1
        and directive["source"]
        == "manager.autonomous-supervision.ace3-r20-exact-one-shot-execution-v11",
        "Manager V11 directive envelope differs",
    )
    text = directive["text"]
    require(isinstance(text, str), "Manager V11 directive text is malformed")
    for binding in (
        "ACE-3 MANAGER V11",
        R20_NONCE,
        EXPECTED_HASHES["package_manifest"],
        EXPECTED_HASHES["package_seal"],
        EXPECTED_HASHES["independent_review"],
        EXPECTED_HASHES["accepted_authority"],
        EXPECTED_HASHES["adopted_project_authority_preimage"],
        EXPECTED_HASHES["v10_adoption_receipt"],
        EXPECTED_HASHES["v11_no_execution_preflight"],
        RECEIPT_CHAIN_TASK_ID,
        "72ce3ac",
        f"Exact cwd: {LAUNCH_CWD}.",
        "EXECUTION_CARDINALITY=1",
        "RETRY=false",
        "REPLAY=false",
        "RESUME=false",
        "WATCHER=false",
    ):
        require(binding in text, f"Manager V11 directive missing {binding}")
    require(
        " ".join(LAUNCH_ARGV) in text,
        "Manager V11 directive launch argv differs",
    )


def _validate_preflight(preflight: dict[str, Any]) -> None:
    require(
        preflight.get("kind")
        == "ace3_model24_r20_v11_no_execution_directive_preflight"
        and preflight.get("schema_version") == 1
        and preflight.get("status") == "PASS"
        and preflight.get("manager_revision") == "V11"
        and preflight.get("nonce") == R20_NONCE
        and preflight.get("task_id") == R20_TASK_ID,
        "V11 no-execution preflight identity differs",
    )
    state = preflight.get("authorization_state")
    require(
        isinstance(state, dict)
        and state.get("preflight_only") is True
        and all(
            state.get(field) is False
            for field in (
                "authority_consumed",
                "authority_consumption_authorized",
                "durable_submission_authorized",
                "execution_authority_issued",
                "execution_authorized",
                "output_or_terminal_publication_authorized",
                "transaction009_replay_authorized",
                "workload_authorized",
            )
        ),
        "V11 preflight no-execution state differs",
    )
    require(
        all(preflight["current_state"]["namespace_absence"].values()),
        "V11 preflight records a present runtime namespace",
    )
    require_zero_map(
        preflight["current_state"]["activity_counters"],
        PROJECT_COUNTERS,
        "V11 preflight activity_counters",
    )


def _validate_project_preimage(project: dict[str, Any]) -> None:
    require(
        project.get("kind") == "ace3_project_chip_execution_authority"
        and project.get("revision") == "r20"
        and project.get("nonce") == R20_NONCE
        and project.get("task_id") == R20_TASK_ID
        and project.get("manager_revision") == "V10"
        and project.get("manager_status") == "MANAGER_RELEASE_ACCEPTED"
        and project.get("runtime_pass_status") == "RUNTIME_PASS_ACCEPTED"
        and project.get("authority_status") == "RUNTIME_PASS_ACCEPTED"
        and project.get("authority_consumed") is False
        and project.get("execution_authorized") is False,
        "adopted project-authority preimage differs",
    )
    require_zero_map(
        project.get("activity_counters"),
        PROJECT_COUNTERS,
        "project-authority preimage activity_counters",
    )


def _validate_adoption(adoption: dict[str, Any]) -> None:
    require(
        adoption.get("kind")
        == "ace3_manager_v10_r20_project_authority_adoption_receipt"
        and adoption.get("status") == "PASS"
        and adoption.get("manager_revision") == "V10"
        and adoption.get("manager_status") == "MANAGER_RELEASE_ACCEPTED"
        and adoption.get("runtime_pass_status") == "RUNTIME_PASS_ACCEPTED"
        and adoption.get("authority_status") == "RUNTIME_PASS_ACCEPTED"
        and adoption.get("authority_consumed") is False
        and adoption.get("execution_authorized") is False
        and adoption.get("postimage", {}).get("sha256")
        == EXPECTED_HASHES["adopted_project_authority_preimage"],
        "V10 adoption receipt differs",
    )
    zero = adoption.get("zero_workload_boundary")
    require(isinstance(zero, dict), "V10 adoption zero-workload boundary missing")
    require_zero_map(
        zero.get("activity_counters"),
        PROJECT_COUNTERS,
        "V10 adoption activity_counters",
    )
    require(
        all(
            zero.get(field) is False
            for field in (
                "authority_consumption_performed",
                "controller_invocation_performed",
                "durable_submission_performed",
                "lifecycle_invocation_performed",
                "model_invocation_performed",
                "output_namespace_present",
                "rtl_invocation_performed",
                "subagent_submission_performed",
                "terminal_namespace_present",
                "transaction009_replayed",
            )
        ),
        "V10 adoption reports runtime activity",
    )


def _validate_receipt_chain_review(
    review: dict[str, Any],
    mission: dict[str, Any],
) -> None:
    require(
        review.get("kind") == "round_reviewed_handoff"
        and review.get("mission_id") == RECEIPT_CHAIN_TASK_ID
        and review.get("producer_role") == "reviewer"
        and review.get("review", {}).get("status") == "done"
        and review.get("review", {}).get("reason")
        == "requested outcome is materially complete",
        "accepted receipt-chain review differs",
    )
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == RECEIPT_CHAIN_TASK_ID
        and mission.get("node_key") == "model24-receipt-chain-validation-target"
        and "no authority creation or consumption" in mission.get("objective", "")
        and "Focused make/test target passes" in mission.get(
            "acceptance_check", ""
        ),
        "receipt-chain mission contract differs",
    )


def _validate_external_bindings(
    admission: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = admission.get("accepted_r20_bindings")
    require(
        isinstance(records, dict) and set(records) == set(EXTERNAL_SOURCES),
        "accepted r20 binding set differs",
    )
    documents: dict[str, dict[str, Any]] = {}
    for label, path in EXTERNAL_SOURCES.items():
        payload = read_regular(path)
        actual = source_record(path, payload)
        require(actual == records[label], f"{label} binding record differs")
        require(
            actual["sha256"] == EXPECTED_HASHES[label],
            f"{label} SHA256 differs",
        )
        documents[label] = load_object(path, payload)
    manifest = documents["package_manifest"]
    require(
        manifest.get("kind") == "ace3_model24_r20_durable_launch_package"
        and manifest.get("identity", {}).get("nonce") == R20_NONCE
        and manifest.get("identity", {}).get("task_id") == R20_TASK_ID
        and manifest.get("execution", {}).get("current_invocation_count") == 0,
        "accepted r20 package identity or invocation state differs",
    )
    require_zero_map(
        manifest.get("zero_state"),
        PACKAGE_COUNTERS,
        "accepted r20 package zero_state",
    )
    review = documents["independent_review"]
    require(
        review.get("kind") == "ace3_model24_r20_independent_l2_review"
        and review.get("producer_role") == "reviewer"
        and review.get("independent") is True
        and review.get("verdict") == "ACCEPT"
        and review.get("nonce") == R20_NONCE
        and review.get("task_id") == R20_TASK_ID,
        "accepted r20 independent review differs",
    )
    authority = documents["accepted_authority"]
    require(
        authority.get("kind")
        == "ace3_model24_r20_manager_exactly_once_authority"
        and authority.get("nonce") == R20_NONCE
        and authority.get("task_id") == R20_TASK_ID
        and authority.get("execution_cardinality") == 1
        and authority.get("manager_exactly_once_directive") is True
        and authority.get("replay") is False
        and authority.get("resume") is False
        and authority.get("retry") is False
        and authority.get("watcher") is False,
        "accepted r20 one-shot authority differs",
    )
    namespaces = manifest.get("namespaces")
    require(isinstance(namespaces, dict), "r20 namespace map missing")
    for field in (
        "authority_consumed",
        "registry_root",
        "receipt",
        "stderr_log",
        "stdout_log",
        "output",
        "payload_output_dir",
        "simulation_dir",
        "terminal_manifest",
        "terminal_root",
    ):
        path = namespaces.get(field)
        require(
            isinstance(path, str) and path and not os.path.lexists(path),
            f"r20 runtime namespace present: {field}",
        )
    return documents


def validate_candidate(candidate: Path) -> dict[str, Any]:
    candidate = candidate.resolve(strict=True)
    require(candidate.is_dir() and not candidate.is_symlink(), "candidate directory required")
    admission_path = candidate / "admission.json"
    admission = load_object(admission_path, read_regular(admission_path))
    require(
        admission.get("schema_version") == 1
        and admission.get("kind")
        == "ace3_model24_r20_v11_runtime_pass_admission_candidate"
        and admission.get("status") == "AWAITING_INDEPENDENT_REVIEW"
        and admission.get("manager_revision") == "V11"
        and admission.get("nonce") == R20_NONCE
        and admission.get("task_id") == R20_TASK_ID,
        "runtime-pass admission identity differs",
    )
    _, documents = _validate_snapshot_records(candidate, admission)
    _validate_directive(documents["manager_v11_directive"])
    _validate_preflight(documents["v11_no_execution_preflight"])
    _validate_project_preimage(
        documents["adopted_project_authority_preimage"]
    )
    _validate_adoption(documents["v10_adoption_receipt"])
    _validate_receipt_chain_review(
        documents["receipt_chain_review"],
        documents["receipt_chain_mission"],
    )
    validate_reviewed_source_commit()
    source = admission.get("receipt_chain_validation")
    require(
        source
        == {
            "accepted": True,
            "review_task_id": RECEIPT_CHAIN_TASK_ID,
            "reviewed_source_commit": REVIEWED_SOURCE_COMMIT,
            "reviewed_source_tree": REVIEWED_SOURCE_TREE,
        },
        "receipt-chain validation binding differs",
    )
    _validate_external_bindings(admission)
    require(
        admission.get("launch")
        == {
            "argv": LAUNCH_ARGV,
            "authority": str(R20_AUTHORITY),
            "cwd": LAUNCH_CWD,
            "review": str(R20_REVIEW),
        },
        "runtime-pass admission launch tuple differs",
    )
    require(
        admission.get("one_shot_controls")
        == {
            "execution_cardinality": 1,
            "replay": False,
            "resume": False,
            "retry": False,
            "watcher": False,
        },
        "runtime-pass admission one-shot controls differ",
    )
    require(
        admission.get("authorization_state")
        == {
            "admission_accepted": False,
            "authority_consumed": False,
            "authority_consumption_authorized": False,
            "durable_submission_authorized": False,
            "execution_authorized": False,
            "output_or_terminal_publication_authorized": False,
            "project_authority_updated": False,
            "workload_authorized": False,
        },
        "runtime-pass admission authorization state differs",
    )
    require_zero_map(
        admission.get("activity_counters"),
        PROJECT_COUNTERS,
        "runtime-pass admission activity_counters",
    )
    require_zero_map(
        admission.get("package_zero_state"),
        PACKAGE_COUNTERS,
        "runtime-pass admission package_zero_state",
    )
    return {
        "activity_counters": "all_zero",
        "admission_status": admission["status"],
        "authority_consumed": False,
        "execution_authorized": False,
        "launch_tuple": "exact",
        "manager_v11_directive": "exact_bytes",
        "namespaces": "absent",
        "one_shot_controls": "exact",
        "preflight": "exact",
        "receipt_chain_validation": "accepted_and_source_bound",
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args()
    result = validate_candidate(arguments.candidate)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
