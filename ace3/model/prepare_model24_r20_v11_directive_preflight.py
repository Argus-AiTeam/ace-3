#!/usr/bin/env python3
"""Publish an immutable r20/V11 no-execution directive preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
CURRENT_MANAGER_CONTEXT = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/0c3116a2006b/latest.json"
)
V10_MANAGER_CONTEXT = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/v10bindr20/mission.json"
)
PROJECT_AUTHORITY = REPOSITORY / "chip-execution-authority.json"
ADOPTION_RECEIPT = (
    REPOSITORY
    / "build/model24_selected_token_position3_continuations"
    / "r20-project-authority-adoption/adoption-receipt.json"
)
R20_NONCE = "18357ce16f61ddae"
R20_TASK_ID = f"ace3-model24-r20-42c895c-{R20_NONCE}"
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
EXPECTED_HASHES = {
    "v10_manager_context": (
        "4c84b1848a3074928a389653477bc2e5750a76ded3b429506b612875c44cb5e5"
    ),
    "package_manifest": (
        "0ba86b47a27d137fff9da03ceaeb102d54f6bcdc794892d0c792f6690824f69a"
    ),
    "package_seal": (
        "7c404f49d15c10b359931be3a850ca116e62284251840acca382ef2587dee852"
    ),
    "review": (
        "0f5310d903dbcaf36f3f46c7e5c1c8bf09315dfd7c8f09cce742a75f9c7d2b78"
    ),
    "authority": (
        "4e8c5c94d573ac438c95342e795325a14ab938526f0e169b168ed6d7bf2ef829"
    ),
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
REVIEW_WORKLOAD_COUNTERS = {
    "execution_invocations",
    "launcher_invocations",
    "model_invocations",
    "real_controller_invocations",
    "real_model24_invocations",
    "real_rtl_simulator_invocations",
    "simulator_invocations",
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


def record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def require_hash(label: str, value: dict[str, Any]) -> None:
    require(
        value["sha256"] == EXPECTED_HASHES[label],
        f"{label} SHA256 differs",
    )


def require_bound_record(
    actual: object,
    expected: dict[str, Any],
    label: str,
    *,
    require_path: bool = True,
) -> None:
    require(isinstance(actual, dict), f"{label} record must be an object")
    keys = ("bytes", "sha256", "path") if require_path else ("bytes", "sha256")
    require(
        all(actual.get(key) == expected[key] for key in keys),
        f"{label} record differs",
    )


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


def absent(path: str) -> bool:
    return not os.path.lexists(path)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if os.path.lexists(path):
            path.chmod(0o600)
            path.unlink()
        raise


def checked_sources() -> dict[str, tuple[Path, bytes, dict[str, Any]]]:
    paths = {
        "current_manager_context": CURRENT_MANAGER_CONTEXT,
        "v10_manager_context": V10_MANAGER_CONTEXT,
        "project_authority": PROJECT_AUTHORITY,
        "adoption_receipt": ADOPTION_RECEIPT,
        "package_manifest": R20_PACKAGE_MANIFEST,
        "package_seal": R20_PACKAGE_SEAL,
        "review": R20_REVIEW,
        "authority": R20_AUTHORITY,
    }
    sources: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for label, path in paths.items():
        payload = read_regular(path)
        sources[label] = (path, payload, record(path, payload))
    for label in EXPECTED_HASHES:
        require_hash(label, sources[label][2])
    return sources


def validate_documents(
    sources: dict[str, tuple[Path, bytes, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    documents = {
        label: load_object(path, payload)
        for label, (path, payload, _) in sources.items()
    }
    current_context = documents["current_manager_context"]
    require(
        current_context.get("kind") == "mission_context"
        and current_context.get("mission_id") == "0c3116a2006b"
        and current_context.get("node_key")
        == "v11-r20-immutable-directive-preflight"
        and current_context.get("scope") == "bounded"
        and current_context.get("stage") == "rtl",
        "current V11 preflight Manager context differs",
    )

    v10_context = documents["v10_manager_context"]
    require(
        v10_context.get("kind") == "mission_context"
        and v10_context.get("mission_id") == "v10bindr20"
        and v10_context.get("node_key") == "r20-project-authority-binding",
        "immutable V10 Manager context differs",
    )

    manifest = documents["package_manifest"]
    require(
        manifest.get("kind") == "ace3_model24_r20_durable_launch_package"
        and manifest.get("identity", {}).get("nonce") == R20_NONCE
        and manifest.get("identity", {}).get("task_id") == R20_TASK_ID,
        "r20 package identity differs",
    )
    require_zero_map(
        manifest.get("zero_state"),
        PACKAGE_COUNTERS,
        "r20 package zero_state",
    )
    require(
        manifest.get("execution", {}).get("current_invocation_count") == 0,
        "r20 package reports an execution invocation",
    )

    review = documents["review"]
    require(
        review.get("kind") == "ace3_model24_r20_independent_l2_review"
        and review.get("producer_role") == "reviewer"
        and review.get("active_role") == "reviewer"
        and review.get("independent") is True
        and review.get("verdict") == "ACCEPT"
        and review.get("nonce") == R20_NONCE
        and review.get("task_id") == R20_TASK_ID,
        "independent r20 ACCEPT review differs",
    )
    test_results = review.get("test_results")
    require(isinstance(test_results, dict), "review test_results must be an object")
    require(
        all(test_results.get(key) == 0 for key in REVIEW_WORKLOAD_COUNTERS),
        "review reports nonzero workload counters",
    )

    authority = documents["authority"]
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
    require(
        authority.get("package_manifest_sha256")
        == EXPECTED_HASHES["package_manifest"]
        and authority.get("package_seal_sha256")
        == EXPECTED_HASHES["package_seal"]
        and authority.get("review_sha256") == EXPECTED_HASHES["review"],
        "accepted r20 authority evidence bindings differ",
    )

    adoption = documents["adoption_receipt"]
    require(
        adoption.get("kind")
        == "ace3_manager_v10_r20_project_authority_adoption_receipt"
        and adoption.get("status") == "PASS"
        and adoption.get("manager_revision") == "V10"
        and adoption.get("manager_status") == "MANAGER_RELEASE_ACCEPTED"
        and adoption.get("runtime_pass_status") == "RUNTIME_PASS_ACCEPTED"
        and adoption.get("authority_status") == "RUNTIME_PASS_ACCEPTED"
        and adoption.get("authority_consumed") is False
        and adoption.get("execution_authorized") is False,
        "V10 adoption receipt state differs",
    )
    require_bound_record(
        adoption.get("mission_context"),
        sources["v10_manager_context"][2],
        "V10 adoption Manager context",
        require_path=False,
    )
    chain = adoption.get("r20_chain")
    require(isinstance(chain, dict), "V10 adoption r20_chain must be an object")
    require(
        chain.get("nonce") == R20_NONCE and chain.get("task_id") == R20_TASK_ID,
        "V10 adoption r20 identity differs",
    )
    for field, source_label in (
        ("package_manifest", "package_manifest"),
        ("package_seal", "package_seal"),
        ("independent_review", "review"),
        ("authority", "authority"),
    ):
        require_bound_record(
            chain.get(field),
            sources[source_label][2],
            f"V10 adoption {field}",
        )
    zero_workload = adoption.get("zero_workload_boundary")
    require(
        isinstance(zero_workload, dict),
        "V10 adoption zero_workload_boundary must be an object",
    )
    require_zero_map(
        zero_workload.get("activity_counters"),
        PROJECT_COUNTERS,
        "V10 adoption activity_counters",
    )
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
    ):
        require(zero_workload.get(field) is False, f"V10 adoption {field} is true")

    project = documents["project_authority"]
    require_bound_record(
        adoption.get("postimage"),
        sources["project_authority"][2],
        "V10 adopted project-authority postimage",
    )
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
        "adopted project authority differs",
    )
    require_zero_map(
        project.get("activity_counters"),
        PROJECT_COUNTERS,
        "project activity_counters",
    )
    manager_release = project.get("manager_release")
    require(
        isinstance(manager_release, dict)
        and manager_release.get("manager_revision") == "V10"
        and manager_release.get("manager_status") == "MANAGER_RELEASE_ACCEPTED"
        and manager_release.get("runtime_pass_status") == "RUNTIME_PASS_ACCEPTED"
        and manager_release.get("adoption_only") is True
        and manager_release.get("authority_consumed") is False
        and manager_release.get("authority_consumption_authorized") is False
        and manager_release.get("durable_submission_authorized") is False
        and manager_release.get("execution_authorized") is False
        and manager_release.get("output_or_terminal_publication_authorized")
        is False
        and manager_release.get("transaction009_replay_authorized") is False
        and manager_release.get("workload_authorized") is False
        and manager_release.get("later_distinct_v11_execution_directive_required")
        is True,
        "project Manager release is not preserved V10 adoption-only state",
    )
    require_bound_record(
        manager_release.get("mission_context"),
        sources["v10_manager_context"][2],
        "project V10 Manager context",
        require_path=False,
    )
    for field, source_label in (
        ("manifest", "package_manifest"),
        ("seal", "package_seal"),
    ):
        require_bound_record(
            project.get("package", {}).get(field),
            sources[source_label][2],
            f"project package {field}",
        )
    require_bound_record(
        project.get("source_disjoint_independent_review"),
        sources["review"][2],
        "project independent review",
    )
    require_bound_record(
        project.get("authority"),
        sources["authority"][2],
        "project accepted authority",
    )

    namespaces = manifest.get("namespaces")
    require(isinstance(namespaces, dict), "r20 package namespaces must be an object")
    namespace_paths = {
        "execution.authority_consumed": namespaces.get("authority_consumed"),
        "execution.registry_root": namespaces.get("registry_root"),
        "execution.receipt": namespaces.get("receipt"),
        "execution.stderr_log": namespaces.get("stderr_log"),
        "execution.stdout_log": namespaces.get("stdout_log"),
        "output.root": namespaces.get("output"),
        "output.payload": namespaces.get("payload_output_dir"),
        "output.simulation": namespaces.get("simulation_dir"),
        "terminal.manifest": namespaces.get("terminal_manifest"),
        "terminal.root": namespaces.get("terminal_root"),
    }
    require(
        all(isinstance(path, str) and path for path in namespace_paths.values()),
        "r20 runtime namespace path is malformed",
    )
    namespace_absence = {
        label: absent(path) for label, path in namespace_paths.items()
    }
    require(
        all(namespace_absence.values()),
        "r20 execution/output/terminal namespace is present",
    )
    return documents, namespace_absence


def retained_sources_unchanged(
    sources: dict[str, tuple[Path, bytes, dict[str, Any]]],
) -> None:
    for label, (path, payload, _) in sources.items():
        require(read_regular(path) == payload, f"{label} changed during preflight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    require(
        output
        == REPOSITORY
        / "build/model24_selected_token_position3_continuations"
        / "r20-v11-directive-preflight",
        "noncanonical V11 preflight output path",
    )
    require(not os.path.lexists(output), f"preflight output already exists: {output}")
    require(
        output.parent.is_dir() and not output.parent.is_symlink(),
        f"preflight parent directory required: {output.parent}",
    )

    sources = checked_sources()
    documents, namespace_absence = validate_documents(sources)
    observed_at_unix_ns = time.time_ns()

    output.mkdir(mode=0o700)
    fsync_directory(output.parent)
    current_snapshot = output / "v11-preflight-manager-context.json"
    v10_snapshot = output / "v10-adoption-manager-context.json"
    write_exclusive(current_snapshot, sources["current_manager_context"][1])
    write_exclusive(v10_snapshot, sources["v10_manager_context"][1])
    current_snapshot_record = record(
        current_snapshot,
        sources["current_manager_context"][1],
    )
    v10_snapshot_record = record(v10_snapshot, sources["v10_manager_context"][1])

    project = documents["project_authority"]
    manifest = documents["package_manifest"]
    adoption = documents["adoption_receipt"]
    preflight = {
        "authorization_state": {
            "authority_consumed": False,
            "authority_consumption_authorized": False,
            "durable_submission_authorized": False,
            "execution_authority_issued": False,
            "execution_authorized": False,
            "output_or_terminal_publication_authorized": False,
            "preflight_only": True,
            "transaction009_replay_authorized": False,
            "workload_authorized": False,
        },
        "claim_boundary": (
            "This artifact authenticates readiness fields only. It does not issue "
            "Manager V11 execution authority and permits no authority consumption, "
            "submission, workload invocation, output, or terminal publication."
        ),
        "current_state": {
            "activity_counters": project["activity_counters"],
            "authority_consumed": False,
            "authority_state": project["authority_state"],
            "execution_authorized": False,
            "namespace_absence": namespace_absence,
            "observed_at_unix_ns": observed_at_unix_ns,
        },
        "future_v11_runtime_pass_authorization": {
            "current_status": "NOT_ISSUED",
            "required_exact_bindings": {
                "accepted_authority": sources["authority"][2],
                "accepted_package_manifest": sources["package_manifest"][2],
                "accepted_package_seal": sources["package_seal"][2],
                "accepted_review": sources["review"][2],
                "adopted_project_authority_preimage": sources[
                    "project_authority"
                ][2],
                "adoption_receipt": sources["adoption_receipt"][2],
                "execution_cardinality": 1,
                "launch_argv": project["authorized_launch"]["argv"],
                "launch_authority": project["authorized_launch"]["authority"],
                "launch_cwd": project["authorized_launch"]["cwd"],
                "launch_review": project["authorized_launch"]["review"],
                "manager_revision": "V11",
                "nonce": R20_NONCE,
                "preflight_manager_context": current_snapshot_record,
                "replay": False,
                "resume": False,
                "retry": False,
                "task_id": R20_TASK_ID,
                "v10_adoption_manager_context": v10_snapshot_record,
                "watcher": False,
            },
            "required_explicit_future_directive_fields": {
                "authority_consumption_authorized": True,
                "durable_submission_authorized": True,
                "execution_authorized": True,
                "manager_context_bytes": "REQUIRED_NEW_IMMUTABLE_V11_DIRECTIVE_BYTES",
                "manager_context_sha256": "REQUIRED_SHA256_OF_EXACT_V11_DIRECTIVE_BYTES",
                "output_or_terminal_publication_authorized": True,
                "preflight_bytes": "REQUIRED_EXACT_PREFLIGHT_BYTES",
                "preflight_sha256": "REQUIRED_SHA256_FROM_VALIDATION_LOG",
                "workload_authorized": True,
            },
        },
        "immutable_provenance": {
            "accepted_authority": sources["authority"][2],
            "accepted_package_manifest": sources["package_manifest"][2],
            "accepted_package_seal": sources["package_seal"][2],
            "accepted_review": sources["review"][2],
            "adopted_project_authority": sources["project_authority"][2],
            "adoption_receipt": sources["adoption_receipt"][2],
            "v10_adoption_manager_context": v10_snapshot_record,
            "v11_preflight_manager_context": current_snapshot_record,
        },
        "kind": "ace3_model24_r20_v11_no_execution_directive_preflight",
        "manager_revision": "V11",
        "nonce": R20_NONCE,
        "package_zero_state": manifest["zero_state"],
        "r20_adoption_status": {
            "authority_status": adoption["authority_status"],
            "manager_revision": adoption["manager_revision"],
            "manager_status": adoption["manager_status"],
            "review_status": project["reviewed_binding"]["review_status"],
            "runtime_pass_status": adoption["runtime_pass_status"],
            "status": adoption["status"],
        },
        "schema_version": 1,
        "status": "PASS",
        "task_id": R20_TASK_ID,
    }
    preflight_path = output / "preflight.json"
    preflight_payload = canonical_json(preflight)
    write_exclusive(preflight_path, preflight_payload)
    preflight_record = record(preflight_path, preflight_payload)

    retained_sources_unchanged(sources)
    _, final_namespace_absence = validate_documents(sources)
    require(
        final_namespace_absence == namespace_absence,
        "runtime namespace state changed during preflight",
    )
    validation_lines = [
        "ACE3_R20_V11_NO_EXECUTION_PREFLIGHT PASS",
        f"preflight_path={preflight_path}",
        f"preflight_bytes={preflight_record['bytes']}",
        f"preflight_sha256={preflight_record['sha256']}",
        f"manager_context_path={current_snapshot}",
        f"manager_context_bytes={current_snapshot_record['bytes']}",
        f"manager_context_sha256={current_snapshot_record['sha256']}",
        "authority_consumed=false",
        "execution_authorized=false",
        "workload_counters=all_zero",
        "execution_namespaces=absent",
        "output_namespaces=absent",
        "terminal_namespaces=absent",
        "execution_authority_issued=false",
    ]
    validation_path = output / "validation.log"
    write_exclusive(
        validation_path,
        ("\n".join(validation_lines) + "\n").encode("ascii"),
    )
    fsync_directory(output)
    for path in output.iterdir():
        path.chmod(0o400)
    output.chmod(0o500)
    fsync_directory(output.parent)
    print(
        "ACE3_R20_V11_NO_EXECUTION_PREFLIGHT_PASS "
        f"artifact={preflight_path} sha256={preflight_record['sha256']} "
        "authority_consumed=false execution_authorized=false "
        "workload_counters=all_zero namespaces=absent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
