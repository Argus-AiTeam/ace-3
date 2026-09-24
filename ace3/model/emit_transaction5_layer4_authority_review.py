#!/usr/bin/env python3
"""Source-disjoint review of transaction005/layer04 Manager authority."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping


ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "cd5b6c869531"
PACKAGE_MISSION_ID = "43c0b6ae6c52"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
PACKAGE_ID = "generation5-cursor5-checkpoint004-transaction005-layer04-r1"

RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATIONS = ADOPTION / "state-generations"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION6 = GENERATIONS / "generation-0000000006"
GENERATION6_STAGING = GENERATIONS / ".generation-0000000006.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION5 = TRANSACTIONS / "transaction-005"
FUTURE_ROOT = RUNTIME / "transaction5-authoritative-generation6"

PACKAGE_ROOT = (
    ROOT / "build/model24_selected_token_position3_transaction5_packages" / PACKAGE_ID
)
PACKAGE_SEAL = PACKAGE_ROOT / "package-seal.json"
PACKAGE_MANIFEST = PACKAGE_ROOT / "package-manifest.json"
SOURCE_MANIFEST = PACKAGE_ROOT / "source-manifest.json"
BASELINE = PACKAGE_ROOT / "authoritative-baseline.json"
PACKAGE_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_reviews"
    / PACKAGE_ID
    / "independent-review.json"
)
AUTHORITY_PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_authorities"
    / PACKAGE_ID
)
AUTHORITY = AUTHORITY_PACKAGE / "manager-authority.json"
AUTHORITY_SEAL = AUTHORITY_PACKAGE / "authority-seal.json"
AUTHORITY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_authority_reviews"
    / PACKAGE_ID
    / "independent-review.json"
)
MANAGER_MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/cd5b6c869531/mission.json"
)
PACKAGE_HOST_DECISION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/43c0b6ae6c52/round-0001.json"
)

TRANSACTION_INDEX = 5
LAYER_INDEX = 4
PERMITTED_INDICES = [TRANSACTION_INDEX]
FORBIDDEN_INDICES = [*range(5), *range(6, 26)]

ZERO_COUNTERS = {
    "authority_issuance": 0,
    "authority_consumption": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "payload_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "runtime_mutation": 0,
    "submission": 0,
    "transaction_execution": 0,
    "vector_generation": 0,
}

FIXED_SHA256 = {
    PACKAGE_SEAL: "93d54e75d1dcb2316343770bda7bf404d4f61af9a42818cc28d11dd92e34c262",
    PACKAGE_MANIFEST: "a777568726ae600b946f9fb236dac45292bce003ace734c85eae22a404ce622e",
    SOURCE_MANIFEST: "c395c5458f905d967225095c90376509e5922529abde8cdc46b4f988e504ad24",
    BASELINE: "8fb8c69a68686b08187aceef926da115e2072510c208cf3289809e2be262798f",
    PACKAGE_REVIEW: "60a2c22eaa95942629329e7c00e5b8779f33d72a6c9f8219fd37f6382e1964cb",
    MANAGER_MISSION: "da79162cb2ea91de14649c042aa9d9a19cb1f1a61b9cb042b7647c7a8412c86d",
    PACKAGE_HOST_DECISION: "f3b3027d091b8be939b43a2bd11da12ab31852da08407049849a8803ecab7e24",
    POINTER: "adee838870e758a5cc32a1a79248938b22d8e9d701501dce525c9098958bdda0",
    GENERATION5
    / "generation-manifest.json": "03f73918812f160cb4b9b7f94b239c5d3efb9cab9d4d41bd621a59eadfc9b599",
    GENERATION5
    / "ledger.json": "ca9397fc658ffe0629c8ba4cade0af5640d85298106b4dfc0dfbeb2ca039af14",
    GENERATION5
    / "checkpoints/transaction-004.json": "ff157b01b83d07d8bf167c420115659072a54415054454674625503d7decd610",
    TRANSACTIONS
    / "transaction-004/position004.state": "bc17b1015874eba58edd49c6b29c74b37057576823711f5983476264de0c65ac",
}

AUTHORITY_MEMBERS = {
    "manager-authority.json",
    "review-emitter.py",
    "review-request.json",
}
FORBIDDEN_IMPORTS = {
    "fcntl",
    "importlib",
    "numpy",
    "safetensors",
    "shutil",
    "subprocess",
    "torch",
}
FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "execute_exact_layer_transaction",
    "execute_layer_transaction",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "system",
}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in document, f"duplicate JSON key {key}: {path}")
            document[key] = value
        return document

    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate(
    record: Mapping[str, Any],
    label: str,
    expected_path: Path | None = None,
) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"}
        and isinstance(record.get("path"), str)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str),
        f"{label} record is malformed",
    )
    path = Path(record["path"])
    if expected_path is not None:
        require(path == expected_path, f"{label} path differs")
    require(file_record(path) == dict(record), f"{label} content differs")


def require_fixed(path: Path) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == FIXED_SHA256[path], f"fixed hash differs: {path}")
    return record


def source_boundary(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(
                    alias.name.split(".", 1)[0] not in FORBIDDEN_IMPORTS
                    for alias in node.names
                ),
                "reviewer imports workload or process support",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                (node.module or "").split(".", 1)[0] not in FORBIDDEN_IMPORTS,
                "reviewer imports workload or process support",
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            require(name.lower() not in FORBIDDEN_CALLS, f"forbidden call: {name}")


def manager_provenance() -> dict[str, Any]:
    record = require_fixed(MANAGER_MISSION)
    mission = load_json(MANAGER_MISSION)
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx005-layer04-authority"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and "manager-direct" in mission.get("tags", []),
        "Manager mission provenance differs",
    )
    return {
        "mission": record,
        "mission_id": MISSION_ID,
        "manager_direct": True,
    }


def authoritative_parent() -> dict[str, Any]:
    return {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(GENERATION5 / "generation-manifest.json"),
        "ledger": file_record(GENERATION5 / "ledger.json"),
        "checkpoint004": file_record(
            GENERATION5 / "checkpoints/transaction-004.json"
        ),
        "position004_input_state": file_record(
            TRANSACTIONS / "transaction-004/position004.state"
        ),
    }


def validate_accepted_boundary() -> dict[str, Any]:
    for path in FIXED_SHA256:
        require_fixed(path)
    manifest = load_json(PACKAGE_MANIFEST)
    review = load_json(PACKAGE_REVIEW)
    host_decision = load_json(PACKAGE_HOST_DECISION)
    require(
        manifest.get("kind")
        == "ace3_position3_transaction5_layer4_executor_package"
        and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("mission_id") == PACKAGE_MISSION_ID
        and manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and manifest.get("authoritative_generation") == 5
        and manifest.get("authoritative_cursor") == 5
        and manifest.get("required_parent_checkpoint_index") == 4
        and manifest.get("transaction_index") == TRANSACTION_INDEX
        and manifest.get("layer_index") == LAYER_INDEX
        and manifest.get("permitted_transaction_indices") == PERMITTED_INDICES
        and manifest.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and manifest.get("transaction005_output_namespace") == str(TRANSACTION5)
        and manifest.get("generation6_output_namespace") == str(GENERATION6)
        and manifest.get("execution_authorized") is False
        and manifest.get("authority_created") is False
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "accepted package differs",
    )
    require(
        review.get("kind")
        == "ace3_position3_transaction5_layer4_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == PACKAGE_MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("permitted_launch") == manifest["launch"]
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction005_executed") is False
        and review.get("generation6_published") is False
        and review.get("transactions006_025_absent") is True
        and review.get("this_review_authorizes_execution") is False,
        "accepted package review differs",
    )
    require(
        host_decision.get("kind") == "round_reviewed_handoff"
        and host_decision.get("mission_id") == PACKAGE_MISSION_ID
        and host_decision.get("producer_role") == "reviewer"
        and host_decision.get("review", {}).get("status") == "done",
        "Host-saved package acceptance differs",
    )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION5 / "ledger.json")
    checkpoint = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    require(
        pointer.get("generation") == 5
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation_manifest")
        == file_record(GENERATION5 / "generation-manifest.json")
        and ledger.get("state_generation") == 5
        and ledger.get("next_transaction_index") == 5
        and ledger.get("completed_transaction_count") == 5
        and checkpoint.get("transaction_index") == 4
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("outputs", {}).get("state")
        == file_record(TRANSACTIONS / "transaction-004/position004.state"),
        "authoritative generation5 parent differs",
    )
    return manifest


def expected_accepted_package() -> dict[str, Any]:
    return {
        "root": str(PACKAGE_ROOT),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "authoritative_baseline": file_record(BASELINE),
        "independent_review": file_record(PACKAGE_REVIEW),
        "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
    }


def validate_authority_document(
    authority: Mapping[str, Any],
    *,
    authenticate_records: bool = True,
    workload_call: Callable[[], None] | None = None,
) -> None:
    del workload_call
    manifest = load_json(PACKAGE_MANIFEST)
    require(
        authority.get("schema_version") == 1
        and authority.get("kind")
        == "ace3_position3_transaction5_layer4_manager_authorization"
        and authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("producer_role") == "manager"
        and authority.get("mission_id") == MISSION_ID
        and authority.get("manager_provenance") == manager_provenance()
        and authority.get("authority_cardinality") == 1
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("authoritative_generation") == 5
        and authority.get("authoritative_cursor") == 5
        and authority.get("required_parent_checkpoint_index") == 4
        and authority.get("transaction_index") == TRANSACTION_INDEX
        and authority.get("layer_index") == LAYER_INDEX
        and authority.get("transaction_identity")
        == manifest["transaction_descriptor"]
        and authority.get("permitted_transaction_indices") == PERMITTED_INDICES
        and authority.get("forbidden_transaction_indices") == FORBIDDEN_INDICES,
        "authority identity, parent, or scope differs",
    )
    require(
        authority.get("package_seal") == file_record(PACKAGE_SEAL)
        and authority.get("package_manifest") == file_record(PACKAGE_MANIFEST)
        and authority.get("package_acceptance") == file_record(PACKAGE_REVIEW)
        and authority.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and authority.get("authoritative_baseline") == file_record(BASELINE)
        and authority.get("host_reviewer_decision")
        == file_record(PACKAGE_HOST_DECISION)
        and authority.get("accepted_package") == expected_accepted_package(),
        "accepted package or review evidence differs",
    )
    require(
        authority.get("authoritative_parent") == authoritative_parent()
        and authority.get("evidence_bindings")
        == {
            "position004_input_state": manifest["position004_input_state"],
            "official_frozen_evidence": manifest["official_frozen_evidence"],
        },
        "parent or frozen evidence binding differs",
    )
    require(
        authority.get("authorized_launch") == manifest["launch"],
        "exact authorized launch differs",
    )
    require(
        authority.get("output_namespaces")
        == {
            "transaction005": str(TRANSACTION5),
            "generation6": str(GENERATION6),
            "authority": str(AUTHORITY),
            "authority_consumption": str(
                FUTURE_ROOT / "manager-authorization-consumption.json"
            ),
        },
        "authority output namespace differs",
    )
    require(
        authority.get("prohibitions")
        == [
            "transaction000-004 replay",
            "transaction006-025 execution",
            "any launch argv, cwd, or environment other than the exact bound launch",
            "any transaction005 or generation6 work before authority consumption",
            "authority replay or second consumption",
            "model, oracle, vector, compile, simulation, or submission during review",
        ],
        "authority prohibitions differ",
    )
    require(
        authority.get("activity_counters") == ZERO_COUNTERS
        and authority.get("authority_consumed") is False
        and authority.get("replay_authorized") is False
        and authority.get("transaction005_executed") is False
        and authority.get("generation6_exists") is False
        and authority.get("transactions006_025_absent") is True,
        "authority lifecycle or zero workload counters differ",
    )
    require(
        authority.get("reviewer_acceptance")
        == {
            "required": True,
            "path": str(AUTHORITY_REVIEW),
            "present_at_issuance": False,
        },
        "authority review gate differs",
    )
    if authenticate_records:
        for label, record in {
            "package seal": authority["package_seal"],
            "package manifest": authority["package_manifest"],
            "package acceptance": authority["package_acceptance"],
            "source manifest": authority["source_manifest"],
            "authoritative baseline": authority["authoritative_baseline"],
            "Host Reviewer decision": authority["host_reviewer_decision"],
        }.items():
            authenticate(record, label)
        for label, record in authority["authoritative_parent"].items():
            authenticate(record, f"parent {label}")


def validate_live_namespace(
    *,
    later_transaction_indices: list[int],
    generation6_exists: bool,
    generation6_staging_exists: bool,
    future_root_exists: bool,
    authority_count: int,
) -> None:
    require(not later_transaction_indices, "transaction005-025 artifact exists")
    require(not generation6_exists, "generation6 exists")
    require(not generation6_staging_exists, "generation6 staging exists")
    require(not future_root_exists, "transaction005 runtime evidence exists")
    require(authority_count == 1, "Manager authority cardinality differs")


def expect_rejection(
    authority: Mapping[str, Any],
    label: str,
    mutation: Callable[[dict[str, Any]], None],
) -> tuple[str, int]:
    candidate = copy.deepcopy(dict(authority))
    mutation(candidate)
    workload_calls = 0

    def workload_call() -> None:
        nonlocal workload_calls
        workload_calls += 1

    try:
        validate_authority_document(
            candidate,
            authenticate_records=False,
            workload_call=workload_call,
        )
    except ReviewError:
        require(workload_calls == 0, f"hostile control reached workload: {label}")
        return label, workload_calls
    raise ReviewError(f"hostile control accepted: {label}")


def adversarial_controls(
    authority: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "wrong-parent",
            lambda item: item.update({"authoritative_generation": 4}),
        ),
        (
            "wrong-transaction",
            lambda item: item.update({"transaction_index": 6}),
        ),
        ("wrong-layer", lambda item: item.update({"layer_index": 5})),
        (
            "wrong-argv",
            lambda item: item["authorized_launch"]["argv"].append("--replay"),
        ),
        (
            "wrong-evidence",
            lambda item: item["accepted_package"]["independent_review"].update(
                {"sha256": "0" * 64}
            ),
        ),
        (
            "wrong-namespace",
            lambda item: item["output_namespaces"].update(
                {"transaction005": str(TRANSACTIONS / "transaction-006")}
            ),
        ),
        (
            "replay-authorized",
            lambda item: item.update({"replay_authorized": True}),
        ),
        (
            "authority-consumed",
            lambda item: item.update({"authority_consumed": True}),
        ),
    ]
    mutations.extend(
        (
            f"transaction{index:03d}-scope",
            lambda item, index=index: item.update(
                {"permitted_transaction_indices": [5, index]}
            ),
        )
        for index in range(6, 26)
    )
    controls: list[str] = []
    workload_counts: dict[str, int] = {}
    for label, mutation in mutations:
        name, calls = expect_rejection(authority, label, mutation)
        controls.append(name)
        workload_counts[name] = calls
    require(
        all(count == 0 for count in workload_counts.values()),
        "hostile control workload calls are nonzero",
    )
    return controls, workload_counts


def assess_authority(
    authority_package: Path,
    output: Path,
    *,
    require_output_absent: bool = True,
) -> tuple[str, list[str], dict[str, int], str | None]:
    try:
        require(authority_package == AUTHORITY_PACKAGE, "authority path differs")
        require(output == AUTHORITY_REVIEW, "authority review path differs")
        if require_output_absent:
            require(not output.exists(), "authority review already exists")
        require(
            authority_package.is_dir()
            and not authority_package.is_symlink()
            and stat.S_IMODE(authority_package.stat().st_mode) & 0o222 == 0,
            "authority package is absent or writable",
        )
        actual = {
            path.name for path in authority_package.iterdir() if path.is_file()
        }
        require(
            actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
            "authority package file set differs",
        )
        manifest = validate_accepted_boundary()
        del manifest
        seal = load_json(AUTHORITY_SEAL)
        require(
            seal.get("kind")
            == "ace3_position3_transaction5_layer4_manager_authority_seal"
            and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
            and set(seal.get("members", {})) == AUTHORITY_MEMBERS
            and seal.get("activity_counters") == ZERO_COUNTERS
            and seal.get("authority_cardinality") == 1
            and seal.get("authority_consumed") is False
            and seal.get("replay_authorized") is False
            and seal.get("transaction005_executed") is False
            and seal.get("generation6_exists") is False
            and seal.get("transactions006_025_absent") is True,
            "authority seal differs",
        )
        for name, record in seal["members"].items():
            path = authority_package / name
            authenticate(record, f"authority member {name}", path)
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"authority member is writable: {name}",
            )
        request = load_json(authority_package / "review-request.json")
        require(
            request.get("kind")
            == "ace3_position3_transaction5_layer4_authority_review_request"
            and request.get("mission_id") == MISSION_ID
            and request.get("required_role") == "reviewer"
            and request.get("review_output") == str(output)
            and request.get("authority") == file_record(AUTHORITY)
            and request.get("package_acceptance") == file_record(PACKAGE_REVIEW),
            "authority review request differs",
        )
        source_boundary(
            (authority_package / "review-emitter.py").read_text(encoding="utf-8")
        )
        authority = load_json(AUTHORITY)
        validate_authority_document(authority)
        authority_files = list(
            AUTHORITY_PACKAGE.parent.rglob("manager-authority.json")
        )
        later = [
            index
            for index in range(5, 26)
            if (TRANSACTIONS / f"transaction-{index:03d}").exists()
        ]
        validate_live_namespace(
            later_transaction_indices=later,
            generation6_exists=GENERATION6.exists(),
            generation6_staging_exists=GENERATION6_STAGING.exists(),
            future_root_exists=FUTURE_ROOT.exists(),
            authority_count=len(authority_files),
        )
        require(
            authority_files == [AUTHORITY],
            "unexpected transaction005 Manager authority exists",
        )
        controls, workload_counts = adversarial_controls(authority)
        return "PASS", controls, workload_counts, None
    except (
        ReviewError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        return "REJECT", [], {}, str(error)


def write_review(path: Path, payload: bytes) -> None:
    require(path.is_absolute(), "review output must be absolute")
    require(
        path.parent.is_dir() and not path.parent.is_symlink(),
        "review parent must be an existing real directory",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def emit_review(authority_package: Path, output: Path) -> None:
    status, controls, workload_counts, reason = assess_authority(
        authority_package,
        output,
    )
    authority = load_json(AUTHORITY)
    review = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_layer4_authority_independent_review",
        "status": status,
        "review_type": "source-disjoint-executable-authority-review",
        "producer_role": "reviewer",
        "mission_id": MISSION_ID,
        "preparation_participation": False,
        "authority_seal": file_record(AUTHORITY_SEAL),
        "authority": file_record(AUTHORITY),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "package_acceptance": file_record(PACKAGE_REVIEW),
        "host_package_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
        "authoritative_parent": authoritative_parent(),
        "authorized_launch": copy.deepcopy(authority["authorized_launch"]),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "authority_cardinality": 1,
        "authority_consumed": False,
        "replay_authorized": False,
        "transaction005_executed": False,
        "generation6_exists": False,
        "transactions006_025_absent": True,
        "adversarial_controls": controls,
        "hostile_control_workload_calls": workload_counts,
        "total_workload_calls": sum(workload_counts.values()),
        "reason": reason,
    }
    write_review(output, canonical_json(review))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    emit_review(arguments.authority_package, arguments.output)
    print(f"TRANSACTION005_LAYER04_AUTHORITY_REVIEW_WRITTEN output={arguments.output}")


if __name__ == "__main__":
    try:
        main()
    except (
        ReviewError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"TRANSACTION005_LAYER04_AUTHORITY_REVIEW_REFUSED {error}"
        ) from error
