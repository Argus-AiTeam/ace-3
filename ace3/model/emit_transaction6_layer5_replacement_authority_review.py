#!/usr/bin/env python3
"""Source-disjoint review of the transaction006/layer05 r4 authority."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY

MISSION_ID = "892d5033a335"
PACKAGE_MISSION_ID = "51e05603a20c"
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/892d5033a335/mission.json"
)
PACKAGE_HOST_DECISION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/51e05603a20c/round-0001.json"
)
PACKAGE_MANAGER_MISSION = PACKAGE_HOST_DECISION.with_name("mission.json")

PACKAGE = RUNTIME / "transaction6-layer5-continuation-package-r4"
PACKAGE_MANIFEST = PACKAGE / "package-manifest.json"
PACKAGE_SEAL = PACKAGE / "package-seal.json"
SOURCE_MANIFEST = PACKAGE / "source-manifest.json"
BASELINE = PACKAGE / "authoritative-baseline.json"
PACKAGE_REVIEW = (
    RUNTIME / "transaction6-layer5-continuation-review-r4/independent-review.json"
)

OLD_PACKAGE = RUNTIME / "transaction6-layer5-continuation-package-r3"
OLD_PACKAGE_REVIEW = (
    RUNTIME / "transaction6-layer5-continuation-review-r3/independent-review.json"
)
OLD_AUTHORITY_PACKAGE = RUNTIME / "transaction6-layer5-continuation-authority-r3"
OLD_AUTHORITY = OLD_AUTHORITY_PACKAGE / "manager-authority.json"
OLD_AUTHORITY_SEAL = OLD_AUTHORITY_PACKAGE / "authority-seal.json"
OLD_AUTHORITY_REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-review-r3"
    / "independent-review.json"
)
OLD_AUTHORITY_HOST_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/24854a171c81/round-0001.json"
)

ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION6 = ADOPTION / "state-generations/generation-0000000006"
GENERATION7 = ADOPTION / "state-generations/generation-0000000007"
GENERATION7_STAGING = ADOPTION / "state-generations/.generation-0000000007.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION6 = TRANSACTIONS / "transaction-006"
FUTURE_ROOT = RUNTIME / "transaction6-authoritative-generation7"

AUTHORITY_PACKAGE = (
    RUNTIME / "transaction6-layer5-continuation-authority-r4-replacement"
)
AUTHORITY = AUTHORITY_PACKAGE / "manager-authority.json"
AUTHORITY_SEAL = AUTHORITY_PACKAGE / "authority-seal.json"
AUTHORITY_REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-review-r4-replacement"
    / "independent-review.json"
)

TRANSACTION_INDEX = 6
LAYER_INDEX = 5
PERMITTED_INDICES = [TRANSACTION_INDEX]
FORBIDDEN_INDICES = [*range(6), *range(7, 26)]
AUTHORITY_MEMBERS = {
    "manager-authority.json",
    "review-emitter.py",
    "review-request.json",
}
PACKAGE_MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction6_executor.py",
    "package_manifest": "package-manifest.json",
    "review_emitter": "review-emitter.py",
    "review_request": "review-request.json",
    "source_manifest": "source-manifest.json",
}
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
    PACKAGE_SEAL: "5cf9eab3f1267333cdd32714759e11abdd15b424d0f11e69ce13cdb2d1f151d3",
    PACKAGE_MANIFEST: "1f782fbfd495e8a8dbd66d25482ede488858e554652e68518fb7b482b74a5890",
    SOURCE_MANIFEST: "6294c8d9de9816ab594fed4fab4124c55afe332d6f173378edc06854e157fd86",
    BASELINE: "9e71792a253ad6b860623ad50fcb7ee2cbdc7966f395617ceaef7d7f6eb40003",
    PACKAGE_REVIEW: "7bf470cf31e5e3b1185e051d2191c8e739409e69c0af485180f528e4ff704a1a",
    PACKAGE_HOST_DECISION: "5dc0031bcf4ec67244da03813b8a8434bc236b5b6699b8bd08c1e034c784158b",
    PACKAGE_MANAGER_MISSION: "c7216011351808cb466073ad1e824aa5cca21abee0f5479d08596056c3d742e2",
    MISSION: "6d39e66046dea39e4f9d1ea3bcc1515ff5d444f1cec896a621af474118376fee",
    POINTER: "d3042e1e75afebb7996aa618390f5dc8dfada73bfcdae028189d19d100fb0e47",
    GENERATION6
    / "generation-manifest.json": "4991277d157154eb564f6ebf7a442fa9d7d1375c58025b1bef22d477aaf1981c",
    GENERATION6
    / "ledger.json": "73ae94073b546410f60fae32bcdcf00b99b2819c5a2ca16e9ffa6613bf4892c0",
    GENERATION6
    / "checkpoints/transaction-005.json": "4145c04113658114bfe0e428661212914231465baf4e221737d16d9e6821f896",
    TRANSACTIONS
    / "transaction-005/position004.state": "0b35421e91650d2c131116bade95af38d90230cecf5d78ca6abdc99890d0f865",
    OLD_PACKAGE
    / "authoritative-baseline.json": "9e71792a253ad6b860623ad50fcb7ee2cbdc7966f395617ceaef7d7f6eb40003",
    OLD_PACKAGE
    / "package-manifest.json": "84cd977a9767e1f6413d1f281ecd0991aea1f1f35e26ed507f0df709326dd0f0",
    OLD_PACKAGE
    / "package-seal.json": "9d091b92acba2c58ecf91987c84dd56f66862eaac2df04e4c696e460578d87b8",
    OLD_PACKAGE
    / "review-emitter.py": "5102136a58d7dcddab2b0d3bf4bfa9dd8fd95fdf44cad76d556321c520cb1f2f",
    OLD_PACKAGE
    / "review-request.json": "079c09bb749c3f866459370ccdbaf6e0bec976679c747248fda2789782849e7f",
    OLD_PACKAGE
    / "source-manifest.json": "f3ee39577e35c19b311905f19f20a89b6c24151b77b29bbef1c8b80bafb044cc",
    OLD_PACKAGE
    / "transaction6_executor.py": "8b5ab23311674c824e10f7a6faa202e4118f8f6d03be1f792a2da94d33aa8af6",
    OLD_PACKAGE_REVIEW: "eb5aec0eaa5e017fef6ed2de7ac7cdc128138ceaef0e38224e18a4e772d3666e",
    OLD_AUTHORITY: "899c63413d4a39aa1d10fb3da1043215b415073666866e14a2de8f219ad3af3d",
    OLD_AUTHORITY_SEAL: "ff65290ab114065644419a1fa71fd572d56d8fd27365622fc8427245bc3e6d11",
    OLD_AUTHORITY_PACKAGE
    / "review-emitter.py": "f8b6261fcc419eeeef958a35c1ea2faab088c678861afaa3efe39bc9a888878e",
    OLD_AUTHORITY_PACKAGE
    / "review-request.json": "23c0e16bd04b360f9a9d61a817c203600dc6b484f2ea495efb29d4f1311030b8",
    OLD_AUTHORITY_REVIEW: "f543ac8aaa71285962709af6eb95ae548d44d0a13592ed1cd7359c884b55578b",
    OLD_AUTHORITY_HOST_REVIEW: "d91cd382e3d7917f3d19df2a1ec7908d6dfb89300ffae09c45244a0332411944",
}
FORBIDDEN_IMPORTS = {
    "ctypes",
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


def authenticate(record: Mapping[str, Any], label: str, path: Path) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"}
        and record.get("path") == str(path)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str)
        and file_record(path) == dict(record),
        f"{label} differs",
    )


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


def expected_supersession() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_transaction006_layer05_r3_supersession",
        "status": "SUPERSEDED_NON_AUTHORITATIVE_HISTORY",
        "operator_decision": require_fixed(PACKAGE_MANAGER_MISSION),
        "reviewed_r3_package_seal": require_fixed(
            OLD_PACKAGE / "package-seal.json"
        ),
        "reviewed_r3_package_acceptance": require_fixed(OLD_PACKAGE_REVIEW),
        "rejected_r3_authority": require_fixed(OLD_AUTHORITY),
        "rejected_r3_authority_seal": require_fixed(OLD_AUTHORITY_SEAL),
        "rejected_r3_authority_review": require_fixed(OLD_AUTHORITY_REVIEW),
        "rejected_r3_host_review": require_fixed(OLD_AUTHORITY_HOST_REVIEW),
        "r3_authority_consumed": False,
        "r3_authority_remains_non_authoritative": True,
        "successor_package": str(PACKAGE),
        "replacement_authority": str(AUTHORITY),
    }


def expected_launch() -> dict[str, Any]:
    return {
        "argv": [
            "/home/argustest/miniconda3/bin/python3",
            str(PACKAGE / "transaction6_executor.py"),
            "execute",
            "--package",
            str(PACKAGE),
            "--review",
            str(PACKAGE_REVIEW),
            "--authorization",
            str(AUTHORITY),
            "--runtime-root",
            str(RUNTIME),
            "--transaction-index",
            "6",
        ],
        "cwd": str(ROOT),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "exit_cursor": 7,
        "exit_generation": 7,
        "interpreter": "/home/argustest/miniconda3/bin/python3",
        "layer_index": LAYER_INDEX,
        "start_cursor": 6,
        "start_generation": 6,
        "transaction_index": TRANSACTION_INDEX,
    }


def manager_provenance() -> dict[str, Any]:
    record = require_fixed(MISSION)
    mission = load_json(MISSION)
    objective = mission.get("objective", "")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx006-layer05-r4-replacement-authority"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and "manager-direct" in mission.get("tags", [])
        and isinstance(objective, str)
        and "issue exactly one distinct replacement authority" in objective
        and "Run 29 hostile controls with zero workload calls" in objective,
        "replacement authority mission differs",
    )
    return {
        "mission": record,
        "mission_id": MISSION_ID,
        "manager_direct": True,
    }


def validate_preserved_history() -> None:
    expected_package_files = {
        "authoritative-baseline.json",
        "package-manifest.json",
        "package-seal.json",
        "review-emitter.py",
        "review-request.json",
        "source-manifest.json",
        "transaction6_executor.py",
    }
    expected_authority_files = {
        "authority-seal.json",
        "manager-authority.json",
        "review-emitter.py",
        "review-request.json",
    }
    require(
        {path.name for path in OLD_PACKAGE.iterdir() if path.is_file()}
        == expected_package_files,
        "r3 package file set differs",
    )
    require(
        {path.name for path in OLD_AUTHORITY_PACKAGE.iterdir() if path.is_file()}
        == expected_authority_files,
        "r3 authority file set differs",
    )
    for path in FIXED_SHA256:
        if (
            OLD_PACKAGE in path.parents
            or OLD_AUTHORITY_PACKAGE in path.parents
            or path in {
                OLD_PACKAGE_REVIEW,
                OLD_AUTHORITY_REVIEW,
                OLD_AUTHORITY_HOST_REVIEW,
            }
        ):
            require_fixed(path)
    package_review = load_json(OLD_PACKAGE_REVIEW)
    authority = load_json(OLD_AUTHORITY)
    authority_review = load_json(OLD_AUTHORITY_REVIEW)
    require(
        package_review.get("status") == "PASS"
        and package_review.get("transaction006_executed") is False,
        "r3 package review history differs",
    )
    require(
        authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("authority_consumed") is False
        and authority.get("transaction006_executed") is False,
        "r3 authority history differs",
    )
    require(
        authority_review.get("status") == "REJECT"
        and authority_review.get("total_workload_calls") == 0
        and authority_review.get("transaction006_executed") is False,
        "r3 rejected review history differs",
    )


def authoritative_parent() -> dict[str, Any]:
    return {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(GENERATION6 / "generation-manifest.json"),
        "ledger": file_record(GENERATION6 / "ledger.json"),
        "checkpoint005": file_record(
            GENERATION6 / "checkpoints/transaction-005.json"
        ),
        "position004_input_state": file_record(
            TRANSACTIONS / "transaction-005/position004.state"
        ),
    }


def validate_accepted_boundary() -> dict[str, Any]:
    for path in (
        PACKAGE_SEAL,
        PACKAGE_MANIFEST,
        SOURCE_MANIFEST,
        BASELINE,
        PACKAGE_REVIEW,
        PACKAGE_HOST_DECISION,
        PACKAGE_MANAGER_MISSION,
        MISSION,
        POINTER,
        GENERATION6 / "generation-manifest.json",
        GENERATION6 / "ledger.json",
        GENERATION6 / "checkpoints/transaction-005.json",
        TRANSACTIONS / "transaction-005/position004.state",
    ):
        require_fixed(path)
    require(
        PACKAGE.is_dir()
        and not PACKAGE.is_symlink()
        and stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "sealed r4 package is absent or writable",
    )
    manifest = load_json(PACKAGE_MANIFEST)
    seal = load_json(PACKAGE_SEAL)
    review = load_json(PACKAGE_REVIEW)
    host_decision = load_json(PACKAGE_HOST_DECISION)
    require(
        manifest.get("kind")
        == "ace3_position3_transaction6_layer5_executor_package"
        and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("mission_id") == PACKAGE_MISSION_ID
        and manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and manifest.get("authoritative_generation") == 6
        and manifest.get("authoritative_cursor") == 6
        and manifest.get("required_parent_checkpoint_index") == 5
        and manifest.get("transaction_index") == TRANSACTION_INDEX
        and manifest.get("layer_index") == LAYER_INDEX
        and manifest.get("permitted_transaction_indices") == PERMITTED_INDICES
        and manifest.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and manifest.get("future_manager_authority") == str(AUTHORITY)
        and manifest.get("launch") == expected_launch()
        and manifest.get("stable_host_reviewer_provenance")
        == str(PACKAGE_HOST_DECISION)
        and manifest.get("supersession") == expected_supersession()
        and manifest.get("execution_authorized") is False
        and manifest.get("authority_created") is False
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "sealed r4 package boundary differs",
    )
    require(
        seal.get("kind") == "ace3_position3_transaction6_layer5_executor_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and set(seal.get("members", {})) == set(PACKAGE_MEMBERS)
        and seal.get("execution_authorized") is False
        and seal.get("authority_created") is False
        and seal.get("activity_counters") == ZERO_COUNTERS,
        "r4 package seal differs",
    )
    for label, relative in PACKAGE_MEMBERS.items():
        path = PACKAGE / relative
        authenticate(seal["members"][label], f"r4 package member {label}", path)
        require(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
            f"r4 package member is writable: {label}",
        )
    require(
        review.get("kind")
        == "ace3_position3_transaction6_layer5_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == PACKAGE_MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and review.get("authoritative_baseline") == file_record(BASELINE)
        and review.get("permitted_launch") == expected_launch()
        and review.get("stable_host_reviewer_provenance")
        == str(PACKAGE_HOST_DECISION)
        and review.get("supersession") == expected_supersession()
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction006_executed") is False
        and review.get("generation7_published") is False
        and review.get("transactions007_025_absent") is True
        and review.get("this_review_authorizes_execution") is False,
        "canonical r4 package PASS review differs",
    )
    require(
        stat.S_IMODE(PACKAGE_REVIEW.stat().st_mode) & 0o222 == 0,
        "canonical r4 package review is writable",
    )
    require(
        host_decision.get("kind") == "round_reviewed_handoff"
        and host_decision.get("mission_id") == PACKAGE_MISSION_ID
        and host_decision.get("producer_role") == "reviewer"
        and host_decision.get("review", {}).get("status") == "done",
        "immutable Host Reviewer handoff differs",
    )
    pointer = load_json(POINTER)
    generation = load_json(GENERATION6 / "generation-manifest.json")
    ledger = load_json(GENERATION6 / "ledger.json")
    checkpoint = load_json(GENERATION6 / "checkpoints/transaction-005.json")
    require(
        pointer.get("kind")
        == "ace3_position3_transaction5_generation6_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 6
        and pointer.get("generation_manifest")
        == file_record(GENERATION6 / "generation-manifest.json")
        and generation.get("kind")
        == "ace3_transaction5_publication_recovery_generation6_state"
        and generation.get("generation") == 6
        and generation.get("transactions006_025_executed") is False
        and ledger.get("state_generation") == 6
        and ledger.get("next_transaction_index") == 6
        and ledger.get("completed_transaction_count") == 6
        and checkpoint.get("transaction_index") == 5
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("outputs", {}).get("state")
        == file_record(TRANSACTIONS / "transaction-005/position004.state"),
        "generation6/cursor6/checkpoint005 parent differs",
    )
    manager_provenance()
    validate_preserved_history()
    return manifest


def expected_accepted_package() -> dict[str, Any]:
    return {
        "root": str(PACKAGE),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "authoritative_baseline": file_record(BASELINE),
        "independent_review": file_record(PACKAGE_REVIEW),
        "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
    }


def expected_authority() -> dict[str, Any]:
    manifest = load_json(PACKAGE_MANIFEST)
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction6_layer5_manager_authorization",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "mission_id": MISSION_ID,
        "manager_provenance": manager_provenance(),
        "authority_cardinality": 1,
        "runtime_identity": RUNTIME_IDENTITY,
        "authoritative_generation": 6,
        "authoritative_cursor": 6,
        "required_parent_checkpoint_index": 5,
        "transaction_index": TRANSACTION_INDEX,
        "layer_index": LAYER_INDEX,
        "transaction_identity": copy.deepcopy(manifest["transaction_descriptor"]),
        "permitted_transaction_indices": list(PERMITTED_INDICES),
        "forbidden_transaction_indices": list(FORBIDDEN_INDICES),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "package_acceptance": file_record(PACKAGE_REVIEW),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "authoritative_baseline": file_record(BASELINE),
        "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
        "stable_host_reviewer_provenance": file_record(PACKAGE_HOST_DECISION),
        "supersession": expected_supersession(),
        "accepted_package": expected_accepted_package(),
        "authoritative_parent": authoritative_parent(),
        "evidence_bindings": {
            "position004_input_state": copy.deepcopy(
                manifest["position004_input_state"]
            ),
            "official_frozen_evidence": copy.deepcopy(
                manifest["official_frozen_evidence"]
            ),
        },
        "authorized_launch": copy.deepcopy(manifest["launch"]),
        "output_namespaces": {
            "transaction006": str(TRANSACTION6),
            "generation7": str(GENERATION7),
            "authority": str(AUTHORITY),
            "authority_consumption": str(
                FUTURE_ROOT / "manager-authorization-consumption.json"
            ),
        },
        "prohibitions": [
            "transaction000-005 replay",
            "transaction007-025 execution",
            "any launch argv, cwd, or environment other than the exact bound launch",
            "any transaction006 or generation7 work before authority consumption",
            "authority replay or second consumption",
            "model, oracle, vector, compile, simulation, or submission during review",
        ],
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "authority_consumed": False,
        "replay_authorized": False,
        "transaction006_executed": False,
        "generation7_exists": False,
        "transactions007_025_absent": True,
        "reviewer_acceptance": {
            "required": True,
            "path": str(AUTHORITY_REVIEW),
            "present_at_issuance": False,
        },
    }


def validate_authority_document(
    authority: Mapping[str, Any],
    workload_call: Callable[[], None] | None = None,
) -> None:
    require(dict(authority) == expected_authority(), "authority boundary differs")
    if workload_call is not None:
        workload_call()


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
        validate_authority_document(candidate, workload_call)
    except ReviewError:
        require(workload_calls == 0, f"hostile control reached workload: {label}")
        return label, workload_calls
    raise ReviewError(f"hostile control accepted: {label}")


def adversarial_controls(
    authority: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("replay", lambda item: item.update({"replay_authorized": True})),
        (
            "wrong-r4-package-provenance",
            lambda item: item["package_seal"].update({"sha256": "0" * 64}),
        ),
        (
            "wrong-host-provenance",
            lambda item: item["stable_host_reviewer_provenance"].update(
                {"sha256": "0" * 64}
            ),
        ),
        (
            "wrong-r3-supersession",
            lambda item: item["supersession"]["rejected_r3_authority"].update(
                {"sha256": "0" * 64}
            ),
        ),
        (
            "wrong-generation-cursor-checkpoint-parent",
            lambda item: item.update({"authoritative_cursor": 5}),
        ),
        (
            "wrong-transaction-layer-scope",
            lambda item: item.update({"layer_index": 4}),
        ),
        (
            "wrong-exact-argv",
            lambda item: item["authorized_launch"]["argv"].append("--replay"),
        ),
        (
            "wrong-output-namespace",
            lambda item: item["output_namespaces"].update(
                {"transaction006": str(TRANSACTIONS / "transaction-007")}
            ),
        ),
        (
            "authority-consumed",
            lambda item: item.update({"authority_consumed": True}),
        ),
        (
            "nonzero-workload-counter",
            lambda item: item["activity_counters"].update({"model_execution": 1}),
        ),
    ]
    mutations.extend(
        (
            f"transaction{index:03d}-scope",
            lambda item, index=index: item.update(
                {"permitted_transaction_indices": [6, index]}
            ),
        )
        for index in range(7, 26)
    )
    controls: list[str] = []
    workload_counts: dict[str, int] = {}
    for label, mutation in mutations:
        name, calls = expect_rejection(authority, label, mutation)
        controls.append(name)
        workload_counts[name] = calls
    require(len(controls) == 29, "hostile control cardinality differs")
    require(
        all(count == 0 for count in workload_counts.values()),
        "hostile control workload calls are nonzero",
    )
    return controls, workload_counts


def replacement_authorities() -> list[Path]:
    return sorted(
        RUNTIME.glob(
            "transaction6-layer5-continuation-authority-r4*/"
            "manager-authority.json"
        )
    )


def validate_live_namespace() -> None:
    require(
        not TRANSACTION6.exists()
        and all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(7, 26)
        ),
        "transaction006-025 artifact exists",
    )
    require(
        not GENERATION7.exists()
        and not GENERATION7_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "generation7 or transaction006 runtime evidence exists",
    )
    require(
        replacement_authorities() == [AUTHORITY],
        "r4 replacement authority cardinality differs",
    )
    validate_preserved_history()


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
            AUTHORITY_PACKAGE.is_dir()
            and not AUTHORITY_PACKAGE.is_symlink()
            and stat.S_IMODE(AUTHORITY_PACKAGE.stat().st_mode) & 0o222 == 0,
            "replacement authority package is absent or writable",
        )
        actual = {
            path.name for path in AUTHORITY_PACKAGE.iterdir() if path.is_file()
        }
        require(
            actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
            "replacement authority package file set differs",
        )
        validate_accepted_boundary()
        seal = load_json(AUTHORITY_SEAL)
        require(
            seal.get("kind")
            == (
                "ace3_position3_transaction6_layer5_r4_replacement_"
                "manager_authority_seal"
            )
            and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
            and set(seal.get("members", {})) == AUTHORITY_MEMBERS
            and seal.get("package_seal") == file_record(PACKAGE_SEAL)
            and seal.get("package_acceptance") == file_record(PACKAGE_REVIEW)
            and seal.get("host_reviewer_decision")
            == file_record(PACKAGE_HOST_DECISION)
            and seal.get("supersession") == expected_supersession()
            and seal.get("activity_counters") == ZERO_COUNTERS
            and seal.get("authority_cardinality") == 1
            and seal.get("authority_consumed") is False
            and seal.get("replay_authorized") is False
            and seal.get("transaction006_executed") is False
            and seal.get("generation7_exists") is False
            and seal.get("transactions007_025_absent") is True,
            "replacement authority seal differs",
        )
        for name, record in seal["members"].items():
            path = AUTHORITY_PACKAGE / name
            authenticate(record, f"replacement authority member {name}", path)
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"replacement authority member is writable: {name}",
            )
        request = load_json(AUTHORITY_PACKAGE / "review-request.json")
        require(
            request.get("kind")
            == (
                "ace3_position3_transaction6_layer5_r4_replacement_"
                "authority_review_request"
            )
            and request.get("mission_id") == MISSION_ID
            and request.get("required_role") == "reviewer"
            and request.get("review_output") == str(output)
            and request.get("authority") == file_record(AUTHORITY)
            and request.get("package_acceptance") == file_record(PACKAGE_REVIEW)
            and request.get("host_reviewer_decision")
            == file_record(PACKAGE_HOST_DECISION)
            and request.get("supersession") == expected_supersession(),
            "replacement authority review request differs",
        )
        source_boundary(
            (AUTHORITY_PACKAGE / "review-emitter.py").read_text(encoding="utf-8")
        )
        authority = load_json(AUTHORITY)
        validate_authority_document(authority)
        validate_live_namespace()
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
        "kind": (
            "ace3_position3_transaction6_layer5_r4_replacement_"
            "authority_independent_review"
        ),
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
        "stable_host_reviewer_provenance": file_record(PACKAGE_HOST_DECISION),
        "supersession": expected_supersession(),
        "authoritative_parent": authoritative_parent(),
        "authorized_launch": copy.deepcopy(authority["authorized_launch"]),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "authority_cardinality": 1,
        "authority_consumed": False,
        "replay_authorized": False,
        "transaction006_executed": False,
        "generation7_exists": False,
        "transactions007_025_absent": True,
        "adversarial_controls": controls,
        "hostile_control_workload_calls": workload_counts,
        "total_workload_calls": sum(workload_counts.values()),
        "reason": reason,
    }
    write_review(output, canonical_json(review))
    output.parent.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assess-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.assess_only:
        status, controls, workload_counts, reason = assess_authority(
            arguments.authority_package,
            arguments.output,
            require_output_absent=not arguments.output.exists(),
        )
        require(status == "PASS", reason or "authority assessment rejected")
        print(
            "TRANSACTION006_LAYER05_R4_AUTHORITY_ASSESSMENT_PASS "
            f"controls={len(controls)} workload_calls={sum(workload_counts.values())}"
        )
        return
    emit_review(arguments.authority_package, arguments.output)
    print(f"TRANSACTION006_LAYER05_R4_AUTHORITY_REVIEW_WRITTEN output={arguments.output}")


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
            f"TRANSACTION006_LAYER05_R4_AUTHORITY_REVIEW_REFUSED {error}"
        ) from error
