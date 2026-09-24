#!/usr/bin/env python3
"""Source-disjoint review of the transaction007/layer06 Manager authority."""

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
MISSION_ID = "a9c26249d4b5"
PACKAGE_MISSION_ID = "cd582996d710"
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/a9c26249d4b5/mission.json"
)
PACKAGE_HOST_DECISION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/cd582996d710/round-0001.json"
)
PACKAGE_MANAGER_MISSION = PACKAGE_HOST_DECISION.with_name("mission.json")

PACKAGE = RUNTIME / "transaction7-layer6-continuation-package-r1"
PACKAGE_MANIFEST = PACKAGE / "package-manifest.json"
PACKAGE_SEAL = PACKAGE / "package-seal.json"
SOURCE_MANIFEST = PACKAGE / "source-manifest.json"
BASELINE = PACKAGE / "authoritative-baseline.json"
PACKAGE_REVIEW = (
    RUNTIME / "transaction7-layer6-continuation-review-r1/independent-review.json"
)
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION7 = ADOPTION / "state-generations/generation-0000000007"
GENERATION8 = ADOPTION / "state-generations/generation-0000000008"
GENERATION8_STAGING = ADOPTION / "state-generations/.generation-0000000008.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION7 = TRANSACTIONS / "transaction-007"
FUTURE_ROOT = RUNTIME / "transaction7-authoritative-generation8"

AUTHORITY_PACKAGE = RUNTIME / "transaction7-layer6-continuation-authority-r1"
AUTHORITY = AUTHORITY_PACKAGE / "manager-authority.json"
AUTHORITY_SEAL = AUTHORITY_PACKAGE / "authority-seal.json"
AUTHORITY_REVIEW = (
    RUNTIME
    / "transaction7-layer6-continuation-authority-review-r1"
    / "independent-review.json"
)

TRANSACTION_INDEX = 7
LAYER_INDEX = 6
PERMITTED_INDICES = [TRANSACTION_INDEX]
FORBIDDEN_INDICES = [*range(7), *range(8, 26)]
AUTHORITY_MEMBERS = {
    "manager-authority.json",
    "review-emitter.py",
    "review-request.json",
}
PACKAGE_MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction7_executor.py",
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
    MISSION: "dd96e1371456cd177f4ced74b89f992d54596f834196c3c21a94ce161b5a169b",
    PACKAGE_MANAGER_MISSION: "df93998131e0f9e2111aa92432ef18a9aa36b3c21058589246202848c18ba9c0",
    PACKAGE_HOST_DECISION: "a5c909eba10c927f16ae2f68be097b73e5bdd913feca483851eecb76feadacbe",
    PACKAGE_SEAL: "1386cf9320fe54fba47e9f227f963301f95fae3328f6f4e2c442e9d61aaee46d",
    PACKAGE_MANIFEST: "e7f64fec6ba1c059dd27a3128979c34e3b1a12e31e5543b8697df4c54d750bef",
    SOURCE_MANIFEST: "ecd24a2d312fec79c613d92c278d3914db971ec10187f770c240d5d9e51e8148",
    BASELINE: "62e0bbdd9f2a8fa5d9e13a75a4459d0d2a4123e6c49396945b193faf338703d5",
    PACKAGE_REVIEW: "e503a873f7df236edd4c2fea548ed212fe527fcd8b924390edc22166deb0a266",
    POINTER: "87727196763ad11b666d07b14cf9d885ff58169277b98503d8407b1262d7f60d",
    GENERATION7
    / "generation-manifest.json": "55f2c189cd8694e72a8156c8f6e4fcf29eb8be0963ef04334c52956e21a2ea28",
    GENERATION7
    / "ledger.json": "784dbeeafded4b0d956601c4697a535848ea99ec1350cbdee25bdc4a2d35cc6a",
    GENERATION7
    / "checkpoints/transaction-006.json": "9a2aa573c27e703970d3badb61a2ecbb416ff711c0dc5beb4a7285d0e6bb6929",
    TRANSACTIONS
    / "transaction-006/position004.state": "d8445c2badf3e544e5affd9942f66e2efd71e23647cc20f66e1656ed6589ebb0",
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


def tree_record(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"real tree required: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(path for path in root.iterdir() if path.is_file())
    require(all(not path.is_symlink() for path in files), "symlinked tree member")
    for path in files:
        relative = path.name.encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(path)))
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


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


def expected_launch() -> dict[str, Any]:
    return {
        "argv": [
            "/home/argustest/miniconda3/bin/python3",
            str(PACKAGE / "transaction7_executor.py"),
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
            "7",
        ],
        "cwd": str(ROOT),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "exit_cursor": 8,
        "exit_generation": 8,
        "interpreter": "/home/argustest/miniconda3/bin/python3",
        "layer_index": LAYER_INDEX,
        "start_cursor": 7,
        "start_generation": 7,
        "transaction_index": TRANSACTION_INDEX,
    }


def authoritative_parent() -> dict[str, Any]:
    return {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(GENERATION7 / "generation-manifest.json"),
        "ledger": file_record(GENERATION7 / "ledger.json"),
        "checkpoint006": file_record(
            GENERATION7 / "checkpoints/transaction-006.json"
        ),
        "position004_input_state": file_record(
            TRANSACTIONS / "transaction-006/position004.state"
        ),
    }


def validate_accepted_boundary() -> dict[str, Any]:
    for path in FIXED_SHA256:
        require_fixed(path)
    require(
        PACKAGE.is_dir()
        and not PACKAGE.is_symlink()
        and stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "sealed package is absent or writable",
    )
    manifest = load_json(PACKAGE_MANIFEST)
    seal = load_json(PACKAGE_SEAL)
    review = load_json(PACKAGE_REVIEW)
    host_decision = load_json(PACKAGE_HOST_DECISION)
    mission = load_json(MISSION)
    pointer = load_json(POINTER)
    generation = load_json(GENERATION7 / "generation-manifest.json")
    ledger = load_json(GENERATION7 / "ledger.json")
    checkpoint = load_json(GENERATION7 / "checkpoints/transaction-006.json")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx007-layer06-authority"
        and manifest.get("kind")
        == "ace3_position3_transaction7_layer6_executor_package"
        and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("mission_id") == PACKAGE_MISSION_ID
        and manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and manifest.get("authoritative_generation") == 7
        and manifest.get("authoritative_cursor") == 7
        and manifest.get("required_parent_checkpoint_index") == 6
        and manifest.get("transaction_index") == TRANSACTION_INDEX
        and manifest.get("layer_index") == LAYER_INDEX
        and manifest.get("future_manager_authority") == str(AUTHORITY)
        and manifest.get("launch") == expected_launch()
        and manifest.get("execution_authorized") is False
        and manifest.get("authority_created") is False
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "package or Manager mission boundary differs",
    )
    require(
        seal.get("kind") == "ace3_position3_transaction7_layer6_executor_seal"
        and set(seal.get("members", {})) == set(PACKAGE_MEMBERS)
        and seal.get("activity_counters") == ZERO_COUNTERS,
        "package seal differs",
    )
    for label, relative in PACKAGE_MEMBERS.items():
        authenticate(seal["members"][label], f"package member {label}", PACKAGE / relative)
    require(
        review.get("kind")
        == "ace3_position3_transaction7_layer6_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == PACKAGE_MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("permitted_launch") == expected_launch()
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction007_executed") is False
        and review.get("generation8_published") is False
        and review.get("transactions008_025_absent") is True
        and review.get("this_review_authorizes_execution") is False
        and host_decision.get("kind") == "round_reviewed_handoff"
        and host_decision.get("mission_id") == PACKAGE_MISSION_ID
        and host_decision.get("producer_role") == "reviewer"
        and host_decision.get("review", {}).get("status") == "done",
        "immutable package PASS boundary differs",
    )
    require(
        pointer.get("kind")
        == "ace3_position3_transaction6_generation7_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 7
        and pointer.get("generation_manifest")
        == file_record(GENERATION7 / "generation-manifest.json")
        and generation.get("kind")
        == "ace3_transaction6_publication_recovery_generation7_state"
        and generation.get("generation") == 7
        and generation.get("transactions007_025_executed") is False
        and ledger.get("state_generation") == 7
        and ledger.get("next_transaction_index") == 7
        and ledger.get("completed_transaction_count") == 7
        and checkpoint.get("transaction_index") == 6
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("outputs", {}).get("state")
        == file_record(TRANSACTIONS / "transaction-006/position004.state"),
        "generation7/cursor7/checkpoint006 parent differs",
    )
    return manifest


def validate_live_namespace() -> None:
    require(
        not TRANSACTION7.exists()
        and all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(8, 26)
        ),
        "transaction007-025 artifact exists",
    )
    require(
        not GENERATION8.exists()
        and not GENERATION8_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "generation8 or transaction007 runtime evidence exists",
    )
    authorities = sorted(
        RUNTIME.glob(
            "transaction7-layer6-continuation-authority*/manager-authority.json"
        )
    )
    require(authorities == [AUTHORITY], "authority cardinality differs")


def validate_authority_semantics(authority: Mapping[str, Any]) -> None:
    manifest = load_json(PACKAGE_MANIFEST)
    require(
        authority.get("kind")
        == "ace3_position3_transaction7_layer6_manager_authorization"
        and authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("producer_role") == "manager"
        and authority.get("mission_id") == MISSION_ID
        and authority.get("authority_cardinality") == 1
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("authoritative_generation") == 7
        and authority.get("authoritative_cursor") == 7
        and authority.get("required_parent_checkpoint_index") == 6
        and authority.get("target_generation") == 8
        and authority.get("target_cursor") == 8
        and authority.get("target_checkpoint_index") == 7
        and authority.get("transaction_index") == TRANSACTION_INDEX
        and authority.get("layer_index") == LAYER_INDEX
        and authority.get("transaction_identity") == manifest["transaction_descriptor"]
        and authority.get("permitted_transaction_indices") == PERMITTED_INDICES
        and authority.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and authority.get("package_seal") == file_record(PACKAGE_SEAL)
        and authority.get("package_manifest") == file_record(PACKAGE_MANIFEST)
        and authority.get("package_acceptance") == file_record(PACKAGE_REVIEW)
        and authority.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and authority.get("authoritative_baseline") == file_record(BASELINE)
        and authority.get("host_reviewer_decision")
        == file_record(PACKAGE_HOST_DECISION)
        and authority.get("accepted_package", {}).get("root") == str(PACKAGE)
        and authority.get("accepted_package", {}).get("tree")
        == tree_record(PACKAGE)
        and authority.get("authoritative_parent") == authoritative_parent()
        and authority.get("evidence_bindings", {}).get("layer_index") == LAYER_INDEX
        and authority.get("evidence_bindings", {}).get("position004_input_state")
        == manifest["position004_input_state"]
        and authority.get("evidence_bindings", {}).get("official_frozen_evidence")
        == manifest["official_frozen_evidence"]
        and authority.get("authorized_launch") == expected_launch()
        and authority.get("output_namespaces", {}).get("transaction007")
        == str(TRANSACTION7)
        and authority.get("output_namespaces", {}).get("generation8")
        == str(GENERATION8)
        and authority.get("activity_counters") == ZERO_COUNTERS
        and authority.get("authority_consumed") is False
        and authority.get("retry_authorized") is False
        and authority.get("replay_authorized") is False
        and authority.get("resume_authorized") is False
        and authority.get("preconsumption_workload_authorized") is False
        and authority.get("synthesis_authorized") is False
        and authority.get("u280_authorized") is False
        and authority.get("transaction007_executed") is False
        and authority.get("generation8_exists") is False
        and authority.get("transactions008_025_absent") is True,
        "authority semantics differ",
    )


def validate_authority_document(
    authority: Mapping[str, Any],
    expected: Mapping[str, Any],
    workload_call: Callable[[], None] | None = None,
) -> None:
    require(dict(authority) == dict(expected), "authority boundary differs")
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
        validate_authority_document(candidate, authority, workload_call)
    except ReviewError:
        require(workload_calls == 0, f"hostile control reached workload: {label}")
        return label, workload_calls
    raise ReviewError(f"hostile control accepted: {label}")


def adversarial_controls(
    authority: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("wrong-parent-generation", lambda item: item.update({"authoritative_generation": 6})),
        ("wrong-parent-cursor", lambda item: item.update({"authoritative_cursor": 6})),
        ("wrong-parent-checkpoint", lambda item: item.update({"required_parent_checkpoint_index": 5})),
        ("wrong-parent-pointer", lambda item: item["authoritative_parent"]["pointer"].update({"sha256": "0" * 64})),
        ("wrong-layer", lambda item: item.update({"layer_index": 5})),
        ("wrong-state-evidence", lambda item: item["evidence_bindings"]["position004_input_state"].update({"sha256": "0" * 64})),
        ("wrong-model-evidence", lambda item: item["evidence_bindings"]["official_frozen_evidence"]["model_checkpoint"].update({"sha256": "0" * 64})),
        ("wrong-layer-evidence", lambda item: item["evidence_bindings"].update({"layer_index": 5})),
        ("altered-argv", lambda item: item["authorized_launch"]["argv"].append("--altered")),
        ("altered-cwd", lambda item: item["authorized_launch"].update({"cwd": "/tmp"})),
        ("altered-environment", lambda item: item["authorized_launch"]["environment"].update({"PYTHONHASHSEED": "1"})),
        ("retry", lambda item: item.update({"retry_authorized": True})),
        ("replay", lambda item: item.update({"replay_authorized": True})),
        ("resume", lambda item: item.update({"resume_authorized": True})),
        ("pre-consumption-workload", lambda item: item.update({"preconsumption_workload_authorized": True})),
        ("pre-consumption-counter", lambda item: item["activity_counters"].update({"payload_execution": 1})),
        ("authority-consumed", lambda item: item.update({"authority_consumed": True})),
        ("wrong-generation8-target", lambda item: item.update({"target_generation": 9})),
        ("wrong-checkpoint007-target", lambda item: item.update({"target_checkpoint_index": 8})),
        ("wrong-generation8-namespace", lambda item: item["output_namespaces"].update({"generation8": str(GENERATION7)})),
        ("synthesis", lambda item: item.update({"synthesis_authorized": True})),
        ("u280", lambda item: item.update({"u280_authorized": True})),
    ]
    mutations.extend(
        (
            f"transaction{index:03d}-scope",
            lambda item, index=index: item.update(
                {"permitted_transaction_indices": [7, index]}
            ),
        )
        for index in range(8, 26)
    )
    controls: list[str] = []
    workload_counts: dict[str, int] = {}
    for label, mutation in mutations:
        name, calls = expect_rejection(authority, label, mutation)
        controls.append(name)
        workload_counts[name] = calls
    require(len(controls) == 40, "hostile control cardinality differs")
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
            AUTHORITY_PACKAGE.is_dir()
            and not AUTHORITY_PACKAGE.is_symlink()
            and stat.S_IMODE(AUTHORITY_PACKAGE.stat().st_mode) & 0o222 == 0,
            "authority package is absent or writable",
        )
        actual = {
            path.name for path in AUTHORITY_PACKAGE.iterdir() if path.is_file()
        }
        require(
            actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
            "authority package file set differs",
        )
        validate_accepted_boundary()
        seal = load_json(AUTHORITY_SEAL)
        require(
            seal.get("kind")
            == "ace3_position3_transaction7_layer6_manager_authority_seal"
            and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
            and set(seal.get("members", {})) == AUTHORITY_MEMBERS
            and seal.get("package_tree") == tree_record(PACKAGE)
            and seal.get("package_seal") == file_record(PACKAGE_SEAL)
            and seal.get("package_acceptance") == file_record(PACKAGE_REVIEW)
            and seal.get("host_reviewer_decision")
            == file_record(PACKAGE_HOST_DECISION)
            and seal.get("activity_counters") == ZERO_COUNTERS
            and seal.get("authority_cardinality") == 1
            and seal.get("authority_consumed") is False
            and seal.get("retry_authorized") is False
            and seal.get("replay_authorized") is False
            and seal.get("resume_authorized") is False
            and seal.get("preconsumption_workload_authorized") is False
            and seal.get("synthesis_authorized") is False
            and seal.get("u280_authorized") is False
            and seal.get("transaction007_executed") is False
            and seal.get("generation8_exists") is False
            and seal.get("transactions008_025_absent") is True,
            "authority seal differs",
        )
        for name, record in seal["members"].items():
            path = AUTHORITY_PACKAGE / name
            authenticate(record, f"authority member {name}", path)
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"authority member is writable: {name}",
            )
        request = load_json(AUTHORITY_PACKAGE / "review-request.json")
        require(
            request.get("kind")
            == "ace3_position3_transaction7_layer6_authority_review_request"
            and request.get("mission_id") == MISSION_ID
            and request.get("required_role") == "reviewer"
            and request.get("review_output") == str(output)
            and request.get("authority") == file_record(AUTHORITY)
            and request.get("package_acceptance") == file_record(PACKAGE_REVIEW)
            and request.get("host_reviewer_decision")
            == file_record(PACKAGE_HOST_DECISION)
            and request.get("package_tree") == tree_record(PACKAGE),
            "authority review request differs",
        )
        source_boundary(
            (AUTHORITY_PACKAGE / "review-emitter.py").read_text(encoding="utf-8")
        )
        authority = load_json(AUTHORITY)
        validate_authority_semantics(authority)
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
        authority_package, output
    )
    authority = load_json(AUTHORITY)
    review = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction7_layer6_authority_independent_review",
        "status": status,
        "review_type": "source-disjoint-executable-authority-review",
        "producer_role": "reviewer",
        "mission_id": MISSION_ID,
        "preparation_participation": False,
        "authority_seal": file_record(AUTHORITY_SEAL),
        "authority": file_record(AUTHORITY),
        "package_tree": tree_record(PACKAGE),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "package_acceptance": file_record(PACKAGE_REVIEW),
        "host_package_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
        "authoritative_parent": authoritative_parent(),
        "authorized_launch": copy.deepcopy(authority["authorized_launch"]),
        "target_generation": 8,
        "target_cursor": 8,
        "target_checkpoint_index": 7,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "authority_cardinality": 1,
        "authority_consumed": False,
        "retry_authorized": False,
        "replay_authorized": False,
        "resume_authorized": False,
        "preconsumption_workload_authorized": False,
        "synthesis_authorized": False,
        "u280_authorized": False,
        "transaction007_executed": False,
        "generation8_exists": False,
        "transactions008_025_absent": True,
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
            "TRANSACTION007_LAYER06_AUTHORITY_ASSESSMENT_PASS "
            f"controls={len(controls)} "
            f"workload_calls={sum(workload_counts.values())}"
        )
        return
    emit_review(arguments.authority_package, arguments.output)
    print(f"TRANSACTION007_LAYER06_AUTHORITY_REVIEW_WRITTEN output={arguments.output}")


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
            f"TRANSACTION007_LAYER06_AUTHORITY_REVIEW_REFUSED {error}"
        ) from error
