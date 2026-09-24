#!/usr/bin/env python3
"""Issue the sole transaction007/layer06 Manager execution authority."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping


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

REVIEWER_SOURCE = ROOT / "ace3/model/emit_transaction7_layer6_authority_review.py"
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


class AuthorityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


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


def file_record(
    path: Path, published_path: Path | None = None
) -> dict[str, Any]:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    return {
        "path": str(published_path or path),
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
    require(
        all(not path.is_symlink() for path in files),
        f"symlinked tree member: {root}",
    )
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


def validate_mission() -> dict[str, Any]:
    record = require_fixed(MISSION)
    mission = load_json(MISSION)
    objective = mission.get("objective", "")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx007-layer06-authority"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and "manager-direct" in mission.get("tags", [])
        and isinstance(objective, str)
        and "exactly one fresh Manager execution authority" in objective
        and "AUTHORIZED_NOT_CONSUMED" in objective,
        "transaction007 authority mission differs",
    )
    return {"mission": record, "mission_id": MISSION_ID, "manager_direct": True}


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


def validate_parent() -> None:
    for path in (
        POINTER,
        GENERATION7 / "generation-manifest.json",
        GENERATION7 / "ledger.json",
        GENERATION7 / "checkpoints/transaction-006.json",
        TRANSACTIONS / "transaction-006/position004.state",
    ):
        require_fixed(path)
    pointer = load_json(POINTER)
    generation = load_json(GENERATION7 / "generation-manifest.json")
    ledger = load_json(GENERATION7 / "ledger.json")
    checkpoint = load_json(GENERATION7 / "checkpoints/transaction-006.json")
    require(
        pointer.get("kind")
        == "ace3_position3_transaction6_generation7_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
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
        and checkpoint.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 6
        and checkpoint.get("outputs", {}).get("state")
        == file_record(TRANSACTIONS / "transaction-006/position004.state"),
        "generation7/cursor7/checkpoint006 parent differs",
    )


def validate_package() -> dict[str, Any]:
    require(
        PACKAGE.is_dir()
        and not PACKAGE.is_symlink()
        and stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "sealed transaction007 package is absent or writable",
    )
    for path in (
        MISSION,
        PACKAGE_MANAGER_MISSION,
        PACKAGE_HOST_DECISION,
        PACKAGE_SEAL,
        PACKAGE_MANIFEST,
        SOURCE_MANIFEST,
        BASELINE,
        PACKAGE_REVIEW,
    ):
        require_fixed(path)
    manifest = load_json(PACKAGE_MANIFEST)
    seal = load_json(PACKAGE_SEAL)
    review = load_json(PACKAGE_REVIEW)
    host_decision = load_json(PACKAGE_HOST_DECISION)
    require(
        manifest.get("kind")
        == "ace3_position3_transaction7_layer6_executor_package"
        and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("mission_id") == PACKAGE_MISSION_ID
        and manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and manifest.get("authoritative_generation") == 7
        and manifest.get("authoritative_cursor") == 7
        and manifest.get("required_parent_checkpoint_index") == 6
        and manifest.get("transaction_index") == TRANSACTION_INDEX
        and manifest.get("layer_index") == LAYER_INDEX
        and manifest.get("permitted_transaction_indices") == PERMITTED_INDICES
        and manifest.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and manifest.get("future_manager_authority") == str(AUTHORITY)
        and manifest.get("generation8_output_namespace") == str(GENERATION8)
        and manifest.get("transaction007_output_namespace") == str(TRANSACTION7)
        and manifest.get("launch") == expected_launch()
        and manifest.get("execution_authorized") is False
        and manifest.get("authority_created") is False
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "sealed transaction007 package identity or launch differs",
    )
    require(
        seal.get("kind") == "ace3_position3_transaction7_layer6_executor_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and set(seal.get("members", {})) == set(PACKAGE_MEMBERS)
        and seal.get("execution_authorized") is False
        and seal.get("authority_created") is False
        and seal.get("activity_counters") == ZERO_COUNTERS,
        "transaction007 package seal differs",
    )
    for label, relative in PACKAGE_MEMBERS.items():
        path = PACKAGE / relative
        authenticate(seal["members"][label], f"package member {label}", path)
        require(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
            f"package member is writable: {label}",
        )
    require(
        review.get("kind")
        == "ace3_position3_transaction7_layer6_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == PACKAGE_MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and review.get("authoritative_baseline") == file_record(BASELINE)
        and review.get("permitted_launch") == expected_launch()
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction007_executed") is False
        and review.get("generation8_published") is False
        and review.get("transactions008_025_absent") is True
        and review.get("this_review_authorizes_execution") is False,
        "immutable transaction007 package PASS differs",
    )
    require(
        stat.S_IMODE(PACKAGE_REVIEW.stat().st_mode) & 0o222 == 0
        and host_decision.get("kind") == "round_reviewed_handoff"
        and host_decision.get("mission_id") == PACKAGE_MISSION_ID
        and host_decision.get("producer_role") == "reviewer"
        and host_decision.get("review", {}).get("status") == "done",
        "Host-saved package Reviewer PASS differs",
    )
    return manifest


def authority_paths() -> list[Path]:
    return sorted(
        RUNTIME.glob(
            "transaction7-layer6-continuation-authority*/manager-authority.json"
        )
    )


def validate_live_absence(authority_count: int) -> None:
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
    authorities = authority_paths()
    require(
        len(authorities) == authority_count
        and (authority_count == 0 or authorities == [AUTHORITY]),
        "transaction007 authority cardinality differs",
    )


def accepted_package() -> dict[str, Any]:
    return {
        "root": str(PACKAGE),
        "tree": tree_record(PACKAGE),
        "package_seal": file_record(PACKAGE_SEAL),
        "package_manifest": file_record(PACKAGE_MANIFEST),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "authoritative_baseline": file_record(BASELINE),
        "independent_review": file_record(PACKAGE_REVIEW),
        "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
    }


def authority_document(
    manifest: Mapping[str, Any], authority_path: Path = AUTHORITY
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction7_layer6_manager_authorization",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "mission_id": MISSION_ID,
        "manager_provenance": validate_mission(),
        "authority_cardinality": 1,
        "runtime_identity": RUNTIME_IDENTITY,
        "authoritative_generation": 7,
        "authoritative_cursor": 7,
        "required_parent_checkpoint_index": 6,
        "target_generation": 8,
        "target_cursor": 8,
        "target_checkpoint_index": 7,
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
        "accepted_package": accepted_package(),
        "authoritative_parent": authoritative_parent(),
        "evidence_bindings": {
            "layer_index": LAYER_INDEX,
            "position004_input_state": copy.deepcopy(
                manifest["position004_input_state"]
            ),
            "official_frozen_evidence": copy.deepcopy(
                manifest["official_frozen_evidence"]
            ),
        },
        "authorized_launch": copy.deepcopy(manifest["launch"]),
        "output_namespaces": {
            "transaction007": str(TRANSACTION7),
            "generation8": str(GENERATION8),
            "checkpoint007": str(
                GENERATION8 / "checkpoints/transaction-007.json"
            ),
            "authority": str(authority_path),
            "authority_consumption": str(
                FUTURE_ROOT / "manager-authorization-consumption.json"
            ),
        },
        "prohibitions": [
            "transaction000-006 retry, replay, or resume",
            "transaction008-025 execution",
            "any launch argv, cwd, or environment other than the exact bound launch",
            "any production workload before atomic authority consumption",
            "authority replay or second consumption",
            "synthesis or U280 execution",
            "model, oracle, vector, compile, simulation, or submission during review",
        ],
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
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
        "reviewer_acceptance": {
            "required": True,
            "path": str(AUTHORITY_REVIEW),
            "present_at_issuance": False,
        },
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_new(source: Path, destination: Path) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise AuthorityError(f"authority destination exists: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def validate_authority_package() -> dict[str, Any]:
    manifest = validate_package()
    validate_mission()
    validate_parent()
    validate_live_absence(1)
    require(
        AUTHORITY_PACKAGE.is_dir()
        and not AUTHORITY_PACKAGE.is_symlink()
        and stat.S_IMODE(AUTHORITY_PACKAGE.stat().st_mode) & 0o222 == 0,
        "authority package is absent, misplaced, or writable",
    )
    actual = {path.name for path in AUTHORITY_PACKAGE.iterdir() if path.is_file()}
    require(
        actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
        "authority package file set differs",
    )
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
    authority = load_json(AUTHORITY)
    require(
        authority == authority_document(manifest),
        "Manager authority document differs",
    )
    request = load_json(AUTHORITY_PACKAGE / "review-request.json")
    require(
        request.get("kind")
        == "ace3_position3_transaction7_layer6_authority_review_request"
        and request.get("mission_id") == MISSION_ID
        and request.get("required_role") == "reviewer"
        and request.get("review_output") == str(AUTHORITY_REVIEW)
        and request.get("authority") == file_record(AUTHORITY)
        and request.get("package_acceptance") == file_record(PACKAGE_REVIEW)
        and request.get("host_reviewer_decision")
        == file_record(PACKAGE_HOST_DECISION)
        and request.get("review_invocation")
        == [
            "/usr/bin/python3",
            str(AUTHORITY_PACKAGE / "review-emitter.py"),
            "--authority-package",
            str(AUTHORITY_PACKAGE),
            "--output",
            str(AUTHORITY_REVIEW),
        ],
        "authority review request differs",
    )
    return authority


def issue_authority() -> Path:
    require(not AUTHORITY_PACKAGE.exists(), f"authority exists: {AUTHORITY_PACKAGE}")
    require(not AUTHORITY_REVIEW.exists(), f"review exists: {AUTHORITY_REVIEW}")
    require(
        not AUTHORITY_REVIEW.parent.exists(),
        f"review namespace exists: {AUTHORITY_REVIEW.parent}",
    )
    require(
        REVIEWER_SOURCE.is_file() and not REVIEWER_SOURCE.is_symlink(),
        "source-disjoint reviewer source is absent or symlinked",
    )
    manifest = validate_package()
    validate_mission()
    validate_parent()
    validate_live_absence(0)

    staging = AUTHORITY_PACKAGE.with_name(f".{AUTHORITY_PACKAGE.name}.preparing")
    require(not staging.exists(), f"authority staging exists: {staging}")
    AUTHORITY_REVIEW.parent.mkdir(mode=0o700, parents=False)
    staging.mkdir(mode=0o700)
    try:
        write_new(
            staging / "manager-authority.json",
            canonical_json(authority_document(manifest)),
        )
        write_new(staging / "review-emitter.py", REVIEWER_SOURCE.read_bytes())
        request = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction7_layer6_authority_review_request",
            "mission_id": MISSION_ID,
            "required_role": "reviewer",
            "preparation_participation_required": False,
            "review_output": str(AUTHORITY_REVIEW),
            "authority": file_record(staging / "manager-authority.json", AUTHORITY),
            "package_acceptance": file_record(PACKAGE_REVIEW),
            "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
            "package_tree": tree_record(PACKAGE),
            "reviewer_emitter": file_record(
                staging / "review-emitter.py",
                AUTHORITY_PACKAGE / "review-emitter.py",
            ),
            "review_invocation": [
                "/usr/bin/python3",
                str(AUTHORITY_PACKAGE / "review-emitter.py"),
                "--authority-package",
                str(AUTHORITY_PACKAGE),
                "--output",
                str(AUTHORITY_REVIEW),
            ],
            "requested_judgment": "PASS_OR_REJECT",
            "review_must_not": [
                "consume, retry, replay, or resume authority",
                "execute transaction007 or transaction008-025",
                "create transaction007, generation8, or runtime evidence",
                "execute model, oracle, vectors, compile, simulation, synthesis, U280, or submission",
                "mutate generations3-7 or transactions000-006",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(request))
        members = {
            name: file_record(staging / name, AUTHORITY_PACKAGE / name)
            for name in AUTHORITY_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction7_layer6_manager_authority_seal",
            "status": "AUTHORIZED_NOT_CONSUMED",
            "members": members,
            "package_tree": tree_record(PACKAGE),
            "package_seal": file_record(PACKAGE_SEAL),
            "package_acceptance": file_record(PACKAGE_REVIEW),
            "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
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
        }
        write_new(staging / "authority-seal.json", canonical_json(seal))
        fsync_directory(staging)
        rename_new(staging, AUTHORITY_PACKAGE)
        fsync_directory(AUTHORITY_PACKAGE.parent)
        AUTHORITY_PACKAGE.chmod(0o555)
        AUTHORITY_REVIEW.parent.chmod(0o755)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_authority_package()
    return AUTHORITY_PACKAGE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("issue", "validate"))
    arguments = parser.parse_args()
    if arguments.command == "issue":
        output = issue_authority()
        print(f"TRANSACTION007_LAYER06_AUTHORITY_ISSUED output={output}")
    else:
        validate_authority_package()
        print(f"TRANSACTION007_LAYER06_AUTHORITY_VALID path={AUTHORITY_PACKAGE}")


if __name__ == "__main__":
    try:
        main()
    except (
        AuthorityError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"TRANSACTION007_LAYER06_AUTHORITY_REFUSED {error}"
        ) from error
