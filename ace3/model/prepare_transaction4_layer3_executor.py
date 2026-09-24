#!/usr/bin/env python3
"""Prepare and later execute only position-3 transaction004/layer03."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "06a9a74e72a0"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATIONS = ADOPTION / "state-generations"
GENERATION3 = GENERATIONS / "generation-0000000003"
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION5_STAGING = GENERATIONS / ".generation-0000000005.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION4 = TRANSACTIONS / "transaction-004"
RUNTIME_LOCK = ADOPTION / "execution.lock"
FUTURE_ROOT = RUNTIME / "transaction4-authoritative-generation5"
CONSUMPTION = FUTURE_ROOT / "manager-authorization-consumption.json"
EXECUTION_START = FUTURE_ROOT / "execution-start.json"
RECEIPT_CANDIDATE = FUTURE_ROOT / "receipt-candidate.json"
EXECUTION_EVIDENCE = FUTURE_ROOT / "execution-evidence.json"
TERMINAL = FUTURE_ROOT / "terminal.json"
FAILURE_TERMINAL = FUTURE_ROOT / "fail-closed-terminal.json"

PACKAGE_ROOT = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_packages"
    / "generation4-cursor4-transaction004-layer03-r1"
)
REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_reviews"
    / "generation4-cursor4-transaction004-layer03-r1"
    / "independent-review.json"
)
AUTHORITY = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_authorities"
    / "generation4-cursor4-transaction004-layer03-r1"
    / "manager-authority.json"
)
EXECUTOR = PACKAGE_ROOT / "transaction4_executor.py"
BASELINE = PACKAGE_ROOT / "authoritative-baseline.json"
SOURCE_MANIFEST = PACKAGE_ROOT / "source-manifest.json"
PACKAGE_MANIFEST = PACKAGE_ROOT / "package-manifest.json"
REVIEW_REQUEST = PACKAGE_ROOT / "review-request.json"
PACKAGE_SEAL = PACKAGE_ROOT / "package-seal.json"
REVIEW_EMITTER = PACKAGE_ROOT / "review-emitter.py"

ORIGINAL_EXECUTOR = RUNTIME / "executor.py"
ORIGINAL_INVOCATION = RUNTIME / "invocation.json"
EXACT_EXECUTOR = RUNTIME / "recovery-r12/recovery_executor.py"
ACCEPTED_SOURCE_MANIFEST = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_reviews"
    / "authoritative-generation3-transaction3-only-r5-absence-predicate-repair-final"
    / "source-manifest.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT / "ace3/model/emit_transaction4_layer3_executor_review.py"
)
PYTHON = Path("/home/argustest/miniconda3/bin/python3")

TRANSACTION_INDEX = 4
LAYER_INDEX = 3
START_CURSOR = 4
EXIT_CURSOR = 5
PERMITTED_INDICES = [TRANSACTION_INDEX]
FORBIDDEN_INDICES = [*range(4), *range(5, 26)]

ZERO_COUNTERS = {
    "authority_issuance": 0,
    "authority_consumption": 0,
    "model_execution": 0,
    "payload_execution": 0,
    "reference_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "runtime_mutation": 0,
    "submission": 0,
    "transaction_execution": 0,
}

FIXED_SHA256 = {
    POINTER: "7d7d45423d4e6eb008eef3dd53c5286a9b74e24b56b5737ee56492365be1095a",
    GENERATION3
    / "generation-manifest.json": "bade4d17d6d9fe627ecb9c34f39b08b50cad85341f897be6d38ffbd75096b02c",
    GENERATION3
    / "ledger.json": "f4c44a7717d392dd8e328d4430a62d233c23865efb0bd59f48718d0f31af0af2",
    GENERATION3
    / "checkpoints/transaction-000.json": "9fabdd9a91f1331220ca4e2c5682cc17e8be18fed444799f97cd3c66276cbdd0",
    GENERATION3
    / "checkpoints/transaction-001.json": "6019478026ea51b2cefd34e4a2e261edca8cb1f02034c1c5ed8fb20e0c3250a7",
    GENERATION3
    / "checkpoints/transaction-002.json": "31330e3e421ced0289415a2e864ca428535c01a97f2c7e157fa2bfeda82a7552",
    GENERATION4
    / "generation-manifest.json": "b106d6232b26f72f191b1e91a6137cd8a4bbd4e4730e615b549c5a9ad3fe390e",
    GENERATION4
    / "ledger.json": "e87086bb1b752759f15f5b117542fccaa718b34e23af3333f37131997472752a",
    GENERATION4
    / "checkpoints/transaction-000.json": "9fabdd9a91f1331220ca4e2c5682cc17e8be18fed444799f97cd3c66276cbdd0",
    GENERATION4
    / "checkpoints/transaction-001.json": "6019478026ea51b2cefd34e4a2e261edca8cb1f02034c1c5ed8fb20e0c3250a7",
    GENERATION4
    / "checkpoints/transaction-002.json": "31330e3e421ced0289415a2e864ca428535c01a97f2c7e157fa2bfeda82a7552",
    GENERATION4
    / "checkpoints/transaction-003.json": "5b0a0a4596f43338d27a5257a6fbe4bb2fadefbd4dedc14605aa6b7249387d62",
}

PACKAGE_MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction4_executor.py",
    "package_manifest": "package-manifest.json",
    "review_emitter": "review-emitter.py",
    "review_request": "review-request.json",
    "source_manifest": "source-manifest.json",
}


class Transaction4Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Transaction4Error(message)


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
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, published_path: Path | None = None) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    return {
        "path": str(published_path if published_path is not None else path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate(
    record: Mapping[str, Any],
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
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
    actual = file_record(path)
    require(actual == dict(record), f"{label} content binding mismatch")
    return actual


def require_fixed(path: Path) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == FIXED_SHA256[path], f"fixed hash mismatch: {path}")
    return record


def tree_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"tree root invalid: {root}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        record = file_record(path)
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += record["bytes"]
    return {
        "root": str(root),
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def transaction_descriptor() -> dict[str, Any]:
    invocation = load_json(ORIGINAL_INVOCATION)
    transactions = invocation.get("transactions")
    require(
        isinstance(transactions, list)
        and len(transactions) == 26
        and transactions[TRANSACTION_INDEX].get("transaction_index")
        == TRANSACTION_INDEX,
        "transaction004 descriptor list differs",
    )
    descriptor = copy.deepcopy(transactions[TRANSACTION_INDEX])
    require(
        descriptor.get("operation") == "position3-decoder-layer"
        and descriptor.get("layer_index") == LAYER_INDEX
        and descriptor.get("inputs", {})
        .get("predecessor", {})
        .get("source_transaction_index")
        == 3
        and descriptor.get("inputs", {}).get("transaction_position") == 3,
        "transaction004/layer03 scope differs",
    )
    return descriptor


def expected_argv(command: str = "execute") -> list[str]:
    return [
        str(PYTHON),
        str(EXECUTOR),
        command,
        "--package",
        str(PACKAGE_ROOT),
        "--review",
        str(REVIEW),
        "--authorization",
        str(AUTHORITY),
        "--runtime-root",
        str(RUNTIME),
        "--transaction-index",
        str(TRANSACTION_INDEX),
    ]


def launch_contract() -> dict[str, Any]:
    return {
        "cwd": str(ROOT),
        "argv": expected_argv(),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "interpreter": str(PYTHON),
        "transaction_index": TRANSACTION_INDEX,
        "layer_index": LAYER_INDEX,
        "start_generation": START_CURSOR,
        "start_cursor": START_CURSOR,
        "exit_generation": EXIT_CURSOR,
        "exit_cursor": EXIT_CURSOR,
    }


def validate_generation4() -> dict[str, Any]:
    pointer_record = require_fixed(POINTER)
    pointer = load_json(POINTER)
    generation_manifest_record = require_fixed(
        GENERATION4 / "generation-manifest.json"
    )
    require(
        pointer.get("kind")
        == "ace3_position3_transaction3_generation4_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and pointer.get("generation") == START_CURSOR
        and pointer.get("generation_manifest") == generation_manifest_record,
        "authoritative pointer is not generation4/cursor4",
    )
    generation_manifest = load_json(GENERATION4 / "generation-manifest.json")
    require(
        generation_manifest.get("kind")
        == "ace3_transaction3_publication_recovery_state_generation"
        and generation_manifest.get("status") == "PREPARED"
        and generation_manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and generation_manifest.get("generation") == START_CURSOR,
        "generation4 manifest identity differs",
    )
    for record in generation_manifest.get("files", {}).values():
        authenticate(record, "generation4 manifest file")
    checkpoints = [
        require_fixed(GENERATION4 / f"checkpoints/transaction-{index:03d}.json")
        for index in range(4)
    ]
    require(
        sorted(path.name for path in (GENERATION4 / "checkpoints").iterdir())
        == [f"transaction-{index:03d}.json" for index in range(4)],
        "generation4 checkpoint set differs",
    )
    ledger_record = require_fixed(GENERATION4 / "ledger.json")
    ledger = load_json(GENERATION4 / "ledger.json")
    require(
        ledger.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == START_CURSOR
        and ledger.get("completed_transaction_count") == START_CURSOR
        and ledger.get("next_transaction_index") == START_CURSOR
        and ledger.get("transaction_count") == 26
        and ledger.get("authoritative_state_root") == str(GENERATION4)
        and ledger.get("completed_receipts") == checkpoints,
        "generation4 ledger is not cursor4",
    )
    require(
        pointer_record["sha256"] == FIXED_SHA256[POINTER]
        and ledger_record["sha256"] == FIXED_SHA256[GENERATION4 / "ledger.json"],
        "generation4 fixed parent differs",
    )
    for index in range(3):
        require_fixed(GENERATION3 / f"checkpoints/transaction-{index:03d}.json")
        require(
            (
                GENERATION3 / f"checkpoints/transaction-{index:03d}.json"
            ).read_bytes()
            == (
                GENERATION4 / f"checkpoints/transaction-{index:03d}.json"
            ).read_bytes(),
            f"checkpoint{index:03d} changed across generation3/generation4",
        )
    require_fixed(GENERATION3 / "generation-manifest.json")
    require_fixed(GENERATION3 / "ledger.json")
    checkpoint3 = load_json(
        GENERATION4 / "checkpoints/transaction-003.json"
    )
    require(
        checkpoint3.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint3.get("status") == "COMPLETE"
        and checkpoint3.get("transaction_index") == 3
        and checkpoint3.get("operation") == "position3-decoder-layer"
        and checkpoint3.get("result", {}).get("exact_integer_oracle_match") is True
        and checkpoint3.get("result", {}).get("natural_rtl_terminal") is True,
        "checkpoint003 is not the accepted transaction003 completion",
    )
    return ledger


def validate_absence() -> None:
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(4, 26)
        ),
        "transaction004-025 namespace is not absent",
    )
    require(
        not GENERATION5.exists()
        and not GENERATION5_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "transaction004 generation or runtime namespace already exists",
    )


def baseline_document() -> dict[str, Any]:
    validate_generation4()
    validate_absence()
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_layer3_authoritative_baseline",
        "runtime_identity": RUNTIME_IDENTITY,
        "authoritative_generation": START_CURSOR,
        "authoritative_cursor": START_CURSOR,
        "pointer": file_record(POINTER),
        "generation3": {
            "manifest": file_record(GENERATION3 / "generation-manifest.json"),
            "ledger": file_record(GENERATION3 / "ledger.json"),
            "checkpoints": [
                file_record(
                    GENERATION3 / f"checkpoints/transaction-{index:03d}.json"
                )
                for index in range(3)
            ],
        },
        "generation4": {
            "manifest": file_record(GENERATION4 / "generation-manifest.json"),
            "ledger": file_record(GENERATION4 / "ledger.json"),
            "checkpoints": [
                file_record(
                    GENERATION4 / f"checkpoints/transaction-{index:03d}.json"
                )
                for index in range(4)
            ],
        },
        "transactions000_003": [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(4)
        ],
        "checkpoint003_parent": file_record(
            GENERATION4 / "checkpoints/transaction-003.json"
        ),
        "transaction004_absent": True,
        "transactions005_025_absent": True,
        "generation5_absent": True,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
    }


def validate_baseline(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction4_layer3_authoritative_baseline"
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("activity_counters") == ZERO_COUNTERS
        and document.get("transaction004_absent") is True
        and document.get("transactions005_025_absent") is True
        and document.get("generation5_absent") is True,
        "authoritative baseline identity or zero state differs",
    )
    authenticate(document["pointer"], "baseline pointer", POINTER)
    for section, generation, count in (
        ("generation3", GENERATION3, 3),
        ("generation4", GENERATION4, 4),
    ):
        authenticate(
            document[section]["manifest"],
            f"{section} manifest",
            generation / "generation-manifest.json",
        )
        authenticate(
            document[section]["ledger"],
            f"{section} ledger",
            generation / "ledger.json",
        )
        for index, record in enumerate(document[section]["checkpoints"]):
            require(index < count, f"{section} checkpoint count differs")
            authenticate(
                record,
                f"{section} checkpoint{index:03d}",
                generation / f"checkpoints/transaction-{index:03d}.json",
            )
        require(
            len(document[section]["checkpoints"]) == count,
            f"{section} checkpoint count differs",
        )
    authenticate(
        document["checkpoint003_parent"],
        "checkpoint003 parent",
        GENERATION4 / "checkpoints/transaction-003.json",
    )
    require(
        document.get("transactions000_003")
        == [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(4)
        ],
        "transactions000-003 tree changed",
    )


def resolved_tool_record(requested: Path) -> dict[str, Any]:
    resolved = requested.resolve(strict=True)
    return {
        "requested_path": str(requested),
        "resolved": file_record(resolved),
    }


def build_source_manifest(executor_path: Path, published_executor: Path) -> dict[str, Any]:
    accepted = load_json(ACCEPTED_SOURCE_MANIFEST)
    require(
        accepted.get("kind")
        == "ace3_position3_transaction3_r5_exact_source_toolchain_closure",
        "accepted transaction003 source closure differs",
    )
    for label, record in accepted["sources"].items():
        authenticate(record, f"accepted source {label}")
    for label, record in accepted["toolchain"].items():
        requested = Path(record["requested_path"])
        require(
            requested.resolve(strict=True) == Path(record["resolved"]["path"]),
            f"accepted tool resolution differs: {label}",
        )
        authenticate(record["resolved"], f"accepted tool {label}")
    sources = copy.deepcopy(accepted["sources"])
    sources["transaction004_executor"] = file_record(
        executor_path, published_executor
    )
    sources["accepted_transaction003_source_closure"] = file_record(
        ACCEPTED_SOURCE_MANIFEST
    )
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_layer3_exact_source_toolchain_closure",
        "sources": sources,
        "toolchain": copy.deepcopy(accepted["toolchain"]),
    }


def validate_source_manifest(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction4_layer3_exact_source_toolchain_closure"
        and isinstance(document.get("sources"), dict)
        and isinstance(document.get("toolchain"), dict),
        "transaction004 source/toolchain closure differs",
    )
    for label, record in document["sources"].items():
        authenticate(record, f"source {label}")
    for label, record in document["toolchain"].items():
        requested = Path(record["requested_path"])
        require(
            requested.resolve(strict=True) == Path(record["resolved"]["path"]),
            f"tool resolution differs: {label}",
        )
        authenticate(record["resolved"], f"tool {label}")


def validate_manifest_document(document: Mapping[str, Any]) -> None:
    descriptor = transaction_descriptor()
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_transaction4_layer3_executor_package"
        and document.get("status") == "SEALED_REVIEW_REQUIRED"
        and document.get("mission_id") == MISSION_ID
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("required_parent_checkpoint_index") == 3
        and document.get("transaction_index") == TRANSACTION_INDEX
        and document.get("layer_index") == LAYER_INDEX
        and document.get("permitted_transaction_indices") == PERMITTED_INDICES
        and document.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and document.get("transaction_descriptor") == descriptor
        and document.get("launch") == launch_contract()
        and document.get("execution_authorized") is False
        and document.get("authority_created") is False
        and document.get("activity_counters") == ZERO_COUNTERS
        and document.get("prohibitions")
        == [
            "transaction000-003 replay",
            "transaction005-025 execution",
            "broad continuation execution",
            "authority creation by this package",
            "execution before separate Manager authority and Reviewer PASS",
        ],
        "package identity, scope, launch, or zero state differs",
    )


def validate_package(
    package: Path = PACKAGE_ROOT, *, require_zero_runtime: bool = True
) -> dict[str, Any]:
    require(package == PACKAGE_ROOT, "package path differs")
    manifest = load_json(package / "package-manifest.json")
    seal = load_json(package / "package-seal.json")
    source_manifest = load_json(package / "source-manifest.json")
    baseline = load_json(package / "authoritative-baseline.json")
    validate_manifest_document(manifest)
    validate_source_manifest(source_manifest)
    validate_baseline(baseline)
    authenticate(
        manifest["source_manifest"],
        "manifest source closure",
        package / "source-manifest.json",
    )
    authenticate(
        manifest["authoritative_baseline"],
        "manifest baseline",
        package / "authoritative-baseline.json",
    )
    require(
        seal.get("kind")
        == "ace3_position3_transaction4_layer3_executor_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("execution_authorized") is False
        and seal.get("authority_created") is False
        and seal.get("activity_counters") == ZERO_COUNTERS
        and set(seal.get("members", {})) == set(PACKAGE_MEMBERS),
        "package seal differs",
    )
    for label, relative in PACKAGE_MEMBERS.items():
        authenticate(
            seal["members"][label],
            f"sealed {label}",
            package / relative,
        )
    validate_generation4()
    if require_zero_runtime:
        validate_absence()
        require(not AUTHORITY.exists(), "Manager authority exists during package stage")
    return manifest


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_package() -> Path:
    require(not PACKAGE_ROOT.exists(), f"package already exists: {PACKAGE_ROOT}")
    require(not REVIEW.exists(), f"review already exists: {REVIEW}")
    require(not AUTHORITY.exists(), f"authority already exists: {AUTHORITY}")
    baseline = baseline_document()
    PACKAGE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = PACKAGE_ROOT.with_name(f".{PACKAGE_ROOT.name}.preparing")
    require(not staging.exists(), f"package staging exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        executor_payload = Path(__file__).read_bytes()
        write_new(staging / "transaction4_executor.py", executor_payload)
        write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
        write_new(
            staging / "authoritative-baseline.json",
            canonical_json(baseline),
        )
        source_manifest = build_source_manifest(
            staging / "transaction4_executor.py",
            PACKAGE_ROOT / "transaction4_executor.py",
        )
        write_new(
            staging / "source-manifest.json",
            canonical_json(source_manifest),
        )
        package_manifest = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction4_layer3_executor_package",
            "status": "SEALED_REVIEW_REQUIRED",
            "mission_id": MISSION_ID,
            "runtime_identity": RUNTIME_IDENTITY,
            "authoritative_generation": START_CURSOR,
            "authoritative_cursor": START_CURSOR,
            "required_parent_checkpoint_index": 3,
            "transaction_index": TRANSACTION_INDEX,
            "layer_index": LAYER_INDEX,
            "permitted_transaction_indices": PERMITTED_INDICES,
            "forbidden_transaction_indices": FORBIDDEN_INDICES,
            "transaction_descriptor": transaction_descriptor(),
            "authoritative_baseline": file_record(
                staging / "authoritative-baseline.json",
                PACKAGE_ROOT / "authoritative-baseline.json",
            ),
            "source_manifest": file_record(
                staging / "source-manifest.json",
                PACKAGE_ROOT / "source-manifest.json",
            ),
            "launch": launch_contract(),
            "review_output": str(REVIEW),
            "future_manager_authority": str(AUTHORITY),
            "execution_authorized": False,
            "authority_created": False,
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
            "prohibitions": [
                "transaction000-003 replay",
                "transaction005-025 execution",
                "broad continuation execution",
                "authority creation by this package",
                "execution before separate Manager authority and Reviewer PASS",
            ],
        }
        write_new(
            staging / "package-manifest.json",
            canonical_json(package_manifest),
        )
        review_request = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction4_layer3_review_request",
            "mission_id": MISSION_ID,
            "required_role": "reviewer",
            "review_output": str(REVIEW),
            "reviewer_emitter": file_record(
                staging / "review-emitter.py",
                PACKAGE_ROOT / "review-emitter.py",
            ),
            "review_invocation": [
                "/usr/bin/python3",
                str(PACKAGE_ROOT / "review-emitter.py"),
                "--package",
                str(PACKAGE_ROOT),
                "--output",
                str(REVIEW),
            ],
            "requested_judgment": "PASS_OR_REJECT",
            "this_review_authorizes_execution": False,
            "review_must_not": [
                "create Manager authority",
                "consume authority",
                "execute transaction004",
                "execute transaction005-025",
                "mutate generation3 or generation4",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(review_request))
        members = {
            label: file_record(staging / relative, PACKAGE_ROOT / relative)
            for label, relative in PACKAGE_MEMBERS.items()
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction4_layer3_executor_seal",
            "status": "SEALED_REVIEW_REQUIRED",
            "members": members,
            "execution_authorized": False,
            "authority_created": False,
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        }
        write_new(staging / "package-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, PACKAGE_ROOT)
        fsync_directory(PACKAGE_ROOT.parent)
        PACKAGE_ROOT.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_package()
    return PACKAGE_ROOT


def import_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"module import failed: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def validate_review() -> dict[str, Any]:
    review = load_json(REVIEW)
    require(
        review.get("kind")
        == "ace3_position3_transaction4_layer3_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and review.get("authoritative_baseline") == file_record(BASELINE)
        and review.get("permitted_launch") == launch_contract()
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction004_executed") is False
        and review.get("transactions005_025_absent") is True
        and review.get("this_review_authorizes_execution") is False,
        "independent package review differs",
    )
    return review


def validate_authority() -> dict[str, Any]:
    authority = load_json(AUTHORITY)
    review = validate_review()
    require(
        authority.get("kind")
        == "ace3_position3_transaction4_layer3_manager_authorization"
        and authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("producer_role") == "manager"
        and authority.get("authority_cardinality") == 1
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("transaction_index") == TRANSACTION_INDEX
        and authority.get("layer_index") == LAYER_INDEX
        and authority.get("permitted_transaction_indices") == PERMITTED_INDICES
        and authority.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and authority.get("package_seal") == file_record(PACKAGE_SEAL)
        and authority.get("package_acceptance") == file_record(REVIEW)
        and authority.get("authorized_launch") == launch_contract()
        and authority.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and authority.get("authoritative_parent")
        == {
            "pointer": file_record(POINTER),
            "generation_manifest": file_record(
                GENERATION4 / "generation-manifest.json"
            ),
            "ledger": file_record(GENERATION4 / "ledger.json"),
            "checkpoints": [
                file_record(
                    GENERATION4 / f"checkpoints/transaction-{index:03d}.json"
                )
                for index in range(4)
            ],
        }
        and authority.get("activity_counters") == ZERO_COUNTERS
        and authority.get("authority_consumed") is False
        and authority.get("replay_authorized") is False,
        "Manager transaction004 authority differs",
    )
    host_decision = authority.get("host_reviewer_decision")
    require(isinstance(host_decision, dict), "Host Reviewer decision binding absent")
    authenticate(host_decision, "Host Reviewer decision")
    require(review.get("status") == "PASS", "package acceptance is not PASS")
    return authority


def write_exclusive_json(path: Path, document: object) -> None:
    write_new(path, canonical_json(document))
    fsync_directory(path.parent)


def publish_generation5(
    ledger4: Mapping[str, Any],
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    checkpoint_payloads = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION4 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(4)
    }
    checkpoint_payloads["checkpoints/transaction-004.json"] = canonical_json(
        receipt
    )
    checkpoint_records = [
        {
            "path": str(GENERATION5 / f"checkpoints/transaction-{index:03d}.json"),
            "bytes": len(checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]),
            "sha256": sha256_bytes(
                checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]
            ),
        }
        for index in range(5)
    ]
    ledger5 = copy.deepcopy(dict(ledger4))
    ledger5.update(
        {
            "authoritative_state_root": str(GENERATION5),
            "completed_receipts": checkpoint_records,
            "completed_transaction_count": EXIT_CURSOR,
            "next_transaction_index": EXIT_CURSOR,
            "state_generation": EXIT_CURSOR,
            "status": "IN_PROGRESS",
            "cumulative_execution_seconds": receipt["timing"][
                "cumulative_execution_seconds"
            ],
        }
    )
    files = {**checkpoint_payloads, "ledger.json": canonical_json(ledger5)}
    manifest = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_generation5_state",
        "status": "PREPARED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": EXIT_CURSOR,
        "parent_generation": START_CURSOR,
        "parent_pointer": file_record(POINTER),
        "package_seal": file_record(PACKAGE_SEAL),
        "independent_review": file_record(REVIEW),
        "manager_authorization": file_record(AUTHORITY),
        "authority_consumption": file_record(CONSUMPTION),
        "execution_evidence": file_record(EXECUTION_EVIDENCE),
        "files": {
            relative: {
                "path": str(GENERATION5 / relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for relative, payload in files.items()
        },
        "activity_counters": {
            **ZERO_COUNTERS,
            "authority_consumption": 1,
            "payload_execution": 1,
            "reference_execution": 1,
            "rtl_compile": 1,
            "rtl_simulation": 1,
            "runtime_mutation": 1,
            "transaction_execution": 1,
        },
        "manager_decision": authority.get("decision"),
    }
    manifest_payload = canonical_json(manifest)
    pointer5 = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_generation5_authoritative_pointer",
        "status": "COMMITTED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": EXIT_CURSOR,
        "generation_manifest": {
            "path": str(GENERATION5 / "generation-manifest.json"),
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "previous_authoritative_pointer_sha256": FIXED_SHA256[POINTER],
        "atomic_visibility_contract": (
            "pointer replacement is the sole visibility point from "
            "generation4/cursor4 to generation5/cursor5"
        ),
    }
    GENERATION5_STAGING.mkdir(mode=0o700)
    (GENERATION5_STAGING / "checkpoints").mkdir(mode=0o700)
    for relative, payload in files.items():
        destination = GENERATION5_STAGING / relative
        write_new(destination, payload)
    write_new(
        GENERATION5_STAGING / "generation-manifest.json",
        manifest_payload,
    )
    fsync_directory(GENERATION5_STAGING / "checkpoints")
    fsync_directory(GENERATION5_STAGING)
    os.rename(GENERATION5_STAGING, GENERATION5)
    fsync_directory(GENERATIONS)
    temporary = POINTER.with_name(".authoritative-state.transaction004.tmp")
    write_new(temporary, canonical_json(pointer5), 0o600)
    os.replace(temporary, POINTER)
    fsync_directory(POINTER.parent)


def execute_once() -> None:
    validate_package(require_zero_runtime=False)
    ledger4 = validate_generation4()
    validate_absence()
    authority = validate_authority()
    with RUNTIME_LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Transaction4Error("another transaction invocation is active") from error
        validate_generation4()
        validate_absence()
        FUTURE_ROOT.mkdir(mode=0o700)
        fsync_directory(FUTURE_ROOT.parent)
        try:
            write_exclusive_json(
                CONSUMPTION,
                {
                    "schema_version": 1,
                    "kind": "ace3_position3_transaction4_layer3_authority_consumption",
                    "status": "CONSUMED_ONCE",
                    "runtime_identity": RUNTIME_IDENTITY,
                    "transaction_index": TRANSACTION_INDEX,
                    "layer_index": LAYER_INDEX,
                    "package_seal": file_record(PACKAGE_SEAL),
                    "independent_review": file_record(REVIEW),
                    "manager_authorization": file_record(AUTHORITY),
                    "replay_authorized": False,
                },
            )
            started_ns = time.time_ns()
            write_exclusive_json(
                EXECUTION_START,
                {
                    "schema_version": 1,
                    "kind": "ace3_position3_transaction4_layer3_execution_start",
                    "transaction_index": TRANSACTION_INDEX,
                    "layer_index": LAYER_INDEX,
                    "started_ns": started_ns,
                },
            )
            original = import_module(ORIGINAL_EXECUTOR, "ace3_tx4_original")
            exact = import_module(EXACT_EXECUTOR, "ace3_tx4_exact")
            states = original.seed_reference_states()
            require(len(states) == 24, "independent oracle state count differs")
            receipt = exact.execute_exact_layer_transaction(
                transaction_descriptor(),
                load_json(GENERATION4 / "checkpoints/transaction-003.json"),
                states[LAYER_INDEX],
                ledger4["cumulative_execution_seconds"],
            )
            original.candidate_control.validate_completion_receipt(
                transaction_descriptor(), receipt
            )
            require(
                receipt.get("transaction_index") == TRANSACTION_INDEX
                and receipt.get("result", {}).get("exact_integer_oracle_match") is True
                and receipt.get("result", {}).get("natural_rtl_terminal") is True,
                "transaction004 result differs",
            )
            write_exclusive_json(RECEIPT_CANDIDATE, receipt)
            evidence = {
                "schema_version": 1,
                "kind": "ace3_position3_transaction4_layer3_execution_evidence",
                "transaction_index": TRANSACTION_INDEX,
                "layer_index": LAYER_INDEX,
                "started_ns": started_ns,
                "transaction_tree": tree_digest(TRANSACTION4),
                "cached_evidence": False,
                "model_execution": 0,
                "reference_execution": 1,
                "rtl_compile": 1,
                "rtl_simulation": 1,
                "transaction_execution": 1,
            }
            write_exclusive_json(EXECUTION_EVIDENCE, evidence)
            publish_generation5(ledger4, receipt, authority)
            write_exclusive_json(
                TERMINAL,
                {
                    "schema_version": 1,
                    "kind": "ace3_position3_transaction4_layer3_terminal",
                    "status": "PASS",
                    "runtime_identity": RUNTIME_IDENTITY,
                    "executed_transaction_indices": [TRANSACTION_INDEX],
                    "executed_layer_indices": [LAYER_INDEX],
                    "next_transaction_index": EXIT_CURSOR,
                    "generation": EXIT_CURSOR,
                    "authority_consumption": file_record(CONSUMPTION),
                    "execution_evidence": file_record(EXECUTION_EVIDENCE),
                    "transactions000_003_replayed": False,
                    "transactions005_025_executed": False,
                },
            )
        except BaseException as error:
            if FUTURE_ROOT.exists() and not FAILURE_TERMINAL.exists():
                write_exclusive_json(
                    FAILURE_TERMINAL,
                    {
                        "schema_version": 1,
                        "kind": "ace3_position3_transaction4_layer3_fail_closed_terminal",
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "transaction_index": TRANSACTION_INDEX,
                        "layer_index": LAYER_INDEX,
                    },
                )
            raise


def validate_requested_paths(arguments: argparse.Namespace) -> None:
    require(arguments.package == PACKAGE_ROOT, "package path differs")
    require(arguments.review == REVIEW, "review path differs")
    require(arguments.authorization == AUTHORITY, "authority path differs")
    require(arguments.runtime_root == RUNTIME, "runtime root differs")
    require(
        arguments.transaction_index == TRANSACTION_INDEX,
        "only transaction004 is permitted",
    )
    require(
        [sys.executable, *sys.argv] == expected_argv(arguments.operation),
        "exact argv or interpreter differs",
    )
    require(Path.cwd() == ROOT, "exact repository-root cwd required")
    require(
        os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "deterministic environment differs",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("prepare", "validate-package", "execute"),
    )
    parser.add_argument("--package", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--authorization", type=Path, default=AUTHORITY)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    parser.add_argument("--transaction-index", type=int, default=TRANSACTION_INDEX)
    arguments = parser.parse_args()
    if arguments.operation == "prepare":
        package = prepare_package()
        print(
            "POSITION3_TRANSACTION004_LAYER03_PACKAGE_PREPARED "
            f"package={package} generation=4 cursor=4 checkpoint=003 "
            "authority=0 consumption=0 payload=0 model=0 reference=0 "
            "rtl_compile=0 rtl_simulation=0 transaction=0"
        )
    elif arguments.operation == "validate-package":
        validate_package()
        print(
            "POSITION3_TRANSACTION004_LAYER03_PACKAGE_VALID "
            "generation=4 cursor=4 checkpoint=003 transaction004=absent "
            "transactions005_025=absent authority=0 execution=0"
        )
    else:
        validate_requested_paths(arguments)
        execute_once()
        print(
            "POSITION3_TRANSACTION004_LAYER03_PASS "
            "executed=004 layer=03 generation=5 cursor=5"
        )


if __name__ == "__main__":
    try:
        main()
    except (
        Transaction4Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ArithmeticError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(f"POSITION3_TRANSACTION004_LAYER03_REFUSED {error}") from error
