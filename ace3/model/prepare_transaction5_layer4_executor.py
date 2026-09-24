#!/usr/bin/env python3
"""Seal and later execute only position-3 transaction005/layer04."""

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
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "43c0b6ae6c52"
PARENT_MISSION_ID = "d560dc6d3138"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATIONS = ADOPTION / "state-generations"
GENERATION3 = GENERATIONS / "generation-0000000003"
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION6 = GENERATIONS / "generation-0000000006"
GENERATION6_STAGING = GENERATIONS / ".generation-0000000006.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION5 = TRANSACTIONS / "transaction-005"
RUNTIME_LOCK = ADOPTION / "execution.lock"
FUTURE_ROOT = RUNTIME / "transaction5-authoritative-generation6"
CONSUMPTION = FUTURE_ROOT / "manager-authorization-consumption.json"
EXECUTION_START = FUTURE_ROOT / "execution-start.json"
RECEIPT_CANDIDATE = FUTURE_ROOT / "receipt-candidate.json"
EXECUTION_EVIDENCE = FUTURE_ROOT / "execution-evidence.json"
TERMINAL = FUTURE_ROOT / "terminal.json"
FAILURE_TERMINAL = FUTURE_ROOT / "fail-closed-terminal.json"

PACKAGE_ID = "generation5-cursor5-checkpoint004-transaction005-layer04-r1"
PACKAGE_ROOT = (
    ROOT / "build/model24_selected_token_position3_transaction5_packages" / PACKAGE_ID
)
REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_reviews"
    / PACKAGE_ID
    / "independent-review.json"
)
AUTHORITY = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_authorities"
    / PACKAGE_ID
    / "manager-authority.json"
)
EXECUTOR = PACKAGE_ROOT / "transaction5_executor.py"
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
    / "build/model24_selected_token_position3_transaction4_packages"
    / "generation4-cursor4-transaction004-layer03-r1"
    / "source-manifest.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT / "ace3/model/emit_transaction5_layer4_executor_review.py"
)
PARENT_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/d560dc6d3138/latest.json"
)
PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")
PYTHON = Path("/home/argustest/miniconda3/bin/python3")
MODEL_CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"

TRANSACTION_INDEX = 5
LAYER_INDEX = 4
START_CURSOR = 5
EXIT_CURSOR = 6
POSITION = 3
OUTPUT_POSITION = 4
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
    PARENT_HANDOFF: "8191adb23a56b8ddc463595bdfbd4c414550cfbe8a5c19c11a633f00f255168c",
    PARENT_REVIEW_DECISION: "b6d400df449af5699ce3c84ad001d8e687b54746b3be348ad354b9cc415e902d",
    POINTER: "adee838870e758a5cc32a1a79248938b22d8e9d701501dce525c9098958bdda0",
    GENERATION5
    / "generation-manifest.json": "03f73918812f160cb4b9b7f94b239c5d3efb9cab9d4d41bd621a59eadfc9b599",
    GENERATION5
    / "ledger.json": "ca9397fc658ffe0629c8ba4cade0af5640d85298106b4dfc0dfbeb2ca039af14",
    GENERATION5
    / "checkpoints/transaction-004.json": "ff157b01b83d07d8bf167c420115659072a54415054454674625503d7decd610",
    TRANSACTIONS
    / "transaction-004/position004.state": "bc17b1015874eba58edd49c6b29c74b37057576823711f5983476264de0c65ac",
    ORIGINAL_INVOCATION: "9fb5418fe06c492633399ea79791a08ae25507f3e74b0cf0c003bdb79b43919d",
    MODEL_CHECKPOINT: "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b",
}

PACKAGE_MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction5_executor.py",
    "package_manifest": "package-manifest.json",
    "review_emitter": "review-emitter.py",
    "review_request": "review-request.json",
    "source_manifest": "source-manifest.json",
}


class Transaction5Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Transaction5Error(message)


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
    require_fixed(ORIGINAL_INVOCATION)
    invocation = load_json(ORIGINAL_INVOCATION)
    transactions = invocation.get("transactions")
    require(
        isinstance(transactions, list)
        and len(transactions) == 26
        and transactions[TRANSACTION_INDEX].get("transaction_index")
        == TRANSACTION_INDEX,
        "transaction005 descriptor list differs",
    )
    descriptor = copy.deepcopy(transactions[TRANSACTION_INDEX])
    require(
        descriptor.get("operation") == "position3-decoder-layer"
        and descriptor.get("layer_index") == LAYER_INDEX
        and descriptor.get("inputs", {})
        .get("predecessor", {})
        .get("source_transaction_index")
        == 4
        and descriptor.get("inputs", {}).get("transaction_position") == POSITION,
        "transaction005/layer04 scope differs",
    )
    authenticate(
        descriptor["inputs"]["fixture_manifest"],
        "official layer04 fixture",
    )
    authenticate(
        descriptor["inputs"]["position2_kv_parent"],
        "official layer04 K/V parent",
    )
    return descriptor


def official_frozen_evidence() -> dict[str, Any]:
    invocation = load_json(ORIGINAL_INVOCATION)
    transaction0 = invocation["transactions"][0]
    checkpoint = transaction0["inputs"]["checkpoint"]
    projection = transaction0["inputs"]["tied_weight"]
    authenticate(checkpoint, "official model checkpoint", MODEL_CHECKPOINT)
    require_fixed(MODEL_CHECKPOINT)
    require(
        projection
        == {
            "dtype": "FP16",
            "sha256": "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0",
            "shape": [151936, 896],
            "tensor": "model.embed_tokens.weight",
            "tied_peer": "lm_head.weight",
        },
        "official frozen projection identity differs",
    )
    return {
        "invocation": file_record(ORIGINAL_INVOCATION),
        "model_checkpoint": copy.deepcopy(checkpoint),
        "tied_projection": copy.deepcopy(projection),
        "layer04_fixture_manifest": copy.deepcopy(
            transaction_descriptor()["inputs"]["fixture_manifest"]
        ),
        "layer04_position2_kv_parent": copy.deepcopy(
            transaction_descriptor()["inputs"]["position2_kv_parent"]
        ),
    }


def compile_argv() -> list[str]:
    return [
        "make",
        "--no-print-directory",
        "model24-rtl-layer-compile",
        "MODEL24_RTL_LAYER_INDEX=4",
        "MODEL24_RTL_ACCURATE_SILU=1",
        f"MODEL24_RTL_CASCADE_DIR={TRANSACTION5 / 'build'}",
    ]


def simulation_argv() -> list[str]:
    binary = (
        TRANSACTION5
        / "build/compiled/layer4/obj_dir/Vace3_decoder_layer0_token_engine"
    )
    return [
        str(binary),
        "--layer-index",
        "4",
        "--vector-dir",
        str(TRANSACTION5 / "vectors"),
        "--tensor-dir",
        str(TRANSACTION5 / "vectors"),
        "--raw-dir",
        str(TRANSACTION5 / "position003/raw"),
        "--transaction-position",
        "3",
        "--state-out",
        str(TRANSACTION5 / "position004.state"),
        "--progress-interval",
        "1000000",
        "--state-in",
        str(transaction_descriptor()["inputs"]["position2_kv_parent"]["path"]),
    ]


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


def validate_parent_acceptance() -> dict[str, Any]:
    require_fixed(PARENT_HANDOFF)
    require_fixed(PARENT_REVIEW_DECISION)
    handoff = load_json(PARENT_HANDOFF)
    decision = load_json(PARENT_REVIEW_DECISION)
    require(
        handoff.get("kind") == "handoff_ref"
        and handoff.get("handoff", {}).get("path") == str(PARENT_REVIEW_DECISION)
        and decision.get("kind") == "round_reviewed_handoff"
        and decision.get("mission_id") == PARENT_MISSION_ID
        and decision.get("producer_role") == "reviewer"
        and decision.get("review", {}).get("status") == "done",
        "d560dc6d3138 independent acceptance differs",
    )

    pointer_record = require_fixed(POINTER)
    manifest_record = require_fixed(GENERATION5 / "generation-manifest.json")
    ledger_record = require_fixed(GENERATION5 / "ledger.json")
    checkpoint4_record = require_fixed(
        GENERATION5 / "checkpoints/transaction-004.json"
    )
    position004_record = require_fixed(
        TRANSACTIONS / "transaction-004/position004.state"
    )
    pointer = load_json(POINTER)
    manifest = load_json(GENERATION5 / "generation-manifest.json")
    ledger = load_json(GENERATION5 / "ledger.json")
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    require(
        pointer.get("kind")
        == "ace3_position3_transaction4_generation5_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and pointer.get("generation") == START_CURSOR
        and pointer.get("generation_manifest") == manifest_record,
        "authoritative pointer is not generation5/cursor5",
    )
    require(
        manifest.get("kind")
        == "ace3_transaction4_publication_recovery_generation5_state"
        and manifest.get("status") == "PREPARED"
        and manifest.get("runtime_identity") == RUNTIME_IDENTITY
        and manifest.get("generation") == START_CURSOR
        and manifest.get("transaction004_retry_replay_resume") is False
        and manifest.get("transactions005_025_executed") is False,
        "generation5 manifest semantics differ",
    )
    for record in manifest.get("files", {}).values():
        authenticate(record, "generation5 manifest file")
    checkpoints = [
        file_record(GENERATION5 / f"checkpoints/transaction-{index:03d}.json")
        for index in range(5)
    ]
    require(
        ledger.get("kind") == "ace3_position3_r11_transaction_ledger"
        and ledger.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == START_CURSOR
        and ledger.get("completed_transaction_count") == START_CURSOR
        and ledger.get("next_transaction_index") == TRANSACTION_INDEX
        and ledger.get("transaction_count") == 26
        and ledger.get("authoritative_state_root") == str(GENERATION5)
        and ledger.get("completed_receipts") == checkpoints,
        "generation5 ledger is not cursor5",
    )
    require(
        checkpoint4.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint4.get("status") == "COMPLETE"
        and checkpoint4.get("transaction_index") == 4
        and checkpoint4.get("operation") == "position3-decoder-layer"
        and checkpoint4.get("result", {}).get("exact_integer_oracle_match") is True
        and checkpoint4.get("result", {}).get("natural_rtl_terminal") is True
        and checkpoint4.get("outputs", {}).get("state") == position004_record
        and checkpoint4.get("output_semantics", {}).get("state", {}).get("position")
        == OUTPUT_POSITION,
        "checkpoint004 is not the accepted transaction004 completion",
    )
    authenticate(
        {
            key: checkpoint4["outputs"]["hidden"][key]
            for key in ("path", "bytes", "sha256")
        },
        "checkpoint004 hidden output",
    )
    require(
        pointer_record["sha256"] == FIXED_SHA256[POINTER]
        and ledger_record["sha256"]
        == FIXED_SHA256[GENERATION5 / "ledger.json"]
        and checkpoint4_record["sha256"]
        == FIXED_SHA256[GENERATION5 / "checkpoints/transaction-004.json"],
        "fixed generation5 parent differs",
    )
    official_frozen_evidence()
    return ledger


def validate_absence() -> None:
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(5, 26)
        ),
        "transaction005-025 namespace is not absent",
    )
    require(
        not GENERATION6.exists()
        and not GENERATION6_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "generation6 or transaction005 runtime namespace exists",
    )


def baseline_document() -> dict[str, Any]:
    validate_parent_acceptance()
    validate_absence()
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_layer4_authoritative_baseline",
        "runtime_identity": RUNTIME_IDENTITY,
        "authoritative_generation": START_CURSOR,
        "authoritative_cursor": START_CURSOR,
        "parent_acceptance": {
            "index": file_record(PARENT_HANDOFF),
            "reviewer_decision": file_record(PARENT_REVIEW_DECISION),
        },
        "pointer": file_record(POINTER),
        "generation_trees": [
            tree_digest(generation)
            for generation in (GENERATION3, GENERATION4, GENERATION5)
        ],
        "transactions000_004": [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(5)
        ],
        "checkpoint004_parent": file_record(
            GENERATION5 / "checkpoints/transaction-004.json"
        ),
        "position004_input_state": copy.deepcopy(checkpoint4["outputs"]["state"]),
        "predecessor_hidden": copy.deepcopy(checkpoint4["outputs"]["hidden"]),
        "official_frozen_evidence": official_frozen_evidence(),
        "transaction005_absent": True,
        "transactions006_025_absent": True,
        "generation6_absent": True,
        "generation6_staging_absent": True,
        "transaction005_output_namespace": str(TRANSACTION5),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
    }


def validate_baseline(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction5_layer4_authoritative_baseline"
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("activity_counters") == ZERO_COUNTERS
        and document.get("transaction005_absent") is True
        and document.get("transactions006_025_absent") is True
        and document.get("generation6_absent") is True
        and document.get("generation6_staging_absent") is True
        and document.get("transaction005_output_namespace") == str(TRANSACTION5),
        "authoritative baseline identity or zero state differs",
    )
    authenticate(
        document["parent_acceptance"]["index"],
        "parent acceptance index",
        PARENT_HANDOFF,
    )
    authenticate(
        document["parent_acceptance"]["reviewer_decision"],
        "parent Reviewer decision",
        PARENT_REVIEW_DECISION,
    )
    authenticate(document["pointer"], "baseline pointer", POINTER)
    authenticate(
        document["checkpoint004_parent"],
        "checkpoint004 parent",
        GENERATION5 / "checkpoints/transaction-004.json",
    )
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    require(
        document.get("position004_input_state") == checkpoint4["outputs"]["state"]
        and document.get("predecessor_hidden") == checkpoint4["outputs"]["hidden"]
        and document.get("official_frozen_evidence") == official_frozen_evidence()
        and document.get("generation_trees")
        == [
            tree_digest(generation)
            for generation in (GENERATION3, GENERATION4, GENERATION5)
        ]
        and document.get("transactions000_004")
        == [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(5)
        ],
        "baseline parent, evidence, or preserved trees differ",
    )


def build_source_manifest(
    executor_path: Path,
    published_executor: Path,
) -> dict[str, Any]:
    accepted = load_json(ACCEPTED_SOURCE_MANIFEST)
    require(
        accepted.get("kind")
        == "ace3_position3_transaction4_layer3_exact_source_toolchain_closure",
        "accepted transaction004 source closure differs",
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
    sources["accepted_transaction004_source_closure"] = file_record(
        ACCEPTED_SOURCE_MANIFEST
    )
    sources["transaction005_executor"] = file_record(
        executor_path,
        published_executor,
    )
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_layer4_exact_source_toolchain_closure",
        "sources": sources,
        "toolchain": copy.deepcopy(accepted["toolchain"]),
    }


def validate_source_manifest(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction5_layer4_exact_source_toolchain_closure"
        and isinstance(document.get("sources"), dict)
        and isinstance(document.get("toolchain"), dict),
        "transaction005 source/toolchain closure differs",
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
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_transaction5_layer4_executor_package"
        and document.get("status") == "SEALED_REVIEW_REQUIRED"
        and document.get("mission_id") == MISSION_ID
        and document.get("parent_mission_id") == PARENT_MISSION_ID
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("required_parent_checkpoint_index") == 4
        and document.get("transaction_index") == TRANSACTION_INDEX
        and document.get("layer_index") == LAYER_INDEX
        and document.get("transaction_position") == POSITION
        and document.get("permitted_transaction_indices") == PERMITTED_INDICES
        and document.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and document.get("transaction_descriptor") == transaction_descriptor()
        and document.get("position004_input_state")
        == load_json(GENERATION5 / "checkpoints/transaction-004.json")["outputs"][
            "state"
        ]
        and document.get("official_frozen_evidence") == official_frozen_evidence()
        and document.get("compile") == {"cwd": str(ROOT), "argv": compile_argv()}
        and document.get("simulation")
        == {"cwd": str(ROOT), "argv": simulation_argv()}
        and document.get("launch") == launch_contract()
        and document.get("transaction005_output_namespace") == str(TRANSACTION5)
        and document.get("execution_authorized") is False
        and document.get("authority_created") is False
        and document.get("activity_counters") == ZERO_COUNTERS,
        "package identity, evidence, argv, scope, or zero state differs",
    )
    require(
        document.get("prohibitions")
        == [
            "transaction000-004 replay",
            "transaction006-025 execution",
            "model, oracle, or vector execution during package review",
            "RTL compile or simulation during package review",
            "generation6 publication during package review",
            "authority creation by this package stage",
            "execution before separate Manager authority and Reviewer PASS",
        ],
        "package prohibitions differ",
    )


def validate_package(
    package: Path = PACKAGE_ROOT,
    *,
    require_zero_runtime: bool = True,
) -> dict[str, Any]:
    require(package == PACKAGE_ROOT, "package path differs")
    manifest = load_json(package / "package-manifest.json")
    seal = load_json(package / "package-seal.json")
    sources = load_json(package / "source-manifest.json")
    baseline = load_json(package / "authoritative-baseline.json")
    validate_manifest_document(manifest)
    validate_source_manifest(sources)
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
        seal.get("kind") == "ace3_position3_transaction5_layer4_executor_seal"
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
    validate_parent_acceptance()
    if require_zero_runtime:
        validate_absence()
        require(not REVIEW.exists(), "canonical Reviewer output already exists")
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
        write_new(staging / "transaction5_executor.py", Path(__file__).read_bytes())
        write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
        write_new(
            staging / "authoritative-baseline.json",
            canonical_json(baseline),
        )
        source_manifest = build_source_manifest(
            staging / "transaction5_executor.py",
            PACKAGE_ROOT / "transaction5_executor.py",
        )
        write_new(
            staging / "source-manifest.json",
            canonical_json(source_manifest),
        )
        package_manifest = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction5_layer4_executor_package",
            "status": "SEALED_REVIEW_REQUIRED",
            "mission_id": MISSION_ID,
            "parent_mission_id": PARENT_MISSION_ID,
            "runtime_identity": RUNTIME_IDENTITY,
            "authoritative_generation": START_CURSOR,
            "authoritative_cursor": START_CURSOR,
            "required_parent_checkpoint_index": 4,
            "transaction_index": TRANSACTION_INDEX,
            "layer_index": LAYER_INDEX,
            "transaction_position": POSITION,
            "permitted_transaction_indices": PERMITTED_INDICES,
            "forbidden_transaction_indices": FORBIDDEN_INDICES,
            "transaction_descriptor": transaction_descriptor(),
            "position004_input_state": baseline["position004_input_state"],
            "official_frozen_evidence": baseline["official_frozen_evidence"],
            "authoritative_baseline": file_record(
                staging / "authoritative-baseline.json",
                PACKAGE_ROOT / "authoritative-baseline.json",
            ),
            "source_manifest": file_record(
                staging / "source-manifest.json",
                PACKAGE_ROOT / "source-manifest.json",
            ),
            "compile": {"cwd": str(ROOT), "argv": compile_argv()},
            "simulation": {"cwd": str(ROOT), "argv": simulation_argv()},
            "launch": launch_contract(),
            "transaction005_output_namespace": str(TRANSACTION5),
            "generation6_output_namespace": str(GENERATION6),
            "review_output": str(REVIEW),
            "future_manager_authority": str(AUTHORITY),
            "execution_authorized": False,
            "authority_created": False,
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
            "prohibitions": [
                "transaction000-004 replay",
                "transaction006-025 execution",
                "model, oracle, or vector execution during package review",
                "RTL compile or simulation during package review",
                "generation6 publication during package review",
                "authority creation by this package stage",
                "execution before separate Manager authority and Reviewer PASS",
            ],
        }
        write_new(
            staging / "package-manifest.json",
            canonical_json(package_manifest),
        )
        review_request = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction5_layer4_review_request",
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
                "execute model, oracle, or vector generation",
                "compile or simulate RTL",
                "execute transaction005",
                "publish generation6",
                "create transaction006-025 artifacts",
                "mutate generations3-5 or transactions000-004",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(review_request))
        members = {
            label: file_record(staging / relative, PACKAGE_ROOT / relative)
            for label, relative in PACKAGE_MEMBERS.items()
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction5_layer4_executor_seal",
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
        == "ace3_position3_transaction5_layer4_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == MISSION_ID
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and review.get("authoritative_baseline") == file_record(BASELINE)
        and review.get("permitted_launch") == launch_contract()
        and review.get("activity_counters") == ZERO_COUNTERS
        and review.get("transaction005_executed") is False
        and review.get("generation6_published") is False
        and review.get("transactions006_025_absent") is True
        and review.get("this_review_authorizes_execution") is False,
        "independent package review differs",
    )
    return review


def validate_authority() -> dict[str, Any]:
    authority = load_json(AUTHORITY)
    review = validate_review()
    require(
        authority.get("kind")
        == "ace3_position3_transaction5_layer4_manager_authorization"
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
                GENERATION5 / "generation-manifest.json"
            ),
            "ledger": file_record(GENERATION5 / "ledger.json"),
            "checkpoint004": file_record(
                GENERATION5 / "checkpoints/transaction-004.json"
            ),
            "position004_input_state": file_record(
                TRANSACTIONS / "transaction-004/position004.state"
            ),
        }
        and authority.get("activity_counters") == ZERO_COUNTERS
        and authority.get("authority_consumed") is False
        and authority.get("replay_authorized") is False,
        "Manager transaction005 authority differs",
    )
    host_decision = authority.get("host_reviewer_decision")
    require(isinstance(host_decision, dict), "Host Reviewer decision binding absent")
    authenticate(host_decision, "Host Reviewer decision")
    require(review.get("status") == "PASS", "package acceptance is not PASS")
    return authority


def write_exclusive_json(path: Path, document: object) -> None:
    write_new(path, canonical_json(document))
    fsync_directory(path.parent)


def publish_generation6(
    ledger5: Mapping[str, Any],
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    checkpoint_payloads = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION5 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(5)
    }
    checkpoint_payloads["checkpoints/transaction-005.json"] = canonical_json(
        receipt
    )
    checkpoint_records = [
        {
            "path": str(GENERATION6 / f"checkpoints/transaction-{index:03d}.json"),
            "bytes": len(checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]),
            "sha256": sha256_bytes(
                checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]
            ),
        }
        for index in range(6)
    ]
    ledger6 = copy.deepcopy(dict(ledger5))
    ledger6.update(
        {
            "authoritative_state_root": str(GENERATION6),
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
    files = {**checkpoint_payloads, "ledger.json": canonical_json(ledger6)}
    manifest = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_generation6_state",
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
                "path": str(GENERATION6 / relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for relative, payload in files.items()
        },
        "activity_counters": {
            **ZERO_COUNTERS,
            "authority_consumption": 1,
            "oracle_execution": 1,
            "payload_execution": 1,
            "rtl_compile": 1,
            "rtl_simulation": 1,
            "runtime_mutation": 1,
            "transaction_execution": 1,
            "vector_generation": 1,
        },
        "manager_decision": authority.get("decision"),
    }
    manifest_payload = canonical_json(manifest)
    pointer6 = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_generation6_authoritative_pointer",
        "status": "COMMITTED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": EXIT_CURSOR,
        "generation_manifest": {
            "path": str(GENERATION6 / "generation-manifest.json"),
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "previous_authoritative_pointer_sha256": FIXED_SHA256[POINTER],
        "atomic_visibility_contract": (
            "pointer replacement is the sole visibility point from "
            "generation5/cursor5 to generation6/cursor6"
        ),
    }
    GENERATION6_STAGING.mkdir(mode=0o700)
    (GENERATION6_STAGING / "checkpoints").mkdir(mode=0o700)
    for relative, payload in files.items():
        write_new(GENERATION6_STAGING / relative, payload)
    write_new(
        GENERATION6_STAGING / "generation-manifest.json",
        manifest_payload,
    )
    fsync_directory(GENERATION6_STAGING / "checkpoints")
    fsync_directory(GENERATION6_STAGING)
    os.rename(GENERATION6_STAGING, GENERATION6)
    fsync_directory(GENERATIONS)
    temporary = POINTER.with_name(".authoritative-state.transaction005.tmp")
    write_new(temporary, canonical_json(pointer6), 0o600)
    os.replace(temporary, POINTER)
    fsync_directory(POINTER.parent)


def execute_once() -> None:
    validate_package(require_zero_runtime=False)
    ledger5 = validate_parent_acceptance()
    validate_absence()
    authority = validate_authority()
    with RUNTIME_LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Transaction5Error("another transaction invocation is active") from error
        validate_parent_acceptance()
        validate_absence()
        FUTURE_ROOT.mkdir(mode=0o700)
        fsync_directory(FUTURE_ROOT.parent)
        try:
            write_exclusive_json(
                CONSUMPTION,
                {
                    "schema_version": 1,
                    "kind": "ace3_position3_transaction5_layer4_authority_consumption",
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
                    "kind": "ace3_position3_transaction5_layer4_execution_start",
                    "transaction_index": TRANSACTION_INDEX,
                    "layer_index": LAYER_INDEX,
                    "started_ns": started_ns,
                },
            )
            original = import_module(ORIGINAL_EXECUTOR, "ace3_tx5_original")
            exact = import_module(EXACT_EXECUTOR, "ace3_tx5_exact")
            states = original.seed_reference_states()
            require(len(states) == 24, "independent oracle state count differs")
            receipt = exact.execute_exact_layer_transaction(
                transaction_descriptor(),
                load_json(GENERATION5 / "checkpoints/transaction-004.json"),
                states[LAYER_INDEX],
                ledger5["cumulative_execution_seconds"],
            )
            original.candidate_control.validate_completion_receipt(
                transaction_descriptor(),
                receipt,
            )
            require(
                receipt.get("transaction_index") == TRANSACTION_INDEX
                and receipt.get("result", {}).get("exact_integer_oracle_match") is True
                and receipt.get("result", {}).get("natural_rtl_terminal") is True,
                "transaction005 result differs",
            )
            write_exclusive_json(RECEIPT_CANDIDATE, receipt)
            evidence = {
                "schema_version": 1,
                "kind": "ace3_position3_transaction5_layer4_execution_evidence",
                "transaction_index": TRANSACTION_INDEX,
                "layer_index": LAYER_INDEX,
                "started_ns": started_ns,
                "transaction_tree": tree_digest(TRANSACTION5),
                "cached_evidence": False,
                "model_execution": 0,
                "oracle_execution": 1,
                "rtl_compile": 1,
                "rtl_simulation": 1,
                "transaction_execution": 1,
                "vector_generation": 1,
            }
            write_exclusive_json(EXECUTION_EVIDENCE, evidence)
            publish_generation6(ledger5, receipt, authority)
            write_exclusive_json(
                TERMINAL,
                {
                    "schema_version": 1,
                    "kind": "ace3_position3_transaction5_layer4_terminal",
                    "status": "PASS",
                    "runtime_identity": RUNTIME_IDENTITY,
                    "executed_transaction_indices": [TRANSACTION_INDEX],
                    "executed_layer_indices": [LAYER_INDEX],
                    "next_transaction_index": EXIT_CURSOR,
                    "generation": EXIT_CURSOR,
                    "authority_consumption": file_record(CONSUMPTION),
                    "execution_evidence": file_record(EXECUTION_EVIDENCE),
                    "transactions000_004_replayed": False,
                    "transactions006_025_executed": False,
                },
            )
        except BaseException as error:
            if FUTURE_ROOT.exists() and not FAILURE_TERMINAL.exists():
                write_exclusive_json(
                    FAILURE_TERMINAL,
                    {
                        "schema_version": 1,
                        "kind": "ace3_position3_transaction5_layer4_fail_closed_terminal",
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
        "only transaction005 is permitted",
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
            "POSITION3_TRANSACTION005_LAYER04_PACKAGE_PREPARED "
            f"package={package} generation=5 cursor=5 checkpoint=004 "
            "authority=0 consumption=0 model=0 oracle=0 vectors=0 "
            "rtl_compile=0 rtl_simulation=0 transaction=0 generation6=0"
        )
    elif arguments.operation == "validate-package":
        validate_package()
        print(
            "POSITION3_TRANSACTION005_LAYER04_PACKAGE_VALID "
            "generation=5 cursor=5 checkpoint=004 authority=0 consumption=0 "
            "model=0 oracle=0 vectors=0 rtl_compile=0 rtl_simulation=0 "
            "transaction=0 generation6=0"
        )
    else:
        validate_requested_paths(arguments)
        execute_once()
        print(
            "POSITION3_TRANSACTION005_LAYER04_EXECUTED "
            "transaction=5 layer=4 generation=6 cursor=6"
        )


if __name__ == "__main__":
    try:
        main()
    except (
        Transaction5Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(f"POSITION3_TRANSACTION005_LAYER04_REFUSED {error}") from error
