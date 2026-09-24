#!/usr/bin/env python3
"""Seal and, only after review, apply transaction005 publication recovery."""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
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
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATIONS = ADOPTION / "state-generations"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION6 = GENERATIONS / "generation-0000000006"
GENERATION6_STAGING = GENERATIONS / ".generation-0000000006.prepared"
POINTER = ADOPTION / "authoritative-state.json"
RUNTIME_LOCK = ADOPTION / "execution.lock"
TRANSACTIONS = RUNTIME / "transactions"
FUTURE = RUNTIME / "transaction5-authoritative-generation6"
ORIGINAL_CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
ORIGINAL_FAIL_CLOSED_TERMINAL = FUTURE / "fail-closed-terminal.json"
PUBLICATION_CONSUMPTION = FUTURE / "publication-recovery-authority-consumption.json"
PUBLICATION_TERMINAL = FUTURE / "publication-recovery-terminal.json"
PUBLICATION_FAILURE = FUTURE / "publication-recovery-fail-closed-terminal.json"
SIMULATION_LOG = TRANSACTIONS / "transaction-005/position003/simulation.log"
MANAGER_MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/b66531b35910/mission.json"
)
REVIEWED_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/b66531b35910/round-0004.json"
)
RECOVERY_PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery"
    / "r1-postconsume-none-to-false-sourcebound"
)
RECOVERY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery"
    / "reviews/r1-postconsume-none-to-false-sourcebound/independent-review.json"
)
AUTHORITY_COLLECTION = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery_authorities"
)
SUPERSEDED_R1_OUTPUT = (
    AUTHORITY_COLLECTION / "r1-manager-direct-recovery-bound"
)
DEFAULT_OUTPUT = (
    AUTHORITY_COLLECTION / "r2-manager-direct-recovery-bound"
)
DEFAULT_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery_authority_reviews"
    / "r2-manager-direct-recovery-bound/independent-review.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT
    / "ace3/model/emit_transaction5_publication_recovery_authority_review.py"
)
AUTHORITY_FILENAME = "publication-recovery-authority.json"
PUBLICATION_SOURCE_NAME = "publication-recovery-authority.py"
AUTHORITY_MEMBERS = {
    AUTHORITY_FILENAME,
    PUBLICATION_SOURCE_NAME,
    "review-emitter.py",
    "review-request.json",
}
RECOVERY_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "recovery-adjudication.py",
    "review-emitter.py",
    "review-request.json",
}
FIXED_SHA256 = {
    RECOVERY_PACKAGE
    / "adjudication.json": "199e5ba62b96ef6bb2bd61701483f32c47e49399f4056ce997f084170a3082ad",
    RECOVERY_PACKAGE
    / "package-manifest.json": "eff1b1fbafc084284ca6c3920c5370a252d9dad9b687e6e266adc7589f6cd6ba",
    RECOVERY_PACKAGE
    / "package-seal.json": "21e1d0b0b7864fa57a8a57ac4939c386260661d4c95a63661a046e36b9c65158",
    RECOVERY_PACKAGE
    / "recovery-adjudication.py": "433d72c614260fc24884a9501623f1350690a2c2240c550b8b92853be250421d",
    RECOVERY_REVIEW: "5f3971da761a406718544ac55ad4faf095b50395fbea1dd49520c170ed8f8da4",
    POINTER: "adee838870e758a5cc32a1a79248938b22d8e9d701501dce525c9098958bdda0",
    GENERATION5
    / "generation-manifest.json": "03f73918812f160cb4b9b7f94b239c5d3efb9cab9d4d41bd621a59eadfc9b599",
    GENERATION5
    / "ledger.json": "ca9397fc658ffe0629c8ba4cade0af5640d85298106b4dfc0dfbeb2ca039af14",
    GENERATION5
    / "checkpoints/transaction-004.json": "ff157b01b83d07d8bf167c420115659072a54415054454674625503d7decd610",
    ORIGINAL_CONSUMPTION: "116a87010387331ed546a54e9af49c3882951cef380eaa6c3f607eef86e9fe92",
    ORIGINAL_FAIL_CLOSED_TERMINAL: "12d0aa1e19a083b7027b1ec7568f7870bddf64389019eb5ad5703a1af0b029de",
    SIMULATION_LOG: "1b5af7c5641c1c9dd8d30eb74b4a9c10e3eb76397caa2bfc4ca8c5245f6cfd06",
    MANAGER_MISSION: "fb47f4b429eb54324311fa54b47ac84e7a69984706917ad6abb418fe37da2555",
    REVIEWED_HANDOFF: "4b2f6730783891a83816cccaee83c344acbc23711f3f546fab0ed45580a5e703",
}
RECOVERY_ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation6_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions006_025_execution": 0,
}
ZERO_ACTIVITY = {
    **RECOVERY_ZERO_ACTIVITY,
    "model_generation": 0,
    "oracle_generation": 0,
    "vector_generation": 0,
}
PERMITTED_OPERATIONS = [
    "reconstruct the transaction005 receipt from accepted r1 frozen evidence",
    "atomically publish generation6, checkpoint005, and cursor6",
]
PROHIBITIONS = [
    "transaction005 retry",
    "transaction005 replay",
    "transaction005 resume",
    "new transaction identity",
    "transaction006-025 execution",
    "model execution",
    "model generation",
    "oracle execution",
    "oracle generation",
    "vector generation",
    "RTL compilation",
    "RTL simulation",
    "publication before the bound independent review is PASS",
    "any publish argv other than the exact bound argv",
    "the superseded r1 authority candidate publish argv",
]
FORBIDDEN_IMPORTS = {"importlib", "numpy", "safetensors", "subprocess", "torch"}
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
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    return {
        "path": str(published_path or path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate(record: Mapping[str, Any], label: str) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"},
        f"{label} record malformed",
    )
    require(file_record(Path(record["path"])) == dict(record), f"{label} differs")


def manager_provenance() -> dict[str, Any]:
    mission = load_json(MANAGER_MISSION)
    reviewed_handoff = load_json(REVIEWED_HANDOFF)
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == "b66531b35910"
        and mission.get("node_key") == "tx005-layer04-execute-once"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and "manager-direct" in mission.get("tags", []),
        "manager-direct mission provenance differs",
    )
    review = reviewed_handoff.get("review", {})
    require(
        reviewed_handoff.get("kind") == "round_reviewed_handoff"
        and reviewed_handoff.get("mission_id") == "b66531b35910"
        and reviewed_handoff.get("producer_role") == "reviewer"
        and review.get("status") == "continue"
        and review.get("next_action")
        == (
            "Prepare a separately reviewed publication-only successor authority "
            "bound to the accepted transaction005 adjudication; seal an argv that "
            "only atomically publishes generation6/checkpoint005/cursor6 without "
            "RTL or oracle replay, and do not execute it before independent review."
        ),
        "reviewed Manager issuance direction differs",
    )
    return {
        "mission": file_record(MANAGER_MISSION),
        "reviewed_handoff": file_record(REVIEWED_HANDOFF),
        "mission_id": "b66531b35910",
        "manager_direct": True,
    }


def original_authority_consumption() -> dict[str, Any]:
    consumption = load_json(ORIGINAL_CONSUMPTION)
    manager_authority = consumption.get("manager_authorization", {})
    require(
        consumption.get("kind")
        == "ace3_position3_transaction5_layer4_authority_consumption"
        and consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("runtime_identity") == RUNTIME_IDENTITY
        and consumption.get("transaction_index") == 5
        and consumption.get("layer_index") == 4
        and consumption.get("replay_authorized") is False,
        "original transaction005 authority consumption differs",
    )
    authenticate(manager_authority, "consumed original Manager authority")
    original_manager = load_json(Path(manager_authority["path"]))
    require(
        original_manager.get("kind")
        == "ace3_position3_transaction5_layer4_manager_authorization"
        and original_manager.get("producer_role") == "manager",
        "consumed original authority lacks Manager provenance",
    )
    return {
        "consumption": file_record(ORIGINAL_CONSUMPTION),
        "manager_authority": copy.deepcopy(manager_authority),
        "status": "CONSUMED_ONCE_NON_REPLAYABLE",
    }


def validate_source_boundary(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(
                    alias.name.split(".", 1)[0] not in FORBIDDEN_IMPORTS
                    for alias in node.names
                ),
                "publication source imports model, oracle, RTL, or process support",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                (node.module or "").split(".", 1)[0] not in FORBIDDEN_IMPORTS,
                "publication source imports model, oracle, RTL, or process support",
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            require(name.lower() not in FORBIDDEN_CALLS, f"forbidden call: {name}")


def later_transactions() -> list[int]:
    return [
        index
        for index in range(6, 26)
        if (TRANSACTIONS / f"transaction-{index:03d}").exists()
    ]


def validate_zero_state(
    *,
    generation6_exists: bool,
    generation6_staging_exists: bool,
    publication_consumption_exists: bool,
    publication_terminal_exists: bool,
    publication_failure_exists: bool,
    later_transaction_indices: list[int],
) -> None:
    require(not generation6_exists, "generation6 already exists")
    require(not generation6_staging_exists, "generation6 staging already exists")
    require(not publication_consumption_exists, "publication authority was consumed")
    require(not publication_terminal_exists, "publication terminal already exists")
    require(not publication_failure_exists, "publication failure already exists")
    require(
        not later_transaction_indices,
        "transaction006-025 artifact exists",
    )


def validate_live_zero_state() -> None:
    validate_zero_state(
        generation6_exists=GENERATION6.exists(),
        generation6_staging_exists=GENERATION6_STAGING.exists(),
        publication_consumption_exists=PUBLICATION_CONSUMPTION.exists(),
        publication_terminal_exists=PUBLICATION_TERMINAL.exists(),
        publication_failure_exists=PUBLICATION_FAILURE.exists(),
        later_transaction_indices=later_transactions(),
    )
    require(
        FUTURE.is_dir()
        and {path.name for path in FUTURE.iterdir()}
        == {
            "execution-start.json",
            "fail-closed-terminal.json",
            "manager-authorization-consumption.json",
        },
        "frozen transaction005 failure namespace differs",
    )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION5 / "ledger.json")
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 5
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == 5
        and ledger.get("next_transaction_index") == 5
        and ledger.get("completed_transaction_count") == 5
        and len(ledger.get("completed_receipts", [])) == 5,
        "generation5/cursor5 parent differs",
    )


def frozen_evidence(adjudication: Mapping[str, Any]) -> dict[str, Any]:
    computation = adjudication["computation_evidence"]
    return {
        "fail_closed_terminal": copy.deepcopy(
            adjudication["execution_boundary"]["fail_closed_terminal"]
        ),
        "simulation_log": copy.deepcopy(computation["simulation_log"]),
        "raw_terminal": copy.deepcopy(computation["terminal"]),
        "raw_trace": copy.deepcopy(computation["raw_trace"]),
        "raw_final": copy.deepcopy(computation["raw_final"]),
        "oracle_trace": copy.deepcopy(computation["oracle_trace"]),
        "oracle_final": copy.deepcopy(computation["oracle_final"]),
        "comparison": copy.deepcopy(computation["comparison"]),
        "output_state": copy.deepcopy(computation["output_state"]),
        "hidden_semantic_sha256": computation["hidden_semantic_sha256"],
        "transaction_tree": copy.deepcopy(
            adjudication["frozen_transaction005"]["tree"]
        ),
    }


def validate_accepted_recovery() -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        RECOVERY_PACKAGE.is_dir()
        and not RECOVERY_PACKAGE.is_symlink()
        and stat.S_IMODE(RECOVERY_PACKAGE.stat().st_mode) & 0o222 == 0,
        "accepted r1 recovery package is absent or writable",
    )
    for path, expected in FIXED_SHA256.items():
        require(sha256_file(path) == expected, f"fixed input hash differs: {path}")
    require(
        stat.S_IMODE(RECOVERY_REVIEW.stat().st_mode) & 0o222 == 0,
        "accepted r1 recovery review is writable",
    )
    seal = load_json(RECOVERY_PACKAGE / "package-seal.json")
    require(
        seal.get("kind")
        == "ace3_transaction5_publication_recovery_adjudication_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and seal.get("generation6_publication_authorized") is False
        and seal.get("generation6_publication_performed") is False
        and set(seal.get("members", {})) == RECOVERY_MEMBERS,
        "accepted r1 recovery seal differs",
    )
    for name, record in seal["members"].items():
        require(
            record.get("path") == str(RECOVERY_PACKAGE / name),
            f"accepted r1 member path differs: {name}",
        )
        authenticate(record, f"accepted r1 member {name}")
    review = load_json(RECOVERY_REVIEW)
    require(
        review.get("kind")
        == "ace3_transaction5_publication_recovery_independent_review"
        and review.get("status") == "PASS"
        and review.get("preparation_participation") is False
        and review.get("institutional_role_authority_claimed") is False
        and review.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and review.get("transaction005_retry_replay_resume") is False
        and review.get("transactions006_025_absent") is True
        and review.get("new_authority_consumed") is False
        and review.get("generation6_publication_authorized") is False
        and review.get("generation6_publication_performed") is False
        and review.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and review.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json"),
        "accepted r1 recovery review differs",
    )
    adjudication = load_json(RECOVERY_PACKAGE / "adjudication.json")
    boundary = adjudication.get("recovery_boundary", {})
    require(
        adjudication.get("kind")
        == "ace3_transaction5_postconsume_publication_recovery_adjudication"
        and adjudication.get("status") == "PASS"
        and adjudication.get("transaction_index") == 5
        and adjudication.get("layer_index") == 4
        and boundary.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation6_publication_authorized") is False
        and boundary.get("generation6_publication_performed") is False
        and boundary.get("transactions006_025_absent") is True,
        "accepted r1 adjudication differs",
    )
    parent = adjudication["authoritative_parent"]
    require(
        parent.get("generation") == 5
        and parent.get("cursor") == 5
        and parent.get("pointer") == file_record(POINTER)
        and parent.get("ledger") == file_record(GENERATION5 / "ledger.json")
        and parent.get("checkpoint004")
        == file_record(GENERATION5 / "checkpoints/transaction-004.json"),
        "accepted r1 generation5/cursor5 parent differs",
    )
    for label, record in frozen_evidence(adjudication).items():
        if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
            authenticate(record, f"frozen transaction005 {label}")
    return adjudication, review


def authorized_publish_argv(
    authority_package: Path = DEFAULT_OUTPUT,
    authority_review: Path = DEFAULT_REVIEW,
) -> list[str]:
    return [
        "/usr/bin/python3",
        str(authority_package / PUBLICATION_SOURCE_NAME),
        "publish",
        "--authority-package",
        str(authority_package),
        "--authority-review",
        str(authority_review),
        "--recovery-package",
        str(RECOVERY_PACKAGE),
        "--recovery-review",
        str(RECOVERY_REVIEW),
        "--runtime-root",
        str(RUNTIME),
    ]


def superseded_candidate() -> dict[str, Any]:
    reviewer = SUPERSEDED_R1_OUTPUT / "review-emitter.py"
    source = reviewer.read_text(encoding="utf-8")
    require(
        'item["authoritative_parent"].update({"cursor": 5})' in source,
        "superseded r1 review defect differs",
    )
    return {
        "package_root": str(SUPERSEDED_R1_OUTPUT),
        "authority_seal": file_record(
            SUPERSEDED_R1_OUTPUT / "authority-seal.json"
        ),
        "authority": file_record(
            SUPERSEDED_R1_OUTPUT / AUTHORITY_FILENAME
        ),
        "publication_source": file_record(
            SUPERSEDED_R1_OUTPUT / PUBLICATION_SOURCE_NAME
        ),
        "review_emitter": file_record(reviewer),
        "review_request": file_record(
            SUPERSEDED_R1_OUTPUT / "review-request.json"
        ),
        "status": "SUPERSEDED_UNUSABLE_UNCONSUMED",
        "defect": (
            "review emitter wrong-parent adversarial mutation used the accepted "
            "cursor5 value and therefore rejected its own review assessment"
        ),
        "publish_argv_forbidden": True,
    }


def authority_document(
    authority_package: Path,
    authority_review: Path,
    publication_source: Path,
    published_source: Path,
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_transaction5_publication_only_recovery_manager_authority",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "scope": (
            "receipt reconstruction and atomic generation6/checkpoint005/"
            "cursor6 publication only"
        ),
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 5,
        "layer_index": 4,
        "new_transaction_identity": None,
        "manager_provenance": manager_provenance(),
        "superseded_candidate": superseded_candidate(),
        "original_transaction005_authority_consumption": (
            original_authority_consumption()
        ),
        "accepted_r1_recovery": {
            "package_root": str(RECOVERY_PACKAGE),
            "adjudication": file_record(RECOVERY_PACKAGE / "adjudication.json"),
            "package_manifest": file_record(
                RECOVERY_PACKAGE / "package-manifest.json"
            ),
            "package_seal": file_record(RECOVERY_PACKAGE / "package-seal.json"),
            "recovery_source": file_record(
                RECOVERY_PACKAGE / "recovery-adjudication.py"
            ),
            "independent_review": file_record(RECOVERY_REVIEW),
        },
        "frozen_transaction005_evidence": frozen_evidence(adjudication),
        "authoritative_parent": {
            "generation": 5,
            "cursor": 5,
            "latest_checkpoint_index": 4,
            "pointer": file_record(POINTER),
            "generation_manifest": file_record(
                GENERATION5 / "generation-manifest.json"
            ),
            "ledger": file_record(GENERATION5 / "ledger.json"),
            "checkpoint004": file_record(
                GENERATION5 / "checkpoints/transaction-004.json"
            ),
        },
        "publication_source": file_record(publication_source, published_source),
        "authorized_publish": {
            "argv": authorized_publish_argv(authority_package, authority_review),
            "cwd": str(ROOT),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
            },
        },
        "permitted_operations": list(PERMITTED_OPERATIONS),
        "prohibitions": list(PROHIBITIONS),
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "authority_cardinality": 1,
        "non_superseded_authority_cardinality": 1,
        "historical_superseded_authority_count": 1,
        "authority_consumed": False,
        "publication_performed": False,
        "generation6_exists": False,
        "transactions006_025_absent": True,
        "reviewer_acceptance": {
            "required": True,
            "path": str(authority_review),
            "present_at_issuance": False,
        },
    }


def validate_authority_package(
    authority_package: Path = DEFAULT_OUTPUT,
    authority_review: Path = DEFAULT_REVIEW,
) -> dict[str, Any]:
    require(
        authority_package.is_dir()
        and not authority_package.is_symlink()
        and stat.S_IMODE(authority_package.stat().st_mode) & 0o222 == 0,
        "authority package is absent or writable",
    )
    actual = {path.name for path in authority_package.iterdir() if path.is_file()}
    require(
        actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
        "authority package file set differs",
    )
    seal = load_json(authority_package / "authority-seal.json")
    require(
        seal.get("kind")
        == "ace3_transaction5_publication_recovery_authority_seal"
        and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and seal.get("activity_counters") == ZERO_ACTIVITY
        and seal.get("authority_cardinality") == 1
        and seal.get("non_superseded_authority_cardinality") == 1
        and seal.get("historical_superseded_authority_count") == 1
        and seal.get("authority_consumed") is False
        and seal.get("publication_performed") is False
        and seal.get("generation6_exists") is False
        and set(seal.get("members", {})) == AUTHORITY_MEMBERS,
        "authority seal differs",
    )
    for name, record in seal["members"].items():
        path = authority_package / name
        require(record.get("path") == str(path), f"authority path differs: {name}")
        authenticate(record, f"authority member {name}")
        require(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
            f"authority member is writable: {name}",
        )
    validate_source_boundary(authority_package / PUBLICATION_SOURCE_NAME)
    validate_source_boundary(authority_package / "review-emitter.py")
    adjudication, _ = validate_accepted_recovery()
    expected = authority_document(
        authority_package,
        authority_review,
        authority_package / PUBLICATION_SOURCE_NAME,
        authority_package / PUBLICATION_SOURCE_NAME,
        adjudication,
    )
    authority = load_json(authority_package / AUTHORITY_FILENAME)
    require(authority == expected, "publication-recovery authority differs")
    request = load_json(authority_package / "review-request.json")
    require(
        request.get("required_role") == "reviewer"
        and request.get("preparation_participation_required") is False
        and request.get("manager_provenance_required") is True
        and request.get("review_output") == str(authority_review)
        and request.get("authority")
        == file_record(authority_package / AUTHORITY_FILENAME)
        and request.get("publication_source")
        == file_record(authority_package / PUBLICATION_SOURCE_NAME)
        and request.get("review_invocation")
        == [
            "/usr/bin/python3",
            str(authority_package / "review-emitter.py"),
            "--authority-package",
            str(authority_package),
            "--output",
            str(authority_review),
        ],
        "authority review request differs",
    )
    validate_live_zero_state()
    return authority


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def issue_authority(
    authority_package: Path = DEFAULT_OUTPUT,
    authority_review: Path = DEFAULT_REVIEW,
) -> Path:
    require(authority_package.is_absolute(), "authority package must be absolute")
    require(authority_review.is_absolute(), "authority review must be absolute")
    require(
        not authority_package.exists(),
        f"authority package already exists: {authority_package}",
    )
    require(
        not authority_review.exists(),
        f"authority review already exists: {authority_review}",
    )
    validate_source_boundary(Path(__file__).resolve())
    validate_source_boundary(REVIEW_EMITTER_SOURCE)
    adjudication, _ = validate_accepted_recovery()
    validate_live_zero_state()
    authority_package.parent.mkdir(parents=True, exist_ok=True)
    authority_review.parent.mkdir(parents=True, exist_ok=True)
    competing = list(AUTHORITY_COLLECTION.rglob(AUTHORITY_FILENAME))
    require(
        competing == [SUPERSEDED_R1_OUTPUT / AUTHORITY_FILENAME],
        "unexpected transaction005 recovery authority set",
    )
    staging = authority_package.with_name(f".{authority_package.name}.preparing")
    require(not staging.exists(), f"authority staging exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        write_new(staging / PUBLICATION_SOURCE_NAME, Path(__file__).read_bytes())
        write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
        authority = authority_document(
            authority_package,
            authority_review,
            staging / PUBLICATION_SOURCE_NAME,
            authority_package / PUBLICATION_SOURCE_NAME,
            adjudication,
        )
        write_new(staging / AUTHORITY_FILENAME, canonical_json(authority))
        request = {
            "schema_version": 1,
            "kind": "ace3_transaction5_publication_recovery_authority_review_request",
            "required_role": "reviewer",
            "preparation_participation_required": False,
            "manager_provenance_required": True,
            "review_output": str(authority_review),
            "authority": file_record(
                staging / AUTHORITY_FILENAME,
                authority_package / AUTHORITY_FILENAME,
            ),
            "publication_source": file_record(
                staging / PUBLICATION_SOURCE_NAME,
                authority_package / PUBLICATION_SOURCE_NAME,
            ),
            "reviewer_emitter": file_record(
                staging / "review-emitter.py",
                authority_package / "review-emitter.py",
            ),
            "review_invocation": [
                "/usr/bin/python3",
                str(authority_package / "review-emitter.py"),
                "--authority-package",
                str(authority_package),
                "--output",
                str(authority_review),
            ],
            "requested_judgment": "PASS_OR_REJECT",
            "review_must_not": [
                "consume publication authority",
                "publish or stage generation6",
                "retry, replay, or resume transaction005",
                "generate or execute model, oracle, or vectors",
                "compile or simulate RTL",
                "create transaction006-025 artifacts",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(request))
        members = {
            name: file_record(staging / name, authority_package / name)
            for name in AUTHORITY_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_transaction5_publication_recovery_authority_seal",
            "status": "AUTHORIZED_NOT_CONSUMED",
            "members": members,
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "authority_cardinality": 1,
            "non_superseded_authority_cardinality": 1,
            "historical_superseded_authority_count": 1,
            "authority_consumed": False,
            "publication_performed": False,
            "generation6_exists": False,
            "transactions006_025_absent": True,
        }
        write_new(staging / "authority-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, authority_package)
        fsync_directory(authority_package.parent)
        authority_package.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_authority_package(authority_package, authority_review)
    return authority_package


def validate_authority_review(
    authority_package: Path = DEFAULT_OUTPUT,
    authority_review: Path = DEFAULT_REVIEW,
) -> dict[str, Any]:
    authority = validate_authority_package(authority_package, authority_review)
    review = load_json(authority_review)
    require(
        stat.S_IMODE(authority_review.stat().st_mode) & 0o222 == 0,
        "authority review is writable",
    )
    require(
        review.get("kind")
        == "ace3_transaction5_publication_recovery_authority_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("preparation_participation") is False
        and review.get("manager_provenance_verified") is True
        and review.get("original_consumption_binding_verified") is True
        and review.get("vector_generation_prohibited") is True
        and review.get("authority_seal")
        == file_record(authority_package / "authority-seal.json")
        and review.get("authority")
        == file_record(authority_package / AUTHORITY_FILENAME)
        and review.get("publication_source")
        == file_record(authority_package / PUBLICATION_SOURCE_NAME)
        and review.get("accepted_r1_recovery_review")
        == file_record(RECOVERY_REVIEW)
        and review.get("authorized_publish_argv")
        == authority["authorized_publish"]["argv"]
        and review.get("activity_counters") == ZERO_ACTIVITY
        and review.get("transaction005_retry_replay_resume") is False
        and review.get("model_oracle_rtl_execution") is False
        and review.get("authority_consumed") is False
        and review.get("generation6_publication_performed") is False
        and review.get("transactions006_025_absent") is True,
        "independent authority review differs",
    )
    return review


def generation6_payloads(
    authority_package: Path,
    authority_review: Path,
) -> tuple[dict[str, bytes], bytes, bytes]:
    authority = validate_authority_package(authority_package, authority_review)
    validate_authority_review(authority_package, authority_review)
    adjudication, _ = validate_accepted_recovery()
    receipt = adjudication["reconstructed_receipt"]
    ledger5 = load_json(GENERATION5 / "ledger.json")
    checkpoints = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION5 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(5)
    }
    checkpoints["checkpoints/transaction-005.json"] = canonical_json(receipt)
    checkpoint_records = [
        {
            "path": str(GENERATION6 / f"checkpoints/transaction-{index:03d}.json"),
            "bytes": len(checkpoints[f"checkpoints/transaction-{index:03d}.json"]),
            "sha256": sha256_bytes(
                checkpoints[f"checkpoints/transaction-{index:03d}.json"]
            ),
        }
        for index in range(6)
    ]
    ledger6 = copy.deepcopy(ledger5)
    ledger6.update(
        {
            "authoritative_state_root": str(GENERATION6),
            "completed_receipts": checkpoint_records,
            "completed_transaction_count": 6,
            "next_transaction_index": 6,
            "state_generation": 6,
            "status": "IN_PROGRESS",
            "cumulative_execution_seconds": receipt["timing"][
                "cumulative_execution_seconds"
            ],
        }
    )
    files = {**checkpoints, "ledger.json": canonical_json(ledger6)}
    activity = copy.deepcopy(ZERO_ACTIVITY)
    activity["authority_consumption"] = 1
    activity["generation6_publication"] = 1
    manifest = {
        "schema_version": 1,
        "kind": "ace3_transaction5_publication_recovery_generation6_state",
        "status": "PREPARED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": 6,
        "parent_generation": 5,
        "parent_pointer": file_record(POINTER),
        "accepted_r1_recovery_seal": file_record(
            RECOVERY_PACKAGE / "package-seal.json"
        ),
        "accepted_r1_recovery_review": file_record(RECOVERY_REVIEW),
        "publication_authority": file_record(
            authority_package / AUTHORITY_FILENAME
        ),
        "publication_authority_review": file_record(authority_review),
        "original_transaction005_authority_consumption": file_record(
            ORIGINAL_CONSUMPTION
        ),
        "files": {
            relative: {
                "path": str(GENERATION6 / relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for relative, payload in files.items()
        },
        "activity_counters": activity,
        "transaction005_retry_replay_resume": False,
        "transactions006_025_executed": False,
        "authorized_publish_argv": authority["authorized_publish"]["argv"],
    }
    manifest_payload = canonical_json(manifest)
    pointer6 = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_generation6_authoritative_pointer",
        "status": "COMMITTED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": 6,
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
    return files, manifest_payload, canonical_json(pointer6)


def validate_exact_publish_invocation(
    arguments: argparse.Namespace,
    authority_package: Path,
    authority_review: Path,
) -> None:
    require(arguments.authority_package == authority_package, "authority path differs")
    require(arguments.authority_review == authority_review, "review path differs")
    require(arguments.recovery_package == RECOVERY_PACKAGE, "recovery path differs")
    require(arguments.recovery_review == RECOVERY_REVIEW, "recovery review differs")
    require(arguments.runtime_root == RUNTIME, "runtime root differs")
    require(
        [sys.executable, *sys.argv]
        == authorized_publish_argv(authority_package, authority_review),
        "exact authorized publish argv or interpreter differs",
    )
    require(Path.cwd() == ROOT, "exact repository-root cwd required")
    require(
        os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "deterministic publish environment differs",
    )


def publish(authority_package: Path, authority_review: Path) -> None:
    validate_authority_review(authority_package, authority_review)
    validate_live_zero_state()
    with RUNTIME_LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AuthorityError("another runtime invocation is active") from error
        validate_authority_review(authority_package, authority_review)
        validate_live_zero_state()
        files, manifest_payload, pointer_payload = generation6_payloads(
            authority_package, authority_review
        )
        write_new(
            PUBLICATION_CONSUMPTION,
            canonical_json(
                {
                    "schema_version": 1,
                    "kind": "ace3_transaction5_publication_recovery_authority_consumption",
                    "status": "CONSUMED_ONCE",
                    "runtime_identity": RUNTIME_IDENTITY,
                    "transaction_index": 5,
                    "authority": file_record(
                        authority_package / AUTHORITY_FILENAME
                    ),
                    "independent_review": file_record(authority_review),
                    "authorized_publish_argv": authorized_publish_argv(
                        authority_package, authority_review
                    ),
                    "retry_authorized": False,
                    "replay_authorized": False,
                    "resume_authorized": False,
                }
            ),
        )
        fsync_directory(FUTURE)
        try:
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
            temporary = POINTER.with_name(
                ".authoritative-state.transaction005-publication.tmp"
            )
            write_new(temporary, pointer_payload, 0o600)
            os.replace(temporary, POINTER)
            fsync_directory(POINTER.parent)
            write_new(
                PUBLICATION_TERMINAL,
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "ace3_transaction5_publication_recovery_terminal",
                        "status": "PASS",
                        "generation": 6,
                        "next_transaction_index": 6,
                        "transaction005_retry_replay_resume": False,
                        "transactions006_025_executed": False,
                        "authority_consumption": file_record(
                            PUBLICATION_CONSUMPTION
                        ),
                    }
                ),
            )
            fsync_directory(FUTURE)
        except (
            AuthorityError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            if not PUBLICATION_FAILURE.exists():
                write_new(
                    PUBLICATION_FAILURE,
                    canonical_json(
                        {
                            "schema_version": 1,
                            "kind": (
                                "ace3_transaction5_publication_recovery_"
                                "fail_closed_terminal"
                            ),
                            "status": "FAIL",
                            "error_type": type(error).__name__,
                            "authority_consumed": True,
                            "retry_authorized": False,
                        }
                    ),
                )
                fsync_directory(FUTURE)
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate", "publish"))
    parser.add_argument("--authority-package", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authority-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--recovery-package", type=Path, default=RECOVERY_PACKAGE)
    parser.add_argument("--recovery-review", type=Path, default=RECOVERY_REVIEW)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    arguments = parser.parse_args()
    if arguments.operation == "prepare":
        package = issue_authority(
            arguments.authority_package, arguments.authority_review
        )
        print(
            "TRANSACTION005_PUBLICATION_RECOVERY_AUTHORITY_SEALED "
            f"package={package} authority_consumption=0 publication=0 "
            "model=0 oracle=0 rtl=0 retry=0 replay=0 resume=0 "
            "generation6=absent transactions006_025=absent"
        )
    elif arguments.operation == "validate":
        validate_authority_review(
            arguments.authority_package, arguments.authority_review
        )
        print(
            "TRANSACTION005_PUBLICATION_RECOVERY_AUTHORITY_VALID "
            "authority_consumption=0 publication=0 execution=0 "
            "generation6=absent transactions006_025=absent"
        )
    else:
        validate_exact_publish_invocation(
            arguments,
            arguments.authority_package,
            arguments.authority_review,
        )
        publish(arguments.authority_package, arguments.authority_review)
        print(
            "TRANSACTION005_PUBLICATION_RECOVERY_PUBLISHED "
            "generation=6 cursor=6 transaction005_retry_replay_resume=0"
        )


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
            f"TRANSACTION005_PUBLICATION_RECOVERY_AUTHORITY_REFUSED {error}"
        ) from error
