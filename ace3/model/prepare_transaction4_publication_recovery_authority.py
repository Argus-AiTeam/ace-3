#!/usr/bin/env python3
"""Seal and, only after review, apply transaction004 publication recovery."""

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
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION5_STAGING = GENERATIONS / ".generation-0000000005.prepared"
POINTER = ADOPTION / "authoritative-state.json"
RUNTIME_LOCK = ADOPTION / "execution.lock"
TRANSACTIONS = RUNTIME / "transactions"
FUTURE = RUNTIME / "transaction4-authoritative-generation5"
ORIGINAL_CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
ORIGINAL_FAIL_CLOSED_TERMINAL = FUTURE / "fail-closed-terminal.json"
PUBLICATION_CONSUMPTION = FUTURE / "publication-recovery-authority-consumption.json"
PUBLICATION_TERMINAL = FUTURE / "publication-recovery-terminal.json"
PUBLICATION_FAILURE = FUTURE / "publication-recovery-fail-closed-terminal.json"
SIMULATION_LOG = (
    TRANSACTIONS / "transaction-004/position003/simulation.log"
)
MANAGER_MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/a5b6014a78cf/mission.json"
)
REVIEWED_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/a5b6014a78cf/round-0001.json"
)
RECOVERY_PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery"
    / "r3-postconsume-none-to-false-sourcebound"
)
RECOVERY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery"
    / "reviews/r3-postconsume-none-to-false-sourcebound/independent-review.json"
)
AUTHORITY_COLLECTION = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery_authorities"
)
SUPERSEDED_R3_OUTPUT = (
    AUTHORITY_COLLECTION / "r3-postconsume-none-to-false-sourcebound"
)
SUPERSEDED_R3_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery_authority_reviews"
    / "r3-postconsume-none-to-false-sourcebound/independent-review.json"
)
SUPERSEDED_R4_OUTPUT = (
    AUTHORITY_COLLECTION / "r4-postconsume-sourcebound-publish-order"
)
SUPERSEDED_R4_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery_authority_reviews"
    / "r4-postconsume-sourcebound-publish-order/independent-review.json"
)
DEFAULT_OUTPUT = (
    AUTHORITY_COLLECTION / "r5-manager-direct-consumption-vector-bound"
)
DEFAULT_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_recovery_authority_reviews"
    / "r5-manager-direct-consumption-vector-bound/independent-review.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT
    / "ace3/model/emit_transaction4_publication_recovery_authority_review.py"
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
    / "adjudication.json": "269829ad986045a5aba457df2806b3b245039931e180053658ca23b1d1a77262",
    RECOVERY_PACKAGE
    / "package-manifest.json": "f2c905010de9422bbba4ea02063c6ba64ff7b48af563e757cbf4584e394642b4",
    RECOVERY_PACKAGE
    / "package-seal.json": "65d10506677411212b377ae520f370608b84d7e9d66fa0c5135505b0c548ac48",
    RECOVERY_PACKAGE
    / "recovery-adjudication.py": "a83fef46615999897f768f3f54bfc01d1018ad49d067597d97e68fa700184125",
    RECOVERY_REVIEW: "1e1f343821b4a28c18f7d888e4227f98f95a40314a6ebe03c1cd7421af9d66a5",
    POINTER: "7d7d45423d4e6eb008eef3dd53c5286a9b74e24b56b5737ee56492365be1095a",
    GENERATION4
    / "generation-manifest.json": "b106d6232b26f72f191b1e91a6137cd8a4bbd4e4730e615b549c5a9ad3fe390e",
    GENERATION4
    / "ledger.json": "e87086bb1b752759f15f5b117542fccaa718b34e23af3333f37131997472752a",
    GENERATION4
    / "checkpoints/transaction-003.json": "5b0a0a4596f43338d27a5257a6fbe4bb2fadefbd4dedc14605aa6b7249387d62",
    ORIGINAL_CONSUMPTION: "54e757913086010cf8b13ad48408a903aea912bba51eb85c78c991a20884b2be",
    ORIGINAL_FAIL_CLOSED_TERMINAL: "a2815d8f9fbf53ff63f5af1f9646008ef42826ada52fa0e9c9364819477e0e72",
    SIMULATION_LOG: "b2b0e11ecdaa947af0ed6c659f876c8e1d8e18d562d1a19c048cb64f158fec5e",
    MANAGER_MISSION: "62dc648e21cde01658b50fa3db82f403cee93940fc67220e0a4d4cb6b702282c",
    REVIEWED_HANDOFF: "137b1ca69a9526740a968d2c53b91bc71b731d6067ba21e7180df89522001461",
}
RECOVERY_ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation5_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions005_025_execution": 0,
}
ZERO_ACTIVITY = {
    **RECOVERY_ZERO_ACTIVITY,
    "model_generation": 0,
    "oracle_generation": 0,
    "vector_generation": 0,
}
PERMITTED_OPERATIONS = [
    "reconstruct the transaction004 receipt from accepted r3 frozen evidence",
    "atomically publish generation5, checkpoint004, and cursor5",
]
PROHIBITIONS = [
    "transaction004 retry",
    "transaction004 replay",
    "transaction004 resume",
    "new transaction identity",
    "transaction005-025 execution",
    "model execution",
    "model generation",
    "oracle execution",
    "oracle generation",
    "vector generation",
    "RTL compilation",
    "RTL simulation",
    "publication before the bound independent review is PASS",
    "any publish argv other than the exact bound argv",
    "the superseded r3 authority candidate publish argv",
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
        and mission.get("mission_id") == "a5b6014a78cf"
        and mission.get("node_key") == "tx004-r3-publication-recovery-authority"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and "manager-direct" in mission.get("tags", []),
        "manager-direct mission provenance differs",
    )
    review = reviewed_handoff.get("review", {})
    require(
        reviewed_handoff.get("kind") == "round_reviewed_handoff"
        and reviewed_handoff.get("mission_id") == "a5b6014a78cf"
        and reviewed_handoff.get("producer_role") == "reviewer"
        and review.get("status") == "continue"
        and review.get("next_action")
        == (
            "Preserve the current immutable candidate as rejected/superseded; "
            "create one distinct, sole non-superseded Manager-issued "
            "publication-only authority that directly binds the original "
            "transaction004 consumption record, explicitly forbids vector "
            "generation, and obtains independent PASS checks for Manager "
            "provenance and both requirements."
        ),
        "reviewed Manager issuance direction differs",
    )
    return {
        "mission": file_record(MANAGER_MISSION),
        "reviewed_handoff": file_record(REVIEWED_HANDOFF),
        "mission_id": "a5b6014a78cf",
        "manager_direct": True,
    }


def original_authority_consumption() -> dict[str, Any]:
    consumption = load_json(ORIGINAL_CONSUMPTION)
    manager_authority = consumption.get("manager_authorization", {})
    require(
        consumption.get("kind")
        == "ace3_position3_transaction4_layer3_authority_consumption"
        and consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("runtime_identity") == RUNTIME_IDENTITY
        and consumption.get("transaction_index") == 4
        and consumption.get("layer_index") == 3
        and consumption.get("replay_authorized") is False,
        "original transaction004 authority consumption differs",
    )
    authenticate(manager_authority, "consumed original Manager authority")
    original_manager = load_json(Path(manager_authority["path"]))
    require(
        original_manager.get("kind")
        == "ace3_position3_transaction4_layer3_manager_authorization"
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
        for index in range(5, 26)
        if (TRANSACTIONS / f"transaction-{index:03d}").exists()
    ]


def validate_zero_state(
    *,
    generation5_exists: bool,
    generation5_staging_exists: bool,
    publication_consumption_exists: bool,
    publication_terminal_exists: bool,
    publication_failure_exists: bool,
    later_transaction_indices: list[int],
) -> None:
    require(not generation5_exists, "generation5 already exists")
    require(not generation5_staging_exists, "generation5 staging already exists")
    require(not publication_consumption_exists, "publication authority was consumed")
    require(not publication_terminal_exists, "publication terminal already exists")
    require(not publication_failure_exists, "publication failure already exists")
    require(
        not later_transaction_indices,
        "transaction005-025 artifact exists",
    )


def validate_live_zero_state() -> None:
    validate_zero_state(
        generation5_exists=GENERATION5.exists(),
        generation5_staging_exists=GENERATION5_STAGING.exists(),
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
        "frozen transaction004 failure namespace differs",
    )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION4 / "ledger.json")
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 4
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == 4
        and ledger.get("next_transaction_index") == 4
        and ledger.get("completed_transaction_count") == 4
        and len(ledger.get("completed_receipts", [])) == 4,
        "generation4/cursor4 parent differs",
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
            adjudication["frozen_transaction004"]["tree"]
        ),
    }


def validate_accepted_recovery() -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        RECOVERY_PACKAGE.is_dir()
        and not RECOVERY_PACKAGE.is_symlink()
        and stat.S_IMODE(RECOVERY_PACKAGE.stat().st_mode) & 0o222 == 0,
        "accepted r3 recovery package is absent or writable",
    )
    for path, expected in FIXED_SHA256.items():
        require(sha256_file(path) == expected, f"fixed input hash differs: {path}")
    require(
        stat.S_IMODE(RECOVERY_REVIEW.stat().st_mode) & 0o222 == 0,
        "accepted r3 recovery review is writable",
    )
    seal = load_json(RECOVERY_PACKAGE / "package-seal.json")
    require(
        seal.get("kind")
        == "ace3_transaction4_publication_recovery_adjudication_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and seal.get("generation5_publication_authorized") is False
        and seal.get("generation5_publication_performed") is False
        and set(seal.get("members", {})) == RECOVERY_MEMBERS,
        "accepted r3 recovery seal differs",
    )
    for name, record in seal["members"].items():
        require(
            record.get("path") == str(RECOVERY_PACKAGE / name),
            f"accepted r3 member path differs: {name}",
        )
        authenticate(record, f"accepted r3 member {name}")
    review = load_json(RECOVERY_REVIEW)
    require(
        review.get("kind")
        == "ace3_transaction4_publication_recovery_independent_review"
        and review.get("status") == "PASS"
        and review.get("preparation_participation") is False
        and review.get("institutional_role_authority_claimed") is False
        and review.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and review.get("transaction004_retry_replay_resume") is False
        and review.get("transactions005_025_absent") is True
        and review.get("new_authority_consumed") is False
        and review.get("generation5_publication_authorized") is False
        and review.get("generation5_publication_performed") is False
        and review.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and review.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json"),
        "accepted r3 recovery review differs",
    )
    adjudication = load_json(RECOVERY_PACKAGE / "adjudication.json")
    boundary = adjudication.get("recovery_boundary", {})
    require(
        adjudication.get("kind")
        == "ace3_transaction4_postconsume_publication_recovery_adjudication"
        and adjudication.get("status") == "PASS"
        and adjudication.get("transaction_index") == 4
        and adjudication.get("layer_index") == 3
        and boundary.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation5_publication_authorized") is False
        and boundary.get("generation5_publication_performed") is False
        and boundary.get("transactions005_025_absent") is True,
        "accepted r3 adjudication differs",
    )
    parent = adjudication["authoritative_parent"]
    require(
        parent.get("generation") == 4
        and parent.get("cursor") == 4
        and parent.get("pointer") == file_record(POINTER)
        and parent.get("ledger") == file_record(GENERATION4 / "ledger.json")
        and parent.get("checkpoint003")
        == file_record(GENERATION4 / "checkpoints/transaction-003.json"),
        "accepted r3 generation4/cursor4 parent differs",
    )
    for label, record in frozen_evidence(adjudication).items():
        if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
            authenticate(record, f"frozen transaction004 {label}")
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


def superseded_candidates() -> list[dict[str, Any]]:
    source_path = SUPERSEDED_R3_OUTPUT / PUBLICATION_SOURCE_NAME
    source = source_path.read_text(encoding="utf-8")
    consumption_offset = source.index(
        "write_new(\n            PUBLICATION_CONSUMPTION"
    )
    payload_offset = source.index(
        "files, manifest_payload, pointer_payload = generation5_payloads"
    )
    require(
        consumption_offset < payload_offset
        and "authority = validate_authority_package(" in source
        and "validate_live_zero_state()" in source,
        "superseded candidate defect is not reproducible",
    )
    r4_authority = load_json(SUPERSEDED_R4_OUTPUT / AUTHORITY_FILENAME)
    require(
        r4_authority.get("producer_role") == "engineer"
        and "original_transaction004_authority_consumption" not in r4_authority
        and "vector generation" not in r4_authority.get("prohibitions", []),
        "reviewed r4 rejection basis differs",
    )
    return [
        {
            "package_root": str(SUPERSEDED_R3_OUTPUT),
            "authority_seal": file_record(
                SUPERSEDED_R3_OUTPUT / "authority-seal.json"
            ),
            "authority": file_record(
                SUPERSEDED_R3_OUTPUT / AUTHORITY_FILENAME
            ),
            "publication_source": file_record(source_path),
            "independent_review": file_record(SUPERSEDED_R3_REVIEW),
            "status": "SUPERSEDED_UNUSABLE_UNCONSUMED",
            "defects": [
                (
                    "publish writes publication authority consumption before "
                    "generation5_payloads re-enters the zero-consumption validator"
                )
            ],
            "publish_argv_forbidden": True,
        },
        {
            "package_root": str(SUPERSEDED_R4_OUTPUT),
            "authority_seal": file_record(
                SUPERSEDED_R4_OUTPUT / "authority-seal.json"
            ),
            "authority": file_record(
                SUPERSEDED_R4_OUTPUT / AUTHORITY_FILENAME
            ),
            "publication_source": file_record(
                SUPERSEDED_R4_OUTPUT / PUBLICATION_SOURCE_NAME
            ),
            "independent_review": file_record(SUPERSEDED_R4_REVIEW),
            "status": "SUPERSEDED_REJECTED_UNCONSUMED",
            "defects": [
                "authority producer is Engineer rather than Manager",
                "original transaction004 authority consumption is not directly bound",
                "vector generation is not explicitly prohibited",
            ],
            "publish_argv_forbidden": True,
        },
    ]


def authority_document(
    authority_package: Path,
    authority_review: Path,
    publication_source: Path,
    published_source: Path,
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_only_recovery_manager_authority",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "scope": (
            "receipt reconstruction and atomic generation5/checkpoint004/"
            "cursor5 publication only"
        ),
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 4,
        "layer_index": 3,
        "new_transaction_identity": None,
        "manager_provenance": manager_provenance(),
        "superseded_candidates": superseded_candidates(),
        "original_transaction004_authority_consumption": (
            original_authority_consumption()
        ),
        "accepted_r3_recovery": {
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
        "frozen_transaction004_evidence": frozen_evidence(adjudication),
        "authoritative_parent": {
            "generation": 4,
            "cursor": 4,
            "latest_checkpoint_index": 3,
            "pointer": file_record(POINTER),
            "generation_manifest": file_record(
                GENERATION4 / "generation-manifest.json"
            ),
            "ledger": file_record(GENERATION4 / "ledger.json"),
            "checkpoint003": file_record(
                GENERATION4 / "checkpoints/transaction-003.json"
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
        "historical_superseded_authority_count": 2,
        "authority_consumed": False,
        "publication_performed": False,
        "generation5_exists": False,
        "transactions005_025_absent": True,
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
        == "ace3_transaction4_publication_recovery_authority_seal"
        and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and seal.get("activity_counters") == ZERO_ACTIVITY
        and seal.get("authority_cardinality") == 1
        and seal.get("non_superseded_authority_cardinality") == 1
        and seal.get("historical_superseded_authority_count") == 2
        and seal.get("authority_consumed") is False
        and seal.get("publication_performed") is False
        and seal.get("generation5_exists") is False
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
        set(competing)
        == {
            SUPERSEDED_R3_OUTPUT / AUTHORITY_FILENAME,
            SUPERSEDED_R4_OUTPUT / AUTHORITY_FILENAME,
        },
        "unexpected transaction004 recovery authority set",
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
            "kind": "ace3_transaction4_publication_recovery_authority_review_request",
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
                "publish or stage generation5",
                "retry, replay, or resume transaction004",
                "generate or execute model, oracle, or vectors",
                "compile or simulate RTL",
                "create transaction005-025 artifacts",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(request))
        members = {
            name: file_record(staging / name, authority_package / name)
            for name in AUTHORITY_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_transaction4_publication_recovery_authority_seal",
            "status": "AUTHORIZED_NOT_CONSUMED",
            "members": members,
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "authority_cardinality": 1,
            "non_superseded_authority_cardinality": 1,
            "historical_superseded_authority_count": 2,
            "authority_consumed": False,
            "publication_performed": False,
            "generation5_exists": False,
            "transactions005_025_absent": True,
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
        == "ace3_transaction4_publication_recovery_authority_independent_review"
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
        and review.get("accepted_r3_recovery_review")
        == file_record(RECOVERY_REVIEW)
        and review.get("authorized_publish_argv")
        == authority["authorized_publish"]["argv"]
        and review.get("activity_counters") == ZERO_ACTIVITY
        and review.get("transaction004_retry_replay_resume") is False
        and review.get("model_oracle_rtl_execution") is False
        and review.get("authority_consumed") is False
        and review.get("generation5_publication_performed") is False
        and review.get("transactions005_025_absent") is True,
        "independent authority review differs",
    )
    return review


def generation5_payloads(
    authority_package: Path,
    authority_review: Path,
) -> tuple[dict[str, bytes], bytes, bytes]:
    authority = validate_authority_package(authority_package, authority_review)
    validate_authority_review(authority_package, authority_review)
    adjudication, _ = validate_accepted_recovery()
    receipt = adjudication["reconstructed_receipt"]
    ledger4 = load_json(GENERATION4 / "ledger.json")
    checkpoints = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION4 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(4)
    }
    checkpoints["checkpoints/transaction-004.json"] = canonical_json(receipt)
    checkpoint_records = [
        {
            "path": str(GENERATION5 / f"checkpoints/transaction-{index:03d}.json"),
            "bytes": len(checkpoints[f"checkpoints/transaction-{index:03d}.json"]),
            "sha256": sha256_bytes(
                checkpoints[f"checkpoints/transaction-{index:03d}.json"]
            ),
        }
        for index in range(5)
    ]
    ledger5 = copy.deepcopy(ledger4)
    ledger5.update(
        {
            "authoritative_state_root": str(GENERATION5),
            "completed_receipts": checkpoint_records,
            "completed_transaction_count": 5,
            "next_transaction_index": 5,
            "state_generation": 5,
            "status": "IN_PROGRESS",
            "cumulative_execution_seconds": receipt["timing"][
                "cumulative_execution_seconds"
            ],
        }
    )
    files = {**checkpoints, "ledger.json": canonical_json(ledger5)}
    activity = copy.deepcopy(ZERO_ACTIVITY)
    activity["authority_consumption"] = 1
    activity["generation5_publication"] = 1
    manifest = {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_recovery_generation5_state",
        "status": "PREPARED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": 5,
        "parent_generation": 4,
        "parent_pointer": file_record(POINTER),
        "accepted_r3_recovery_seal": file_record(
            RECOVERY_PACKAGE / "package-seal.json"
        ),
        "accepted_r3_recovery_review": file_record(RECOVERY_REVIEW),
        "publication_authority": file_record(
            authority_package / AUTHORITY_FILENAME
        ),
        "publication_authority_review": file_record(authority_review),
        "original_transaction004_authority_consumption": file_record(
            ORIGINAL_CONSUMPTION
        ),
        "files": {
            relative: {
                "path": str(GENERATION5 / relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for relative, payload in files.items()
        },
        "activity_counters": activity,
        "transaction004_retry_replay_resume": False,
        "transactions005_025_executed": False,
        "authorized_publish_argv": authority["authorized_publish"]["argv"],
    }
    manifest_payload = canonical_json(manifest)
    pointer5 = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_generation5_authoritative_pointer",
        "status": "COMMITTED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": 5,
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
    return files, manifest_payload, canonical_json(pointer5)


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
        files, manifest_payload, pointer_payload = generation5_payloads(
            authority_package, authority_review
        )
        write_new(
            PUBLICATION_CONSUMPTION,
            canonical_json(
                {
                    "schema_version": 1,
                    "kind": "ace3_transaction4_publication_recovery_authority_consumption",
                    "status": "CONSUMED_ONCE",
                    "runtime_identity": RUNTIME_IDENTITY,
                    "transaction_index": 4,
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
            GENERATION5_STAGING.mkdir(mode=0o700)
            (GENERATION5_STAGING / "checkpoints").mkdir(mode=0o700)
            for relative, payload in files.items():
                write_new(GENERATION5_STAGING / relative, payload)
            write_new(
                GENERATION5_STAGING / "generation-manifest.json",
                manifest_payload,
            )
            fsync_directory(GENERATION5_STAGING / "checkpoints")
            fsync_directory(GENERATION5_STAGING)
            os.rename(GENERATION5_STAGING, GENERATION5)
            fsync_directory(GENERATIONS)
            temporary = POINTER.with_name(
                ".authoritative-state.transaction004-publication.tmp"
            )
            write_new(temporary, pointer_payload, 0o600)
            os.replace(temporary, POINTER)
            fsync_directory(POINTER.parent)
            write_new(
                PUBLICATION_TERMINAL,
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "ace3_transaction4_publication_recovery_terminal",
                        "status": "PASS",
                        "generation": 5,
                        "next_transaction_index": 5,
                        "transaction004_retry_replay_resume": False,
                        "transactions005_025_executed": False,
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
                                "ace3_transaction4_publication_recovery_"
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
            "TRANSACTION004_PUBLICATION_RECOVERY_AUTHORITY_SEALED "
            f"package={package} authority_consumption=0 publication=0 "
            "model=0 oracle=0 rtl=0 retry=0 replay=0 resume=0 "
            "generation5=absent transactions005_025=absent"
        )
    elif arguments.operation == "validate":
        validate_authority_review(
            arguments.authority_package, arguments.authority_review
        )
        print(
            "TRANSACTION004_PUBLICATION_RECOVERY_AUTHORITY_VALID "
            "authority_consumption=0 publication=0 execution=0 "
            "generation5=absent transactions005_025=absent"
        )
    else:
        validate_exact_publish_invocation(
            arguments,
            arguments.authority_package,
            arguments.authority_review,
        )
        publish(arguments.authority_package, arguments.authority_review)
        print(
            "TRANSACTION004_PUBLICATION_RECOVERY_PUBLISHED "
            "generation=5 cursor=5 transaction004_retry_replay_resume=0"
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
            f"TRANSACTION004_PUBLICATION_RECOVERY_AUTHORITY_REFUSED {error}"
        ) from error
