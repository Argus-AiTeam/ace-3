#!/usr/bin/env python3
"""Issue one review-gated Manager authority for transaction005/layer04."""

from __future__ import annotations

import argparse
import copy
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
REVIEW_EMITTER_SOURCE = (
    ROOT / "ace3/model/emit_transaction5_layer4_authority_review.py"
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


def validate_accepted_package() -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        PACKAGE_ROOT.is_dir()
        and not PACKAGE_ROOT.is_symlink()
        and stat.S_IMODE(PACKAGE_ROOT.stat().st_mode) & 0o222 == 0,
        "accepted package is absent or writable",
    )
    for path in FIXED_SHA256:
        require_fixed(path)
    require(
        stat.S_IMODE(PACKAGE_REVIEW.stat().st_mode) & 0o222 == 0,
        "accepted package review is writable",
    )

    manifest = load_json(PACKAGE_MANIFEST)
    seal = load_json(PACKAGE_SEAL)
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
        and manifest.get("position004_input_state")
        == file_record(TRANSACTIONS / "transaction-004/position004.state")
        and manifest.get("transaction005_output_namespace") == str(TRANSACTION5)
        and manifest.get("generation6_output_namespace") == str(GENERATION6)
        and manifest.get("execution_authorized") is False
        and manifest.get("authority_created") is False
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "accepted package identity, scope, or zero state differs",
    )
    require(
        seal.get("kind") == "ace3_position3_transaction5_layer4_executor_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("execution_authorized") is False
        and seal.get("authority_created") is False
        and seal.get("activity_counters") == ZERO_COUNTERS,
        "accepted package seal differs",
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
    return manifest, review


def validate_parent() -> None:
    pointer = load_json(POINTER)
    generation_manifest = load_json(GENERATION5 / "generation-manifest.json")
    ledger = load_json(GENERATION5 / "ledger.json")
    checkpoint = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    position = file_record(TRANSACTIONS / "transaction-004/position004.state")
    require(
        pointer.get("kind")
        == "ace3_position3_transaction4_generation5_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and pointer.get("generation") == 5
        and pointer.get("generation_manifest")
        == file_record(GENERATION5 / "generation-manifest.json"),
        "authoritative generation5 pointer differs",
    )
    require(
        generation_manifest.get("kind")
        == "ace3_transaction4_publication_recovery_generation5_state"
        and generation_manifest.get("generation") == 5
        and generation_manifest.get("transactions005_025_executed") is False,
        "generation5 manifest differs",
    )
    require(
        ledger.get("state_generation") == 5
        and ledger.get("next_transaction_index") == 5
        and ledger.get("completed_transaction_count") == 5
        and checkpoint.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 4
        and checkpoint.get("outputs", {}).get("state") == position,
        "generation5 ledger or checkpoint004 differs",
    )


def validate_live_absence() -> None:
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(5, 26)
        ),
        "transaction005-025 namespace exists",
    )
    require(
        not GENERATION6.exists()
        and not GENERATION6_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "transaction005 or generation6 runtime namespace exists",
    )


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


def authority_document(
    manifest: Mapping[str, Any],
    authority_path: Path = AUTHORITY,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_layer4_manager_authorization",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "mission_id": MISSION_ID,
        "manager_provenance": manager_provenance(),
        "authority_cardinality": 1,
        "runtime_identity": RUNTIME_IDENTITY,
        "authoritative_generation": 5,
        "authoritative_cursor": 5,
        "required_parent_checkpoint_index": 4,
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
        "accepted_package": {
            "root": str(PACKAGE_ROOT),
            "package_seal": file_record(PACKAGE_SEAL),
            "package_manifest": file_record(PACKAGE_MANIFEST),
            "source_manifest": file_record(SOURCE_MANIFEST),
            "authoritative_baseline": file_record(BASELINE),
            "independent_review": file_record(PACKAGE_REVIEW),
            "host_reviewer_decision": file_record(PACKAGE_HOST_DECISION),
        },
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
            "transaction005": str(TRANSACTION5),
            "generation6": str(GENERATION6),
            "authority": str(authority_path),
            "authority_consumption": str(
                FUTURE_ROOT / "manager-authorization-consumption.json"
            ),
        },
        "prohibitions": [
            "transaction000-004 replay",
            "transaction006-025 execution",
            "any launch argv, cwd, or environment other than the exact bound launch",
            "any transaction005 or generation6 work before authority consumption",
            "authority replay or second consumption",
            "model, oracle, vector, compile, simulation, or submission during review",
        ],
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "authority_consumed": False,
        "replay_authorized": False,
        "transaction005_executed": False,
        "generation6_exists": False,
        "transactions006_025_absent": True,
        "reviewer_acceptance": {
            "required": True,
            "path": str(AUTHORITY_REVIEW),
            "present_at_issuance": False,
        },
    }


def validate_authority_package(
    authority_package: Path = AUTHORITY_PACKAGE,
) -> dict[str, Any]:
    manifest, _ = validate_accepted_package()
    validate_parent()
    validate_live_absence()
    require(
        authority_package == AUTHORITY_PACKAGE
        and authority_package.is_dir()
        and not authority_package.is_symlink()
        and stat.S_IMODE(authority_package.stat().st_mode) & 0o222 == 0,
        "authority package is absent, misplaced, or writable",
    )
    actual = {path.name for path in authority_package.iterdir() if path.is_file()}
    require(
        actual == AUTHORITY_MEMBERS | {"authority-seal.json"},
        "authority package file set differs",
    )
    seal = load_json(AUTHORITY_SEAL)
    require(
        seal.get("kind")
        == "ace3_position3_transaction5_layer4_manager_authority_seal"
        and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and seal.get("members", {}).keys() == AUTHORITY_MEMBERS
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
    authority = load_json(AUTHORITY)
    require(
        authority == authority_document(manifest),
        "Manager authority document differs",
    )
    request = load_json(authority_package / "review-request.json")
    require(
        request.get("kind")
        == "ace3_position3_transaction5_layer4_authority_review_request"
        and request.get("mission_id") == MISSION_ID
        and request.get("required_role") == "reviewer"
        and request.get("review_output") == str(AUTHORITY_REVIEW)
        and request.get("authority") == file_record(AUTHORITY)
        and request.get("package_acceptance") == file_record(PACKAGE_REVIEW)
        and request.get("review_invocation")
        == [
            "/usr/bin/python3",
            str(authority_package / "review-emitter.py"),
            "--authority-package",
            str(authority_package),
            "--output",
            str(AUTHORITY_REVIEW),
        ],
        "authority review request differs",
    )
    return authority


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


def issue_authority() -> Path:
    require(not AUTHORITY_PACKAGE.exists(), f"authority exists: {AUTHORITY_PACKAGE}")
    require(not AUTHORITY_REVIEW.exists(), f"authority review exists: {AUTHORITY_REVIEW}")
    require(
        REVIEW_EMITTER_SOURCE.is_file()
        and not REVIEW_EMITTER_SOURCE.is_symlink(),
        "review emitter source is absent or symlinked",
    )
    manifest, _ = validate_accepted_package()
    validate_parent()
    validate_live_absence()
    competing = (
        list(AUTHORITY_PACKAGE.parent.rglob("manager-authority.json"))
        if AUTHORITY_PACKAGE.parent.exists()
        else []
    )
    require(not competing, "transaction005 Manager authority already exists")

    AUTHORITY_PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    AUTHORITY_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    staging = AUTHORITY_PACKAGE.with_name(f".{AUTHORITY_PACKAGE.name}.preparing")
    require(not staging.exists(), f"authority staging exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        write_new(
            staging / "manager-authority.json",
            canonical_json(authority_document(manifest)),
        )
        write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
        request = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction5_layer4_authority_review_request",
            "mission_id": MISSION_ID,
            "required_role": "reviewer",
            "preparation_participation_required": False,
            "review_output": str(AUTHORITY_REVIEW),
            "authority": file_record(
                staging / "manager-authority.json",
                AUTHORITY,
            ),
            "package_acceptance": file_record(PACKAGE_REVIEW),
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
                "consume or replay authority",
                "execute transaction005 or transaction006-025",
                "create transaction005, generation6, or runtime evidence",
                "execute model, oracle, vectors, compile, simulation, or submission",
                "mutate generations3-5 or transactions000-004",
            ],
        }
        write_new(
            staging / "review-request.json",
            canonical_json(request),
        )
        members = {
            name: file_record(staging / name, AUTHORITY_PACKAGE / name)
            for name in AUTHORITY_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction5_layer4_manager_authority_seal",
            "status": "AUTHORIZED_NOT_CONSUMED",
            "members": members,
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
            "authority_cardinality": 1,
            "authority_consumed": False,
            "replay_authorized": False,
            "transaction005_executed": False,
            "generation6_exists": False,
            "transactions006_025_absent": True,
        }
        write_new(staging / "authority-seal.json", canonical_json(seal))

        fsync_directory(staging)
        os.rename(staging, AUTHORITY_PACKAGE)
        fsync_directory(AUTHORITY_PACKAGE.parent)
        AUTHORITY_PACKAGE.chmod(0o555)
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
        print(f"TRANSACTION005_LAYER04_AUTHORITY_ISSUED output={output}")
    else:
        validate_authority_package()
        print(f"TRANSACTION005_LAYER04_AUTHORITY_VALID path={AUTHORITY_PACKAGE}")


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
        raise SystemExit(f"TRANSACTION005_LAYER04_AUTHORITY_REFUSED {error}") from error
