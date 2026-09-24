#!/usr/bin/env python3
"""Create the review-gated transaction006/layer05 r4 successor package."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "51e05603a20c"
PARENT_MISSION_ID = "d560dc6d3138"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY

OLD_PACKAGE = RUNTIME / "transaction6-layer5-continuation-package-r3"
OLD_REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-review-r3"
    / "independent-review.json"
)
OLD_AUTHORITY = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-r3"
    / "manager-authority.json"
)
OLD_AUTHORITY_SEAL = OLD_AUTHORITY.with_name("authority-seal.json")
OLD_AUTHORITY_REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-review-r3"
    / "independent-review.json"
)
OLD_AUTHORITY_HOST_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/24854a171c81/round-0001.json"
)

PACKAGE_ID = "generation6-cursor6-checkpoint005-transaction006-layer05-r4"
PACKAGE = RUNTIME / "transaction6-layer5-continuation-package-r4"
REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-review-r4"
    / "independent-review.json"
)
AUTHORITY = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-r4-replacement"
    / "manager-authority.json"
)
AUTHORITY_REVIEW = (
    RUNTIME
    / "transaction6-layer5-continuation-authority-review-r4-replacement"
    / "independent-review.json"
)
PACKAGE_HOST_DECISION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/51e05603a20c/round-0001.json"
)
MANAGER_MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/51e05603a20c/mission.json"
)

ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION6 = ADOPTION / "state-generations/generation-0000000006"
GENERATION7 = ADOPTION / "state-generations/generation-0000000007"
GENERATION7_STAGING = ADOPTION / "state-generations/.generation-0000000007.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION6 = TRANSACTIONS / "transaction-006"
FUTURE_ROOT = RUNTIME / "transaction6-authoritative-generation7"

OLD_FIXED_SHA256 = {
    OLD_PACKAGE
    / "authoritative-baseline.json": "9e71792a253ad6b860623ad50fcb7ee2cbdc7966f395617ceaef7d7f6eb40003",
    OLD_PACKAGE
    / "transaction6_executor.py": "8b5ab23311674c824e10f7a6faa202e4118f8f6d03be1f792a2da94d33aa8af6",
    OLD_PACKAGE
    / "package-manifest.json": "84cd977a9767e1f6413d1f281ecd0991aea1f1f35e26ed507f0df709326dd0f0",
    OLD_PACKAGE
    / "review-emitter.py": "5102136a58d7dcddab2b0d3bf4bfa9dd8fd95fdf44cad76d556321c520cb1f2f",
    OLD_PACKAGE
    / "review-request.json": "079c09bb749c3f866459370ccdbaf6e0bec976679c747248fda2789782849e7f",
    OLD_PACKAGE
    / "source-manifest.json": "f3ee39577e35c19b311905f19f20a89b6c24151b77b29bbef1c8b80bafb044cc",
    OLD_PACKAGE
    / "package-seal.json": "9d091b92acba2c58ecf91987c84dd56f66862eaac2df04e4c696e460578d87b8",
    OLD_REVIEW: "eb5aec0eaa5e017fef6ed2de7ac7cdc128138ceaef0e38224e18a4e772d3666e",
    OLD_AUTHORITY: "899c63413d4a39aa1d10fb3da1043215b415073666866e14a2de8f219ad3af3d",
    OLD_AUTHORITY_SEAL: "ff65290ab114065644419a1fa71fd572d56d8fd27365622fc8427245bc3e6d11",
    OLD_AUTHORITY_REVIEW: "f543ac8aaa71285962709af6eb95ae548d44d0a13592ed1cd7359c884b55578b",
    OLD_AUTHORITY_HOST_REVIEW: "d91cd382e3d7917f3d19df2a1ec7908d6dfb89300ffae09c45244a0332411944",
    POINTER: "d3042e1e75afebb7996aa618390f5dc8dfada73bfcdae028189d19d100fb0e47",
    GENERATION6
    / "generation-manifest.json": "4991277d157154eb564f6ebf7a442fa9d7d1375c58025b1bef22d477aaf1981c",
    GENERATION6
    / "ledger.json": "73ae94073b546410f60fae32bcdcf00b99b2819c5a2ca16e9ffa6613bf4892c0",
    GENERATION6
    / "checkpoints/transaction-005.json": "4145c04113658114bfe0e428661212914231465baf4e221737d16d9e6821f896",
    TRANSACTIONS
    / "transaction-005/position004.state": "0b35421e91650d2c131116bade95af38d90230cecf5d78ca6abdc99890d0f865",
}

MEMBERS = {
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


class SuccessorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorError(message)


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
    expected_path: Path,
) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"}
        and record.get("path") == str(expected_path)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str)
        and file_record(expected_path) == dict(record),
        f"{label} differs",
    )


def require_fixed(path: Path) -> dict[str, Any]:
    record = file_record(path)
    require(
        record["sha256"] == OLD_FIXED_SHA256[path],
        f"preserved fixed artifact differs: {path}",
    )
    return record


def replace_once(source: str, old: str, new: str, label: str) -> str:
    require(source.count(old) == 1, f"{label} replacement count differs")
    return source.replace(old, new, 1)


def replace_all(source: str, old: str, new: str, label: str) -> str:
    require(old in source, f"{label} replacement source absent")
    return source.replace(old, new)


def supersession_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_transaction006_layer05_r3_supersession",
        "status": "SUPERSEDED_NON_AUTHORITATIVE_HISTORY",
        "operator_decision": file_record(MANAGER_MISSION),
        "reviewed_r3_package_seal": require_fixed(
            OLD_PACKAGE / "package-seal.json"
        ),
        "reviewed_r3_package_acceptance": require_fixed(OLD_REVIEW),
        "rejected_r3_authority": require_fixed(OLD_AUTHORITY),
        "rejected_r3_authority_seal": require_fixed(OLD_AUTHORITY_SEAL),
        "rejected_r3_authority_review": require_fixed(OLD_AUTHORITY_REVIEW),
        "rejected_r3_host_review": require_fixed(OLD_AUTHORITY_HOST_REVIEW),
        "r3_authority_consumed": False,
        "r3_authority_remains_non_authoritative": True,
        "successor_package": str(PACKAGE),
        "replacement_authority": str(AUTHORITY),
    }


def validate_preserved_history() -> None:
    for path in OLD_FIXED_SHA256:
        require_fixed(path)
    package_review = load_json(OLD_REVIEW)
    authority = load_json(OLD_AUTHORITY)
    authority_review = load_json(OLD_AUTHORITY_REVIEW)
    require(
        package_review.get("status") == "PASS"
        and package_review.get("transaction006_executed") is False,
        "reviewed r3 package history differs",
    )
    require(
        authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("authority_consumed") is False
        and authority.get("transaction006_executed") is False,
        "rejected r3 authority history differs",
    )
    require(
        authority_review.get("status") == "REJECT"
        and authority_review.get("total_workload_calls") == 0
        and authority_review.get("transaction006_executed") is False,
        "rejected r3 review history differs",
    )


def validate_manager_decision() -> None:
    mission = load_json(MANAGER_MISSION)
    objective = mission.get("objective", "")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key")
        == "tx006-layer05-authorize-once-operator-answer-operator-answer"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and isinstance(objective, str)
        and "Authorize option 1." in objective
        and "Create exactly one create-exclusive successor package revision"
        in objective,
        "Manager successor decision differs",
    )


def validate_zero_runtime() -> None:
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION6 / "ledger.json")
    checkpoint = load_json(GENERATION6 / "checkpoints/transaction-005.json")
    require(
        pointer.get("generation") == 6
        and pointer.get("status") == "COMMITTED"
        and ledger.get("state_generation") == 6
        and ledger.get("next_transaction_index") == 6
        and ledger.get("completed_transaction_count") == 6
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 5,
        "generation6/cursor6/checkpoint005 parent differs",
    )
    require(
        not TRANSACTION6.exists()
        and all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(7, 26)
        )
        and not GENERATION7.exists()
        and not GENERATION7_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "transaction006-025 or generation7 activity exists",
    )


def source_protocol_constants() -> str:
    return f'''PACKAGE_HOST_DECISION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/{MISSION_ID}/round-0001.json"
)
SUPERSEDED_PACKAGE_SEAL = Path(
    "{OLD_PACKAGE / "package-seal.json"}"
)
SUPERSEDED_PACKAGE_REVIEW = Path(
    "{OLD_REVIEW}"
)
SUPERSEDED_AUTHORITY = Path(
    "{OLD_AUTHORITY}"
)
SUPERSEDED_AUTHORITY_SEAL = Path(
    "{OLD_AUTHORITY_SEAL}"
)
SUPERSEDED_AUTHORITY_REVIEW = Path(
    "{OLD_AUTHORITY_REVIEW}"
)
SUPERSEDED_AUTHORITY_HOST_REVIEW = Path(
    "{OLD_AUTHORITY_HOST_REVIEW}"
)
MANAGER_SUCCESSOR_MISSION = Path(
    "{MANAGER_MISSION}"
)'''


def source_supersession_function() -> str:
    return '''
def expected_supersession() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_transaction006_layer05_r3_supersession",
        "status": "SUPERSEDED_NON_AUTHORITATIVE_HISTORY",
        "operator_decision": file_record(MANAGER_SUCCESSOR_MISSION),
        "reviewed_r3_package_seal": file_record(SUPERSEDED_PACKAGE_SEAL),
        "reviewed_r3_package_acceptance": file_record(SUPERSEDED_PACKAGE_REVIEW),
        "rejected_r3_authority": file_record(SUPERSEDED_AUTHORITY),
        "rejected_r3_authority_seal": file_record(SUPERSEDED_AUTHORITY_SEAL),
        "rejected_r3_authority_review": file_record(SUPERSEDED_AUTHORITY_REVIEW),
        "rejected_r3_host_review": file_record(SUPERSEDED_AUTHORITY_HOST_REVIEW),
        "r3_authority_consumed": False,
        "r3_authority_remains_non_authoritative": True,
        "successor_package": str(PACKAGE_ROOT),
        "replacement_authority": str(AUTHORITY),
    }

'''


def transform_common(source: str) -> str:
    source = replace_once(
        source,
        'MISSION_ID = "12fb90389c40"',
        f'MISSION_ID = "{MISSION_ID}"',
        "mission",
    )
    source = replace_once(
        source,
        'PACKAGE_ID = "generation6-cursor6-checkpoint005-transaction006-layer05-r3"',
        f'PACKAGE_ID = "{PACKAGE_ID}"',
        "package identity",
    )
    source = replace_all(
        source,
        "transaction6-layer5-continuation-package-r3",
        "transaction6-layer5-continuation-package-r4",
        "package path",
    )
    source = replace_all(
        source,
        "transaction6-layer5-continuation-review-r3",
        "transaction6-layer5-continuation-review-r4",
        "review path",
    )
    source = replace_all(
        source,
        "transaction6-layer5-continuation-authority-r3",
        "transaction6-layer5-continuation-authority-r4-replacement",
        "authority path",
    )
    source = replace_once(
        source,
        'PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")',
        'PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")\n'
        + source_protocol_constants(),
        "stable Host provenance constants",
    )
    source = replace_once(
        source,
        "\ndef tree_digest(",
        source_supersession_function() + "def tree_digest(",
        "supersession function",
    )
    return source


def transform_executor() -> bytes:
    source = (OLD_PACKAGE / "transaction6_executor.py").read_text(
        encoding="utf-8"
    )
    source = transform_common(source)
    source = replace_once(
        source,
        "and document.get(\"activity_counters\") == ZERO_COUNTERS,",
        "and document.get(\"activity_counters\") == ZERO_COUNTERS\n"
        "        and document.get(\"stable_host_reviewer_provenance\")\n"
        "        == str(PACKAGE_HOST_DECISION)\n"
        "        and document.get(\"supersession\") == expected_supersession(),",
        "manifest protocol bindings",
    )
    source = replace_once(
        source,
        'and authority.get("package_acceptance") == file_record(REVIEW)',
        'and authority.get("package_acceptance") == file_record(REVIEW)\n'
        "        and authority.get(\"host_reviewer_decision\")\n"
        "        == file_record(PACKAGE_HOST_DECISION)",
        "authority Host provenance record",
    )
    source = replace_once(
        source,
        'authenticate(host_decision, "Host Reviewer decision")',
        'authenticate(\n'
        '        host_decision, "Host Reviewer decision", PACKAGE_HOST_DECISION\n'
        "    )\n"
        "    decision = load_json(PACKAGE_HOST_DECISION)\n"
        "    require(\n"
        '        decision.get("kind") == "round_reviewed_handoff"\n'
        "        and decision.get(\"mission_id\") == MISSION_ID\n"
        '        and decision.get("producer_role") == "reviewer"\n'
        '        and decision.get("review", {}).get("status") == "done",\n'
        '        "stable Host-saved package Reviewer provenance differs",\n'
        "    )",
        "runtime Host provenance validation",
    )
    ast.parse(source)
    return source.encode("utf-8")


def transform_reviewer() -> bytes:
    source = (OLD_PACKAGE / "review-emitter.py").read_text(encoding="utf-8")
    source = transform_common(source)
    source = replace_once(
        source,
        "and document.get(\"activity_counters\") == ZERO_COUNTERS,",
        "and document.get(\"activity_counters\") == ZERO_COUNTERS\n"
        "        and document.get(\"stable_host_reviewer_provenance\")\n"
        "        == str(PACKAGE_HOST_DECISION)\n"
        "        and document.get(\"supersession\") == expected_supersession(),",
        "Reviewer manifest protocol bindings",
    )
    source = replace_once(
        source,
        '        "executor transaction/layer/parent constants differ",\n    )',
        '        "executor transaction/layer/parent constants differ",\n'
        "    )\n"
        "    require(\n"
        "        MISSION_ID in source\n"
        "        and \"file_record(PACKAGE_HOST_DECISION)\" in source\n"
        "        and \"expected_supersession()\" in source,\n"
        '        "executor lacks stable Host provenance or supersession binding",\n'
        "    )",
        "Reviewer executor protocol check",
    )
    source = replace_once(
        source,
        '        "package_seal": file_record(package / "package-seal.json"),',
        '        "package_seal": file_record(package / "package-seal.json"),\n'
        '        "supersession": expected_supersession(),\n'
        '        "stable_host_reviewer_provenance": str(PACKAGE_HOST_DECISION),',
        "Reviewer result protocol records",
    )
    ast.parse(source)
    return source.encode("utf-8")


def replace_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, str):
        return (
            value.replace(
                "transaction6-layer5-continuation-package-r3",
                "transaction6-layer5-continuation-package-r4",
            )
            .replace(
                "transaction6-layer5-continuation-review-r3",
                "transaction6-layer5-continuation-review-r4",
            )
            .replace(
                "transaction6-layer5-continuation-authority-r3",
                "transaction6-layer5-continuation-authority-r4-replacement",
            )
        )
    return value


def write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def protocol_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "runtime_identity",
        "authoritative_generation",
        "authoritative_cursor",
        "required_parent_checkpoint_index",
        "transaction_index",
        "layer_index",
        "transaction_position",
        "permitted_transaction_indices",
        "forbidden_transaction_indices",
        "transaction_descriptor",
        "position004_input_state",
        "official_frozen_evidence",
        "compile",
        "simulation",
        "transaction006_output_namespace",
        "generation7_output_namespace",
        "restart_adoption",
        "execution_authorized",
        "authority_created",
        "activity_counters",
        "prohibitions",
    )
    return {key: copy.deepcopy(document[key]) for key in keys}


def validate_package() -> dict[str, Any]:
    require(
        PACKAGE.is_dir()
        and not PACKAGE.is_symlink()
        and stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "successor package is absent or writable",
    )
    old_manifest = load_json(OLD_PACKAGE / "package-manifest.json")
    manifest = load_json(PACKAGE / "package-manifest.json")
    seal = load_json(PACKAGE / "package-seal.json")
    sources = load_json(PACKAGE / "source-manifest.json")
    require(
        manifest.get("mission_id") == MISSION_ID
        and manifest.get("parent_mission_id") == PARENT_MISSION_ID
        and manifest.get("package_id") == PACKAGE_ID
        and manifest.get("review_output") == str(REVIEW)
        and manifest.get("future_manager_authority") == str(AUTHORITY)
        and manifest.get("stable_host_reviewer_provenance")
        == str(PACKAGE_HOST_DECISION)
        and manifest.get("supersession") == supersession_document()
        and protocol_fields(manifest) == protocol_fields(old_manifest),
        "successor package changed a frozen workload or parent field",
    )
    require(
        replace_paths(old_manifest["launch"]) == manifest["launch"],
        "successor sealed launch differs beyond bound paths",
    )
    require(
        set(seal.get("members", {})) == set(MEMBERS)
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("activity_counters") == ZERO_COUNTERS,
        "successor package seal differs",
    )
    for label, relative in MEMBERS.items():
        authenticate(seal["members"][label], label, PACKAGE / relative)
    old_sources = load_json(OLD_PACKAGE / "source-manifest.json")
    require(
        {
            key: value
            for key, value in sources["sources"].items()
            if key != "transaction006_executor"
        }
        == {
            key: value
            for key, value in old_sources["sources"].items()
            if key != "transaction006_executor"
        }
        and sources["toolchain"] == old_sources["toolchain"],
        "successor source/toolchain closure changed outside the executor",
    )
    authenticate(
        sources["sources"]["transaction006_executor"],
        "successor executor source",
        PACKAGE / "transaction6_executor.py",
    )
    require(
        not REVIEW.exists()
        and not AUTHORITY.exists()
        and not AUTHORITY_REVIEW.exists()
        and not PACKAGE_HOST_DECISION.exists(),
        "successor review, authority, or Host decision exists prematurely",
    )
    validate_preserved_history()
    validate_zero_runtime()
    return manifest


def prepare_package() -> Path:
    validate_manager_decision()
    validate_preserved_history()
    validate_zero_runtime()
    require(not PACKAGE.exists(), f"successor package exists: {PACKAGE}")
    require(not REVIEW.exists(), f"successor review exists: {REVIEW}")
    require(not AUTHORITY.exists(), f"replacement authority exists: {AUTHORITY}")
    require(
        not AUTHORITY_REVIEW.exists(),
        f"replacement authority review exists: {AUTHORITY_REVIEW}",
    )
    require(
        not PACKAGE_HOST_DECISION.exists(),
        f"Host package Reviewer decision exists prematurely: {PACKAGE_HOST_DECISION}",
    )

    executor = transform_executor()
    reviewer = transform_reviewer()
    baseline = (OLD_PACKAGE / "authoritative-baseline.json").read_bytes()
    old_manifest = load_json(OLD_PACKAGE / "package-manifest.json")
    old_sources = load_json(OLD_PACKAGE / "source-manifest.json")
    old_request = load_json(OLD_PACKAGE / "review-request.json")

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    staging = PACKAGE.with_name(f".{PACKAGE.name}.preparing")
    require(not staging.exists(), f"successor staging exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        write_new(staging / "transaction6_executor.py", executor)
        write_new(staging / "review-emitter.py", reviewer)
        write_new(staging / "authoritative-baseline.json", baseline)

        sources = copy.deepcopy(old_sources)
        sources["sources"]["transaction006_executor"] = file_record(
            staging / "transaction6_executor.py",
            PACKAGE / "transaction6_executor.py",
        )
        write_new(staging / "source-manifest.json", canonical_json(sources))

        manifest = replace_paths(copy.deepcopy(old_manifest))
        manifest["mission_id"] = MISSION_ID
        manifest["package_id"] = PACKAGE_ID
        manifest["authoritative_baseline"] = file_record(
            staging / "authoritative-baseline.json",
            PACKAGE / "authoritative-baseline.json",
        )
        manifest["source_manifest"] = file_record(
            staging / "source-manifest.json",
            PACKAGE / "source-manifest.json",
        )
        manifest["stable_host_reviewer_provenance"] = str(PACKAGE_HOST_DECISION)
        manifest["supersession"] = supersession_document()
        write_new(staging / "package-manifest.json", canonical_json(manifest))

        request = replace_paths(copy.deepcopy(old_request))
        request["mission_id"] = MISSION_ID
        request["reviewer_emitter"] = file_record(
            staging / "review-emitter.py",
            PACKAGE / "review-emitter.py",
        )
        request["stable_host_reviewer_provenance"] = str(PACKAGE_HOST_DECISION)
        request["supersession"] = supersession_document()
        write_new(staging / "review-request.json", canonical_json(request))

        members = {
            label: file_record(staging / relative, PACKAGE / relative)
            for label, relative in MEMBERS.items()
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_position3_transaction6_layer5_executor_seal",
            "status": "SEALED_REVIEW_REQUIRED",
            "members": members,
            "execution_authorized": False,
            "authority_created": False,
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        }
        write_new(staging / "package-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, PACKAGE)
        fsync_directory(PACKAGE.parent)
        PACKAGE.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_package()
    return PACKAGE


def independent_assessment() -> tuple[str, int, str | None]:
    specification = importlib.util.spec_from_file_location(
        "tx006_layer05_r4_package_reviewer",
        PACKAGE / "review-emitter.py",
    )
    require(
        specification is not None and specification.loader is not None,
        "sealed Reviewer import failed",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    status, controls, reason = module.assess_package(PACKAGE)
    return status, len(controls), reason


def preflight() -> None:
    validate_manager_decision()
    validate_preserved_history()
    validate_zero_runtime()
    require(not PACKAGE.exists(), f"successor package exists: {PACKAGE}")
    require(not REVIEW.exists(), f"successor review exists: {REVIEW}")
    require(not AUTHORITY.exists(), f"replacement authority exists: {AUTHORITY}")
    require(
        not AUTHORITY_REVIEW.exists(),
        f"replacement authority review exists: {AUTHORITY_REVIEW}",
    )
    executor = transform_executor().decode("utf-8")
    reviewer = transform_reviewer().decode("utf-8")
    require(
        MISSION_ID in executor
        and AUTHORITY.parent.name in executor
        and "expected_supersession()" in executor
        and MISSION_ID in reviewer
        and AUTHORITY.parent.name in reviewer
        and "expected_supersession()" in reviewer,
        "transformed protocol bindings are incomplete",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "validate"))
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        preflight()
        print("TRANSACTION006_LAYER05_SUCCESSOR_PREFLIGHT_PASS")
        return
    if arguments.command == "prepare":
        output = prepare_package()
    else:
        output = PACKAGE
        validate_package()
    status, controls, reason = independent_assessment()
    require(status == "PASS", f"independent assessment failed: {reason}")
    print(
        "TRANSACTION006_LAYER05_SUCCESSOR_PACKAGE_READY "
        f"path={output} assessment={status} controls={controls} "
        f"review_output_absent={not REVIEW.exists()}"
    )


if __name__ == "__main__":
    try:
        main()
    except (SuccessorError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(
            f"TRANSACTION006_LAYER05_SUCCESSOR_PACKAGE_REFUSED {error}"
        ) from error
