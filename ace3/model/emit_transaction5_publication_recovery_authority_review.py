#!/usr/bin/env python3
"""Source-disjoint review of transaction005 publication-recovery authority."""

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
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATIONS = ADOPTION / "state-generations"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION6 = GENERATIONS / "generation-0000000006"
GENERATION6_STAGING = GENERATIONS / ".generation-0000000006.prepared"
POINTER = ADOPTION / "authoritative-state.json"
TRANSACTIONS = RUNTIME / "transactions"
FUTURE = RUNTIME / "transaction5-authoritative-generation6"
ORIGINAL_CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
ORIGINAL_FAIL_CLOSED_TERMINAL = FUTURE / "fail-closed-terminal.json"
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
AUTHORITY_PACKAGE = (
    AUTHORITY_COLLECTION / "r2-manager-direct-recovery-bound"
)
AUTHORITY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery_authority_reviews"
    / "r2-manager-direct-recovery-bound/independent-review.json"
)
AUTHORITY_FILENAME = "publication-recovery-authority.json"
PUBLICATION_SOURCE_NAME = "publication-recovery-authority.py"
AUTHORITY_MEMBERS = {
    AUTHORITY_FILENAME,
    PUBLICATION_SOURCE_NAME,
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


def authorized_publish_argv() -> list[str]:
    return [
        "/usr/bin/python3",
        str(AUTHORITY_PACKAGE / PUBLICATION_SOURCE_NAME),
        "publish",
        "--authority-package",
        str(AUTHORITY_PACKAGE),
        "--authority-review",
        str(AUTHORITY_REVIEW),
        "--recovery-package",
        str(RECOVERY_PACKAGE),
        "--recovery-review",
        str(RECOVERY_REVIEW),
        "--runtime-root",
        str(RUNTIME),
    ]


def source_boundary(source: str) -> None:
    tree = ast.parse(source)
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


def validate_recovery_review(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_transaction5_publication_recovery_independent_review"
        and document.get("status") == "PASS"
        and document.get("preparation_participation") is False
        and document.get("institutional_role_authority_claimed") is False
        and document.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and document.get("transaction005_retry_replay_resume") is False
        and document.get("transactions006_025_absent") is True
        and document.get("new_authority_consumed") is False
        and document.get("generation6_publication_authorized") is False
        and document.get("generation6_publication_performed") is False
        and document.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and document.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json"),
        "accepted r1 recovery review differs",
    )


def validate_authority_document(
    authority: Mapping[str, Any], authenticate_records: bool = True
) -> None:
    require(
        authority.get("kind")
        == "ace3_transaction5_publication_only_recovery_manager_authority"
        and authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("producer_role") == "manager"
        and authority.get("scope")
        == (
            "receipt reconstruction and atomic generation6/checkpoint005/"
            "cursor6 publication only"
        )
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("transaction_index") == 5
        and authority.get("layer_index") == 4
        and authority.get("new_transaction_identity") is None,
        "publication-recovery authority identity differs",
    )
    require(
        authority.get("manager_provenance") == manager_provenance(),
        "Manager provenance binding differs",
    )
    require(
        authority.get("superseded_candidate") == superseded_candidate(),
        "superseded candidate binding differs",
    )
    require(
        authority.get("original_transaction005_authority_consumption")
        == original_authority_consumption(),
        "original transaction005 authority consumption binding differs",
    )
    accepted = authority.get("accepted_r1_recovery", {})
    require(
        accepted.get("package_root") == str(RECOVERY_PACKAGE)
        and accepted.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json")
        and accepted.get("package_manifest")
        == file_record(RECOVERY_PACKAGE / "package-manifest.json")
        and accepted.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and accepted.get("recovery_source")
        == file_record(RECOVERY_PACKAGE / "recovery-adjudication.py")
        and accepted.get("independent_review") == file_record(RECOVERY_REVIEW),
        "accepted r1 recovery binding differs",
    )
    adjudication = load_json(RECOVERY_PACKAGE / "adjudication.json")
    require(
        authority.get("frozen_transaction005_evidence")
        == frozen_evidence(adjudication),
        "frozen transaction005 evidence binding differs",
    )
    parent = authority.get("authoritative_parent", {})
    require(
        parent.get("generation") == 5
        and parent.get("cursor") == 5
        and parent.get("latest_checkpoint_index") == 4
        and parent.get("pointer") == file_record(POINTER)
        and parent.get("generation_manifest")
        == file_record(GENERATION5 / "generation-manifest.json")
        and parent.get("ledger") == file_record(GENERATION5 / "ledger.json")
        and parent.get("checkpoint004")
        == file_record(GENERATION5 / "checkpoints/transaction-004.json"),
        "generation5/cursor5/checkpoint004 binding differs",
    )
    require(
        authority.get("publication_source")
        == file_record(AUTHORITY_PACKAGE / PUBLICATION_SOURCE_NAME),
        "publication source binding differs",
    )
    require(
        authority.get("authorized_publish")
        == {
            "argv": authorized_publish_argv(),
            "cwd": str(ROOT),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
            },
        },
        "exact authorized publish invocation differs",
    )
    require(
        authority.get("permitted_operations") == PERMITTED_OPERATIONS,
        "permitted publication-only operations differ",
    )
    require(authority.get("prohibitions") == PROHIBITIONS, "prohibitions differ")
    require(
        authority.get("activity_counters") == ZERO_ACTIVITY
        and authority.get("authority_cardinality") == 1
        and authority.get("non_superseded_authority_cardinality") == 1
        and authority.get("historical_superseded_authority_count") == 1
        and authority.get("authority_consumed") is False
        and authority.get("publication_performed") is False
        and authority.get("generation6_exists") is False
        and authority.get("transactions006_025_absent") is True,
        "authority lifecycle or zero counters differ",
    )
    reviewer = authority.get("reviewer_acceptance", {})
    require(
        reviewer.get("required") is True
        and reviewer.get("path") == str(AUTHORITY_REVIEW)
        and reviewer.get("present_at_issuance") is False,
        "independent review gate differs",
    )
    if authenticate_records:
        for record in frozen_evidence(adjudication).values():
            if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
                authenticate(record, "frozen transaction005 evidence")


def validate_live_namespace(
    *,
    generation6_exists: bool,
    generation6_staging_exists: bool,
    publication_consumption_exists: bool,
    publication_terminal_exists: bool,
    publication_failure_exists: bool,
    later_transaction_indices: list[int],
    authority_count: int,
) -> None:
    require(not generation6_exists, "generation6 already exists")
    require(not generation6_staging_exists, "generation6 staging already exists")
    require(not publication_consumption_exists, "publication authority was consumed")
    require(not publication_terminal_exists, "publication terminal exists")
    require(not publication_failure_exists, "publication failure exists")
    require(not later_transaction_indices, "transaction006-025 artifact exists")
    require(authority_count == 1, "publication authority cardinality differs")


def mutate(
    authority: Mapping[str, Any],
    operation: Callable[[dict[str, Any]], None],
) -> None:
    candidate = copy.deepcopy(dict(authority))
    operation(candidate)
    validate_authority_document(candidate, authenticate_records=False)


def expect_rejection(operation: Callable[[], None], label: str) -> str:
    try:
        operation()
    except ReviewError:
        return label
    raise ReviewError(f"adversarial control was accepted: {label}")


def adversarial_controls(
    authority: Mapping[str, Any],
    recovery_review: Mapping[str, Any],
    publication_source: str,
) -> list[str]:
    rejected_recovery = copy.deepcopy(dict(recovery_review))
    rejected_recovery["status"] = "REJECT"
    controls = [
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item.update({"producer_role": "engineer"}),
            ),
            "manager-provenance",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item[
                    "original_transaction005_authority_consumption"
                ]["consumption"].update({"sha256": "0" * 64}),
            ),
            "original-authority-consumption-binding",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["prohibitions"].remove("vector generation"),
            ),
            "vector-generation-prohibition",
        ),
        expect_rejection(
            lambda: validate_recovery_review(rejected_recovery),
            "rejected-r1-recovery-review",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["frozen_transaction005_evidence"][
                    "raw_terminal"
                ].update({"sha256": "0" * 64}),
            ),
            "altered-frozen-evidence",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["authoritative_parent"].update({"cursor": 4}),
            ),
            "wrong-parent",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["publication_source"].update(
                    {"sha256": "0" * 64}
                ),
            ),
            "altered-publication-source",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["authorized_publish"]["argv"].append("--retry"),
            ),
            "altered-publish-argv",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"authority_consumption": 1}
                ),
            ),
            "authority-consumption",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"generation6_publication": 1}
                ),
            ),
            "generation6-publication",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"transaction_retry": 1}
                ),
            ),
            "transaction005-retry",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"transaction_replay": 1}
                ),
            ),
            "transaction005-replay",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"transaction_resume": 1}
                ),
            ),
            "transaction005-resume",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"model_execution": 1}
                ),
            ),
            "model-execution",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"oracle_execution": 1}
                ),
            ),
            "oracle-execution",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"vector_generation": 1}
                ),
            ),
            "vector-generation",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["activity_counters"].update(
                    {"rtl_simulation": 1}
                ),
            ),
            "rtl-execution",
        ),
        expect_rejection(
            lambda: validate_live_namespace(
                generation6_exists=False,
                generation6_staging_exists=False,
                publication_consumption_exists=False,
                publication_terminal_exists=False,
                publication_failure_exists=False,
                later_transaction_indices=[6],
                authority_count=1,
            ),
            "transaction006-artifact",
        ),
        expect_rejection(
            lambda: source_boundary(publication_source + "\nimport torch\n"),
            "forbidden-computation-import",
        ),
    ]
    return controls


def assess(
    authority_package: Path, output: Path
) -> tuple[str, list[str], str | None]:
    try:
        require(authority_package == AUTHORITY_PACKAGE, "authority path differs")
        require(output == AUTHORITY_REVIEW, "review output path differs")
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
            and seal.get("transactions006_025_absent") is True
            and set(seal.get("members", {})) == AUTHORITY_MEMBERS,
            "authority seal differs",
        )
        for name, record in seal["members"].items():
            path = authority_package / name
            require(record.get("path") == str(path), f"member path differs: {name}")
            authenticate(record, f"authority member {name}")
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"authority member is writable: {name}",
            )
        for path, expected in FIXED_SHA256.items():
            require(sha256_file(path) == expected, f"fixed hash differs: {path}")
        require(
            stat.S_IMODE(RECOVERY_REVIEW.stat().st_mode) & 0o222 == 0,
            "accepted r1 recovery review is writable",
        )
        recovery_review = load_json(RECOVERY_REVIEW)
        validate_recovery_review(recovery_review)
        authority = load_json(authority_package / AUTHORITY_FILENAME)
        validate_authority_document(authority)
        request = load_json(authority_package / "review-request.json")
        require(
            request.get("required_role") == "reviewer"
            and request.get("preparation_participation_required") is False
            and request.get("manager_provenance_required") is True
            and request.get("review_output") == str(output)
            and request.get("authority")
            == file_record(authority_package / AUTHORITY_FILENAME)
            and request.get("publication_source")
            == file_record(authority_package / PUBLICATION_SOURCE_NAME),
            "authority review request differs",
        )
        publication_source = (
            authority_package / PUBLICATION_SOURCE_NAME
        ).read_text(encoding="utf-8")
        reviewer_source = (authority_package / "review-emitter.py").read_text(
            encoding="utf-8"
        )
        source_boundary(publication_source)
        source_boundary(reviewer_source)
        authority_files = list(AUTHORITY_COLLECTION.rglob(AUTHORITY_FILENAME))
        later = [
            index
            for index in range(6, 26)
            if (TRANSACTIONS / f"transaction-{index:03d}").exists()
        ]
        validate_live_namespace(
            generation6_exists=GENERATION6.exists(),
            generation6_staging_exists=GENERATION6_STAGING.exists(),
            publication_consumption_exists=(
                FUTURE / "publication-recovery-authority-consumption.json"
            ).exists(),
            publication_terminal_exists=(
                FUTURE / "publication-recovery-terminal.json"
            ).exists(),
            publication_failure_exists=(
                FUTURE / "publication-recovery-fail-closed-terminal.json"
            ).exists(),
            later_transaction_indices=later,
            authority_count=1,
        )
        require(
            set(authority_files)
            == {
                SUPERSEDED_R1_OUTPUT / AUTHORITY_FILENAME,
                authority_package / AUTHORITY_FILENAME,
            },
            "authority history differs",
        )
        require(
            {path.name for path in FUTURE.iterdir()}
            == {
                "execution-start.json",
                "fail-closed-terminal.json",
                "manager-authorization-consumption.json",
            },
            "frozen transaction005 failure namespace differs",
        )
        controls = adversarial_controls(
            authority, recovery_review, publication_source
        )
        return "PASS", controls, None
    except (
        ReviewError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        return "REJECT", [], str(error)


def write_review(path: Path, payload: bytes) -> None:
    require(path.is_absolute(), "review output must be absolute")
    parent = path.parent
    require(
        parent.is_dir() and not parent.is_symlink(),
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
    require(
        stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
        "authority review output is writable",
    )
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    status, controls, reason = assess(
        arguments.authority_package, arguments.output
    )
    authority = load_json(arguments.authority_package / AUTHORITY_FILENAME)
    review = {
        "schema_version": 1,
        "kind": (
            "ace3_transaction5_publication_recovery_authority_independent_review"
        ),
        "status": status,
        "review_type": "source-disjoint-executable-authority-review",
        "producer_role": "reviewer",
        "preparation_participation": False,
        "manager_provenance_verified": status == "PASS",
        "original_consumption_binding_verified": status == "PASS",
        "vector_generation_prohibited": status == "PASS",
        "authority_seal": file_record(
            arguments.authority_package / "authority-seal.json"
        ),
        "authority": file_record(
            arguments.authority_package / AUTHORITY_FILENAME
        ),
        "publication_source": file_record(
            arguments.authority_package / PUBLICATION_SOURCE_NAME
        ),
        "accepted_r1_recovery_review": file_record(RECOVERY_REVIEW),
        "superseded_candidate_seal": file_record(
            SUPERSEDED_R1_OUTPUT / "authority-seal.json"
        ),
        "authorized_publish_argv": authority.get("authorized_publish", {}).get(
            "argv"
        ),
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "transaction005_retry_replay_resume": False,
        "model_oracle_rtl_execution": False,
        "authority_consumed": False,
        "generation6_publication_performed": False,
        "transactions006_025_absent": True,
        "adversarial_controls": controls,
        "reason": reason,
    }
    write_review(arguments.output, canonical_json(review))
    print(
        "TRANSACTION005_PUBLICATION_RECOVERY_AUTHORITY_REVIEW "
        f"status={status} output={arguments.output} "
        "authority_consumption=0 publication=0 model=0 oracle=0 rtl=0 "
        "retry=0 replay=0 resume=0 generation6=absent "
        "transactions006_025=absent"
    )


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
            f"TRANSACTION005_PUBLICATION_RECOVERY_AUTHORITY_REVIEW_REFUSED {error}"
        ) from error
