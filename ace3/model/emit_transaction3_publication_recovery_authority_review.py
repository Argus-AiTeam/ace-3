#!/usr/bin/env python3
"""Independently review a transaction-003 publication-recovery authority."""

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
GENERATION3 = GENERATIONS / "generation-0000000003"
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION4_STAGING = GENERATIONS / ".generation-0000000004.prepared"
POINTER = ADOPTION / "authoritative-state.json"
TRANSACTIONS = RUNTIME / "transactions"
PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery"
    / "r5-postconsume-natural-terminal-mapping-r7"
)
RECOVERY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery_reviews"
    / "r5-postconsume-natural-terminal-mapping-r7"
    / "independent-review.json"
)
AUTHORITY_COLLECTION = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery_authorities"
)
AUTHORITY_PACKAGE = AUTHORITY_COLLECTION / "r5-postconsume-natural-terminal-mapping-r7"
AUTHORITY_FILENAME = "manager-publication-recovery-authority.json"
AUTHORITY_MEMBERS = {
    AUTHORITY_FILENAME,
    "review-emitter.py",
    "review-request.json",
}
EXPECTED_SHA256 = {
    POINTER: "9489297fb3230c5b281c746c86a4bde6fba92cbb8af1d339f81178591c3de13d",
    GENERATION3 / "ledger.json": "f4c44a7717d392dd8e328d4430a62d233c23865efb0bd59f48718d0f31af0af2",
    GENERATION3 / "checkpoints/transaction-002.json": "31330e3e421ced0289415a2e864ca428535c01a97f2c7e157fa2bfeda82a7552",
    PACKAGE / "adjudication.json": "850026965fa2c55b902756e6e59b5a4140cd3785997c34f7e618da4a4e894b7e",
    PACKAGE / "package-manifest.json": "87c6f0aaf03316834e01b209de073bd610fda482ce3e6ff7028f4991f7ca212b",
    PACKAGE / "package-seal.json": "dcb02706a3daaf510771d7cccc5bc782fa83c8c4aaee8ff6876dca0840183587",
    PACKAGE / "publication_recovery.py": "89f6ecbda9cf10dc972974a706136a69a38db12bb091ba8c3aea0ff837a058a6",
    RECOVERY_REVIEW: "cf6f92873167909c8799d68920d83b7cf55edfefd598f3b658cbbe908ea46baf",
}
EXPECTED_BOUND_SHA256 = {
    "consumed_original_r5.manager_authorization": "05105bf6f7f04252f04f13c5b5de584d2112558df6c7b17af8f2971922d2e6d0",
    "consumed_original_r5.authority_consumption": "803a89340967094fd4e2875635ab27e94a3314cab31c9795d7a948d617b66a89",
    "execution_identity.execution_start": "24be70a905665174391b63653b1a0c7521c850bb78d502e4553a2973a09ef320",
    "execution_identity.durable_runner_status": "6c1c866bc65c945b081649d99b1dd87a01d73b0223286a9e570c78442d39e816",
    "sealed_failure_terminal": "7c0245d0aafe3a2065bf01659c5a81b4ee3b40da03bccc3776c5ebaf15ad98ab",
    "frozen_evidence.raw_rtl_terminal": "d94b645e69d075ca952d0bc13af43c699a72158700592c94028897478c4f807c",
    "frozen_evidence.raw_rtl_trace": "d90d1382eec19f296571fe639cc16312364c55e31a24296cfedd2601ca260f84",
    "frozen_evidence.raw_rtl_final": "07457cc78c7064b0c9802233e6f2c7a3e18e634b1f66acdb84f4001f084a7a9c",
    "frozen_evidence.oracle_trace": "d90d1382eec19f296571fe639cc16312364c55e31a24296cfedd2601ca260f84",
    "frozen_evidence.oracle_final": "07457cc78c7064b0c9802233e6f2c7a3e18e634b1f66acdb84f4001f084a7a9c",
    "frozen_evidence.comparison": "23c5792963b9e4bf58abe4ec948982608a4b7bbc80f80ff77fcd7af36d70ead2",
    "frozen_evidence.output_state": "f179ad258de8f8aedb060592334d8f6d732dd559e93878a2c7daa7a84fcb567c",
}
EXPECTED_HIDDEN_SHA256 = (
    "95e7911ce6afbc8dba048e3910b22489288fedfdeee33b47021622d937b67463"
)
PROHIBITIONS = [
    "model generation",
    "oracle generation",
    "RTL compilation",
    "RTL simulation",
    "transaction003 replay",
    "new transaction identity",
    "transaction004-025 execution",
]
FORBIDDEN_IMPORTS = {"importlib", "numpy", "safetensors", "subprocess", "torch"}
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
REVIEWER_FORBIDDEN_CALLS = FORBIDDEN_CALLS | {
    "flock",
    "mkdir",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "unlink",
}


class AuthorityReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityReviewError(message)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate_record(record: Mapping[str, Any], label: str) -> None:
    require(
        isinstance(record.get("path"), str)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str),
        f"{label} record is malformed",
    )
    require(file_record(Path(record["path"])) == dict(record), f"{label} differs")


def zero_activity_counters() -> dict[str, int]:
    return {
        "authority_consumption": 0,
        "model_generation": 0,
        "new_transaction_identity": 0,
        "oracle_generation": 0,
        "publication": 0,
        "rtl_compilation": 0,
        "rtl_simulation": 0,
        "transaction003_replay": 0,
        "transaction004_025_execution": 0,
    }


def recovery_argv() -> list[str]:
    return [
        "/usr/bin/python3",
        str(PACKAGE / "publication_recovery.py"),
        "publish",
        "--package",
        str(PACKAGE),
        "--review",
        str(RECOVERY_REVIEW),
    ]


def nested(document: Mapping[str, Any], dotted: str) -> Mapping[str, Any]:
    value: Any = document
    for part in dotted.split("."):
        require(isinstance(value, Mapping), f"record path is absent: {dotted}")
        value = value.get(part)
    require(isinstance(value, Mapping), f"record is absent: {dotted}")
    return value


def source_boundary(payload: str, reviewer_only: bool = False) -> None:
    tree = ast.parse(payload)
    forbidden_calls = REVIEWER_FORBIDDEN_CALLS if reviewer_only else FORBIDDEN_CALLS
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(
                    alias.name.split(".", 1)[0] not in FORBIDDEN_IMPORTS
                    for alias in node.names
                ),
                "source imports model, oracle, RTL, or process support",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                (node.module or "").split(".", 1)[0] not in FORBIDDEN_IMPORTS,
                "source imports model, oracle, RTL, or process support",
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            require(name.lower() not in forbidden_calls, f"forbidden call: {name}")


def validate_recovery_review(document: Mapping[str, Any]) -> None:
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_transaction3_publication_recovery_independent_review"
        and document.get("status") == "PASS"
        and document.get("producer_role") == "reviewer"
        and document.get("preparation_participation") is False
        and document.get("publication_performed") is False
        and document.get("recovery_authority_created") is False
        and document.get("adjudication")
        == file_record(PACKAGE / "adjudication.json")
        and document.get("package_seal")
        == file_record(PACKAGE / "package-seal.json"),
        "accepted r7 Reviewer artifact differs",
    )


def validate_readonly_mode(mode: int) -> None:
    require(mode & 0o222 == 0, "accepted r7 review is writable")


def validate_authority_document(
    authority: Mapping[str, Any], authenticate: bool = True
) -> None:
    require(
        authority.get("schema_version") == 1
        and authority.get("kind")
        == "ace3_transaction3_publication_only_recovery_manager_authority"
        and authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("producer_role") == "manager"
        and authority.get("scope")
        == "publication-only frozen-evidence receipt recovery"
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("transaction_index") == 3
        and authority.get("new_transaction_identity") is None,
        "Manager authority identity differs",
    )
    require(
        authority.get("authorized_argv") == recovery_argv(),
        "authorized recovery argv differs",
    )
    require(authority.get("prohibitions") == PROHIBITIONS, "prohibitions differ")
    require(
        authority.get("activity_counters") == zero_activity_counters()
        and authority.get("authority_cardinality") == 1
        and authority.get("authority_consumed") is False
        and authority.get("publication_performed") is False,
        "authority lifecycle or activity counters differ",
    )
    parent = authority.get("authoritative_parent", {})
    require(
        parent.get("generation") == 3
        and parent.get("cursor") == 3
        and parent.get("latest_checkpoint_index") == 2
        and parent.get("pointer") == file_record(POINTER)
        and parent.get("ledger") == file_record(GENERATION3 / "ledger.json")
        and parent.get("checkpoint002")
        == file_record(GENERATION3 / "checkpoints/transaction-002.json"),
        "generation3/cursor3/checkpoint002 parentage differs",
    )
    accepted = authority.get("accepted_recovery", {})
    require(
        accepted.get("package_root") == str(PACKAGE)
        and accepted.get("adjudication")
        == file_record(PACKAGE / "adjudication.json")
        and accepted.get("package_manifest")
        == file_record(PACKAGE / "package-manifest.json")
        and accepted.get("package_seal")
        == file_record(PACKAGE / "package-seal.json")
        and accepted.get("publication_recovery")
        == file_record(PACKAGE / "publication_recovery.py")
        and accepted.get("independent_reviewer_artifact")
        == file_record(RECOVERY_REVIEW),
        "accepted recovery package or Reviewer binding differs",
    )
    consumed = authority.get("consumed_original_r5", {})
    require(
        consumed.get("replay_authorized") is False,
        "consumed r5 replay boundary differs",
    )
    execution = authority.get("execution_identity", {})
    require(
        execution.get("runtime_identity") == RUNTIME_IDENTITY
        and execution.get("transaction_index") == 3,
        "execution identity differs",
    )
    require(
        authority.get("frozen_evidence", {}).get("hidden_semantic_sha256")
        == EXPECTED_HIDDEN_SHA256,
        "frozen hidden semantic hash differs",
    )
    for dotted, expected_sha256 in EXPECTED_BOUND_SHA256.items():
        record = nested(authority, dotted)
        require(
            record.get("sha256") == expected_sha256,
            f"frozen binding hash differs: {dotted}",
        )
        if authenticate:
            authenticate_record(record, dotted)
    reviewer = authority.get("reviewer_acceptance", {})
    require(
        reviewer.get("required") is True
        and reviewer.get("present_at_issuance") is False
        and isinstance(reviewer.get("path"), str),
        "Reviewer acceptance boundary differs",
    )


def validate_live_namespace(
    *,
    generation4_exists: bool,
    staging_exists: bool,
    later_transactions: list[int],
    authority_count: int,
) -> None:
    require(not generation4_exists, "generation4 already exists")
    require(not staging_exists, "generation4 staging already exists")
    require(not later_transactions, "transaction004-025 namespace exists")
    require(authority_count == 1, "Manager authority cardinality differs")


def mutate(
    authority: Mapping[str, Any],
    operation: Callable[[dict[str, Any]], None],
) -> None:
    candidate = copy.deepcopy(authority)
    operation(candidate)
    validate_authority_document(candidate, authenticate=False)


def expect_rejection(operation: Callable[[], None], label: str) -> str:
    try:
        operation()
    except AuthorityReviewError:
        return label
    raise AuthorityReviewError(f"adversarial control was accepted: {label}")


def adversarial_controls(
    authority: Mapping[str, Any],
    recovery_review: Mapping[str, Any],
    publication_source: str,
) -> list[str]:
    rejected_review = dict(recovery_review)
    rejected_review["status"] = "REJECT"
    return [
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["frozen_evidence"]["raw_rtl_final"].update(
                    {"sha256": "0" * 64}
                ),
            ),
            "altered-frozen-evidence",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["authoritative_parent"].update(
                    {"latest_checkpoint_index": 3}
                ),
            ),
            "wrong-parentage",
        ),
        expect_rejection(
            lambda: validate_recovery_review(rejected_review),
            "rejected-r7-review",
        ),
        expect_rejection(
            lambda: validate_readonly_mode(0o644),
            "writable-r7-review",
        ),
        expect_rejection(
            lambda: validate_live_namespace(
                generation4_exists=False,
                staging_exists=False,
                later_transactions=[],
                authority_count=2,
            ),
            "competing-authority",
        ),
        expect_rejection(
            lambda: mutate(
                authority,
                lambda item: item["authorized_argv"].append("--replay"),
            ),
            "replay-capable-argv",
        ),
        expect_rejection(
            lambda: source_boundary(publication_source + "\nimport torch\n"),
            "forbidden-computation-import",
        ),
        expect_rejection(
            lambda: source_boundary(publication_source + "\nos.system('false')\n"),
            "forbidden-computation-call",
        ),
        expect_rejection(
            lambda: validate_live_namespace(
                generation4_exists=True,
                staging_exists=False,
                later_transactions=[],
                authority_count=1,
            ),
            "pre-existing-generation4",
        ),
        expect_rejection(
            lambda: validate_live_namespace(
                generation4_exists=False,
                staging_exists=False,
                later_transactions=[4],
                authority_count=1,
            ),
            "later-transaction-namespace",
        ),
    ]


def assess(
    authority_package: Path, output: Path
) -> tuple[str, list[str], str]:
    try:
        require(
            authority_package == AUTHORITY_PACKAGE,
            "authority package path differs",
        )
        require(output.is_absolute(), "review output must be absolute")
        require(not output.exists(), "authority review output already exists")
        metadata = authority_package.lstat()
        require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) & 0o222 == 0,
            "authority package is absent or writable",
        )
        actual_files = {
            path.name for path in authority_package.iterdir() if path.is_file()
        }
        require(
            actual_files == AUTHORITY_MEMBERS | {"authority-seal.json"},
            "authority package file set differs",
        )
        seal = load_json(authority_package / "authority-seal.json")
        require(
            seal.get("kind")
            == "ace3_transaction3_publication_recovery_authority_seal"
            and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
            and seal.get("authority_cardinality") == 1
            and seal.get("activity_counters") == zero_activity_counters()
            and seal.get("authority_consumed") is False
            and seal.get("publication_performed") is False,
            "authority seal differs",
        )
        members = seal.get("members")
        require(
            isinstance(members, dict) and set(members) == AUTHORITY_MEMBERS,
            "authority seal members differ",
        )
        for name, record in members.items():
            path = authority_package / name
            require(record.get("path") == str(path), f"member path differs: {name}")
            authenticate_record(record, f"authority member {name}")
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"authority member is writable: {name}",
            )
        for path, expected_sha256 in EXPECTED_SHA256.items():
            require(
                sha256_file(path) == expected_sha256,
                f"fixed evidence hash differs: {path}",
            )
        recovery_review = load_json(RECOVERY_REVIEW)
        require(
            stat.S_ISREG(RECOVERY_REVIEW.stat().st_mode),
            "accepted r7 review is not a regular file",
        )
        validate_readonly_mode(stat.S_IMODE(RECOVERY_REVIEW.stat().st_mode))
        validate_recovery_review(recovery_review)
        authority = load_json(authority_package / AUTHORITY_FILENAME)
        validate_authority_document(authority)
        request = load_json(authority_package / "review-request.json")
        require(
            request.get("required_role") == "reviewer"
            and request.get("manager_participation_required") is False
            and request.get("review_output") == str(output)
            and request.get("authority")
            == file_record(authority_package / AUTHORITY_FILENAME),
            "authority review request differs",
        )
        authority_files = list(AUTHORITY_COLLECTION.rglob(AUTHORITY_FILENAME))
        later = [
            index
            for index in range(4, 26)
            if (TRANSACTIONS / f"transaction-{index:03d}").exists()
        ]
        validate_live_namespace(
            generation4_exists=GENERATION4.exists(),
            staging_exists=GENERATION4_STAGING.exists(),
            later_transactions=later,
            authority_count=len(authority_files),
        )
        require(
            authority_files == [authority_package / AUTHORITY_FILENAME],
            "competing Manager authority path exists",
        )
        publication_source = (PACKAGE / "publication_recovery.py").read_text(
            encoding="utf-8"
        )
        emitter_source = (authority_package / "review-emitter.py").read_text(
            encoding="utf-8"
        )
        source_boundary(publication_source)
        source_boundary(emitter_source, reviewer_only=True)
        controls = adversarial_controls(
            authority, recovery_review, publication_source
        )
        return "PASS", controls, "all independent authority checks passed"
    except (AuthorityReviewError, FileNotFoundError) as error:
        return "REJECT", [], str(error)


def write_review(path: Path, payload: bytes) -> None:
    require(path.is_absolute(), "review output must be absolute")
    parent = path.parent
    metadata = parent.lstat()
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
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
            "ace3_transaction3_publication_recovery_authority_independent_review"
        ),
        "status": status,
        "producer_role": "reviewer",
        "manager_participation": False,
        "authority_status": authority.get("status"),
        "authority_seal": file_record(
            arguments.authority_package / "authority-seal.json"
        ),
        "manager_authority": file_record(
            arguments.authority_package / AUTHORITY_FILENAME
        ),
        "accepted_recovery_review": file_record(RECOVERY_REVIEW),
        "activity_counters": zero_activity_counters(),
        "authority_consumed": False,
        "publication_performed": False,
        "adversarial_controls": controls,
        "reason": reason,
    }
    write_review(arguments.output, canonical_json(review))
    print(
        "TRANSACTION3_PUBLICATION_RECOVERY_AUTHORITY_REVIEW "
        f"status={status} output={arguments.output} "
        "authority=AUTHORIZED_NOT_CONSUMED publication=0 consumption=0 "
        "model=0 oracle=0 rtl_compile=0 rtl_simulation=0 replay=0"
    )


if __name__ == "__main__":
    main()
