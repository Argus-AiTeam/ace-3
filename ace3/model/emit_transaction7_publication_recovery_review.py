#!/usr/bin/env python3
"""Independently review transaction007's publication-recovery package."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path("/home/argustest/ace3-argus")
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATIONS = ADOPTION / "state-generations"
POINTER = ADOPTION / "authoritative-state.json"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION7 = TRANSACTIONS / "transaction-007"
FUTURE = RUNTIME / "transaction7-authoritative-generation8"
GENERATION8 = GENERATIONS / "generation-0000000008"
GENERATION8_STAGING = GENERATIONS / ".generation-0000000008.prepared"
ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation8_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "synthesis": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions008_025_execution": 0,
    "u280": 0,
}
PACKAGE_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "recovery-adjudication.py",
    "review-emitter.py",
    "review-request.json",
}
FORBIDDEN_IMPORTS = {"numpy", "safetensors", "subprocess", "torch"}
FORBIDDEN_CALLS = {
    "execute_exact_layer_transaction",
    "execute_layer_transaction",
    "popen",
    "run",
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


def tree_summary(root: Path) -> dict[str, Any]:
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


def source_boundary(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    require(
        imports.isdisjoint(FORBIDDEN_IMPORTS),
        f"computation import in recovery package: {path}",
    )
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    require(
        {name.lower() for name in calls}.isdisjoint(FORBIDDEN_CALLS),
        f"execution call in recovery package: {path}",
    )


def validate_live_evidence(document: Mapping[str, Any]) -> None:
    require(not GENERATION8.exists(), "generation8 already exists")
    require(not GENERATION8_STAGING.exists(), "generation8 staging exists")
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(8, 26)
        ),
        "transaction008-025 artifact exists",
    )
    pointer = load_json(POINTER)
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 7
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY,
        "generation7 pointer differs",
    )
    require(
        FUTURE.is_dir()
        and {path.name for path in FUTURE.iterdir()}
        == {
            "execution-start.json",
            "fail-closed-terminal.json",
            "manager-authorization-consumption.json",
        },
        "transaction007 fail-closed namespace differs",
    )
    finding = document.get("finding", {})
    evidence = document.get("computation_evidence", {})
    counts = evidence.get("terminal_counts", {})
    require(
        finding.get("natural_terminal") == 1
        and finding.get("rtl_exit_code") == 0
        and finding.get("done_count") == 1
        and finding.get("final_count") == 896
        and finding.get("integer_mismatches") == 0
        and finding.get("exact_oracle_agreement") is True
        and counts
        == {
            "natural_terminal": 1,
            "exit_code": 0,
            "trace_count": 23408,
            "final_count": 896,
            "done_count": 1,
        },
        "natural RTL completion evidence differs",
    )
    for label in (
        "terminal",
        "simulation_log",
        "comparison",
        "raw_trace",
        "oracle_trace",
        "raw_final",
        "oracle_final",
        "output_state",
    ):
        authenticate(evidence[label], f"computation evidence {label}")
    require(
        Path(evidence["raw_trace"]["path"]).read_bytes()
        == Path(evidence["oracle_trace"]["path"]).read_bytes()
        and Path(evidence["raw_final"]["path"]).read_bytes()
        == Path(evidence["oracle_final"]["path"]).read_bytes(),
        "RTL and independent integer-oracle outputs differ",
    )
    comparison = load_json(Path(evidence["comparison"]["path"]))
    require(
        comparison.get("integer_mismatches") == 0
        and comparison.get("rtl_matches_exact_integer_oracle") is True
        and comparison.get("exact_integer_oracle_output_sha256")
        == evidence.get("hidden_semantic_sha256"),
        "integer-oracle comparison differs",
    )
    receipt = document.get("reconstructed_receipt", {})
    recovery = receipt.get("publication_recovery", {})
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 7
        and receipt.get("result", {}).get("natural_rtl_terminal") is True
        and receipt.get("result", {}).get("exact_integer_oracle_match") is True
        and receipt.get("result", {}).get("output_hidden_elements") == 896
        and receipt.get("timing", {}).get("transaction_seconds") is None
        and recovery.get("model_rerun") is False
        and recovery.get("oracle_rerun") is False
        and recovery.get("rtl_rerun") is False
        and recovery.get("transaction_retry") is False
        and recovery.get("transaction_replay") is False
        and recovery.get("transaction_resume") is False,
        "reconstructed receipt differs",
    )
    boundary = document.get("recovery_boundary", {})
    require(
        boundary.get("activity_counters") == ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation8_publication_authorized") is False
        and boundary.get("generation8_publication_performed") is False
        and boundary.get("separately_reviewed_publication_authority_required")
        is True
        and boundary.get("transactions008_025_absent") is True,
        "publication-only boundary differs",
    )
    for index in range(3, 8):
        require(
            document["preserved_history"]["generations"][str(index)]
            == tree_summary(GENERATIONS / f"generation-{index:010d}"),
            f"generation{index} preservation differs",
        )
    for index in range(7):
        require(
            document["preserved_history"]["transactions"][str(index)]
            == tree_summary(TRANSACTIONS / f"transaction-{index:03d}"),
            f"transaction{index:03d} preservation differs",
        )
    require(
        document["frozen_transaction007"]["tree"] == tree_summary(TRANSACTION7),
        "transaction007 frozen tree differs",
    )
    for record in document["frozen_transaction007"]["inventory"]:
        authenticate(
            {key: record[key] for key in ("path", "bytes", "sha256")},
            "transaction007 inventory",
        )


def mutated(document: Mapping[str, Any], *path: str, value: object) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(document))
    target: dict[str, Any] = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return candidate


def expect_rejection(document: Mapping[str, Any], label: str) -> str:
    try:
        validate_live_evidence(document)
    except ReviewError:
        return label
    raise ReviewError(f"adversarial mutation accepted: {label}")


def assess(package: Path, output: Path) -> tuple[str, list[str], str | None]:
    try:
        require(package.is_absolute(), "package path must be absolute")
        require(output.is_absolute(), "review output must be absolute")
        require(not output.exists(), "review output already exists")
        require(
            package.is_dir()
            and not package.is_symlink()
            and stat.S_IMODE(package.stat().st_mode) & 0o222 == 0,
            "sealed package is absent or writable",
        )
        seal = load_json(package / "package-seal.json")
        require(
            seal.get("kind")
            == "ace3_transaction7_publication_recovery_adjudication_seal"
            and seal.get("status") == "SEALED_REVIEW_REQUIRED"
            and seal.get("activity_counters") == ZERO_ACTIVITY
            and seal.get("generation8_publication_authorized") is False
            and seal.get("generation8_publication_performed") is False
            and set(seal.get("members", {})) == PACKAGE_MEMBERS,
            "recovery seal differs",
        )
        require(
            {path.name for path in package.iterdir() if path.is_file()}
            == PACKAGE_MEMBERS | {"package-seal.json"},
            "recovery package file set differs",
        )
        for name, record in seal["members"].items():
            require(record.get("path") == str(package / name), f"path differs: {name}")
            authenticate(record, f"sealed member {name}")
        manifest = load_json(package / "package-manifest.json")
        require(
            manifest.get("kind") == "ace3_transaction7_publication_recovery_package"
            and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
            and manifest.get("review_path") == str(output)
            and manifest.get("publication_entrypoint") is None
            and manifest.get("publication_authority") is None
            and manifest.get("activity_counters") == ZERO_ACTIVITY
            and manifest.get("generation8_publication_authorized") is False
            and manifest.get("generation8_publication_performed") is False
            and manifest.get("separately_reviewed_publication_authority_required")
            is True,
            "recovery package boundary differs",
        )
        source_boundary(package / "recovery-adjudication.py")
        source_boundary(package / "review-emitter.py")
        document = load_json(package / "adjudication.json")
        require(
            document.get("kind")
            == "ace3_transaction7_postconsume_publication_recovery_adjudication"
            and document.get("status") == "PASS",
            "recovery adjudication identity differs",
        )
        validate_live_evidence(document)
        controls = [
            expect_rejection(
                mutated(document, "finding", "natural_terminal", value=0),
                "altered-terminal",
            ),
            expect_rejection(
                mutated(document, "finding", "integer_mismatches", value=1),
                "altered-comparison",
            ),
            expect_rejection(
                mutated(
                    document,
                    "reconstructed_receipt",
                    "result",
                    "natural_rtl_terminal",
                    value=False,
                ),
                "altered-receipt",
            ),
            expect_rejection(
                mutated(
                    document,
                    "recovery_boundary",
                    "generation8_publication_authorized",
                    value=True,
                ),
                "premature-publication-authority",
            ),
            expect_rejection(
                mutated(
                    document,
                    "reconstructed_receipt",
                    "publication_recovery",
                    "transaction_retry",
                    value=True,
                ),
                "transaction-retry",
            ),
        ]
        return "PASS", controls, None
    except (ReviewError, OSError, ValueError, KeyError, TypeError) as error:
        return "REJECT", [], str(error)


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def emit_review(package: Path, output: Path) -> None:
    require(not output.exists(), f"review output already exists: {output}")
    status, controls, reason = assess(package, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    review = {
        "schema_version": 1,
        "kind": "ace3_transaction7_publication_recovery_independent_review",
        "status": status,
        "producer_role": "reviewer",
        "review_type": "source-disjoint-executable-evidence-review",
        "preparation_participation": False,
        "package_seal": file_record(package / "package-seal.json"),
        "adjudication": file_record(package / "adjudication.json"),
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "natural_rtl_terminal": True if status == "PASS" else None,
        "final_output_count": 896 if status == "PASS" else None,
        "integer_oracle_mismatches": 0 if status == "PASS" else None,
        "transaction007_retry_replay_resume": False,
        "generations3_7_preserved": status == "PASS",
        "transactions000_006_preserved": status == "PASS",
        "transactions008_025_absent": status == "PASS",
        "generation8_publication_authorized": False,
        "generation8_publication_performed": False,
        "separately_reviewed_publication_authority_required": True,
        "synthesis_or_u280": False,
        "adversarial_controls": controls,
        "reason": reason,
    }
    write_new(output, canonical_json(review))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    emit_review(arguments.package, arguments.output)
    print(
        "TRANSACTION007_RECOVERY_INDEPENDENT_REVIEW_WRITTEN "
        f"output={arguments.output} execution=0 publication=0"
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
        raise SystemExit(f"TRANSACTION007_RECOVERY_REVIEW_REFUSED {error}") from error
