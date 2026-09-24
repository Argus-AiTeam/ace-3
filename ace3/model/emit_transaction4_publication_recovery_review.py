#!/usr/bin/env python3
"""Source-disjoint review of the transaction004 recovery adjudication."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
RUNTIME = (
    ROOT
    / "build/model24_selected_token_position3_runs"
    / "ace3-position3-fresh-r11-20260831t215500z"
)
TRANSACTION4 = RUNTIME / "transactions/transaction-004"
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATION5 = ADOPTION / "state-generations/generation-0000000005"
GENERATION5_STAGING = ADOPTION / "state-generations/.generation-0000000005.prepared"
FUTURE = RUNTIME / "transaction4-authoritative-generation5"
TRANSACTION4_PACKAGE_MANIFEST = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_packages"
    / "generation4-cursor4-transaction004-layer03-r1"
    / "package-manifest.json"
)
PRODUCTION_CONTROL = (
    ROOT
    / "build/model24_selected_token_position3_continuations"
    / "v10-accepted-review-gated-r10-readable-authority-schema"
    / "continuation_control.py"
)
EXPECTED_PRODUCTION_CONTROL = {
    "path": str(PRODUCTION_CONTROL),
    "bytes": 151006,
    "sha256": "001f26990ee7628106600100c2de91e7305723c38efe31a3c81b2fe044e09db6",
}
EXPECTED_HIDDEN_SHA256 = (
    "152347b94d879e16b8e5d514ac7a0b38c954336f138bd3f5e27b57965d350cfa"
)
PACKAGE_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "recovery-adjudication.py",
    "review-emitter.py",
    "review-request.json",
}
ZERO_ACTIVITY = {
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
    require(file_record(Path(record["path"])) == dict(record), f"{label} differs")


def source_boundary(source: str) -> None:
    tree = ast.parse(source)
    imported = {
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
        {
            "fcntl",
            "numpy",
            "safetensors",
            "shutil",
            "subprocess",
            "torch",
        }.isdisjoint(imported),
        "package source imports execution support",
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
        {
            "execute_exact_layer_transaction",
            "execute_layer_transaction",
            "publish_generation5",
            "run",
            "system",
        }.isdisjoint(calls),
        "package source contains execution or publication call",
    )


def terminal_fields(payload: bytes) -> None:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReviewError("RTL terminal is not ASCII") from error
    require(len(lines) == 1, "RTL terminal line count differs")
    fields: dict[str, str] = {}
    for item in lines[0].split():
        require(item.count("=") == 1, "RTL terminal field malformed")
        name, value = item.split("=", 1)
        require(name not in fields, f"duplicate terminal field: {name}")
        fields[name] = value
    require(
        fields
        == {
            "schema": "ace3_decoder_token_transaction_v1",
            "layer_index": "3",
            "position": "3",
            "natural_terminal": "1",
            "exit_code": "0",
            "trace_count": "23408",
            "final_count": "896",
            "done_count": "1",
        },
        "RTL terminal facts differ",
    )


def validate_with_production_completion_validator(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        file_record(PRODUCTION_CONTROL) == EXPECTED_PRODUCTION_CONTROL,
        "production completion validator source binding differs",
    )
    transaction = load_json(TRANSACTION4_PACKAGE_MANIFEST)[
        "transaction_descriptor"
    ]
    spec = importlib.util.spec_from_file_location(
        "ace3_transaction4_review_production_control",
        PRODUCTION_CONTROL,
    )
    require(
        spec is not None and spec.loader is not None,
        "production completion validator could not be loaded",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        validated = module.validate_completion_receipt(transaction, receipt)
    except module.ControlError as error:
        raise ReviewError(
            f"production completion validator rejected reconstructed receipt: {error}"
        ) from error
    require(
        isinstance(validated, dict)
        and validated.get("result") == receipt.get("result"),
        "production completion validator returned an unexpected receipt",
    )
    return {
        "status": "PASS",
        "validator": copy.deepcopy(EXPECTED_PRODUCTION_CONTROL),
        "transaction_descriptor": file_record(TRANSACTION4_PACKAGE_MANIFEST),
        "validated_result": copy.deepcopy(validated["result"]),
    }


def validate_adjudication(document: Mapping[str, Any], authenticate_all: bool) -> None:
    require(
        document.get("kind")
        == "ace3_transaction4_postconsume_publication_recovery_adjudication"
        and document.get("status") == "PASS"
        and document.get("transaction_index") == 4
        and document.get("layer_index") == 3,
        "adjudication identity differs",
    )
    finding = document.get("finding", {})
    require(
        finding.get("natural_terminal") == 1
        and finding.get("rtl_exit_code") == 0
        and finding.get("done_count") == 1
        and finding.get("final_count") == 896
        and finding.get("integer_mismatches") == 0
        and finding.get("exact_oracle_agreement") is True
        and finding.get("sole_blocking_defect")
        == "completion receipt natural_rtl_terminal None-to-False mapping"
        and finding.get("generation5_publication_attempted") is False,
        "adjudicated finding differs",
    )
    boundary = document.get("recovery_boundary", {})
    require(
        boundary.get("activity_counters") == ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation5_publication_authorized") is False
        and boundary.get("generation5_publication_performed") is False
        and boundary.get("separately_reviewed_successor_required") is True
        and boundary.get("transactions005_025_absent") is True,
        "recovery boundary differs",
    )
    receipt = document.get("reconstructed_receipt", {})
    recovery = receipt.get("publication_recovery", {})
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 4
        and receipt.get("result", {}).get("natural_rtl_terminal") is True
        and receipt.get("result", {}).get("exact_integer_oracle_match") is True
        and receipt.get("output_semantics", {}).get("hidden", {}).get(
            "semantic_sha256"
        )
        == EXPECTED_HIDDEN_SHA256
        and receipt.get("timing", {}).get("transaction_seconds") is None
        and recovery.get("transaction_retry") is False
        and recovery.get("transaction_replay") is False
        and recovery.get("transaction_resume") is False,
        "reconstructed receipt boundary differs",
    )
    source = document.get("source_boundary", {})
    require(
        source.get("terminal_parser_authenticates")
        == {"natural_terminal": 1, "exit_code": 0}
        and source.get("terminal_parser_returns")
        == ["trace_count", "final_count", "done_count"]
        and source.get("returned_rtl_record_top_level_natural_terminal") is None
        and source.get("returned_rtl_record_raw_natural_terminal") is None
        and source.get("receipt_adapter_observed_value_type") == "NoneType"
        and source.get("receipt_adapter_observed_value") is None
        and source.get("failing_expression") == "natural_terminal is True"
        and source.get("failing_expression_result") is False
        and source.get("corrected_reconstructed_receipt_value") is True,
        "source mapping finding differs",
    )
    production_validation = document.get("production_completion_validation", {})
    require(
        production_validation.get("status") == "PASS"
        and production_validation.get("validator")
        == EXPECTED_PRODUCTION_CONTROL
        and production_validation.get("transaction_descriptor")
        == file_record(TRANSACTION4_PACKAGE_MANIFEST)
        and production_validation.get("validated_result")
        == receipt.get("result"),
        "production completion validation proof differs",
    )
    if not authenticate_all:
        return
    require(
        not GENERATION5.exists()
        and not GENERATION5_STAGING.exists()
        and all(
            not (RUNTIME / f"transactions/transaction-{index:03d}").exists()
            for index in range(5, 26)
        ),
        "generation5 or transaction005-025 exists",
    )
    require(
        FUTURE.is_dir()
        and {path.name for path in FUTURE.iterdir()}
        == {
            "execution-start.json",
            "fail-closed-terminal.json",
            "manager-authorization-consumption.json",
        },
        "failure namespace differs",
    )
    inventory = document.get("frozen_transaction004", {}).get("inventory")
    require(isinstance(inventory, list), "transaction004 inventory absent")
    actual_paths = {
        path for path in TRANSACTION4.rglob("*") if path.is_file() or path.is_symlink()
    }
    recorded_paths = {Path(record["path"]) for record in inventory}
    require(actual_paths == recorded_paths, "partial transaction004 inventory")
    for record in inventory:
        authenticate(
            {key: record[key] for key in ("path", "bytes", "sha256")},
            "transaction004 inventory",
        )
    terminal_fields(
        (TRANSACTION4 / "position003/raw/terminal.txt").read_bytes()
    )
    comparison = load_json(TRANSACTION4 / "comparison.json")
    require(
        comparison
        == {
            "exact_integer_oracle_output_sha256": EXPECTED_HIDDEN_SHA256,
            "implementation": "independent ACE-3 integer W4A16 oracle",
            "integer_mismatches": 0,
            "inter_layer_boundary": "binary16 round after every RTL layer",
            "rtl_matches_exact_integer_oracle": True,
        },
        "comparison facts differ",
    )
    require(
        (TRANSACTION4 / "position003/raw/trace.hex").read_bytes()
        == (TRANSACTION4 / "position003/exact_oracle/trace.hex").read_bytes()
        and (TRANSACTION4 / "position003/raw/final.hex").read_bytes()
        == (TRANSACTION4 / "position003/exact_oracle/final.hex").read_bytes(),
        "raw RTL output differs from exact oracle",
    )
    for section in (
        "authoritative_parent",
        "execution_boundary",
        "source_boundary",
        "computation_evidence",
    ):
        for value in document[section].values():
            if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
                authenticate(value, f"adjudication {section}")
    require(
        document.get("production_completion_validation")
        == validate_with_production_completion_validator(receipt),
        "production completion validation proof differs",
    )


def mutate(
    document: Mapping[str, Any], operation: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(document))
    operation(candidate)
    return candidate


def adversarial_controls(document: Mapping[str, Any], source: str) -> list[str]:
    controls = []
    mutations = (
        (
            "altered-terminal",
            lambda item: item["finding"].update({"natural_terminal": 0}),
        ),
        (
            "altered-comparison",
            lambda item: item["finding"].update({"integer_mismatches": 1}),
        ),
        (
            "altered-mapping",
            lambda item: item["source_boundary"].update(
                {"receipt_adapter_observed_value": 1}
            ),
        ),
        (
            "altered-production-validation",
            lambda item: item["production_completion_validation"].update(
                {"status": "REJECT"}
            ),
        ),
        (
            "retry",
            lambda item: item["recovery_boundary"]["activity_counters"].update(
                {"transaction_retry": 1}
            ),
        ),
        (
            "replay",
            lambda item: item["reconstructed_receipt"][
                "publication_recovery"
            ].update({"transaction_replay": True}),
        ),
        (
            "resume",
            lambda item: item["reconstructed_receipt"][
                "publication_recovery"
            ].update({"transaction_resume": True}),
        ),
        (
            "authority-consumption",
            lambda item: item["recovery_boundary"]["activity_counters"].update(
                {"authority_consumption": 1}
            ),
        ),
        (
            "generation5-publication",
            lambda item: item["recovery_boundary"].update(
                {"generation5_publication_authorized": True}
            ),
        ),
        (
            "transaction005-execution",
            lambda item: item["recovery_boundary"].update(
                {"transactions005_025_absent": False}
            ),
        ),
    )
    for name, operation in mutations:
        try:
            validate_adjudication(mutate(document, operation), False)
        except ReviewError:
            controls.append(name)
        else:
            raise ReviewError(f"adversarial mutation accepted: {name}")
    try:
        source_boundary(source + "\nimport subprocess\n")
    except ReviewError:
        controls.append("execution-import")
    else:
        raise ReviewError("execution import accepted")
    return controls


def assess_package(
    package: Path, output: Path
) -> tuple[str, list[str], str | None]:
    try:
        require(package.is_absolute(), "package path must be absolute")
        require(output.is_absolute(), "review output must be absolute")
        require(
            package.is_dir()
            and not package.is_symlink()
            and stat.S_IMODE(package.stat().st_mode) & 0o222 == 0,
            "sealed package directory is absent or writable",
        )
        seal = load_json(package / "package-seal.json")
        require(
            seal.get("kind")
            == "ace3_transaction4_publication_recovery_adjudication_seal"
            and seal.get("status") == "SEALED_REVIEW_REQUIRED"
            and seal.get("activity_counters") == ZERO_ACTIVITY
            and seal.get("generation5_publication_authorized") is False
            and seal.get("generation5_publication_performed") is False
            and set(seal.get("members", {})) == PACKAGE_MEMBERS,
            "package seal differs",
        )
        actual = {path.name for path in package.iterdir() if path.is_file()}
        require(actual == PACKAGE_MEMBERS | {"package-seal.json"}, "file set differs")
        for name, record in seal["members"].items():
            require(record["path"] == str(package / name), f"path differs: {name}")
            authenticate(record, f"sealed member {name}")
        manifest = load_json(package / "package-manifest.json")
        require(
            manifest.get("review_path") == str(output)
            and manifest.get("activity_counters") == ZERO_ACTIVITY
            and manifest.get("publication_entrypoint") is None
            and manifest.get("recovery_authority") is None
            and manifest.get("generation5_publication_authorized") is False
            and manifest.get("separately_reviewed_successor_required") is True,
            "package execution/publication boundary differs",
        )
        source = (package / "recovery-adjudication.py").read_text(encoding="utf-8")
        emitter = (package / "review-emitter.py").read_text(encoding="utf-8")
        source_boundary(source)
        source_boundary(emitter)
        adjudication = load_json(package / "adjudication.json")
        validate_adjudication(adjudication, True)
        controls = adversarial_controls(adjudication, source)
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
    status, controls, reason = assess_package(package, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adjudication = load_json(package / "adjudication.json")
    source = adjudication.get("source_boundary", {})
    production_validation = adjudication.get(
        "production_completion_validation", {}
    )
    review = {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_recovery_independent_review",
        "status": status,
        "review_type": "source-disjoint-executable-evidence-review",
        "institutional_role_authority_claimed": False,
        "preparation_participation": False,
        "package_seal": file_record(package / "package-seal.json"),
        "adjudication": file_record(package / "adjudication.json"),
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "transaction004_retry_replay_resume": False,
        "transactions005_025_absent": True,
        "new_authority_consumed": False,
        "generation5_publication_authorized": False,
        "generation5_publication_performed": False,
        "separately_reviewed_successor_required": True,
        "receipt_adapter_observed_value": source.get(
            "receipt_adapter_observed_value"
        ),
        "receipt_adapter_mapping_result": source.get(
            "failing_expression_result"
        ),
        "production_completion_validator": production_validation.get(
            "validator"
        ),
        "production_completion_validator_result": (
            production_validation.get("status")
        ),
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
        "TRANSACTION004_RECOVERY_INDEPENDENT_REVIEW_WRITTEN "
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
        raise SystemExit(f"TRANSACTION004_RECOVERY_REVIEW_REFUSED {error}") from error
