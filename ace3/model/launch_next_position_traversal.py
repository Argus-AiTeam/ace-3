#!/usr/bin/env python3
"""Fail-closed launcher for one authority-bound next-position transaction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import validate_next_position_traversal_preflight as preflight


TRANSACTION_COUNT = 26
AUTHORITY_KIND = re.compile(
    r"^ace3_position[0-9]+_transaction[0-9]+_layer[0-9]+_launch_authority$"
)
ZERO_EFFECT_FIELDS = (
    "authority_package_created",
    "authority_consumed",
    "consumption_artifact_created",
    "traversal_performed",
    "traversal_artifact_created",
    "runtime_state_mutated",
    "transaction_consumed",
    "execution_artifact_created",
    "publication_performed",
    "publication_artifact_created",
)
FUTURE_TRANSACTION_EFFECT_FIELDS = (
    "authority_created",
    "authority_consumed",
    "traversal_performed",
    "runtime_state_mutated",
    "transaction_consumed",
    "execution_performed",
    "publication_performed",
)


class AuthorityError(RuntimeError):
    """Raised before any runtime effect when authority is absent or invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def load_authority(path: Path | None) -> tuple[Path, dict[str, Any]]:
    if path is None:
        raise AuthorityError("authority package was not provided")
    resolved = path.resolve()
    if not resolved.is_file():
        raise AuthorityError(f"authority package is missing: {resolved}")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError(
            f"authority package is not valid JSON: {resolved}: {error}"
        ) from error
    require(isinstance(document, dict), "authority package root is not an object")
    return resolved, document


def validate_authority(
    path: Path | None,
    readiness_path: Path,
    preflight_report: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    authority_path, authority = load_authority(path)
    transaction_index = preflight_report["transaction_index"]
    layer_index = preflight_report["layer_index"]
    transaction_position = preflight_report["transaction_position"]
    generation = preflight_report["generation"]
    expected_kind = (
        f"ace3_position{transaction_position}_transaction{transaction_index}"
        f"_layer{layer_index}_launch_authority"
    )
    require(
        authority.get("schema_version") == 1
        and isinstance(authority.get("kind"), str)
        and AUTHORITY_KIND.fullmatch(authority["kind"]) is not None
        and authority["kind"] == expected_kind,
        "authority schema or kind differs from the pending transaction",
    )
    require(
        authority_path.stat().st_mode & 0o222 == 0,
        "authority package is writable",
    )
    require(
        authority.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and authority.get("authority_consumed") is False
        and authority.get("authority_cardinality") == 1,
        "authority is not an unconsumed cardinality-one grant",
    )
    require(
        authority.get("runtime_identity") == preflight_report["runtime_identity"]
        and authority.get("authoritative_generation") == generation
        and authority.get("transaction_index") == transaction_index
        and authority.get("layer_index") == layer_index
        and authority.get("target_checkpoint_index") == transaction_index,
        "authority runtime, generation, transaction, or layer binding differs",
    )
    require(
        authority.get("permitted_transaction_indices") == [transaction_index]
        and authority.get("excluded_transaction_indices")
        == [
            index
            for index in range(TRANSACTION_COUNT)
            if index != transaction_index
        ],
        "authority does not permit exactly the pending transaction",
    )
    absence_key = f"transactions{transaction_index:03d}_025_absent"
    require(
        authority.get(absence_key) is True,
        f"authority does not bind initial {absence_key}",
    )
    readiness_record = preflight.file_record(readiness_path.resolve())
    bindings = authority.get("bindings")
    require(
        isinstance(bindings, dict)
        and bindings.get("runtime_readiness") == readiness_record,
        "authority does not bind the exact runtime-readiness input",
    )
    identity = authority.get("transaction_identity")
    require(
        isinstance(identity, dict)
        and identity.get("transaction_index") == transaction_index
        and identity.get("layer_index") == layer_index
        and identity.get("operation") == "position3-decoder-layer"
        and identity.get("input_binding_sha256")
        == preflight_report["input_binding_sha256"]
        and identity.get("inputs", {}).get("transaction_position")
        == transaction_position
        and identity.get("required_outputs", {}).get("state", {}).get("position")
        == preflight_report["output_state_position"],
        "authority transaction identity differs from authenticated inputs",
    )
    launch = authority.get("authorized_launch")
    require(
        isinstance(launch, dict)
        and launch.get("transaction_index") == transaction_index
        and launch.get("layer_index") == layer_index
        and launch.get("start_cursor") == transaction_index
        and launch.get("start_generation") == generation
        and launch.get("exit_generation") == generation + 1,
        "authorized launch bounds differ from the pending transaction",
    )
    argv = launch.get("argv")
    cwd = launch.get("cwd")
    require(
        isinstance(argv, list)
        and argv
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(cwd, str)
        and Path(cwd).is_absolute(),
        "authorized launch command is malformed",
    )
    require(
        argv.count("--transaction-index") == 1
        and argv.index("--transaction-index") + 1 < len(argv)
        and argv[argv.index("--transaction-index") + 1]
        == str(transaction_index),
        "authorized launch command is not transaction bounded",
    )
    consumption = authority.get("authority_consumption")
    require(
        isinstance(consumption, dict)
        and isinstance(consumption.get("path"), str)
        and Path(consumption["path"]).is_absolute(),
        "authority consumption path is missing or not absolute",
    )
    consumption_path = Path(consumption["path"])
    require(
        consumption_path.parent.is_dir(),
        "authority consumption parent directory is absent",
    )
    require(
        not consumption_path.exists(),
        "authority was already consumed",
    )
    return authority_path, authority


def refusal_report(
    report: Mapping[str, Any],
    reason: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_next_position_traversal_launch_refusal",
        "status": "REFUSED",
        "reason_code": reason_code,
        "reason": reason,
        "preflight": dict(report),
        "authority_package_created": False,
        "authority_consumed": False,
        "consumption_artifact_created": False,
        "traversal_performed": False,
        "traversal_artifact_created": False,
        "runtime_state_mutated": False,
        "transaction_consumed": False,
        "execution_artifact_created": False,
        "publication_performed": False,
        "publication_artifact_created": False,
    }


def consume_authority(
    authority_path: Path,
    authority: Mapping[str, Any],
    readiness_path: Path,
    report: Mapping[str, Any],
) -> Path:
    consumption_path = Path(authority["authority_consumption"]["path"])
    consumption = {
        "schema_version": 1,
        "kind": "ace3_next_position_traversal_authority_consumption",
        "status": "CONSUMED_BEFORE_TRAVERSAL",
        "authority": preflight.file_record(authority_path),
        "runtime_readiness": preflight.file_record(readiness_path),
        "runtime_identity": report["runtime_identity"],
        "generation": report["generation"],
        "transaction_index": report["transaction_index"],
        "layer_index": report["layer_index"],
    }
    preflight.write_new(consumption_path, preflight.canonical_json(consumption))
    return consumption_path


def launch(
    readiness_path: Path,
    authority_package: Path | None,
) -> tuple[dict[str, Any], int]:
    resolved_readiness = readiness_path.resolve()
    report = preflight.validate_preflight(resolved_readiness)
    report["input_binding_sha256"] = json.loads(
        resolved_readiness.read_text(encoding="utf-8")
    )["next_layer_inputs"]["template_input_binding_sha256"]
    try:
        authority_path, authority = validate_authority(
            authority_package,
            resolved_readiness,
            report,
        )
    except AuthorityError as error:
        code = (
            "MISSING_AUTHORITY_PACKAGE"
            if authority_package is None
            or not authority_package.resolve().is_file()
            else "INVALID_AUTHORITY_PACKAGE"
        )
        return refusal_report(report, str(error), code), 2

    consumption_path = consume_authority(
        authority_path,
        authority,
        resolved_readiness,
        report,
    )
    command = authority["authorized_launch"]
    completed = subprocess.run(
        command["argv"],
        cwd=command["cwd"],
        check=False,
    )
    return {
        "schema_version": 1,
        "kind": "ace3_next_position_traversal_launch_result",
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "preflight": report,
        "authority_consumption": preflight.file_record(consumption_path),
        "authority_consumed": True,
        "traversal_performed": True,
        "transaction_index": report["transaction_index"],
        "returncode": completed.returncode,
    }, completed.returncode


def dry_run_readiness(readiness_path: Path) -> tuple[dict[str, Any], int]:
    resolved_readiness = readiness_path.resolve()
    refusal, returncode = launch(resolved_readiness, None)
    require(
        returncode == 2
        and refusal.get("status") == "REFUSED"
        and refusal.get("reason_code") == "MISSING_AUTHORITY_PACKAGE",
        "dry-run authority guard did not refuse missing authority",
    )
    report = refusal["preflight"]
    readiness = preflight.load_json(resolved_readiness, "runtime readiness")
    cursor = readiness["cursor"]
    parent_checkpoint_index = report["parent_checkpoint_index"]
    parent_checkpoint_name = f"checkpoint{parent_checkpoint_index:03d}"
    parent_generation = cursor["parent_generation"]

    return {
        "schema_version": 1,
        "kind": "ace3_next_position_runtime_readiness_contract",
        "status": "NOT_READY",
        "runtime_identity": report["runtime_identity"],
        "cursor_identity": {
            "cursor_index": report["cursor"],
            "generation": report["generation"],
            "parent_generation": parent_generation,
            "parent_checkpoint": parent_checkpoint_name,
            "parent_checkpoint_index": parent_checkpoint_index,
        },
        "required_parent_artifacts": {
            "authoritative_pointer": cursor["authoritative_pointer"],
            "generation_manifest": cursor["generation_manifest"],
            "ledger": cursor["ledger"],
            parent_checkpoint_name: cursor[parent_checkpoint_name],
        },
        "authenticated_input_hashes": {
            "input_binding_sha256": report["input_binding_sha256"],
            "parent_hidden_semantic_sha256": (
                report["parent_hidden_semantic_sha256"]
            ),
            "selected_token_feedback_semantic_sha256": (
                report["selected_token_feedback_semantic_sha256"]
            ),
        },
        "next_position_bounds": {
            "transaction_index": report["transaction_index"],
            "layer_index": report["layer_index"],
            "transaction_position": report["transaction_position"],
            "output_state_position": report["output_state_position"],
            "transaction_count_bound": report["transaction_count_bound"],
            "model_layer_count_bound": report["model_layer_count_bound"],
        },
        "launch_verdict": {
            "status": refusal["status"],
            "launch_authorized": False,
            "reason_code": refusal["reason_code"],
            "reason": refusal["reason"],
        },
        "zero_effects": {
            field: refusal[field]
            for field in ZERO_EFFECT_FIELDS
        },
    }, returncode


def dry_run_feedback_package(
    package_path: Path,
) -> tuple[dict[str, Any], int]:
    import prepare_selected_token_feedback_input_package as feedback_package

    try:
        package, package_record = feedback_package.validate_input_package(
            package_path
        )
    except (
        feedback_package.InputPackageError,
        feedback_package.TokenizerContractError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise preflight.PreflightError(
            f"selected-token feedback package validation failed: {error}"
        ) from error

    report, returncode = dry_run_readiness(
        Path(package["source_readiness"]["path"])
    )
    parentage = package["parentage"]
    parent_checkpoint = parentage["parent_checkpoint"]
    preflight.require(
        report["cursor_identity"] == {
            key: parentage[key]
            for key in (
                "cursor_index",
                "generation",
                "parent_generation",
                "parent_checkpoint",
                "parent_checkpoint_index",
            )
        }
        and report["required_parent_artifacts"]
        == parentage["required_parent_artifacts"]
        and report["next_position_bounds"] == package["next_position_bounds"],
        "selected-token feedback package dry-run parentage differs",
    )
    report["selected_token_feedback_package_validation"] = {
        "status": "PASS",
        "kind": package["kind"],
        "package": package_record,
        "package_sha256_validated": True,
        "source_readiness": package["source_readiness"],
        "token_id": package["selected_token"]["token_id"],
        "decoded_text_fragment": package["selected_token"][
            "decoded_text_fragment"
        ],
        "embedding_semantic_sha256": package["tied_embedding"][
            "selected_row_semantic_sha256"
        ],
        "parent_checkpoint_identity": {
            "name": parent_checkpoint,
            "index": parentage["parent_checkpoint_index"],
            "artifact": parentage["required_parent_artifacts"][
                parent_checkpoint
            ],
        },
        "next_position_bounds": package["next_position_bounds"],
    }
    return report, returncode


def dry_run_authority_readiness(
    package_path: Path,
) -> tuple[dict[str, Any], int]:
    import prepare_transaction8_layer7_authority_readiness as authority_readiness

    package, package_record = (
        authority_readiness.validate_authority_readiness_package(package_path)
    )
    transaction = package["transaction_identity"]
    cursor = package["cursor_identity"]
    return {
        "schema_version": 1,
        "kind": (
            "ace3_transaction8_layer7_authority_readiness_validation"
        ),
        "status": "PASS",
        "runtime_identity": package["runtime_identity"],
        "authority_readiness": {
            "status": package["status"],
            "package": package_record,
            "scope": package["authority"]["scope"],
            "authority_present": True,
            "authority_consumed": False,
            "missing_authority": False,
            "workload_authorized": False,
        },
        "cursor_identity": cursor,
        "transaction_identity": transaction,
        "reviewer_gate": {
            "status": package["bindings"]["status"],
            "mission_id": package["bindings"]["mission_id"],
            "post_publication_review": package["bindings"][
                "post_publication_review"
            ],
            "reviewer_role_receipt": package["bindings"][
                "reviewer_role_receipt"
            ],
        },
        "selected_token_feedback_package": package["bindings"][
            "selected_token_feedback_package"
        ],
        "launch_verdict": {
            "status": "AUTHORITY_VALIDATED",
            "reason_code": "EXECUTION_NOT_REQUESTED",
            "launch_authorized": False,
            "authority_missing": False,
        },
        "zero_effects": {
            field: False
            for field in ZERO_EFFECT_FIELDS
        },
        "transactions009_025": package["transactions009_025"],
    }, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--selected-token-feedback-package", type=Path)
    parser.add_argument("--authority-package", type=Path)
    parser.add_argument("--authority-readiness-package", type=Path)
    parser.add_argument("--dry-run-readiness", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.dry_run_readiness and arguments.authority_package is not None:
        parser.error("--dry-run-readiness does not accept --authority-package")
    if (
        arguments.selected_token_feedback_package is not None
        and not arguments.dry_run_readiness
    ):
        parser.error(
            "--selected-token-feedback-package requires "
            "--dry-run-readiness"
        )
    if (
        arguments.selected_token_feedback_package is not None
        and arguments.readiness is not None
    ):
        parser.error(
            "--selected-token-feedback-package derives its source readiness"
        )
    if arguments.authority_readiness_package is not None:
        if not arguments.dry_run_readiness:
            parser.error(
                "--authority-readiness-package requires --dry-run-readiness"
            )
        if (
            arguments.readiness is not None
            or arguments.selected_token_feedback_package is not None
            or arguments.authority_package is not None
        ):
            parser.error(
                "--authority-readiness-package is a complete validation input"
            )
    if (
        arguments.authority_readiness_package is None
        and arguments.selected_token_feedback_package is None
        and arguments.readiness is None
    ):
        parser.error(
            "--readiness or --selected-token-feedback-package is required"
        )
    try:
        if arguments.authority_readiness_package is not None:
            report, returncode = dry_run_authority_readiness(
                arguments.authority_readiness_package
            )
        elif arguments.selected_token_feedback_package is not None:
            report, returncode = dry_run_feedback_package(
                arguments.selected_token_feedback_package
            )
        elif arguments.dry_run_readiness:
            report, returncode = dry_run_readiness(arguments.readiness)
        else:
            report, returncode = launch(
                arguments.readiness,
                arguments.authority_package,
            )
        payload = preflight.canonical_json(report)
        if arguments.output is not None:
            preflight.write_new(arguments.output.resolve(), payload)
        sys.stdout.buffer.write(payload)
        return returncode
    except (OSError, preflight.PreflightError) as error:
        print(f"NEXT_POSITION_LAUNCH_PREFLIGHT_FAILED {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
