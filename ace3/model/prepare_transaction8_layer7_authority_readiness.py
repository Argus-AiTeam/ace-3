#!/usr/bin/env python3
"""Prepare transaction-8 authority readiness without authorizing workload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import prepare_selected_token_feedback_input_package as feedback
import validate_next_position_traversal_preflight as preflight


TRANSACTION_INDEX = 8
LAYER_INDEX = 7
GENERATION = 8
PARENT_CHECKPOINT_INDEX = 7
TRANSACTION_COUNT = 26
REVIEW_ACTIVITY_KEYS = {
    "authority_consumption",
    "authority_issuance",
    "generation_publication",
    "model_execution",
    "oracle_execution",
    "rtl_compile",
    "rtl_simulation",
    "submission",
    "transaction_execution",
    "transaction_replay",
    "transaction_resume",
    "transaction_retry",
    "vector_generation",
}
FUTURE_EFFECT_FIELDS = (
    "authority_created",
    "authority_consumed",
    "traversal_performed",
    "runtime_state_mutated",
    "transaction_consumed",
    "execution_performed",
    "publication_performed",
)


class AuthorityReadinessError(RuntimeError):
    """Raised when a transaction-8 readiness gate is not authentic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityReadinessError(message)


def _same_content(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    label: str,
) -> None:
    require(
        left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256"),
        f"{label} content binding differs",
    )


def _load_authenticated_json(path: Path, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    resolved = path.resolve()
    record = preflight.file_record(resolved)
    document = preflight.load_json(resolved, label)
    return document, record


def _validate_reviewer_gate(
    review_path: Path,
    receipt_path: Path,
    parent_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    review, review_record = _load_authenticated_json(
        review_path,
        "generation8 post-publication review",
    )
    require(
        review_path.resolve().stat().st_mode & 0o222 == 0,
        "generation8 post-publication review is writable",
    )
    counters = review.get("activity_counters")
    require(
        review.get("schema_version") == 1
        and review.get("kind")
        == "ace3_generation8_post_publication_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("authoritative_generation") == GENERATION
        and review.get("authoritative_cursor") == TRANSACTION_INDEX
        and review.get("authoritative_checkpoint_index")
        == PARENT_CHECKPOINT_INDEX
        and review.get("transaction007_non_replayable") is True
        and review.get("transaction007_retry_replay_resume") is False
        and review.get("transaction008_authority_created") is False
        and review.get("transaction008_authority_consumed") is False
        and review.get("transaction008_runtime_output_present") is False
        and review.get("transactions008_025_absent") is True
        and review.get("this_review_authorizes_execution") is False
        and review.get("host_role_receipt_bound_at_emission") is False
        and review.get("host_role_receipt_required_for_future_authority")
        is True
        and isinstance(counters, dict)
        and REVIEW_ACTIVITY_KEYS.issubset(counters)
        and all(counters[key] == 0 for key in REVIEW_ACTIVITY_KEYS),
        "generation8 Reviewer gate semantics differ",
    )
    for review_key, parent_key in (
        ("authoritative_pointer", "authoritative_pointer"),
        ("generation8_manifest", "generation_manifest"),
        ("generation8_ledger", "ledger"),
        ("checkpoint007", "checkpoint007"),
    ):
        left = review.get(review_key)
        right = parent_artifacts.get(parent_key)
        require(
            isinstance(left, dict) and isinstance(right, dict),
            f"{review_key} Reviewer binding is missing",
        )
        _same_content(left, right, review_key)

    receipt, receipt_record = _load_authenticated_json(
        receipt_path,
        "Reviewer role receipt",
    )
    receipt_review = receipt.get("review")
    require(
        receipt.get("schema_version") == 3
        and receipt.get("kind") == "round_reviewed_handoff"
        and receipt.get("producer_role") == "reviewer"
        and receipt.get("mission_id") == review.get("mission_id")
        and isinstance(receipt_review, dict)
        and receipt_review.get("status") == "continue"
        and isinstance(receipt_review.get("reason"), str)
        and "Reviewer PASS" in receipt_review["reason"],
        "durable Reviewer role receipt differs from the gate",
    )
    return (
        {
            "post_publication_review": review_record,
            "reviewer_role_receipt": receipt_record,
            "mission_id": review["mission_id"],
            "status": "PASS",
        },
        review,
    )


def build_authority_readiness_package(
    feedback_package_path: Path,
    post_publication_review_path: Path,
    reviewer_role_receipt_path: Path,
) -> dict[str, Any]:
    selected, selected_record = feedback.validate_input_package(
        feedback_package_path.resolve()
    )
    source_report = preflight.validate_preflight(
        Path(selected["source_readiness"]["path"])
    )
    parentage = selected["parentage"]
    bounds = selected["next_position_bounds"]
    require(
        parentage.get("cursor_index") == TRANSACTION_INDEX
        and parentage.get("generation") == GENERATION
        and parentage.get("parent_generation") == GENERATION - 1
        and parentage.get("parent_checkpoint") == "checkpoint007"
        and parentage.get("parent_checkpoint_index")
        == PARENT_CHECKPOINT_INDEX
        and bounds.get("transaction_index") == TRANSACTION_INDEX
        and bounds.get("layer_index") == LAYER_INDEX,
        "selected-token package is not bound to transaction008/layer7",
    )
    reviewer_gate, _ = _validate_reviewer_gate(
        post_publication_review_path,
        reviewer_role_receipt_path,
        parentage["required_parent_artifacts"],
    )
    return {
        "schema_version": 1,
        "kind": (
            "ace3_position3_transaction8_layer7_"
            "authority_readiness_package"
        ),
        "status": "AUTHORITY_READY_NOT_CONSUMED",
        "runtime_identity": source_report["runtime_identity"],
        "cursor_identity": {
            "generation": GENERATION,
            "cursor": TRANSACTION_INDEX,
            "parent_generation": GENERATION - 1,
            "parent_checkpoint": "checkpoint007",
            "parent_checkpoint_index": PARENT_CHECKPOINT_INDEX,
        },
        "transaction_identity": {
            **bounds,
            "authority_cardinality": 1,
            "permitted_transaction_indices": [TRANSACTION_INDEX],
            "excluded_transaction_indices": [
                index
                for index in range(TRANSACTION_COUNT)
                if index != TRANSACTION_INDEX
            ],
        },
        "bindings": {
            "selected_token_feedback_package": selected_record,
            "source_readiness": selected["source_readiness"],
            "required_parent_artifacts": parentage[
                "required_parent_artifacts"
            ],
            **reviewer_gate,
        },
        "authority": {
            "scope": "transaction008-layer7-host-validation-only",
            "present": True,
            "consumed": False,
            "workload_authorized": False,
            "traversal_authorized": False,
            "publication_authorized": False,
        },
        "transactions009_025": {
            "transaction_indices": list(range(9, TRANSACTION_COUNT)),
            **{field: False for field in FUTURE_EFFECT_FIELDS},
        },
        "claim_boundary": (
            "Transaction008/layer7 authority-readiness validation only. "
            "No traversal, runtime mutation, authority consumption, workload "
            "execution, generation publication, or transaction009-025 "
            "authority or effect is authorized or performed."
        ),
    }


def validate_authority_readiness_package(
    package_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = package_path.resolve()
    package = preflight.load_json(
        resolved,
        "transaction008 authority-readiness package",
    )
    require(
        package.get("schema_version") == 1
        and package.get("kind")
        == "ace3_position3_transaction8_layer7_authority_readiness_package"
        and package.get("status") == "AUTHORITY_READY_NOT_CONSUMED"
        and resolved.stat().st_mode & 0o222 == 0,
        "transaction008 authority-readiness package identity differs",
    )
    bindings = package.get("bindings")
    require(isinstance(bindings, dict), "authority-readiness bindings are missing")
    selected_record = preflight.authenticate_record(
        bindings.get("selected_token_feedback_package"),
        "selected-token feedback package",
    )
    review_record = preflight.authenticate_record(
        bindings.get("post_publication_review"),
        "post-publication review",
    )
    receipt_record = preflight.authenticate_record(
        bindings.get("reviewer_role_receipt"),
        "Reviewer role receipt",
    )
    expected = build_authority_readiness_package(
        Path(selected_record["path"]),
        Path(review_record["path"]),
        Path(receipt_record["path"]),
    )
    payload = resolved.read_bytes()
    require(
        package == expected and payload == preflight.canonical_json(expected),
        "transaction008 authority-readiness package differs from its bindings",
    )
    return package, preflight.file_record(resolved)


def emit_authority_readiness_package(
    feedback_package_path: Path,
    post_publication_review_path: Path,
    reviewer_role_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    package = build_authority_readiness_package(
        feedback_package_path,
        post_publication_review_path,
        reviewer_role_receipt_path,
    )
    resolved_output = output_path.resolve()
    preflight.write_new(resolved_output, preflight.canonical_json(package))
    os.chmod(resolved_output, 0o400)
    return package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-token-feedback-package",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--post-publication-review",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reviewer-role-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        package = emit_authority_readiness_package(
            arguments.selected_token_feedback_package,
            arguments.post_publication_review,
            arguments.reviewer_role_receipt,
            arguments.output,
        )
    except (
        AuthorityReadinessError,
        feedback.InputPackageError,
        feedback.TokenizerContractError,
        preflight.PreflightError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        print(
            f"TRANSACTION8_AUTHORITY_READINESS_FAILED {error}",
            file=sys.stderr,
        )
        return 1
    print(
        "TRANSACTION8_AUTHORITY_READINESS_READY "
        f"generation={package['cursor_identity']['generation']} "
        f"cursor={package['cursor_identity']['cursor']} "
        f"transaction={package['transaction_identity']['transaction_index']} "
        f"layer={package['transaction_identity']['layer_index']} "
        "consumed=0 workload=0 publication=0 transaction009_025=0 "
        f"artifact={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
