#!/usr/bin/env python3
"""Prepare a verified selected-token input package without launch authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open

sys.dont_write_bytecode = True

from model24_execution_oracle import (
    ContractError as TokenizerContractError,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    OFFICIAL_CONFIG,
    TIED_WEIGHT_SHA256,
)
import validate_next_position_traversal_preflight as preflight


EMBEDDING_TENSOR = "model.embed_tokens.weight"
HIDDEN_ELEMENTS = 896


class InputPackageError(RuntimeError):
    """Raised when selected-token input package preparation fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputPackageError(message)


def _authenticated_record(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} record is missing")
    return preflight.authenticate_record(value, label)


def _serialized_embedding_bits(path: Path) -> np.ndarray:
    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise InputPackageError(
            f"selected-token embedding is not readable ASCII: {path}"
        ) from error
    require(
        len(rows) == HIDDEN_ELEMENTS,
        "selected-token embedding element count differs",
    )
    values: list[int] = []
    for index, row in enumerate(rows):
        require(
            len(row) == 4
            and all(character in "0123456789abcdef" for character in row),
            f"selected-token embedding row {index} is malformed",
        )
        values.append(int(row, 16))
    return np.asarray(values, dtype="<u2")


def build_input_package(readiness_path: Path) -> dict[str, Any]:
    import launch_next_position_traversal as launcher

    resolved_readiness = readiness_path.resolve()
    readiness = preflight.load_json(resolved_readiness, "runtime readiness")
    preflight_report = preflight.validate_preflight(resolved_readiness)
    dry_run, returncode = launcher.dry_run_readiness(resolved_readiness)
    require(
        returncode == 2
        and dry_run.get("status") == "NOT_READY"
        and dry_run.get("launch_verdict", {}).get("reason_code")
        == "MISSING_AUTHORITY_PACKAGE",
        "dry-run readiness did not remain fail closed without authority",
    )
    require(
        all(value is False for value in dry_run.get("zero_effects", {}).values())
        and set(dry_run.get("zero_effects", {})) == set(launcher.ZERO_EFFECT_FIELDS),
        "dry-run readiness did not report the complete zero-effect boundary",
    )

    selected = readiness.get("selected_token")
    feedback = readiness.get("embedding_feedback")
    require(
        isinstance(selected, dict) and isinstance(feedback, dict),
        "selected-token feedback inputs are missing",
    )
    token_id = selected.get("token_id")
    require(
        type(token_id) is int
        and token_id == preflight_report.get("selected_token_id")
        and 0 <= token_id < OFFICIAL_CONFIG["vocab_size"],
        "selected token ID differs from the authenticated preflight",
    )

    tokenizer_record = _authenticated_record(
        selected.get("official_tokenizer"),
        "official tokenizer",
    )
    tokenizer_config_record = _authenticated_record(
        selected.get("official_tokenizer_config"),
        "official tokenizer config",
    )
    tokenizer_path = Path(tokenizer_record["path"])
    tokenizer_config_path = Path(tokenizer_config_record["path"])
    require(
        tokenizer_path.name == "tokenizer.json"
        and tokenizer_config_path.name == "tokenizer_config.json"
        and tokenizer_path.parent == tokenizer_config_path.parent
        and tokenizer_record["sha256"] == TOKENIZER_SHA256
        and tokenizer_config_record["sha256"] == TOKENIZER_CONFIG_SHA256,
        "official tokenizer artifact binding differs",
    )
    tokenizer = authenticate_tokenizer(tokenizer_path.parent)
    decoded_fragment = tokenizer.decode([token_id], skip_special_tokens=False)
    require(
        decoded_fragment
        and tokenizer.encode(
            decoded_fragment,
            add_special_tokens=False,
        ).ids
        == [token_id]
        and selected.get("tokenizer_piece") == decoded_fragment,
        "official tokenizer decode or token round-trip differs",
    )

    semantics = feedback.get("semantics")
    require(
        isinstance(semantics, dict)
        and semantics.get("token_id") == token_id
        and semantics.get("tensor") == EMBEDDING_TENSOR
        and semantics.get("dtype") == "FP16"
        and semantics.get("elements") == HIDDEN_ELEMENTS
        and semantics.get("semantic_sha256")
        == preflight_report.get("selected_token_feedback_semantic_sha256"),
        "selected-token embedding semantics differ from preflight",
    )
    checkpoint_record = semantics.get("checkpoint")
    require(
        isinstance(checkpoint_record, dict)
        and checkpoint_record.get("bytes") == CHECKPOINT_SIZE
        and checkpoint_record.get("sha256") == CHECKPOINT_SHA256,
        "official checkpoint record differs",
    )
    authenticated_checkpoint = _authenticated_record(
        checkpoint_record,
        "official AWQ checkpoint",
    )

    embedding_record = _authenticated_record(
        feedback.get("vector"),
        "selected-token embedding vector",
    )
    serialized_bits = _serialized_embedding_bits(Path(embedding_record["path"]))
    checkpoint_path = Path(authenticated_checkpoint["path"])
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embedding = checkpoint.get_slice(EMBEDDING_TENSOR)
        embedding_shape = list(embedding.get_shape())
        row = np.asarray(
            embedding[token_id : token_id + 1],
            dtype="<f2",
        ).reshape(-1)
    require(
        embedding_shape
        == [OFFICIAL_CONFIG["vocab_size"], OFFICIAL_CONFIG["hidden_size"]]
        and list(row.shape) == [HIDDEN_ELEMENTS],
        "official tied embedding geometry differs",
    )
    checkpoint_bits = row.view("<u2")
    semantic_sha256 = preflight.sha256_bytes(checkpoint_bits.tobytes())
    require(
        np.array_equal(checkpoint_bits, serialized_bits)
        and semantic_sha256 == semantics["semantic_sha256"],
        "checkpoint embedding row differs from selected-token feedback",
    )

    source_receipt = _authenticated_record(
        selected.get("lm_head_receipt"),
        "selected-token lm_head receipt",
    )
    source_readiness = preflight.file_record(resolved_readiness)
    return {
        "schema_version": 1,
        "kind": "ace3_selected_token_feedback_input_package",
        "status": "VERIFIED_INPUT_ONLY",
        "source_readiness": source_readiness,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": authenticated_checkpoint,
        },
        "selected_token": {
            "token_id": token_id,
            "decoded_text_fragment": decoded_fragment,
            "selected_logit_f16_bits": selected.get(
                "selected_logit_f16_bits"
            ),
            "lm_head_receipt": source_receipt,
            "official_tokenizer": tokenizer_record,
            "official_tokenizer_config": tokenizer_config_record,
        },
        "tied_embedding": {
            "tensor": EMBEDDING_TENSOR,
            "tied_peer": "lm_head.weight",
            "tied_weight_sha256": TIED_WEIGHT_SHA256,
            "dtype": "FP16",
            "tensor_shape": embedding_shape,
            "selected_row_shape": [HIDDEN_ELEMENTS],
            "selected_row_semantic_sha256": semantic_sha256,
            "serialized_vector": embedding_record,
        },
        "parentage": {
            **dry_run["cursor_identity"],
            "required_parent_artifacts": dry_run[
                "required_parent_artifacts"
            ],
        },
        "next_position_bounds": dry_run["next_position_bounds"],
        "dry_run_readiness_contract": {
            "kind": dry_run["kind"],
            "status": dry_run["status"],
            "authenticated_input_hashes": dry_run[
                "authenticated_input_hashes"
            ],
            "launch_verdict": dry_run["launch_verdict"],
            "zero_effects": dry_run["zero_effects"],
        },
        "authority_package_present": False,
        "launch_authorized": False,
        "claim_boundary": (
            "Verified selected-token decode and tied-embedding input only. "
            "No authority, traversal, runtime mutation, consumption, execution, "
            "generation publication, synthesis, PPA, FPGA, latency, or dialogue "
            "claim is created."
        ),
    }


def validate_input_package(
    package_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_package = package_path.resolve()
    package = preflight.load_json(
        resolved_package,
        "selected-token feedback input package",
    )
    require(
        package.get("schema_version") == 1
        and package.get("kind")
        == "ace3_selected_token_feedback_input_package"
        and package.get("status") == "VERIFIED_INPUT_ONLY"
        and package.get("authority_package_present") is False
        and package.get("launch_authorized") is False,
        "selected-token feedback input package identity differs",
    )
    source_readiness = _authenticated_record(
        package.get("source_readiness"),
        "source runtime readiness",
    )
    expected = build_input_package(Path(source_readiness["path"]))
    expected_payload = preflight.canonical_json(expected)
    try:
        payload = resolved_package.read_bytes()
    except OSError as error:
        raise InputPackageError(
            f"selected-token feedback input package is unreadable: "
            f"{resolved_package}"
        ) from error
    require(
        payload == expected_payload and package == expected,
        "selected-token feedback input package differs from rebuilt evidence",
    )
    package_record = preflight.file_record(resolved_package)
    require(
        package_record["sha256"] == preflight.sha256_bytes(expected_payload),
        "selected-token feedback input package SHA256 differs",
    )
    return package, package_record


def emit_input_package(
    readiness_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    package = build_input_package(readiness_path)
    preflight.write_new(output_path.resolve(), preflight.canonical_json(package))
    return package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        package = emit_input_package(arguments.readiness, arguments.output)
    except (
        InputPackageError,
        TokenizerContractError,
        preflight.PreflightError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        print(f"SELECTED_TOKEN_INPUT_PACKAGE_FAILED {error}", file=sys.stderr)
        return 1
    print(
        "SELECTED_TOKEN_INPUT_PACKAGE_VERIFIED "
        f"token={package['selected_token']['token_id']} "
        f"decode={json.dumps(package['selected_token']['decoded_text_fragment'])} "
        f"embedding={package['tied_embedding']['selected_row_semantic_sha256']} "
        f"parent={package['parentage']['parent_checkpoint']} "
        "launch=MISSING_AUTHORITY_PACKAGE effects=0 "
        f"artifact={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
