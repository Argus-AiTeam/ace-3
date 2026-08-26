#!/usr/bin/env python3
"""Fail-closed validation for deterministic model24 execution vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

from model24_execution_oracle import (
    ContractError,
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    build_vector_artifacts,
    load_json_bytes,
    require,
    require_provenance_commit,
    sha256_bytes,
    validate_decoder_snapshot,
    validate_execution_contract,
    validate_vector_bindings,
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=repository_root / "build" / "model24_execution_prep_cf11" / "vectors",
    )
    parser.add_argument(
        "--official-tokenizer-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_TOKENIZER_DIR,
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=DEFAULT_OFFICIAL_CHECKPOINT,
    )
    return parser.parse_args()


def validate_vector_directory(
    repository_root: Path,
    vector_dir: Path,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
) -> dict[str, int]:
    contracts = repository_root / "ace3" / "contracts"
    contract_path = contracts / "model24_execution.json"
    bindings_path = contracts / "model24_execution_vector_bindings.json"
    contract_payload = contract_path.read_bytes()
    bindings_payload = bindings_path.read_bytes()
    tensor_payload = (contracts / "model24_tensor_map.json").read_bytes()
    control_payload = (contracts / "model24_control.json").read_bytes()

    require_provenance_commit(repository_root)
    validate_decoder_snapshot(repository_root)
    contract = load_json_bytes(contract_payload, str(contract_path))
    bindings = load_json_bytes(bindings_payload, str(bindings_path))
    validate_execution_contract(contract, tensor_payload, control_payload)
    validate_vector_bindings(bindings, sha256_bytes(contract_payload))

    require(vector_dir.is_dir(), f"vector directory is missing: {vector_dir}")
    expected_names = set(bindings["artifact_set"])
    actual_paths = list(vector_dir.iterdir())
    require(all(path.is_file() for path in actual_paths), "vector directory contains a non-file")
    actual_names = {path.name for path in actual_paths}
    require(
        actual_names == expected_names,
        (
            "vector artifact set mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        ),
    )

    actual_payloads: dict[str, bytes] = {}
    for name in sorted(actual_names):
        payload = (vector_dir / name).read_bytes()
        load_json_bytes(payload, name)
        actual_payloads[name] = payload

    expected_payloads = build_vector_artifacts(
        sha256_bytes(contract_payload),
        sha256_bytes(bindings_payload),
        tokenizer_dir,
        checkpoint_path,
    )
    for name in sorted(expected_names):
        require(
            actual_payloads[name] == expected_payloads[name],
            f"{name} differs from independent oracle regeneration",
        )

    manifest = load_json_bytes(actual_payloads["manifest.json"], "manifest.json")
    for name, record in manifest["artifacts"].items():
        require(
            record["sha256"] == sha256_bytes(actual_payloads[name]),
            f"{name} manifest SHA256 mismatch",
        )
        require(record["bytes"] == len(actual_payloads[name]), f"{name} byte count mismatch")
    return manifest["summary"]


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        summary = validate_vector_directory(
            repository_root,
            args.vector_dir.resolve(),
            args.official_tokenizer_dir.resolve(),
            args.official_checkpoint.resolve(),
        )
    except (ContractError, OSError) as error:
        raise SystemExit(f"MODEL24_EXECUTION_VALIDATION_FAIL {error}") from error
    print(
        "MODEL24_EXECUTION_VALIDATION_PASS "
        f"layers={summary['official_layers']} "
        f"events={summary['official_events']} "
        f"tensors={summary['official_tensor_count']} "
        f"small_geometries={summary['small_geometry_cases']} "
        f"host_tokens={summary['host_total_structural_token_steps']} "
        f"layer0_tokens={summary['official_layer0_tokens']} "
        f"layer0_stages={summary['official_layer0_stages']} "
        f"projection_bit_checks={summary['official_layer0_projection_bit_checks']} "
        f"official_logits={summary['official_logits']} "
        f"argmax_token_id={summary['official_argmax_token_id']}"
    )


if __name__ == "__main__":
    main()
