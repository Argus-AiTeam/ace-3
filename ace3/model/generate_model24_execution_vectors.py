#!/usr/bin/env python3
"""Generate deterministic decoder-independent model24 execution vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

from model24_execution_oracle import (
    ContractError,
    build_vector_artifacts,
    load_json_bytes,
    require,
    require_parent_commit,
    sha256_bytes,
    validate_execution_contract,
    validate_vector_bindings,
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "build" / "model24_execution_prep_cf11" / "vectors",
    )
    return parser.parse_args()


def generate(repository_root: Path, output_dir: Path) -> dict[str, bytes]:
    contracts = repository_root / "ace3" / "contracts"
    contract_path = contracts / "model24_execution.json"
    bindings_path = contracts / "model24_execution_vector_bindings.json"
    tensor_path = contracts / "model24_tensor_map.json"
    control_path = contracts / "model24_control.json"

    require_parent_commit(repository_root)
    contract_payload = contract_path.read_bytes()
    bindings_payload = bindings_path.read_bytes()
    tensor_payload = tensor_path.read_bytes()
    control_payload = control_path.read_bytes()
    contract = load_json_bytes(contract_payload, str(contract_path))
    bindings = load_json_bytes(bindings_payload, str(bindings_path))
    validate_execution_contract(contract, tensor_payload, control_payload)
    validate_vector_bindings(bindings, sha256_bytes(contract_payload))

    expected_names = set(bindings["artifact_set"])
    artifacts = build_vector_artifacts(
        sha256_bytes(contract_payload),
        sha256_bytes(bindings_payload),
    )
    require(set(artifacts) == expected_names, "generator artifact set mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {path.name for path in output_dir.iterdir()}
    require(
        existing_names <= expected_names,
        f"output directory contains unexpected artifacts: {sorted(existing_names - expected_names)}",
    )
    for name, payload in artifacts.items():
        (output_dir / name).write_bytes(payload)
    return artifacts


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        artifacts = generate(repository_root, args.output_dir.resolve())
    except (ContractError, OSError) as error:
        raise SystemExit(f"MODEL24_EXECUTION_GENERATION_FAIL {error}") from error
    print(
        "MODEL24_EXECUTION_GENERATION_PASS "
        f"artifacts={len(artifacts)} events=483 layers=24 "
        f"manifest_sha256={sha256_bytes(artifacts['manifest.json'])}"
    )


if __name__ == "__main__":
    main()
