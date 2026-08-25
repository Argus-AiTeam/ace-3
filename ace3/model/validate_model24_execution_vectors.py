#!/usr/bin/env python3
"""Fail-closed validation for deterministic model24 execution vectors."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from model24_execution_oracle import (
    ContractError,
    ORACLE_GEOMETRY,
    argmax_lowest,
    build_vector_artifacts,
    expected_schedule,
    load_json_bytes,
    require,
    require_provenance_commit,
    sha256_bytes,
    validate_execution_contract,
    validate_vector_bindings,
)


def _validate_reduced_execution(document: object) -> None:
    require(isinstance(document, dict), "reduced execution root must be an object")
    require(
        document.get("kind") == "ace3_model24_reduced_software_oracle_execution",
        "reduced execution kind mismatch",
    )
    require(document.get("geometry") == ORACLE_GEOMETRY.document(), "geometry mismatch")
    inventory = document.get("tensor_inventory")
    consumed = document.get("consumed_tensor_names")
    require(isinstance(inventory, list) and len(inventory) == 627, "tensor inventory")
    require(
        isinstance(consumed, list)
        and consumed == sorted(record["name"] for record in inventory),
        "tensor consumption coverage mismatch",
    )
    require(len(set(consumed)) == 627, "tensor inventory contains duplicates")
    schedule = expected_schedule(ORACLE_GEOMETRY.schedule_geometry())
    executions = document.get("executions")
    require(isinstance(executions, list) and len(executions) == 2, "execution count")
    for execution_index, execution in enumerate(executions):
        events = execution.get("events")
        require(isinstance(events, list) and len(events) == 483, "event count mismatch")
        residual_outputs: list[str] = []
        kv_actions: list[tuple[int, str]] = []
        for expected, actual in zip(schedule, events, strict=True):
            for field in ("ordinal", "operation", "layer_id", "tensor_names"):
                require(
                    actual.get(field) == expected[field],
                    f"execution {execution_index} event {expected['ordinal']} {field} mismatch",
                )
            output = actual.get("output")
            require(
                isinstance(output, dict)
                and output.get("dtype") == "FP16"
                and output.get("elements", 0) > 0,
                "event output is not non-empty FP16",
            )
            if "residual_output" in expected:
                transition = actual.get("residual_transition")
                require(
                    isinstance(transition, dict)
                    and transition.get("input") == expected["residual_input"]
                    and transition.get("output") == expected["residual_output"]
                    and transition.get("format") == "FP16",
                    "residual transition mismatch",
                )
                residual_outputs.append(transition["output"])
            if expected["operation"] in ("kv_write", "kv_read"):
                transition = actual.get("kv_transition")
                require(
                    isinstance(transition, dict)
                    and transition.get("format") == "FP16"
                    and transition.get("layer_id") == expected["layer_id"],
                    "KV transition mismatch",
                )
                kv_actions.append((transition["layer_id"], transition["action"]))
                require(
                    sorted(transition["owners"].values()) == sorted(expected["kv_owners"]),
                    "KV ownership mismatch",
                )
                if transition["action"] == "read":
                    require(
                        transition["positions"] == list(range(execution_index + 1)),
                        "KV read history mismatch",
                    )
        require(len(residual_outputs) == 48, "residual transition count mismatch")
        require(
            kv_actions
            == [
                (layer_id, action)
                for layer_id in range(24)
                for action in ("write", "read")
            ],
            "KV write/read order mismatch",
        )
        lm_head = execution.get("lm_head")
        require(
            isinstance(lm_head, dict)
            and lm_head.get("tied") is True
            and lm_head.get("tied_to") == "model.embed_tokens.weight"
            and lm_head.get("group_size") == 128
            and lm_head.get("groups_per_logit") == 2,
            "tied grouped lm_head mismatch",
        )
        logits = [
            struct.unpack("<e", bits.to_bytes(2, "little"))[0]
            for bits in lm_head["logit_bits"]
        ]
        require(
            execution.get("output_token_id") == argmax_lowest(logits),
            "deterministic argmax mismatch",
        )
        require(
            execution.get("final_norm", {}).get("dtype") == "FP16",
            "final RMSNorm result mismatch",
        )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=repository_root / "build" / "model24_execution_prep_cf11" / "vectors",
    )
    return parser.parse_args()


def validate_vector_directory(repository_root: Path, vector_dir: Path) -> dict[str, int]:
    contracts = repository_root / "ace3" / "contracts"
    contract_path = contracts / "model24_execution.json"
    bindings_path = contracts / "model24_execution_vector_bindings.json"
    contract_payload = contract_path.read_bytes()
    bindings_payload = bindings_path.read_bytes()
    tensor_payload = (contracts / "model24_tensor_map.json").read_bytes()
    control_payload = (contracts / "model24_control.json").read_bytes()
    oracle_source_payload = Path(__file__).with_name(
        "model24_execution_oracle.py"
    ).read_bytes()

    require_provenance_commit(repository_root)
    contract = load_json_bytes(contract_payload, str(contract_path))
    bindings = load_json_bytes(bindings_payload, str(bindings_path))
    validate_execution_contract(contract, tensor_payload, control_payload)
    validate_vector_bindings(
        bindings,
        sha256_bytes(contract_payload),
        oracle_source_payload,
    )

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
    _validate_reduced_execution(
        load_json_bytes(
            actual_payloads["reduced_execution.json"],
            "reduced_execution.json",
        )
    )

    expected_payloads = build_vector_artifacts(
        sha256_bytes(contract_payload),
        sha256_bytes(bindings_payload),
        oracle_source_payload,
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
        summary = validate_vector_directory(repository_root, args.vector_dir.resolve())
    except (ContractError, OSError) as error:
        raise SystemExit(f"MODEL24_EXECUTION_VALIDATION_FAIL {error}") from error
    print(
        "MODEL24_EXECUTION_VALIDATION_PASS "
        f"layers={summary['official_layers']} "
        f"events={summary['official_events']} "
        f"tensors={summary['official_tensor_count']} "
        f"generated_tokens={summary['generated_token_count']}"
    )


if __name__ == "__main__":
    main()
