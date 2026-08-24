#!/usr/bin/env python3
"""Authenticate and semantically recheck ACE-3 FP16 adaptation vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fp16_adaptation_oracle import residual_add, rmsnorm, silu_gate

EXPECTED_FILES = {
    "manifest.json",
    "residual_cases.hex",
    "silu_cases.hex",
    "rms_inputs.hex",
    "rms_expected.hex",
    "rms_meta.hex",
    "fp16_adaptation_params.svh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_hex(path: Path, width: int) -> list[int]:
    records = path.read_text(encoding="ascii").splitlines()
    require(
        all(
            len(record) == width
            and all(character in "0123456789abcdef" for character in record)
            for record in records
        ),
        f"{path.name}: malformed hexadecimal record",
    )
    return [int(record, 16) for record in records]


def authenticate(generated_dir: Path, bindings: dict[str, Any]) -> None:
    require(
        bindings.get("kind") == "ace3_fp16_adaptation_serialized_bindings",
        "unexpected FP16 adaptation binding kind",
    )
    artifacts = bindings.get("serialized_artifacts")
    require(isinstance(artifacts, list), "serialized artifact bindings missing")
    by_name = {artifact.get("file"): artifact for artifact in artifacts}
    require(set(by_name) == EXPECTED_FILES, "binding artifact set mismatch")
    for name in sorted(EXPECTED_FILES):
        payload = (generated_dir / name).read_bytes()
        binding = by_name[name]
        actual = hashlib.sha256(payload).hexdigest()
        require(
            actual == binding.get("sha256"),
            f"{name} SHA256 mismatch: expected {binding.get('sha256')}, got {actual}",
        )
        require(len(payload) == binding.get("byte_count"), f"{name} byte count mismatch")
        require(
            payload.count(b"\n") == binding.get("line_count"),
            f"{name} line count mismatch",
        )


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    contract = load_json(args.contract)
    bindings = load_json(args.bindings)
    manifest = load_json(generated_dir / "manifest.json")
    for path, document in (
        (args.contract, contract),
        (args.bindings, bindings),
        (generated_dir / "manifest.json", manifest),
    ):
        require(isinstance(document, dict), f"{path}: JSON root must be an object")
        require(document.get("schema_version") == 1, f"{path}: schema_version != 1")
    authenticate(generated_dir, bindings)

    require(
        manifest.get("kind") == "ace3_fp16_adaptation_vectors",
        "unexpected FP16 adaptation manifest kind",
    )
    require(
        manifest.get("model_repository")
        == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        and manifest.get("model_revision")
        == "db09cd27ead7fee40cdee309693cf83601b9c899",
        "model identity mismatch",
    )
    require(
        manifest.get("source_sha256") == contract.get("official_source_sha256"),
        "official source hash set mismatch",
    )
    require(
        manifest.get("hidden_size") == 896
        and manifest.get("intermediate_size") == 4864
        and manifest.get("rms_epsilon") == "1e-6"
        and manifest.get("rms_epsilon_q48") == 281_474_977,
        "frozen model geometry or RMS epsilon mismatch",
    )

    residual_stream = read_hex(generated_dir / "residual_cases.hex", 13)
    silu_stream = read_hex(generated_dir / "silu_cases.hex", 13)
    rms_inputs = read_hex(generated_dir / "rms_inputs.hex", 8)
    rms_expected = read_hex(generated_dir / "rms_expected.hex", 5)
    rms_meta = read_hex(generated_dir / "rms_meta.hex", 12)
    residual_cases = manifest["residual_cases"]
    silu_cases = manifest["silu_cases"]
    rms_transactions = manifest["rms_transactions"]
    require(len(residual_stream) == len(residual_cases) >= 8, "residual case count")
    require(len(silu_stream) == len(silu_cases) >= 8, "SiLU case count")
    require(len(rms_transactions) >= 4, "RMS transaction count")

    for packed, case in zip(residual_stream, residual_cases, strict=True):
        expected = residual_add(case["left"], case["right"])
        serialized = (
            case["left"]
            | (case["right"] << 16)
            | (expected[0] << 32)
            | (int(expected[1]) << 48)
            | (int(expected[2]) << 49)
        )
        require(packed == serialized, f"{case['name']}: residual oracle mismatch")
        require(
            [case["result"], case["invalid"], case["saturation"]]
            == [expected[0], expected[1], expected[2]],
            f"{case['name']}: residual manifest mismatch",
        )

    for packed, case in zip(silu_stream, silu_cases, strict=True):
        expected = silu_gate(case["gate"], case["up"])
        serialized = (
            case["gate"]
            | (case["up"] << 16)
            | (expected[0] << 32)
            | (int(expected[1]) << 48)
            | (int(expected[2]) << 49)
        )
        require(packed == serialized, f"{case['name']}: SiLU oracle mismatch")
        require(
            [case["result"], case["invalid"], case["saturation"]]
            == [expected[0], expected[1], expected[2]],
            f"{case['name']}: SiLU manifest mismatch",
        )

    input_index = 0
    output_index = 0
    for transaction_index, transaction in enumerate(rms_transactions):
        size = manifest["rms_test_size"]
        inputs = rms_inputs[input_index : input_index + size]
        activations = [record & 0xFFFF for record in inputs]
        weights = [(record >> 16) & 0xFFFF for record in inputs]
        outputs, mean_q48, rms_q24 = rmsnorm(activations, weights)
        require(
            activations == transaction["activations"]
            and weights == transaction["weights"]
            and mean_q48 == transaction["mean_q48"]
            and rms_q24 == transaction["rms_q24"],
            f"RMS transaction {transaction_index} manifest mismatch",
        )
        for local_index, expected in enumerate(outputs):
            packed = rms_expected[output_index + local_index]
            serialized = expected[0] | (int(expected[1]) << 16) | (
                int(expected[2]) << 17
            )
            require(
                packed == serialized,
                f"RMS transaction {transaction_index} output mismatch",
            )
        invalid = any(item[1] for item in outputs)
        require(
            rms_meta[transaction_index] == rms_q24 | (int(invalid) << 46),
            f"RMS transaction {transaction_index} metadata mismatch",
        )
        input_index += size
        output_index += size
    require(
        input_index == len(rms_inputs) and output_index == len(rms_expected),
        "unconsumed RMS stream records",
    )

    all_cases = residual_cases + silu_cases + rms_transactions
    require(
        sum(case["source"].startswith("official_") for case in all_cases) == 5,
        "official-derived case coverage mismatch",
    )
    require(
        any(case.get("invalid") for case in residual_cases)
        and any(case.get("saturation") for case in residual_cases)
        and any(case.get("invalid") for case in silu_cases)
        and any(case.get("saturation") for case in silu_cases)
        and any(case.get("invalid") for case in rms_transactions),
        "non-vacuous exceptional-value coverage missing",
    )
    print(
        "FP16_ADAPTATION_VECTOR_VALIDATION_PASS "
        f"residual={len(residual_cases)} silu={len(silu_cases)} "
        f"rms_transactions={len(rms_transactions)} official=5 "
        "sha256=pass oracle_recompute=pass nonvacuous=pass"
    )


if __name__ == "__main__":
    main()
