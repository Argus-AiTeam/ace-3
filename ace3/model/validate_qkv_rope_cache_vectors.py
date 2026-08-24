#!/usr/bin/env python3
"""Authenticate and independently validate QKV/RoPE/KV-cache vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from qwen2_rope_oracle import qwen2_coefficient, rotate_pair

EXPECTED_FILES = {
    "manifest.json": (None, None),
    "rope_cases.hex": (512, 32),
    "cache_cases.hex": (128, 16),
    "qkv_params.svh": (3, None),
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
    return json.loads(path.read_text(encoding="utf-8"))


def authenticate(generated_dir: Path, bindings: dict[str, Any]) -> None:
    require(
        bindings.get("kind") == "ace3_qkv_rope_cache_serialized_bindings",
        "unexpected QKV binding kind",
    )
    artifacts = bindings.get("serialized_artifacts")
    require(isinstance(artifacts, list), "serialized artifact bindings missing")
    by_name = {item.get("file"): item for item in artifacts}
    require(
        set(by_name) == set(EXPECTED_FILES),
        "QKV bindings do not cover the exact simulator artifact set",
    )
    for name, item in by_name.items():
        payload = (generated_dir / name).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        require(
            actual_hash == item.get("sha256"),
            f"{name} SHA256 mismatch: expected {item.get('sha256')}, got {actual_hash}",
        )
        require(len(payload) == item.get("byte_count"), f"{name} byte count mismatch")
        require(
            payload.count(b"\n") == item.get("line_count"),
            f"{name} line count mismatch",
        )


def checked_records(path: Path, count: int, width: int) -> list[int]:
    lines = path.read_text(encoding="ascii").splitlines()
    require(len(lines) == count, f"{path.name}: record count mismatch")
    require(
        all(
            len(line) == width
            and all(character in "0123456789abcdef" for character in line)
            for line in lines
        ),
        f"{path.name}: malformed hexadecimal record",
    )
    return [int(line, 16) for line in lines]


def validate_rope(records: list[int]) -> None:
    coverage: dict[bool, set[tuple[int, int]]] = {False: set(), True: set()}
    for index, record in enumerate(records):
        low_f16 = record & 0xFFFF
        high_f16 = (record >> 16) & 0xFFFF
        cos_f16 = (record >> 32) & 0xFFFF
        sin_f16 = (record >> 48) & 0xFFFF
        expected_low = (record >> 64) & 0xFFFF
        expected_high = (record >> 80) & 0xFFFF
        expected_invalid = bool((record >> 96) & 1)
        expected_saturation = bool((record >> 97) & 1)
        is_key = bool((record >> 98) & 1)
        head = (record >> 99) & 0xF
        pair = (record >> 103) & 0x1F
        position = (record >> 108) & 0x7FFF
        require(
            (cos_f16, sin_f16) == qwen2_coefficient(position, pair),
            f"RoPE coefficient mismatch at record {index}",
        )
        actual = rotate_pair(low_f16, high_f16, cos_f16, sin_f16)
        require(
            actual
            == (
                expected_low,
                expected_high,
                expected_invalid,
                expected_saturation,
            ),
            f"RoPE oracle mismatch at record {index}",
        )
        require(head < (2 if is_key else 14), f"illegal head at record {index}")
        coverage[is_key].add((head, pair))
    require(
        coverage[False] == {(head, pair) for head in range(14) for pair in range(32)},
        "query-head rotary coverage is incomplete",
    )
    require(
        coverage[True] == {(head, pair) for head in range(2) for pair in range(32)},
        "key-head rotary coverage is incomplete",
    )


def validate_cache(records: list[int]) -> None:
    coverage: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        k_f16 = record & 0xFFFF
        v_f16 = (record >> 16) & 0xFFFF
        dimension = (record >> 32) & 0x3F
        head = (record >> 38) & 0x1
        position = (record >> 39) & 0x7FFF
        cache_slot = (record >> 54) & 0x3
        require(head < 2 and dimension < 64, f"cache geometry mismatch at {index}")
        require(
            cache_slot == 0 and position == 3,
            f"cache vector address mismatch at {index}",
        )
        require(
            (k_f16 & 0x7C00) != 0x7C00 and (v_f16 & 0x7C00) != 0x7C00,
            f"non-finite cache value at {index}",
        )
        coverage.add((head, dimension))
    require(
        coverage == {(head, dimension) for head in range(2) for dimension in range(64)},
        "K/V head-dimension coverage is incomplete",
    )


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    contract = load_json(args.contract)
    bindings = load_json(args.bindings)
    require(contract.get("schema_version") == 1, "contract schema_version != 1")
    require(
        contract.get("conclusion") == "QKV_ROPE_FP16_KV_CACHE",
        "unexpected QKV contract conclusion",
    )
    require(bindings.get("schema_version") == 1, "bindings schema_version != 1")
    authenticate(generated_dir, bindings)
    manifest = load_json(generated_dir / "manifest.json")
    require(
        manifest.get("kind") == "ace3_qwen2_qkv_rope_cache_vectors",
        "unexpected QKV vector manifest kind",
    )
    require(
        manifest.get("model_repository") == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        and manifest.get("model_revision")
        == "db09cd27ead7fee40cdee309693cf83601b9c899",
        "QKV model identity mismatch",
    )
    require(
        manifest.get("source_sha256")
        == {
            "config": "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
            "model_api": "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
            "scales": "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
        },
        "QKV source hash set mismatch",
    )
    rope_records = checked_records(
        generated_dir / "rope_cases.hex", *EXPECTED_FILES["rope_cases.hex"]
    )
    cache_records = checked_records(
        generated_dir / "cache_cases.hex", *EXPECTED_FILES["cache_cases.hex"]
    )
    validate_rope(rope_records)
    validate_cache(cache_records)
    print(
        "QKV_ROPE_CACHE_VECTOR_VALIDATION_PASS serialized_sha256=pass "
        "rope_cases=512 cache_cases=128 query_heads=14 kv_heads=2 "
        "head_dim=64 oracle=bit_exact"
    )


if __name__ == "__main__":
    main()
