#!/usr/bin/env python3
"""Authenticate and independently recompute ACE-3 attention vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from attention_oracle import (
    HEAD_DIM,
    attention_score,
    attention_softmax,
    attention_value,
    mapped_kv_head,
)

EXPECTED_FILES = {
    "manifest.json": (None, None),
    "attention_params.svh": (6, None),
    "attention_score_expected.hex": (20, 32),
    "attention_score_terms.hex": (1280, 16),
    "attention_softmax_rows.hex": (20, 16),
    "attention_softmax_terms.hex": (140, 16),
    "attention_value_cases.hex": (20, 32),
    "attention_value_terms.hex": (122, 16),
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


def authenticate(generated_dir: Path, bindings: dict[str, Any]) -> None:
    require(
        bindings.get("kind") == "ace3_attention_serialized_bindings",
        "unexpected attention binding kind",
    )
    artifacts = bindings.get("serialized_artifacts")
    require(isinstance(artifacts, list), "attention artifact bindings missing")
    by_name = {artifact.get("file"): artifact for artifact in artifacts}
    require(
        set(by_name) == set(EXPECTED_FILES),
        "attention bindings do not cover the exact simulator artifact set",
    )
    for name, artifact in by_name.items():
        payload = (generated_dir / name).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        require(
            actual_hash == artifact.get("sha256"),
            f"{name} SHA256 mismatch: expected {artifact.get('sha256')}, got {actual_hash}",
        )
        require(
            len(payload) == artifact.get("byte_count"),
            f"{name} byte count mismatch",
        )
        require(
            payload.count(b"\n") == artifact.get("line_count"),
            f"{name} line count mismatch",
        )


def validate_scores(expected: list[int], terms: list[int]) -> None:
    covered_heads: set[int] = set()
    for case_index, record in enumerate(expected):
        base = case_index * HEAD_DIM
        q_values = [term & 0xFFFF for term in terms[base : base + HEAD_DIM]]
        k_values = [
            (term >> 16) & 0xFFFF for term in terms[base : base + HEAD_DIM]
        ]
        hits = [
            bool((term >> 32) & 1) for term in terms[base : base + HEAD_DIM]
        ]
        query_head = (record >> 16) & 0xF
        key_head = (record >> 20) & 0xF
        query_position = (record >> 24) & 0x7FFF
        key_position = (record >> 39) & 0x7FFF
        result = attention_score(
            q_values, k_values, hits, query_position, key_position
        )
        require(
            key_head == mapped_kv_head(query_head),
            f"score GQA mapping mismatch at case {case_index}",
        )
        observed = (
            record & 0xFFFF,
            bool((record >> 54) & 1),
            bool((record >> 55) & 1),
            bool((record >> 56) & 1),
            bool((record >> 57) & 1),
        )
        require(
            observed
            == (
                result.score_f16,
                result.causal,
                result.cache_miss,
                result.invalid,
                result.saturation,
            ),
            f"score oracle mismatch at case {case_index}",
        )
        if case_index < 14:
            covered_heads.add(query_head)
    require(covered_heads == set(range(14)), "official score head coverage incomplete")


def validate_softmax(rows: list[int], terms: list[int]) -> None:
    term_index = 0
    covered_heads: set[int] = set()
    for row_index, header in enumerate(rows):
        query_head = header & 0xF
        query_position = (header >> 4) & 0x7FFF
        count = (header >> 19) & 0xFFFF
        row_terms = terms[term_index : term_index + count]
        term_index += count
        scores = [term & 0xFFFF for term in row_terms]
        key_positions = [(term >> 16) & 0x7FFF for term in row_terms]
        causal = [bool((term >> 31) & 1) for term in row_terms]
        misses = [bool((term >> 32) & 1) for term in row_terms]
        invalid = [bool((term >> 33) & 1) for term in row_terms]
        expected_probabilities = [
            (term >> 34) & 0xFFFF for term in row_terms
        ]
        result = attention_softmax(
            scores,
            key_positions,
            causal,
            misses,
            invalid,
            query_position,
        )
        require(
            expected_probabilities == list(result.probabilities_f16),
            f"softmax probabilities mismatch at row {row_index}",
        )
        require(
            (
                bool((header >> 35) & 1),
                bool((header >> 36) & 1),
                bool((header >> 37) & 1),
            )
            == (result.row_error, result.cache_miss, result.invalid),
            f"softmax status mismatch at row {row_index}",
        )
        if row_index < 14:
            covered_heads.add(query_head)
    require(term_index == len(terms), "softmax term framing mismatch")
    require(covered_heads == set(range(14)), "official softmax head coverage incomplete")


def validate_values(cases: list[int], terms: list[int]) -> None:
    term_index = 0
    covered_heads: set[int] = set()
    covered_kv_heads: set[int] = set()
    for case_index, header in enumerate(cases):
        query_head = header & 0xF
        value_head = (header >> 4) & 0xF
        count = (header >> 29) & 0xFFFF
        case_terms = terms[term_index : term_index + count]
        term_index += count
        probabilities = [term & 0xFFFF for term in case_terms]
        values = [(term >> 16) & 0xFFFF for term in case_terms]
        hits = [bool((term >> 32) & 1) for term in case_terms]
        row_errors = [bool((term >> 33) & 1) for term in case_terms]
        result = attention_value(probabilities, values, hits, row_errors)
        require(
            value_head == mapped_kv_head(query_head),
            f"value GQA mapping mismatch at case {case_index}",
        )
        require(
            (
                (header >> 45) & 0xFFFF,
                bool((header >> 61) & 1),
                bool((header >> 62) & 1),
                bool((header >> 63) & 1),
                bool((header >> 64) & 1),
            )
            == (
                result.value_f16,
                result.row_error,
                result.cache_miss,
                result.invalid,
                result.saturation,
            ),
            f"value oracle mismatch at case {case_index}",
        )
        if case_index < 14:
            covered_heads.add(query_head)
            covered_kv_heads.add(value_head)
    require(term_index == len(terms), "value term framing mismatch")
    require(covered_heads == set(range(14)), "official value head coverage incomplete")
    require(covered_kv_heads == {0, 1}, "both GQA value groups were not covered")


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    contract = load_json(args.contract)
    bindings = load_json(args.bindings)
    require(contract.get("schema_version") == 1, "contract schema_version != 1")
    require(
        contract.get("conclusion")
        == "SCALED_QK_CAUSAL_SOFTMAX_FP16_VALUE_COMPOSITION",
        "unexpected attention contract conclusion",
    )
    authenticate(generated_dir, bindings)
    manifest = load_json(generated_dir / "manifest.json")
    require(
        manifest.get("kind") == "ace3_attention_vectors",
        "unexpected attention manifest kind",
    )
    require(
        manifest.get("model_repository")
        == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        and manifest.get("model_revision")
        == "db09cd27ead7fee40cdee309693cf83601b9c899",
        "attention model identity mismatch",
    )
    require(
        manifest.get("source_sha256")
        == {
            "config": "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
            "model_api": "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
            "scales": "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
        },
        "attention source hash set mismatch",
    )
    records = {
        name: checked_records(generated_dir / name, count, width)
        for name, (count, width) in EXPECTED_FILES.items()
        if count is not None and width is not None
    }
    validate_scores(
        records["attention_score_expected.hex"],
        records["attention_score_terms.hex"],
    )
    validate_softmax(
        records["attention_softmax_rows.hex"],
        records["attention_softmax_terms.hex"],
    )
    validate_values(
        records["attention_value_cases.hex"],
        records["attention_value_terms.hex"],
    )
    print(
        "ACE3_ATTENTION_VECTOR_VALIDATION_PASS serialized_sha256=pass "
        "oracle=bit_exact score_cases=20 softmax_rows=20 value_cases=20 "
        "query_heads=14 kv_heads=2 directed_errors=pass"
    )


if __name__ == "__main__":
    main()
