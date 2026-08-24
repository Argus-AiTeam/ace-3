#!/usr/bin/env python3
"""Generate authenticated official-derived and directed attention vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from attention_oracle import (
    HEAD_DIM,
    QUERY_HEADS,
    attention_score,
    attention_softmax,
    attention_value,
    mapped_kv_head,
)

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
SOURCES: dict[str, tuple[str, str]] = {
    "config": (
        "config.json",
        "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
    ),
    "model_api": (
        "model-api.json",
        "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
    ),
    "scales": (
        "sample-model_layers_0_self_attn_q_proj-scales.bin",
        "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def checked_bytes(source_dir: Path, source: str) -> bytes:
    filename, expected_hash = SOURCES[source]
    payload = (source_dir / filename).read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{source} source hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return payload


def signed_sample(samples: list[int], index: int, negative: bool = False) -> int:
    value = samples[index % len(samples)]
    if value & 0x7C00 == 0x7C00:
        raise RuntimeError("official FP16 scale sample is non-finite")
    return value ^ (0x8000 if negative else 0)


def official_qk(
    samples: list[int], query_head: int, key_position: int
) -> tuple[list[int], list[int]]:
    kv_head = mapped_kv_head(query_head)
    q_values = [
        signed_sample(
            samples,
            query_head * HEAD_DIM + dimension,
            (query_head + dimension) % 5 == 0,
        )
        for dimension in range(HEAD_DIM)
    ]
    k_values = [
        signed_sample(
            samples,
            2048 + kv_head * HEAD_DIM + key_position * 131 + dimension,
            (key_position + dimension) % 7 == 0,
        )
        for dimension in range(HEAD_DIM)
    ]
    return q_values, k_values


def build_score_cases(samples: list[int]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for query_head in range(QUERY_HEADS):
        key_position = query_head % 8
        q_values, k_values = official_qk(samples, query_head, key_position)
        cases.append(
            {
                "query_head": query_head,
                "key_head": mapped_kv_head(query_head),
                "query_position": 7,
                "key_position": key_position,
                "q": q_values,
                "k": k_values,
                "hits": [True] * HEAD_DIM,
            }
        )
    directed = [
        (0, 0, 3, 4, [0x3C00] * HEAD_DIM, [0x3C00] * HEAD_DIM, [True] * HEAD_DIM),
        (7, 1, 3, 3, [0x3C00] * HEAD_DIM, [0x3C00] * HEAD_DIM, [False] + [True] * 63),
        (1, 0, 2, 2, [0x7C00] + [0x3C00] * 63, [0x3C00] * HEAD_DIM, [True] * HEAD_DIM),
        (8, 1, 2, 2, [0x7BFF] * HEAD_DIM, [0x7BFF] * HEAD_DIM, [True] * HEAD_DIM),
        (2, 0, 2, 2, [0x7BFF] * HEAD_DIM, [0xFBFF] * HEAD_DIM, [True] * HEAD_DIM),
        (
            13,
            1,
            2,
            2,
            [0x3C00] * HEAD_DIM,
            [0x3C00 if dimension % 2 == 0 else 0xBC00 for dimension in range(HEAD_DIM)],
            [True] * HEAD_DIM,
        ),
    ]
    for query_head, key_head, query_position, key_position, q_values, k_values, hits in directed:
        cases.append(
            {
                "query_head": query_head,
                "key_head": key_head,
                "query_position": query_position,
                "key_position": key_position,
                "q": q_values,
                "k": k_values,
                "hits": hits,
            }
        )
    return cases


def build_softmax_rows(samples: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_head in range(QUERY_HEADS):
        scores: list[int] = []
        for key_position in range(8):
            q_values, k_values = official_qk(samples, query_head, key_position)
            scores.append(
                attention_score(
                    q_values, k_values, [True] * HEAD_DIM, 7, key_position
                ).score_f16
            )
        rows.append(
            {
                "query_head": query_head,
                "query_position": 7,
                "scores": scores,
                "key_positions": list(range(8)),
                "causal": [True] * 8,
                "misses": [False] * 8,
                "invalid": [False] * 8,
            }
        )
    rows.extend(
        [
            {
                "query_head": 0,
                "query_position": 3,
                "scores": [0x3C00, 0x3C00, 0x4000, 0x4000, 0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF],
                "key_positions": list(range(8)),
                "causal": [True] * 4 + [False] * 4,
                "misses": [False] * 8,
                "invalid": [False] * 8,
            },
            {
                "query_head": 7,
                "query_position": 3,
                "scores": [0x7BFF, 0x0000, 0xC000, 0xFBFF],
                "key_positions": list(range(4)),
                "causal": [True] * 4,
                "misses": [False] * 4,
                "invalid": [False] * 4,
            },
            {
                "query_head": 1,
                "query_position": 3,
                "scores": [0x3C00] * 4,
                "key_positions": list(range(4)),
                "causal": [True] * 4,
                "misses": [False] * 4,
                "invalid": [False] * 4,
            },
            {
                "query_head": 8,
                "query_position": 3,
                "scores": [0x0000] * 4,
                "key_positions": list(range(4)),
                "causal": [True] * 4,
                "misses": [False, True, False, False],
                "invalid": [False] * 4,
            },
            {
                "query_head": 2,
                "query_position": 3,
                "scores": [0x0000, 0x7C00, 0x0000, 0x0000],
                "key_positions": list(range(4)),
                "causal": [True] * 4,
                "misses": [False] * 4,
                "invalid": [False, True, False, False],
            },
            {
                "query_head": 13,
                "query_position": 0,
                "scores": [0x0000] * 4,
                "key_positions": [1, 2, 3, 4],
                "causal": [False] * 4,
                "misses": [False] * 4,
                "invalid": [False] * 4,
            },
        ]
    )
    return rows


def build_value_cases(
    samples: list[int], softmax_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for query_head, row in enumerate(softmax_rows[:QUERY_HEADS]):
        softmax = attention_softmax(
            row["scores"],
            row["key_positions"],
            row["causal"],
            row["misses"],
            row["invalid"],
            row["query_position"],
        )
        dimension = (query_head * 5) % HEAD_DIM
        values = [
            signed_sample(
                samples,
                6144 + mapped_kv_head(query_head) * HEAD_DIM +
                key_position * 67 + dimension,
                (query_head + key_position) % 6 == 0,
            )
            for key_position in range(8)
        ]
        cases.append(
            {
                "query_head": query_head,
                "value_head": mapped_kv_head(query_head),
                "query_position": 7,
                "dimension": dimension,
                "probabilities": list(softmax.probabilities_f16),
                "values": values,
                "hits": [True] * 8,
                "row_errors": [False] * 8,
            }
        )
    cases.extend(
        [
            {
                "query_head": 0,
                "value_head": 0,
                "query_position": 3,
                "dimension": 0,
                "probabilities": [0x3800, 0x3800],
                "values": [0x3C00, 0x4000],
                "hits": [True, False],
                "row_errors": [False, False],
            },
            {
                "query_head": 7,
                "value_head": 1,
                "query_position": 3,
                "dimension": 1,
                "probabilities": [0x3C00, 0x0000],
                "values": [0x4000, 0x7C00],
                "hits": [True, False],
                "row_errors": [False, False],
            },
            {
                "query_head": 1,
                "value_head": 0,
                "query_position": 3,
                "dimension": 2,
                "probabilities": [0x3C00],
                "values": [0x7C00],
                "hits": [True],
                "row_errors": [False],
            },
            {
                "query_head": 8,
                "value_head": 1,
                "query_position": 3,
                "dimension": 3,
                "probabilities": [0x0000],
                "values": [0x0000],
                "hits": [True],
                "row_errors": [True],
            },
            {
                "query_head": 2,
                "value_head": 0,
                "query_position": 3,
                "dimension": 4,
                "probabilities": [0x3C00, 0x3C00],
                "values": [0x7BFF, 0x7BFF],
                "hits": [True, True],
                "row_errors": [False, False],
            },
            {
                "query_head": 13,
                "value_head": 1,
                "query_position": 3,
                "dimension": 63,
                "probabilities": [0x3800, 0x3800],
                "values": [0x3C00, 0xBC00],
                "hits": [True, True],
                "row_errors": [False, False],
            },
        ]
    )
    return cases


def serialize(output_dir: Path, samples: list[int]) -> dict[str, int]:
    score_cases = build_score_cases(samples)
    softmax_rows = build_softmax_rows(samples)
    value_cases = build_value_cases(samples, softmax_rows)

    score_terms: list[int] = []
    score_expected: list[int] = []
    for case in score_cases:
        result = attention_score(
            case["q"],
            case["k"],
            case["hits"],
            case["query_position"],
            case["key_position"],
        )
        for q_bits, k_bits, hit in zip(
            case["q"], case["k"], case["hits"], strict=True
        ):
            score_terms.append(q_bits | (k_bits << 16) | (int(hit) << 32))
        record = result.score_f16
        record |= case["query_head"] << 16
        record |= case["key_head"] << 20
        record |= case["query_position"] << 24
        record |= case["key_position"] << 39
        record |= int(result.causal) << 54
        record |= int(result.cache_miss) << 55
        record |= int(result.invalid) << 56
        record |= int(result.saturation) << 57
        score_expected.append(record)

    softmax_headers: list[int] = []
    softmax_terms: list[int] = []
    for row in softmax_rows:
        result = attention_softmax(
            row["scores"],
            row["key_positions"],
            row["causal"],
            row["misses"],
            row["invalid"],
            row["query_position"],
        )
        count = len(row["scores"])
        header = row["query_head"]
        header |= row["query_position"] << 4
        header |= count << 19
        header |= int(result.row_error) << 35
        header |= int(result.cache_miss) << 36
        header |= int(result.invalid) << 37
        softmax_headers.append(header)
        for score, key_position, causal, miss, invalid, probability in zip(
            row["scores"],
            row["key_positions"],
            row["causal"],
            row["misses"],
            row["invalid"],
            result.probabilities_f16,
            strict=True,
        ):
            term = score
            term |= key_position << 16
            term |= int(causal) << 31
            term |= int(miss) << 32
            term |= int(invalid) << 33
            term |= probability << 34
            softmax_terms.append(term)

    value_headers: list[int] = []
    value_terms: list[int] = []
    for case in value_cases:
        result = attention_value(
            case["probabilities"],
            case["values"],
            case["hits"],
            case["row_errors"],
        )
        count = len(case["probabilities"])
        header = case["query_head"]
        header |= case["value_head"] << 4
        header |= case["query_position"] << 8
        header |= case["dimension"] << 23
        header |= count << 29
        header |= result.value_f16 << 45
        header |= int(result.row_error) << 61
        header |= int(result.cache_miss) << 62
        header |= int(result.invalid) << 63
        header |= int(result.saturation) << 64
        value_headers.append(header)
        for probability, value, hit, row_error in zip(
            case["probabilities"],
            case["values"],
            case["hits"],
            case["row_errors"],
            strict=True,
        ):
            value_terms.append(
                probability
                | (value << 16)
                | (int(hit) << 32)
                | (int(row_error) << 33)
            )

    files = {
        "attention_score_terms.hex": "".join(
            f"{record:016x}\n" for record in score_terms
        ),
        "attention_score_expected.hex": "".join(
            f"{record:032x}\n" for record in score_expected
        ),
        "attention_softmax_rows.hex": "".join(
            f"{record:016x}\n" for record in softmax_headers
        ),
        "attention_softmax_terms.hex": "".join(
            f"{record:016x}\n" for record in softmax_terms
        ),
        "attention_value_cases.hex": "".join(
            f"{record:032x}\n" for record in value_headers
        ),
        "attention_value_terms.hex": "".join(
            f"{record:016x}\n" for record in value_terms
        ),
        "attention_params.svh": (
            f"`define ATTENTION_SCORE_CASES {len(score_cases)}\n"
            f"`define ATTENTION_SCORE_TERMS {len(score_terms)}\n"
            f"`define ATTENTION_SOFTMAX_ROWS {len(softmax_rows)}\n"
            f"`define ATTENTION_SOFTMAX_TERMS {len(softmax_terms)}\n"
            f"`define ATTENTION_VALUE_CASES {len(value_cases)}\n"
            f"`define ATTENTION_VALUE_TERMS {len(value_terms)}\n"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in files.items():
        (output_dir / filename).write_text(payload, encoding="ascii")
    return {
        "score_cases": len(score_cases),
        "score_terms": len(score_terms),
        "softmax_rows": len(softmax_rows),
        "softmax_terms": len(softmax_terms),
        "value_cases": len(value_cases),
        "value_terms": len(value_terms),
    }


def main() -> None:
    args = parse_args()
    source_dir = args.official_tensor_dir.resolve(strict=True)
    config = json.loads(checked_bytes(source_dir, "config"))
    if (
        config.get("hidden_size") != 896
        or config.get("num_attention_heads") != 14
        or config.get("num_key_value_heads") != 2
        or config.get("max_position_embeddings") != 32768
    ):
        raise RuntimeError("authenticated attention geometry mismatch")
    model_api = json.loads(checked_bytes(source_dir, "model_api"))
    if (
        model_api.get("id") != MODEL_REPOSITORY
        or model_api.get("sha") != MODEL_REVISION
    ):
        raise RuntimeError("authenticated model identity mismatch")
    scales_raw = checked_bytes(source_dir, "scales")
    samples = list(struct.unpack(f"<{len(scales_raw) // 2}H", scales_raw))
    counts = serialize(args.output_dir, samples)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_attention_vectors",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_sha256": {name: source[1] for name, source in SOURCES.items()},
        "geometry": {
            "query_heads": 14,
            "kv_heads": 2,
            "head_dim": 64,
            "gqa_group_size": 7,
            "score_scale": "1/8",
        },
        "counts": counts,
        "operand_source": (
            "Official-derived operands are deterministic signed selections from "
            "the authenticated layer-0 q_proj FP16 scale sample; they are not "
            "captured runtime Q/K/V activations. Directed rows cover errors and "
            "FP16 boundaries."
        ),
        "directed_coverage": [
            "causal boundary",
            "future mask",
            "ties",
            "exponential underflow",
            "K cache miss",
            "V cache miss",
            "non-finite operand",
            "positive and negative saturation",
            "cancellation",
            "upstream row error",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE3_ATTENTION_VECTOR_GENERATION_PASS "
        f"score_cases={counts['score_cases']} "
        f"softmax_rows={counts['softmax_rows']} "
        f"value_cases={counts['value_cases']} "
        "official_derived=42 directed=18 query_heads=14 kv_heads=2"
    )


if __name__ == "__main__":
    main()
