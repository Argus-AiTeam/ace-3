#!/usr/bin/env python3
"""Generate authenticated Qwen2.5 RoPE and K/V cache vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from qwen2_rope_oracle import qwen2_coefficient, rotate_pair

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
QUERY_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
PAIR_COUNT = HEAD_DIM // 2
VECTOR_POSITION = 37
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


def checked_bytes(source_dir: Path, name: str) -> bytes:
    filename, expected_hash = SOURCES[name]
    payload = (source_dir / filename).read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{name} source hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return payload


def signed_sample(samples: list[int], index: int, negative: bool) -> int:
    value = samples[index % len(samples)]
    if value & 0x7C00 == 0x7C00:
        raise RuntimeError("official FP16 scale sample is non-finite")
    return value ^ (0x8000 if negative else 0)


def build_rope_cases(samples: list[int]) -> list[int]:
    records: list[int] = []
    for is_key, head_count, sample_offset in (
        (False, QUERY_HEADS, 0),
        (True, KV_HEADS, 4096),
    ):
        for head in range(head_count):
            for pair in range(PAIR_COUNT):
                channel = head * HEAD_DIM + pair
                low = signed_sample(
                    samples, sample_offset + channel, (head + pair) % 3 == 0
                )
                high = signed_sample(
                    samples,
                    sample_offset + channel + PAIR_COUNT,
                    (head + pair) % 4 == 0,
                )
                cos_f16, sin_f16 = qwen2_coefficient(VECTOR_POSITION, pair)
                out_low, out_high, invalid, saturation = rotate_pair(
                    low, high, cos_f16, sin_f16
                )
                record = low
                record |= high << 16
                record |= cos_f16 << 32
                record |= sin_f16 << 48
                record |= out_low << 64
                record |= out_high << 80
                record |= int(invalid) << 96
                record |= int(saturation) << 97
                record |= int(is_key) << 98
                record |= head << 99
                record |= pair << 103
                record |= VECTOR_POSITION << 108
                records.append(record)
    return records


def build_cache_cases(samples: list[int]) -> list[int]:
    records: list[int] = []
    for head in range(KV_HEADS):
        for dimension in range(HEAD_DIM):
            k_f16 = signed_sample(
                samples, 2048 + head * HEAD_DIM + dimension, dimension % 5 == 0
            )
            v_f16 = signed_sample(
                samples, 6144 + head * HEAD_DIM + dimension, dimension % 7 == 0
            )
            record = k_f16
            record |= v_f16 << 16
            record |= dimension << 32
            record |= head << 38
            record |= 3 << 39
            record |= 0 << 54
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    source_dir = args.official_tensor_dir.resolve(strict=True)
    output_dir = args.output_dir
    config = json.loads(checked_bytes(source_dir, "config"))
    expected_config = {
        "hidden_size": 896,
        "num_attention_heads": QUERY_HEADS,
        "num_key_value_heads": KV_HEADS,
        "max_position_embeddings": 32768,
        "rope_theta": 1_000_000.0,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("authenticated Qwen2.5 RoPE geometry mismatch")
    if config["hidden_size"] // config["num_attention_heads"] != HEAD_DIM:
        raise RuntimeError("authenticated head dimension mismatch")
    model_api = json.loads(checked_bytes(source_dir, "model_api"))
    if (
        model_api.get("id") != MODEL_REPOSITORY
        or model_api.get("sha") != MODEL_REVISION
    ):
        raise RuntimeError("authenticated model identity mismatch")
    scales_raw = checked_bytes(source_dir, "scales")
    samples = list(struct.unpack(f"<{len(scales_raw) // 2}H", scales_raw))

    rope_records = build_rope_cases(samples)
    cache_records = build_cache_cases(samples)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_qwen2_qkv_rope_cache_vectors",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_sha256": {name: value[1] for name, value in SOURCES.items()},
        "geometry": {
            "hidden_size": 896,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rotary_pairs": PAIR_COUNT,
            "max_position_embeddings": 32768,
            "rope_theta": 1_000_000.0,
        },
        "numerical_order": "binary16 multiply, binary16 multiply, binary16 add",
        "rotate_half": "[x[0:32],x[32:64]] -> [-x[32:64],x[0:32]]",
        "vector_position": VECTOR_POSITION,
        "rope_case_count": len(rope_records),
        "cache_case_count": len(cache_records),
        "operand_source": (
            "FP16 operands are deterministic signed selections from authenticated "
            "layer-0 q_proj scale samples; they are not captured Q/K/V activations."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "rope_cases.hex").write_text(
        "".join(f"{record:032x}\n" for record in rope_records), encoding="ascii"
    )
    (output_dir / "cache_cases.hex").write_text(
        "".join(f"{record:016x}\n" for record in cache_records), encoding="ascii"
    )
    (output_dir / "qkv_params.svh").write_text(
        "`define QKV_ROPE_CASES 512\n"
        "`define QKV_CACHE_CASES 128\n"
        "`define QKV_VECTOR_POSITION 37\n",
        encoding="ascii",
    )
    print(
        "QKV_ROPE_CACHE_VECTOR_GENERATION_PASS "
        f"rope_cases={len(rope_records)} cache_cases={len(cache_records)} "
        "query_heads=14 kv_heads=2 head_dim=64"
    )


if __name__ == "__main__":
    main()
