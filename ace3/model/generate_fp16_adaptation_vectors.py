#!/usr/bin/env python3
"""Generate deterministic authenticated vectors for ACE-3 FP16 adaptation RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from fp16_adaptation_oracle import residual_add, rmsnorm, silu_gate

RMS_SIZE = 8
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


def checked_bytes(source_dir: Path, key: str) -> bytes:
    filename, expected = SOURCES[key]
    payload = (source_dir / filename).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{key} source SHA256 mismatch: expected {expected}, got {actual}"
        )
    return payload


def artifact_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "line_count": payload.count(b"\n"),
    }


def write_lines(path: Path, records: list[int], width: int) -> None:
    path.write_text(
        "".join(f"{record:0{width}x}\n" for record in records),
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.official_tensor_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(checked_bytes(source_dir, "config"))
    model_api = json.loads(checked_bytes(source_dir, "model_api"))
    if (
        config.get("hidden_size") != 896
        or config.get("intermediate_size") != 4864
        or config.get("rms_norm_eps") != 1e-6
    ):
        raise RuntimeError("authenticated Qwen geometry or RMS epsilon mismatch")
    if (
        model_api.get("id") != MODEL_REPOSITORY
        or model_api.get("sha") != MODEL_REVISION
    ):
        raise RuntimeError("authenticated model repository/revision mismatch")

    scale_payload = checked_bytes(source_dir, "scales")
    scales = list(struct.unpack(f"<{len(scale_payload) // 2}H", scale_payload))
    if len(scales) != 7 * 896:
        raise RuntimeError("official q_proj scale sample geometry mismatch")

    residual_specs = [
        ("both-positive-zero", 0x0000, 0x0000, "directed"),
        ("both-negative-zero", 0x8000, 0x8000, "directed"),
        ("one-plus-one", 0x3C00, 0x3C00, "directed"),
        ("exact-cancellation", 0x3C00, 0xBC00, "directed"),
        ("subnormal-add", 0x0001, 0x0001, "directed"),
        ("tie-even", 0x3C00, 0x1000, "directed"),
        ("positive-overflow", 0x7BFF, 0x7BFF, "directed"),
        ("invalid-nan", 0x7E00, 0x3C00, "directed"),
        ("official-scale-pair-0", scales[0], scales[1], "official_q_proj_scales"),
        ("official-scale-pair-1", scales[896], scales[897], "official_q_proj_scales"),
    ]
    residual_cases: list[dict[str, object]] = []
    residual_stream: list[int] = []
    for name, left, right, source in residual_specs:
        result, invalid, saturation = residual_add(left, right)
        residual_cases.append(
            {
                "name": name,
                "left": left,
                "right": right,
                "result": result,
                "invalid": invalid,
                "saturation": saturation,
                "source": source,
            }
        )
        residual_stream.append(
            left
            | (right << 16)
            | (result << 32)
            | (int(invalid) << 48)
            | (int(saturation) << 49)
        )

    silu_specs = [
        ("positive-zero", 0x0000, 0x3C00, "directed"),
        ("negative-zero", 0x8000, 0x3C00, "directed"),
        ("one-one", 0x3C00, 0x3C00, "directed"),
        ("negative-one", 0xBC00, 0x3C00, "directed"),
        ("negative-up", 0x3C00, 0xBC00, "directed"),
        ("subnormal", 0x0001, 0x3C00, "directed"),
        ("positive-overflow", 0x7BFF, 0x7BFF, "directed"),
        ("invalid-infinity", 0x7C00, 0x3C00, "directed"),
        ("official-scale-pair-0", scales[2], scales[3], "official_q_proj_scales"),
        ("official-scale-pair-1", scales[898], scales[899], "official_q_proj_scales"),
    ]
    silu_cases: list[dict[str, object]] = []
    silu_stream: list[int] = []
    for name, gate, up, source in silu_specs:
        result, invalid, saturation = silu_gate(gate, up)
        silu_cases.append(
            {
                "name": name,
                "gate": gate,
                "up": up,
                "result": result,
                "invalid": invalid,
                "saturation": saturation,
                "source": source,
            }
        )
        silu_stream.append(
            gate
            | (up << 16)
            | (result << 32)
            | (int(invalid) << 48)
            | (int(saturation) << 49)
        )

    rms_specs = [
        (
            "unit-symmetric",
            [0x3C00, 0xBC00, 0x3C00, 0xBC00] * 2,
            [0x3C00] * RMS_SIZE,
            "directed",
        ),
        (
            "all-zero-signed",
            [0x0000, 0x8000] * 4,
            [0x3C00] * RMS_SIZE,
            "directed",
        ),
        (
            "invalid-element",
            [0x3C00, 0x7E00] + [0x0000] * 6,
            [0x3C00] * RMS_SIZE,
            "directed",
        ),
        (
            "official-scale-window",
            scales[16:24],
            scales[912:920],
            "official_q_proj_scales",
        ),
    ]
    rms_transactions: list[dict[str, object]] = []
    rms_input_stream: list[int] = []
    rms_expected_stream: list[int] = []
    rms_meta_stream: list[int] = []
    for name, activations, weights, source in rms_specs:
        outputs, mean_q48, rms_q24 = rmsnorm(activations, weights)
        invalid = any(item[1] for item in outputs)
        rms_transactions.append(
            {
                "name": name,
                "activations": activations,
                "weights": weights,
                "outputs": [item[0] for item in outputs],
                "invalid": invalid,
                "saturations": [item[2] for item in outputs],
                "mean_q48": mean_q48,
                "rms_q24": rms_q24,
                "source": source,
            }
        )
        rms_input_stream.extend(
            activation | (weight << 16)
            for activation, weight in zip(activations, weights, strict=True)
        )
        rms_expected_stream.extend(
            result | (int(item_invalid) << 16) | (int(saturation) << 17)
            for result, item_invalid, saturation in outputs
        )
        rms_meta_stream.append(rms_q24 | (int(invalid) << 46))

    paths = {
        "residual_cases.hex": (residual_stream, 13),
        "silu_cases.hex": (silu_stream, 13),
        "rms_inputs.hex": (rms_input_stream, 8),
        "rms_expected.hex": (rms_expected_stream, 5),
        "rms_meta.hex": (rms_meta_stream, 12),
    }
    for filename, (records, width) in paths.items():
        write_lines(output_dir / filename, records, width)

    params = (
        f"localparam integer FP16_RESIDUAL_CASES = {len(residual_stream)};\n"
        f"localparam integer FP16_SILU_CASES = {len(silu_stream)};\n"
        f"localparam integer FP16_RMS_CASES = {len(rms_transactions)};\n"
        f"localparam integer FP16_RMS_SIZE = {RMS_SIZE};\n"
    )
    (output_dir / "fp16_adaptation_params.svh").write_text(
        params, encoding="ascii"
    )

    manifest = {
        "schema_version": 1,
        "kind": "ace3_fp16_adaptation_vectors",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_sha256": {key: value[1] for key, value in SOURCES.items()},
        "hidden_size": 896,
        "intermediate_size": 4864,
        "rms_epsilon": "1e-6",
        "rms_epsilon_q48": 281_474_977,
        "rms_test_size": RMS_SIZE,
        "residual_cases": residual_cases,
        "silu_cases": silu_cases,
        "rms_transactions": rms_transactions,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    artifacts = [
        artifact_record(output_dir / filename)
        for filename in [
            "manifest.json",
            "residual_cases.hex",
            "silu_cases.hex",
            "rms_inputs.hex",
            "rms_expected.hex",
            "rms_meta.hex",
            "fp16_adaptation_params.svh",
        ]
    ]
    print(
        "FP16_ADAPTATION_VECTOR_GENERATION_PASS "
        f"residual={len(residual_stream)} silu={len(silu_stream)} "
        f"rms_transactions={len(rms_transactions)} rms_elements={len(rms_input_stream)} "
        f"official_cases=5 artifacts={len(artifacts)}"
    )
    for artifact in artifacts:
        print(
            f"{artifact['file']} sha256={artifact['sha256']} "
            f"bytes={artifact['byte_count']} lines={artifact['line_count']}"
        )


if __name__ == "__main__":
    main()
