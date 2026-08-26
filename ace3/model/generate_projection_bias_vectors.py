#!/usr/bin/env python3
"""Generate small, independently recomputable biased AWQ projection vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from projection_oracle import complete_projection_output

IN_FEATURES = 128
OUT_FEATURES = 8
SOURCE_BIAS = "layer0_self_attn_q_proj_bias.fp16le.bin"


def word(lane: int, nibble: int) -> int:
    physical = (0, 4, 1, 5, 2, 6, 3, 7)[lane]
    return nibble << (4 * physical)


def make_case(name: str, lane: int, bias: int, active: dict[int, int],
              weights: dict[int, int], scale: int = 0x3C00) -> dict[str, object]:
    activations = [0] * IN_FEATURES
    qweights = [word(lane, 0)] * IN_FEATURES
    for index, value in active.items():
        activations[index] = value
    for index, value in weights.items():
        qweights[index] = word(lane, value)
    accumulator, result, invalid, saturation, groups = complete_projection_output(
        activations, qweights, [word(lane, 0)], [scale], lane, bias
    )
    return {
        "name": name, "lane": lane, "bias": bias, "activations": activations,
        "qweights": qweights, "qzero": word(lane, 0), "scale": scale,
        "accumulator": accumulator, "result": result, "invalid": invalid,
        "saturation": saturation, "group_accumulator": groups[0],
    }


def build_cases(source_dir: Path) -> list[dict[str, object]]:
    raw = (source_dir / SOURCE_BIAS).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = "e9612c72520a62dd903796d9535de2081e1cd15a724b2972513599ff276b9e72"
    if digest != expected or len(raw) != 1792:
        raise RuntimeError("authenticated layer-0 Q bias source mismatch")
    official_bias = struct.unpack("<896H", raw)[17]
    cases = [
        make_case("post-round-bias-distinguishes-preaccumulator", 0, 0x0001,
                  {0: 0x0001}, {0: 1}, 0x3800),
        make_case("official-layer0-q-bias", 1, official_bias, {}, {}),
        make_case("nonfinite-bias-invalid", 2, 0x7E00, {}, {}),
        make_case("finite-saturation", 3, 0x7BFF, {0: 0x7BFF}, {0: 15}, 0x4000),
        make_case("negative-zero-bias", 4, 0x8000, {}, {}),
    ]
    if cases[0]["result"] != 0x0001:
        raise AssertionError("directed post-round case did not retain min-subnormal bias")
    return cases


def write(output_dir: Path, source_dir: Path) -> None:
    cases = build_cases(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers, metadata, pairs, expected = [], [], [], []
    summaries = []
    for number, case in enumerate(cases):
        headers.append(f"{number:02x}{int(case['lane']):02x}{int(case['bias']):04x}")
        metadata.append(f"{(int(case['scale']) << 32) | int(case['qzero']):012x}")
        for activation, qweight in zip(case["activations"], case["qweights"], strict=True):
            pairs.append(f"{(int(activation) << 32) | int(qweight):012x}")
        packed = number | (int(case["accumulator"]) & ((1 << 102) - 1)) << 8
        packed |= int(case["result"]) << 110
        packed |= int(bool(case["invalid"])) << 126
        packed |= int(bool(case["saturation"])) << 127
        expected.append(f"{packed:032x}")
        summaries.append({key: case[key] for key in (
            "name", "lane", "bias", "accumulator", "group_accumulator", "result",
            "invalid", "saturation")})
    source = source_dir / SOURCE_BIAS
    manifest = {
        "schema_version": 1,
        "kind": "ace3_awq_w4a16_projection_bias_vectors",
        "geometry": {"in_features": IN_FEATURES, "out_features": OUT_FEATURES,
                     "bias_enable": 1, "group_size": 128},
        "source_sha256": {SOURCE_BIAS: hashlib.sha256(source.read_bytes()).hexdigest()},
        "rounding": "complete Q53.48 projection rounds to FP16 before FP16 bias residual_add",
        "cases": summaries,
    }
    files = {
        "cases.hex": "\n".join(headers) + "\n",
        "meta.hex": "\n".join(metadata) + "\n",
        "pairs.hex": "\n".join(pairs) + "\n",
        "expected.hex": "\n".join(expected) + "\n",
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    write(args.output_dir.resolve(), args.official_tensor_dir.resolve(strict=True))
    print("PROJECTION_BIAS_VECTOR_PASS cases=5 pairs=640 in_features=128 "
          "out_features=8 bias_enable=1 source_q_bias=pass post_round=pass")


if __name__ == "__main__":
    main()
