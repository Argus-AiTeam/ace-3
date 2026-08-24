#!/usr/bin/env python3
"""Generate deterministic official-tensor and synthetic contract vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

from awq_bit_oracle import AWQ_REVERSE_ORDER, GROUP_SIZE, complete_dot

SEED = 0xACE3CF01
SOURCES: dict[str, tuple[str, str]] = {
    "qweight": (
        "sample-model_layers_0_self_attn_q_proj-qweight.bin",
        "db4770023698611ff0115d220590fdb8232fbe5dcbd22fbe80e0bcdc838caf87",
    ),
    "qzeros": (
        "sample-model_layers_0_self_attn_q_proj-qzeros.bin",
        "3cf7cd5712dd7523db3c7dd47c2b1d582e19545036f75b95ff0331c1fc0c596c",
    ),
    "scales": (
        "sample-model_layers_0_self_attn_q_proj-scales.bin",
        "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic ACE-3 AWQ G128 vectors"
    )
    parser.add_argument(
        "--official-tensor-dir",
        required=True,
        type=Path,
        help="read-only directory containing the three official layer-0 q_proj samples",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for generated vectors (the root build uses build/vectors)",
    )
    return parser.parse_args()


def checked_bytes(name: str, source_base: Path) -> bytes:
    filename, expected = SOURCES[name]
    path = source_base / filename
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{name} source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return payload


def packed_word(values: list[int]) -> int:
    if len(values) != 8 or any(not 0 <= value < 16 for value in values):
        raise ValueError("packed I32 needs eight nibbles")
    return sum(value << (4 * slot) for slot, value in enumerate(values))


def lane_word(logical_lane: int, value: int, fill: int = 0) -> int:
    values = [fill] * 8
    values[AWQ_REVERSE_ORDER[logical_lane]] = value
    return packed_word(values)


def one_hot_activations(bits: int) -> list[int]:
    return [bits] + [0] * (GROUP_SIZE - 1)


def add_case(
    cases: list[dict[str, object]],
    name: str,
    lane: int,
    qzero: int,
    scale: int,
    activations: list[int],
    qweights: list[int],
    source: str,
) -> None:
    accumulator, result, invalid, saturation = complete_dot(
        activations, qweights, qzero, scale, lane
    )
    cases.append(
        {
            "name": name,
            "lane": lane,
            "qzero": qzero,
            "scale": scale,
            "activations": activations,
            "qweights": qweights,
            "accumulator": accumulator,
            "result": result,
            "invalid": invalid,
            "saturation": saturation,
            "source": source,
        }
    )


def finite_random_f16(rng: random.Random) -> int:
    sign = rng.randrange(2)
    selector = rng.randrange(16)
    if selector == 0:
        exponent = 0
    else:
        exponent = rng.randrange(8, 20)
    return (sign << 15) | (exponent << 10) | rng.randrange(1024)


def build_cases(source_base: Path) -> list[dict[str, object]]:
    qweight_raw = checked_bytes("qweight", source_base)
    qzero_raw = checked_bytes("qzeros", source_base)
    scale_raw = checked_bytes("scales", source_base)
    qweights = list(struct.unpack(f"<{len(qweight_raw) // 4}I", qweight_raw))
    qzeros = list(struct.unpack(f"<{len(qzero_raw) // 4}I", qzero_raw))
    scales = list(struct.unpack(f"<{len(scale_raw) // 2}H", scale_raw))
    if (len(qweights), len(qzeros), len(scales)) != (896 * 112, 7 * 112, 7 * 896):
        raise RuntimeError("official layer-0 q_proj tensor geometry mismatch")

    cases: list[dict[str, object]] = []
    rng = random.Random(SEED)
    for lane in range(8):
        group = 0
        output_word = 0
        logical_output = lane
        activations = [finite_random_f16(rng) for _ in range(GROUP_SIZE)]
        official_weights = [
            qweights[(group * GROUP_SIZE + index) * 112 + output_word]
            for index in range(GROUP_SIZE)
        ]
        add_case(
            cases,
            f"official-g0-output-{logical_output}-seeded",
            lane,
            qzeros[group * 112 + output_word],
            scales[group * 896 + logical_output],
            activations,
            official_weights,
            "official_layer0_q_proj",
        )

    for group, output_word, lane in ((3, 17, 2), (6, 111, 7)):
        logical_output = output_word * 8 + lane
        activations = [finite_random_f16(rng) for _ in range(GROUP_SIZE)]
        official_weights = [
            qweights[(group * GROUP_SIZE + index) * 112 + output_word]
            for index in range(GROUP_SIZE)
        ]
        add_case(
            cases,
            f"official-g{group}-output-{logical_output}-seeded",
            lane,
            qzeros[group * 112 + output_word],
            scales[group * 896 + logical_output],
            activations,
            official_weights,
            "official_layer0_q_proj",
        )

    reorder_weights = packed_word([index + 1 for index in range(8)])
    for lane in range(8):
        add_case(
            cases,
            f"synthetic-reorder-lane-{lane}",
            lane,
            packed_word([0] * 8),
            0x3C00,
            one_hot_activations(0x3C00),
            [reorder_weights] + [0] * 127,
            "contract_valid_synthetic",
        )

    directed = [
        ("zero", 0x0, 0x0, 0x3C00, 0x0000),
        ("no-plus-one-negative", 0x7, 0x8, 0x3C00, 0x3C00),
        ("negative-activation", 0x1, 0x0, 0x3C00, 0xBC00),
        ("negative-scale", 0x1, 0x0, 0xBC00, 0x3C00),
        ("activation-min-subnormal", 0x1, 0x0, 0x3C00, 0x0001),
        ("scale-min-subnormal", 0x1, 0x0, 0x0001, 0x3C00),
        ("subnormal-product-underflow", 0x1, 0x0, 0x0001, 0x0001),
        ("subnormal-rne-tie-even", 0x1, 0x0, 0x3800, 0x0003),
        ("positive-saturation", 0x1, 0x0, 0x4000, 0x7BFF),
        ("negative-saturation", 0x1, 0x0, 0x4000, 0xFBFF),
        ("invalid-activation", 0x1, 0x0, 0x3C00, 0x7C00),
        ("invalid-scale", 0x1, 0x0, 0x7C00, 0x3C00),
    ]
    for name, weight, zero, scale, activation in directed:
        add_case(
            cases,
            f"synthetic-{name}",
            0,
            lane_word(0, zero),
            scale,
            one_hot_activations(activation),
            [lane_word(0, weight)] + [0] * 127,
            "contract_valid_synthetic",
        )
    return cases


def write_vectors(cases: list[dict[str, object]], generated: Path) -> None:
    generated.mkdir(parents=True, exist_ok=True)
    meta_lines: list[str] = []
    pair_lines: list[str] = []
    manifest_cases: list[dict[str, object]] = []
    for case in cases:
        accumulator = int(case["accumulator"])
        meta = int(case["qzero"])
        meta |= int(case["scale"]) << 32
        meta |= int(case["lane"]) << 48
        meta |= (accumulator & ((1 << 96) - 1)) << 51
        meta |= int(case["result"]) << 147
        meta |= int(bool(case["invalid"])) << 163
        meta |= int(bool(case["saturation"])) << 164
        meta_lines.append(f"{meta:042x}")
        activations = case["activations"]
        qweights = case["qweights"]
        assert isinstance(activations, list) and isinstance(qweights, list)
        for activation, qweight in zip(activations, qweights, strict=True):
            pair_lines.append(f"{((int(activation) << 32) | int(qweight)):012x}")
        manifest_cases.append(
            {
                key: case[key]
                for key in (
                    "name",
                    "lane",
                    "qzero",
                    "scale",
                    "accumulator",
                    "result",
                    "invalid",
                    "saturation",
                    "source",
                )
            }
        )
    (generated / "meta.hex").write_text("\n".join(meta_lines) + "\n")
    (generated / "pairs.hex").write_text("\n".join(pair_lines) + "\n")
    (generated / "cases.txt").write_text(
        "\n".join(
            f"{case['name']} {int(case['qzero']):08x} {int(case['scale']):04x} "
            f"{int(case['lane'])} "
            f"{(int(case['accumulator']) & ((1 << 96) - 1)):024x} "
            f"{int(case['result']):04x} {int(bool(case['invalid']))} "
            f"{int(bool(case['saturation']))}"
            for case in cases
        )
        + "\n"
    )
    (generated / "vector_params.svh").write_text(
        f"localparam integer VECTOR_CASES = {len(cases)};\n"
        f"localparam integer VECTOR_PAIRS = {len(cases) * GROUP_SIZE};\n"
    )
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "group_size": GROUP_SIZE,
        "official_case_count": sum(
            case["source"] == "official_layer0_q_proj" for case in cases
        ),
        "synthetic_case_count": sum(
            case["source"] == "contract_valid_synthetic" for case in cases
        ),
        "exact_ulp_bound": 0,
        "cases": manifest_cases,
    }
    (generated / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    arguments = parse_args()
    official_tensor_dir = arguments.official_tensor_dir.resolve(strict=True)
    output_dir = arguments.output_dir.resolve()
    if not official_tensor_dir.is_dir():
        raise NotADirectoryError(official_tensor_dir)
    if output_dir == official_tensor_dir or official_tensor_dir in output_dir.parents:
        raise ValueError("output directory must not be inside the official tensor directory")
    generated_cases = build_cases(official_tensor_dir)
    write_vectors(generated_cases, output_dir)
    print(
        "VECTOR_PASS "
        f"seed=0x{SEED:08x} cases={len(generated_cases)} "
        f"pairs={len(generated_cases) * GROUP_SIZE} official_sha256=pass "
        f"official_tensor_dir={official_tensor_dir} output_dir={output_dir}"
    )
