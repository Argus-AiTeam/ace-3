#!/usr/bin/env python3
"""Generate deterministic complete-input AWQ projection vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

from awq_bit_oracle import AWQ_REVERSE_ORDER, GROUP_SIZE
from projection_oracle import CROSS_ACC_BITS, complete_projection_output

SEED = 0xACE3CF02
IN_FEATURES = 896
OUT_FEATURES = 896
GROUPS = IN_FEATURES // GROUP_SIZE
OUTPUT_WORDS = OUT_FEATURES // 8
OFFICIAL_FIRST_OUTPUT = 4
OFFICIAL_OUTPUT_COUNT = 8
SOURCES: dict[str, tuple[str, str]] = {
    "model_api": (
        "model-api.json",
        "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
    ),
    "packing_utils": (
        "autoawq-v0.2.9-packing_utils.py",
        "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe",
    ),
    "config": (
        "config.json",
        "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
    ),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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


def finite_random_f16(rng: random.Random) -> int:
    sign = rng.randrange(2)
    selector = rng.randrange(16)
    exponent = 0 if selector == 0 else rng.randrange(8, 18)
    return (sign << 15) | (exponent << 10) | rng.randrange(1024)


def make_output(
    name: str,
    output_channel: int,
    activations: list[int],
    qweights: list[int],
    qzeros: list[int],
    scales: list[int],
    source: str,
) -> dict[str, object]:
    lane = output_channel % 8
    accumulator, result, invalid, saturation, group_accumulators = (
        complete_projection_output(
            activations, qweights, qzeros, scales, lane
        )
    )
    return {
        "name": name,
        "output_channel": output_channel,
        "logical_lane": lane,
        "output_word": output_channel // 8,
        "activations": activations,
        "qweights": qweights,
        "qzeros": qzeros,
        "scales": scales,
        "group_accumulators": group_accumulators,
        "accumulator": accumulator,
        "result": result,
        "invalid": invalid,
        "saturation": saturation,
        "source": source,
    }


def directed_output(
    name: str,
    output_channel: int,
    activations_by_index: dict[int, int],
    qweights_by_index: dict[int, int],
    scales_by_group: dict[int, int] | None = None,
) -> dict[str, object]:
    lane = output_channel % 8
    activations = [0] * IN_FEATURES
    qweights = [0] * IN_FEATURES
    qzeros = [lane_word(lane, 0)] * GROUPS
    scales = [0x3C00] * GROUPS
    for index, value in activations_by_index.items():
        activations[index] = value
    for index, value in qweights_by_index.items():
        qweights[index] = lane_word(lane, value)
    for group, value in (scales_by_group or {}).items():
        scales[group] = value
    return make_output(
        name,
        output_channel,
        activations,
        qweights,
        qzeros,
        scales,
        "contract_valid_synthetic",
    )


def build_transactions(source_base: Path) -> list[dict[str, object]]:
    model_api = json.loads(checked_bytes("model_api", source_base))
    if (
        model_api.get("id") != "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        or model_api.get("sha") != "db09cd27ead7fee40cdee309693cf83601b9c899"
    ):
        raise RuntimeError("authenticated model repository/revision mismatch")
    siblings = {
        sibling.get("rfilename"): sibling
        for sibling in model_api.get("siblings", [])
        if isinstance(sibling, dict)
    }
    model_tensor = siblings.get("model.safetensors", {})
    if (
        model_tensor.get("lfs", {}).get("sha256")
        != "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
    ):
        raise RuntimeError("authenticated model.safetensors identity mismatch")
    checked_bytes("packing_utils", source_base)

    config = json.loads(checked_bytes("config", source_base))
    expected_config = {
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("authenticated Qwen2.5-0.5B config geometry mismatch")
    quantization = config.get("quantization_config", {})
    if (
        quantization.get("bits") != 4
        or quantization.get("group_size") != GROUP_SIZE
        or quantization.get("version") != "gemm"
        or quantization.get("zero_point") is not True
    ):
        raise RuntimeError("authenticated AWQ config mismatch")

    qweight_raw = checked_bytes("qweight", source_base)
    qzero_raw = checked_bytes("qzeros", source_base)
    scale_raw = checked_bytes("scales", source_base)
    qweight_tensor = list(
        struct.unpack(f"<{len(qweight_raw) // 4}I", qweight_raw)
    )
    qzero_tensor = list(struct.unpack(f"<{len(qzero_raw) // 4}I", qzero_raw))
    scale_tensor = list(struct.unpack(f"<{len(scale_raw) // 2}H", scale_raw))
    if (
        len(qweight_tensor),
        len(qzero_tensor),
        len(scale_tensor),
    ) != (
        IN_FEATURES * OUTPUT_WORDS,
        GROUPS * OUTPUT_WORDS,
        GROUPS * OUT_FEATURES,
    ):
        raise RuntimeError("official layer-0 q_proj tensor geometry mismatch")

    rng = random.Random(SEED)
    official_activations = [
        finite_random_f16(rng) for _ in range(IN_FEATURES)
    ]
    official_outputs: list[dict[str, object]] = []
    for output_channel in range(
        OFFICIAL_FIRST_OUTPUT,
        OFFICIAL_FIRST_OUTPUT + OFFICIAL_OUTPUT_COUNT,
    ):
        output_word = output_channel // 8
        official_outputs.append(
            make_output(
                f"official-output-{output_channel}-full-input",
                output_channel,
                official_activations,
                [
                    qweight_tensor[input_index * OUTPUT_WORDS + output_word]
                    for input_index in range(IN_FEATURES)
                ],
                [
                    qzero_tensor[group * OUTPUT_WORDS + output_word]
                    for group in range(GROUPS)
                ],
                [
                    scale_tensor[group * OUT_FEATURES + output_channel]
                    for group in range(GROUPS)
                ],
                "official_layer0_q_proj",
            )
        )

    directed = [
        directed_output(
            "directed-cross-group-round-once-cancellation",
            0,
            {0: 0x0001, GROUP_SIZE: 0x8001},
            {0: 1, GROUP_SIZE: 1},
            {1: 0x3800},
        ),
        directed_output(
            "directed-positive-saturation",
            7,
            {0: 0x7BFF},
            {0: 1},
            {0: 0x4000},
        ),
        directed_output(
            "directed-negative-saturation",
            8,
            {0: 0xFBFF},
            {0: 1},
            {0: 0x4000},
        ),
        directed_output(
            "directed-min-subnormal",
            137,
            {0: 0x0001},
            {0: 1},
        ),
        directed_output("directed-zero", 511, {}, {}),
        directed_output(
            "directed-invalid-activation",
            895,
            {0: 0x7C00},
            {0: 1},
        ),
    ]
    transactions: list[dict[str, object]] = [
        {
            "name": "official-output-tile-4-through-11",
            "first_output": OFFICIAL_FIRST_OUTPUT,
            "outputs": official_outputs,
            "source": "official_layer0_q_proj",
        }
    ]
    transactions.extend(
        {
            "name": f"transaction-{output['name']}",
            "first_output": output["output_channel"],
            "outputs": [output],
            "source": output["source"],
        }
        for output in directed
    )
    return transactions


def write_vectors(
    transactions: list[dict[str, object]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transaction_lines: list[str] = []
    expected_lines: list[str] = []
    meta_lines: list[str] = []
    pair_lines: list[str] = []
    manifest_transactions: list[dict[str, object]] = []
    output_count = 0
    for transaction in transactions:
        outputs = transaction["outputs"]
        assert isinstance(outputs, list)
        first_output = int(transaction["first_output"])
        transaction_lines.append(
            f"{(first_output | (len(outputs) << 13)):07x}"
        )
        manifest_outputs: list[dict[str, object]] = []
        for output in outputs:
            assert isinstance(output, dict)
            accumulator = int(output["accumulator"])
            packed_expected = int(output["output_channel"])
            packed_expected |= (
                accumulator & ((1 << CROSS_ACC_BITS) - 1)
            ) << 13
            packed_expected |= int(output["result"]) << 115
            packed_expected |= int(bool(output["invalid"])) << 131
            packed_expected |= int(bool(output["saturation"])) << 132
            expected_lines.append(f"{packed_expected:034x}")
            qzeros = output["qzeros"]
            scales = output["scales"]
            activations = output["activations"]
            qweights = output["qweights"]
            assert isinstance(qzeros, list) and isinstance(scales, list)
            assert isinstance(activations, list) and isinstance(qweights, list)
            for qzero, scale in zip(qzeros, scales, strict=True):
                meta_lines.append(f"{((int(scale) << 32) | int(qzero)):012x}")
            for activation, qweight in zip(
                activations, qweights, strict=True
            ):
                pair_lines.append(
                    f"{((int(activation) << 32) | int(qweight)):012x}"
                )
            manifest_outputs.append(
                {
                    key: output[key]
                    for key in (
                        "name",
                        "output_channel",
                        "logical_lane",
                        "output_word",
                        "group_accumulators",
                        "accumulator",
                        "result",
                        "invalid",
                        "saturation",
                        "source",
                    )
                }
            )
            output_count += 1
        manifest_transactions.append(
            {
                "name": transaction["name"],
                "first_output": first_output,
                "output_count": len(outputs),
                "source": transaction["source"],
                "outputs": manifest_outputs,
            }
        )

    (output_dir / "transactions.hex").write_text(
        "\n".join(transaction_lines) + "\n"
    )
    (output_dir / "expected.hex").write_text(
        "\n".join(expected_lines) + "\n"
    )
    (output_dir / "meta.hex").write_text("\n".join(meta_lines) + "\n")
    (output_dir / "pairs.hex").write_text("\n".join(pair_lines) + "\n")
    (output_dir / "projection_params.svh").write_text(
        f"localparam integer PROJECTION_TRANSACTIONS = {len(transactions)};\n"
        f"localparam integer PROJECTION_OUTPUTS = {output_count};\n"
        f"localparam integer PROJECTION_GROUPS = {output_count * GROUPS};\n"
        f"localparam integer PROJECTION_PAIRS = {output_count * IN_FEATURES};\n"
    )
    manifest = {
        "schema_version": 1,
        "kind": "ace3_native_awq_full_input_projection_vectors",
        "model_repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "model_revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
        "source_sha256": {
            name: digest for name, (_, digest) in SOURCES.items()
        },
        "seed": SEED,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "group_size": GROUP_SIZE,
        "group_count": GROUPS,
        "cross_acc_bits": CROSS_ACC_BITS,
        "rounding": "once_after_all_groups",
        "transaction_count": len(transactions),
        "output_count": output_count,
        "metadata_count": output_count * GROUPS,
        "pair_count": output_count * IN_FEATURES,
        "official_output_count": OFFICIAL_OUTPUT_COUNT,
        "synthetic_output_count": output_count - OFFICIAL_OUTPUT_COUNT,
        "transactions": manifest_transactions,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    source_dir = args.official_tensor_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output directory must not be inside the official source")
    transactions = build_transactions(source_dir)
    write_vectors(transactions, output_dir)
    output_count = sum(len(transaction["outputs"]) for transaction in transactions)
    print(
        "PROJECTION_VECTOR_PASS "
        f"seed=0x{SEED:08x} transactions={len(transactions)} "
        f"outputs={output_count} official_outputs={OFFICIAL_OUTPUT_COUNT} "
        f"groups={output_count * GROUPS} pairs={output_count * IN_FEATURES} "
        "revision=db09cd27ead7fee40cdee309693cf83601b9c899 "
        "config_sha256=pass official_tensor_sha256=pass"
    )


if __name__ == "__main__":
    main()
