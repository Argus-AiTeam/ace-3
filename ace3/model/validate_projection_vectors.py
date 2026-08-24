#!/usr/bin/env python3
"""Authenticate and validate complete-input projection simulator vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from projection_oracle import complete_projection_output

EXPECTED_FILES = {
    "manifest.json": (365, None),
    "transactions.hex": (7, 7),
    "expected.hex": (14, 34),
    "meta.hex": (98, 12),
    "pairs.hex": (12544, 12),
    "projection_params.svh": (4, None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def authenticate(
    generated_dir: Path, bindings: dict[str, Any]
) -> None:
    require(
        bindings.get("kind")
        == "ace3_full_input_projection_serialized_vector_bindings",
        "unexpected projection binding kind",
    )
    artifacts = bindings.get("serialized_artifacts")
    require(isinstance(artifacts, list), "projection artifact bindings missing")
    by_name: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        require(isinstance(artifact, dict), "artifact binding is not an object")
        name = artifact.get("file")
        require(isinstance(name, str), "artifact file name missing")
        require(name not in by_name, f"duplicate artifact binding: {name}")
        by_name[name] = artifact
    require(
        set(by_name) == set(EXPECTED_FILES),
        "projection bindings do not cover the exact simulator artifact set",
    )
    for name in sorted(EXPECTED_FILES):
        artifact = by_name[name]
        payload = (generated_dir / name).read_bytes()
        expected_hash = artifact.get("sha256")
        actual_hash = hashlib.sha256(payload).hexdigest()
        require(
            actual_hash == expected_hash,
            f"{name} SHA256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
        require(
            len(payload) == artifact.get("byte_count"),
            f"{name} byte count mismatch",
        )
        require(
            payload.count(b"\n") == artifact.get("line_count"),
            f"{name} line count mismatch",
        )


def checked_hex(path: Path, lines: int, width: int) -> None:
    records = path.read_text(encoding="ascii").splitlines()
    require(len(records) == lines, f"{path}: record count mismatch")
    require(
        all(
            len(record) == width
            and all(char in "0123456789abcdef" for char in record)
            for record in records
        ),
        f"{path}: malformed hexadecimal record",
    )


def read_hex(path: Path) -> list[int]:
    return [
        int(record, 16)
        for record in path.read_text(encoding="ascii").splitlines()
    ]


def validate_stream_semantics(
    generated_dir: Path, manifest: dict[str, Any]
) -> None:
    transaction_stream = read_hex(generated_dir / "transactions.hex")
    expected_stream = read_hex(generated_dir / "expected.hex")
    metadata_stream = read_hex(generated_dir / "meta.hex")
    pair_stream = read_hex(generated_dir / "pairs.hex")
    transactions = manifest["transactions"]
    expected_index = 0
    metadata_index = 0
    pair_index = 0
    for transaction_index, transaction in enumerate(transactions):
        outputs = transaction["outputs"]
        packed_transaction = transaction_stream[transaction_index]
        require(
            (packed_transaction & 0x1FFF) == transaction["first_output"]
            and ((packed_transaction >> 13) & 0x1FFF) == len(outputs),
            f"transaction {transaction_index} stream mismatch",
        )
        for local_output_index, output in enumerate(outputs):
            channel = transaction["first_output"] + local_output_index
            require(
                output["output_channel"] == channel,
                f"output {expected_index} sequence mismatch",
            )
            qzeros: list[int] = []
            scales: list[int] = []
            for _ in range(7):
                metadata = metadata_stream[metadata_index]
                metadata_index += 1
                qzeros.append(metadata & 0xFFFFFFFF)
                scales.append((metadata >> 32) & 0xFFFF)
            activations: list[int] = []
            qweights: list[int] = []
            for _ in range(896):
                pair = pair_stream[pair_index]
                pair_index += 1
                qweights.append(pair & 0xFFFFFFFF)
                activations.append((pair >> 32) & 0xFFFF)
            accumulator, result, invalid, saturation, group_accumulators = (
                complete_projection_output(
                    activations,
                    qweights,
                    qzeros,
                    scales,
                    channel % 8,
                )
            )
            packed_expected = channel
            packed_expected |= (accumulator & ((1 << 102) - 1)) << 13
            packed_expected |= result << 115
            packed_expected |= int(invalid) << 131
            packed_expected |= int(saturation) << 132
            require(
                expected_stream[expected_index] == packed_expected,
                f"output {expected_index} serialized oracle mismatch",
            )
            require(
                output["logical_lane"] == channel % 8
                and output["output_word"] == channel // 8
                and output["group_accumulators"] == group_accumulators
                and output["accumulator"] == accumulator
                and output["result"] == result
                and output["invalid"] is invalid
                and output["saturation"] is saturation,
                f"output {expected_index} manifest oracle mismatch",
            )
            expected_index += 1
    require(
        expected_index == len(expected_stream)
        and metadata_index == len(metadata_stream)
        and pair_index == len(pair_stream),
        "serialized streams contain unconsumed records",
    )


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    contract = load_json(args.contract)
    bindings = load_json(args.bindings)
    for path, document in ((args.contract, contract), (args.bindings, bindings)):
        require(isinstance(document, dict), f"{path}: JSON root must be an object")
        require(document.get("schema_version") == 1, f"{path}: schema_version != 1")
    authenticate(generated_dir, bindings)

    manifest = load_json(generated_dir / "manifest.json")
    require(
        manifest.get("kind") == "ace3_native_awq_full_input_projection_vectors",
        "unexpected projection manifest kind",
    )
    require(
        manifest.get("model_repository")
        == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        and manifest.get("model_revision")
        == "db09cd27ead7fee40cdee309693cf83601b9c899",
        "projection model identity mismatch",
    )
    require(
        manifest.get("source_sha256")
        == {
            "config": "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
            "model_api": "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
            "packing_utils": "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe",
            "qweight": "db4770023698611ff0115d220590fdb8232fbe5dcbd22fbe80e0bcdc838caf87",
            "qzeros": "3cf7cd5712dd7523db3c7dd47c2b1d582e19545036f75b95ff0331c1fc0c596c",
            "scales": "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
        },
        "projection source hash set mismatch",
    )
    generation = bindings.get("generation")
    require(isinstance(generation, dict), "projection generation binding missing")
    for key in (
        "seed",
        "in_features",
        "out_features",
        "group_size",
        "group_count",
        "cross_acc_bits",
        "transaction_count",
        "output_count",
        "metadata_count",
        "pair_count",
        "official_output_count",
        "synthetic_output_count",
    ):
        require(manifest.get(key) == generation.get(key), f"{key} mismatch")
    require(
        manifest.get("rounding") == "once_after_all_groups",
        "projection rounding policy mismatch",
    )
    require(
        contract.get("composition", {}).get("final_rounding")
        == "one binary16 round-to-nearest-ties-to-even conversion after every input group has been summed",
        "projection contract final-rounding policy mismatch",
    )
    authenticated_config = contract.get("authenticated_model_config", {})
    require(
        authenticated_config.get("sha256")
        == "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090",
        "authenticated model config hash mismatch",
    )
    require(
        authenticated_config.get("repository")
        == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
        and authenticated_config.get("revision")
        == "db09cd27ead7fee40cdee309693cf83601b9c899"
        and authenticated_config.get("model_api_sha256")
        == "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768",
        "authenticated fixed model revision mismatch",
    )
    authenticated_tensors = contract.get("authenticated_tensor_contract", {})
    require(
        authenticated_tensors.get("sha256")
        == "b3754c03658534b79ddf8f667049e9122d631f84005fb219faf0e5e9de56e2aa",
        "authenticated tensor contract hash mismatch",
    )
    require(
        authenticated_tensors.get("native_awq_correction_sha256")
        == "601e726d6a524d01bc48ef435831d9fe23cf9a99ad86e22c2382d5af74cded66"
        and authenticated_tensors.get("packing_source_sha256")
        == "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe",
        "native AWQ correction provenance mismatch",
    )
    geometry_tuples = {
        (
            tuple(geometry.get("modules", [])),
            geometry.get("in_features"),
            geometry.get("out_features"),
            geometry.get("groups"),
            geometry.get("packed_output_words"),
        )
        for geometry in contract.get("supported_geometries", [])
    }
    require(
        geometry_tuples
        == {
            (("q_proj", "o_proj"), 896, 896, 7, 112),
            (("k_proj", "v_proj"), 896, 128, 7, 16),
            (("gate_proj", "up_proj"), 896, 4864, 7, 608),
            (("down_proj",), 4864, 896, 38, 112),
        },
        "authenticated projection geometry table mismatch",
    )
    transactions = manifest.get("transactions")
    require(
        isinstance(transactions, list) and len(transactions) == 7,
        "projection transaction count mismatch",
    )
    outputs = [
        output
        for transaction in transactions
        for output in transaction.get("outputs", [])
    ]
    require(len(outputs) == 14, "projection output count mismatch")
    official = [
        output for output in outputs
        if output.get("source") == "official_layer0_q_proj"
    ]
    require(
        [output.get("output_channel") for output in official]
        == list(range(4, 12)),
        "official output tile does not span channels 4 through 11",
    )
    cancellation = next(
        output for output in outputs
        if output.get("name")
        == "directed-cross-group-round-once-cancellation"
    )
    require(
        cancellation.get("group_accumulators", [])[:2]
        == [1 << 24, -(1 << 23)]
        and cancellation.get("accumulator") == 1 << 23
        and cancellation.get("result") == 0,
        "cross-group round-once cancellation vector mismatch",
    )
    directed_expectations = {
        "directed-positive-saturation": (0x7BFF, False, True),
        "directed-negative-saturation": (0xFBFF, False, True),
        "directed-min-subnormal": (0x0001, False, False),
        "directed-zero": (0x0000, False, False),
        "directed-invalid-activation": (0x0000, True, False),
    }
    for name, expected in directed_expectations.items():
        output = next(item for item in outputs if item.get("name") == name)
        require(
            (
                output.get("result"),
                output.get("invalid"),
                output.get("saturation"),
            )
            == expected,
            f"{name} directed expectation mismatch",
        )
    require(
        all(
            len(output.get("group_accumulators", [])) == 7
            and any(output["group_accumulators"])
            for output in official
        )
        and len({output.get("accumulator") for output in official}) == 8,
        "official outputs are vacuous or insufficiently distinct",
    )

    for name, (line_count, width) in EXPECTED_FILES.items():
        if width is not None:
            checked_hex(generated_dir / name, line_count, width)
    expected_params = (
        "localparam integer PROJECTION_TRANSACTIONS = 7;\n"
        "localparam integer PROJECTION_OUTPUTS = 14;\n"
        "localparam integer PROJECTION_GROUPS = 98;\n"
        "localparam integer PROJECTION_PAIRS = 12544;\n"
    )
    require(
        (generated_dir / "projection_params.svh").read_text(encoding="ascii")
        == expected_params,
        "projection parameter include mismatch",
    )
    validate_stream_semantics(generated_dir, manifest)
    print(
        "PROJECTION_JSON_VALIDATION_PASS json_files=3 serialized_artifacts=6 "
        "sha256=pass transactions=7 outputs=14 official_outputs=8 "
        "groups=98 pairs=12544 cross_acc_bits=102 round_once=pass "
        "stream_oracle=pass nonvacuity=pass"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PROJECTION_JSON_VALIDATION_FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
