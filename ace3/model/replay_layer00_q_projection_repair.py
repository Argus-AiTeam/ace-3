#!/usr/bin/env python3
"""Replay the repaired layer-0 Q dependency cone against the accepted oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decoder_layer0_oracle
from diagnose_layer01_position2_stage00 import (
    ABSOLUTE_TOLERANCE,
    HIDDEN_SIZE,
    MAX_ULP_DISTANCE,
    POSITION,
    RELATIVE_TOLERANCE,
    authenticated_trace,
    file_record,
    load_json,
    manifest_input,
    sha256_bytes,
    trace_stage,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTEMPT = (
    ROOT / "build/model24_persistent_kv_second_token_attempt002/traversal"
)
DEFAULT_REFERENCE = (
    ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/model24_persistent_kv_second_token_layer00_q_repair_attempt001"
)


class ReplayError(RuntimeError):
    """Raised when an authenticated replay boundary or result is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def ordered_f16(bits: np.ndarray) -> np.ndarray:
    values = bits.astype(np.int64)
    return np.where(
        values & 0x8000,
        0x8000 - (values & 0x7FFF),
        0x8000 + values,
    )


def comparison(
    actual_bits: np.ndarray,
    expected_bits: np.ndarray,
) -> dict[str, Any]:
    require(actual_bits.shape == expected_bits.shape, "comparison shape mismatch")
    actual = actual_bits.view("<f2").astype(np.float64)
    expected = expected_bits.view("<f2").astype(np.float64)
    finite = np.isfinite(actual) & np.isfinite(expected)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 2.0**-14)
    ulp = np.abs(ordered_f16(actual_bits) - ordered_f16(expected_bits))
    passed = finite & (
        (absolute <= ABSOLUTE_TOLERANCE)
        | ((relative < RELATIVE_TOLERANCE) & (ulp <= MAX_ULP_DISTANCE))
    )
    different = np.flatnonzero(actual_bits != expected_bits)
    failed = np.flatnonzero(~passed)

    def difference(index: int) -> dict[str, Any]:
        return {
            "element_index": index,
            "actual_bits": f"{int(actual_bits[index]):04x}",
            "expected_bits": f"{int(expected_bits[index]):04x}",
            "actual": float(actual[index]),
            "expected": float(expected[index]),
            "absolute_error": float(absolute[index]),
            "relative_error": float(relative[index]),
            "ulp_distance": int(ulp[index]),
        }

    return {
        "records": int(actual_bits.size),
        "finite_count": int(np.sum(finite)),
        "exact_match_count": int(np.sum(actual_bits == expected_bits)),
        "different_count": int(different.size),
        "failure_count": int(failed.size),
        "within_tolerance": not failed.size,
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp.max()),
        "first_difference": (
            None if not different.size else difference(int(different[0]))
        ),
        "first_failure": None if not failed.size else difference(int(failed[0])),
    }


def write_hex(path: Path, values: np.ndarray) -> dict[str, Any]:
    payload = "".join(f"{int(value):04x}\n" for value in values).encode("ascii")
    path.write_bytes(payload)
    return {
        **file_record(path),
        "records": int(values.size),
        "semantic_sha256": sha256_bytes(
            np.asarray(values, dtype="<u2").tobytes()
        ),
    }


def layer_values(layer_dir: Path, layer: int) -> dict[str, list[int]]:
    manifest_path = layer_dir / "vectors/manifest.json"
    manifest = load_json(manifest_path)
    source_prefix = f"model.layers.{layer}."
    values: dict[str, list[int]] = {}
    for tensor in manifest["tensors"]:
        checkpoint = tensor["checkpoint_tensor"]
        name = str(checkpoint["name"])
        require(
            name.startswith(source_prefix),
            f"unexpected layer{layer:02d} tensor: {name}",
        )
        serialized = tensor["serialized"]
        path = Path(serialized["path"])
        actual = file_record(path)
        require(
            actual["bytes"] == serialized["bytes"]
            and actual["sha256"] == serialized["sha256"]
            and path.parent == layer_dir / "vectors/tensors",
            f"layer{layer:02d} tensor binding mismatch: {path}",
        )
        digits = 4 if checkpoint["dtype"] == "F16" else 8
        rows = path.read_text(encoding="ascii").splitlines()
        require(
            rows and all(len(row) == digits for row in rows),
            f"malformed layer{layer:02d} tensor: {path}",
        )
        suffix = name.removeprefix(source_prefix)
        values[f"model.layers.0.{suffix}:"] = [
            int(row, 16) for row in rows
        ]
    return values


def repaired_layer_replay(
    values: dict[str, list[int]],
    activation: np.ndarray,
    cache_k: list[list[int]],
    cache_v: list[list[int]],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    final, trace = decoder_layer0_oracle.run_token(
        values,
        activation.tolist(),
        POSITION,
        [row.copy() for row in cache_k],
        [row.copy() for row in cache_v],
        accurate_silu=True,
    )
    stages: dict[int, np.ndarray] = {}
    for stage in (0, 1, 8, 18):
        stages[stage] = np.asarray(
            [value for item_stage, _, value, _ in trace if item_stage == stage],
            dtype="<u2",
        )
    require(
        stages[0].shape == (HIDDEN_SIZE,)
        and stages[1].shape == (HIDDEN_SIZE,)
        and stages[8].shape
        == (decoder_layer0_oracle.HEADS * (POSITION + 1),)
        and stages[18].shape == (HIDDEN_SIZE,),
        "repaired replay trace geometry mismatch",
    )
    result = np.asarray(final, dtype="<u2")
    require(
        np.array_equal(result, stages[18]),
        "repaired final output differs from stage18 trace",
    )
    return result, stages


def reference_array(
    reference: Path,
    layer: int,
    stage: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    record_path = reference / f"layer{layer:02d}_generation1.json"
    record = load_json(record_path)
    suffix = f"/layer{layer:02d}/position002/stage{stage:02d}.npy"
    matches = [
        item
        for item in record["stages"]
        if str(item["path"]).endswith(suffix)
    ]
    require(len(matches) == 1, f"reference layer {layer} stage {stage} mismatch")
    array_record = matches[0]
    array_path = Path(array_record["path"])
    require(
        sha256_bytes(array_path.read_bytes()) == array_record["file_sha256"],
        f"independent array file binding mismatch: {array_path}",
    )
    values = np.load(array_path, allow_pickle=False)
    require(
        values.dtype == np.dtype("<u2")
        and list(values.shape) == array_record["shape"]
        and sha256_bytes(np.ascontiguousarray(values).tobytes())
        == array_record["semantic_sha256"],
        f"independent array semantic binding mismatch: {array_path}",
    )
    return values, {
        "layer_record": file_record(record_path),
        "array": array_record,
    }


def generate(attempt: Path, reference: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"refusing to replace immutable attempt: {output}")
    output.mkdir(parents=True)
    trace_paths = {
        (layer, position): (
            attempt
            / f"layer{layer:02d}"
            / (
                f"position{position:03d}/raw/trace.hex"
                if position < POSITION
                else "raw/trace.hex"
            )
        )
        for layer in (0, 1)
        for position in range(POSITION + 1)
    }
    comparison_paths = {
        key: (
            reference
            / f"rtl_comparison_layer{key[0]:02d}_position{key[1]:03d}.json"
        )
        for key in trace_paths
    }
    traces = {
        key: authenticated_trace(comparison_paths[key], path)
        for key, path in trace_paths.items()
    }

    def cache(layer: int, stage: int) -> list[list[int]]:
        return [
            trace_stage(traces[(layer, position)], position, stage, 128).tolist()
            for position in range(POSITION)
        ]

    layer_dirs = {
        layer: attempt / f"layer{layer:02d}" for layer in (0, 1)
    }
    layer00_input, layer00_input_record = manifest_input(layer_dirs[0])
    layer00_final, layer00_stages = repaired_layer_replay(
        layer_values(layer_dirs[0], 0),
        layer00_input,
        cache(0, 6),
        cache(0, 7),
    )
    layer01_final, layer01_stages = repaired_layer_replay(
        layer_values(layer_dirs[1], 1),
        layer00_final,
        cache(1, 6),
        cache(1, 7),
    )

    references: dict[tuple[int, int], np.ndarray] = {}
    reference_records: dict[str, Any] = {}
    for layer, stages in ((0, (1, 18)), (1, (0, 1, 8))):
        for stage in stages:
            values, records = reference_array(reference, layer, stage)
            references[(layer, stage)] = values
            reference_records[f"layer{layer:02d}_stage{stage:02d}"] = records

    baseline = {
        "layer00_stage18": comparison(
            trace_stage(
                traces[(0, POSITION)], POSITION, 18, HIDDEN_SIZE
            ),
            references[(0, 18)],
        ),
        "layer01_stage00": comparison(
            trace_stage(
                traces[(1, POSITION)], POSITION, 0, HIDDEN_SIZE
            ),
            references[(1, 0)],
        ),
        "layer01_stage08": comparison(
            trace_stage(
                traces[(1, POSITION)],
                POSITION,
                8,
                decoder_layer0_oracle.HEADS * (POSITION + 1),
                POSITION + 1,
            ),
            references[(1, 8)],
        ),
    }
    repaired = {
        "layer00_stage01": comparison(
            layer00_stages[1], references[(0, 1)]
        ),
        "layer00_stage18": comparison(
            layer00_stages[18], references[(0, 18)]
        ),
        "layer01_stage00": comparison(
            layer01_stages[0], references[(1, 0)]
        ),
        "layer01_stage01": comparison(
            layer01_stages[1], references[(1, 1)]
        ),
        "layer01_stage08": comparison(
            layer01_stages[8], references[(1, 8)]
        ),
    }
    require(
        all(item["within_tolerance"] for item in repaired.values()),
        "repaired dependency cone remains outside accepted policy",
    )
    require(
        repaired["layer00_stage18"]["max_absolute_error"]
        < baseline["layer00_stage18"]["max_absolute_error"],
        "layer00 stage18 drift was not reduced",
    )
    require(
        repaired["layer01_stage00"]["max_absolute_error"]
        < baseline["layer01_stage00"]["max_absolute_error"],
        "layer01 stage00 drift was not reduced",
    )
    require(
        repaired["layer01_stage08"]["failure_count"]
        < baseline["layer01_stage08"]["failure_count"],
        "layer01 stage08 material failures were not reduced",
    )

    raw_dir = output / "raw"
    raw_dir.mkdir()
    outputs = {
        "layer00_stage01": write_hex(
            raw_dir / "layer00_position002_stage01.hex",
            layer00_stages[1],
        ),
        "layer00_stage18": write_hex(
            raw_dir / "layer00_position002_stage18.hex",
            layer00_stages[18],
        ),
        "layer01_stage00": write_hex(
            raw_dir / "layer01_position002_stage00.hex",
            layer01_stages[0],
        ),
        "layer01_stage01": write_hex(
            raw_dir / "layer01_position002_stage01.hex",
            layer01_stages[1],
        ),
        "layer01_stage08": write_hex(
            raw_dir / "layer01_position002_stage08.hex",
            layer01_stages[8],
        ),
        "layer01_stage18": write_hex(
            raw_dir / "layer01_position002_stage18.hex",
            layer01_final,
        ),
    }
    source_paths = (
        Path(__file__),
        ROOT / "ace3/model/accepted_awq_projection.py",
        ROOT / "ace3/model/decoder_layer0_oracle.py",
        ROOT / "ace3/model/diagnose_layer01_position2_stage00.py",
        ROOT / "ace3/model/projection_oracle.py",
    )
    document = {
        "schema_version": 1,
        "kind": "ace3_layer00_q_projection_repair_replay",
        "attempt_id": output.name,
        "status": "PASS",
        "policy": {
            "profile": "native AWQ W4A16 G128 asymmetric, FP16 activations/KV",
            "projection_arithmetic": (
                "binary64 dequantized native packed AWQ dot and bias, "
                "FP16RNE output at each projection boundary"
            ),
            "finite_required": True,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance_strict": RELATIVE_TOLERANCE,
            "relative_denominator_floor": 2.0**-14,
            "max_ulp_distance": MAX_ULP_DISTANCE,
            "acceptance_expression": (
                "finite AND (abs_error <= 0.125 OR "
                "(relative_error < 0.001 AND ordered_FP16_ULP <= 1))"
            ),
        },
        "authenticated_inputs": {
            "preserved_attempt": str(attempt.resolve()),
            "layer00_input": layer00_input_record,
            "traces": [
                file_record(trace_paths[key]) for key in sorted(trace_paths)
            ],
            "trace_comparisons": [
                file_record(comparison_paths[key])
                for key in sorted(comparison_paths)
            ],
            "independent_reference": reference_records,
            "sources": [file_record(path) for path in source_paths],
        },
        "repair": {
            "scope": (
                "shared layer-parameterized Q projection numerical contract "
                "applied to layer00 and its layer01 Q consumer"
            ),
            "native_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
            "qzero_adjustment": "none",
            "group_size": 128,
        },
        "baseline_comparisons": baseline,
        "repaired_comparisons": repaired,
        "reduction": {
            name: {
                "failure_count_before": baseline[name]["failure_count"],
                "failure_count_after": repaired[name]["failure_count"],
                "different_count_before": baseline[name]["different_count"],
                "different_count_after": repaired[name]["different_count"],
                "max_absolute_error_before": baseline[name][
                    "max_absolute_error"
                ],
                "max_absolute_error_after": repaired[name][
                    "max_absolute_error"
                ],
            }
            for name in baseline
        },
        "outputs": outputs,
        "claim_boundary": (
            "bounded computer-local numerical replay of the real preserved "
            "layer00 position002 input/KV path through layer01 stage08; no "
            "whole-model, synthesis, PPA, FPGA, or deployed-hardware claim"
        ),
    }
    result_path = output / "result.json"
    result_path.write_bytes(canonical_json(document))
    print(
        "LAYER00_Q_PROJECTION_REPAIR_REPLAY_PASS "
        f"attempt={output.name} "
        f"layer00_stage18_max_abs="
        f"{repaired['layer00_stage18']['max_absolute_error']} "
        f"layer01_stage00_max_abs="
        f"{repaired['layer01_stage00']['max_absolute_error']} "
        f"layer01_stage08_failures="
        f"{repaired['layer01_stage08']['failure_count']} "
        f"result_sha256={sha256_bytes(result_path.read_bytes())}"
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.attempt, arguments.reference, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
