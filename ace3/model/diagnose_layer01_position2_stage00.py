#!/usr/bin/env python3
"""Localize the position-2 layer00-to-layer01 stage00 divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decoder_layer0_oracle
from fp16_adaptation_oracle import (
    EPSILON_Q48,
    decode_f16_q24,
    q24_to_f16,
    rmsnorm,
    round_div_even_unsigned,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTEMPT = (
    ROOT / "build/model24_persistent_kv_second_token_attempt002/traversal"
)
DEFAULT_ATTEMPT004 = (
    ROOT / "build/model24_persistent_kv_second_token_attempt004"
)
DEFAULT_ATTEMPT005 = (
    ROOT / "build/model24_persistent_kv_second_token_attempt005"
)
DEFAULT_REFERENCE = (
    ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/model24_persistent_kv_second_token_upstream_diagnostic_attempt003"
)
POSITION = 2
HIDDEN_SIZE = 896
ABSOLUTE_TOLERANCE = 0.125
RELATIVE_TOLERANCE = 0.001
MAX_ULP_DISTANCE = 1


class DiagnosticError(RuntimeError):
    """Raised when an authenticated boundary or controlled result is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def authenticated_trace(comparison_path: Path, trace_path: Path) -> bytes:
    comparison_document = load_json(comparison_path)
    expected = comparison_document.get("binding", {}).get("trace.hex", {})
    actual = file_record(trace_path)
    require(
        actual["bytes"] == expected.get("bytes")
        and actual["sha256"] == expected.get("sha256"),
        f"preserved trace binding mismatch: {trace_path}",
    )
    return trace_path.read_bytes()


def trace_stage(
    payload: bytes,
    position: int,
    stage: int,
    count: int,
    index_period: int | None = None,
) -> np.ndarray:
    records: list[tuple[int, int]] = []
    for ordinal, row in enumerate(payload.decode("ascii").splitlines()):
        require(len(row) == 16, f"trace row {ordinal} width mismatch")
        require(
            int(row[0:2], 16) == 0 and int(row[2:6], 16) == position,
            f"trace row {ordinal} owner mismatch",
        )
        if int(row[6:8], 16) == stage:
            records.append((int(row[8:12], 16), int(row[12:16], 16)))
    if index_period is None:
        records.sort()
        expected_indices = list(range(count))
    else:
        require(
            index_period > 0 and count % index_period == 0,
            f"trace stage {stage} index period mismatch",
        )
        expected_indices = list(range(index_period)) * (count // index_period)
    require(
        [index for index, _ in records] == expected_indices,
        f"trace stage {stage} index mismatch",
    )
    return np.asarray([value for _, value in records], dtype="<u2")


def load_hidden(path: Path) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    require(len(rows) == HIDDEN_SIZE, f"hidden row count mismatch: {path}")
    values = np.empty(HIDDEN_SIZE, dtype="<u2")
    for ordinal, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and int(row[2:6], 16) == ordinal,
            f"hidden row {ordinal} ordering mismatch: {path}",
        )
        values[ordinal] = int(row[6:10], 16)
    return values


def load_hex_f16(path: Path) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    require(
        len(rows) == HIDDEN_SIZE and all(len(row) == 4 for row in rows),
        f"FP16 tensor geometry mismatch: {path}",
    )
    return np.asarray([int(row, 16) for row in rows], dtype="<u2")


def authenticated_array(record: dict[str, Any]) -> np.ndarray:
    path = Path(record["path"])
    require(
        sha256_bytes(path.read_bytes()) == record["file_sha256"],
        f"independent array file binding mismatch: {path}",
    )
    value = np.load(path, allow_pickle=False)
    require(
        value.dtype == np.dtype("<u2")
        and value.shape == (HIDDEN_SIZE,)
        and sha256_bytes(np.ascontiguousarray(value).tobytes())
        == record["semantic_sha256"],
        f"independent array semantic binding mismatch: {path}",
    )
    return value


def reference_stage(
    layer_record: dict[str, Any],
    layer: int,
    stage: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = f"/layer{layer:02d}/position002/stage{stage:02d}.npy"
    matches = [
        record
        for record in layer_record["stages"]
        if str(record["path"]).endswith(suffix)
    ]
    require(len(matches) == 1, f"reference layer {layer} stage {stage} mismatch")
    return authenticated_array(matches[0]), matches[0]


def integer_rmsnorm(
    activation_bits: np.ndarray,
    weight_bits: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    outputs, mean_q48, root_q24 = rmsnorm(
        activation_bits.tolist(),
        weight_bits.tolist(),
    )
    require(
        all(not invalid and not saturated for _, invalid, saturated in outputs),
        "integer RMSNorm produced invalid or saturated output",
    )
    return (
        np.asarray([value for value, _, _ in outputs], dtype="<u2"),
        mean_q48,
        root_q24,
    )


def accepted_rmsnorm(
    activation_bits: np.ndarray,
    weight_bits: np.ndarray,
) -> np.ndarray:
    activation = torch.from_numpy(
        activation_bits.view("<f2").astype(np.float64)
    )
    weight = torch.from_numpy(weight_bits.view("<f2").astype(np.float64))
    variance = activation.pow(2).mean()
    normalized = activation * torch.rsqrt(variance + 1e-6)
    return (
        (normalized * weight)
        .to(torch.float16)
        .detach()
        .cpu()
        .numpy()
        .view("<u2")
    )


def rounded_root_rmsnorm(
    activation_bits: np.ndarray,
    weight_bits: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    activations = [decode_f16_q24(int(value)) for value in activation_bits]
    weights = [decode_f16_q24(int(value)) for value in weight_bits]
    require(
        all(value[1] for value in activations + weights),
        "rounded-root RMSNorm input is non-finite",
    )
    mean_q48 = round_div_even_unsigned(
        sum(value[0] * value[0] for value in activations)
        + EPSILON_Q48 * HIDDEN_SIZE,
        HIDDEN_SIZE,
    )
    floor_root = math.isqrt(mean_q48)
    ceiling_root = floor_root + 1
    lower_distance = mean_q48 - floor_root * floor_root
    upper_distance = ceiling_root * ceiling_root - mean_q48
    root_q24 = (
        ceiling_root
        if upper_distance < lower_distance
        or (
            upper_distance == lower_distance
            and floor_root & 1
        )
        else floor_root
    )
    outputs = []
    for activation, weight in zip(activations, weights, strict=True):
        product = activation[0] * weight[0]
        magnitude = round_div_even_unsigned(abs(product), root_q24)
        result, saturated = q24_to_f16(
            -magnitude if product < 0 else magnitude,
            zero_sign=activation[3] ^ weight[3],
        )
        require(not saturated, "rounded-root RMSNorm saturated")
        outputs.append(result)
    return np.asarray(outputs, dtype="<u2"), mean_q48, root_q24


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
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 2.0**-14)
    ulp = np.abs(ordered_f16(actual_bits) - ordered_f16(expected_bits))
    passed = (absolute <= ABSOLUTE_TOLERANCE) | (
        (relative < RELATIVE_TOLERANCE) & (ulp <= MAX_ULP_DISTANCE)
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


def manifest_input(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = path / "vectors/manifest.json"
    manifest = load_json(manifest_path)
    input_record = manifest["input"]
    input_path = Path(input_record["path"])
    require(
        file_record(input_path)["bytes"] == input_record["bytes"]
        and file_record(input_path)["sha256"] == input_record["sha256"],
        f"vector input binding mismatch: {input_path}",
    )
    bits = load_hidden(input_path)
    require(
        sha256_bytes(bits.tobytes()) == manifest["input_activation_sha256"],
        f"vector input semantic mismatch: {input_path}",
    )
    return bits, {
        "manifest": file_record(manifest_path),
        "input": file_record(input_path),
        "semantic_sha256": sha256_bytes(bits.tobytes()),
    }


def oracle_values(
    manifest: dict[str, Any],
    layer_dir: Path,
) -> dict[str, list[int]]:
    prefix = "model.layers.0."
    values: dict[str, list[int]] = {}
    for tensor in manifest["tensors"]:
        checkpoint = tensor["checkpoint_tensor"]
        name = checkpoint["name"]
        require(name.startswith(prefix), f"unexpected layer00 tensor: {name}")
        serialized = tensor["serialized"]
        path = Path(serialized["path"])
        actual = file_record(path)
        require(
            actual["bytes"] == serialized["bytes"]
            and actual["sha256"] == serialized["sha256"]
            and path.parent == layer_dir / "vectors/tensors",
            f"layer00 serialized tensor binding mismatch: {path}",
        )
        digits = 4 if checkpoint["dtype"] == "F16" else 8
        rows = path.read_text(encoding="ascii").splitlines()
        require(
            rows and all(len(row) == digits for row in rows),
            f"malformed serialized tensor: {path}",
        )
        values[f"{prefix}{name.removeprefix(prefix)}:"] = [
            int(row, 16) for row in rows
        ]
    return values


def run_with_injected_first_norm(
    values: dict[str, list[int]],
    activation: np.ndarray,
    cache_k: list[list[int]],
    cache_v: list[list[int]],
    *,
    accepted_norm: np.ndarray | None = None,
    accepted_q: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    original_rmsnorm = decoder_layer0_oracle.rmsnorm
    original_module = decoder_layer0_oracle._module
    rmsnorm_calls = 0
    q_injected = False

    def injected_rmsnorm(
        activations_f16: list[int],
        weights_f16: list[int],
        epsilon_q48: int = EPSILON_Q48,
    ) -> tuple[list[tuple[int, bool, bool]], int, int]:
        nonlocal rmsnorm_calls
        rmsnorm_calls += 1
        if rmsnorm_calls == 1 and accepted_norm is not None:
            require(
                activations_f16 == activation.tolist(),
                "injected RMSNorm activation mismatch",
            )
            return (
                [(int(value), False, False) for value in accepted_norm],
                0,
                0,
            )
        return original_rmsnorm(
            activations_f16,
            weights_f16,
            epsilon_q48,
        )

    def injected_module(
        module_values: dict[str, list[int]],
        prefix: str,
        activations: list[int],
        out_features: int,
        bias: list[int] | None = None,
    ) -> list[int]:
        nonlocal q_injected
        if prefix == "self_attn.q_proj" and accepted_q is not None:
            require(not q_injected, "duplicate Q injection")
            q_injected = True
            return accepted_q.astype(np.uint64).tolist()
        return original_module(
            module_values,
            prefix,
            activations,
            out_features,
            bias,
        )

    decoder_layer0_oracle.rmsnorm = injected_rmsnorm
    decoder_layer0_oracle._module = injected_module
    try:
        final, trace = decoder_layer0_oracle.run_token(
            values,
            activation.tolist(),
            POSITION,
            [row.copy() for row in cache_k],
            [row.copy() for row in cache_v],
            accurate_silu=True,
        )
    finally:
        decoder_layer0_oracle.rmsnorm = original_rmsnorm
        decoder_layer0_oracle._module = original_module
    require(
        rmsnorm_calls == 2,
        "injected layer00 replay RMSNorm call count mismatch",
    )
    require(
        q_injected == (accepted_q is not None),
        "injected layer00 replay Q call count mismatch",
    )
    stage00 = np.asarray(
        [value for stage, _, value, _ in trace if stage == 0],
        dtype="<u2",
    )
    expected_stage00 = (
        accepted_norm
        if accepted_norm is not None
        else integer_rmsnorm(activation, load_hex_f16(
            Path(
                DEFAULT_ATTEMPT
                / "layer00/vectors/tensors/"
                "layer0_input_layernorm_weight.fp16le.bin.hex"
            )
        ))[0]
    )
    require(
        np.array_equal(stage00, expected_stage00),
        "injected layer00 replay stage00 mismatch",
    )
    return np.asarray(final, dtype="<u2"), stage00


def generate(
    attempt: Path,
    attempt004: Path,
    attempt005: Path,
    reference: Path,
    output: Path,
) -> dict[str, Any]:
    status004_path = attempt004 / "status.json"
    evidence004_path = attempt004 / "evidence.json"
    diagnosis005_path = attempt005 / "diagnosis.json"
    status004 = load_json(status004_path)
    diagnosis005 = load_json(diagnosis005_path)
    require(
        status004.get("attempt_id")
        == "model24_persistent_kv_second_token_attempt004"
        and status004.get("status") == "NUMERIC_MISMATCH_REVIEW_REQUIRED",
        "attempt004 status mismatch",
    )
    require(
        diagnosis005.get("status") == "ROOT_CAUSE_REFINED_UPSTREAM",
        "attempt005 diagnostic status mismatch",
    )

    comparison_paths = {
        (layer, position): (
            reference
            / f"rtl_comparison_layer{layer:02d}_position{position:03d}.json"
        )
        for layer in (0, 1)
        for position in range(POSITION + 1)
    }
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
    traces = {
        key: authenticated_trace(comparison_paths[key], trace_paths[key])
        for key in comparison_paths
    }
    rtl_layer00_stage00 = trace_stage(
        traces[(0, POSITION)],
        POSITION,
        0,
        HIDDEN_SIZE,
    )
    rtl_layer00_stage18 = trace_stage(
        traces[(0, POSITION)],
        POSITION,
        18,
        HIDDEN_SIZE,
    )
    rtl_layer01_stage00 = trace_stage(
        traces[(1, POSITION)],
        POSITION,
        0,
        HIDDEN_SIZE,
    )
    layer00_cache_k = [
        trace_stage(traces[(0, position)], position, 6, 128).tolist()
        for position in range(POSITION)
    ]
    layer00_cache_v = [
        trace_stage(traces[(0, position)], position, 7, 128).tolist()
        for position in range(POSITION)
    ]

    layer00_record_path = reference / "layer00_generation1.json"
    layer01_record_path = reference / "layer01_generation1.json"
    layer00_record = load_json(layer00_record_path)
    layer01_record = load_json(layer01_record_path)
    reference_layer00_stage00, reference_layer00_stage00_record = (
        reference_stage(layer00_record, 0, 0)
    )
    reference_layer00_stage01, reference_layer00_stage01_record = (
        reference_stage(layer00_record, 0, 1)
    )
    reference_layer00_stage18, reference_layer00_stage18_record = (
        reference_stage(layer00_record, 0, 18)
    )
    reference_layer01_stage00, reference_layer01_stage00_record = (
        reference_stage(layer01_record, 1, 0)
    )

    layer00_dir = attempt / "layer00"
    layer01_dir = attempt / "layer01"
    layer00_input, layer00_input_record = manifest_input(layer00_dir)
    layer01_input, layer01_input_record = manifest_input(layer01_dir)
    layer00_manifest = load_json(layer00_dir / "vectors/manifest.json")
    layer00_final_path = layer00_dir / "raw/final.hex"
    layer00_final = load_hidden(layer00_final_path)
    require(
        np.array_equal(layer00_final, rtl_layer00_stage18)
        and np.array_equal(layer00_final, layer01_input),
        "layer00 final to layer01 input handoff is not bit exact",
    )

    weight_paths = {
        layer: (
            attempt
            / f"layer{layer:02d}/vectors/tensors/"
            f"layer{layer}_input_layernorm_weight.fp16le.bin.hex"
        )
        for layer in (0, 1)
    }
    weights = {
        layer: load_hex_f16(weight_paths[layer])
        for layer in (0, 1)
    }

    integer_layer00, layer00_mean_q48, layer00_floor_root_q24 = (
        integer_rmsnorm(layer00_input, weights[0])
    )
    accepted_layer00 = accepted_rmsnorm(layer00_input, weights[0])
    rounded_layer00, _, layer00_rounded_root_q24 = rounded_root_rmsnorm(
        layer00_input,
        weights[0],
    )
    require(
        np.array_equal(integer_layer00, rtl_layer00_stage00),
        "integer RMSNorm does not reproduce preserved layer00 stage00",
    )
    require(
        np.array_equal(accepted_layer00, reference_layer00_stage00),
        "accepted RMSNorm does not reproduce reference layer00 stage00",
    )
    layer00_oracle_values = oracle_values(layer00_manifest, layer00_dir)
    stage00_injected_final, _ = run_with_injected_first_norm(
        layer00_oracle_values,
        layer00_input,
        layer00_cache_k,
        layer00_cache_v,
        accepted_norm=reference_layer00_stage00,
    )
    q_injected_final, _ = run_with_injected_first_norm(
        layer00_oracle_values,
        layer00_input,
        layer00_cache_k,
        layer00_cache_v,
        accepted_q=reference_layer00_stage01,
    )

    integer_layer01_rtl_input, layer01_rtl_mean_q48, layer01_rtl_root_q24 = (
        integer_rmsnorm(layer01_input, weights[1])
    )
    accepted_layer01_rtl_input = accepted_rmsnorm(layer01_input, weights[1])
    integer_layer01_reference_input, _, _ = integer_rmsnorm(
        reference_layer00_stage18,
        weights[1],
    )
    accepted_layer01_reference_input = accepted_rmsnorm(
        reference_layer00_stage18,
        weights[1],
    )
    require(
        np.array_equal(integer_layer01_rtl_input, rtl_layer01_stage00),
        "integer RMSNorm does not reproduce preserved layer01 stage00",
    )
    require(
        np.array_equal(
            accepted_layer01_reference_input,
            reference_layer01_stage00,
        ),
        "accepted RMSNorm does not reproduce reference layer01 stage00",
    )

    layer00_stage00_comparison = comparison(
        rtl_layer00_stage00,
        reference_layer00_stage00,
    )
    require(
        layer00_stage00_comparison["different_count"] == 1,
        "unexpected earliest layer00 stage00 divergence count",
    )
    layer00_handoff_comparison = comparison(
        layer00_final,
        reference_layer00_stage18,
    )
    layer01_stage00_comparison = comparison(
        rtl_layer01_stage00,
        reference_layer01_stage00,
    )
    crossed = {
        "rtl_arithmetic_rtl_handoff_vs_accepted": layer01_stage00_comparison,
        "accepted_arithmetic_rtl_handoff_vs_accepted": comparison(
            accepted_layer01_rtl_input,
            reference_layer01_stage00,
        ),
        "rtl_arithmetic_reference_handoff_vs_accepted": comparison(
            integer_layer01_reference_input,
            reference_layer01_stage00,
        ),
        "accepted_arithmetic_reference_handoff_vs_accepted": comparison(
            accepted_layer01_reference_input,
            reference_layer01_stage00,
        ),
        "rtl_vs_accepted_arithmetic_same_rtl_handoff": comparison(
            rtl_layer01_stage00,
            accepted_layer01_rtl_input,
        ),
    }
    rounded_root_comparison = comparison(
        rounded_layer00,
        reference_layer00_stage00,
    )
    stage00_injected_final_vs_rtl = comparison(
        stage00_injected_final,
        layer00_final,
    )
    stage00_injected_final_vs_accepted = comparison(
        stage00_injected_final,
        reference_layer00_stage18,
    )
    q_injected_final_vs_rtl = comparison(q_injected_final, layer00_final)
    q_injected_final_vs_accepted = comparison(
        q_injected_final,
        reference_layer00_stage18,
    )
    q_injected_layer01_stage00 = accepted_rmsnorm(
        q_injected_final,
        weights[1],
    )
    q_injected_layer01_stage00_comparison = comparison(
        q_injected_layer01_stage00,
        reference_layer01_stage00,
    )
    require(
        stage00_injected_final_vs_rtl["different_count"] == 0,
        "isolated accepted layer00 stage00 was not quantized away",
    )
    require(
        q_injected_final_vs_rtl["different_count"] > 0,
        "accepted layer00 Q injection did not affect layer00 final",
    )

    earliest = layer00_stage00_comparison["first_difference"]
    require(earliest is not None, "missing earliest exact divergence")
    document = {
        "schema_version": 1,
        "kind": "ace3_layer00_to_layer01_position002_stage00_diagnosis",
        "status": "ROOT_CAUSE_LOCALIZED",
        "policy": {
            "profile": "native AWQ W4A16 G128 asymmetric, FP16 activations/KV",
            "accepted_layer_arithmetic": (
                "binary64 intra-stage arithmetic with FP16RNE at each stage"
            ),
            "rtl_rmsnorm_arithmetic": (
                "exact FP16-to-Q24 decode, rounded Q48 mean, floor Q24 "
                "square root, rounded Q24 division, FP16RNE output"
            ),
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance_strict": RELATIVE_TOLERANCE,
            "max_ulp_distance": MAX_ULP_DISTANCE,
        },
        "authenticated_inputs": {
            "attempt004_status": file_record(status004_path),
            "attempt004_evidence": file_record(evidence004_path),
            "attempt005_diagnosis": file_record(diagnosis005_path),
            "rtl_comparisons": [
                file_record(comparison_paths[(layer, position)])
                for layer in (0, 1)
                for position in range(POSITION + 1)
            ],
            "rtl_traces": [
                file_record(trace_paths[(layer, position)])
                for layer in (0, 1)
                for position in range(POSITION + 1)
            ],
            "layer00_reference_record": file_record(layer00_record_path),
            "layer01_reference_record": file_record(layer01_record_path),
            "reference_arrays": [
                reference_layer00_stage00_record,
                reference_layer00_stage01_record,
                reference_layer00_stage18_record,
                reference_layer01_stage00_record,
            ],
            "layer00_input": layer00_input_record,
            "layer00_final": file_record(layer00_final_path),
            "layer01_input": layer01_input_record,
            "input_layernorm_weights": [
                file_record(weight_paths[layer]) for layer in (0, 1)
            ],
            "diagnostic_source": file_record(Path(__file__)),
        },
        "layer00_stage00_control": {
            "common_input_semantic_sha256": sha256_bytes(
                layer00_input.tobytes()
            ),
            "rtl_reproduced_by_integer_rmsnorm": True,
            "reference_reproduced_by_accepted_rmsnorm": True,
            "rtl_vs_accepted": layer00_stage00_comparison,
            "earliest_exact_divergence": {
                **earliest,
                "layer": 0,
                "position": POSITION,
                "stage": 0,
                "stage_name": "input_rmsnorm",
            },
            "integer_mean_q48": layer00_mean_q48,
            "integer_floor_root_q24": layer00_floor_root_q24,
            "nearest_root_q24": layer00_rounded_root_q24,
            "nearest_root_vs_accepted": rounded_root_comparison,
            "nearest_root_changes_floor_root": (
                layer00_rounded_root_q24 != layer00_floor_root_q24
            ),
            "accepted_stage00_injected_replay": {
                "final_vs_preserved_rtl": (
                    stage00_injected_final_vs_rtl
                ),
                "final_vs_accepted": stage00_injected_final_vs_accepted,
                "causal_effect_reaches_final": False,
            },
            "accepted_q_injected_replay": {
                "injected_stage01_semantic_sha256": sha256_bytes(
                    reference_layer00_stage01.tobytes()
                ),
                "final_vs_preserved_rtl": q_injected_final_vs_rtl,
                "final_vs_accepted": q_injected_final_vs_accepted,
                "accepted_layer01_stage00_vs_accepted": (
                    q_injected_layer01_stage00_comparison
                ),
                "causal_effect_reaches_final": True,
            },
        },
        "layer00_to_layer01_handoff": {
            "rtl_final_equals_layer01_input": True,
            "rtl_semantic_sha256": sha256_bytes(layer00_final.tobytes()),
            "accepted_semantic_sha256": sha256_bytes(
                reference_layer00_stage18.tobytes()
            ),
            "rtl_vs_accepted": layer00_handoff_comparison,
        },
        "layer01_stage00_control": {
            "rtl_reproduced_by_integer_rmsnorm": True,
            "reference_reproduced_by_accepted_rmsnorm": True,
            "rtl_input_integer_mean_q48": layer01_rtl_mean_q48,
            "rtl_input_integer_floor_root_q24": layer01_rtl_root_q24,
            "crossed_comparisons": crossed,
        },
        "root_cause": (
            "The first exact accepted-policy split is layer00 position002 "
            "stage00 input RMSNorm, not the layer00-to-layer01 file handoff: "
            "the handoff is bit exact from layer00 final.hex into layer01 "
            "inputs.hex. On the same authenticated layer00 input and weights, "
            "the integer floor-root RMSNorm exactly reproduces RTL while the "
            "binary64 RMSNorm exactly reproduces the accepted reference. "
            "That one-ULP stage00 split is quantized away by the integer "
            "layer00 datapath and does not change stage18. The earliest tested "
            "boundary with a causal effect on stage18 is layer00 stage01 Q "
            "projection: injecting the accepted Q tensor changes the final "
            "handoff. "
            "At layer01, integer and accepted RMSNorm are bit exact on the "
            "preserved handoff, so the observed layer01 stage00 difference is "
            "entirely inherited from the divergent layer00 final tensor, not "
            "introduced by layer01 RMSNorm."
        ),
        "minimal_repair_required": (
            "Repair the layer00 Q projection numerical contract against the "
            "frozen accepted FP16 stage01 tensor; neither changing layer01 "
            "RMSNorm nor merely rounding the layer00 Q24 RMS root addresses "
            "the causal split. Then rerun only layer00 position002 through "
            "stage18 and its layer01 stage00/stage08 consumers. Do not start "
            "a new 24-layer traversal until that bounded cone is within the "
            "accepted policy."
        ),
        "claim_boundary": (
            "bounded position002 layer00-to-layer01 numerical diagnosis only; "
            "no repaired RTL, full traversal, synthesis, PPA, FPGA, or "
            "dialogue claim"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "diagnosis.json"
    require(not result_path.exists(), f"refusing to replace immutable {result_path}")
    result_path.write_bytes(canonical_json(document))
    print(
        "LAYER01_POSITION002_STAGE00_ROOT_CAUSE_LOCALIZED "
        f"result={result_path} "
        f"earliest_index={earliest['element_index']} "
        f"rtl_bits={earliest['actual_bits']} "
        f"accepted_bits={earliest['expected_bits']} "
        f"handoff_sha256={document['layer00_to_layer01_handoff']['rtl_semantic_sha256']}"
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--attempt004", type=Path, default=DEFAULT_ATTEMPT004)
    parser.add_argument("--attempt005", type=Path, default=DEFAULT_ATTEMPT005)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(
        args.attempt.resolve(strict=True),
        args.attempt004.resolve(strict=True),
        args.attempt005.resolve(strict=True),
        args.reference.resolve(strict=True),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
